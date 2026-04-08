#!/usr/bin/env python3
"""
chad/core/orchestrator.py

CHAD Orchestrator — Risk Budget Publisher (Allocator Spine) [Production v2]
=========================================================================

What this service is
--------------------
This orchestrator is CHAD’s canonical *risk-budget publisher*. It computes and publishes:

- runtime/dynamic_caps.json           (authoritative per-strategy dollar caps for throttle/execution)
- runtime/savage_alloc_state.json     (Phase 11 receipt; advisory; best-effort)
- runtime/allocator_v3_state.json     (Phase 11.5 receipt; advisory; best-effort)
- data/traces/decision_trace_YYYYMMDD.ndjson  (Phase 8 recorder; one record per cycle; best-effort)

What this service is NOT
------------------------
It NEVER:
- calls broker APIs
- executes trades
- mutates LiveGate / STOP / operator intent
- writes secrets

Design guarantees
-----------------
- Deterministic given runtime/config inputs (best-effort logging cannot affect outputs).
- Strict numeric hygiene (finite floats; clamps; safe defaults).
- Atomic runtime writes (tmp -> fsync -> replace; fsync dir best-effort).
- DecisionTrace recorder is append-only, best-effort, and cannot break caps publishing.
- Stdlib-only; no third-party deps.

CLI
---
Run once:
    python -m chad.core.orchestrator --once --log-level INFO

Run forever:
    export CHAD_ORCH_RUN_FOREVER=1
    python -m chad.core.orchestrator --log-level INFO
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from chad.risk.dynamic_risk_allocator import (
    DynamicRiskAllocator,
    PortfolioSnapshot,
    StrategyAllocation,
    default_output_path as default_dynamic_caps_path,
    enforce_chassis,
)

# Phase 11 (Savage)
from chad.risk.savage_allocator import (
    SavageConfig,
    apply_savage_overlay,
    write_state_report_atomic,
)

# Phase 11.5 (V3)
from chad.risk.allocator_v3 import (
    AllocV3Config,
    compute_allocator_v3,
    write_state_atomic as write_allocator_v3_state_atomic,
)

LOGGER_NAME = "chad.orchestrator"
logger = logging.getLogger(LOGGER_NAME)

# -----------------------------------------------------------------------------
# Canonical paths
# -----------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
RUNTIME_DIR: Path = REPO_ROOT / "runtime"
DATA_DIR: Path = REPO_ROOT / "data"
TRACES_DIR: Path = DATA_DIR / "traces"
PORTFOLIO_SNAPSHOT_PATH: Path = RUNTIME_DIR / "portfolio_snapshot.json"


# -----------------------------------------------------------------------------
# Hardened helpers (stdlib-only)
# -----------------------------------------------------------------------------

def _utc_now_iso() -> str:
    # Stable format without importing datetime on hot path.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_ymd() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return float(x) if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    s = str(v).strip() if v is not None else ""
    return s if s else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return _finite_float(v, default=float(default))


def _env_bool01(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Dict[str, Any]) -> str:
    # Deterministic: stable ordering, stable whitespace.
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    return _sha256_bytes(b)


def _atomic_write_json(path: Path, payload: Dict[str, Any], *, indent: int = 2) -> None:
    """
    Atomic JSON write:
      - write tmp
      - fsync tmp
      - replace
      - fsync dir best-effort
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=indent, sort_keys=True) + "\n").encode("utf-8")

    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        # Never fail publish on fsync-dir
        pass


def _append_ndjson_locked(path: Path, line_obj: Dict[str, Any]) -> None:
    """
    Append one NDJSON line. Best-effort fsync, with a simple POSIX advisory lock.

    Why not atomic rename?
      - NDJSON is append-only; writers must not truncate. We lock, append, flush.
      - If lock fails, we still attempt append without lock (fail-soft) by writing
        to a _nolock file to preserve evidence rather than losing it.

    NOTE: This recorder is best-effort and MUST NOT raise in orchestrator hot path.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    ymd = _utc_ymd()

    # Normal file first
    target = path

    line = (json.dumps(line_obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    try:
        with target.open("ab", buffering=0) as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Could not lock; fall back to nolock file to preserve trace evidence.
                fallback = path.with_name(f"{path.stem}_{ymd}_nolock.ndjson")
                with fallback.open("ab", buffering=0) as fb:
                    fb.write(line)
                    fb.flush()
                    try:
                        os.fsync(fb.fileno())
                    except Exception:
                        pass
                return

            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception:
        # Best-effort: swallow
        return


# -----------------------------------------------------------------------------
# Live trace enrichment helpers (best-effort, fail-soft)
#
# These replace the previous reads of runtime/full_execution_cycle_last.json and
# runtime/decision_trace_heartbeat.json, both of which had been frozen since
# 2026-04-03 when their writer processes (full_execution_cycle.py oneshot and
# decision_trace_heartbeat.py timer) stopped running. The orchestrator now
# composes a fresh gate snapshot every cycle from authoritative sources:
#   - HTTP /live-gate            (live evaluation, source of truth for permissions)
#   - runtime/scr_state.json     (chad-scr-sync writes from /shadow)
#   - runtime/live_readiness.json (chad-live-readiness oneshot)
#   - runtime/fast_loop_state.json (orchestrator fast loop)
#   - runtime/last_route_decision.json (chad-live-loop)
# and writes a fresh runtime/decision_trace_heartbeat.json so downstream readers
# (operator surface, daily reports) see live data.
# -----------------------------------------------------------------------------

LIVE_GATE_URL_DEFAULT = "http://127.0.0.1:9618/live-gate"
DECISION_TRACE_HEARTBEAT_PATH: Path = RUNTIME_DIR / "decision_trace_heartbeat.json"


def _fetch_live_gate_snapshot(
    url: Optional[str] = None,
    timeout_s: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Fail-soft HTTP fetch of /live-gate. Returns the parsed JSON dict on success
    or None on any error (network, timeout, parse, non-dict). Stdlib only.
    """
    import urllib.request

    target = url or os.environ.get("LIVE_GATE_URL") or LIVE_GATE_URL_DEFAULT
    try:
        with urllib.request.urlopen(target, timeout=float(timeout_s)) as r:
            raw = r.read().decode("utf-8", errors="replace")
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _compose_gate_results(
    live_gate: Optional[Dict[str, Any]],
    scr_state: Optional[Dict[str, Any]],
    live_readiness: Optional[Dict[str, Any]],
    fast_loop: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build trace `gate_results` from fresh runtime sources. Prefers live HTTP
    /live-gate, falls back to file state if the endpoint is unreachable.
    """
    out: Dict[str, Any] = {}

    if isinstance(live_gate, dict):
        for key in (
            "allow_exits_only",
            "allow_ibkr_live",
            "allow_ibkr_paper",
            "execution",
            "mode",
            "operator_mode",
            "operator_reason",
            "reasons",
            "shadow",
        ):
            if key in live_gate:
                out[key] = live_gate[key]

    # If /live-gate fetch failed or returned no shadow, splice in scr_state file.
    if "shadow" not in out and isinstance(scr_state, dict):
        out["shadow"] = {
            "state": scr_state.get("state"),
            "sizing_factor": scr_state.get("sizing_factor"),
            "paper_only": scr_state.get("paper_only"),
            "reasons": list(scr_state.get("reasons") or [])[:10],
            "stats": scr_state.get("stats") or {},
            "ts_utc": scr_state.get("ts_utc"),
        }

    if isinstance(live_readiness, dict):
        out["live_readiness"] = {
            "ready_for_live": bool(live_readiness.get("ready_for_live", False)),
            "ts_utc": live_readiness.get("ts_utc"),
        }

    if isinstance(fast_loop, dict):
        out["fast_loop"] = {
            "status": fast_loop.get("status"),
            "broker_ok": fast_loop.get("broker_ok"),
            "price_ok": fast_loop.get("price_ok"),
            "ibkr_bars_ok": fast_loop.get("ibkr_bars_ok"),
            "profit_lock_active": fast_loop.get("profit_lock_active"),
            "ts_utc": fast_loop.get("ts_utc"),
        }

    return out


def _resolve_execution_mode(live_gate: Optional[Dict[str, Any]]) -> str:
    """
    Pull current execution mode from the live-gate response, falling back to
    env vars and finally 'unknown'. Never raises.
    """
    if isinstance(live_gate, dict):
        ex = live_gate.get("execution")
        if isinstance(ex, dict):
            m = ex.get("exec_mode")
            if isinstance(m, str) and m.strip():
                return m.strip()
        md = live_gate.get("mode")
        if isinstance(md, dict):
            cm = md.get("chad_mode")
            if isinstance(cm, str) and cm.strip():
                return cm.strip()
    env_m = os.environ.get("CHAD_EXECUTION_MODE")
    return env_m.strip() if env_m and env_m.strip() else "unknown"


def _count_fills_today(repo_root: Path) -> Dict[str, Any]:
    """
    Best-effort line-count of today's executed fills from
    data/fills/FILLS_YYYYMMDD.ndjson. Returns a small summary dict; never raises.
    """
    try:
        ymd = time.strftime("%Y%m%d", time.gmtime())
        path = repo_root / "data" / "fills" / f"FILLS_{ymd}.ndjson"
        if not path.is_file():
            return {"fills_today": 0, "exists": False, "path": str(path)}
        n = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    n += 1
        return {"fills_today": int(n), "exists": True, "path": str(path)}
    except Exception:
        return {"fills_today": 0, "exists": False}


def _write_decision_trace_heartbeat(
    *,
    ts_utc: str,
    live_gate: Optional[Dict[str, Any]],
    scr_state: Optional[Dict[str, Any]],
    live_readiness: Optional[Dict[str, Any]],
    fast_loop: Optional[Dict[str, Any]],
    executions_today: Dict[str, Any],
    out_path: Path = DECISION_TRACE_HEARTBEAT_PATH,
) -> None:
    """
    Atomically write a fresh runtime/decision_trace_heartbeat.json composed
    from current live state. Replaces the dead chad/core/decision_trace_heartbeat.py
    timer-driven oneshot.
    """
    payload: Dict[str, Any] = {
        "ts_utc": str(ts_utc),
        "ok": bool(live_gate is not None),
        "source": "orchestrator",
        "live_gate": live_gate if isinstance(live_gate, dict) else None,
        "mode": (live_gate.get("mode") if isinstance(live_gate, dict) else None),
        "allow_ibkr_paper": (
            live_gate.get("allow_ibkr_paper") if isinstance(live_gate, dict) else None
        ),
        "allow_ibkr_live": (
            live_gate.get("allow_ibkr_live") if isinstance(live_gate, dict) else None
        ),
        "scr_state": (
            {
                "state": scr_state.get("state"),
                "sizing_factor": scr_state.get("sizing_factor"),
                "paper_only": scr_state.get("paper_only"),
                "ts_utc": scr_state.get("ts_utc"),
                "stats": scr_state.get("stats") or {},
            }
            if isinstance(scr_state, dict)
            else None
        ),
        "live_readiness": (
            {
                "ready_for_live": bool(live_readiness.get("ready_for_live", False)),
                "ts_utc": live_readiness.get("ts_utc"),
            }
            if isinstance(live_readiness, dict)
            else None
        ),
        "fast_loop": (
            {
                "status": fast_loop.get("status"),
                "broker_ok": fast_loop.get("broker_ok"),
                "price_ok": fast_loop.get("price_ok"),
                "ts_utc": fast_loop.get("ts_utc"),
            }
            if isinstance(fast_loop, dict)
            else None
        ),
        "executions_today": executions_today,
        # Orchestrator does not own order submission (chad-live-loop does);
        # this list is intentionally empty for trace contract compatibility.
        "submitted_orders": [],
    }
    _atomic_write_json(out_path, payload, indent=2)


# -----------------------------------------------------------------------------
# Settings / results
# -----------------------------------------------------------------------------

FAST_LOOP_STATE_PATH: Path = RUNTIME_DIR / "fast_loop_state.json"


@dataclass(frozen=True)
class OrchestratorSettings:
    daily_risk_pct: float
    loop_interval_seconds: float
    run_forever: bool
    log_level: str
    portfolio_snapshot_path: Path
    dynamic_caps_path: Path
    decision_trace_enabled: bool
    fast_loop_enabled: bool
    fast_loop_interval_seconds: float

    @classmethod
    def from_env(
        cls,
        *,
        portfolio_snapshot_path: Optional[Path] = None,
        dynamic_caps_path: Optional[Path] = None,
        log_level_override: Optional[str] = None,
    ) -> "OrchestratorSettings":
        daily_risk_pct = _finite_float(_env_float("CHAD_DAILY_RISK_PCT", 5.0), 5.0)
        daily_risk_pct = max(0.0, min(100.0, float(daily_risk_pct)))

        loop_interval_seconds = _finite_float(_env_float("CHAD_ORCH_INTERVAL_SECONDS", 60.0), 60.0)
        loop_interval_seconds = max(1.0, float(loop_interval_seconds))

        run_forever = _env_bool01("CHAD_ORCH_RUN_FOREVER", False)
        log_level = (log_level_override or _env_str("CHAD_ORCH_LOG_LEVEL", "INFO")).upper()

        snap_path = Path(portfolio_snapshot_path or PORTFOLIO_SNAPSHOT_PATH)
        caps_path = Path(dynamic_caps_path or default_dynamic_caps_path())

        # DecisionTrace is required by SSOT; default ON.
        decision_trace_enabled = _env_bool01("CHAD_DECISION_TRACE_ENABLED", True)

        # Fast loop settings
        fast_loop_enabled = _env_bool01("CHAD_FAST_LOOP_ENABLED", True)
        fast_loop_interval = _finite_float(_env_float("CHAD_FAST_LOOP_INTERVAL_SECONDS", 8.0), 8.0)
        fast_loop_interval = max(1.0, min(30.0, float(fast_loop_interval)))

        return cls(
            daily_risk_pct=daily_risk_pct,
            loop_interval_seconds=loop_interval_seconds,
            run_forever=bool(run_forever),
            log_level=str(log_level),
            portfolio_snapshot_path=snap_path,
            dynamic_caps_path=caps_path,
            decision_trace_enabled=bool(decision_trace_enabled),
            fast_loop_enabled=bool(fast_loop_enabled),
            fast_loop_interval_seconds=fast_loop_interval,
        )


@dataclass(frozen=True)
class OrchestratorCycleResult:
    trace_id: str
    cycle_id: str
    total_equity: float
    daily_risk_fraction: float
    portfolio_risk_cap: float
    dynamic_caps_path: Path
    dynamic_caps_hash: str
    used_fallback_snapshot: bool
    allocator_mode: str


# -----------------------------------------------------------------------------
# Allocation Strategy Pattern (DI-friendly)
# -----------------------------------------------------------------------------

class AllocationStrategy(Protocol):
    """
    Strategy Pattern: produce adjusted raw weights and write an audit receipt (best-effort).

    Implementations MUST:
    - never increase the total risk budget (handled later by DynamicRiskAllocator)
    - fail closed (raise -> caller falls back to base)
    """
    name: str

    def apply(
        self,
        *,
        repo_root: Path,
        base_weights: Mapping[str, float],
        log: logging.Logger,
    ) -> Mapping[str, float]:
        ...


@dataclass(frozen=True)
class SavageAllocatorStrategy:
    name: str = "SAVAGE"

    def apply(
        self,
        *,
        repo_root: Path,
        base_weights: Mapping[str, float],
        log: logging.Logger,
    ) -> Mapping[str, float]:
        cfg = SavageConfig.from_env()
        adjusted, report = apply_savage_overlay(repo_root=repo_root, base_weights=base_weights, cfg=cfg)
        try:
            write_state_report_atomic(repo_root=repo_root, report=report)
        except Exception as exc:  # noqa: BLE001
            log.warning("orchestrator.savage_receipt_write_failed", extra={"error": str(exc)})
        return adjusted


@dataclass(frozen=True)
class AllocV3Strategy:
    name: str = "V3"

    def apply(
        self,
        *,
        repo_root: Path,
        base_weights: Mapping[str, float],
        log: logging.Logger,
    ) -> Mapping[str, float]:
        prev_state = _safe_json_load(repo_root / "runtime" / "allocator_v3_state.json")
        cfg = AllocV3Config.from_env()
        adjusted, state = compute_allocator_v3(
            repo_root=repo_root,
            base_weights=base_weights,
            cfg=cfg,
            prev_state=prev_state,
        )
        try:
            write_allocator_v3_state_atomic(repo_root, state)
        except Exception as exc:  # noqa: BLE001
            log.warning("orchestrator.v3_receipt_write_failed", extra={"error": str(exc)})
        return adjusted


def _allocator_factory():
    from chad.risk.correlation_strategy import CorrelationOverlayStrategy
    return CorrelationOverlayStrategy()
# -----------------------------------------------------------------------------
# DecisionTrace recorder (best-effort, SSOT-aligned)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionTraceConfig:
    enabled: bool
    traces_dir: Path
    filename_prefix: str = "decision_trace"

    def trace_path(self) -> Path:
        return self.traces_dir / f"{self.filename_prefix}_{_utc_ymd()}.ndjson"


class DecisionTraceRecorder:
    """
    Append-only DecisionTrace recorder.

    NOTE: This is intentionally best-effort:
      - It must never break orchestrator publishing.
      - Any error is logged and swallowed.
    """

    def __init__(self, cfg: DecisionTraceConfig) -> None:
        self._cfg = cfg

    def record_cycle(
        self,
        *,
        trace_id: str,
        cycle_id: str,
        ts_utc: str,
        allocator_mode: str,
        snapshot_used_fallback: bool,
        snapshot_path: str,
        dynamic_caps_path: str,
        dynamic_caps_hash: str,
        dynamic_caps: Dict[str, Any],
        execution_mode: str = "unknown",
        gate_results: Optional[Dict[str, Any]] = None,
        submitted_orders: Optional[list] = None,
        strategy_detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._cfg.enabled:
            return

        # Minimal SSOT-grade record: one per cycle, reconstructable without code.
        rec: Dict[str, Any] = {
            "schema_version": 3,
            "trace_id": str(trace_id),
            "cycle_id": str(cycle_id),
            "ts_utc": str(ts_utc),
            "component": "orchestrator",
            "mode": str(execution_mode),
            "allocator_mode": str(allocator_mode),
            "gate_results": gate_results if gate_results else {},
            "submitted_orders": submitted_orders if submitted_orders is not None else [],
            "inputs": {
                "portfolio_snapshot_path": str(snapshot_path),
                "used_fallback_snapshot": bool(snapshot_used_fallback),
            },
            "artifacts": {
                "dynamic_caps_path": str(dynamic_caps_path),
                "dynamic_caps_hash": f"sha256:{dynamic_caps_hash}",
                "trace_file_hint": "data/traces/decision_trace_YYYYMMDD.ndjson",
            },
            "caps_summary": {
                "total_equity": _finite_float(dynamic_caps.get("total_equity"), 0.0),
                "daily_risk_fraction": _finite_float(dynamic_caps.get("daily_risk_fraction"), 0.0),
                "portfolio_risk_cap": _finite_float(dynamic_caps.get("portfolio_risk_cap"), 0.0),
                "strategy_caps_keys": sorted(list((dynamic_caps.get("strategy_caps") or {}).keys())),
            },
            "strategy_detail": strategy_detail if strategy_detail is not None else {
                "available_strategies": {},
                "rejected_strategies": {},
                "selected_strategy": None,
                "selected_strategy_reason": None,
                "affordability_rejections": [],
                "guard_rejections": [],
            },
            "notes": {
                "deterministic": True,
                "no_broker_calls": True,
                "best_effort_trace": True,
            },
        }

        try:
            _append_ndjson_locked(self._cfg.trace_path(), rec)
        except Exception:
            return


# -----------------------------------------------------------------------------
# Orchestrator core
# -----------------------------------------------------------------------------

class Orchestrator:
    """
    Risk-budget publisher:
      snapshot -> allocation overlay -> DynamicRiskAllocator -> runtime/dynamic_caps.json
      + DecisionTrace record per cycle (best-effort)
    """

    def __init__(
        self,
        settings: OrchestratorSettings,
        *,
        repo_root: Path = REPO_ROOT,
        trace_recorder: Optional[DecisionTraceRecorder] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._settings = settings
        self._repo_root = repo_root
        self._log = log or logger
        self._trace = trace_recorder or DecisionTraceRecorder(
            DecisionTraceConfig(enabled=settings.decision_trace_enabled, traces_dir=TRACES_DIR)
        )

    @property
    def settings(self) -> OrchestratorSettings:
        return self._settings

    # --------------------------
    # Snapshot loader
    # --------------------------

    def _load_portfolio_snapshot(self) -> Tuple[PortfolioSnapshot, bool]:
        """
        Load snapshot from JSON if available, else fallback env vars.

        Supported keys:
          - ibkr_equity / coinbase_equity / kraken_equity
          - legacy: ibkr_equity_usd / coinbase_equity_usd / total_equity_usd
        """
        path = self._settings.portfolio_snapshot_path
        obj = _safe_json_load(path)

        if obj is not None:
            try:
                ibkr_raw = obj.get("ibkr_equity_usd", obj.get("ibkr_equity", 0.0))
                coin_raw = obj.get("coinbase_equity_usd", obj.get("coinbase_equity", 0.0))
                kraken_raw = obj.get("kraken_equity_usd", obj.get("kraken_equity", 0.0))

                snap = PortfolioSnapshot(
                    ibkr_equity=_finite_float(ibkr_raw, 0.0),
                    coinbase_equity=_finite_float(coin_raw, 0.0),
                    kraken_equity=_finite_float(kraken_raw, 0.0),
                )

                self._log.info(
                    "orchestrator.portfolio_snapshot_loaded",
                    extra={
                        "path": str(path),
                        "ibkr_equity": snap.ibkr_equity,
                        "coinbase_equity": snap.coinbase_equity,
                        "kraken_equity": snap.kraken_equity,
                        "total_equity": snap.total_equity,
                    },
                )
                return snap, False
            except Exception as exc:  # noqa: BLE001
                self._log.warning("orchestrator.portfolio_snapshot_invalid", extra={"path": str(path), "error": str(exc)})

        # fallback
        ibkr_f = _env_float("CHAD_IBKR_EQUITY_FALLBACK", 0.0)
        coin_f = _env_float("CHAD_COINBASE_EQUITY_FALLBACK", 0.0)
        kraken_f = _env_float("CHAD_KRAKEN_EQUITY_FALLBACK", 0.0)

        snap = PortfolioSnapshot(
            ibkr_equity=max(0.0, float(_finite_float(ibkr_f, 0.0))),
            coinbase_equity=max(0.0, float(_finite_float(coin_f, 0.0))),
            kraken_equity=max(0.0, float(_finite_float(kraken_f, 0.0))),
        )

        self._log.info(
            "orchestrator.portfolio_snapshot_fallback",
            extra={
                "ibkr_equity": snap.ibkr_equity,
                "coinbase_equity": snap.coinbase_equity,
                "kraken_equity": snap.kraken_equity,
                "total_equity": snap.total_equity,
            },
        )
        return snap, True

    # --------------------------
    # Allocation -> allocator
    # --------------------------

    def _build_allocator(self) -> Tuple[DynamicRiskAllocator, str]:
        """
        Build DynamicRiskAllocator with selected allocation overlay.

        Fail-closed:
        - Any overlay failure => base weights.
        """
        base = StrategyAllocation.from_env_or_default()
        allocation = base
        strategy = _allocator_factory()
        mode = getattr(strategy, "name", "unknown")

        try:
            adjusted = strategy.apply(repo_root=self._repo_root, base_weights=base.weights, log=self._log)

            # Preserve base zero weights (never resurrect)
            adjusted_clean: Dict[str, float] = {}
            for k, bw in base.weights.items():
                if float(bw) <= 0.0:
                    adjusted_clean[k] = 0.0
                else:
                    adjusted_clean[k] = max(0.0, float(adjusted.get(k, bw)))
            # 50/30/20 chassis enforcement (after overlays, before cap build)
            adjusted_clean = enforce_chassis(adjusted_clean)
            allocation = StrategyAllocation(weights=adjusted_clean)

            self._log.info("orchestrator.allocation_overlay_applied", extra={"mode": mode})
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "orchestrator.allocation_overlay_failed_fallback_to_base",
                extra={"mode": mode, "error": str(exc)},
            )
            allocation = StrategyAllocation(weights=enforce_chassis(base.weights))

        daily_fraction = float(self._settings.daily_risk_pct) / 100.0
        daily_fraction = max(0.0, min(1.0, float(_finite_float(daily_fraction, 0.0))))

        return DynamicRiskAllocator(
            strategy_allocation=allocation,
            daily_risk_fraction=daily_fraction,
        ), mode

    # --------------------------
    # Core publish action
    # --------------------------

    def refresh_dynamic_caps(self) -> OrchestratorCycleResult:
        """
        The one critical action:
          snapshot -> allocator payload -> atomic write runtime/dynamic_caps.json
          + DecisionTrace record (best-effort)
        """
        ts_utc = _utc_now_iso()
        trace_id = str(uuid.uuid4())
        cycle_id = ts_utc.replace("-", "").replace(":", "")  # compact but stable

        self._log.info(
            "orchestrator.cycle_start",
            extra={
                "trace_id": trace_id,
                "cycle_id": cycle_id,
                "daily_risk_pct": self._settings.daily_risk_pct,
                "dynamic_caps_path": str(self._settings.dynamic_caps_path),
            },
        )

        snapshot, used_fallback = self._load_portfolio_snapshot()
        allocator, allocator_mode = self._build_allocator()

        # Domain 2: gather fresh runtime state for trace enrichment (best-effort).
        # Replaces the previous reads of full_execution_cycle_last.json and
        # decision_trace_heartbeat.json which had been stale since their writers
        # stopped running. None of these calls can break the publish path.
        _live_gate = _fetch_live_gate_snapshot()
        _scr_state = _safe_json_load(RUNTIME_DIR / "scr_state.json")
        _live_readiness = _safe_json_load(RUNTIME_DIR / "live_readiness.json")
        _fast_loop = _safe_json_load(RUNTIME_DIR / "fast_loop_state.json")
        _route_decision = _safe_json_load(RUNTIME_DIR / "last_route_decision.json")
        _gate_results = _compose_gate_results(_live_gate, _scr_state, _live_readiness, _fast_loop)
        _exec_mode = _resolve_execution_mode(_live_gate)
        _executions_today = _count_fills_today(self._repo_root)

        payload = allocator.build_payload(snapshot=snapshot)

        out_path = self._settings.dynamic_caps_path
        _atomic_write_json(out_path, payload, indent=2)

        # Publish to Redis state bus (non-blocking, fail-soft)
        try:
            from chad.core.state_bus import get_publisher
            get_publisher().publish_dynamic_caps(payload)
        except Exception:
            pass

        caps_hash = _sha256_json(payload)

        total_equity = float(_finite_float(payload.get("total_equity"), 0.0))
        daily_risk_fraction = float(_finite_float(payload.get("daily_risk_fraction"), 0.0))
        portfolio_risk_cap = float(_finite_float(payload.get("portfolio_risk_cap"), 0.0))

        # Best-effort: write a fresh runtime/decision_trace_heartbeat.json so
        # operator surface and downstream readers see live state instead of the
        # frozen snapshot left behind by the dead heartbeat oneshot. MUST NOT
        # break orchestrator publishing.
        try:
            _write_decision_trace_heartbeat(
                ts_utc=ts_utc,
                live_gate=_live_gate,
                scr_state=_scr_state,
                live_readiness=_live_readiness,
                fast_loop=_fast_loop,
                executions_today=_executions_today,
            )
        except Exception:
            pass

        # DecisionTrace (best-effort, cannot fail publish)
        try:
            self._trace.record_cycle(
                trace_id=trace_id,
                cycle_id=cycle_id,
                ts_utc=ts_utc,
                allocator_mode=allocator_mode,
                snapshot_used_fallback=used_fallback,
                snapshot_path=str(self._settings.portfolio_snapshot_path),
                dynamic_caps_path=str(out_path),
                dynamic_caps_hash=caps_hash,
                dynamic_caps=payload,
                execution_mode=_exec_mode,
                gate_results=_gate_results,
                # Orchestrator no longer owns order submission; chad-live-loop
                # is the execution path. Submission counts surface via the
                # decision_trace_heartbeat.json executions_today block instead.
                submitted_orders=[],
                strategy_detail=_route_decision if isinstance(_route_decision, dict) else None,
            )
        except Exception:
            # absolute fail-soft
            pass

        self._log.info(
            "orchestrator.cycle_summary",
            extra={
                "trace_id": trace_id,
                "cycle_id": cycle_id,
                "allocator_mode": allocator_mode,
                "total_equity": total_equity,
                "daily_risk_fraction": daily_risk_fraction,
                "portfolio_risk_cap": portfolio_risk_cap,
                "dynamic_caps_hash": f"sha256:{caps_hash}",
                "used_fallback_snapshot": used_fallback,
            },
        )

        return OrchestratorCycleResult(
            trace_id=trace_id,
            cycle_id=cycle_id,
            total_equity=total_equity,
            daily_risk_fraction=daily_risk_fraction,
            portfolio_risk_cap=portfolio_risk_cap,
            dynamic_caps_path=out_path,
            dynamic_caps_hash=caps_hash,
            used_fallback_snapshot=used_fallback,
            allocator_mode=allocator_mode,
        )

    # --------------------------
    # Async runners
    # --------------------------

    async def run_once(self) -> OrchestratorCycleResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.refresh_dynamic_caps)

    async def run_forever(self) -> None:
        interval = float(self._settings.loop_interval_seconds)
        self._log.info(
            "orchestrator.loop_start",
            extra={
                "interval_seconds": interval,
                "daily_risk_pct": self._settings.daily_risk_pct,
                "portfolio_snapshot_path": str(self._settings.portfolio_snapshot_path),
                "dynamic_caps_path": str(self._settings.dynamic_caps_path),
                "decision_trace_enabled": self._settings.decision_trace_enabled,
                "fast_loop_enabled": self._settings.fast_loop_enabled,
                "fast_loop_interval_seconds": self._settings.fast_loop_interval_seconds,
            },
        )

        while True:
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                self._log.exception("orchestrator.cycle_error", extra={"error": str(exc)})
            await asyncio.sleep(interval)

    # --------------------------
    # Fast loop (8-second cadence)
    # --------------------------

    def _run_fast_loop_cycle(self) -> Dict[str, Any]:
        """
        Fast loop cycle: broker sync, price freshness, risk monitoring.

        This runs at 8-second cadence for latency-sensitive operations:
        - Broker position sync (read portfolio_snapshot.json freshness)
        - Price cache freshness check
        - Profit lock / stop monitoring readiness
        - Write runtime/fast_loop_state.json

        Non-fatal: any error is logged and swallowed.
        """
        ts_utc = _utc_now_iso()

        # Check broker position freshness
        snap = _safe_json_load(self._settings.portfolio_snapshot_path)
        broker_ok = snap is not None
        snap_age_s = -1.0
        if snap and isinstance(snap.get("ts_utc"), str):
            try:
                snap_ts = snap["ts_utc"]
                # Parse ISO timestamp to compute age
                import re
                # Simple epoch diff: compare formatted strings
                snap_age_s = -1.0  # unknown unless we can parse
            except Exception:
                snap_age_s = -1.0

        # Check price cache freshness
        price_cache = _safe_json_load(RUNTIME_DIR / "price_cache.json")
        price_ok = price_cache is not None
        price_symbols = 0
        if price_cache and isinstance(price_cache.get("prices"), dict):
            price_symbols = len(price_cache["prices"])

        # Check IBKR bars cache freshness
        ibkr_bars = _safe_json_load(RUNTIME_DIR / "ibkr_bars_cache.json")
        ibkr_bars_ok = ibkr_bars is not None
        ibkr_bar_count = 0
        if ibkr_bars and isinstance(ibkr_bars.get("symbols"), dict):
            ibkr_bar_count = sum(len(v) for v in ibkr_bars["symbols"].values() if isinstance(v, list))

        # Check profit lock state
        profit_lock = _safe_json_load(RUNTIME_DIR / "profit_lock_state.json")
        profit_lock_active = bool(profit_lock and profit_lock.get("locked"))

        # Build state
        state: Dict[str, Any] = {
            "ts_utc": ts_utc,
            "fast_loop_interval_seconds": self._settings.fast_loop_interval_seconds,
            "cycle_count": getattr(self, "_fast_loop_cycle_count", 0) + 1,
            "broker_ok": broker_ok,
            "price_ok": price_ok,
            "price_symbols": price_symbols,
            "ibkr_bars_ok": ibkr_bars_ok,
            "ibkr_bar_count": ibkr_bar_count,
            "profit_lock_active": profit_lock_active,
            "status": "ok",
        }

        # Persist cycle count
        self._fast_loop_cycle_count = state["cycle_count"]  # type: ignore[attr-defined]

        # Write state file
        _atomic_write_json(FAST_LOOP_STATE_PATH, state)

        # Publish fast loop heartbeat to Redis (non-blocking, fail-soft)
        try:
            from chad.core.state_bus import get_publisher
            get_publisher().publish_fast_loop(state)
        except Exception:
            pass

        return state

    async def _run_fast_loop(self) -> None:
        """
        Fast loop coroutine: runs independently at 8-second cadence.

        Non-fatal: any error is logged and the loop continues.
        Never kills the signal loop.
        """
        interval = float(self._settings.fast_loop_interval_seconds)
        self._fast_loop_cycle_count = 0  # type: ignore[attr-defined]

        self._log.info(
            "orchestrator.fast_loop_start",
            extra={"interval_seconds": interval},
        )

        while True:
            try:
                loop = asyncio.get_running_loop()
                state = await loop.run_in_executor(None, self._run_fast_loop_cycle)

                if state.get("cycle_count", 0) % 50 == 1:
                    # Log every 50th cycle (~7 minutes) to avoid spam
                    self._log.info(
                        "orchestrator.fast_loop_heartbeat",
                        extra={
                            "cycle_count": state.get("cycle_count"),
                            "broker_ok": state.get("broker_ok"),
                            "price_ok": state.get("price_ok"),
                            "ibkr_bars_ok": state.get("ibkr_bars_ok"),
                        },
                    )
                    # Write Redis health state (best-effort, ~every 7 min)
                    try:
                        from chad.core.state_bus import write_redis_state_json
                        write_redis_state_json()
                    except Exception:
                        pass
            except Exception as exc:  # noqa: BLE001
                self._log.warning(
                    "orchestrator.fast_loop_error",
                    extra={"error": str(exc)},
                )
            await asyncio.sleep(interval)


# -----------------------------------------------------------------------------
# CLI / bootstrap
# -----------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHAD Orchestrator — publish runtime/dynamic_caps.json (+ DecisionTrace)")
    p.add_argument("--once", action="store_true", help="Run one cycle then exit.")
    p.add_argument("--log-level", default=None, help="Override CHAD_ORCH_LOG_LEVEL.")
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    settings = OrchestratorSettings.from_env(log_level_override=args.log_level)
    _configure_logging(settings.log_level)

    orch = Orchestrator(settings)

    if args.once or not settings.run_forever:
        await orch.run_once()
        return 0

    # Launch signal loop + fast loop concurrently
    tasks = [asyncio.create_task(orch.run_forever())]
    if settings.fast_loop_enabled:
        tasks.append(asyncio.create_task(orch._run_fast_loop()))
        logger.info(
            "orchestrator.tiered_loop_enabled",
            extra={
                "signal_loop_interval": settings.loop_interval_seconds,
                "fast_loop_interval": settings.fast_loop_interval_seconds,
            },
        )

    await asyncio.gather(*tasks)
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

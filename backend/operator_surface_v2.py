"""
backend/operator_surface_v2.py

CHAD Operator Surface V2 (Phase 10)
A hardened, audit-first, read-only operator API surface.

This is a clean rebuild (V2) designed to be mounted into any FastAPI app:
    from backend.operator_surface_v2 import build_operator_router
    app.include_router(build_operator_router(), prefix="")

Non-negotiable guarantees:
- No broker calls
- No state mutation (no file writes, no config edits)
- Fail-soft behavior (stable response shapes even if artifacts are missing)
- Bounded output (Telegram/dashboard safe)
- Caching with TTL (prevents expensive system calls per request)
- Safe subprocess calls (timeouts, never raises)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import anyio
from fastapi import APIRouter
from pydantic import BaseModel, Field


# ----------------------------
# Small utilities
# ----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _bounded_list(items: Sequence[Any], limit: int) -> List[Any]:
    if limit <= 0:
        return []
    return list(items[:limit])


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# ----------------------------
# TTL cache (thread-safe enough for FastAPI sync usage; no shared mutation outside GIL)
# ----------------------------

@dataclass
class _CacheEntry:
    expires_at: float
    value: Any


class TTLCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = float(ttl_seconds)
        self._store: Dict[str, _CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        ent = self._store.get(key)
        if ent is None:
            return None
        if now >= ent.expires_at:
            self._store.pop(key, None)
            return None
        return ent.value

    def set(self, key: str, value: Any) -> None:
        now = time.monotonic()
        self._store[key] = _CacheEntry(expires_at=now + self._ttl, value=value)


# ----------------------------
# Safe subprocess execution
# ----------------------------

@dataclass(frozen=True)
class ExecResult:
    rc: int
    out: str
    err: str


async def safe_run(cmd: Sequence[str], *, cwd: Optional[Path], timeout_s: float) -> ExecResult:
    """
    Runs subprocess in a worker thread. Never raises.
    """
    def _run() -> ExecResult:
        try:
            p = subprocess.run(
                list(cmd),
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return ExecResult(int(p.returncode), p.stdout or "", p.stderr or "")
        except Exception as exc:
            return ExecResult(255, "", f"exec_error:{type(exc).__name__}:{exc}")

    return await anyio.to_thread.run_sync(_run)


# ----------------------------
# File access (read-only)
# ----------------------------

@dataclass(frozen=True)
class JsonRead:
    ok: bool
    error: Optional[str]
    data: Dict[str, Any]


def read_json_dict(path: Path) -> JsonRead:
    """
    Reads a JSON file and guarantees dict shape. Never raises.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return JsonRead(False, "missing", {})
    except Exception as exc:
        return JsonRead(False, f"read_error:{type(exc).__name__}", {})

    try:
        obj = json.loads(raw)
    except Exception as exc:
        return JsonRead(False, f"json_parse_error:{type(exc).__name__}", {})

    if not isinstance(obj, dict):
        return JsonRead(False, "invalid_shape:not_dict", {})
    return JsonRead(True, None, obj)


def file_meta(path: Path) -> Dict[str, Any]:
    """
    Never raises. Includes ts_utc/ttl_seconds if present.
    """
    meta: Dict[str, Any] = {"path": str(path)}
    try:
        st = path.stat()
    except FileNotFoundError:
        meta["exists"] = False
        return meta
    except Exception as exc:
        meta["exists"] = False
        meta["error"] = f"stat_error:{type(exc).__name__}"
        return meta

    meta["exists"] = True
    meta["size_bytes"] = int(st.st_size)
    meta["mtime_epoch"] = float(st.st_mtime)

    j = read_json_dict(path)
    meta["json_ok"] = j.ok
    if not j.ok:
        meta["json_error"] = j.error
        return meta

    if "ts_utc" in j.data:
        meta["ts_utc"] = j.data.get("ts_utc")
    if "ttl_seconds" in j.data:
        meta["ttl_seconds"] = j.data.get("ttl_seconds")
    return meta


# ----------------------------
# Config + DI
# ----------------------------

@dataclass(frozen=True)
class OperatorConfig:
    repo_dir: Path
    ssot_runtime_dir: Path
    timers: Tuple[str, ...]
    max_reasons: int
    max_failed_units: int
    systemd_timeout_s: float
    git_timeout_s: float
    cache_ttl_s: float


def default_config() -> OperatorConfig:
    repo_dir = Path(os.environ.get("CHAD_REPO_DIR", "/home/ubuntu/chad_finale"))
    ssot_runtime_dir = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
    return OperatorConfig(
        repo_dir=repo_dir,
        ssot_runtime_dir=ssot_runtime_dir,
        timers=("chad-scr-sync.timer", "chad-tier-sync.timer"),
        max_reasons=10,
        max_failed_units=50,
        systemd_timeout_s=2.0,
        git_timeout_s=2.0,
        cache_ttl_s=3.0,  # tiny TTL to avoid spamming systemctl/git
    )


class OperatorService:
    """
    Core service implementing operator queries.
    """

    def __init__(self, cfg: OperatorConfig) -> None:
        self.cfg = cfg
        self.cache = TTLCache(cfg.cache_ttl_s)

    # ---- git fingerprint (cached) ----
    async def git_fingerprint(self) -> Dict[str, Any]:
        cached = self.cache.get("git")
        if cached is not None:
            return cached

        if not self.cfg.repo_dir.exists():
            out = {"ok": False, "repo_dir": str(self.cfg.repo_dir), "error": "repo_dir_missing"}
            self.cache.set("git", out)
            return out

        head = await safe_run(["git", "rev-parse", "HEAD"], cwd=self.cfg.repo_dir, timeout_s=1.5)
        if head.rc != 0:
            out = {"ok": False, "repo_dir": str(self.cfg.repo_dir), "error": head.err.strip()[:300], "rc": head.rc}
            self.cache.set("git", out)
            return out

        st = await safe_run(["git", "status", "--porcelain"], cwd=self.cfg.repo_dir, timeout_s=2.0)
        dirty = bool(st.out.strip()) if st.rc == 0 else None

        out = {
            "ok": True,
            "repo_dir": str(self.cfg.repo_dir),
            "head_sha": head.out.strip(),
            "dirty": dirty,
            "status_rc": st.rc,
            "status_err": st.err.strip()[:200] if st.rc != 0 else None,
        }
        self.cache.set("git", out)
        return out

    # ---- systemd status (cached) ----
    async def systemd_is_active(self, unit: str) -> Dict[str, Any]:
        key = f"active:{unit}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        r = await safe_run(["systemctl", "is-active", unit], cwd=self.cfg.repo_dir, timeout_s=self.cfg.systemd_timeout_s)
        out = {"unit": unit, "rc": r.rc, "state": (r.out.strip() if r.out else None), "err": (r.err.strip()[:200] if r.rc != 0 else None)}
        self.cache.set(key, out)
        return out

    async def systemd_is_enabled(self, unit: str) -> Dict[str, Any]:
        key = f"enabled:{unit}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        r = await safe_run(["systemctl", "is-enabled", unit], cwd=self.cfg.repo_dir, timeout_s=self.cfg.systemd_timeout_s)
        out = {"unit": unit, "rc": r.rc, "state": (r.out.strip() if r.out else None), "err": (r.err.strip()[:200] if r.rc != 0 else None)}
        self.cache.set(key, out)
        return out

    async def failed_units(self) -> Dict[str, Any]:
        key = "failed_units"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        r = await safe_run(["systemctl", "--failed", "--no-pager"], cwd=self.cfg.repo_dir, timeout_s=self.cfg.systemd_timeout_s)
        if r.rc != 0:
            out = {"ok": False, "rc": r.rc, "error": r.err.strip()[:300], "failed_units": []}
            self.cache.set(key, out)
            return out

        failed: List[str] = []
        for line in r.out.splitlines():
            line = line.strip()
            if line.startswith("chad-"):
                failed.append(line.split()[0])
        failed = list(dict.fromkeys(failed))[: self.cfg.max_failed_units]
        out = {"ok": True, "rc": 0, "failed_units": failed}
        self.cache.set(key, out)
        return out

    # ---- runtime paths ----
    def runtime_paths(self) -> Dict[str, Path]:
        # repo runtime (legacy)
        rt = self.cfg.repo_dir / "runtime"
        # ssot runtime (absolute)
        ss = self.cfg.ssot_runtime_dir
        return {
            "feed_state": rt / "feed_state.json",
            "positions_snapshot": rt / "positions_snapshot.json",
            "reconciliation_state": rt / "reconciliation_state.json",
            "dynamic_caps": rt / "dynamic_caps.json",
            "operator_intent": rt / "operator_intent.json",
            "portfolio_snapshot": rt / "portfolio_snapshot.json",
            "decision_trace_heartbeat": rt / "decision_trace_heartbeat.json",
            "scr_state": ss / "scr_state.json",
            "tier_state": ss / "tier_state.json",
        }

    # ---- explainers ----
    def risk_explain(self) -> Dict[str, Any]:
        caps_path = self.runtime_paths()["dynamic_caps"]
        scr_path = self.runtime_paths()["scr_state"]

        caps = read_json_dict(caps_path)
        scr = read_json_dict(scr_path)

        total_equity = _safe_float(caps.data.get("total_equity"))
        daily_risk_fraction = _safe_float(caps.data.get("daily_risk_fraction"))
        portfolio_risk_cap = _safe_float(caps.data.get("portfolio_risk_cap"))

        sc = caps.data.get("strategy_caps")
        if not isinstance(sc, dict):
            sc = {}

        denom = portfolio_risk_cap if portfolio_risk_cap > 0 else 0.0
        rows: List[Dict[str, Any]] = []
        for name in sorted(sc.keys()):
            cap = _safe_float(sc.get(name))
            pct = round((cap / denom) * 100.0, 4) if denom > 0 else None
            rows.append({"strategy": str(name), "cap_usd": cap, "cap_pct_of_portfolio_risk_cap": pct})

        posture = []
        if caps.ok:
            posture.append(f"total_equity=${total_equity:,.2f}")
            posture.append(f"daily_risk_fraction={daily_risk_fraction:.4f}")
            posture.append(f"portfolio_risk_cap≈${portfolio_risk_cap:,.2f}")
        else:
            posture.append(f"dynamic_caps=ERROR({caps.error})")

        why: List[str] = []
        if scr.ok:
            why.append(f"SCR.state={scr.data.get('state')}")
            if bool(scr.data.get("paper_only")):
                why.append("SCR.paper_only=true")
            rs = scr.data.get("reasons")
            if isinstance(rs, list):
                why.extend([str(x) for x in rs[: self.cfg.max_reasons]])
        else:
            why.append(f"SCR=ERROR({scr.error})")

        return {
            "ts_utc": utc_now_iso(),
            "schema_version": "risk_explain.v2",
            "dynamic_caps": {
                "ok": caps.ok,
                "error": caps.error,
                "total_equity": total_equity,
                "daily_risk_fraction": daily_risk_fraction,
                "portfolio_risk_cap": portfolio_risk_cap,
            },
            "strategy_caps": rows,
            "posture_summary": " | ".join(posture),
            "why_blocked_summary": why[: 12],
        }

    def what_if_caps(self, *, equity: float, daily_risk_fraction: float) -> Dict[str, Any]:
        caps_path = self.runtime_paths()["dynamic_caps"]
        caps = read_json_dict(caps_path)

        if equity <= 0:
            return {"ts_utc": utc_now_iso(), "schema_version": "what_if_caps.v2", "ok": False, "error": "invalid_equity"}

        if daily_risk_fraction <= 0 or daily_risk_fraction > 1.0:
            return {"ts_utc": utc_now_iso(), "schema_version": "what_if_caps.v2", "ok": False, "error": "invalid_daily_risk_fraction"}

        sc = caps.data.get("strategy_caps")
        prc = _safe_float(caps.data.get("portfolio_risk_cap"))
        if not isinstance(sc, dict) or prc <= 0:
            return {"ts_utc": utc_now_iso(), "schema_version": "what_if_caps.v2", "ok": False, "error": f"missing_strategy_fractions:{caps.error or unknown}"}

        fractions: Dict[str, float] = {}
        for k, v in sc.items():
            frac = _safe_float(v) / prc if prc > 0 else 0.0
            if frac > 0:
                fractions[str(k)] = frac

        portfolio_risk_cap = float(equity) * float(daily_risk_fraction)
        rows = [{"strategy": k, "fraction": round(fractions[k], 6), "cap_usd": portfolio_risk_cap * fractions[k]} for k in sorted(fractions.keys())]

        return {
            "ts_utc": utc_now_iso(),
            "schema_version": "what_if_caps.v2",
            "ok": True,
            "inputs": {"equity": float(equity), "daily_risk_fraction": float(daily_risk_fraction)},
            "portfolio_risk_cap": portfolio_risk_cap,
            "strategy_caps": rows,
            "source_fractions": {"ok": True, "from": str(caps_path), "count": len(rows)},
        }

    def perf_snapshot(self) -> Dict[str, Any]:
        paths = self.runtime_paths()
        scr = read_json_dict(paths["scr_state"])
        port = read_json_dict(paths["portfolio_snapshot"])
        hb = read_json_dict(paths["decision_trace_heartbeat"])

        stats = scr.data.get("stats") if scr.ok and isinstance(scr.data.get("stats"), dict) else {}
        reasons = scr.data.get("reasons") if scr.ok and isinstance(scr.data.get("reasons"), list) else []
        reasons = [str(x) for x in reasons[: self.cfg.max_reasons]]

        summary: Dict[str, Any] = {}
        if port.ok:
            for k in ("ibkr_equity", "kraken_equity", "coinbase_equity", "total_equity", "ts_utc", "ttl_seconds"):
                if k in port.data:
                    summary[k] = port.data.get(k)

        narrative: List[str] = []
        if scr.ok:
            narrative.append(f"SCR.state={scr.data.get('state')}")
            narrative.append(f"SCR.paper_only={bool(scr.data.get('paper_only'))}")
            narrative.append(f"SCR.sizing_factor={_safe_float(scr.data.get('sizing_factor')):.4f}")

        def _sf(k: str) -> Optional[float]:
            v = stats.get(k)
            try:
                return None if v is None else float(v)
            except Exception:
                return None

        wr = _sf("win_rate")
        sh = _sf("sharpe_like")
        dd = _sf("max_drawdown")
        pnl = _sf("total_pnl")
        tt = stats.get("total_trades")

        if wr is not None: narrative.append(f"win_rate={wr:.3f}")
        if sh is not None: narrative.append(f"sharpe_like={sh:.3f}")
        if dd is not None: narrative.append(f"max_drawdown={dd:.2f}")
        if pnl is not None: narrative.append(f"total_pnl={pnl:.2f}")
        if isinstance(tt, (int, float)): narrative.append(f"total_trades={int(tt)}")

        return {
            "ts_utc": utc_now_iso(),
            "schema_version": "perf_snapshot.v2",
            "scr": {"ok": scr.ok, "error": scr.error, "state": scr.data.get("state"), "paper_only": scr.data.get("paper_only"), "sizing_factor": scr.data.get("sizing_factor"), "stats": stats, "reasons": reasons},
            "portfolio_snapshot": {"ok": port.ok, "error": port.error, "summary": summary},
            "decision_trace_heartbeat": {"ok": hb.ok, "error": hb.error, "data": hb.data if hb.ok else {}},
            "operator_narrative": " | ".join(narrative),
        }

    def brief(self) -> Dict[str, Any]:
        caps = self.risk_explain()
        perf = self.perf_snapshot()

        # Short, deterministic, human-readable text block
        lines: List[str] = []
        lines.append(f"CHAD Brief @ {utc_now_iso()} UTC")
        lines.append(caps.get("posture_summary", ""))
        wb = caps.get("why_blocked_summary") or []
        if isinstance(wb, list) and wb:
            lines.append("Why blocked: " + " ".join([str(x) for x in wb[:4]]))
        lines.append("Perf: " + (perf.get("operator_narrative") or "UNKNOWN"))
        return {"ts_utc": utc_now_iso(), "schema_version": "brief.v2", "ok": True, "text": "\n".join([x for x in lines if x])}

    def why_blocked(self) -> Dict[str, Any]:
        op = read_json_dict(self.runtime_paths()["operator_intent"])
        scr = read_json_dict(self.runtime_paths()["scr_state"])
        # best-effort hint: if live_gate.json exists
        lg = read_json_dict(self.cfg.repo_dir / "runtime" / "live_gate.json")

        reasons: List[str] = []
        reasons.append(f"OperatorIntent={op.data.get('mode')}" if op.ok else "OperatorIntent=MISSING")
        if scr.ok:
            reasons.append(f"SCR.state={scr.data.get('state')}")
            if bool(scr.data.get("paper_only")):
                reasons.append("SCR.paper_only=true")
            rs = scr.data.get("reasons")
            if isinstance(rs, list):
                reasons.extend([str(x) for x in rs[: self.cfg.max_reasons]])
        else:
            reasons.append("SCR=MISSING")

        if lg.ok:
            rs2 = lg.data.get("reasons")
            if isinstance(rs2, list):
                reasons.extend([str(x) for x in rs2[: self.cfg.max_reasons]])

        return {
            "ts_utc": utc_now_iso(),
            "schema_version": "why_blocked.v2",
            "summary": " | ".join(reasons[:12]),
            "operator_intent": {"ok": op.ok, "error": op.error, "data": op.data},
            "scr_state": {"ok": scr.ok, "error": scr.error, "data": scr.data},
            "live_gate_hint": {"ok": lg.ok, "error": lg.error, "data": lg.data},
        }


# ----------------------------
# Response models (stable)
# ----------------------------

class VersionResponse(BaseModel):
    ts_utc: str
    repo_dir: str
    ssot_runtime_dir: str
    git: Dict[str, Any]


class StatusResponse(BaseModel):
    ts_utc: str
    service: str = Field(default="CHAD Operator Surface V2")
    version: Dict[str, Any]
    timers: Dict[str, Any]
    failed_units: Dict[str, Any]
    runtime_files: Dict[str, Any]


# ----------------------------
# Router factory
# ----------------------------

def build_operator_router(cfg: Optional[OperatorConfig] = None) -> APIRouter:
    cfg = cfg or default_config()
    svc = OperatorService(cfg)
    router = APIRouter(tags=["operator"])

    @router.get("/op/version", response_model=VersionResponse)
    async def op_version() -> VersionResponse:
        git = await svc.git_fingerprint()
        return VersionResponse(ts_utc=utc_now_iso(), repo_dir=str(cfg.repo_dir), ssot_runtime_dir=str(cfg.ssot_runtime_dir), git=git)

    @router.get("/op/status", response_model=StatusResponse)
    async def op_status() -> StatusResponse:
        git = await svc.git_fingerprint()
        timers: Dict[str, Any] = {}
        for t in cfg.timers:
            timers[t] = {"enabled": await svc.systemd_is_enabled(t), "active": await svc.systemd_is_active(t)}
        failed = await svc.failed_units()

        files = {k: file_meta(p) for k, p in svc.runtime_paths().items() if k in ("feed_state","positions_snapshot","reconciliation_state","dynamic_caps","operator_intent","portfolio_snapshot","decision_trace_heartbeat","scr_state","tier_state")}

        return StatusResponse(
            ts_utc=utc_now_iso(),
            version=git,
            timers=timers,
            failed_units=failed,
            runtime_files=files,
        )

    @router.get("/op/why_blocked")
    def op_why_blocked() -> Dict[str, Any]:
        return svc.why_blocked()

    @router.get("/op/risk_explain")
    def op_risk_explain() -> Dict[str, Any]:
        return svc.risk_explain()

    @router.get("/op/perf_snapshot")
    def op_perf_snapshot() -> Dict[str, Any]:
        return svc.perf_snapshot()

    @router.get("/op/what_if_caps")
    def op_what_if_caps(equity: float, daily_risk_fraction: float = 0.05) -> Dict[str, Any]:
        return svc.what_if_caps(equity=equity, daily_risk_fraction=daily_risk_fraction)

    @router.get("/op/brief")
    def op_brief() -> Dict[str, Any]:
        return svc.brief()
    @router.get("/op/readiness")
    def op_readiness() -> Dict[str, Any]:
        """
        Operator Readiness Gate (Phase 10 Part 6)

        Read-only, fail-closed checklist that answers:
          - Are we safe to proceed to next phases?
          - What EXACT blockers remain?

        This endpoint never enables live trading. It only reports readiness.
        """
        paths = svc.runtime_paths()

        # Runtime artifacts
        feed = read_json_dict(paths["feed_state"])
        rec = read_json_dict(paths["reconciliation_state"])
        scr = read_json_dict(paths["scr_state"])
        tier = read_json_dict(paths["tier_state"])
        caps = read_json_dict(paths["dynamic_caps"])
        intent = read_json_dict(paths["operator_intent"])

        # systemd health
        failed = anyio.run(svc.failed_units)  # cached + bounded
        timers: Dict[str, Any] = {}
        for t in cfg.timers:
            timers[t] = {
                "enabled": anyio.run(svc.systemd_is_enabled, t),
                "active": anyio.run(svc.systemd_is_active, t),
            }

        blockers: List[str] = []
        warnings: List[str] = []

        # 1) Failed units must be empty
        if not bool(failed.get("ok", False)):
            blockers.append("systemd_failed_units_check_error")
        else:
            fu = failed.get("failed_units") or []
            if fu:
                blockers.append(f"failed_units_present:{len(fu)}")

        # 2) Timers must be enabled+active (SCR/Tier are core operator correctness)
        for t, st in timers.items():
            en = (st.get("enabled") or {}).get("state")
            ac = (st.get("active") or {}).get("state")
            if en != "enabled":
                blockers.append(f"timer_not_enabled:{t}")
            if ac != "active":
                blockers.append(f"timer_not_active:{t}")

        # 3) SCR must exist and not be stale logically
        if not scr.ok:
            blockers.append(f"scr_state_missing:{scr.error}")
        else:
            state = str(scr.data.get("state") or "")
            if not state:
                blockers.append("scr_state_missing_state")

        # 4) Tier must exist (SSOT visibility)
        if not tier.ok:
            warnings.append(f"tier_state_missing:{tier.error}")

        # 5) Caps must exist (risk math visibility)
        if not caps.ok:
            warnings.append(f"dynamic_caps_missing:{caps.error}")
        else:
            prc = _safe_float(caps.data.get("portfolio_risk_cap"))
            if prc <= 0:
                warnings.append("portfolio_risk_cap_nonpositive")

        # 6) Feed state must exist and be parseable
        if not feed.ok:
            warnings.append(f"feed_state_missing:{feed.error}")
        else:
            if not feed.data.get("ts_utc"):
                warnings.append("feed_state_missing_ts_utc")

        # 7) Reconciliation should be GREEN (warn if missing; block if explicitly RED)
        if not rec.ok:
            warnings.append(f"reconciliation_state_missing:{rec.error}")
        else:
            status = str(rec.data.get("status") or "").upper()
            if status == "RED":
                blockers.append("reconciliation_red")
            elif status not in ("GREEN", "UNKNOWN", "ERROR"):
                warnings.append(f"reconciliation_unexpected:{status}")

        # 8) Operator intent existence (warn only; live is still gated elsewhere)
        if not intent.ok:
            warnings.append(f"operator_intent_missing:{intent.error}")

        ok = (len(blockers) == 0)

        next_actions: List[str] = []
        if not ok:
            # deterministic ordering
            for b in blockers:
                if b.startswith("failed_units_present"):
                    next_actions.append("Run: systemctl --failed --no-pager")
                if b.startswith("timer_not_"):
                    next_actions.append("Run: systemctl status chad-scr-sync.timer chad-tier-sync.timer --no-pager")
                if b.startswith("scr_state_missing"):
                    next_actions.append("Run: ls -lh /home/ubuntu/CHAD\\ FINALE/runtime/scr_state.json && journalctl -u chad-scr-sync.service -n 40 --no-pager")
                if b == "reconciliation_red":
                    next_actions.append("Run: curl -sS http://127.0.0.1:9618/reconciliation-state | python3 -m json.tool")
            # dedupe
            next_actions = list(dict.fromkeys(next_actions))

        return {
            "ts_utc": utc_now_iso(),
            "schema_version": "readiness.v1",
            "ok": ok,
            "blockers": blockers,
            "warnings": warnings,
            "timers": timers,
            "failed_units": failed,
            "artifacts": {
                "scr_state": {"ok": scr.ok, "error": scr.error},
                "tier_state": {"ok": tier.ok, "error": tier.error},
                "dynamic_caps": {"ok": caps.ok, "error": caps.error},
                "feed_state": {"ok": feed.ok, "error": feed.error},
                "reconciliation_state": {"ok": rec.ok, "error": rec.error},
                "operator_intent": {"ok": intent.ok, "error": intent.error},
            },
            "next_actions": next_actions,
        }

    return router

#!/usr/bin/env python3
"""
chad/risk/savage_allocator.py

PHASE 11 — "Savage Allocator Mode"
----------------------------------
This module adds a *bounded*, *deterministic*, *fail-closed* capital-allocation overlay
that sits **before** DynamicRiskAllocator writes runtime/dynamic_caps.json.

What it does (in plain terms)
- Start with base strategy weights (from env / defaults).
- Read recent PAPER trade outcomes (append-only NDJSON trade_history files).
- Compute per-strategy performance stats (wins/losses, sharpe-like, drawdown proxy).
- Read macro regime label (runtime/macro_state.json: risk_on / risk_off / neutral).
- Produce per-strategy multipliers (e.g., 0.25..1.35) and apply them to weights.
- Normalize to keep the total portfolio risk cap unchanged (it reallocates, not inflates).

Hard safety rules
- Never touches broker APIs.
- Never increases daily risk fraction or total risk cap; only redistributes within it.
- Fail-closed: if inputs are missing/stale/insufficient, return neutral multipliers (1.0)
  OR tighten (downscale) depending on policy, but never "invent confidence".
- Deterministic: same inputs -> same outputs.

Wiring expectation (next step)
- Orchestrator._build_allocator() will call `apply_savage_overlay(...)` and
  then feed the adjusted weights into StrategyAllocation.

Runtime artifacts (optional but supported)
- This module can produce a JSON-serializable report dict. The orchestrator
  can choose to persist it under runtime/savage_alloc_state.json (TTL’d).

No third-party dependencies.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


# -----------------------------
# Defaults / Env config
# -----------------------------

DEFAULT_LOOKBACK_DAYS: int = 30
DEFAULT_MAX_TRADES: int = 5000
DEFAULT_MIN_TRADES_PER_STRAT: int = 50

# Multipliers are ALWAYS bounded. These bounds prevent "runaway".
DEFAULT_MIN_FACTOR: float = 0.20
DEFAULT_MAX_FACTOR: float = 1.35

# How strongly performance impacts allocation (0 -> neutral).
DEFAULT_ALPHA_PERF: float = 0.60

# Regime tilt is an additional multiplier (bounded separately).
DEFAULT_REGIME_TILT_ON: float = 1.05
DEFAULT_REGIME_TILT_OFF: float = 0.95

# If true, missing/insufficient perf data returns neutral (1.0) per strategy.
# If false, missing/insufficient data tightens (downscales toward MIN_FACTOR).
DEFAULT_FAIL_CLOSED_NEUTRAL: bool = True

# File naming + locations (SSOT-aligned)
TRADE_FILE_GLOB = "trade_history_*.ndjson"
DEFAULT_TTL_SECONDS = 300  # matches dynamic_caps cadence


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = int(str(raw).strip())
        return v if v >= 0 else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(str(raw).strip())
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# -----------------------------
# Domain: stats + decisions
# -----------------------------

@dataclass(frozen=True)
class SavageConfig:
    enabled: bool
    lookback_days: int
    max_trades: int
    min_trades_per_strategy: int
    min_factor: float
    max_factor: float
    alpha_perf: float
    regime_tilt_on: float
    regime_tilt_off: float
    fail_closed_neutral: bool
    ttl_seconds: int

    @classmethod
    def from_env(cls) -> "SavageConfig":
        enabled = _env_bool("CHAD_SAVAGE_ALLOCATOR_ENABLED", True)
        lookback_days = _env_int("CHAD_SAVAGE_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)
        max_trades = _env_int("CHAD_SAVAGE_MAX_TRADES", DEFAULT_MAX_TRADES)
        min_trades = _env_int("CHAD_SAVAGE_MIN_TRADES_PER_STRAT", DEFAULT_MIN_TRADES_PER_STRAT)

        min_factor = _env_float("CHAD_SAVAGE_MIN_FACTOR", DEFAULT_MIN_FACTOR)
        max_factor = _env_float("CHAD_SAVAGE_MAX_FACTOR", DEFAULT_MAX_FACTOR)
        alpha_perf = _env_float("CHAD_SAVAGE_ALPHA_PERF", DEFAULT_ALPHA_PERF)

        regime_on = _env_float("CHAD_SAVAGE_REGIME_TILT_ON", DEFAULT_REGIME_TILT_ON)
        regime_off = _env_float("CHAD_SAVAGE_REGIME_TILT_OFF", DEFAULT_REGIME_TILT_OFF)

        fail_closed_neutral = _env_bool("CHAD_SAVAGE_FAIL_CLOSED_NEUTRAL", DEFAULT_FAIL_CLOSED_NEUTRAL)
        ttl_seconds = _env_int("CHAD_SAVAGE_TTL_SECONDS", DEFAULT_TTL_SECONDS)

        # Defensive clamps
        lookback_days = max(0, lookback_days)
        max_trades = max(0, max_trades)
        min_trades = max(0, min_trades)

        if not math.isfinite(min_factor) or min_factor <= 0.0:
            min_factor = DEFAULT_MIN_FACTOR
        if not math.isfinite(max_factor) or max_factor <= 0.0:
            max_factor = DEFAULT_MAX_FACTOR
        if min_factor > max_factor:
            min_factor, max_factor = max_factor, min_factor

        if not math.isfinite(alpha_perf) or alpha_perf < 0.0:
            alpha_perf = 0.0
        if alpha_perf > 1.0:
            alpha_perf = 1.0

        # Regime tilts must be finite and positive
        if not (math.isfinite(regime_on) and regime_on > 0.0):
            regime_on = DEFAULT_REGIME_TILT_ON
        if not (math.isfinite(regime_off) and regime_off > 0.0):
            regime_off = DEFAULT_REGIME_TILT_OFF

        ttl_seconds = max(30, ttl_seconds)

        return cls(
            enabled=enabled,
            lookback_days=lookback_days,
            max_trades=max_trades,
            min_trades_per_strategy=min_trades,
            min_factor=min_factor,
            max_factor=max_factor,
            alpha_perf=alpha_perf,
            regime_tilt_on=regime_on,
            regime_tilt_off=regime_off,
            fail_closed_neutral=fail_closed_neutral,
            ttl_seconds=ttl_seconds,
        )


@dataclass(frozen=True)
class StrategyPerf:
    trades: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    sharpe_like: float
    max_drawdown: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trades": int(self.trades),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "total_pnl": float(self.total_pnl),
            "win_rate": float(self.win_rate),
            "sharpe_like": float(self.sharpe_like),
            "max_drawdown": float(self.max_drawdown),
        }


# -----------------------------
# Data parsing (paper trades)
# -----------------------------

def _safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    s = line.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _finite_or_zero(x: float) -> float:
    return x if math.isfinite(x) else 0.0


def _compute_sharpe_like(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    mu = sum(pnls) / float(len(pnls))
    var = sum((x - mu) ** 2 for x in pnls) / float(len(pnls))
    sd = math.sqrt(var) if var > 0.0 else 0.0
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    return float(mu / sd)


def _compute_max_drawdown(pnls: List[float]) -> float:
    # Peak-to-trough drawdown on cumulative PnL (negative number).
    if not pnls:
        return 0.0
    peak = 0.0
    cur = 0.0
    max_dd = 0.0
    for p in pnls:
        cur += float(p)
        if cur > peak:
            peak = cur
        dd = cur - peak
        if dd < max_dd:
            max_dd = dd
    return float(max_dd)


def _iter_trade_files(trades_dir: Path) -> List[Path]:
    if not trades_dir.exists():
        return []
    return sorted(trades_dir.glob(TRADE_FILE_GLOB), reverse=True)


def _extract_payload(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize record -> payload.

    We accept either:
    - {"payload": {...}} envelope style
    - {"strategy": "...", ...} flat style
    """
    payload = rec.get("payload")
    if isinstance(payload, dict):
        return payload
    # Flat style fallback
    return rec if isinstance(rec, dict) else None


def _is_live_trade(payload: Dict[str, Any]) -> bool:
    # Many records use is_live / live flags. Treat any truthy as live.
    v = payload.get("is_live", payload.get("live", False))
    return bool(v is True)


def _extract_strategy(payload: Dict[str, Any]) -> str:
    s = payload.get("strategy", "")
    if s is None:
        return "unknown"
    return str(s).strip().lower() or "unknown"


def _extract_pnl(payload: Dict[str, Any]) -> Optional[float]:
    """
    We support multiple possible keys; keep deterministic priority.
    """
    for k in ("pnl", "pnl_usd", "net_pnl", "profit", "profit_usd"):
        if k in payload:
            try:
                v = float(payload.get(k))
                return v if math.isfinite(v) else None
            except Exception:
                return None
    return None


def load_recent_paper_trades(
    *,
    repo_root: Path,
    lookback_days: int,
    max_trades: int,
) -> List[Dict[str, Any]]:
    """
    Load paper trade payloads from data/trades trade_history_*.ndjson files.

    Deterministic behavior:
    - Files are processed newest-first.
    - Within a file, lines are processed in order.
    - Stop once max_trades is reached.

    We intentionally do NOT attempt to parse dates inside each record here,
    because current trade_history schema may not expose a uniform timestamp field.
    lookback_days is therefore a best-effort limiter via file naming.
    """
    trades_dir = repo_root / "data" / "trades"
    files = _iter_trade_files(trades_dir)
    if not files:
        return []

    out: List[Dict[str, Any]] = []

    # Best-effort filter via filename date if present
    cutoff = datetime.now(timezone.utc).date()

    def within_window(fname: str) -> bool:
        # trade_history_YYYYMMDD.ndjson
        try:
            ymd = fname.split("_", 2)[2].split(".", 1)[0]
            dt = datetime.strptime(ymd, "%Y%m%d").date()
            delta = (cutoff - dt).days
            return 0 <= delta <= max(0, lookback_days)
        except Exception:
            # If filename doesn't parse, keep it (fail-safe).
            return True

    for f in files:
        if lookback_days > 0 and not within_window(f.name):
            continue
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            rec = _safe_json_loads(line)
            if not rec:
                continue
            payload = _extract_payload(rec)
            if not isinstance(payload, dict):
                continue
            if _is_live_trade(payload):
                continue
            out.append(payload)
            if max_trades > 0 and len(out) >= max_trades:
                return out

    return out


def compute_perf_by_strategy(trades: Iterable[Dict[str, Any]]) -> Dict[str, StrategyPerf]:
    pnls_by: Dict[str, List[float]] = {}
    wins_by: Dict[str, int] = {}
    losses_by: Dict[str, int] = {}
    total_by: Dict[str, float] = {}
    trades_by: Dict[str, int] = {}

    for t in trades:
        if not isinstance(t, dict):
            continue
        strat = _extract_strategy(t)
        pnl = _extract_pnl(t)
        if pnl is None:
            continue
        pnl_f = float(pnl)
        if not math.isfinite(pnl_f):
            continue

        pnls_by.setdefault(strat, []).append(pnl_f)
        trades_by[strat] = trades_by.get(strat, 0) + 1
        total_by[strat] = total_by.get(strat, 0.0) + pnl_f

        if pnl_f > 0:
            wins_by[strat] = wins_by.get(strat, 0) + 1
        elif pnl_f < 0:
            losses_by[strat] = losses_by.get(strat, 0) + 1
        else:
            # pnl == 0 counts as a trade but not win/loss
            pass

    out: Dict[str, StrategyPerf] = {}
    for strat, n in trades_by.items():
        pnls = pnls_by.get(strat, [])
        wins = wins_by.get(strat, 0)
        losses = losses_by.get(strat, 0)
        total = total_by.get(strat, 0.0)
        wr = float(wins) / float(n) if n > 0 else 0.0
        sharpe_like = _compute_sharpe_like(pnls)
        max_dd = _compute_max_drawdown(pnls)
        out[strat] = StrategyPerf(
            trades=int(n),
            wins=int(wins),
            losses=int(losses),
            total_pnl=float(total),
            win_rate=float(wr),
            sharpe_like=float(sharpe_like),
            max_drawdown=float(max_dd),
        )
    return out


# -----------------------------
# Regime reader (macro_state.json)
# -----------------------------

def read_macro_risk_label(*, repo_root: Path) -> str:
    """
    Best-effort read of runtime/macro_state.json risk_label.
    Returns: "risk_on" | "risk_off" | "neutral" | "unknown"
    """
    path = repo_root / "runtime" / "macro_state.json"
    try:
        if not path.is_file():
            return "unknown"
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return "unknown"
        lab = str(obj.get("risk_label", "")).strip().lower()
        if lab in ("risk_on", "risk_off", "neutral"):
            return lab
        return "unknown"
    except Exception:
        return "unknown"


# -----------------------------
# Scoring -> multipliers
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sigmoid(x: float) -> float:
    # Stable sigmoid mapping to (0,1)
    # Clamp x to prevent overflow
    x = _clamp(x, -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-x))


def _score_from_perf(perf: StrategyPerf) -> float:
    """
    Build a conservative score in [0,1] from win_rate + sharpe_like + drawdown.

    - win_rate dominates early (robust, easy to interpret).
    - sharpe_like adjusts for consistency.
    - drawdown penalizes tail ugliness.

    This is deterministic and bounded; it is not ML.
    """
    # win_rate in [0,1]
    wr = _clamp(perf.win_rate, 0.0, 1.0)

    # sharpe_like mapped via sigmoid around 0
    # -2 -> ~0.12, 0 -> 0.5, +2 -> ~0.88
    sh = _sigmoid(perf.sharpe_like)

    # drawdown is negative or zero; penalize large negative
    # -10 -> heavy penalty; 0 -> no penalty
    dd = float(perf.max_drawdown)
    dd_pen = _sigmoid(dd / 2.0)  # dd=-6 -> sigmoid(-3)~0.047 ; dd=0 -> 0.5

    # Weighted blend; keep simple and stable
    # Note: dd_pen is 0..1 where larger is better (less drawdown).
    score = (0.55 * wr) + (0.35 * sh) + (0.10 * dd_pen)
    return _clamp(score, 0.0, 1.0)


def _mult_from_score(score: float, cfg: SavageConfig) -> float:
    """
    Convert score [0,1] to multiplier [min_factor,max_factor],
    with alpha_perf controlling how much we depart from neutral.
    """
    score = _clamp(score, 0.0, 1.0)

    # Base mapping: 0 -> min_factor, 0.5 -> 1.0-ish, 1 -> max_factor
    # We shape it so that score=0.5 yields 1.0 exactly.
    if score >= 0.5:
        # scale up toward max
        up = (score - 0.5) / 0.5  # 0..1
        raw = 1.0 + up * (cfg.max_factor - 1.0)
    else:
        # scale down toward min
        down = (0.5 - score) / 0.5  # 0..1
        raw = 1.0 - down * (1.0 - cfg.min_factor)

    # Blend with neutral based on alpha_perf
    blended = (1.0 - cfg.alpha_perf) * 1.0 + cfg.alpha_perf * raw
    return _clamp(float(blended), cfg.min_factor, cfg.max_factor)


def _apply_regime_tilt(strategy: str, mult: float, risk_label: str, cfg: SavageConfig) -> float:
    """
    Regime tilt is deliberately mild and bounded, and never overrides perf clamps.

    Default behavior:
    - risk_off: slightly tilt down "alpha/gamma/delta/crypto/forex", tilt up "omega/beta"
    - risk_on: slightly tilt up "alpha/gamma/delta/crypto/forex", tilt down "omega"
    - neutral/unknown: no change
    """
    s = strategy.strip().lower()
    m = float(mult)

    if risk_label == "risk_off":
        if s in ("omega",):
            m *= cfg.regime_tilt_on
        elif s in ("beta",):
            m *= 1.02  # mild preference for base sleeve
        else:
            m *= cfg.regime_tilt_off
    elif risk_label == "risk_on":
        if s in ("omega",):
            m *= cfg.regime_tilt_off
        else:
            m *= cfg.regime_tilt_on

    return _clamp(m, cfg.min_factor, cfg.max_factor)


# -----------------------------
# Public API
# -----------------------------

def compute_savage_multipliers(
    *,
    repo_root: Path,
    base_weights: Mapping[str, float],
    cfg: SavageConfig,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Compute multipliers per strategy given base weights and recent paper performance.

    Returns:
      (multipliers, report)
    """
    ts = _utc_now_iso()
    risk_label = read_macro_risk_label(repo_root=repo_root)

    report: Dict[str, Any] = {
        "schema_version": "savage_alloc_state.v1",
        "ts_utc": ts,
        "ttl_seconds": int(cfg.ttl_seconds),
        "enabled": bool(cfg.enabled),
        "risk_label": risk_label,
        "config": {
            "lookback_days": int(cfg.lookback_days),
            "max_trades": int(cfg.max_trades),
            "min_trades_per_strategy": int(cfg.min_trades_per_strategy),
            "min_factor": float(cfg.min_factor),
            "max_factor": float(cfg.max_factor),
            "alpha_perf": float(cfg.alpha_perf),
            "regime_tilt_on": float(cfg.regime_tilt_on),
            "regime_tilt_off": float(cfg.regime_tilt_off),
            "fail_closed_neutral": bool(cfg.fail_closed_neutral),
        },
        "per_strategy": {},
        "notes": [],
    }

    # If disabled: neutral multipliers
    if not cfg.enabled:
        multipliers = {k: 1.0 for k in base_weights.keys()}
        report["notes"].append("disabled: returning neutral multipliers=1.0")
        return multipliers, report

    trades = load_recent_paper_trades(
        repo_root=repo_root,
        lookback_days=cfg.lookback_days,
        max_trades=cfg.max_trades,
    )
    perf_by = compute_perf_by_strategy(trades)

    if not trades:
        report["notes"].append("no_paper_trades_found")
    else:
        report["notes"].append(f"paper_trades_loaded={len(trades)}")

    multipliers: Dict[str, float] = {}

    # Deterministic strategy order: base weight keys sorted
    for strat in sorted(base_weights.keys()):
        p = perf_by.get(strat)

        if p is None or p.trades < cfg.min_trades_per_strategy:
            # Fail-closed: either neutral or tighten
            if cfg.fail_closed_neutral:
                mult = 1.0
                reason = "insufficient_data_neutral"
            else:
                mult = cfg.min_factor
                reason = "insufficient_data_tighten"
            multipliers[strat] = float(mult)
            report["per_strategy"][strat] = {
                "multiplier": float(mult),
                "reason": reason,
                "perf": p.to_dict() if p is not None else None,
                "score": None,
            }
            continue

        score = _score_from_perf(p)
        mult = _mult_from_score(score, cfg=cfg)
        mult = _apply_regime_tilt(strat, mult, risk_label=risk_label, cfg=cfg)

        multipliers[strat] = float(mult)
        report["per_strategy"][strat] = {
            "multiplier": float(mult),
            "reason": "perf_and_regime",
            "perf": p.to_dict(),
            "score": float(score),
        }

    # Note: strategies present in perf but not base weights are ignored by design.
    return multipliers, report


def apply_savage_overlay(
    *,
    repo_root: Path,
    base_weights: Mapping[str, float],
    cfg: SavageConfig,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Apply computed multipliers to base weights and return adjusted raw weights.

    Important:
    - We do NOT normalize here; DynamicRiskAllocator will normalize.
    - We preserve zero-weight strategies as zero (never resurrect a disabled brain).

    Returns:
      (adjusted_raw_weights, report)
    """
    multipliers, report = compute_savage_multipliers(
        repo_root=repo_root,
        base_weights=base_weights,
        cfg=cfg,
    )

    adjusted: Dict[str, float] = {}
    for strat, w in base_weights.items():
        w_f = float(w)
        if not math.isfinite(w_f) or w_f <= 0.0:
            adjusted[strat] = 0.0
            continue
        m = float(multipliers.get(strat, 1.0))
        if not math.isfinite(m) or m <= 0.0:
            m = 1.0
        adjusted[strat] = float(w_f * m)

    report["base_weights"] = {k: float(base_weights[k]) for k in sorted(base_weights.keys())}
    report["multipliers"] = {k: float(multipliers.get(k, 1.0)) for k in sorted(base_weights.keys())}
    report["adjusted_raw_weights"] = {k: float(adjusted[k]) for k in sorted(adjusted.keys())}

    # If everything got zeroed (shouldn't happen), fail-closed to base.
    if sum(adjusted.values()) <= 0.0:
        report["notes"].append("all_adjusted_weights_zero: reverting_to_base_weights")
        adjusted = {k: float(max(0.0, float(v))) for k, v in base_weights.items()}
        report["adjusted_raw_weights"] = {k: float(adjusted[k]) for k in sorted(adjusted.keys())}

    return adjusted, report


def write_state_report_atomic(
    *,
    repo_root: Path,
    report: Mapping[str, Any],
    filename: str = "savage_alloc_state.json",
) -> Path:
    """
    Optional helper for orchestrator to persist the overlay report under runtime/.

    Atomic write:
      tmp -> replace
    """
    runtime_dir = repo_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out_path = runtime_dir / filename
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")

    payload = dict(report)
    # Ensure ts/ttl exist
    payload.setdefault("ts_utc", _utc_now_iso())
    payload.setdefault("ttl_seconds", int(DEFAULT_TTL_SECONDS))

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(out_path)
    return out_path

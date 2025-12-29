#!/usr/bin/env python3
"""
CHAD — Shadow Confidence Router (SCR)

Purpose
-------
Given aggregate trade stats, decide CHAD's confidence band:
- WARMUP
- CAUTIOUS
- CONFIDENT
- PAUSED

This is a safety governor used by LiveGate and status surfaces.

Critical behavior (production)
------------------------------
- The confidence decision must be deterministic.
- It must never crash on missing keys or bad values.
- It must keep live trading blocked unless explicitly allowed by config.

Key upgrade for your current system state
-----------------------------------------
Your trade_stats_engine now produces:
- total_trades (raw)
- effective_trades (performance sample)
- excluded_* counters

Warmup/manual/untrusted trades should NOT poison performance gating.
Therefore SCR must use effective_trades for the warmup threshold and
for performance band eligibility.

Result:
- If effective_trades < warmup_min_trades => WARMUP (paper-only continues)
- Avoid PAUSED solely because there is "no effective sample" yet.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# Authoritative runtime config path on your host
SCR_CONFIG_PATH = Path("/home/ubuntu/CHAD FINALE/runtime/scr_config.json")


# -----------------------------
# Data models
# -----------------------------

@dataclass(frozen=True)
class SCRConfig:
    # Warmup gating
    warmup_min_trades: int = 50
    warmup_allow_live: bool = False

    # CONFIDENT thresholds
    confident_min_win_rate: float = 0.55
    confident_min_sharpe: float = 0.70
    confident_max_drawdown: float = -2000.0  # Max DD must be >= this (less negative is better)

    # CAUTIOUS thresholds
    cautious_min_win_rate: float = 0.45
    cautious_min_sharpe: float = 0.30
    cautious_max_drawdown: float = -5000.0

    # Sizing factors per band
    warmup_sizing_factor: float = 0.10
    confident_sizing_factor: float = 1.00
    cautious_sizing_factor: float = 0.25
    paused_sizing_factor: float = 0.00


@dataclass(frozen=True)
class ShadowState:
    state: str
    sizing_factor: float
    paper_only: bool
    reasons: List[str]
    stats: Dict[str, Any]


# -----------------------------
# Config loading
# -----------------------------

def _safe_int(x: Any, default: int) -> int:
    try:
        v = int(x)
        return v
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def load_scr_config(path: Path = SCR_CONFIG_PATH) -> SCRConfig:
    """
    Load SCR config from disk if present; otherwise return defaults.

    The runtime file is treated as operator-authoritative.
    """
    base = SCRConfig()
    try:
        if not path.exists():
            return base
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return base

        return SCRConfig(
            warmup_min_trades=_safe_int(raw.get("warmup_min_trades", base.warmup_min_trades), base.warmup_min_trades),
            warmup_allow_live=bool(raw.get("warmup_allow_live", base.warmup_allow_live)),

            confident_min_win_rate=_safe_float(raw.get("confident_min_win_rate", base.confident_min_win_rate), base.confident_min_win_rate),
            confident_min_sharpe=_safe_float(raw.get("confident_min_sharpe", base.confident_min_sharpe), base.confident_min_sharpe),
            confident_max_drawdown=_safe_float(raw.get("confident_max_drawdown", base.confident_max_drawdown), base.confident_max_drawdown),

            cautious_min_win_rate=_safe_float(raw.get("cautious_min_win_rate", base.cautious_min_win_rate), base.cautious_min_win_rate),
            cautious_min_sharpe=_safe_float(raw.get("cautious_min_sharpe", base.cautious_min_sharpe), base.cautious_min_sharpe),
            cautious_max_drawdown=_safe_float(raw.get("cautious_max_drawdown", base.cautious_max_drawdown), base.cautious_max_drawdown),

            warmup_sizing_factor=_safe_float(raw.get("warmup_sizing_factor", base.warmup_sizing_factor), base.warmup_sizing_factor),
            confident_sizing_factor=_safe_float(raw.get("confident_sizing_factor", base.confident_sizing_factor), base.confident_sizing_factor),
            cautious_sizing_factor=_safe_float(raw.get("cautious_sizing_factor", base.cautious_sizing_factor), base.cautious_sizing_factor),
            paused_sizing_factor=_safe_float(raw.get("paused_sizing_factor", base.paused_sizing_factor), base.paused_sizing_factor),
        )
    except Exception:
        # Never crash SCR on config parse; fall back to defaults.
        return base


# -----------------------------
# Metrics extraction
# -----------------------------

def _extract_metric(stats: Dict[str, Any], key: str, default: float) -> float:
    return _safe_float(stats.get(key, default), default)


def _extract_int(stats: Dict[str, Any], key: str, default: int) -> int:
    return _safe_int(stats.get(key, default), default)


# -----------------------------
# Core evaluation
# -----------------------------

def evaluate_confidence(stats: Dict[str, Any], config: Optional[SCRConfig] = None) -> ShadowState:
    """
    Evaluate confidence band given aggregate stats.

    IMPORTANT:
    - Uses effective_trades (performance sample) as the gating counter.
    - total_trades remains informational and is still surfaced in stats.
    """
    reasons: List[str] = []

    if config is None:
        config = load_scr_config()

    total_trades = _extract_int(stats, "total_trades", 0)
    effective_trades = _extract_int(stats, "effective_trades", total_trades)

    win_rate = _extract_metric(stats, "win_rate", 0.0)
    sharpe_like = _extract_metric(stats, "sharpe_like", 0.0)
    max_drawdown = _extract_metric(stats, "max_drawdown", 0.0)
    total_pnl = _extract_metric(stats, "total_pnl", 0.0)

    # ---- WARMUP band (effective sample too small) ---------------------------
    if effective_trades < config.warmup_min_trades:
        if effective_trades == 0:
            reasons.append(
                "Warmup: 0 effective trades (all trades excluded/untrusted). Paper-only continues until real strategy sample exists."
            )
        reasons.append(
            f"Warmup: only {effective_trades} effective trades (< {config.warmup_min_trades} required)."
        )
        reasons.append(
            f"Current win_rate={win_rate:.3f}, sharpe_like={sharpe_like:.3f}, "
            f"max_drawdown={max_drawdown:.2f}, total_pnl={total_pnl:.2f} "
            f"(total_trades={total_trades}, effective_trades={effective_trades})"
        )

        sizing = config.warmup_sizing_factor
        paper_only = not config.warmup_allow_live

        return ShadowState(
            state="WARMUP",
            sizing_factor=sizing,
            paper_only=paper_only,
            reasons=reasons,
            stats=stats,
        )

    # ---- CONFIDENT band -----------------------------------------------------
    confident_conditions: List[str] = []

    if win_rate >= config.confident_min_win_rate:
        confident_conditions.append("win_rate OK")
    else:
        reasons.append(
            f"CONFIDENT blocked: win_rate={win_rate:.3f} < {config.confident_min_win_rate:.3f}"
        )

    if sharpe_like >= config.confident_min_sharpe:
        confident_conditions.append("sharpe OK")
    else:
        reasons.append(
            f"CONFIDENT blocked: sharpe_like={sharpe_like:.3f} < {config.confident_min_sharpe:.3f}"
        )

    if max_drawdown >= config.confident_max_drawdown:
        confident_conditions.append("drawdown OK")
    else:
        reasons.append(
            f"CONFIDENT blocked: max_drawdown={max_drawdown:.2f} < {config.confident_max_drawdown:.2f}"
        )

    if len(confident_conditions) == 3:
        reasons.append(
            "CONFIDENT: win_rate, sharpe, and drawdown all within confident band."
        )
        return ShadowState(
            state="CONFIDENT",
            sizing_factor=config.confident_sizing_factor,
            paper_only=False,
            reasons=reasons,
            stats=stats,
        )

    # ---- CAUTIOUS band ------------------------------------------------------
    cautious_ok = True

    if win_rate < config.cautious_min_win_rate:
        cautious_ok = False
        reasons.append(
            f"CAUTIOUS blocked by win_rate={win_rate:.3f} < {config.cautious_min_win_rate:.3f}"
        )

    if sharpe_like < config.cautious_min_sharpe:
        cautious_ok = False
        reasons.append(
            f"CAUTIOUS blocked by sharpe_like={sharpe_like:.3f} < {config.cautious_min_sharpe:.3f}"
        )

    if max_drawdown < config.cautious_max_drawdown:
        cautious_ok = False
        reasons.append(
            f"CAUTIOUS blocked by max_drawdown={max_drawdown:.2f} < {config.cautious_max_drawdown:.2f}"
        )

    if cautious_ok:
        reasons.append(
            "CAUTIOUS: stats not strong enough for CONFIDENT, but above cautious thresholds."
        )
        return ShadowState(
            state="CAUTIOUS",
            sizing_factor=config.cautious_sizing_factor,
            paper_only=False,
            reasons=reasons,
            stats=stats,
        )

    # ---- PAUSED band --------------------------------------------------------
    reasons.append(
        "PAUSED: performance below cautious thresholds – live trading should be disabled."
    )
    reasons.append(
        f"Summary: total_trades={total_trades}, effective_trades={effective_trades}, "
        f"win_rate={win_rate:.3f}, sharpe_like={sharpe_like:.3f}, "
        f"max_drawdown={max_drawdown:.2f}, total_pnl={total_pnl:.2f}"
    )

    return ShadowState(
        state="PAUSED",
        sizing_factor=config.paused_sizing_factor,
        paper_only=True,
        reasons=reasons,
        stats=stats,
    )
# --- Phase 4 audit: always print SCR decision when run as a module ---
if __name__ == "__main__":
    import json

    from chad.analytics.trade_stats_engine import load_and_compute

    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    state = evaluate_confidence(stats)

    payload = {
        "state": getattr(state, "state", None),
        "paper_only": bool(getattr(state, "paper_only", False)),
        "sizing_factor": float(getattr(state, "sizing_factor", 0.0) or 0.0),
        "reasons": list(getattr(state, "reasons", []) or []),
        "stats": {
            "total_trades": int(stats.get("total_trades", 0) or 0),
            "paper_trades": int(stats.get("paper_trades", 0) or 0),
            "live_trades": int(stats.get("live_trades", 0) or 0),
            "effective_trades": int(stats.get("effective_trades", 0) or 0),
            "excluded_manual": int(stats.get("excluded_manual", 0) or 0),
            "excluded_untrusted": int(stats.get("excluded_untrusted", 0) or 0),
            "excluded_nonfinite": int(stats.get("excluded_nonfinite", 0) or 0),
            "win_rate": float(stats.get("win_rate", 0.0) or 0.0),
            "total_pnl": float(stats.get("total_pnl", 0.0) or 0.0),
            "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
            "sharpe_like": float(stats.get("sharpe_like", 0.0) or 0.0),
        },
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

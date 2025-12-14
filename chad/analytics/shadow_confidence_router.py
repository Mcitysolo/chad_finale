"""
Shadow Confidence Router (SCR) – Phase 5

This module turns recent trade performance into a **confidence state** that
other CHAD components can use to gate live trading and position sizing.

Inputs:
    * Aggregated stats dict from `trade_stats_engine.load_and_compute(...)`

Outputs:
    * ShadowState object describing:
        - state:       "WARMUP" | "CONFIDENT" | "CAUTIOUS" | "PAUSED"
        - sizing_factor: 0.0–1.0 (how much of normal size to allow)
        - paper_only:  True/False (whether live trading should be disabled)
        - reasons:     list of human-readable strings explaining the decision
        - stats:       the original stats dict for downstream consumers

Design goals:
    * Deterministic and explainable.
    * Conservative by default (err on the side of PAUSED / paper-only).
    * Simple thresholds, easy to tune in later phases.

Typical usage:
    from chad.analytics.trade_stats_engine import load_and_compute
    from chad.analytics.shadow_confidence_router import evaluate_confidence

    stats = load_and_compute(max_trades=200, days_back=30)
    shadow_state = evaluate_confidence(stats)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


ConfidenceState = Literal["WARMUP", "CONFIDENT", "CAUTIOUS", "PAUSED"]


@dataclass(frozen=True)
class SCRConfig:
    """
    Thresholds and settings for Shadow Confidence Router.

    All thresholds are intentionally conservative and can be tuned later.
    """

    # Minimum trades required before we trust any stats at all.
    warmup_min_trades: int = 50

    # CONFIDENT band thresholds.
    confident_min_win_rate: float = 0.55    # 55%+ win rate
    confident_min_sharpe: float = 0.7       # Sharpe-like threshold
    confident_max_drawdown: float = -2000.0  # Max DD >= -2k (unit: PnL)

    # CAUTIOUS band thresholds (softer than CONFIDENT).
    cautious_min_win_rate: float = 0.45
    cautious_min_sharpe: float = 0.3
    cautious_max_drawdown: float = -5000.0  # Max DD >= -5k

    # Sizing factors for each band.
    warmup_sizing_factor: float = 0.1       # Tiny canary sizing
    confident_sizing_factor: float = 1.0    # Full intended size
    cautious_sizing_factor: float = 0.25    # Reduced sizing
    paused_sizing_factor: float = 0.0       # No live size

    # Whether to allow *any* live trading in WARMUP band.
    warmup_allow_live: bool = False


@dataclass(frozen=True)
class ShadowState:
    """
    Result of SCR evaluation.

    Fields:
        state:
            High-level confidence band:
                - "WARMUP": Not enough data yet; rely on paper/canary.
                - "CONFIDENT": Full live trading allowed.
                - "CAUTIOUS": Reduced size / canary while we watch performance.
                - "PAUSED": No live trading (paper-only).

        sizing_factor:
            Scaling multiplier for normal position sizes (0.0–1.0).

        paper_only:
            If True, CHAD should run in paper mode only; no live orders.

        reasons:
            List of human-readable explanations for the decision.

        stats:
            The original stats dict used for the decision (for transparency
            and logging by the coach, reports, etc.).
    """

    state: ConfidenceState
    sizing_factor: float
    paper_only: bool
    reasons: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


DEFAULT_CONFIG = SCRConfig()


def _extract_metric(stats: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    Safely extract a float metric from stats.
    """
    value = stats.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def evaluate_confidence(
    stats: Dict[str, Any],
    config: SCRConfig = DEFAULT_CONFIG,
) -> ShadowState:
    """
    Evaluate CHAD's current confidence band based on aggregate trade stats.

    Args:
        stats:
            Dict as returned by `trade_stats_engine.load_and_compute(...)`.
        config:
            SCRConfig thresholds.

    Returns:
        ShadowState describing confidence band, sizing, and paper/live gating.
    """
    reasons: List[str] = []

    total_trades = int(stats.get("total_trades", 0))
    win_rate = _extract_metric(stats, "win_rate", 0.0)
    sharpe_like = _extract_metric(stats, "sharpe_like", 0.0)
    max_drawdown = _extract_metric(stats, "max_drawdown", 0.0)
    total_pnl = _extract_metric(stats, "total_pnl", 0.0)

    # ---- WARMUP band: not enough data yet ----------------------------------
    if total_trades < config.warmup_min_trades:
        reasons.append(
            f"Warmup: only {total_trades} trades (< {config.warmup_min_trades} required)."
        )
        reasons.append(
            f"Current win_rate={win_rate:.3f}, sharpe_like={sharpe_like:.3f}, "
            f"max_drawdown={max_drawdown:.2f}, total_pnl={total_pnl:.2f}"
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
    # All conditions must pass for CONFIDENT.
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
    # Softer requirements than CONFIDENT.
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
    # If we don't qualify for CONFIDENT or CAUTIOUS, we pause live trading.
    reasons.append(
        "PAUSED: performance below cautious thresholds – live trading should be disabled."
    )
    reasons.append(
        f"Summary: total_trades={total_trades}, win_rate={win_rate:.3f}, "
        f"sharpe_like={sharpe_like:.3f}, max_drawdown={max_drawdown:.2f}, "
        f"total_pnl={total_pnl:.2f}"
    )

    return ShadowState(
        state="PAUSED",
        sizing_factor=config.paused_sizing_factor,
        paper_only=True,
        reasons=reasons,
        stats=stats,
    )

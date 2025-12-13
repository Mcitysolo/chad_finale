"""
Shadow Formatting Helpers (Phase 6 – Coach / Status Messages)

This module turns ShadowState + stats into **human-readable text** suitable
for:

    * Telegram coach replies
    * HTTP /status endpoints
    * CLI tools

It does NOT compute any stats or state itself – it only formats the output
from:

    - chad.analytics.shadow_confidence_router.evaluate_confidence(...)
    - chad.analytics.trade_stats_engine.load_and_compute(...)
    - chad.analytics.shadow_state_snapshot.write_shadow_snapshot(...)
"""

from __future__ import annotations

from typing import Any, Dict, List

from chad.analytics.shadow_confidence_router import ShadowState


def format_shadow_summary(shadow_state: ShadowState) -> str:
    """
    Produce a compact, single-message summary suitable for Telegram / status.

    Includes:
        - State (WARMUP / CONFIDENT / CAUTIOUS / PAUSED)
        - Sizing factor
        - Paper-only flag
        - Totals (trades, win-rate, total PnL)
        - Top reasons
    """
    stats: Dict[str, Any] = shadow_state.stats or {}

    total_trades = int(stats.get("total_trades", 0))
    win_rate = float(stats.get("win_rate", 0.0))
    total_pnl = float(stats.get("total_pnl", 0.0))
    live_trades = int(stats.get("live_trades", 0))
    paper_trades = int(stats.get("paper_trades", 0))

    lines: List[str] = []

    lines.append("CHAD Shadow Confidence Snapshot")
    lines.append("--------------------------------")
    lines.append(f"State        : {shadow_state.state}")
    lines.append(f"Sizing factor: {shadow_state.sizing_factor:.3f}")
    lines.append(f"Paper only   : {shadow_state.paper_only}")
    lines.append("")
    lines.append("Performance")
    lines.append(f"  Total trades : {total_trades}")
    lines.append(f"  Win rate     : {win_rate:.3f}")
    lines.append(f"  Total PnL    : {total_pnl:.2f}")
    lines.append(f"  Live trades  : {live_trades}")
    lines.append(f"  Paper trades : {paper_trades}")
    lines.append("")
    lines.append("Reasons")
    if not shadow_state.reasons:
        lines.append("  (no reasons recorded)")
    else:
        for idx, reason in enumerate(shadow_state.reasons, start=1):
            lines.append(f"  {idx}. {reason}")

    return "\n".join(lines)


def format_shadow_state_simple(shadow_state: ShadowState) -> str:
    """
    Minimal one-liner summary, useful for logs or short status messages.
    """
    stats: Dict[str, Any] = shadow_state.stats or {}
    total_trades = int(stats.get("total_trades", 0))
    win_rate = float(stats.get("win_rate", 0.0))
    total_pnl = float(stats.get("total_pnl", 0.0))

    return (
        f"ShadowState={shadow_state.state} "
        f"(sizing_factor={shadow_state.sizing_factor:.3f}, "
        f"paper_only={shadow_state.paper_only}, "
        f"trades={total_trades}, win_rate={win_rate:.3f}, "
        f"total_pnl={total_pnl:.2f})"
    )

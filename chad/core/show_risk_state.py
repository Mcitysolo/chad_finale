"""
CHAD Risk State Snapshot – CLI (Phase 7)

This tool prints a unified snapshot of CHAD's current risk state:

    * Global mode (CHAD_MODE):
        - DRY_RUN or LIVE (normalized)
        - live_enabled flag

    * Dynamic caps (if available) from runtime/dynamic_caps.json:
        - total_equity
        - portfolio_risk_cap
        - daily_risk_fraction
        - strategy_caps (alpha, beta, gamma, omega, delta, crypto, forex)

    * Shadow Confidence State (SCR):
        - state (WARMUP / CONFIDENT / CAUTIOUS / PAUSED)
        - sizing_factor
        - paper_only
        - reasons
        - key performance stats (trades, win_rate, total_pnl)

Usage:

    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.show_risk_state

This command is STRICTLY READ-ONLY – it does NOT execute trades,
modify config, or change system mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from chad.analytics.trade_stats_engine import load_and_compute
from chad.analytics.shadow_confidence_router import evaluate_confidence
from chad.analytics.shadow_formatting import format_shadow_state_simple
from chad.core.mode import get_chad_mode, is_live_mode_enabled


DYNAMIC_CAPS_PATH = Path("runtime") / "dynamic_caps.json"


def _load_dynamic_caps(path: Path) -> Dict[str, Any]:
    """
    Load dynamic caps if present, else return a stub dict.
    """
    if not path.exists():
        return {
            "available": False,
            "reason": f"dynamic caps file not found: {str(path)}",
        }

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"error reading dynamic caps: {exc}",
        }

    total_equity = float(data.get("total_equity", 0.0))
    portfolio_risk_cap = float(data.get("portfolio_risk_cap", 0.0))
    daily_risk_fraction = float(data.get("daily_risk_fraction", 0.0))
    strategy_caps = data.get("strategy_caps", {}) or {}

    return {
        "available": True,
        "total_equity": total_equity,
        "portfolio_risk_cap": portfolio_risk_cap,
        "daily_risk_fraction": daily_risk_fraction,
        "strategy_caps": strategy_caps,
        "raw": data,
    }


def main() -> None:
    # --- Dynamic Caps ---
    dyn_caps = _load_dynamic_caps(DYNAMIC_CAPS_PATH)

    # --- CHAD Mode ---
    mode = get_chad_mode()
    live_enabled = is_live_mode_enabled()

    # --- Shadow Confidence State ---
    stats = load_and_compute(
        max_trades=200,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    shadow_state = evaluate_confidence(stats)

    # --- Print unified snapshot ---
    print("=== CHAD Risk State Snapshot ===\n")

    # Mode section
    print("Mode")
    print(f"  CHAD_MODE       : {mode.value}")
    print(f"  live_enabled    : {live_enabled}")
    print()

    # Dynamic caps section
    print("Dynamic Caps")
    if not dyn_caps.get("available", False):
        print(f"  [!] Not available: {dyn_caps.get('reason')}")
    else:
        total_equity = dyn_caps["total_equity"]
        portfolio_risk_cap = dyn_caps["portfolio_risk_cap"]
        daily_risk_fraction = dyn_caps["daily_risk_fraction"]
        strategy_caps = dyn_caps["strategy_caps"]

        print(f"  total_equity        : {total_equity:,.2f}")
        print(f"  daily_risk_fraction : {daily_risk_fraction:.3f}")
        print(f"  portfolio_risk_cap  : {portfolio_risk_cap:,.2f}")
        print("  strategy_caps:")
        if not strategy_caps:
            print("    (none)")
        else:
            for name, cap in strategy_caps.items():
                try:
                    cap_val = float(cap)
                except Exception:
                    cap_val = cap
                print(f"    {name:8s}: {cap_val:,.2f}")

    print("\nShadow Confidence (SCR)")
    print("------------------------")
    print(format_shadow_state_simple(shadow_state))
    print()
    print("Reasons:")
    if not shadow_state.reasons:
        print("  (no reasons recorded)")
    else:
        for idx, reason in enumerate(shadow_state.reasons, start=1):
            print(f"  {idx}. {reason}")

    sys.exit(0)


if __name__ == "__main__":
    main()

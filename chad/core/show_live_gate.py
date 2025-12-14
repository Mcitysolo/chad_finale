"""
CHAD Live Gate Snapshot – CLI (Phase 8)

This tool prints the unified live gating decision for IBKR, based on:

    * ExecutionConfig (CHAD_EXECUTION_MODE, ibkr_enabled, ibkr_dry_run)
    * CHAD_MODE (DRY_RUN / LIVE)
    * Shadow Confidence State (SCR)

It is READ-ONLY and does NOT execute any trades.
"""

from __future__ import annotations

from pprint import pprint

from chad.core.live_gate import evaluate_live_gate


def main() -> None:
    decision = evaluate_live_gate()
    ctx = decision.context

    print("=== CHAD Live Gate Snapshot ===\n")

    print("ExecutionConfig (adapter-level)")
    print("--------------------------------")
    print(f"  exec_mode      : {ctx.exec_mode}")
    print(f"  ibkr_enabled   : {ctx.ibkr_enabled}")
    print(f"  ibkr_dry_run   : {ctx.ibkr_dry_run}")
    print()
    print("CHAD_MODE (global)")
    print("------------------")
    print(f"  chad_mode      : {ctx.chad_mode.value}")
    print(f"  live_enabled   : {ctx.chad_mode == type(ctx.chad_mode).LIVE}")
    print()
    print("Shadow Confidence (SCR)")
    print("------------------------")
    print(f"  state          : {ctx.shadow_state.state}")
    print(f"  sizing_factor  : {ctx.shadow_state.sizing_factor}")
    print(f"  paper_only     : {ctx.shadow_state.paper_only}")
    print()
    print("Final Live Gate Decision")
    print("------------------------")
    print(f"  allow_ibkr_live  : {decision.allow_ibkr_live}")
    print(f"  allow_ibkr_paper : {decision.allow_ibkr_paper}")
    print()
    print("Reasons:")
    if not decision.reasons:
        print("  (no reasons)")
    else:
        for idx, reason in enumerate(decision.reasons, start=1):
            print(f"  {idx}. {reason}")


if __name__ == "__main__":
    main()

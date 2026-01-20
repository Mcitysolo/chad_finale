"""
chad.core.show_live_gate

Operator-facing LiveGate snapshot printer.

Purpose
-------
This module prints a human-readable snapshot of CHAD's LiveGate evaluation.
It is read-only and safe. It never executes trades.

It is used for:
- CLI sanity checks
- runbooks
- operator verification
- CI smoke checks (basic)

This script must NEVER crash in production. If evaluation fails, it prints a
clear error and exits non-zero.

"""

from __future__ import annotations

import sys
from typing import Any

from chad.core.live_gate import evaluate_live_gate


def _p(s: str = "") -> None:
    print(s)


def main() -> int:
    try:
        decision = evaluate_live_gate()
        ctx = decision.context
    except Exception as exc:
        _p("=== CHAD Live Gate Snapshot ===")
        _p("ERROR: Failed to evaluate LiveGate.")
        _p(f"Exception: {type(exc).__name__}: {exc}")
        return 2

    _p("=== CHAD Live Gate Snapshot ===")
    _p("")

    _p("ExecutionConfig (adapter-level)")
    _p("--------------------------------")
    _p(f"  exec_mode      : {ctx.execution.exec_mode}")
    _p(f"  ibkr_enabled   : {ctx.execution.ibkr_enabled}")
    _p(f"  ibkr_dry_run   : {ctx.execution.ibkr_dry_run}")
    _p(f"  kraken_enabled : {ctx.execution.kraken_enabled}")
    _p(f"  CHAD_MODE      : {ctx.chad_mode}")
    _p("")

    _p("STOP (authoritative)")
    _p("--------------------")
    _p(f"  stop   : {ctx.stop_state.stop}")
    _p(f"  reason : {ctx.stop_state.reason!r}")
    _p("")

    _p("Operator intent")
    _p("--------------")
    _p(f"  mode   : {ctx.operator_intent.mode}")
    _p(f"  reason : {ctx.operator_intent.reason!r}")
    _p("")

    _p("Shadow Confidence Router (SCR)")
    _p("------------------------------")
    _p(f"  state         : {ctx.shadow_state.state}")
    _p(f"  sizing_factor : {ctx.shadow_state.sizing_factor:.3f}")
    _p(f"  paper_only    : {ctx.shadow_state.paper_only}")
    if ctx.shadow_state.reasons:
        _p("  reasons       :")
        for r in ctx.shadow_state.reasons[:10]:
            _p(f"    - {r}")
    _p("")

    _p("LiveGate decision")
    _p("-----------------")
    _p(f"  mode            : {decision.mode}")
    _p(f"  allow_exits_only : {decision.allow_exits_only}")
    _p(f"  allow_ibkr_live  : {decision.allow_ibkr_live}")
    _p(f"  allow_ibkr_paper : {decision.allow_ibkr_paper}")
    _p("")
    _p("Reasons:")
    for r in decision.reasons:
        _p(f"  - {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

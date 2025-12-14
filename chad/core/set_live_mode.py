"""
CHAD Live Mode Setter CLI (Phase 8 – Operator Intent Only)

This module provides a small command-line interface for updating
CHAD's live-mode intent via the runtime/live_mode.json file.

IMPORTANT (Phase 7 / Early Phase 8):
------------------------------------
* This DOES NOT enable real live trading by itself.
* ExecutionConfig is still hard-locked to DRY_RUN for IBKR.
* LiveGate still returns allow_ibkr_live=False.
* This simply records operator intent ("live" vs "stop") so that
  future Phase 8 wiring can consider it as one of several gates.

Usage
-----
From the CHAD runtime root:

    # Show current live-mode state
    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.set_live_mode show

    # Set STOP / DRY_RUN intent
    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.set_live_mode stop --reason "operator_stop"

    # Set LIVE intent (still DRY_RUN in this build)
    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.set_live_mode live --reason "operator_go_live"

The live_mode.json file is read/written via chad.core.live_mode, which
ensures safe defaults and atomic writes.
"""

from __future__ import annotations

import argparse
from typing import Final

from chad.core.live_mode import LiveModeState, load_live_mode, set_live_mode

_DEFAULT_LIVE_REASON: Final[str] = "operator_go_live"
_DEFAULT_STOP_REASON: Final[str] = "operator_stop"


def _print_state(state: LiveModeState, header: str | None = None) -> None:
    """
    Pretty-print the LiveModeState for CLI usage.
    """
    if header:
        print(header)
    print("=== CHAD Live Mode State ===")
    print(f"live   : {state.live}")
    print(f"reason : {state.reason}")
    print()
    print("Notes:")
    print("  • In this Phase 7 build, adapters and execution remain hard-locked")
    print("    to DRY_RUN regardless of this flag.")
    print("  • This state reflects operator intent only and will be combined")
    print("    with CHAD_MODE, SCR, LiveGate, caps, and STOP flags in Phase 8.")


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the CLI.

    Subcommands:
        show   -> print current state
        live   -> set live=True
        stop   -> set live=False
    """
    parser = argparse.ArgumentParser(
        prog="python -m chad.core.set_live_mode",
        description="CHAD Live Mode Setter (operator intent only).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # show
    subparsers.add_parser(
        "show",
        help="Show current live-mode state.",
    )

    # live
    live_parser = subparsers.add_parser(
        "live",
        help="Set live-mode intent to LIVE (still DRY_RUN in this build).",
    )
    live_parser.add_argument(
        "--reason",
        type=str,
        default=_DEFAULT_LIVE_REASON,
        help=f"Reason for go-live intent (default: {_DEFAULT_LIVE_REASON!r}).",
    )

    # stop
    stop_parser = subparsers.add_parser(
        "stop",
        help="Set live-mode intent to STOP / DRY_RUN.",
    )
    stop_parser.add_argument(
        "--reason",
        type=str,
        default=_DEFAULT_STOP_REASON,
        help=f"Reason for stop intent (default: {_DEFAULT_STOP_REASON!r}).",
    )

    return parser


def main() -> int:
    """
    Entry point for the set_live_mode CLI.

    Returns
    -------
    int
        0 on success, non-zero on error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "show":
        state = load_live_mode()
        _print_state(state, header="Current live-mode state:")
        return 0

    if args.command == "live":
        reason: str = args.reason
        state = set_live_mode(True, reason)
        _print_state(state, header="Updated live-mode state (LIVE intent):")
        return 0

    if args.command == "stop":
        reason = args.reason
        state = set_live_mode(False, reason)
        _print_state(state, header="Updated live-mode state (STOP/DRY_RUN intent):")
        return 0

    # Should not reach here due to required=True on subparsers.
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
CHAD ExecutionConfig Inspector (Phase 7)

This CLI tool prints a unified view of:

    - CHAD_EXECUTION_MODE   → adapter-level ExecutionConfig
    - CHAD_MODE             → high-level global mode flag
    - Effective broker flags (IBKR enabled / dry-run)

Usage
-----
From the CHAD runtime root:

    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.show_execution_config

This is a *read-only* diagnostic. It never touches any broker.
"""

from __future__ import annotations

import os
from typing import Final

from chad.core.mode import get_chad_mode, is_live_mode_enabled
from chad.execution.execution_config import (
    ExecutionConfig,
    ExecutionMode,
    get_execution_config,
)

_ENV_VAR_NAME: Final[str] = "CHAD_EXECUTION_MODE"


def _format_bool(value: bool) -> str:
    """Format booleans as 'Yes' / 'No' for readability."""
    return "Yes" if value else "No"


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _print_execution_section(cfg: ExecutionConfig) -> None:
    _print_header("ExecutionConfig (adapter-level)")
    print(f"{_ENV_VAR_NAME:<18}: {os.getenv(_ENV_VAR_NAME, '(unset)')}")
    print(f"{'mode':<18}: {cfg.mode.value} ({cfg.mode!r})")
    print(f"{'ibkr_enabled':<18}: {_format_bool(cfg.ibkr_enabled)}")
    print(f"{'ibkr_dry_run':<18}: {_format_bool(cfg.ibkr_dry_run)}")
    print(f"{'kraken_enabled':<18}: {_format_bool(cfg.kraken_enabled)}")
    print()

    if cfg.mode is ExecutionMode.DRY_RUN and cfg.ibkr_dry_run:
        print("Effective IBKR behaviour:")
        print("  • IBKR adapter may initialise, but will ONLY operate in")
        print("    dry-run / what-if mode (no real orders placed).")
    else:
        print("Effective IBKR behaviour:")
        print("  • Non-standard configuration detected. In this Phase-7")
        print("    build, all IBKR_* modes should be forced back to DRY_RUN.")
    print()


def _print_chad_mode_section() -> None:
    mode = get_chad_mode()
    live_flag = is_live_mode_enabled()

    _print_header("CHAD_MODE (global)")
    env_val = os.environ.get("CHAD_MODE", "(unset)")
    print(f"{'CHAD_MODE env':<18}: {env_val}")
    print(f"{'normalized':<18}: {mode.value}")
    print(f"{'live_enabled':<18}: {_format_bool(live_flag)}")
    print()
    print("Notes:")
    print("  • In this Phase-7 build, CHAD_MODE is informational only.")
    print("  • Actual broker behaviour is governed by ExecutionConfig,")
    print("    which is currently hard-locked to DRY_RUN for IBKR.")
    print()


def main() -> int:
    """
    Entry point for the show_execution_config CLI.

    Returns
    -------
    int
        0 on success, non-zero on error.
    """
    try:
        cfg = get_execution_config()
    except Exception as exc:  # pragma: no cover - defensive
        print("ERROR: Failed to load ExecutionConfig:", exc)
        return 1

    print("=== CHAD Execution Configuration Snapshot ===")
    print()

    _print_execution_section(cfg)
    _print_chad_mode_section()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

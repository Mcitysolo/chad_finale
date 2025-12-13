"""
CHAD Mode Helper (Phase 8 – DRY_RUN vs LIVE)

This module provides a single, canonical way to read CHAD's global mode
from the environment:

    CHAD_MODE = "DRY_RUN" | "LIVE"

Defaults to DRY_RUN if unset or invalid.

IMPORTANT:
    This module does NOT execute trades or talk to brokers. It only
    reports the intended mode so other components can decide what to do.

Current policy on this EC2:
    - All execution runners (ibkr_execution_runner, etc.) are still
      hard-coded to WHAT-IF / dry-run. CHAD_MODE is informational-only
      until LIVE is explicitly wired and approved.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Final


class CHADMode(str, Enum):
    DRY_RUN = "DRY_RUN"
    LIVE = "LIVE"


DEFAULT_MODE: Final[CHADMode] = CHADMode.DRY_RUN


def get_chad_mode() -> CHADMode:
    """
    Read CHAD_MODE from the environment and normalize it.

    Allowed values (case-insensitive):
        - "DRY_RUN"
        - "LIVE"

    Anything else (or unset) ⇒ DRY_RUN.
    """
    raw = os.environ.get("CHAD_MODE", "").strip().upper()

    if raw in ("LIVE", "LIVE_TRADING"):
        return CHADMode.LIVE

    # Default and fallback: DRY_RUN
    return CHADMode.DRY_RUN


def is_live_mode_enabled() -> bool:
    """
    Convenience helper: True iff CHAD_MODE resolves to LIVE.
    """
    return get_chad_mode() == CHADMode.LIVE


def main() -> None:
    """
    CLI entrypoint:

        PYTHONPATH="..." python -m chad.core.mode
    """
    mode = get_chad_mode()
    live_flag = is_live_mode_enabled()

    print("=== CHAD MODE ===")
    print(f"CHAD_MODE env    : {os.environ.get('CHAD_MODE', '(unset)')}")
    print(f"normalized mode  : {mode.value}")
    print(f"live_enabled     : {live_flag}")


if __name__ == "__main__":
    main()

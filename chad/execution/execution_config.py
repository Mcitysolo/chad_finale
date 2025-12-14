"""
CHAD Execution Mode Configuration (Phase 7 – DRY_RUN Hard Lock)

This module centralises execution-mode handling for CHAD.

Key guarantees in this Phase-7 build:
-------------------------------------
* All IBKR-related modes are HARD-FORCED to DRY_RUN.
* No combination of environment variables can cause IBKR to place live orders.
* Other broker integrations (e.g. Kraken) are disabled by default.

Environment:
    CHAD_EXECUTION_MODE:
        - "dry_run"   -> ExecutionMode.DRY_RUN
        - "ibkr_paper" (forced to DRY_RUN)
        - "ibkr_live"  (forced to DRY_RUN)
        - Any other / invalid value -> DRY_RUN

This file is the single source of truth for execution behaviour at the
adapter level. Higher-level knobs like CHAD_MODE are *informational* in
this phase and do not override the hard DRY_RUN safety here.

Phase-8+ will relax this with an explicit, multi-step promotion flow
(CHAD_MODE + SCR + caps + Shadow Router + operator approval).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Final


LOGGER = logging.getLogger(__name__)

# Name of the environment variable controlling execution mode.
_ENV_VAR_NAME: Final[str] = "CHAD_EXECUTION_MODE"


class ExecutionMode(str, Enum):
    """
    Supported execution modes for CHAD.

    DRY_RUN:
        Default mode. IBKR adapter runs in dry-run / what-if mode and
        never places real orders. Safe for development, testing, and
        Phase-7 paper trading.

    IBKR_PAPER:
        Logical "paper" mode for IBKR. In this Phase-7 build it is
        HARD-FORCED back to DRY_RUN for safety.

    IBKR_LIVE:
        Logical "live" mode for IBKR. In this Phase-7 build it is
        HARD-FORCED back to DRY_RUN for safety.
    """

    DRY_RUN = "dry_run"
    IBKR_PAPER = "ibkr_paper"
    IBKR_LIVE = "ibkr_live"


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Concrete execution configuration used by broker adapters.

    Attributes
    ----------
    mode:
        The normalised high-level execution mode.

    ibkr_enabled:
        True if IBKR adapter should be available at all. In this Phase-7
        build we allow IBKR to initialise in dry-run mode only.

    ibkr_dry_run:
        True if IBKR adapter must operate in dry-run / what-if mode
        (no real orders placed). In this Phase-7 build this is ALWAYS
        True whenever ibkr_enabled is True.

    kraken_enabled:
        Placeholder for future Kraken integration. Always False in this
        build to avoid accidental activation.
    """

    mode: ExecutionMode
    ibkr_enabled: bool
    ibkr_dry_run: bool
    kraken_enabled: bool = False


def _parse_raw_mode(raw: str | None) -> ExecutionMode:
    """
    Parse the raw CHAD_EXECUTION_MODE value into an ExecutionMode.

    Any invalid, empty, or None value falls back to DRY_RUN.
    """
    if not raw:
        return ExecutionMode.DRY_RUN

    raw_normalised = raw.strip().lower()
    try:
        return ExecutionMode(raw_normalised)
    except ValueError:
        LOGGER.warning(
            "Invalid CHAD_EXECUTION_MODE value %r – falling back to DRY_RUN",
            raw,
        )
        return ExecutionMode.DRY_RUN


def get_execution_config() -> ExecutionConfig:
    """
    Compute the effective ExecutionConfig for the current process.

    Phase-7 Safety Override
    -----------------------
    Regardless of CHAD_EXECUTION_MODE:

        - ExecutionMode.IBKR_PAPER
        - ExecutionMode.IBKR_LIVE

    are both HARD-FORCED back to ExecutionMode.DRY_RUN.

    This ensures:
        * IBKR adapter remains in dry-run / what-if mode.
        * No live orders can be placed by accident in this build.

    Examples
    --------
    CHAD_EXECUTION_MODE=dry_run   -> mode=DRY_RUN,  ibkr_enabled=True,  ibkr_dry_run=True
    CHAD_EXECUTION_MODE=ibkr_live -> mode=DRY_RUN*, ibkr_enabled=True*, ibkr_dry_run=True*
    CHAD_EXECUTION_MODE=ibkr_paper-> mode=DRY_RUN*, ibkr_enabled=True*, ibkr_dry_run=True*

    (*) forced by Phase-7 safety override.
    """
    raw = os.getenv(_ENV_VAR_NAME, ExecutionMode.DRY_RUN.value)
    mode = _parse_raw_mode(raw)

    # ------------------------------------------------------------------
    # Phase-7 Safety Override:
    # Force ALL IBKR_* modes back to DRY_RUN so no live trading is
    # possible in this build, regardless of CHAD_EXECUTION_MODE.
    # ------------------------------------------------------------------
    if mode in (ExecutionMode.IBKR_PAPER, ExecutionMode.IBKR_LIVE):
        LOGGER.info(
            "Execution mode %s requested via %s, but Phase-7 safety override "
            "forces mode=DRY_RUN (ibkr_dry_run=True).",
            mode.value,
            _ENV_VAR_NAME,
        )
        mode = ExecutionMode.DRY_RUN

    # Map high-level mode to concrete adapter behaviour.
    if mode is ExecutionMode.DRY_RUN:
        # Current behaviour: IBKR adapter is allowed to initialise but
        # must operate in dry-run / what-if mode only.
        return ExecutionConfig(
            mode=mode,
            ibkr_enabled=True,
            ibkr_dry_run=True,
            kraken_enabled=False,
        )

    # In this Phase-7 build we should never reach this point because all
    # non-DRY_RUN modes are forced back to DRY_RUN above. The following
    # branch exists only as a defensive fallback and to make future
    # Phase-8+ diffing easier.
    LOGGER.warning(
        "Unexpected non-DRY_RUN mode %s after Phase-7 override – "
        "falling back to DRY_RUN ExecutionConfig.",
        mode.value,
    )
    return ExecutionConfig(
        mode=ExecutionMode.DRY_RUN,
        ibkr_enabled=True,
        ibkr_dry_run=True,
        kraken_enabled=False,
    )


if __name__ == "__main__":
    # Small self-check helper:
    #   PYTHONPATH="..." python -m chad.execution.execution_config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = get_execution_config()
    print("=== CHAD ExecutionConfig ===")
    print(f"{_ENV_VAR_NAME} =", os.getenv(_ENV_VAR_NAME, "(unset)"))
    print(f"mode            : {cfg.mode}")
    print(f"ibkr_enabled    : {cfg.ibkr_enabled}")
    print(f"ibkr_dry_run    : {cfg.ibkr_dry_run}")
    print(f"kraken_enabled  : {cfg.kraken_enabled}")

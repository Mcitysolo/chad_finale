"""
CHAD Execution Mode Configuration (Phase 7 – DRY_RUN Hard Lock)

This module centralises execution-mode handling for CHAD.

Key guarantees in this Phase-7 build:
-------------------------------------
* All IBKR-related modes are HARD-FORCED to DRY_RUN.
* No combination of environment variables can cause IBKR to place live orders.
* Kraken integration remains OFF by default and requires explicit opt-in.
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

# Explicit opt-in flag for Kraken in Phase 7.
# Must be set to a truthy value to enable Kraken adapter paths.
_KRAKEN_ENV: Final[str] = "CHAD_KRAKEN_ENABLED"


class ExecutionMode(str, Enum):
    """
    Supported execution modes for CHAD.

    DRY_RUN:
        Default mode. IBKR adapter runs in dry-run / what-if mode and
        never places real orders. Safe for development, testing, and
        Phase-7 paper trading.

    IBKR_PAPER / IBKR_LIVE:
        Logical modes that are forcibly coerced back to DRY_RUN in Phase 7.
    """

    DRY_RUN = "dry_run"
    IBKR_PAPER = "ibkr_paper"
    IBKR_LIVE = "ibkr_live"


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Concrete execution configuration used by broker adapters.

    kraken_enabled:
        In Phase 7, Kraken remains OFF by default. It can only be enabled by
        setting CHAD_KRAKEN_ENABLED=1 (or yes/true/on).
    """

    mode: ExecutionMode
    ibkr_enabled: bool
    ibkr_dry_run: bool
    kraken_enabled: bool = False


def _parse_raw_mode(raw: str | None) -> ExecutionMode:
    if not raw:
        return ExecutionMode.DRY_RUN
    raw_normalised = raw.strip().lower()
    try:
        return ExecutionMode(raw_normalised)
    except ValueError:
        LOGGER.warning("Invalid %s value %r – falling back to DRY_RUN", _ENV_VAR_NAME, raw)
        return ExecutionMode.DRY_RUN


def _truthy_env(name: str) -> bool:
    """
    Interpret environment variable as boolean truthy.

    Truthy values:
      1, true, yes, y, on
    """
    v = os.environ.get(name, "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def get_execution_config() -> ExecutionConfig:
    """
    Compute effective ExecutionConfig for the current process.

    Phase-7 Safety Override:
      - IBKR modes are always forced to DRY_RUN.
      - Kraken is OFF unless explicitly enabled via CHAD_KRAKEN_ENABLED.
    """
    raw = os.getenv(_ENV_VAR_NAME, ExecutionMode.DRY_RUN.value)
    mode = _parse_raw_mode(raw)

    # Force all IBKR modes back to DRY_RUN in Phase 7.
    if mode in (ExecutionMode.IBKR_PAPER, ExecutionMode.IBKR_LIVE):
        LOGGER.info(
            "Execution mode %s requested via %s, but Phase-7 safety override forces DRY_RUN.",
            mode.value,
            _ENV_VAR_NAME,
        )
        mode = ExecutionMode.DRY_RUN

    kraken_enabled = _truthy_env(_KRAKEN_ENV)

    # In Phase 7 we always return DRY_RUN adapter behavior.
    return ExecutionConfig(
        mode=ExecutionMode.DRY_RUN,
        ibkr_enabled=True,
        ibkr_dry_run=True,
        kraken_enabled=bool(kraken_enabled),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = get_execution_config()
    print("=== CHAD ExecutionConfig ===")
    print(f"{_ENV_VAR_NAME} =", os.getenv(_ENV_VAR_NAME, "(unset)"))
    print(f"{_KRAKEN_ENV} =", os.getenv(_KRAKEN_ENV, "(unset)"))
    print(f"mode            : {cfg.mode}")
    print(f"ibkr_enabled    : {cfg.ibkr_enabled}")
    print(f"ibkr_dry_run    : {cfg.ibkr_dry_run}")
    print(f"kraken_enabled  : {cfg.kraken_enabled}")

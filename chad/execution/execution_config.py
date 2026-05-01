"""
CHAD Execution Mode Configuration

Purpose
-------
Single authoritative execution-mode resolver for broker adapters.

This module determines the effective adapter posture from environment,
while staying explicit, auditable, and fail-closed.

Supported modes
---------------
- dry_run     : safe what-if mode
- ibkr_paper  : paper-trading lane
- ibkr_live   : live-trading lane

Design rules
------------
- No hidden coercion.
- No silent upgrade from safer mode to riskier mode.
- Invalid values fail closed to dry_run.
- Kraken remains separately controlled by CHAD_KRAKEN_ENABLED.
- This module only resolves config. It does NOT authorize trading.
  Final authorization still belongs to LiveGate and the caller.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Final

LOGGER = logging.getLogger(__name__)

_ENV_VAR_NAME: Final[str] = "CHAD_EXECUTION_MODE"
_KRAKEN_ENV: Final[str] = "CHAD_KRAKEN_ENABLED"


class ExecutionMode(str, Enum):
    DRY_RUN = "dry_run"
    IBKR_PAPER = "ibkr_paper"
    IBKR_LIVE = "ibkr_live"


# Vocabulary aliases — CHAD_EXECUTION_MODE env var has historically used the
# short forms "paper" / "live"; the canonical enum uses "ibkr_paper" /
# "ibkr_live". Map both forms to the enum so direct os.environ readers can be
# migrated without changing operator-facing env values.
_ALIASES: Final[Dict[str, str]] = {
    "paper": "ibkr_paper",
    "live": "ibkr_live",
    "ibkr_paper": "ibkr_paper",
    "ibkr_live": "ibkr_live",
    "dry_run": "dry_run",
}


@dataclass(frozen=True)
class ExecutionConfig:
    mode: ExecutionMode
    ibkr_enabled: bool
    ibkr_dry_run: bool
    kraken_enabled: bool = False


def _parse_raw_mode(raw: str | None) -> ExecutionMode:
    if not raw:
        return ExecutionMode.DRY_RUN

    raw_normalised = _ALIASES.get(raw.strip().lower(), raw.strip().lower())
    try:
        return ExecutionMode(raw_normalised)
    except ValueError:
        LOGGER.warning(
            "Invalid %s value %r - falling back to %s",
            _ENV_VAR_NAME,
            raw,
            ExecutionMode.DRY_RUN.value,
        )
        return ExecutionMode.DRY_RUN


def get_execution_mode() -> ExecutionMode:
    """Single canonical reader for CHAD_EXECUTION_MODE.

    Normalizes paper -> ibkr_paper, live -> ibkr_live. All production code
    must call this instead of reading os.environ directly so that the
    paper/live vs ibkr_paper/ibkr_live vocabulary mismatch is contained
    in one place (ISSUE-78).
    """
    return _parse_raw_mode(os.getenv(_ENV_VAR_NAME))


def is_paper_mode() -> bool:
    """True when execution mode is paper or dry_run (i.e. NOT live)."""
    return get_execution_mode() in (ExecutionMode.DRY_RUN, ExecutionMode.IBKR_PAPER)


def is_live_mode() -> bool:
    """True only when execution mode is ibkr_live."""
    return get_execution_mode() == ExecutionMode.IBKR_LIVE


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def get_execution_config() -> ExecutionConfig:
    """
    Resolve effective execution configuration.

    Behavior:
    - dry_run    -> IBKR enabled, dry-run=True
    - ibkr_paper -> IBKR enabled, dry-run=True
    - ibkr_live  -> IBKR enabled, dry-run=False

    This function does NOT decide whether live trading is allowed.
    It only exposes the adapter posture requested by environment.
    """
    raw = os.getenv(_ENV_VAR_NAME, ExecutionMode.DRY_RUN.value)
    mode = _parse_raw_mode(raw)
    kraken_enabled = _truthy_env(_KRAKEN_ENV)

    if mode == ExecutionMode.IBKR_LIVE:
        ibkr_dry_run = False
    else:
        ibkr_dry_run = True

    return ExecutionConfig(
        mode=mode,
        ibkr_enabled=True,
        ibkr_dry_run=ibkr_dry_run,
        kraken_enabled=bool(kraken_enabled),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = get_execution_config()
    print("=== CHAD ExecutionConfig ===")
    print(f"{_ENV_VAR_NAME} = {os.getenv(_ENV_VAR_NAME, '(unset)')}")
    print(f"{_KRAKEN_ENV} = {os.getenv(_KRAKEN_ENV, '(unset)')}")
    print(f"mode            : {cfg.mode.value}")
    print(f"ibkr_enabled    : {cfg.ibkr_enabled}")
    print(f"ibkr_dry_run    : {cfg.ibkr_dry_run}")
    print(f"kraken_enabled  : {cfg.kraken_enabled}")

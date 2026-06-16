"""Shared futures-execution off-switch — single source of truth.

This module owns the reversible env-flag predicate that hard-blocks futures
(FUT/FOP) order submission, plus the secType classifier used to decide which
contracts the off-switch applies to. Both ``chad.core.live_loop`` (early skip)
and the broker-submit chokepoints (``chad.execution.ibkr_adapter`` and
``chad.execution.ibkr_trade_router``) import from here so the flag logic lives
in exactly one place and is never copy-pasted.

Design notes:
- Imports nothing from ``chad`` — safe to import from any execution module
  (no circular-import risk).
- ``futures_execution_disabled`` recognises the same three flags the original
  ``live_loop._futures_execution_disabled`` did, with identical truthy/falsy
  vocabularies, so the move is behaviour-preserving.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

# secTypes governed by the futures off-switch. FOP (futures options) is included
# alongside FUT so the gate covers every futures-rooted instrument.
FUTURES_SEC_TYPES = frozenset({"FUT", "FOP"})


def futures_execution_disabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when any futures-disable flag is set. Reversible env stopgap.

    Recognised flags (ANY one triggers the block):
      - ``CHAD_DISABLE_FUTURES_EXECUTION`` truthy
      - ``CHAD_DISABLE_FUTURES`` truthy
      - ``CHAD_FUTURES_EXECUTION_ENABLED`` falsy

    ``env`` defaults to ``os.environ`` when omitted. Unrecognised values are
    treated as "not set" (fail-open only for garbage; explicit truthy/falsy
    are honoured).
    """
    if env is None:
        env = os.environ
    if env.get("CHAD_DISABLE_FUTURES_EXECUTION", "").strip().lower() in _TRUTHY:
        return True
    if env.get("CHAD_DISABLE_FUTURES", "").strip().lower() in _TRUTHY:
        return True
    if env.get("CHAD_FUTURES_EXECUTION_ENABLED", "").strip().lower() in _FALSY:
        return True
    return False


def is_futures_sec_type(sec_type: object) -> bool:
    """True for FUT/FOP secType (case-insensitive, whitespace-tolerant)."""
    return str(sec_type or "").strip().upper() in FUTURES_SEC_TYPES

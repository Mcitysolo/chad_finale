"""chad/execution/rth_gate.py — regular-trading-hours (RTH) submission gate.

WKF U2. A pure, side-effect-free decision helper that blocks EQUITY/ETF order
intents which would submit OUTSIDE the US equity regular session. It sits at
the same pre-claim chokepoint as the margin gate (``ibkr_adapter._submit_intent``)
so a blocked intent writes no idempotency row.

Scope / exemptions (deliberate):
  * Gated:   asset_class in {equity, etf}.
  * Exempt:  futures / futures-options (nearly 24h sessions), forex, options,
             and unknown. Crypto never reaches the IBKR adapter at all — it
             flows through the separate Kraken lane — so it is exempt by
             construction; this module additionally treats "crypto" as exempt
             for defence in depth.

Toggle: ``CHAD_RTH_GATE`` env var, DEFAULT ON. Only an explicit falsy value
(0/false/no/off) disables it — this mirrors the futures off-switch idiom in
``chad/execution/futures_gate.py``. Tests disable with ``CHAD_RTH_GATE=0``.

RTH window is delegated to :func:`chad.utils.market_hours.equity_rth_is_open`
(DST-aware US equity 09:30-16:00 ET; 13:30-20:00 UTC in the current EDT regime,
14:30-21:00 UTC in EST). Half-day / exchange-holiday handling is a named TODO
there (WKF-U2-HALFDAY).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Mapping, Optional

from chad.utils.market_hours import equity_rth_is_open

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

RTH_GATE_ENV = "CHAD_RTH_GATE"

# Asset classes subject to the RTH gate. Everything else is exempt.
_RTH_GATED_ASSET_CLASSES = frozenset({"equity", "etf"})

# The canonical block-reason / status string (also added to live_loop's
# _UNCONFIRMED_BROKER_STATUSES so it can never be mistaken for a fill).
RTH_BLOCK_STATUS = "market_closed"


def rth_gate_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the RTH gate is active. DEFAULT ON.

    Only an explicit falsy ``CHAD_RTH_GATE`` value disables it; an unset or
    unrecognised value leaves the gate ON (fail-safe toward gating).
    """
    if env is None:
        env = os.environ
    raw = str(env.get(RTH_GATE_ENV, "") or "").strip().lower()
    if raw in _FALSY:
        return False
    return True


def is_rth_gated_asset(asset_class: Any) -> bool:
    """True iff ``asset_class`` (equity/ETF) is subject to the RTH gate."""
    return str(asset_class or "").strip().lower() in _RTH_GATED_ASSET_CLASSES


def rth_block_reason(
    asset_class: Any,
    now_utc: datetime,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return a block-reason string iff this intent must be RTH-blocked, else None.

    Blocks only when ALL hold: the gate is enabled, the asset is equity/ETF,
    and the US equity regular session is CLOSED at ``now_utc``. Futures / crypto
    / options / forex / unknown always return None (exempt).
    """
    if not rth_gate_enabled(env):
        return None
    if not is_rth_gated_asset(asset_class):
        return None
    if equity_rth_is_open(now_utc):
        return None
    return "market_closed_outside_rth"


__all__ = [
    "RTH_GATE_ENV",
    "RTH_BLOCK_STATUS",
    "rth_gate_enabled",
    "is_rth_gated_asset",
    "rth_block_reason",
]

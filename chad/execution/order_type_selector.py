"""
chad/execution/order_type_selector.py

Phase-8 Session 6 (E4): passive vs aggressive order-type selector.

Maps intent.order_urgency + an estimated spread into a concrete order-type
record that the execution pipeline uses to shape the limit price it sends
to IBKR / Kraken. Never returns a MKT order — "aggressive" means a
marketable limit priced through the top of book, not a market order.

Decision matrix
---------------

    urgency="normal" and spread <= threshold → PASSIVE
        order_type="LMT", aggressive=False, price_offset_pct=0.0
        (strategy's limit_price is used as-is; fills patiently at mid
         or one tick inside)

    urgency="high" OR spread > threshold     → AGGRESSIVE
        order_type="LMT", aggressive=True, price_offset_pct=0.001
        (limit is pushed 10 bps through the market on the active side:
         BUY priced slightly above ask, SELL slightly below bid;
         guarantees marketability without using MKT)

The 10 bps offset is conservative: 1 bps buys roughly 1 penny on a $100
name and 1 cent on a mid-three-figure crypto. Wider-spread instruments
already trip the aggressive branch via the spread check and do not need
a larger offset.

Spread estimation
-----------------

A bar's high-low range is a *daily* spread estimate — far too wide to
use directly as the current bid-ask width. `estimate_spread_pct` divides
that range by 10 as a rough heuristic (intraday spread ≪ daily range);
this is a stand-in until real top-of-book quotes are plumbed through.
Missing bar data returns a conservative 10 bps default.

Safety contract
---------------

  * Never returns order_type="MKT".
  * Unknown / garbage urgency values are treated as "normal".
  * Negative or NaN spread values fall back to the default.
  * The function is pure — no I/O, no side effects, safe to call from
    the hot path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default spread estimate used when we cannot derive one from bar data.
# 10 bps is a reasonable mid-liquidity equity spread; below this threshold
# the passive branch fires.
DEFAULT_SPREAD_PCT: float = 0.001

# When the estimated spread is above this threshold the aggressive branch
# fires regardless of urgency — a wide spread means the passive limit at
# mid is unlikely to fill in a reasonable time.
DEFAULT_SPREAD_THRESHOLD_PCT: float = 0.003

# Offset applied through the market for aggressive orders. 10 bps.
AGGRESSIVE_PRICE_OFFSET_PCT: float = 0.001

# The only order_type this module ever emits. Market orders are banned.
ORDER_TYPE_LIMIT: str = "LMT"

# Canonical urgency vocabulary. Anything else is treated as "normal".
_VALID_URGENCY = frozenset({"normal", "high"})


def _safe_float(value: Any, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _normalize_urgency(urgency: Any) -> str:
    u = str(urgency or "normal").strip().lower()
    if u not in _VALID_URGENCY:
        return "normal"
    return u


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_order_type(
    urgency: str = "normal",
    estimated_spread_pct: float = DEFAULT_SPREAD_PCT,
    spread_threshold_pct: float = DEFAULT_SPREAD_THRESHOLD_PCT,
) -> Dict[str, Any]:
    """Return order-type parameters for one intent.

    Returned dict schema::

        {
            "order_type": "LMT",              # always LMT — never MKT
            "aggressive": bool,
            "price_offset_pct": float,        # 0.0 for passive; +0.001 aggressive
            "reason": str,                    # short diagnostic for logs
        }

    Parameters
    ----------
    urgency
        'normal' (default, passive) or 'high' (always aggressive). Unknown
        values are treated as 'normal'.
    estimated_spread_pct
        Current estimated spread as a fraction of mid price (e.g. 0.001
        == 10 bps). Values above ``spread_threshold_pct`` force aggressive.
    spread_threshold_pct
        Cutoff above which the spread itself forces aggressive routing.
    """
    u = _normalize_urgency(urgency)
    spread = _safe_float(estimated_spread_pct, DEFAULT_SPREAD_PCT)
    if spread < 0.0:
        spread = DEFAULT_SPREAD_PCT
    threshold = _safe_float(spread_threshold_pct, DEFAULT_SPREAD_THRESHOLD_PCT)
    if threshold <= 0.0:
        threshold = DEFAULT_SPREAD_THRESHOLD_PCT

    aggressive = (u == "high") or (spread > threshold)

    if aggressive:
        reason = "urgency_high" if u == "high" else "wide_spread"
        return {
            "order_type": ORDER_TYPE_LIMIT,
            "aggressive": True,
            "price_offset_pct": AGGRESSIVE_PRICE_OFFSET_PCT,
            "reason": reason,
        }

    return {
        "order_type": ORDER_TYPE_LIMIT,
        "aggressive": False,
        "price_offset_pct": 0.0,
        "reason": "passive_tight_spread",
    }


def estimate_spread_pct(
    symbol: str = "",
    bar_data: Optional[Mapping[str, Any]] = None,
    default_pct: float = DEFAULT_SPREAD_PCT,
) -> float:
    """Estimate current spread as a fraction of mid from a bar.

    Uses (high - low) / mid * 0.1 as a rough stand-in for the intraday
    bid-ask spread — the 0.1 factor shrinks the *daily* range down to a
    plausible tick-level spread, since intraday top-of-book is much
    narrower than the full session range. Returns ``default_pct`` for
    missing, malformed, or non-positive inputs.
    """
    if not bar_data:
        return default_pct
    try:
        high = float(bar_data.get("high", 0.0) or 0.0)
        low = float(bar_data.get("low", 0.0) or 0.0)
    except (TypeError, ValueError):
        return default_pct
    if high <= 0.0 or low <= 0.0 or high < low:
        return default_pct
    mid = (high + low) / 2.0
    if mid <= 0.0:
        return default_pct
    return max(0.0, (high - low) / mid * 0.1)


def compute_aggressive_limit_price(
    side: str,
    reference_price: float,
    price_offset_pct: float = AGGRESSIVE_PRICE_OFFSET_PCT,
) -> float:
    """Push a reference price through the market on the active side.

    BUY  → reference × (1 + offset)   (pay up through the ask)
    SELL → reference × (1 - offset)   (cross down through the bid)

    Works with any non-empty ``side`` string; anything starting with 'S'
    (case-insensitive) is treated as SELL. Returns ``reference_price``
    unchanged when ``reference_price`` is non-positive so downstream
    arithmetic cannot produce a negative limit.
    """
    ref = _safe_float(reference_price, 0.0)
    if ref <= 0.0:
        return ref
    off = _safe_float(price_offset_pct, 0.0)
    is_sell = str(side or "").strip().upper().startswith("S")
    if is_sell:
        return ref * (1.0 - off)
    return ref * (1.0 + off)


__all__ = [
    "AGGRESSIVE_PRICE_OFFSET_PCT",
    "DEFAULT_SPREAD_PCT",
    "DEFAULT_SPREAD_THRESHOLD_PCT",
    "ORDER_TYPE_LIMIT",
    "compute_aggressive_limit_price",
    "estimate_spread_pct",
    "select_order_type",
]

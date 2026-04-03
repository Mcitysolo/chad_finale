#!/usr/bin/env python3
"""
chad/options/strike_selector.py

Strike and expiry selection for vertical spread construction.

Given an options chain, current price, and directional bias, selects
the optimal expiry and strike pair for a vertical spread (bull call
or bear put).

Design
------
- Deterministic, no I/O, no side effects.
- Strict typing, fail-closed (returns None on bad inputs).
- DTE utility for expiry date math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional

from chad.options.chain_provider import OptionsChain


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpreadSpec:
    """
    Specification for a vertical option spread.

    Attributes
    ----------
    symbol : str
        Underlying symbol (e.g., "SPY").
    expiry : str
        Expiration date (YYYYMMDD format).
    long_strike : float
        Strike price for the long leg.
    short_strike : float
        Strike price for the short leg.
    right : str
        Option right: "C" for call, "P" for put.
    spread_type : str
        "BULL_CALL" or "BEAR_PUT".
    max_loss_per_contract : float
        Maximum loss per contract in dollars (spread_width * 100).
    net_debit_estimate : float
        Estimated net debit per contract (approximate, actual depends on fills).
    dte : int
        Days to expiration.
    """
    symbol: str
    expiry: str
    long_strike: float
    short_strike: float
    right: str  # "C" or "P"
    spread_type: str  # "BULL_CALL" or "BEAR_PUT"
    max_loss_per_contract: float
    net_debit_estimate: float
    dte: int


# ---------------------------------------------------------------------------
# DTE utility
# ---------------------------------------------------------------------------

def dte_from_expiry(expiry_str: str) -> int:
    """
    Calculate days to expiration from an expiry date string.

    Parameters
    ----------
    expiry_str : str
        Expiration date in YYYYMMDD format.

    Returns
    -------
    int
        Days until expiration (0 if today, negative if past).
    """
    try:
        exp_date = datetime.strptime(expiry_str.strip(), "%Y%m%d").date()
        today = datetime.now(timezone.utc).date()
        return (exp_date - today).days
    except (ValueError, TypeError):
        return -1


# ---------------------------------------------------------------------------
# Strike selection helpers
# ---------------------------------------------------------------------------

def _nearest_strike(strikes: List[float], target: float) -> Optional[float]:
    """Find the strike nearest to target price."""
    if not strikes or target <= 0:
        return None
    return min(strikes, key=lambda s: abs(s - target))


def _nearest_strike_above(strikes: List[float], target: float) -> Optional[float]:
    """Find the nearest strike at or above target."""
    above = [s for s in strikes if s >= target]
    return min(above) if above else None


def _nearest_strike_below(strikes: List[float], target: float) -> Optional[float]:
    """Find the nearest strike at or below target."""
    below = [s for s in strikes if s <= target]
    return max(below) if below else None


# ---------------------------------------------------------------------------
# Spread selector
# ---------------------------------------------------------------------------

def select_vertical_spread(
    chain: OptionsChain,
    current_price: float,
    direction: str,
    *,
    target_dte_min: int = 21,
    target_dte_max: int = 45,
    otm_offset_pct: float = 0.02,
    spread_width_pct: float = 0.05,
) -> Optional[SpreadSpec]:
    """
    Select a vertical spread from the options chain.

    Parameters
    ----------
    chain : OptionsChain
        Available expirations and strikes.
    current_price : float
        Current underlying price.
    direction : str
        "bullish" for bull call spread, "bearish" for bear put spread.
    target_dte_min : int
        Minimum acceptable days to expiration (default 21).
    target_dte_max : int
        Maximum acceptable days to expiration (default 45).
    otm_offset_pct : float
        How far OTM to place the long leg as a fraction of price (default 0.02 = 2%).
    spread_width_pct : float
        Spread width as a fraction of price (default 0.05 = 5%).

    Returns
    -------
    Optional[SpreadSpec]
        Spread specification, or None if no suitable expiry/strikes found.

    Spread Construction
    -------------------
    Bull call spread (bullish):
      - Long: call at ATM (nearest to current_price)
      - Short: call at ATM + spread_width
      - Max loss = spread_width * 100 (per contract)

    Bear put spread (bearish):
      - Long: put at ATM (nearest to current_price)
      - Short: put at ATM - spread_width
      - Max loss = spread_width * 100 (per contract)
    """
    if current_price <= 0:
        return None

    direction = direction.strip().lower()
    if direction not in ("bullish", "bearish"):
        return None

    if not chain.expirations or not chain.strikes:
        return None

    # Step 1: Select expiry in DTE range
    expiry = _select_expiry(chain.expirations, target_dte_min, target_dte_max)
    if expiry is None:
        return None
    dte = dte_from_expiry(expiry)

    # Step 2: Calculate strike targets
    spread_width = current_price * spread_width_pct
    strikes = chain.strikes

    if direction == "bullish":
        # Bull call spread: buy ATM call, sell higher call
        long_strike = _nearest_strike(strikes, current_price)
        if long_strike is None:
            return None
        short_target = long_strike + spread_width
        short_strike = _nearest_strike_above(strikes, short_target)
        if short_strike is None or short_strike <= long_strike:
            return None
        right = "C"
        spread_type = "BULL_CALL"
        actual_width = short_strike - long_strike

    else:
        # Bear put spread: buy ATM put, sell lower put
        long_strike = _nearest_strike(strikes, current_price)
        if long_strike is None:
            return None
        short_target = long_strike - spread_width
        short_strike = _nearest_strike_below(strikes, short_target)
        if short_strike is None or short_strike >= long_strike:
            return None
        right = "P"
        spread_type = "BEAR_PUT"
        actual_width = long_strike - short_strike

    max_loss = actual_width * 100.0  # per contract in dollars

    # Estimate net debit (rough: ~40-60% of spread width for ATM spreads)
    net_debit_estimate = actual_width * 100.0 * 0.50

    return SpreadSpec(
        symbol=chain.symbol,
        expiry=expiry,
        long_strike=long_strike,
        short_strike=short_strike,
        right=right,
        spread_type=spread_type,
        max_loss_per_contract=max_loss,
        net_debit_estimate=net_debit_estimate,
        dte=dte,
    )


def _select_expiry(
    expirations: List[str],
    dte_min: int,
    dte_max: int,
) -> Optional[str]:
    """
    Select the nearest expiry within the DTE range.

    Prefers the expiry closest to the midpoint of [dte_min, dte_max].
    Returns None if no expiry falls in range.
    """
    target_mid = (dte_min + dte_max) / 2.0
    best: Optional[str] = None
    best_dist = float("inf")

    for exp in expirations:
        dte = dte_from_expiry(exp)
        if dte < dte_min or dte > dte_max:
            continue
        dist = abs(dte - target_mid)
        if dist < best_dist:
            best = exp
            best_dist = dist

    return best


__all__ = [
    "SpreadSpec",
    "dte_from_expiry",
    "select_vertical_spread",
]

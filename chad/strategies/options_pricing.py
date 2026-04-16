#!/usr/bin/env python3
"""
chad/strategies/options_pricing.py

Black-Scholes pricing engine for synthetic options pricing.

Used by omega_momentum_options when real options chain cache is unavailable
(IBKR options subscription not active). Provides:

- black_scholes_price(): vanilla European option pricing
- estimate_iv_from_vix(): IV estimation from VIX level by asset class
- select_strike(): OTM strike selection with rounding
- select_expiry_dte(): next Friday >= target DTE
- estimate_contract_price(): full synthetic contract specification
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_FREE_RATE = 0.05

# IV scaling factors relative to VIX (VIX is 30-day SPY IV in pct points)
_IV_SCALE: Dict[str, float] = {
    "SPY": 1.0,
    "QQQ": 1.0,
    "AAPL": 1.3,
    "MSFT": 1.3,
    "GOOGL": 1.3,
    "NVDA": 1.3,
    "BTC-USD": 3.0,
    "ETH-USD": 3.0,
}
_IV_SCALE_DEFAULT = 1.5


# ---------------------------------------------------------------------------
# Normal CDF approximation (avoids scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the Abramowitz & Stegun approximation."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    European option price via Black-Scholes.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to expiry in years
    r : risk-free rate
    sigma : implied volatility (annualized, decimal)
    option_type : 'C' for call, 'P' for put

    Returns
    -------
    float : option price per share, minimum 0.01
    """
    if S <= 0 or K <= 0 or sigma <= 0:
        return 0.01
    if T <= 0:
        # At expiry: intrinsic value only
        if option_type == "C":
            return max(0.01, S - K)
        return max(0.01, K - S)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "C":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return max(0.01, price)


# ---------------------------------------------------------------------------
# IV estimation
# ---------------------------------------------------------------------------

def estimate_iv_from_vix(vix: float, symbol: str) -> float:
    """
    Estimate annualized implied volatility from VIX for a given symbol.

    VIX is SPY 30-day IV expressed in percentage points.
    Individual stocks and crypto have higher IV scaled by asset class.
    """
    scale = _IV_SCALE.get(symbol, _IV_SCALE_DEFAULT)
    iv = (vix / 100.0) * scale
    return max(0.10, iv)


# ---------------------------------------------------------------------------
# Strike selection
# ---------------------------------------------------------------------------

def select_strike(
    S: float,
    direction: str,
    otm_pct: float = 0.02,
) -> float:
    """
    Select an OTM strike rounded to standard option increments.

    For calls: round up from S * (1 + otm_pct)
    For puts:  round down from S * (1 - otm_pct)

    Rounding: $1 increments for underlyings < $100, $5 for >= $100.
    """
    step = 1.0 if S < 100.0 else 5.0

    if direction == "BUY_CALL":
        raw = S * (1.0 + otm_pct)
        return math.ceil(raw / step) * step
    else:  # BUY_PUT
        raw = S * (1.0 - otm_pct)
        return math.floor(raw / step) * step


# ---------------------------------------------------------------------------
# Expiry selection
# ---------------------------------------------------------------------------

def select_expiry_dte(target_dte: int = 7) -> str:
    """
    Return the next Friday that is >= target_dte days away.

    Returns YYYYMMDD string.
    """
    now = datetime.now(timezone.utc)
    candidate = now + timedelta(days=target_dte)
    # Advance to next Friday (weekday 4)
    days_until_friday = (4 - candidate.weekday()) % 7
    if days_until_friday == 0 and candidate.date() == now.date():
        days_until_friday = 7
    expiry_date = candidate + timedelta(days=days_until_friday)
    return expiry_date.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Full synthetic contract
# ---------------------------------------------------------------------------

def estimate_contract_price(
    S: float,
    direction: str,
    vix: float,
    symbol: str,
    dte: int = 7,
) -> dict:
    """
    Build a full synthetic contract specification using Black-Scholes.

    Returns dict with strike, expiry, right, estimated_price, iv, delta, synthetic flag.
    """
    right = "C" if direction == "BUY_CALL" else "P"
    strike = select_strike(S, direction)
    expiry = select_expiry_dte(dte)
    iv = estimate_iv_from_vix(vix, symbol)
    T = dte / 365.0

    price = black_scholes_price(S, strike, T, RISK_FREE_RATE, iv, right)

    # Approximate delta using BS d1
    if T > 0 and iv > 0:
        d1 = (math.log(S / strike) + (RISK_FREE_RATE + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
        if right == "C":
            delta = round(_norm_cdf(d1), 3)
        else:
            delta = round(_norm_cdf(d1) - 1.0, 3)
    else:
        delta = 0.5 if right == "C" else -0.5

    return {
        "strike": strike,
        "expiry": expiry,
        "right": right,
        "estimated_price": round(price, 2),
        "iv": round(iv, 4),
        "delta": delta,
        "synthetic": True,
    }

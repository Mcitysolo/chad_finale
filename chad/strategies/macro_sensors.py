#!/usr/bin/env python3
"""
chad/strategies/macro_sensors.py

Shared macro sensing utilities for CHAD macro strategies.

Provides reusable building blocks for macro regime detection:
- VIX extraction from multiple context key formats
- Portfolio drawdown estimation (peak-to-trough)
- ATR as percentage of price (volatility proxy)
- EMA slope for trend direction
- MacroRegime 4-state classifier

Design
------
- Deterministic, no I/O, no side effects.
- Strict typing throughout.
- Fail-closed: missing data -> NEUTRAL regime, None values.
- Imported by omega_macro.py and potentially future macro strategies.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    """Safely convert any value to float; return default on failure or NaN/Inf."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


# ---------------------------------------------------------------------------
# MacroRegime enum
# ---------------------------------------------------------------------------

class MacroRegime(str, Enum):
    """
    Four-state macro regime classification.

    RISK_ON:     Low volatility, healthy equity, growth conditions.
    RISK_OFF:    High volatility or significant drawdown — flight to safety.
    STAGFLATION: Elevated volatility with rising commodities and falling bonds.
    NEUTRAL:     No strong macro signal — stay flat or minimal exposure.
    """
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    STAGFLATION = "stagflation"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# VIX extraction
# ---------------------------------------------------------------------------

def _vix_value(ctx: Any) -> Optional[float]:
    """
    Extract VIX value from market context with multiple key fallbacks.

    Searches for VIX data across common context attribute names:
    - ctx.vix (numeric or dict with 'value' key)
    - ctx.vol_index
    - ctx.volatility_index

    Parameters
    ----------
    ctx : Any
        Market context object (typically MarketContext).

    Returns
    -------
    Optional[float]
        VIX value if found and valid, else None.
    """
    for attr in ("vix", "vol_index", "volatility_index"):
        raw = getattr(ctx, attr, None)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            v = _safe_float(raw)
            if v > 0:
                return v
        elif isinstance(raw, dict):
            for key in ("value", "last", "close"):
                if key in raw:
                    v = _safe_float(raw[key])
                    if v > 0:
                        return v

    # Fallback: check ctx.prices for VIX scalar
    try:
        prices = getattr(ctx, "prices", None) or {}
        if isinstance(prices, dict):
            for key in ("VIX", "^VIX", "vix"):
                val = prices.get(key)
                if val is not None:
                    v = float(val)
                    if 0 < v < 200:
                        return v
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Portfolio drawdown
# ---------------------------------------------------------------------------

def _portfolio_drawdown_pct(ctx: Any) -> Optional[float]:
    """
    Estimate portfolio drawdown as a percentage from equity peak.

    Looks for equity_peak and equity in ctx.portfolio.extra dict.

    Parameters
    ----------
    ctx : Any
        Market context object with a portfolio attribute.

    Returns
    -------
    Optional[float]
        Drawdown as a negative fraction (e.g., -0.06 for -6%), or None
        if equity data is unavailable. Returns 0.0 if at or above peak.
    """
    portfolio = getattr(ctx, "portfolio", None)
    if portfolio is None:
        return None
    extra = getattr(portfolio, "extra", None)
    if not isinstance(extra, dict):
        return None
    peak = extra.get("equity_peak")
    current = extra.get("equity")
    if peak is None or current is None:
        return None
    peak_f = _safe_float(peak, 0.0)
    current_f = _safe_float(current, 0.0)
    if peak_f <= 0.0:
        return None
    dd = (current_f - peak_f) / peak_f
    return float(dd)


# ---------------------------------------------------------------------------
# ATR percentage
# ---------------------------------------------------------------------------

def _atr_pct_from_bars(
    bars: Sequence[Mapping[str, Any]],
    period: int = 14,
) -> Optional[float]:
    """
    Compute ATR as a percentage of the latest close price.

    Uses true range (max of H-L, |H-prevC|, |L-prevC|) averaged over
    the specified period using a simple moving average of the last
    `period` true range values.

    Parameters
    ----------
    bars : Sequence[Mapping[str, Any]]
        OHLCV bar data. Each bar must have 'high', 'low', 'close' keys.
    period : int
        ATR lookback period. Default 14.

    Returns
    -------
    Optional[float]
        ATR / close as a fraction (e.g., 0.025 for 2.5%), or None
        if insufficient data.
    """
    if len(bars) < period + 2:
        return None

    trs: List[float] = []
    prev_close = _safe_float(bars[0].get("close"), 0.0)
    if prev_close <= 0:
        return None

    for bar in bars[1:]:
        high = _safe_float(bar.get("high"), 0.0)
        low = _safe_float(bar.get("low"), 0.0)
        close = _safe_float(bar.get("close"), 0.0)
        if high <= 0 or low <= 0 or close <= 0 or high < low:
            prev_close = close if close > 0 else prev_close
            continue
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if tr > 0:
            trs.append(tr)
        prev_close = close

    if len(trs) < period:
        return None

    atr_val = sum(trs[-period:]) / period
    last_close = _safe_float(bars[-1].get("close"), 0.0)
    if last_close <= 0:
        return None
    return float(atr_val / last_close)


# ---------------------------------------------------------------------------
# EMA slope
# ---------------------------------------------------------------------------

def _ema_slope(
    prices: Sequence[float],
    period: int = 20,
) -> Optional[float]:
    """
    Compute normalized EMA slope for trend direction detection.

    Calculates two EMA values — the current and one bar prior — and
    returns the normalized difference: (ema_now - ema_prev) / ema_prev.

    A positive value indicates an uptrend; negative indicates downtrend.
    The magnitude reflects the rate of change.

    Parameters
    ----------
    prices : Sequence[float]
        Close price series, oldest first.
    period : int
        EMA period. Default 20.

    Returns
    -------
    Optional[float]
        Normalized EMA slope, or None if insufficient data.
        Positive = uptrend, negative = downtrend.
    """
    if len(prices) < period + 2:
        return None

    valid = [p for p in prices if isinstance(p, (int, float)) and math.isfinite(p) and p > 0]
    if len(valid) < period + 2:
        return None

    alpha = 2.0 / (period + 1.0)
    ema_val = float(valid[0])
    ema_prev = ema_val

    for i, px in enumerate(valid[1:], 1):
        ema_prev = ema_val
        ema_val = alpha * px + (1.0 - alpha) * ema_val

    if ema_prev <= 0 or not math.isfinite(ema_val):
        return None
    return float((ema_val - ema_prev) / ema_prev)


# ---------------------------------------------------------------------------
# Macro regime classifier
# ---------------------------------------------------------------------------

def classify_macro_regime(
    vix: Optional[float],
    drawdown_pct: Optional[float],
    bond_trend: Optional[float],
    commodity_trend: Optional[float],
) -> MacroRegime:
    """
    Classify the current macro environment into one of four regimes.

    Classification rules (evaluated in priority order):

    1. **RISK_OFF**: VIX >= 25 OR drawdown <= -5%.
       Flight-to-safety conditions dominate all other signals.

    2. **STAGFLATION**: VIX >= 20 AND commodities trending up (>0)
       AND bonds trending down (<0). Elevated vol with rising input
       costs and falling bond prices signals stagflationary pressure.

    3. **RISK_ON**: VIX < 18 AND drawdown > -2%.
       Low volatility and minimal drawdown indicate healthy conditions.

    4. **NEUTRAL**: Everything else — mixed signals, insufficient data,
       or transitional regime. Fail-closed default.

    Parameters
    ----------
    vix : Optional[float]
        Current VIX level (e.g., 15.0, 25.0). None if unavailable.
    drawdown_pct : Optional[float]
        Portfolio drawdown as negative fraction (e.g., -0.06). None if unavailable.
    bond_trend : Optional[float]
        Normalized EMA slope for bonds. Positive = rising prices (falling yields).
    commodity_trend : Optional[float]
        Normalized EMA slope for commodities. Positive = rising prices.

    Returns
    -------
    MacroRegime
        The classified macro regime.
    """
    # Normalize None values to safe defaults (fail-closed toward NEUTRAL)
    vix_val = _safe_float(vix, 20.0) if vix is not None else 20.0
    dd_val = _safe_float(drawdown_pct, 0.0) if drawdown_pct is not None else 0.0

    # Rule 1: RISK_OFF — extreme volatility or significant drawdown
    if vix_val >= 25.0 or dd_val <= -0.05:
        return MacroRegime.RISK_OFF

    # Rule 2: STAGFLATION — elevated vol + rising commodities + falling bonds
    if (
        vix_val >= 20.0
        and commodity_trend is not None
        and commodity_trend > 0.0
        and bond_trend is not None
        and bond_trend < 0.0
    ):
        return MacroRegime.STAGFLATION

    # Rule 3: RISK_ON — calm markets, healthy equity
    if vix_val < 18.0 and dd_val > -0.02:
        return MacroRegime.RISK_ON

    # Rule 4: NEUTRAL — everything else
    return MacroRegime.NEUTRAL


__all__ = [
    "MacroRegime",
    "_vix_value",
    "_portfolio_drawdown_pct",
    "_atr_pct_from_bars",
    "_ema_slope",
    "classify_macro_regime",
]

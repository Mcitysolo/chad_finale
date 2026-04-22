"""
chad/analytics/threshold_adapter.py

Phase-8 Session 5 (S4): adaptive entry thresholds by market regime.

Utility module — strategies *opt in* by calling adjust(base, regime) or
adjust_rsi(oversold, overbought, regime) instead of using hardcoded
values. Nothing in Session 5 refactors existing strategies; that is
done incrementally as each strategy is touched.

Rationale
---------
A z-score cutoff that is correct in a calm market is too loose in a
volatile one (and too tight in a range). Scaling the cutoff by a
regime-aware multiplier is the simplest durable way to make entry
thresholds self-adaptive without per-strategy bespoke logic.

Regime multipliers (product default):

    trending_bull / trending_bear : 1.0   (no change — trend follows itself)
    ranging                       : 0.8   (easier to enter — range trades
                                           want more entries, not fewer)
    volatile                      : 1.5   (harder to enter — false signals
                                           dominate in chop)
    unknown                       : 1.2   (slightly harder — be conservative
                                           when the classifier can't decide)
    adverse                       : 2.0   (very hard to enter; almost off)

Callers may override the map by passing a custom dict into the helpers.

Safety
------
  * Unknown regime labels fall back to the 'unknown' multiplier (1.2).
  * adjust_rsi clamps to [0, 100] so no strategy ever sees an invalid
    RSI band.
  * All helpers return defaults on non-numeric input rather than raising.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple


# Default multipliers — consumers that want operator-controlled values
# should pass a dict loaded from config instead of mutating this constant.
REGIME_MULTIPLIERS: Mapping[str, float] = {
    "trending_bull": 1.0,
    "trending_bear": 1.0,
    "trending": 1.0,
    "risk_on": 1.0,
    "risk_off": 1.0,
    "ranging": 0.8,
    "volatile": 1.5,
    "choppy": 1.5,
    "unknown": 1.2,
    "adverse": 2.0,
}

DEFAULT_MULTIPLIER: float = 1.2  # matches the 'unknown' bucket


def _resolve_multiplier(
    regime_state: str,
    multipliers: Optional[Mapping[str, float]] = None,
) -> float:
    m = multipliers if multipliers is not None else REGIME_MULTIPLIERS
    key = str(regime_state or "").strip().lower()
    if key in m:
        try:
            return float(m[key])
        except (TypeError, ValueError):
            return DEFAULT_MULTIPLIER
    # Fallback bucket: prefer a configured 'unknown', else hard default.
    fallback = m.get("unknown")
    try:
        return float(fallback) if fallback is not None else DEFAULT_MULTIPLIER
    except (TypeError, ValueError):
        return DEFAULT_MULTIPLIER


def adjust(
    base_threshold: float,
    regime_state: str,
    multipliers: Optional[Mapping[str, float]] = None,
) -> float:
    """Scale a base threshold by the regime-specific multiplier.

    Higher return value = harder to trigger. Strategies call this
    in place of using a hardcoded z-score/indicator cutoff.
    """
    try:
        base = float(base_threshold)
    except (TypeError, ValueError):
        return 0.0
    return base * _resolve_multiplier(regime_state, multipliers)


def adjust_rsi(
    base_oversold: float,
    base_overbought: float,
    regime_state: str,
    multipliers: Optional[Mapping[str, float]] = None,
) -> Tuple[float, float]:
    """Widen or tighten RSI bands by regime.

    Volatile regimes push the bands OUTWARD (lower oversold, higher
    overbought) so random spikes don't trigger entries. Ranging regimes
    pull the bands INWARD so more setups qualify for mean-reversion
    trades. Output is clamped to [0, 100].
    """
    try:
        low = float(base_oversold)
        high = float(base_overbought)
    except (TypeError, ValueError):
        return 0.0, 100.0
    if high < low:
        low, high = high, low

    mult = _resolve_multiplier(regime_state, multipliers)
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0
    widened_half = half_range * mult

    new_low = max(0.0, center - widened_half)
    new_high = min(100.0, center + widened_half)
    if new_low > new_high:
        new_low, new_high = new_high, new_low
    return new_low, new_high


__all__ = [
    "REGIME_MULTIPLIERS",
    "DEFAULT_MULTIPLIER",
    "adjust",
    "adjust_rsi",
]

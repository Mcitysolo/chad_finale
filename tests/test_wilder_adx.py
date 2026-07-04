"""Tests for the real Wilder ADX in market_metrics_publisher.compute_adx_proxy.

These assert the fix for the regime-classifier bug: the old body measured
*volatility* (``atr_pct * 1600``) and therefore read individual stocks
(2-7% ATR) as "strongly trending" even in a flat market, which drove the
classifier to a false ``trending_bear``. The replacement is a standard
Wilder Average Directional Index which measures *directional strength*:

    * flat / choppy markets  -> LOW  ADX (< 20)
    * strong sustained trend -> HIGH ADX (> 25), either direction
    * always bounded to roughly [0, 100]
    * < 2*period+1 bars      -> 0.0 (graceful degrade)

No network, no broker calls, no files — pure in-memory OHLC fixtures.
The formula is the standard Wilder definition (no external library API),
so no Context7 documentation lookup was required.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from chad.analytics.market_metrics_publisher import (
    DEFAULT_ADX_PERIOD,
    compute_adx_proxy,
)


# ---------------------------------------------------------------------------
# Fixtures (deterministic OHLC bar builders)
# ---------------------------------------------------------------------------


def _bar(close: float, half_range: float = 0.5) -> Dict[str, float]:
    """One OHLC bar centred on ``close`` with a symmetric high/low band."""
    return {
        "high": close + half_range,
        "low": close - half_range,
        "close": close,
        "open": close,
    }


def _bars_from_closes(closes: List[float], half_range: float = 0.2) -> List[dict]:
    return [_bar(c, half_range) for c in closes]


def _uptrend_bars(n: int, start: float = 100.0, step: float = 1.0) -> List[dict]:
    """Strictly monotonic uptrend: every high/low steps up by ``step``."""
    return [_bar(start + k * step) for k in range(n)]


def _downtrend_bars(n: int, start: float = 200.0, step: float = 1.0) -> List[dict]:
    """Strictly monotonic downtrend: every high/low steps down by ``step``."""
    return [_bar(start - k * step) for k in range(n)]


def _flat_choppy_bars(n: int) -> List[dict]:
    """Prices oscillating in a tight [99.5, 100.7] band with no net drift.

    Frequent reversals mean +DM and -DM stay balanced, so DX (and hence
    the ADX) collapses toward zero — a calm market must NOT read as
    "trending".
    """
    pattern = [100.0, 100.7, 100.2, 99.5, 100.1, 99.6, 100.4, 99.9, 100.3, 99.7]
    closes = [pattern[k % len(pattern)] for k in range(n)]
    return _bars_from_closes(closes)


# ---------------------------------------------------------------------------
# (a) flat/choppy market => LOW adx (the core fix)
# ---------------------------------------------------------------------------


def test_flat_market_reads_low_adx() -> None:
    adx = compute_adx_proxy(_flat_choppy_bars(70))
    assert adx < 20.0, f"flat market should read low ADX, got {adx}"
    assert adx >= 0.0


# ---------------------------------------------------------------------------
# (b) strong uptrend => HIGH adx
# ---------------------------------------------------------------------------


def test_strong_uptrend_reads_high_adx() -> None:
    adx = compute_adx_proxy(_uptrend_bars(40))
    assert adx > 25.0, f"strong uptrend should read high ADX, got {adx}"


# ---------------------------------------------------------------------------
# (c) strong downtrend => HIGH adx (direction-agnostic)
# ---------------------------------------------------------------------------


def test_strong_downtrend_reads_high_adx() -> None:
    adx = compute_adx_proxy(_downtrend_bars(40))
    assert adx > 25.0, f"strong downtrend should read high ADX, got {adx}"


# ---------------------------------------------------------------------------
# (d) insufficient bars => 0.0 (graceful degrade contract)
# ---------------------------------------------------------------------------


def test_insufficient_bars_returns_zero() -> None:
    period = DEFAULT_ADX_PERIOD
    minimum = 2 * period + 1
    # empty, singleton, and one short of the minimum all degrade to 0.0.
    assert compute_adx_proxy([]) == 0.0
    assert compute_adx_proxy(_uptrend_bars(1)) == 0.0
    assert compute_adx_proxy(_uptrend_bars(minimum - 1)) == 0.0


def test_minimum_bar_boundary() -> None:
    period = DEFAULT_ADX_PERIOD
    minimum = 2 * period + 1  # 29 for period=14
    # One short of the minimum => 0.0; exactly the minimum => a real value.
    assert compute_adx_proxy(_uptrend_bars(minimum - 1)) == 0.0
    assert compute_adx_proxy(_uptrend_bars(minimum)) > 0.0


# ---------------------------------------------------------------------------
# (e) bounded to [0, 100] for every fixture above
# ---------------------------------------------------------------------------


def test_adx_is_bounded_zero_to_hundred() -> None:
    fixtures = [
        _flat_choppy_bars(70),
        _uptrend_bars(40),
        _downtrend_bars(40),
        _bars_from_closes([100.0, 100.5, 101.3, 100.9, 102.0, 101.4] * 12),
    ]
    for bars in fixtures:
        adx = compute_adx_proxy(bars)
        assert 0.0 <= adx <= 100.0, f"ADX out of bounds: {adx}"


def test_monotonic_trend_saturates_at_hundred() -> None:
    # With no counter-directional movement, -DM is always 0, DX is always
    # 100, and the smoothed ADX saturates at exactly 100 (bounded, unlike
    # the old proxy which returned 114 for AMD).
    assert compute_adx_proxy(_uptrend_bars(40)) == pytest.approx(100.0)
    assert compute_adx_proxy(_downtrend_bars(40)) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# (f) known reference vector (hand-computed, period=2)
# ---------------------------------------------------------------------------


def test_reference_vector_period_two() -> None:
    """Hand-computed Wilder ADX for a 6-bar series with period=2.

    Bars (high, low, close):
        0: 10.0,  8.0,  9.0
        1: 11.0,  9.0, 10.0
        2: 12.0, 10.0, 11.0
        3: 11.5,  9.5, 10.0
        4: 13.0, 11.0, 12.5
        5: 14.0, 12.0, 13.5

    Directional movement / true range (i = 1..5):
        +DM = [1,   1,   0,   1.5, 1  ]
        -DM = [0,   0,   0.5, 0,   0  ]
        TR  = [2,   2,   2,   3,   2  ]

    Wilder smoothing over period=2 (first = sum of first 2, then
    prior - prior/2 + current):
        sm(+DM) = [2, 1,   2,    2    ]
        sm(-DM) = [0, 0.5, 0.25, 0.125]
        sm(TR)  = [4, 4,   5,    4.5  ]

    +DI = 100*sm(+DM)/sm(TR); -DI = 100*sm(-DM)/sm(TR);
    DX  = 100*|+DI - -DI| / (+DI + -DI):
        DX = [100, 33.3333..., 77.7778..., 88.2353...]

    ADX seed = mean(first 2 DX) = 200/3; then adx = (adx + dx)/2:
        seed        = 66.66667
        after DX[2] = 72.22222
        after DX[3] = 24550/306 = 80.22876
    """
    bars = [
        {"high": 10.0, "low": 8.0, "close": 9.0},
        {"high": 11.0, "low": 9.0, "close": 10.0},
        {"high": 12.0, "low": 10.0, "close": 11.0},
        {"high": 11.5, "low": 9.5, "close": 10.0},
        {"high": 13.0, "low": 11.0, "close": 12.5},
        {"high": 14.0, "low": 12.0, "close": 13.5},
    ]
    adx = compute_adx_proxy(bars, period=2)
    assert adx == pytest.approx(80.22876, abs=1e-4)


# ---------------------------------------------------------------------------
# Robustness: malformed / non-positive bars are skipped, never raise
# ---------------------------------------------------------------------------


def test_malformed_bars_do_not_raise() -> None:
    bars: List[dict] = _uptrend_bars(40)
    bars.insert(5, {"high": 0.0, "low": 0.0, "close": 0.0})  # non-positive -> skipped
    bars.insert(9, "not-a-dict")  # type: ignore[arg-type]  # skipped
    adx = compute_adx_proxy(bars)
    assert 0.0 <= adx <= 100.0

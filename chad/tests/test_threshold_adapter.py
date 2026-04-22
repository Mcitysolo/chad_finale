"""Tests for Phase-8 Session 5 threshold_adapter (S4)."""

from __future__ import annotations

import pytest

from chad.analytics.threshold_adapter import (
    DEFAULT_MULTIPLIER,
    REGIME_MULTIPLIERS,
    adjust,
    adjust_rsi,
)


def test_volatile_regime_widens_threshold():
    base = 2.0
    adjusted = adjust(base, "volatile")
    assert adjusted > base
    assert adjusted == pytest.approx(base * REGIME_MULTIPLIERS["volatile"])


def test_ranging_regime_tightens_threshold():
    base = 2.0
    adjusted = adjust(base, "ranging")
    assert adjusted < base
    assert adjusted == pytest.approx(base * REGIME_MULTIPLIERS["ranging"])


def test_trending_bull_is_neutral_to_threshold():
    assert adjust(2.5, "trending_bull") == pytest.approx(2.5)


def test_adverse_regime_heavily_tightens_entries():
    # adverse=2.0× makes entries much harder.
    assert adjust(1.5, "adverse") == pytest.approx(3.0)


def test_unknown_regime_slight_tightening():
    assert adjust(1.0, "unknown") == pytest.approx(REGIME_MULTIPLIERS["unknown"])


def test_unknown_regime_label_falls_back():
    """A label we don't recognize falls to the 'unknown' bucket (1.2)."""
    assert adjust(1.0, "completely_made_up") == pytest.approx(REGIME_MULTIPLIERS["unknown"])


def test_adjust_non_numeric_base_returns_zero():
    assert adjust("nope", "volatile") == pytest.approx(0.0)


def test_adjust_case_insensitive():
    assert adjust(1.0, "VOLATILE") == pytest.approx(REGIME_MULTIPLIERS["volatile"])


# ---------------------------------------------------------------------------
# RSI band adjustment
# ---------------------------------------------------------------------------


def test_rsi_bands_widen_in_volatile():
    low, high = adjust_rsi(30.0, 70.0, "volatile")
    assert low < 30.0
    assert high > 70.0


def test_rsi_bands_tighten_in_ranging():
    low, high = adjust_rsi(30.0, 70.0, "ranging")
    assert low > 30.0
    assert high < 70.0


def test_rsi_bands_clamped_to_0_100():
    # A very-aggressive multiplier shouldn't push bands outside [0, 100].
    low, high = adjust_rsi(20.0, 80.0, "adverse")
    assert low >= 0.0
    assert high <= 100.0


def test_rsi_bands_swapped_inputs_normalized():
    low, high = adjust_rsi(70.0, 30.0, "trending_bull")
    assert low < high
    assert low == pytest.approx(30.0)
    assert high == pytest.approx(70.0)


def test_rsi_bands_non_numeric_returns_full_range():
    low, high = adjust_rsi("x", "y", "volatile")
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(100.0)


def test_custom_multipliers_override_defaults():
    custom = {"volatile": 5.0, "unknown": 1.0}
    assert adjust(1.0, "volatile", multipliers=custom) == pytest.approx(5.0)

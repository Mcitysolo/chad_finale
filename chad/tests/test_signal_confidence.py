"""Tests for Phase-8 Session 3 signal_confidence (S3)."""

from __future__ import annotations

import pytest

from chad.analytics.signal_confidence import (
    DEFAULT_CONFIDENCE,
    SIZING_FLOOR,
    compute_confidence,
    confidence_from_signal,
    normalize_signal_strength,
    regime_quality_from_state,
    sizing_multiplier,
)


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------


def test_compute_confidence_all_max_returns_1():
    assert compute_confidence(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_compute_confidence_clamped_to_0_1():
    # Values outside [0, 1] get clamped before multiplication.
    assert compute_confidence(5.0, 2.0, -1.0) == pytest.approx(0.0)
    assert compute_confidence(-0.5, 0.5, 0.5) == pytest.approx(0.0)


def test_compute_confidence_product_rule():
    # 0.8 × 0.5 × 1.0 = 0.4
    assert compute_confidence(0.8, 0.5, 1.0) == pytest.approx(0.4)


def test_compute_confidence_handles_garbage_input():
    assert compute_confidence("nope", 1.0, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# regime_quality_from_state
# ---------------------------------------------------------------------------


def test_regime_quality_known_regime():
    assert regime_quality_from_state("trending_bull") == pytest.approx(1.0)
    assert regime_quality_from_state("ranging") == pytest.approx(0.7)
    assert regime_quality_from_state("adverse") == pytest.approx(0.2)


def test_regime_quality_unknown_defaults_to_0_5():
    assert regime_quality_from_state("garbage_label") == pytest.approx(0.5)
    assert regime_quality_from_state(None) == pytest.approx(0.5)


def test_regime_quality_case_insensitive():
    assert regime_quality_from_state("TRENDING_BULL") == pytest.approx(1.0)
    assert regime_quality_from_state(" Ranging ") == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# normalize_signal_strength
# ---------------------------------------------------------------------------


def test_normalize_signal_strength_tanh_center_returns_half():
    # tanh(0) = 0 -> (0 + 1) / 2 = 0.5
    assert normalize_signal_strength(0.0, method="tanh") == pytest.approx(0.5)


def test_normalize_signal_strength_tanh_bounds_approach_0_and_1():
    # Large positive -> ~1, large negative -> ~0
    assert normalize_signal_strength(5.0, method="tanh") > 0.99
    assert normalize_signal_strength(-5.0, method="tanh") < 0.01


def test_normalize_signal_strength_clip_method():
    assert normalize_signal_strength(0.75, method="clip") == pytest.approx(0.75)
    assert normalize_signal_strength(1.5, method="clip") == pytest.approx(1.0)
    assert normalize_signal_strength(-0.2, method="clip") == pytest.approx(0.0)


def test_normalize_signal_strength_nan_and_inf_treated_as_zero():
    assert normalize_signal_strength(float("nan")) == pytest.approx(0.0)
    assert normalize_signal_strength(float("inf")) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sizing_multiplier (S3 — floor)
# ---------------------------------------------------------------------------


def test_confidence_sizing_multiplier_floor():
    """Low confidence never produces zero-size orders — SIZING_FLOOR kicks in."""
    assert sizing_multiplier(0.0) == pytest.approx(SIZING_FLOOR)
    assert sizing_multiplier(0.05) == pytest.approx(SIZING_FLOOR)
    # High confidence passes through unchanged.
    assert sizing_multiplier(0.9) == pytest.approx(0.9)
    assert sizing_multiplier(1.0) == pytest.approx(1.0)


def test_sizing_multiplier_custom_floor():
    assert sizing_multiplier(0.1, floor=0.25) == pytest.approx(0.25)
    assert sizing_multiplier(0.5, floor=0.25) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# confidence_from_signal (end-to-end helper)
# ---------------------------------------------------------------------------


def test_confidence_from_signal_none_strength_returns_default():
    assert confidence_from_signal(None, regime_state="trending_bull") == pytest.approx(
        DEFAULT_CONFIDENCE
    )


def test_confidence_from_signal_combines_all_three_factors():
    # strength=inf-ish -> ~1 after tanh, regime trending_bull -> 1.0, liq 1.0
    c = confidence_from_signal(5.0, regime_state="trending_bull", liquidity_quality=1.0)
    assert c > 0.99
    # Adverse regime knocks it down to 0.2 × ~1 × 1 = 0.2
    c_adv = confidence_from_signal(5.0, regime_state="adverse", liquidity_quality=1.0)
    assert c_adv == pytest.approx(0.2, abs=0.01)

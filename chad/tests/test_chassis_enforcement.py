#!/usr/bin/env python3
"""Tests for 50/30/20 chassis enforcement in dynamic_risk_allocator."""

import os
from unittest.mock import patch

import pytest

from chad.risk.dynamic_risk_allocator import (
    enforce_chassis,
    ALPHA_STRATEGIES,
    BETA_STRATEGIES,
    ADAPTIVE_STRATEGIES,
    ALPHA_TARGET,
    BETA_TARGET,
    ADAPTIVE_TARGET,
    CHASSIS_TOLERANCE,
)


def _sleeve_totals(w):
    alpha = sum(w.get(s, 0.0) for s in ALPHA_STRATEGIES)
    beta = sum(w.get(s, 0.0) for s in BETA_STRATEGIES)
    adaptive = sum(w.get(s, 0.0) for s in ADAPTIVE_STRATEGIES)
    return alpha, beta, adaptive


# ── Drifted weights (beta pushed to 10%) ──────────────────────────────

DRIFTED = {
    "alpha": 0.30, "alpha_futures": 0.10, "alpha_options": 0.08,
    "gamma": 0.08, "gamma_futures": 0.06, "gamma_reversion": 0.05,
    "beta": 0.10,
    "omega": 0.07, "omega_macro": 0.05, "omega_vol": 0.06,
    "delta": 0.04, "crypto": 0.01,
}

# ── Weights within tolerance ──────────────────────────────────────────

NEAR_TARGET = {
    "alpha": 0.22, "alpha_futures": 0.08, "alpha_options": 0.06,
    "gamma": 0.06, "gamma_futures": 0.04, "gamma_reversion": 0.03,
    "beta": 0.28,
    "omega": 0.07, "omega_macro": 0.05, "omega_vol": 0.04,
    "delta": 0.04, "crypto": 0.03,
}


def test_beta_protected_when_pushed_below():
    """Beta at 10% should be corrected to 30%."""
    result = enforce_chassis(DRIFTED)
    _, beta, _ = _sleeve_totals(result)
    assert abs(beta - BETA_TARGET) < 0.001


def test_alpha_sleeve_near_50():
    result = enforce_chassis(DRIFTED)
    alpha, _, _ = _sleeve_totals(result)
    assert abs(alpha - ALPHA_TARGET) < 0.001


def test_adaptive_sleeve_near_20():
    result = enforce_chassis(DRIFTED)
    _, _, adaptive = _sleeve_totals(result)
    assert abs(adaptive - ADAPTIVE_TARGET) < 0.001


def test_weights_sum_to_one():
    result = enforce_chassis(DRIFTED)
    total = sum(result.values())
    assert abs(total - 1.0) < 0.001


def test_tolerance_band_no_enforcement():
    """Weights within tolerance should pass through unchanged."""
    result = enforce_chassis(NEAR_TARGET)
    # Should be normalized but not chassis-adjusted
    total_input = sum(NEAR_TARGET.values())
    for k, v in NEAR_TARGET.items():
        expected = v / total_input
        assert abs(result[k] - expected) < 0.001, f"{k}: {result[k]} != {expected}"


def test_chassis_disabled_via_env():
    """When CHAD_CHASSIS_ENFORCEMENT=0, weights pass through as-is."""
    with patch.dict(os.environ, {"CHAD_CHASSIS_ENFORCEMENT": "0"}):
        result = enforce_chassis(DRIFTED)
    assert result == DRIFTED


def test_intra_sleeve_proportions_preserved():
    """Within a sleeve, relative proportions should be maintained."""
    result = enforce_chassis(DRIFTED)
    # Alpha sleeve: alpha was 0.30, alpha_futures was 0.10 => ratio 3:1
    if result.get("alpha_futures", 0) > 0:
        ratio_before = DRIFTED["alpha"] / DRIFTED["alpha_futures"]
        ratio_after = result["alpha"] / result["alpha_futures"]
        assert abs(ratio_before - ratio_after) < 0.01


def test_zero_weight_strategy_stays_zero():
    """A strategy with zero weight should remain zero."""
    w = dict(DRIFTED)
    w["gamma_reversion"] = 0.0
    result = enforce_chassis(w)
    assert result["gamma_reversion"] == 0.0


def test_beta_pushed_high():
    """Beta at 55% should be corrected down to 30%."""
    w = {
        "alpha": 0.10, "alpha_futures": 0.02, "alpha_options": 0.02,
        "gamma": 0.02, "gamma_futures": 0.01, "gamma_reversion": 0.01,
        "beta": 0.55,
        "omega": 0.05, "omega_macro": 0.05, "omega_vol": 0.05,
        "delta": 0.05, "crypto": 0.07,
    }
    result = enforce_chassis(w)
    _, beta, _ = _sleeve_totals(result)
    assert abs(beta - BETA_TARGET) < 0.001
    assert abs(sum(result.values()) - 1.0) < 0.001

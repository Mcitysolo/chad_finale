"""Phase A Item 3 regression tests — pre-entry R:R gate."""

from __future__ import annotations

import chad.utils.risk_reward as rr_module
from chad.utils.risk_reward import (
    MIN_RR_RATIO,
    compute_rr_ratio,
    passes_rr_gate,
)


def test_compute_rr_ratio_normal_case():
    assert compute_rr_ratio(3.0, 2.0) == 1.5


def test_compute_rr_ratio_degenerate_stop_zero():
    assert compute_rr_ratio(3.0, 0.0) is None


def test_compute_rr_ratio_degenerate_target_zero():
    assert compute_rr_ratio(0.0, 2.0) is None


def test_passes_rr_gate_at_exact_threshold():
    assert passes_rr_gate(3.0, 2.0) is True


def test_passes_rr_gate_below_threshold():
    assert passes_rr_gate(2.0, 2.0) is False


def test_passes_rr_gate_degenerate_fail_open():
    assert passes_rr_gate(0.0, 2.0) is True


def test_alpha_futures_strategy_tuning_default_rr_meets_floor():
    from chad.strategies.alpha_futures import StrategyTuning

    tuning = StrategyTuning()
    assert tuning.target_atr_multiple == 3.0
    assert tuning.stop_loss_atr_multiple == 2.0
    ratio = tuning.target_atr_multiple / tuning.stop_loss_atr_multiple
    assert ratio >= MIN_RR_RATIO


def test_alpha_futures_misconfigured_low_target_blocked():
    assert passes_rr_gate(1.0, 2.0) is False


def test_alpha_alpha_params_default_rr_meets_floor():
    from chad.strategies.alpha import AlphaParams

    params = AlphaParams()
    assert params.target_atr_multiple == 3.0
    assert params.atr_trail_mult > 0
    ratio = params.target_atr_multiple / params.atr_trail_mult
    assert ratio >= MIN_RR_RATIO


def test_alpha_intraday_fixed_constants_meet_floor():
    assert (4.5 / 1.5) >= MIN_RR_RATIO


def test_risk_reward_module_fail_open_on_negative_inputs():
    assert passes_rr_gate(-1.0, 2.0) is True
    assert passes_rr_gate(2.0, -1.0) is True


def test_risk_reward_module_exports():
    assert "MIN_RR_RATIO" in rr_module.__all__
    assert "compute_rr_ratio" in rr_module.__all__
    assert "passes_rr_gate" in rr_module.__all__

"""Phase A Item 1 regression tests — stop-distance-driven sizing.
Verifies fallback contract: original alpha_futures behavior is preserved
when tier_profile / tier_max_risk_usd is absent."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from chad.strategies.alpha_futures import (
    FuturesInstrumentSpec,
    StrategyTuning,
    _compute_contract_size,
)


def test_alpha_futures_fallback_uses_spec_risk_multiple():
    """
    alpha_futures fallback: tier_max_risk_usd=None uses spec.risk_multiple legacy sizing.
    """
    spec = FuturesInstrumentSpec(
        symbol="MES",
        family="ES",
        exchange="CME",
        point_value=5.0,
        min_tick=0.25,
        risk_multiple=1.25,
        max_contracts=10,
    )
    tuning = StrategyTuning(
        stop_loss_atr_multiple=2.0,
        risk_budget_pct=0.02,
        min_risk_budget_usd=10.0,
        max_trade_notional=500_000.0,
    )

    contracts, _alloc, _budget, risk_per_contract_usd = _compute_contract_size(
        symbol="MES",
        spec=spec,
        price=5000.0,
        atr_val=10.0,
        equity=100_000.0,
        tuning=tuning,
        confidence=0.7,
        tier_max_risk_usd=None,
    )

    assert abs(risk_per_contract_usd - 62.5) < 1e-6, (
        f"legacy sizing broken: rpc={risk_per_contract_usd}"
    )
    assert contracts == 10, f"max_contracts cap broken: contracts={contracts}"


def test_alpha_futures_tier_active_uses_stop_loss_atr_multiple():
    """
    alpha_futures tier-active path: tier_max_risk_usd uses stop_loss_atr_multiple.
    """
    spec = FuturesInstrumentSpec(
        symbol="MES",
        family="ES",
        exchange="CME",
        point_value=5.0,
        min_tick=0.25,
        risk_multiple=1.25,
        max_contracts=10,
    )
    tuning = StrategyTuning(
        stop_loss_atr_multiple=2.0,
        risk_budget_pct=0.02,
        min_risk_budget_usd=10.0,
        max_trade_notional=500_000.0,
    )

    contracts, _alloc, _budget, risk_per_contract_usd = _compute_contract_size(
        symbol="MES",
        spec=spec,
        price=5000.0,
        atr_val=10.0,
        equity=100_000.0,
        tuning=tuning,
        confidence=0.7,
        tier_max_risk_usd=200.0,
    )

    assert abs(risk_per_contract_usd - 100.0) < 1e-6, (
        f"tier path sizing broken: rpc={risk_per_contract_usd}"
    )
    assert contracts == 2, f"tier contract cap broken: contracts={contracts}"


def test_alpha_intraday_insufficient_budget_returns_zero():
    """
    alpha_intraday futures branch returns zero when tier budget cannot support one MES contract.
    """
    from chad.strategies.alpha_intraday import _size_for

    result = _size_for("MES", 0.7, atr=10.0, tier_max_risk_usd=10.0)
    assert result == 0.0, f"expected 0.0 contracts, got {result}"


def test_alpha_handler_tier_profile_none_silent_fallback():
    """
    alpha fallback: tier_profile=None remains a silent fallback and emits no signal on empty data.
    """
    from chad.strategies.alpha import alpha_handler

    ctx = SimpleNamespace(
        now=datetime.now(timezone.utc),
        prices={},
        legend=None,
        bars={},
        tier_profile=None,
    )

    result = alpha_handler(ctx)
    assert result == [], f"expected empty signal list, got {result}"

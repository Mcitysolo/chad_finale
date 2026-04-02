from __future__ import annotations

import types

import pytest


def test_policy_symbol_cap_blocks_buy(monkeypatch: pytest.MonkeyPatch):
    """
    Prove the new symbol cap gate blocks BUY when _symbol_cap_check denies.
    This is a pure unit test: no ledger reads, no config reads.
    """
    import chad.policy as policy
    from chad.types import StrategyName, TradeSignal, SignalSide, AssetClass

    # Force the cap check to deny for AAPL
    monkeypatch.setattr(policy, "_symbol_cap_check", lambda symbol: (False, "TEST_DENY"), raising=True)

    # Build a minimal PolicyEngine with BETA limits enabled
    eng = policy.PolicyEngine(strategy_limits=policy.build_default_strategy_limits(), global_limits=policy.build_default_global_limits())

    sig = TradeSignal(
        strategy=StrategyName.BETA,
        symbol="AAPL",
        side=SignalSide.BUY,  # keep consistent with existing enum usage
        size=1.0,
        confidence=1.0,
        asset_class=AssetClass.EQUITY,
    )

    prices = {"AAPL": 100.0}
    out = eng.evaluate_signals([sig], current_symbol_notional={}, current_total_notional=0.0, prices=prices)

    assert len(out) == 1
    assert out[0].decision.accepted is False
    assert out[0].decision.reason.startswith("symbol_cap_block:AAPL:TEST_DENY")


def test_policy_symbol_cap_allows_sell_even_if_denied(monkeypatch: pytest.MonkeyPatch):
    """
    Prove SELL is not blocked by symbol cap gate (so exits can still happen).
    """
    import chad.policy as policy
    from chad.types import StrategyName, TradeSignal, SignalSide, AssetClass

    monkeypatch.setattr(policy, "_symbol_cap_check", lambda symbol: (False, "TEST_DENY"), raising=True)

    eng = policy.PolicyEngine(strategy_limits=policy.build_default_strategy_limits(), global_limits=policy.build_default_global_limits())

    sig = TradeSignal(
        strategy=StrategyName.BETA,
        symbol="AAPL",
        side=SignalSide.SELL,
        size=1.0,
        confidence=1.0,
        asset_class=AssetClass.EQUITY,
    )

    prices = {"AAPL": 100.0}
    out = eng.evaluate_signals([sig], current_symbol_notional={}, current_total_notional=0.0, prices=prices)

    assert len(out) == 1
    assert out[0].decision.accepted is True
    # could still be resized by other caps, but must not be rejected by symbol cap
    assert "symbol_cap_block" not in out[0].decision.reason


def test_policy_symbol_cap_fail_closed_on_eval_error(monkeypatch: pytest.MonkeyPatch):
    import chad.policy as policy
    from chad.types import TradeSignal, SignalSide, StrategyName, AssetClass
    from datetime import datetime, timezone

    # Force caps enabled + configured symbol (even if policy re-loads config)
    monkeypatch.setattr(
        policy,
        "_load_symbol_caps_config",
        lambda: {"enabled": True, "symbols": {"AAPL": {"max_trades_per_day": 1, "max_consecutive_losses": 1}}},
        raising=True,
    )

    # Force the symbol cap check to fail-closed
    monkeypatch.setattr(policy, "_symbol_cap_check", lambda symbol: (False, "EVAL_ERROR"), raising=True)

    eng = policy.PolicyEngine(
        strategy_limits=policy.build_default_strategy_limits(),
        global_limits=policy.build_default_global_limits(),
    )

    sig = TradeSignal(
        symbol="AAPL",
        side=SignalSide.BUY,
        size=1.0,
        strategy=StrategyName.ALPHA,
        confidence=0.9,
        asset_class=AssetClass.EQUITY,
        created_at=datetime.now(timezone.utc),
        meta={},
    )

    out = eng.evaluate_signals([sig], current_symbol_notional={}, current_total_notional=0.0, prices={"AAPL": 10.0})

    # If symbol-cap guard is applied to BUY entries, this must reject.
    # If the guard is not wired in evaluate_signals yet, this will pass accepted=True and we must wire it.
    assert len(out) == 1
    assert out[0].decision.accepted is False


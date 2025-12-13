#!/usr/bin/env python3
"""
chad/tests/test_core_pipeline.py

Core tests for CHAD Phase-3 decision stack:

- types + router wiring
- PolicyEngine basic behavior
- DecisionPipeline end-to-end on a simple context
"""

from __future__ import annotations

from datetime import datetime, timezone

import math

from chad.engine import StrategyEngine
from chad.policy import (
    PolicyDecision,
    PolicyEngine,
    build_default_global_limits,
    build_default_strategy_limits,
)
from chad.strategies import register_core_strategies
from chad.types import (
    AssetClass,
    LegendConsensus,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyName,
    TradeSignal,
)
from chad.utils.pipeline import DecisionPipeline
from chad.utils.signal_router import SignalRouter


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_signal_router_merges_by_symbol_and_side() -> None:
    now = _now()
    sigs = [
        TradeSignal(
            strategy=StrategyName.ALPHA,
            symbol="AAPL",
            side=SignalSide.BUY,
            size=10,
            confidence=0.8,
            asset_class=AssetClass.EQUITY,
            created_at=now,
        ),
        TradeSignal(
            strategy=StrategyName.BETA,
            symbol="AAPL",
            side=SignalSide.BUY,
            size=5,
            confidence=0.6,
            asset_class=AssetClass.EQUITY,
            created_at=now,
        ),
    ]

    router = SignalRouter()
    routed = router.route(sigs)

    assert len(routed) == 1
    r = routed[0]
    assert r.symbol == "AAPL"
    assert r.side == SignalSide.BUY
    assert math.isclose(r.net_size, 15.0)
    assert set(s.value for s in r.source_strategies) == {"alpha", "beta"}
    # weighted confidence = (10*0.8 + 5*0.6) / 15 = 0.7333...
    assert 0.73 < r.confidence < 0.74


def test_policy_engine_rejects_missing_price() -> None:
    limits = build_default_strategy_limits()
    globals_ = build_default_global_limits()
    engine = PolicyEngine(strategy_limits=limits, global_limits=globals_)

    sig = TradeSignal(
        strategy=StrategyName.ALPHA,
        symbol="XYZ",
        side=SignalSide.BUY,
        size=10,
        confidence=0.9,
        asset_class=AssetClass.EQUITY,
        created_at=_now(),
    )

    evaluated = engine.evaluate_signals(
        [sig],
        current_symbol_notional={},
        current_total_notional=0.0,
        prices={},  # no XYZ price provided
    )[0]

    decision: PolicyDecision = evaluated.decision
    assert decision.accepted is False
    assert decision.adjusted_size == 0.0
    assert decision.reason == "missing_or_invalid_price"


def test_policy_engine_respects_trade_notional_cap() -> None:
    limits = build_default_strategy_limits()
    globals_ = build_default_global_limits()

    # Shrink Alpha's max_trade_notional to force resizing
    alpha_limits = limits[StrategyName.ALPHA]
    limits[StrategyName.ALPHA] = type(alpha_limits)(
        enabled=alpha_limits.enabled,
        max_symbol_notional=alpha_limits.max_symbol_notional,
        max_total_notional=alpha_limits.max_total_notional,
        max_trade_notional=1_000.0,  # lower than intended notional
        allow_short=alpha_limits.allow_short,
    )

    engine = PolicyEngine(strategy_limits=limits, global_limits=globals_)

    sig = TradeSignal(
        strategy=StrategyName.ALPHA,
        symbol="SPY",
        side=SignalSide.BUY,
        size=10,  # 10 * 200 = 2000 > 1000 cap
        confidence=0.9,
        asset_class=AssetClass.ETF,
        created_at=_now(),
    )

    prices = {"SPY": 200.0}
    evaluated = engine.evaluate_signals(
        [sig],
        current_symbol_notional={"SPY": 0.0},
        current_total_notional=0.0,
        prices=prices,
    )[0]

    dec = evaluated.decision
    assert dec.accepted is True
    assert dec.reason == "resized_to_max_trade_notional"
    # New size should be exactly max_trade_notional / price
    assert math.isclose(dec.adjusted_size, 1000.0 / 200.0)


def test_decision_pipeline_with_alpha_and_beta() -> None:
    now = _now()

    # Market ticks (SPY cheap vs anchor, QQQ present but unused here)
    ticks = {
        "SPY": MarketTick(
            symbol="SPY",
            price=675.0,
            size=100.0,
            exchange=11,
            timestamp=now,
        ),
        "QQQ": MarketTick(
            symbol="QQQ",
            price=610.0,
            size=50.0,
            exchange=4,
            timestamp=now,
        ),
    }

    # Legend favors AAPL, SPY, MSFT
    legend = LegendConsensus(
        as_of=now,
        weights={
            "AAPL": 0.18,
            "SPY": 0.08,
            "MSFT": 0.05,
        },
    )

    portfolio = PortfolioSnapshot(
        timestamp=now,
        cash=50_000.0,
        positions={
            "SPY": Position(
                symbol="SPY",
                asset_class=AssetClass.ETF,
                quantity=0.0,
                avg_price=680.0,
            )
        },
    )

    ctx = MarketContext(
        now=now,
        ticks=ticks,
        legend=legend,
        portfolio=portfolio,
    )

    engine = StrategyEngine()
    register_core_strategies(engine)

    policy = PolicyEngine(
        strategy_limits=build_default_strategy_limits(),
        global_limits=build_default_global_limits(),
    )
    router = SignalRouter()

    pipeline = DecisionPipeline(engine=engine, policy=policy, router=router)

    prices = {"SPY": 675.0, "QQQ": 610.0, "AAPL": 190.0, "MSFT": 410.0}
    current_symbol_notional = {"SPY": 0.0, "AAPL": 0.0, "MSFT": 0.0}
    current_total_notional = 0.0

    result = pipeline.run(
        ctx=ctx,
        prices=prices,
        current_symbol_notional=current_symbol_notional,
        current_total_notional=current_total_notional,
    )

    # There should be at least one signal from Alpha (SPY) and some from Beta
    assert len(result.raw_signals) >= 2

    # All evaluated signals must have a valid PolicyDecision
    assert len(result.evaluated_signals) == len(result.raw_signals)
    assert all(ev.decision.adjusted_size >= 0.0 for ev in result.evaluated_signals)

    # Routed signals should reflect accepted, possibly resized signals
    for r in result.routed_signals:
        assert r.net_size > 0.0
        assert r.symbol in {"SPY", "AAPL", "MSFT"}

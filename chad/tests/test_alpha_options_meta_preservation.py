"""Regression tests for alpha_options OPT/BAG metadata preservation through
the execution pipeline.

Background
----------
2026-05-08 forensic audit: alpha_options SPY OPT submits were failing with
``ContractResolutionError: Options contract for SPY requires 'expiry' in
intent.meta`` after the OPTIONS_INTENT_PROXIED log. The strategy emits
expiry/long_strike/short_strike/long_right/short_right in TradeSignal.meta,
the SignalRouter preserves it, and build_execution_plan copies it into
``order.metadata['signal_meta']`` (nested). However, the IBKR adapter reads
contract fields from the TOP level of ``intent.meta`` — so without flattening
the nested ``signal_meta`` keys, the adapter never sees them.

Additionally, ``_resolve_options_spec`` hardcodes ``sec_type='OPT'`` for every
OPTIONS asset class, but alpha_options actually emits BAG (vertical spread)
intents. The intent's effective sec_type must be derived from the preserved
signal meta so the adapter takes the combo-resolution path.

These tests assert that:
1. alpha_options BAG intents flow through build_execution_plan +
   build_ibkr_intents_from_plan with expiry/long_strike/short_strike/
   long_right/short_right at the TOP level of intent.meta and sec_type=BAG.
2. Single-leg OPT intents preserve expiry/strike/right at the top level.
3. Missing required contract metadata triggers a pre-submit skip with a
   clear OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META log instead of
   ContractResolutionError from the adapter.
4. Futures contract_month resolution is not affected.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from chad.execution import execution_pipeline as ep
from chad.types import AssetClass, RoutedSignal, SignalSide, StrategyName


@pytest.fixture(autouse=True)
def _bypass_routing_gates(monkeypatch):
    """Routing gates depend on disk bar data and would reject these
    intents on price drift. The patch under test sits before those
    gates — the gate behavior is exercised by other test modules."""
    monkeypatch.setattr(ep, "run_all_gates", lambda **_kwargs: (True, ""))


def _routed_alpha_options_bag(symbol: str = "SPY", side: SignalSide = SignalSide.BUY,
                              meta_override: dict | None = None) -> RoutedSignal:
    meta = {
        "engine": "alpha_options.v1",
        "spread_id": "test-spread-001",
        "spread_type": "BULL_CALL",
        "expiry": "20260620",
        "long_strike": 470.0,
        "short_strike": 475.0,
        "long_right": "C",
        "short_right": "C",
        "dte": 43,
        "max_loss_per_contract": 250.0,
        "net_debit_estimate": 2.50,
        "contracts": 1,
        "required_asset_class": "options",
        "sec_type": "BAG",
    }
    if meta_override is not None:
        meta = {**meta, **meta_override}
    return RoutedSignal(
        symbol=symbol,
        side=side,
        net_size=1.0,
        source_strategies=(StrategyName.ALPHA_OPTIONS,),
        primary_strategy=StrategyName.ALPHA_OPTIONS.value,
        confidence=0.75,
        asset_class=AssetClass.OPTIONS,
        created_at=datetime.now(timezone.utc),
        meta=meta,
    )


def _routed_single_leg_opt(symbol: str = "SPY") -> RoutedSignal:
    return RoutedSignal(
        symbol=symbol,
        side=SignalSide.BUY,
        net_size=1.0,
        source_strategies=(StrategyName.ALPHA_OPTIONS,),
        primary_strategy=StrategyName.ALPHA_OPTIONS.value,
        confidence=0.7,
        asset_class=AssetClass.OPTIONS,
        created_at=datetime.now(timezone.utc),
        meta={
            "engine": "alpha_options.v1",
            "sec_type": "OPT",
            "expiry": "20260620",
            "strike": 470.0,
            "right": "C",
            "required_asset_class": "options",
        },
    )


def _build_intents(routed):
    plan = ep.build_execution_plan([routed], prices={routed.symbol: 470.0})
    assert plan.orders, "expected planner to accept the routed signal"
    return ep.build_ibkr_intents_from_plan(plan)


def test_alpha_options_bag_intent_preserves_contract_meta_and_sec_type():
    """A BAG vertical-spread intent reaches the IBKR adapter with expiry,
    long_strike, short_strike, long_right, short_right at the TOP level
    of intent.meta and sec_type='BAG'."""
    routed = _routed_alpha_options_bag()
    intents = _build_intents(routed)
    assert len(intents) == 1, f"expected exactly 1 intent, got {len(intents)}"
    intent = intents[0]

    assert intent.symbol == "SPY"
    assert intent.sec_type == "BAG", (
        f"effective sec_type must be BAG when survivor signal carries "
        f"sec_type=BAG; got {intent.sec_type!r}"
    )

    assert intent.meta.get("expiry") == "20260620", (
        "expiry must be at the top level of intent.meta — the adapter "
        "looks for it there, not under signal_meta"
    )
    assert float(intent.meta.get("long_strike")) == 470.0
    assert float(intent.meta.get("short_strike")) == 475.0
    assert intent.meta.get("long_right") == "C"
    assert intent.meta.get("short_right") == "C"


def test_alpha_options_bag_intent_resolves_through_ibkr_adapter():
    """End-to-end: the meta-preserving intent must be acceptable to
    _ContractResolver._resolve_combo without ContractResolutionError."""
    from chad.execution.ibkr_adapter import (
        IbkrConfig,
        NormalizedIntent,
        _ContractResolver,
    )

    routed = _routed_alpha_options_bag()
    intent = _build_intents(routed)[0]

    config = IbkrConfig(dry_run=True)
    resolver = _ContractResolver(config, now_fn=lambda: datetime.now(timezone.utc))
    normalized = NormalizedIntent(
        strategy=intent.strategy,
        symbol=intent.symbol,
        sec_type=intent.sec_type,
        exchange=intent.exchange or "SMART",
        currency=intent.currency or "USD",
        side=intent.side,
        order_type=intent.order_type,
        quantity=intent.quantity,
        notional_estimate=intent.notional_estimate,
        asset_class="options",
        source_strategies=(intent.strategy,),
        created_at=datetime.now(timezone.utc),
        meta=dict(intent.meta),
    )
    resolved = resolver.resolve(None, normalized)
    assert resolved.summary["sec_type"] == "BAG"


def test_single_leg_opt_intent_preserves_expiry_strike_right():
    """A single-leg OPT intent (sec_type=OPT in meta) must reach the
    adapter with expiry, strike, right at the top level."""
    routed = _routed_single_leg_opt()
    intents = _build_intents(routed)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.sec_type == "OPT"
    assert intent.meta.get("expiry") == "20260620"
    assert float(intent.meta.get("strike")) == 470.0
    assert intent.meta.get("right") == "C"


def test_options_intent_missing_expiry_is_skipped_pre_submit(caplog):
    """A routed OPTIONS signal with no expiry must be skipped before
    the adapter sees it, with a clear OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META
    log — not allowed to surface as ContractResolutionError downstream."""
    routed = _routed_alpha_options_bag(meta_override={"expiry": None})
    with caplog.at_level(logging.WARNING, logger="chad.live_loop"):
        intents = _build_intents(routed)
    assert intents == [], (
        "OPT/BAG intents missing required contract meta must NOT reach "
        "the IBKR adapter — they should be skipped pre-submit"
    )
    assert any(
        "OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META" in r.getMessage()
        for r in caplog.records
    ), "expected OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META audit log"


def test_options_intent_missing_strikes_is_skipped_pre_submit(caplog):
    """BAG intent missing long_strike/short_strike must be skipped."""
    routed = _routed_alpha_options_bag(
        meta_override={"long_strike": None, "short_strike": None}
    )
    with caplog.at_level(logging.WARNING, logger="chad.live_loop"):
        intents = _build_intents(routed)
    assert intents == []
    assert any(
        "OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META" in r.getMessage()
        and "long_strike" in r.getMessage()
        for r in caplog.records
    )


def test_futures_contract_month_safety_unaffected(monkeypatch):
    """The OPTIONS preservation patch must not regress the FUT
    contract_month resolution path."""
    monkeypatch.setattr(ep, "_resolve_contract_month", lambda sym: "202606")

    routed = RoutedSignal(
        symbol="MES",
        side=SignalSide.BUY,
        net_size=1.0,
        source_strategies=(StrategyName.ALPHA_FUTURES,),
        primary_strategy=StrategyName.ALPHA_FUTURES.value,
        confidence=0.6,
        asset_class=AssetClass.FUTURES,
        created_at=datetime.now(timezone.utc),
        meta={},
    )
    plan = ep.build_execution_plan([routed], prices={"MES": 5800.0})
    intents = ep.build_ibkr_intents_from_plan(plan)
    assert len(intents) == 1
    intent = intents[0]
    assert intent.sec_type == "FUT"
    assert intent.meta.get("contract_month") == "202606"


def test_alpha_options_bag_no_signal_meta_falls_back_safely(caplog):
    """When the planner has not propagated signal_meta (older path),
    a bare OPTIONS order with no contract fields should be skipped
    cleanly rather than reaching the adapter as an unresolvable OPT."""
    # Construct a RoutedSignal whose meta has *no* contract fields at all.
    routed = RoutedSignal(
        symbol="SPY",
        side=SignalSide.BUY,
        net_size=1.0,
        source_strategies=(StrategyName.ALPHA_OPTIONS,),
        primary_strategy=StrategyName.ALPHA_OPTIONS.value,
        confidence=0.5,
        asset_class=AssetClass.OPTIONS,
        created_at=datetime.now(timezone.utc),
        meta={},
    )
    with caplog.at_level(logging.WARNING, logger="chad.live_loop"):
        intents = _build_intents(routed)
    assert intents == []
    assert any(
        "OPTIONS_INTENT_SKIPPED_MISSING_CONTRACT_META" in r.getMessage()
        for r in caplog.records
    )

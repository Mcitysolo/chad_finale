"""Regression tests for omega_macro execution lane.

Covers the 2026-05-09 forensic-audit fix where ZN, ZB, and M6E (the
omega_macro futures universe) were missing from the
``_futures_spec_registry()`` cache in ``chad.execution.execution_pipeline``.
The intent builder swallowed the resulting ``ValueError`` silently, so
omega_macro generated signals every cycle but never produced an IBKR
intent — the strategy showed `signals_after_regime_gate=0` with empty
``blocked_reasons`` because the drop happened before the regime gate.

The fix:
1. Adds ZN, ZB, M6E to ``_futures_spec_registry()``.
2. Replaces the silent ``except ValueError: continue`` in
   ``build_ibkr_intents_from_plan`` with a logged WARNING so future
   silent drops are observable.

These tests lock both behaviors in.
"""

from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from chad.execution import execution_pipeline as ep
from chad.execution.execution_pipeline import (
    ExecutionPlan,
    PlannedOrder,
    _futures_spec_registry,
    _resolve_futures_spec,
    build_ibkr_intents_from_plan,
)
from chad.strategies.alpha_futures import ALPHA_FUTURES_UNIVERSE
from chad.strategies.omega_macro import OMEGA_MACRO_SPECS
from chad.types import AssetClass, SignalSide, StrategyName


@pytest.fixture(autouse=True)
def _isolate_from_disk_bars(monkeypatch):
    """Stub out the latest-bar reader so post-spec routing gates do not
    compare the test's synthetic price against real on-disk bar prices.

    With no bar, ``too_late_to_chase_gate`` falls back to ``order.price``
    as the current price, drift is zero, and the gate passes — letting
    these tests focus on registry resolution rather than market-state
    drift, which is what we are asserting.
    """
    monkeypatch.setattr(ep, "_load_latest_bar_for_symbol", lambda _symbol: None)
    monkeypatch.setattr(
        ep, "_load_latest_bar_for_symbol_with_timeframe", lambda _s, _tf: None
    )
    monkeypatch.setattr(ep, "_load_latest_bar_ts_for_symbol", lambda _s: "")


_OMEGA_MACRO_EXPECTED_EXCHANGE = {
    "ZN": "CBOT",
    "ZB": "CBOT",
    "M6E": "CME",
}


def _make_plan(symbol: str, *, asset_class: AssetClass = AssetClass.FUTURES,
               primary_strategy: StrategyName = StrategyName.OMEGA_MACRO) -> ExecutionPlan:
    order = PlannedOrder(
        symbol=symbol,
        side=SignalSide.BUY,
        size=1.0,
        asset_class=asset_class,
        price=110.0,
        notional=110.0,
        primary_strategy=primary_strategy,
        contributing_strategies=(primary_strategy,),
        metadata={"netted": True, "raw_asset_class": asset_class.value},
    )
    return ExecutionPlan(orders=[order])


# ---------------------------------------------------------------------------
# Test 1: omega_macro symbols build into FUT intents with correct exchanges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol,expected_exchange",
    sorted(_OMEGA_MACRO_EXPECTED_EXCHANGE.items()),
)
def test_execution_pipeline_omega_macro_intent(symbol, expected_exchange):
    plan = _make_plan(symbol)
    intents = build_ibkr_intents_from_plan(plan)

    assert len(intents) == 1, (
        f"expected exactly one intent for {symbol}; got {len(intents)} "
        f"(registry must include {symbol} as a FUT spec)"
    )
    intent = intents[0]
    assert intent.sec_type == "FUT", (
        f"omega_macro {symbol} must route as FUT, got {intent.sec_type!r}"
    )
    assert intent.exchange == expected_exchange, (
        f"omega_macro {symbol} must route to {expected_exchange}, "
        f"got {intent.exchange!r}"
    )
    assert intent.symbol == symbol


# ---------------------------------------------------------------------------
# Test 2: registry covers every symbol any futures strategy actually trades
# ---------------------------------------------------------------------------

def test_futures_spec_registry_covers_strategy_universes():
    universe = set(OMEGA_MACRO_SPECS.keys()) | set(ALPHA_FUTURES_UNIVERSE)
    assert universe, "test universe must be non-empty"

    missing = []
    for symbol in sorted(universe):
        try:
            spec = _resolve_futures_spec(symbol)
        except ValueError:
            missing.append(symbol)
            continue
        assert spec.sec_type == "FUT", (
            f"{symbol} resolved but sec_type={spec.sec_type!r} (expected 'FUT')"
        )

    assert not missing, (
        "futures_spec_registry is missing strategy-universe symbols: "
        f"{sorted(missing)}. Either add them to _futures_spec_registry() or "
        "remove them from the strategy universe."
    )


# ---------------------------------------------------------------------------
# Test 3: unknown FUTURES symbol logs WARNING and produces no intent
# ---------------------------------------------------------------------------

def test_intent_builder_logs_unknown_symbol(caplog):
    bogus_symbol = "ZZZ_NOT_A_REAL_FUTURE"
    # Defensive: if a future cleanup accidentally registers this, fail loudly
    # rather than producing a misleading green test.
    assert bogus_symbol not in _futures_spec_registry(), (
        f"{bogus_symbol} must remain unregistered for this test to be valid"
    )

    plan = _make_plan(bogus_symbol)
    with caplog.at_level(logging.WARNING, logger="chad.execution.execution_pipeline"):
        intents = build_ibkr_intents_from_plan(plan)

    assert intents == [], (
        f"unknown FUTURES symbol must not produce an intent; got {intents!r}"
    )

    matching = [
        rec for rec in caplog.records
        if "INTENT_DROPPED_NO_SPEC" in rec.getMessage()
    ]
    assert matching, (
        "expected a WARNING containing INTENT_DROPPED_NO_SPEC when an "
        "unknown FUTURES symbol is dropped; got: "
        f"{[rec.getMessage() for rec in caplog.records]}"
    )
    msg = matching[0].getMessage()
    assert bogus_symbol in msg, f"log line must include the dropped symbol: {msg!r}"


# ---------------------------------------------------------------------------
# Test 4: regime activation matrix gates target strategies as expected
# ---------------------------------------------------------------------------

def test_regime_matrix_blocks_delta_pairs_gamma_in_trending_bull():
    from chad.portfolio.regime_activation import is_strategy_allowed

    assert is_strategy_allowed("delta_pairs", "trending_bull") is False, (
        "delta_pairs must remain blocked under trending_bull (it is a "
        "ranging-regime mean-reversion strategy)"
    )
    assert is_strategy_allowed("gamma", "trending_bull") is False, (
        "gamma must remain blocked under trending_bull (it is a "
        "ranging/volatile-regime strategy)"
    )
    assert is_strategy_allowed("omega_macro", "trending_bull") is True, (
        "omega_macro must remain enabled under trending_bull — if this "
        "asserts False, the matrix has regressed"
    )


# ---------------------------------------------------------------------------
# Test 5: end-to-end smoke for omega_macro futures intent creation
# ---------------------------------------------------------------------------

def test_execution_pipeline_omega_macro_end_to_end():
    """Run the full plan→intent path for every omega_macro symbol.

    No IBKR connection, no order placement — pure planning + intent
    construction. Each input symbol must produce exactly one IBKR FUT
    intent with the strategy attribution preserved.
    """
    for symbol in sorted(OMEGA_MACRO_SPECS.keys()):
        plan = _make_plan(symbol)
        intents = build_ibkr_intents_from_plan(plan)
        assert len(intents) == 1, (
            f"omega_macro {symbol}: expected 1 intent, got {len(intents)}"
        )
        intent = intents[0]
        assert intent.sec_type == "FUT"
        assert intent.symbol == symbol
        assert intent.currency == "USD"
        # The intent's strategy name comes from PlannedOrder.primary_strategy.value.
        assert str(intent.strategy) == StrategyName.OMEGA_MACRO.value
        assert intent.quantity > 0


# ---------------------------------------------------------------------------
# Test 6: gamma_futures M2K builds into a FUT intent on CME
# ---------------------------------------------------------------------------
#
# Locks in the 2026-05-09 forensic-audit fix that added M2K (Micro E-mini
# Russell 2000) to ``_futures_spec_registry()``. Same gap class as the
# omega_macro fix above (commits 53fdb98 / 9e784ee): M2K is in the
# gamma_futures universe but was missing from the routing registry, so
# every M2K intent dropped silently with INTENT_DROPPED_NO_SPEC.

def test_execution_pipeline_m2k_intent():
    plan = _make_plan("M2K", primary_strategy=StrategyName.GAMMA_FUTURES)
    intents = build_ibkr_intents_from_plan(plan)

    assert len(intents) == 1, (
        f"expected exactly one intent for M2K; got {len(intents)} "
        "(registry must include M2K as a FUT spec)"
    )
    intent = intents[0]
    assert intent.sec_type == "FUT", (
        f"gamma_futures M2K must route as FUT, got {intent.sec_type!r}"
    )
    assert intent.exchange == "CME", (
        f"gamma_futures M2K must route to CME, got {intent.exchange!r}"
    )
    assert intent.symbol == "M2K"


def test_futures_spec_registry_covers_m2k():
    """Regression lock that M2K resolves through ``_resolve_futures_spec``.

    Without M2K in the registry, ``build_ibkr_intents_from_plan`` drops the
    intent with INTENT_DROPPED_NO_SPEC and gamma_futures never reaches the
    broker. This test fails fast if the registry entry is removed.
    """
    spec = _resolve_futures_spec("M2K")
    assert spec.sec_type == "FUT"
    assert spec.exchange == "CME"
    assert spec.currency == "USD"

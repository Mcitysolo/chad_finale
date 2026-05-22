#!/usr/bin/env python3
"""
chad/tests/test_ibkr_adapter_tick_snap.py

Focused tests for the LMT-price tick-snap path added to chad.execution.
ibkr_adapter to fix Box-9 / GAP-036 blocker: IBKR rejected MCL SELL orders
with Error 110 ("price does not conform to the minimum price variation for
this contract") because the strategy-computed lmtPrice 102.41748 was not
snapped to MCL minTick=0.01 before being placed.

The tests pin:
  * the exact observed reproducer (MCL SELL @ 102.41748 -> 102.42)
  * the side-safe policy (BUY=floor, SELL=ceil)
  * pass-through for unknown FUT symbols and for non-FUT sec_types
  * the order-factory path actually consults the snapper end-to-end
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from chad.execution.ibkr_adapter import (
    IbkrConfig,
    NormalizedIntent,
    _FUT_MIN_TICK_BY_SYMBOL,
    _OrderFactory,
    _snap_lmt_price_to_tick,
)
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Pure helper: _snap_lmt_price_to_tick
# ---------------------------------------------------------------------------


def test_mcl_sell_observed_reproducer_snaps_up_to_one_cent() -> None:
    """The exact production-observed reproducer.

    IBKR error log shows lmtPrice=102.41748 SELL on MCL@NYMEX cancelled with
    Error 110. After the patch, SELL must snap UP to the next 0.01 boundary
    (102.42), and the price the broker would receive must be a valid MCL tick.
    """
    snapped, tick = _snap_lmt_price_to_tick(
        102.41748, side="SELL", sec_type="FUT", symbol="MCL"
    )
    assert tick == Decimal("0.01")
    assert snapped == pytest.approx(102.42, abs=1e-9)
    # The whole point: the post-snap price must be on the tick lattice.
    assert (Decimal(str(snapped)) / tick) == (Decimal(str(snapped)) / tick).to_integral_value()


def test_mcl_buy_same_price_floors_to_one_cent() -> None:
    """BUY at the same offending price must round DOWN (less aggressive).

    A BUY buyer rounding UP would pay more than the strategy intended;
    policy is conservative.
    """
    snapped, tick = _snap_lmt_price_to_tick(
        102.41748, side="BUY", sec_type="FUT", symbol="MCL"
    )
    assert tick == Decimal("0.01")
    assert snapped == pytest.approx(102.41, abs=1e-9)


def test_unsnapped_mcl_price_can_no_longer_pass_unchanged() -> None:
    """Pins the bug fix: the previously-rejected price must NOT be
    forwarded verbatim once it has traversed the snapper for either side."""
    for side in ("BUY", "SELL"):
        snapped, _ = _snap_lmt_price_to_tick(
            102.41748, side=side, sec_type="FUT", symbol="MCL"
        )
        assert snapped != 102.41748, (
            f"side={side}: unsnapped price 102.41748 leaked through tick snap"
        )


def test_already_snapped_price_is_idempotent() -> None:
    """If the strategy gives us a tick-valid price, both BUY and SELL must
    return that price unchanged (idempotent within float tolerance)."""
    for side in ("BUY", "SELL"):
        snapped, tick = _snap_lmt_price_to_tick(
            102.42, side=side, sec_type="FUT", symbol="MCL"
        )
        assert tick == Decimal("0.01")
        assert snapped == pytest.approx(102.42, abs=1e-9)


@pytest.mark.parametrize(
    "symbol,raw,side,expected",
    [
        # MES tick 0.25 — BUY floors, SELL ceils
        ("MES", 5800.13, "BUY",  5800.00),
        ("MES", 5800.13, "SELL", 5800.25),
        # MGC tick 0.10
        ("MGC", 2345.678, "BUY",  2345.60),
        ("MGC", 2345.678, "SELL", 2345.70),
        # M6E tick 0.0001 (FX micro)
        ("M6E", 1.10245, "BUY",  1.1024),
        ("M6E", 1.10245, "SELL", 1.1025),
        # MYM tick 1.0 (whole-point)
        ("MYM", 39871.4, "BUY",  39871.0),
        ("MYM", 39871.4, "SELL", 39872.0),
    ],
)
def test_known_futures_symbols_snap_side_safely(
    symbol: str, raw: float, side: str, expected: float
) -> None:
    snapped, tick = _snap_lmt_price_to_tick(
        raw, side=side, sec_type="FUT", symbol=symbol
    )
    assert tick == _FUT_MIN_TICK_BY_SYMBOL[symbol]
    assert snapped == pytest.approx(expected, abs=1e-9)


def test_zn_eighth_of_thirtysecond_tick_snaps_correctly() -> None:
    """ZN minTick is 1/64 = 0.015625; verify both sides on a value that is
    NOT already on the tick lattice."""
    raw = 110.5234375 + 0.005  # = 110.5284375 — between two ZN ticks
    buy_snap, tick = _snap_lmt_price_to_tick(
        raw, side="BUY", sec_type="FUT", symbol="ZN"
    )
    sell_snap, _ = _snap_lmt_price_to_tick(
        raw, side="SELL", sec_type="FUT", symbol="ZN"
    )
    assert tick == Decimal("0.015625")
    # Floor and ceil must straddle the input by exactly one tick.
    assert Decimal(str(sell_snap)) - Decimal(str(buy_snap)) == tick
    assert Decimal(str(buy_snap)) <= Decimal(str(raw)) <= Decimal(str(sell_snap))


def test_unknown_futures_symbol_passes_through_with_no_tick() -> None:
    """Conservative: do NOT guess a tick for symbols we have not vetted.
    The caller (and the LMT_PRICE_SNAPPED log) can detect this via tick=None
    rather than silently substituting an assumed value."""
    snapped, tick = _snap_lmt_price_to_tick(
        123.456789, side="BUY", sec_type="FUT", symbol="UNKNOWN_FUT"
    )
    assert tick is None
    assert snapped == 123.456789


def test_equity_passes_through_unchanged() -> None:
    """STK orders intentionally bypass the FUT tick map; existing equity
    submission behaviour must be preserved exactly."""
    snapped, tick = _snap_lmt_price_to_tick(
        276.157, side="BUY", sec_type="STK", symbol="IWM"
    )
    assert tick is None
    assert snapped == 276.157


def test_option_passes_through_unchanged() -> None:
    snapped, tick = _snap_lmt_price_to_tick(
        1.234567, side="SELL", sec_type="OPT", symbol="SPY"
    )
    assert tick is None
    assert snapped == 1.234567


def test_non_finite_or_non_positive_price_is_left_alone() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        snapped, tick = _snap_lmt_price_to_tick(
            bad, side="BUY", sec_type="FUT", symbol="MCL"
        )
        # tick may still be reported (MCL is in the map), but we must not
        # invent a positive price out of a non-positive / non-finite input.
        if bad != bad:  # NaN
            assert snapped != snapped
        else:
            assert snapped == bad


# ---------------------------------------------------------------------------
# Integration: _OrderFactory.build() actually calls the snapper
# ---------------------------------------------------------------------------


class _StubOrder:
    """Minimal stand-in for ib_async.order.Order to avoid importing it
    during unit tests; mirrors the attributes _OrderFactory.build sets."""

    action = ""
    orderType = ""
    totalQuantity = 0.0
    tif = ""
    outsideRth = False
    whatIf = False
    account = ""
    lmtPrice = 0.0


def _stub_contract_classes() -> tuple[Any, ...]:
    return (object, object, object, _StubOrder, object)


def _make_intent(
    *, symbol: str, side: str, sec_type: str, limit_price: float
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy="gamma_futures",
        symbol=symbol,
        sec_type=sec_type,
        exchange="NYMEX",
        currency="USD",
        side=side,
        order_type="LMT",
        quantity=2.0,
        notional_estimate=0.0,
        asset_class="futures",
        source_strategies=("gamma_futures",),
        created_at=datetime.now(timezone.utc),
        limit_price=limit_price,
    )


def test_order_factory_build_snaps_mcl_sell_reproducer(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: feed _OrderFactory the exact production reproducer and
    assert the lmtPrice that would land on the wire is tick-valid."""
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = _make_intent(
        symbol="MCL", side="SELL", sec_type="FUT", limit_price=102.41748
    )
    prepared = factory.build(intent, what_if=True)
    # SELL ceil to MCL 0.01 -> 102.42
    assert prepared.order.lmtPrice == pytest.approx(102.42, abs=1e-9)
    # And the wire price must be on the tick lattice.
    assert Decimal(str(prepared.order.lmtPrice)) % Decimal("0.01") == Decimal("0")


def test_order_factory_build_snaps_mcl_buy_reproducer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = _make_intent(
        symbol="MCL", side="BUY", sec_type="FUT", limit_price=102.41748
    )
    prepared = factory.build(intent, what_if=True)
    # BUY floor to MCL 0.01 -> 102.41
    assert prepared.order.lmtPrice == pytest.approx(102.41, abs=1e-9)


def test_order_factory_build_passes_equity_price_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """STK orders must NOT be snapped via the FUT tick table; the lmtPrice
    set on the order is exactly what the strategy provided."""
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = _make_intent(
        symbol="IWM", side="BUY", sec_type="STK", limit_price=276.157
    )
    prepared = factory.build(intent, what_if=True)
    assert prepared.order.lmtPrice == pytest.approx(276.157, abs=1e-9)


def test_order_factory_build_passes_unknown_futures_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown FUT symbol must not be silently snapped to a guessed tick."""
    monkeypatch.setattr(
        "chad.execution.ibkr_adapter._lazy_import_contract_classes",
        _stub_contract_classes,
    )
    factory = _OrderFactory(IbkrConfig())
    intent = _make_intent(
        symbol="UNKNOWN_FUT", side="SELL", sec_type="FUT", limit_price=42.123456
    )
    prepared = factory.build(intent, what_if=True)
    assert prepared.order.lmtPrice == pytest.approx(42.123456, abs=1e-9)

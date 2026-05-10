#!/usr/bin/env python3
"""
chad/tests/test_ibkr_open_order_guard.py

Tests for the broker-side open-order guard added to IbkrAdapter.

The guard reads ib.openTrades() before each placeOrder and:
  * suppresses with status="duplicate_open_order" when a working order on the
    same broker lane (sec_type/symbol/side[/exchange/currency/contract_month/
    strike/right/limit_price]) already exists,
  * suppresses with status="suppressed_open_orders_cap" when the per-lane
    working-order count would breach IBKR's working-order cap.

Both statuses are unconfirmed and must NOT produce paper_fill evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List

import pytest

from chad.execution.ibkr_adapter import (
    IbkrAdapter,
    IbkrConfig,
    _open_trade_lane_key,
    _normalize_lmt_price,
)
from chad.types import AssetClass, RoutedSignal, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeContract:
    def __init__(
        self,
        *,
        symbol: str,
        secType: str,
        exchange: str = "",
        currency: str = "USD",
        lastTradeDateOrContractMonth: str = "",
        strike: Any = "",
        right: str = "",
        multiplier: str = "",
    ) -> None:
        self.symbol = symbol
        self.secType = secType
        self.exchange = exchange
        self.currency = currency
        self.lastTradeDateOrContractMonth = lastTradeDateOrContractMonth
        self.strike = strike
        self.right = right
        self.multiplier = multiplier
        self.primaryExchange = ""
        self.comboLegs = []


class _FakeOrder:
    def __init__(self, *, action: str, lmtPrice: Any = None, orderId: int = 0) -> None:
        self.action = action
        self.lmtPrice = lmtPrice if lmtPrice is not None else 1.7976931348623157e308
        self.orderId = orderId
        self.permId = 0
        self.totalQuantity = 1.0
        self.orderType = "MKT" if lmtPrice is None else "LMT"


class _FakeStatus:
    def __init__(self, *, status: str, remaining: float = 1.0) -> None:
        self.status = status
        self.remaining = remaining


class _FakeTrade:
    def __init__(self, contract: _FakeContract, order: _FakeOrder, status: _FakeStatus) -> None:
        self.contract = contract
        self.order = order
        self.orderStatus = status
        self.fills: List[Any] = []
        self.commissionReport: List[Any] = []


class _FakeIB:
    def __init__(self, open_trades: List[_FakeTrade] | None = None) -> None:
        self._connected = False
        self.open_trades_value = list(open_trades or [])
        self.place_order_calls: List[Any] = []

    def isConnected(self) -> bool:
        return self._connected

    def connect(self, host: str, port: int, clientId: int, timeout: float) -> None:  # noqa: N803
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def managedAccounts(self) -> List[str]:
        return ["DU000000"]

    def qualifyContracts(self, *contracts: Any) -> List[Any]:
        return list(contracts)

    def whatIfOrder(self, contract: Any, order: Any) -> Any:
        return order

    def placeOrder(self, contract: Any, order: Any) -> Any:
        self.place_order_calls.append((contract, order))
        return _FakeTrade(
            contract,
            order,
            _FakeStatus(status="Submitted", remaining=getattr(order, "totalQuantity", 1.0)),
        )

    def sleep(self, seconds: float) -> None:
        return None

    def openTrades(self) -> List[_FakeTrade]:
        return list(self.open_trades_value)


# ---------------------------------------------------------------------------
# Lane-key helpers
# ---------------------------------------------------------------------------


def test_open_trade_lane_key_skips_non_working_status() -> None:
    contract = _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD")
    order = _FakeOrder(action="BUY", orderId=1)
    status = _FakeStatus(status="Filled", remaining=0.0)
    assert _open_trade_lane_key(_FakeTrade(contract, order, status)) is None


def test_open_trade_lane_key_includes_inactive_with_remaining() -> None:
    contract = _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD")
    order = _FakeOrder(action="BUY", orderId=1)
    inactive_with_qty = _FakeStatus(status="Inactive", remaining=5.0)
    inactive_without_qty = _FakeStatus(status="Inactive", remaining=0.0)
    assert _open_trade_lane_key(_FakeTrade(contract, order, inactive_with_qty)) is not None
    assert _open_trade_lane_key(_FakeTrade(contract, order, inactive_without_qty)) is None


def test_normalize_lmt_price_handles_unset_double() -> None:
    assert _normalize_lmt_price(None) is None
    assert _normalize_lmt_price(0.0) is None
    assert _normalize_lmt_price(-1.0) is None
    assert _normalize_lmt_price(1.7976931348623157e308) is None
    assert _normalize_lmt_price(100.5) == 100.5


# ---------------------------------------------------------------------------
# Guard wired into the live submit path (drives _submit_via_ib end-to-end)
# ---------------------------------------------------------------------------


def _make_routed(
    symbol: str,
    side: SignalSide,
    qty: float,
    strategies: tuple[StrategyName, ...],
    asset_class: AssetClass = AssetClass.ETF,
    meta: dict | None = None,
) -> RoutedSignal:
    return RoutedSignal(
        symbol=symbol,
        side=side,
        net_size=qty,
        source_strategies=strategies,
        confidence=0.75,
        asset_class=asset_class,
        created_at=datetime.now(timezone.utc),
        meta=dict(meta or {}),
    )


def _live_cfg() -> IbkrConfig:
    return IbkrConfig(
        dry_run=False,
        enable_idempotency=False,
        max_submit_retries=1,
        initial_status_wait_s=0.0,
        retry_backoff_s=0.0,
    )


def test_guard_blocks_cross_strategy_duplicate_spy_buy_stk() -> None:
    """SPY BUY STK already working at broker → second submit from a different
    strategy is suppressed with duplicate_open_order; placeOrder is not called."""
    existing = _FakeTrade(
        _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD"),
        _FakeOrder(action="BUY", orderId=42),
        _FakeStatus(status="Submitted", remaining=10.0),
    )
    fake_ib = _FakeIB(open_trades=[existing])
    adapter = IbkrAdapter(config=_live_cfg(), ib_factory=lambda: fake_ib)

    routed = _make_routed("SPY", SignalSide.BUY, 5.0, (StrategyName.BETA,), AssetClass.ETF)
    submitted = adapter.submit_routed_signals([routed])

    assert len(submitted) == 1
    assert submitted[0].status == "duplicate_open_order"
    assert submitted[0].ib_order_id is None
    assert fake_ib.place_order_calls == []


def test_guard_does_not_block_different_side() -> None:
    existing = _FakeTrade(
        _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD"),
        _FakeOrder(action="SELL", orderId=42),
        _FakeStatus(status="Submitted", remaining=10.0),
    )
    fake_ib = _FakeIB(open_trades=[existing])
    adapter = IbkrAdapter(config=_live_cfg(), ib_factory=lambda: fake_ib)

    routed = _make_routed("SPY", SignalSide.BUY, 5.0, (StrategyName.BETA,), AssetClass.ETF)
    submitted = adapter.submit_routed_signals([routed])

    assert len(submitted) == 1
    assert submitted[0].status.lower() == "submitted"
    assert len(fake_ib.place_order_calls) == 1


def test_guard_does_not_block_different_symbol() -> None:
    existing = _FakeTrade(
        _FakeContract(symbol="QQQ", secType="STK", exchange="SMART", currency="USD"),
        _FakeOrder(action="BUY", orderId=42),
        _FakeStatus(status="Submitted", remaining=10.0),
    )
    fake_ib = _FakeIB(open_trades=[existing])
    adapter = IbkrAdapter(config=_live_cfg(), ib_factory=lambda: fake_ib)

    routed = _make_routed("SPY", SignalSide.BUY, 5.0, (StrategyName.BETA,), AssetClass.ETF)
    submitted = adapter.submit_routed_signals([routed])

    assert len(submitted) == 1
    assert submitted[0].status.lower() == "submitted"
    assert len(fake_ib.place_order_calls) == 1


def test_guard_does_not_block_different_futures_contract_month() -> None:
    """MES front-month FUT working at broker → MES with a different contract
    month must not be blocked (different broker-side lane)."""
    existing = _FakeTrade(
        _FakeContract(
            symbol="MES",
            secType="FUT",
            exchange="CME",
            currency="USD",
            lastTradeDateOrContractMonth="20260619",
            multiplier="5",
        ),
        _FakeOrder(action="BUY", orderId=42),
        _FakeStatus(status="Submitted", remaining=1.0),
    )
    fake_ib = _FakeIB(open_trades=[existing])
    adapter = IbkrAdapter(config=_live_cfg(), ib_factory=lambda: fake_ib)

    routed = _make_routed(
        "MES",
        SignalSide.BUY,
        1.0,
        (StrategyName.ALPHA_FUTURES,),
        AssetClass.FUTURES,
        meta={"contract_month": "20260918"},
    )
    submitted = adapter.submit_routed_signals([routed])

    assert len(submitted) == 1
    assert submitted[0].status.lower() == "submitted"
    assert len(fake_ib.place_order_calls) == 1


def test_guard_blocks_at_per_lane_cap() -> None:
    """12 working SPY BUY STK orders at broker → next submit suppressed with
    suppressed_open_orders_cap; placeOrder is not called."""
    existing_trades: List[_FakeTrade] = []
    for i in range(12):
        existing_trades.append(
            _FakeTrade(
                _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD"),
                _FakeOrder(action="BUY", lmtPrice=100.0 + i, orderId=100 + i),
                _FakeStatus(status="Submitted", remaining=1.0),
            )
        )
    fake_ib = _FakeIB(open_trades=existing_trades)
    adapter = IbkrAdapter(config=_live_cfg(), ib_factory=lambda: fake_ib)

    # Use a fresh limit price so the duplicate check does NOT fire — only the
    # per-lane cap should suppress this submission.
    routed = _make_routed(
        "SPY",
        SignalSide.BUY,
        5.0,
        (StrategyName.BETA,),
        AssetClass.ETF,
        meta={"order_type": "LMT", "limit_price": 999.0},
    )
    submitted = adapter.submit_routed_signals([routed])

    assert len(submitted) == 1
    assert submitted[0].status == "suppressed_open_orders_cap"
    assert submitted[0].ib_order_id is None
    assert fake_ib.place_order_calls == []


def test_guard_below_cap_allows_submit() -> None:
    """11 working SPY BUY STK orders (cap=12) → next submit goes through.

    Use distinct lmtPrice values so the duplicate-key check does not fire.
    """
    existing_trades: List[_FakeTrade] = []
    for i in range(11):
        existing_trades.append(
            _FakeTrade(
                _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD"),
                _FakeOrder(action="BUY", lmtPrice=100.0 + i, orderId=100 + i),
                _FakeStatus(status="Submitted", remaining=1.0),
            )
        )
    fake_ib = _FakeIB(open_trades=existing_trades)
    adapter = IbkrAdapter(config=_live_cfg(), ib_factory=lambda: fake_ib)

    routed = _make_routed(
        "SPY",
        SignalSide.BUY,
        5.0,
        (StrategyName.BETA,),
        AssetClass.ETF,
        meta={"order_type": "LMT", "limit_price": 999.0},
    )
    submitted = adapter.submit_routed_signals([routed])

    assert len(submitted) == 1
    assert submitted[0].status.lower() == "submitted"
    assert len(fake_ib.place_order_calls) == 1


def test_guard_disabled_via_config() -> None:
    existing = _FakeTrade(
        _FakeContract(symbol="SPY", secType="STK", exchange="SMART", currency="USD"),
        _FakeOrder(action="BUY", orderId=42),
        _FakeStatus(status="Submitted", remaining=10.0),
    )
    fake_ib = _FakeIB(open_trades=[existing])
    cfg = IbkrConfig(
        dry_run=False,
        enable_idempotency=False,
        enable_open_order_guard=False,
        max_submit_retries=1,
        initial_status_wait_s=0.0,
        retry_backoff_s=0.0,
    )
    adapter = IbkrAdapter(config=cfg, ib_factory=lambda: fake_ib)

    routed = _make_routed("SPY", SignalSide.BUY, 5.0, (StrategyName.BETA,), AssetClass.ETF)
    submitted = adapter.submit_routed_signals([routed])

    assert submitted[0].status.lower() == "submitted"
    assert len(fake_ib.place_order_calls) == 1


# ---------------------------------------------------------------------------
# Paper-evidence safety: new statuses must be classified as unconfirmed
# ---------------------------------------------------------------------------


def test_new_statuses_are_unconfirmed_in_evidence_writer() -> None:
    from chad.core.live_loop import _UNCONFIRMED_BROKER_STATUSES, _should_persist_paper_evidence

    assert "duplicate_open_order" in _UNCONFIRMED_BROKER_STATUSES
    assert "suppressed_open_orders_cap" in _UNCONFIRMED_BROKER_STATUSES

    class _StubOrder:
        def __init__(self, status: str) -> None:
            self.status = status

    persist, reason = _should_persist_paper_evidence(_StubOrder("duplicate_open_order"), {})
    assert persist is False
    assert reason and "duplicate_open_order" in reason

    persist, reason = _should_persist_paper_evidence(_StubOrder("suppressed_open_orders_cap"), {})
    assert persist is False
    assert reason and "suppressed_open_orders_cap" in reason

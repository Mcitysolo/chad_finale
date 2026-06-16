#!/usr/bin/env python3
"""Futures off-switch hard gate at the broker-submit chokepoints.

Covers the wired off-switch added in chad.execution.futures_gate and enforced
at BOTH FUT-capable broker chokepoints:

  * chad.execution.ibkr_adapter.IbkrAdapter._submit_via_ib  (the must-have;
    the live-loop submit path funnels through here)
  * chad.execution.ibkr_trade_router.IBKRTradeRouter.execute (second
    FUT-capable broker path)

Operator decision under test: when any futures-disable flag is set, hard-block
EVERY futures order (FUT/FOP, BOTH sides, ALL intent classes incl. exit/flip)
— NO carve-out — fail-closed, without ever calling placeOrder. Equities are
unaffected. All tests are deterministic and never touch the network: the IB
session is a local fake whose placeOrder/whatIfOrder calls are recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List

import pytest

from chad.execution.futures_gate import (
    futures_execution_disabled,
    is_futures_sec_type,
)
from chad.execution.ibkr_adapter import (
    IbkrAdapter,
    IbkrConfig,
    NormalizedIntent,
    SubmittedOrder,
    _PreparedOrder,
    _ResolvedContract,
)
from chad.execution.ibkr_trade_router import (
    IBKRTradeRequest,
    IBKRTradeResponse,
    IBKRTradeRouter,
)


_FLAG_FORMS = [
    {"CHAD_DISABLE_FUTURES_EXECUTION": "1"},
    {"CHAD_DISABLE_FUTURES": "1"},
    {"CHAD_FUTURES_EXECUTION_ENABLED": "0"},
]
_ALL_FLAGS = (
    "CHAD_DISABLE_FUTURES_EXECUTION",
    "CHAD_DISABLE_FUTURES",
    "CHAD_FUTURES_EXECUTION_ENABLED",
)


def _now() -> datetime:
    return datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc)


def _clear_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALL_FLAGS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Fakes — record broker calls; never reach the network.
# --------------------------------------------------------------------------- #


@dataclass
class _FakeContract:
    secType: str = "FUT"
    symbol: str = "MES"


@dataclass
class _FakeOrderObj:
    orderId: int = 555
    permId: int = 0


@dataclass
class _FakeOrderStatus:
    status: str = "Filled"


@dataclass
class _FakeStatusEvent:
    subs: List[Any] = field(default_factory=list)

    def __iadd__(self, handler: Any) -> "_FakeStatusEvent":
        self.subs.append(handler)
        return self


@dataclass
class _FakeTrade:
    order: _FakeOrderObj = field(default_factory=_FakeOrderObj)
    orderStatus: _FakeOrderStatus = field(default_factory=_FakeOrderStatus)
    fills: List[Any] = field(default_factory=list)
    commissionReport: List[Any] = field(default_factory=list)
    statusEvent: _FakeStatusEvent = field(default_factory=_FakeStatusEvent)


class _RecordingIB:
    """Records placeOrder / whatIfOrder so tests can assert they never fire."""

    def __init__(self) -> None:
        self.place_calls: List[Any] = []
        self.whatif_calls: List[Any] = []

    def qualifyContracts(self, contract: Any) -> List[Any]:
        return [contract]

    def openTrades(self) -> List[Any]:
        return []

    def isConnected(self) -> bool:
        return True

    def sleep(self, _s: float) -> None:
        return None

    def placeOrder(self, contract: Any, order: Any) -> _FakeTrade:
        self.place_calls.append((contract, order))
        return _FakeTrade()

    def whatIfOrder(self, contract: Any, order: Any) -> _FakeOrderObj:
        self.whatif_calls.append((contract, order))
        return _FakeOrderObj(orderId=7)


def _make_adapter(tmp_path) -> IbkrAdapter:
    cfg = IbkrConfig(
        dry_run=False,
        state_db_path=tmp_path / "exec_state.sqlite3",
        terminal_wait_s=0.0,
        initial_status_wait_s=0.0,
    )
    # ib_factory=lambda: None bypasses the real ib_async factory (which raises
    # by design); we always hand _submit_via_ib our fake ib explicitly.
    return IbkrAdapter(config=cfg, ib_factory=lambda: None)  # type: ignore[arg-type]


def _intent(
    *,
    sec_type: str,
    side: str = "BUY",
    symbol: str = "MES",
    meta: Any = None,
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy="omega_macro",
        symbol=symbol,
        sec_type=sec_type,
        exchange="CME",
        currency="USD",
        side=side,
        order_type="MKT",
        quantity=1.0,
        notional_estimate=1000.0,
        asset_class="futures" if is_futures_sec_type(sec_type) else "equities",
        source_strategies=("omega_macro",),
        created_at=_now(),
        limit_price=None,
        meta=meta or {},
    )


def _submit(adapter: IbkrAdapter, ib: Any, intent: NormalizedIntent) -> SubmittedOrder:
    contract = _FakeContract(secType=intent.sec_type, symbol=intent.symbol)
    resolved = _ResolvedContract(contract=contract, summary={"sec_type": intent.sec_type})
    prepared = _PreparedOrder(order=_FakeOrderObj(), quantity=intent.quantity, what_if=False)
    return adapter._submit_via_ib(
        ib=ib,
        intent=intent,
        resolved_contract=resolved,
        prepared=prepared,
        submitted_at=_now(),
        idempotency_key=f"k::{intent.symbol}::{intent.side}::{intent.sec_type}",
    )


# --------------------------------------------------------------------------- #
# Shared predicate / classifier (single source of truth).
# --------------------------------------------------------------------------- #


def test_is_futures_sec_type():
    assert is_futures_sec_type("FUT") is True
    assert is_futures_sec_type("fop") is True
    assert is_futures_sec_type("  Fut ") is True
    assert is_futures_sec_type("STK") is False
    assert is_futures_sec_type("OPT") is False
    assert is_futures_sec_type("CASH") is False
    assert is_futures_sec_type(None) is False


def test_predicate_unset_false():
    assert futures_execution_disabled({}) is False


# --------------------------------------------------------------------------- #
# Adapter chokepoint (the must-have).
# --------------------------------------------------------------------------- #


def test_fut_buy_flag_set_blocked_placeorder_not_called(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES_EXECUTION", "1")
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr"):
        result = _submit(adapter, ib, _intent(sec_type="FUT", side="BUY"))
    assert result.status == "futures_execution_disabled"
    assert result.ib_order_id is None
    assert ib.place_calls == []
    assert ib.whatif_calls == []
    assert "FUTURES_EXECUTION_GATE_BLOCKED" in caplog.text


def test_fut_sell_flag_set_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES_EXECUTION", "1")
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    result = _submit(adapter, ib, _intent(sec_type="FUT", side="SELL"))
    assert result.status == "futures_execution_disabled"
    assert ib.place_calls == []


def test_fop_flag_set_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES_EXECUTION", "1")
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    result = _submit(adapter, ib, _intent(sec_type="FOP", side="BUY"))
    assert result.status == "futures_execution_disabled"
    assert ib.place_calls == []


@pytest.mark.parametrize("intent_meta", [{"intent": "exit"}, {"intent": "flip"}, {"reason": "max_hold_exit"}])
def test_exit_flip_fut_flag_set_blocked_no_carveout(tmp_path, monkeypatch, intent_meta):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES_EXECUTION", "1")
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    # Both sides exercised; the chokepoint has no notion of exit/flip → blocked.
    for side in ("BUY", "SELL"):
        result = _submit(adapter, ib, _intent(sec_type="FUT", side=side, meta=intent_meta))
        assert result.status == "futures_execution_disabled"
    assert ib.place_calls == []


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_stk_flag_set_not_blocked_placeorder_called(tmp_path, monkeypatch, side):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES_EXECUTION", "1")
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    result = _submit(adapter, ib, _intent(sec_type="STK", side=side, symbol="SPY"))
    assert result.status != "futures_execution_disabled"
    assert len(ib.place_calls) == 1


def test_fut_flag_not_set_allowed_placeorder_called(tmp_path, monkeypatch):
    _clear_flags(monkeypatch)
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    result = _submit(adapter, ib, _intent(sec_type="FUT", side="BUY"))
    assert result.status != "futures_execution_disabled"
    assert len(ib.place_calls) == 1


@pytest.mark.parametrize("flag_env", _FLAG_FORMS)
def test_each_flag_form_independently_blocks_adapter(tmp_path, monkeypatch, flag_env):
    _clear_flags(monkeypatch)
    for k, v in flag_env.items():
        monkeypatch.setenv(k, v)
    adapter = _make_adapter(tmp_path)
    ib = _RecordingIB()
    result = _submit(adapter, ib, _intent(sec_type="FUT", side="BUY"))
    assert result.status == "futures_execution_disabled"
    assert ib.place_calls == []


# --------------------------------------------------------------------------- #
# Trade-router chokepoint (second FUT-capable broker path).
# --------------------------------------------------------------------------- #


def _router_req(sec_type: str, side: str = "BUY", symbol: str = "MES") -> IBKRTradeRequest:
    return IBKRTradeRequest(
        symbol=symbol,
        sec_type=sec_type,
        exchange="CME",
        currency="USD",
        side=side,
        order_type="MKT",
        quantity=1.0,
        limit_price=None,
        what_if=False,
    )


def test_router_fut_flag_set_blocked_placeorder_not_called(monkeypatch, caplog):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES", "1")
    ib = _RecordingIB()
    router = IBKRTradeRouter(host="127.0.0.1", port=4002, client_id=99, ib=ib)
    with caplog.at_level(logging.WARNING, logger="chad.execution.ibkr_trade_router"):
        resp = router.execute(_router_req("FUT", "SELL"))
    assert isinstance(resp, IBKRTradeResponse)
    assert resp.status == "futures_execution_disabled"
    assert resp.order_id == 0
    assert ib.place_calls == []
    assert "FUTURES_EXECUTION_GATE_BLOCKED" in caplog.text


def test_router_fop_flag_set_blocked(monkeypatch):
    monkeypatch.setenv("CHAD_FUTURES_EXECUTION_ENABLED", "0")
    ib = _RecordingIB()
    router = IBKRTradeRouter(host="127.0.0.1", port=4002, client_id=99, ib=ib)
    resp = router.execute(_router_req("FOP", "BUY"))
    assert resp.status == "futures_execution_disabled"
    assert ib.place_calls == []


def test_router_stk_flag_set_not_blocked(monkeypatch):
    monkeypatch.setenv("CHAD_DISABLE_FUTURES", "1")
    ib = _RecordingIB()
    router = IBKRTradeRouter(host="127.0.0.1", port=4002, client_id=99, ib=ib)
    resp = router.execute(_router_req("STK", "BUY", symbol="SPY"))
    assert resp.status != "futures_execution_disabled"
    assert len(ib.place_calls) == 1


def test_router_fut_flag_not_set_allowed(monkeypatch):
    _clear_flags(monkeypatch)
    ib = _RecordingIB()
    router = IBKRTradeRouter(host="127.0.0.1", port=4002, client_id=99, ib=ib)
    resp = router.execute(_router_req("FUT", "BUY"))
    assert resp.status != "futures_execution_disabled"
    assert len(ib.place_calls) == 1


@pytest.mark.parametrize("flag_env", _FLAG_FORMS)
def test_each_flag_form_independently_blocks_router(monkeypatch, flag_env):
    _clear_flags(monkeypatch)
    for k, v in flag_env.items():
        monkeypatch.setenv(k, v)
    ib = _RecordingIB()
    router = IBKRTradeRouter(host="127.0.0.1", port=4002, client_id=99, ib=ib)
    resp = router.execute(_router_req("FUT", "BUY"))
    assert resp.status == "futures_execution_disabled"
    assert ib.place_calls == []

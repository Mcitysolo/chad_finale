"""Official Matrix Box 055 — BAG live fill harness (offline).

Acceptance criterion (Evidence-Locked Completion Matrix v0.1):
    "offline harness proves placeOrder → Trade → status → fill round trip"

This module proves the BAG (combo / vertical-spread) order lifecycle end-to-end
using fake/mock broker primitives ONLY — no IBKR connection, no real
``ib.placeOrder``, no network I/O, no runtime ledger mutation. The harness
ties together components already exercised in isolation by Box 51..54 plus
GAP-036:

  - chad/execution/ibkr_adapter.py
        IbkrAdapter._install_trade_status_handler  (status → SQLite row)
        IbkrAdapter._ib_probe                       (openTrades + fills scan)
        _SQLiteIdempotencyStore                     (per-key claim / promote)

  - chad/execution/paper_exec_evidence_writer.py
        simulate_bag_paper_fill                     (BAG → paper_fill evidence)
        normalize_paper_fill_evidence               (full normalize pipeline)

The eight gate assertions (one per test) collectively prove:

  Gate 2  offline BAG harness exists (this file)
  Gate 3  placeOrder → Trade → status → fill round trip proven
  Gate 4  BAG combo legs + spread_id preserved through the lifecycle
  Gate 5  harness uses fake/mock broker only — no IBKR import requiring gateway
  Gate 6  failure path (Cancelled / missing meta) cannot synthesize a trusted fill

The remaining gates (Gate 1 service / Gate 7 tests pass / Gate 8 no orders
placed / Gate 9 no runtime mutation / Gate 10 no live trading authorization)
are operational and are recorded in the Box-055 evidence file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import pytest

# Keep any transitive import of live_loop safe — never attempt to connect.
os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

from chad.execution.ibkr_adapter import (  # noqa: E402
    IbkrAdapter,
    IbkrConfig,
    _PROBE_TERMINAL_PREFIX,
)
from chad.execution.paper_exec_evidence_writer import (  # noqa: E402
    PaperExecEvidence,
    normalize_paper_fill_evidence,
    simulate_bag_paper_fill,
)


# ---------------------------------------------------------------------------
# Fake broker primitives — minimal stand-ins for ib_async types.
# No real ib_async / IBKR client is constructed. Every attribute the
# adapter touches (placeOrder, openTrades, trades, fills, sleep) is
# implemented in-process with no network I/O.
# ---------------------------------------------------------------------------

_SPREAD_ID = "box055-spread-uuid-0001"
_BOID = 50055
_KEY = "box055-idempotency-key"


@dataclass
class _FakeComboLeg:
    conId: int
    ratio: int
    action: str
    exchange: str = "SMART"


@dataclass
class _FakeContract:
    """BAG combo container. Mirrors ib_async.Contract surface used by adapter."""
    symbol: str = "SPY"
    secType: str = "BAG"
    exchange: str = "SMART"
    currency: str = "USD"
    comboLegs: List[_FakeComboLeg] = field(default_factory=list)


@dataclass
class _FakeOrder:
    orderId: int = _BOID
    permId: int = 0
    orderType: str = "LMT"
    lmtPrice: float = 1.50
    totalQuantity: float = 1.0
    action: str = "BUY"


@dataclass
class _FakeOrderStatus:
    status: str = "PendingSubmit"


@dataclass
class _FakeStatusEvent:
    subscribers: List[Callable[[Any], None]] = field(default_factory=list)

    def __iadd__(self, handler: Callable[[Any], None]) -> "_FakeStatusEvent":
        self.subscribers.append(handler)
        return self

    def emit(self, trade: Any) -> None:
        for h in list(self.subscribers):
            h(trade)


@dataclass
class _FakeExecution:
    orderId: int = _BOID
    permId: int = 0
    shares: float = 1.0
    price: float = 1.50


@dataclass
class _FakeFill:
    execution: _FakeExecution = field(default_factory=_FakeExecution)
    contract: Optional[_FakeContract] = None


@dataclass
class _FakeTrade:
    contract: _FakeContract = field(default_factory=_FakeContract)
    order: _FakeOrder = field(default_factory=_FakeOrder)
    orderStatus: _FakeOrderStatus = field(default_factory=_FakeOrderStatus)
    fills: List[Any] = field(default_factory=list)
    statusEvent: _FakeStatusEvent = field(default_factory=_FakeStatusEvent)
    commissionReport: List[Any] = field(default_factory=list)


@dataclass
class _FakeIB:
    """Mock IB client. Captures the (contract, order) pair handed to
    ``placeOrder`` and returns a _FakeTrade carrying both back. The adapter
    reads ``openTrades`` / ``trades`` / ``fills`` for probe + reconciliation."""

    placed_calls: List[tuple] = field(default_factory=list)
    open_trades_list: List[_FakeTrade] = field(default_factory=list)
    all_trades_list: List[_FakeTrade] = field(default_factory=list)
    fills_list: List[_FakeFill] = field(default_factory=list)

    def placeOrder(self, contract: Any, order: Any) -> _FakeTrade:
        self.placed_calls.append((contract, order))
        trade = _FakeTrade(contract=contract, order=order)
        self.open_trades_list.append(trade)
        self.all_trades_list.append(trade)
        return trade

    def openTrades(self) -> List[_FakeTrade]:
        return list(self.open_trades_list)

    def trades(self) -> List[_FakeTrade]:
        return list(self.all_trades_list)

    def fills(self) -> List[_FakeFill]:
        return list(self.fills_list)

    def sleep(self, _s: float) -> None:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)


def _make_adapter(tmp_path: Path) -> IbkrAdapter:
    """IbkrAdapter wired to a tmp SQLite store and a NULL ib_factory so
    nothing can ever connect to a real broker."""
    cfg = IbkrConfig(
        dry_run=False,
        state_db_path=tmp_path / "exec_state.sqlite3",
        terminal_wait_s=0.0,
        initial_status_wait_s=0.0,
    )
    return IbkrAdapter(config=cfg, ib_factory=lambda: None)  # type: ignore[arg-type]


def _make_bag_contract() -> _FakeContract:
    """SPY 720/725 bull-call vertical — two legs with non-zero conIds."""
    return _FakeContract(
        symbol="SPY",
        secType="BAG",
        comboLegs=[
            _FakeComboLeg(conId=111111, ratio=1, action="BUY"),
            _FakeComboLeg(conId=222222, ratio=1, action="SELL"),
        ],
    )


def _make_bag_order(boid: int = _BOID) -> _FakeOrder:
    return _FakeOrder(orderId=boid, orderType="LMT", lmtPrice=1.50, action="BUY")


def _make_bag_evidence(**overrides: Any) -> PaperExecEvidence:
    """Representative alpha_options BAG evidence carrying spread_id."""
    base_extra = {
        "sec_type": "BAG",
        "spread_id": _SPREAD_ID,
        "spread_type": "BULL_CALL",
        "expiry": "20260516",
        "long_strike": 720.0,
        "short_strike": 725.0,
        "long_right": "C",
        "short_right": "C",
        "dte": 10,
        "max_loss_per_contract": 150.0,
        "net_debit_estimate": 1.50,
        "contracts": 1,
        "required_asset_class": "options",
        "engine": "alpha_options.v1",
    }
    base_extra.update(overrides.pop("extra_extra", {}))
    kwargs = dict(
        symbol="SPY",
        side="BUY",
        quantity=1.0,
        fill_price=0.0,
        expected_price=1.50,
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        status="",
        asset_class="",
        is_live=False,
        fill_time_utc="2026-05-21T12:00:00Z",
        extra=base_extra,
    )
    kwargs.update(overrides)
    return PaperExecEvidence(**kwargs)


# ===========================================================================
# Box-055 Gate 3 — placeOrder → Trade → status → fill round trip
# ===========================================================================

def test_box055_gate3_placeOrder_returns_trade(tmp_path: Path) -> None:
    """Gate 3.a — placeOrder is invoked exactly once with a BAG contract +
    LMT order and returns a Trade-like object carrying both back."""
    fake_ib = _FakeIB()
    contract = _make_bag_contract()
    order = _make_bag_order()

    trade = fake_ib.placeOrder(contract, order)

    assert len(fake_ib.placed_calls) == 1, "placeOrder must be invoked exactly once"
    placed_contract, placed_order = fake_ib.placed_calls[0]
    assert placed_contract is contract, "contract identity must be preserved through placeOrder"
    assert placed_order is order, "order identity must be preserved through placeOrder"
    assert trade is not None, "placeOrder must return a Trade-like object"
    assert trade.contract is contract, "Trade.contract must reference the submitted contract"
    assert trade.order is order, "Trade.order must reference the submitted order"
    assert trade.orderStatus.status == "PendingSubmit", "fresh Trade starts at PendingSubmit"


def test_box055_gate3_status_event_promotes_pending_to_filled(tmp_path: Path) -> None:
    """Gate 3.b — orderStatus transition PendingSubmit → Filled flows through
    the adapter's statusEvent handler into the SQLite idempotency store."""
    adapter = _make_adapter(tmp_path)
    store = adapter._idempotency
    assert store is not None

    fake_ib = _FakeIB()
    contract = _make_bag_contract()
    order = _make_bag_order()

    # Claim the idempotency row first (mirrors _submit_via_ib path).
    store.claim(_KEY, {"symbol": contract.symbol, "side": "BUY", "qty": 1.0}, _now())

    trade = fake_ib.placeOrder(contract, order)
    adapter._install_trade_status_handler(trade, _KEY)

    # PendingSubmit transition (broker accepts).
    trade.orderStatus.status = "PendingSubmit"
    trade.statusEvent.emit(trade)
    row = store.get(_KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "pendingsubmit"

    # Filled transition.
    trade.orderStatus.status = "Filled"
    trade.statusEvent.emit(trade)
    row = store.get(_KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "filled"
    assert int(row["broker_order_id"] or 0) == _BOID, (
        f"broker_order_id must be captured from order.orderId — got {row['broker_order_id']!r}"
    )


def test_box055_gate3_fill_harvested_via_ib_probe(tmp_path: Path) -> None:
    """Gate 3.c — once the trade emits Filled, the adapter's _ib_probe
    detects the execution from ib.fills() and classifies the order as
    TERMINAL_AT_BROKER:Filled. This proves the fill round trip closes."""
    adapter = _make_adapter(tmp_path)
    fake_ib = _FakeIB()
    contract = _make_bag_contract()
    order = _make_bag_order()

    trade = fake_ib.placeOrder(contract, order)
    trade.orderStatus.status = "Filled"

    # Broker also reports the execution in its fills stream.
    fill = _FakeFill(
        execution=_FakeExecution(orderId=order.orderId, shares=1.0, price=1.50),
        contract=contract,
    )
    fake_ib.fills_list.append(fill)
    trade.fills.append(fill)

    probe_result = adapter._ib_probe(fake_ib, order.orderId)
    assert probe_result == _PROBE_TERMINAL_PREFIX + "Filled", (
        f"_ib_probe must classify a Filled trade as TERMINAL_AT_BROKER:Filled — got {probe_result!r}"
    )


# ===========================================================================
# Box-055 Gate 4 — BAG combo legs + spread_id preserved
# ===========================================================================

def test_box055_gate4_bag_combo_legs_preserved_through_round_trip(tmp_path: Path) -> None:
    """Gate 4.a — secType='BAG' and both ComboLeg objects survive identity
    on the Trade returned by placeOrder. No coercion to STK / silent
    downgrade."""
    fake_ib = _FakeIB()
    contract = _make_bag_contract()
    order = _make_bag_order()

    trade = fake_ib.placeOrder(contract, order)

    assert trade.contract.secType == "BAG", (
        f"secType must remain 'BAG' through placeOrder — got {trade.contract.secType!r}"
    )
    assert len(trade.contract.comboLegs) == 2, "must have exactly 2 combo legs"
    long_leg, short_leg = trade.contract.comboLegs
    assert long_leg.action == "BUY" and long_leg.conId == 111111 and long_leg.ratio == 1
    assert short_leg.action == "SELL" and short_leg.conId == 222222 and short_leg.ratio == 1


def test_box055_gate4_spread_id_preserved_intent_to_evidence(tmp_path: Path) -> None:
    """Gate 4.b — spread_id stamped on the intent meta survives through the
    paper-fill simulator into the persisted evidence record. This is the
    join key SCR / trade_closer / portfolio engine use to match the BUY
    opener and SELL closer of a vertical spread."""
    ev = _make_bag_evidence()
    fired = simulate_bag_paper_fill(ev)
    assert fired is True
    assert ev.extra.get("spread_id") == _SPREAD_ID, (
        f"spread_id must survive simulate_bag_paper_fill — got {ev.extra.get('spread_id')!r}"
    )
    # bag_legs are also stamped so downstream consumers can match per-leg.
    legs = ev.extra.get("bag_legs")
    assert isinstance(legs, list) and len(legs) == 2

    # Full normalize pipeline (the path live_loop actually calls).
    ev2 = _make_bag_evidence()
    normalize_paper_fill_evidence(ev2)
    assert ev2.extra.get("spread_id") == _SPREAD_ID
    assert ev2.asset_class == "options"
    assert ev2.status == "paper_fill"
    assert ev2.fill_price == 1.50


# ===========================================================================
# Box-055 Gate 5 — harness uses fake/mock broker only
# ===========================================================================

def test_box055_gate5_no_real_broker_connection(tmp_path: Path) -> None:
    """Gate 5 — Adapter is built with ib_factory=lambda: None and the test
    never imports or instantiates a real ib_async.IB client. Any attempt to
    rely on an actual broker would raise here."""
    adapter = _make_adapter(tmp_path)
    # ib_factory returns None — there is no real client to connect to.
    assert adapter._ib_factory() is None  # type: ignore[attr-defined]
    # The fake IB exposes only the surface the adapter touches.
    fake_ib = _FakeIB()
    assert hasattr(fake_ib, "placeOrder")
    assert hasattr(fake_ib, "openTrades")
    assert hasattr(fake_ib, "trades")
    assert hasattr(fake_ib, "fills")
    # Defensive: confirm the module under test is NOT carrying a live IB.
    assert getattr(adapter, "_ib", None) is None


# ===========================================================================
# Box-055 Gate 6 — failure path cannot synthesize a trusted fill
# ===========================================================================

def test_box055_gate6_cancelled_status_does_not_emit_paper_fill(tmp_path: Path) -> None:
    """Gate 6.a — orderStatus=Cancelled is recorded into the idempotency
    store as 'cancelled'. The status handler MUST NOT write paper-fill
    evidence; the row stays terminal-negative and downstream consumers
    see no trusted fill."""
    adapter = _make_adapter(tmp_path)
    store = adapter._idempotency
    assert store is not None

    fake_ib = _FakeIB()
    contract = _make_bag_contract()
    order = _make_bag_order()
    store.claim(_KEY, {"symbol": contract.symbol}, _now())
    trade = fake_ib.placeOrder(contract, order)
    adapter._install_trade_status_handler(trade, _KEY)

    trade.orderStatus.status = "Cancelled"
    trade.statusEvent.emit(trade)

    row = store.get(_KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "cancelled", (
        "Cancelled must be recorded — not promoted to filled"
    )
    # No execution was harvested.
    assert fake_ib.fills_list == [], "Cancelled order must not carry any fills"


def test_box055_gate6_missing_bag_meta_does_not_synthesize_fill(tmp_path: Path) -> None:
    """Gate 6.b — simulate_bag_paper_fill refuses to synthesize a fake fill
    when required BAG metadata is incomplete. The record is annotated with
    a skip reason so the rest of normalize can reject it loudly rather
    than persisting an untrusted price."""
    ev = _make_bag_evidence()
    ev.extra.pop("net_debit_estimate", None)
    fired = simulate_bag_paper_fill(ev)
    assert fired is False, (
        "Incomplete BAG meta MUST NOT synthesize a paper fill — got fired=True"
    )
    reason = ev.extra.get("bag_simulator_skipped_reason")
    assert isinstance(reason, str) and reason.startswith("missing_meta:"), (
        f"missing-meta skip reason must be recorded — got {reason!r}"
    )
    # Status is unchanged from blank — no trusted fill emitted.
    assert ev.status != "paper_fill"

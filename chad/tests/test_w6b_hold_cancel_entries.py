"""
chad/tests/test_w6b_hold_cancel_entries.py

W6B-13 — cancel ENTRY orders working at the broker when a hold is applied.

This is the highest-risk item in the lane, and the tests are weighted
accordingly: almost all of them assert that something is NOT cancelled.

The two traps, both pinned below:

  1. clientId-scoped enumeration. `ib.openOrders()` silently returns nothing
     for other clients' orders. A tool built on it reports "cancelled 0, all
     clear" while orders keep working — a confident, wrong success. So the
     enumerator must abort rather than fall back, and there is a named test
     for exactly that.

  2. indiscriminate cancellation. paper_position_closer cancels EVERY open
     order; applied to a hold that would strip protective exits and leave
     positions naked — strictly worse than the problem being solved.

Classification has no CHAD tag to lean on: CHAD does not set `orderRef`, so a
broker-enumerated order carries no intent_class. Everything is inferred, and
every inference failure resolves to "leave it alone".
"""

from __future__ import annotations

import json

import pytest

from ops import hold_cancel_entries as hce


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeOrder:
    def __init__(self, order_id=1, action="BUY", qty=10, order_type="LMT",
                 parent_id=0, oca_group="", client_id=5, perm_id=100):
        self.orderId = order_id
        self.action = action
        self.totalQuantity = qty
        self.orderType = order_type
        self.parentId = parent_id
        self.ocaGroup = oca_group
        self.clientId = client_id
        self.permId = perm_id


class FakeContract:
    def __init__(self, symbol="AAPL", sec_type="STK"):
        self.symbol = symbol
        self.secType = sec_type


class FakeStatus:
    def __init__(self, status="Submitted"):
        self.status = status


class FakeTrade:
    def __init__(self, order=None, contract=None, status="Submitted"):
        self.order = order or FakeOrder()
        self.contract = contract or FakeContract()
        self.orderStatus = FakeStatus(status)


class FakePosition:
    def __init__(self, symbol, qty):
        self.contract = FakeContract(symbol)
        self.position = qty


class FakeIB:
    """Exposes reqAllOpenOrders (cross-client) — the required surface."""

    def __init__(self, orders=None, positions=None, positions_raise=False):
        self._orders = orders or []
        self._positions = positions if positions is not None else []
        self._positions_raise = positions_raise
        self.cancelled = []

    def reqAllOpenOrders(self):
        return list(self._orders)

    def positions(self):
        if self._positions_raise:
            raise ConnectionError("no broker")
        return list(self._positions)

    def cancelOrder(self, order):
        self.cancelled.append(order.orderId)


PAPER_ENV = {"CHAD_EXECUTION_MODE": "paper", "CHAD_HOLD_CANCEL_ENTRIES": "1"}


# --------------------------------------------------------------------------
# Trap 1 — cross-client enumeration
# --------------------------------------------------------------------------

def test_enumeration_uses_the_cross_client_call():
    ib = FakeIB(orders=[FakeTrade()])
    assert len(hce.enumerate_all_open_orders(ib)) == 1


def test_aborts_when_only_the_clientid_scoped_call_exists():
    """THE trap. openOrders() returns [] for other clients' orders WITHOUT
    erroring, so falling back to it would report a confident, wrong success."""

    class ScopedOnlyIB:
        def openOrders(self):
            return []  # would look like "nothing to cancel"

    with pytest.raises(hce.HoldCancelAbort) as exc:
        hce.enumerate_all_open_orders(ScopedOnlyIB())
    assert "clientId-scoped" in str(exc.value)


def test_aborts_rather_than_degrading_when_enumeration_raises():
    class BrokenIB:
        def reqAllOpenOrders(self):
            raise ConnectionError("socket closed")

        def openOrders(self):
            return []

    with pytest.raises(hce.HoldCancelAbort):
        hce.enumerate_all_open_orders(BrokenIB())


def test_report_records_every_client_id_seen():
    """Evidence that the enumeration really was cross-client."""
    ib = FakeIB(
        orders=[
            FakeTrade(FakeOrder(order_id=1, client_id=5)),
            FakeTrade(FakeOrder(order_id=2, client_id=77)),
        ],
        positions=[],
    )
    report = hce.run(ib, execute=False, env=PAPER_ENV)
    assert report["client_ids_seen"] == [5, 77]


# --------------------------------------------------------------------------
# Trap 2 — protective orders must survive
# --------------------------------------------------------------------------

def _classify(order, positions=None, known=True):
    row = hce._row_from_broker(FakeTrade(order))
    return hce.classify_order(row, positions or {}, positions_known=known)


def test_bracket_child_is_protective():
    c = _classify(FakeOrder(parent_id=42))
    assert c.verdict == "protective" and c.cancellable is False


def test_oca_leg_is_protective():
    c = _classify(FakeOrder(oca_group="OCA_123"))
    assert c.verdict == "protective" and c.cancellable is False


@pytest.mark.parametrize("otype", ["STP", "STP LMT", "TRAIL", "TRAIL LIMIT", "MIT", "LIT"])
def test_stop_and_trail_families_are_protective(otype):
    c = _classify(FakeOrder(order_type=otype))
    assert c.verdict == "protective" and c.cancellable is False


def test_sell_against_a_long_is_reduce_only():
    c = _classify(FakeOrder(action="SELL"), {"AAPL": 100.0})
    assert c.verdict == "reduce_only" and c.cancellable is False


def test_buy_against_a_short_is_reduce_only():
    c = _classify(FakeOrder(action="BUY"), {"AAPL": -100.0})
    assert c.verdict == "reduce_only" and c.cancellable is False


def test_oversized_opposing_order_is_still_reduce_only_not_entry():
    """A flip's closing half is still an exit. Cancelling it is not this
    tool's business, and treating a flip as an entry would strip the close."""
    c = _classify(FakeOrder(action="SELL", qty=500), {"AAPL": 100.0})
    assert c.verdict == "reduce_only"


# --------------------------------------------------------------------------
# Fail-closed on ambiguity
# --------------------------------------------------------------------------

def test_unknown_order_type_is_left_alone():
    c = _classify(FakeOrder(order_type="VWAP"))
    assert c.verdict == "unknown" and c.cancellable is False


def test_unknown_action_is_left_alone():
    c = _classify(FakeOrder(action="SSHORT"), {"AAPL": 0.0})
    assert c.verdict == "unknown" and c.cancellable is False


def test_unavailable_positions_leave_everything_alone():
    """Without positions an entry cannot be told from a reduce-only order.
    Reading "no positions" as "flat" would make every SELL look like an
    entry — the single most dangerous misreading available here."""
    c = _classify(FakeOrder(action="SELL"), {}, known=False)
    assert c.verdict == "unknown" and c.cancellable is False


def test_broker_position_failure_cancels_nothing_end_to_end():
    ib = FakeIB(orders=[FakeTrade(FakeOrder(action="SELL"))], positions_raise=True)
    report = hce.run(ib, execute=True, env=PAPER_ENV)
    assert ib.cancelled == []
    assert report["positions_known"] is False
    assert any("UNAVAILABLE" in n for n in report["loud_notes"])


def test_unparseable_order_is_recorded_and_left_alone():
    ib = FakeIB(orders=[object()], positions=[])
    report = hce.run(ib, execute=True, env=PAPER_ENV)
    assert ib.cancelled == []
    assert len(report["orders_unparseable"]) == 1


# --------------------------------------------------------------------------
# The positive case
# --------------------------------------------------------------------------

def test_plain_entry_on_a_flat_book_is_cancellable():
    c = _classify(FakeOrder(action="BUY", order_type="LMT"), {"AAPL": 0.0})
    assert c.verdict == "entry" and c.cancellable is True


def test_adding_to_an_existing_long_is_an_entry():
    c = _classify(FakeOrder(action="BUY"), {"AAPL": 50.0})
    assert c.verdict == "entry" and c.cancellable is True


def test_execute_cancels_only_the_entry():
    entry = FakeOrder(order_id=1, action="BUY", order_type="LMT")
    stop = FakeOrder(order_id=2, action="SELL", order_type="STP")
    child = FakeOrder(order_id=3, action="SELL", order_type="LMT", parent_id=1)
    ib = FakeIB(
        orders=[FakeTrade(entry), FakeTrade(stop), FakeTrade(child)],
        positions=[FakePosition("AAPL", 100.0)],
    )
    report = hce.run(ib, execute=True, env=PAPER_ENV)
    assert ib.cancelled == [1]
    assert report["counts"]["cancelled"] == 1
    assert report["counts"]["left_alone"] == 2
    assert report["outcome"] == "complete"


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def test_execute_refuses_without_the_flag():
    with pytest.raises(hce.HoldCancelAbort) as exc:
        hce.check_gates({"CHAD_EXECUTION_MODE": "paper"}, execute=True)
    assert hce.ENABLE_FLAG in str(exc.value)


def test_plan_works_without_the_flag():
    """An operator must always be able to SEE what would be cancelled without
    arming anything."""
    hce.check_gates({"CHAD_EXECUTION_MODE": "paper"}, execute=False)


@pytest.mark.parametrize("mode", ["live", "LIVE", "", "production"])
def test_non_paper_mode_refuses_even_for_a_plan(mode):
    with pytest.raises(hce.HoldCancelAbort):
        hce.check_gates({"CHAD_EXECUTION_MODE": mode}, execute=False)


def test_plan_never_cancels():
    ib = FakeIB(
        orders=[FakeTrade(FakeOrder(action="BUY"))],
        positions=[FakePosition("AAPL", 0.0)],
    )
    report = hce.run(ib, execute=False, env={"CHAD_EXECUTION_MODE": "paper"})
    assert ib.cancelled == []
    assert report["mode"] == "plan"
    assert report["outcome"] == "planned"
    assert report["counts"]["cancellable"] == 1


# --------------------------------------------------------------------------
# The report must be loud about partial and zero outcomes
# --------------------------------------------------------------------------

def test_zero_cancelled_is_not_reported_as_clean_success():
    """"cancelled 0" must never read as "the book is clear"."""
    ib = FakeIB(orders=[FakeTrade(FakeOrder(order_type="STP"))], positions=[])
    report = hce.run(ib, execute=True, env=PAPER_ENV)
    assert report["outcome"] == "nothing_cancelled"
    assert any("NOT proof" in n for n in report["loud_notes"])


def test_cancel_failure_is_loud_and_marks_the_outcome_partial():
    class FailingIB(FakeIB):
        def cancelOrder(self, order):
            if order.orderId == 2:
                raise TimeoutError("no ack")
            self.cancelled.append(order.orderId)

    ib = FailingIB(
        orders=[
            FakeTrade(FakeOrder(order_id=1, action="BUY")),
            FakeTrade(FakeOrder(order_id=2, action="BUY")),
        ],
        positions=[FakePosition("AAPL", 0.0)],
    )
    report = hce.run(ib, execute=True, env=PAPER_ENV)
    assert report["counts"]["cancel_failed"] == 1
    assert report["outcome"] == "partial"
    assert any("FAILED" in n for n in report["loud_notes"])


def test_not_cancelled_entries_carry_a_reason_each():
    ib = FakeIB(
        orders=[FakeTrade(FakeOrder(order_id=9, order_type="STP"))], positions=[]
    )
    report = hce.run(ib, execute=False, env={"CHAD_EXECUTION_MODE": "paper"})
    (nc,) = report["not_cancelled"]
    assert nc["order_id"] == 9
    assert nc["verdict"] == "protective"
    assert "protective by construction" in nc["reason"]


def test_report_is_a_pinned_schema_and_writes_atomically(tmp_path):
    ib = FakeIB(orders=[], positions=[])
    report = hce.run(ib, execute=False, env={"CHAD_EXECUTION_MODE": "paper"})
    out = tmp_path / "hold_cancel_report.json"
    hce.write_report(report, out)
    assert json.loads(out.read_text())["schema_version"] == "hold_cancel_report.v1"
    assert not list(tmp_path.glob("*.tmp.*"))


# --------------------------------------------------------------------------
# Channel-1 posture
# --------------------------------------------------------------------------

def test_entrypoint_is_blocked_by_the_order_guard():
    """This tool cancels working broker orders, so it must sit behind the same
    guard as the flatten tools — operator-invoked, never agent-invoked."""
    from pathlib import Path

    guard = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "chad-order-guard.sh"
    assert "hold_cancel_entries" in guard.read_text(encoding="utf-8")

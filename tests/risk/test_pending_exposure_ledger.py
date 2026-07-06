"""Tests for chad/risk/pending_exposure_ledger.py (Margin BLOCK Phase B, ledger).

Fixtures only — no network, no broker, injected time (deterministic). Exercises:
reserve+release round-trip; the reducing bound that the gate depends on
(pending_reducing_qty accumulates across a burst, and is ZERO if a reducer skips
the ledger); total_reserved sums increasing + reducing; release of an unknown id
is a no-op; a duplicate order_id is rejected deterministically; expire()
reconciles against injected broker truth and NEVER blind-releases (the
leaked-reservation guard) — including the empty-set vs None distinction;
rebuild_from_broker fully replaces state from broker open-orders and sets the
restart-rebuild flag; determinism (same ops → same state); the conservative
partial-fill retention; strict fail-loud validation; and the module imports NO
live/execution/broker path and exposes NO order method.
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from chad.risk.pending_exposure_ledger import (
    DEFAULT_RESERVATION_TTL_SECONDS,
    PendingExposureLedger,
    Reservation,
    ReservedTotals,
)

# A broker-open-order spec as an attribute object (rebuild accepts Mapping OR
# attribute object — mirror the Phase-A dual-access convention).
OpenOrder = namedtuple("OpenOrder", "order_id side symbol reducing margin notional qty")

_T0 = 1000.0  # injected base epoch


def _led(ttl=DEFAULT_RESERVATION_TTL_SECONDS):
    return PendingExposureLedger(ttl_seconds=ttl)


def _reserve_increasing(led, oid, symbol="AAPL", *, qty=10.0, margin=500.0,
                        notional=1000.0, side="BUY", now=_T0):
    return led.reserve(oid, side, symbol, False, margin, notional, qty=qty, now=now)


def _reserve_reducing(led, oid, symbol="AAPL", *, qty=100.0, margin=0.0,
                      notional=1000.0, side="SELL", now=_T0):
    return led.reserve(oid, side, symbol, True, margin, notional, qty=qty, now=now)


# ---------------------------------------------------------------------------
# reserve + release round-trip
# ---------------------------------------------------------------------------
def test_reserve_then_release_round_trip():
    led = _led()
    assert len(led) == 0
    res = _reserve_increasing(led, "o1")
    assert isinstance(res, Reservation)
    assert led.is_reserved("o1") is True
    assert len(led) == 1
    assert led.get("o1") is res
    removed = led.release("o1")
    assert removed is True
    assert led.is_reserved("o1") is False
    assert len(led) == 0
    assert led.get("o1") is None


def test_reserve_normalizes_symbol_side_and_order_id():
    led = _led()
    led.reserve("  o1 ", " buy ", " aapl ", False, 1.0, 2.0, qty=3.0, now=_T0)
    res = led.get("o1")
    assert res is not None
    assert res.order_id == "o1"
    assert res.side == "BUY"
    assert res.symbol == "AAPL"
    # Reserved-at is the injected time, not wall-clock.
    assert res.reserved_at_epoch == _T0


# ---------------------------------------------------------------------------
# the reducing bound (the load-bearing invariant, §1.3 / CHANGELOG 2.1→2.2)
# ---------------------------------------------------------------------------
def test_reducing_reservation_accumulates_pending_reducing_qty():
    led = _led()
    assert led.pending_reducing_qty("AAPL") == 0.0
    _reserve_reducing(led, "r1", "AAPL", qty=100.0)
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)


def test_burst_of_three_reducers_sums_the_qty_the_gate_bounds_against():
    """Three SELL-100 reducers → pending_reducing_qty == 300 on that symbol.

    This is precisely the value the gate subtracts to shrink the reducible
    remainder across a burst; if it did not accumulate, a reduce-burst would
    over-flatten past flat into a short (the exact 2.1→2.2 bug)."""
    led = _led()
    _reserve_reducing(led, "r1", "AAPL", qty=100.0)
    _reserve_reducing(led, "r2", "AAPL", qty=100.0)
    _reserve_reducing(led, "r3", "AAPL", qty=100.0)
    assert led.pending_reducing_qty("AAPL") == pytest.approx(300.0)


def test_pending_reducing_qty_counts_only_reducers_on_that_symbol():
    led = _led()
    _reserve_reducing(led, "r1", "AAPL", qty=100.0)
    _reserve_increasing(led, "i1", "AAPL", qty=50.0)   # increasing — excluded
    _reserve_reducing(led, "r2", "MSFT", qty=70.0)      # other symbol — excluded
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)
    assert led.pending_reducing_qty("MSFT") == pytest.approx(70.0)
    assert led.pending_reducing_qty("TSLA") == 0.0


def test_pending_reducing_qty_symbol_is_normalized():
    led = _led()
    _reserve_reducing(led, "r1", "aapl", qty=40.0)
    assert led.pending_reducing_qty("  AAPL ") == pytest.approx(40.0)


def test_releasing_a_reducer_shrinks_pending_reducing_qty():
    """A reducer that fills/cancels frees its qty — the remainder re-grows."""
    led = _led()
    _reserve_reducing(led, "r1", "AAPL", qty=100.0)
    _reserve_reducing(led, "r2", "AAPL", qty=100.0)
    assert led.pending_reducing_qty("AAPL") == pytest.approx(200.0)
    led.release("r1")
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# total_reserved sums increasing + reducing
# ---------------------------------------------------------------------------
def test_total_reserved_sums_increasing_and_reducing_separately():
    led = _led()
    _reserve_increasing(led, "i1", margin=500.0, notional=1000.0)
    _reserve_reducing(led, "r1", margin=250.0, notional=800.0)
    totals = led.total_reserved()
    assert isinstance(totals, ReservedTotals)
    assert totals.margin == pytest.approx(750.0)
    assert totals.notional == pytest.approx(1800.0)
    assert totals.count == 2


def test_total_reserved_empty_ledger_is_zero():
    totals = _led().total_reserved()
    assert totals.margin == 0.0
    assert totals.notional == 0.0
    assert totals.count == 0


def test_total_reserved_reflects_release():
    led = _led()
    _reserve_increasing(led, "i1", margin=500.0, notional=1000.0)
    _reserve_increasing(led, "i2", margin=300.0, notional=600.0)
    led.release("i1")
    totals = led.total_reserved()
    assert totals.margin == pytest.approx(300.0)
    assert totals.notional == pytest.approx(600.0)
    assert totals.count == 1


# ---------------------------------------------------------------------------
# release: unknown id no-op, partial-fill conservative retention
# ---------------------------------------------------------------------------
def test_release_unknown_id_is_noop_not_crash():
    led = _led()
    assert led.release("never-reserved") is False
    assert len(led) == 0


def test_release_unknown_id_leaves_other_reservations_intact():
    led = _led()
    _reserve_increasing(led, "i1")
    assert led.release("ghost") is False
    assert led.is_reserved("i1") is True


def test_partial_fill_retains_full_reservation_conservative():
    """filled_qty is the future per-fill-shrink hook; today it must NOT free
    exposure — the full reservation persists (conservative, §1.3)."""
    led = _led()
    _reserve_reducing(led, "r1", "AAPL", qty=100.0, margin=200.0, notional=900.0)
    kept = led.release("r1", filled_qty=40.0)
    assert kept is False  # nothing freed
    assert led.is_reserved("r1") is True
    # Reducing qty and totals are unchanged (full reservation retained).
    assert led.pending_reducing_qty("AAPL") == pytest.approx(100.0)
    assert led.total_reserved().notional == pytest.approx(900.0)
    # A subsequent terminal release (no filled_qty) removes it.
    assert led.release("r1") is True
    assert led.is_reserved("r1") is False


def test_partial_fill_validates_filled_qty():
    led = _led()
    _reserve_reducing(led, "r1", "AAPL", qty=100.0)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            led.release("r1", filled_qty=bad)
    # Reservation survives the rejected partial.
    assert led.is_reserved("r1") is True


# ---------------------------------------------------------------------------
# duplicate order_id handled deterministically (reject, loud)
# ---------------------------------------------------------------------------
def test_duplicate_order_id_is_rejected():
    led = _led()
    _reserve_increasing(led, "o1", notional=1000.0)
    with pytest.raises(ValueError, match="duplicate order_id"):
        _reserve_increasing(led, "o1", notional=9999.0)
    # First reservation stands unchanged; the second's amount never leaks in.
    assert led.total_reserved().notional == pytest.approx(1000.0)
    assert len(led) == 1


def test_reserve_after_release_reuses_id_ok():
    """A re-used id is fine once the prior reservation is released (terminal)."""
    led = _led()
    _reserve_increasing(led, "o1", notional=1000.0)
    led.release("o1")
    _reserve_increasing(led, "o1", notional=2000.0)  # no longer a duplicate
    assert led.total_reserved().notional == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# strict fail-loud validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field_kwargs", [
    {"margin": float("nan")},
    {"margin": float("inf")},
    {"margin": -1.0},
    {"notional": float("nan")},
    {"notional": -0.01},
    {"qty": float("inf")},
    {"qty": -5.0},
    {"margin": "not-a-number"},
])
def test_reserve_rejects_nonfinite_negative_or_nonnumeric_amounts(field_kwargs):
    led = _led()
    kwargs = {"qty": 10.0, "margin": 1.0, "notional": 1.0}
    kwargs.update(field_kwargs)
    with pytest.raises(ValueError):
        led.reserve("o1", "BUY", "AAPL", False,
                    kwargs["margin"], kwargs["notional"], qty=kwargs["qty"], now=_T0)
    assert len(led) == 0  # nothing partially inserted


@pytest.mark.parametrize("bad_id", ["", "   ", None])
def test_reserve_rejects_empty_order_id(bad_id):
    led = _led()
    with pytest.raises(ValueError):
        led.reserve(bad_id, "BUY", "AAPL", False, 1.0, 1.0, qty=1.0, now=_T0)


@pytest.mark.parametrize("bad_reducing", [0, 1, "true", "false", None])
def test_reserve_rejects_non_bool_reducing(bad_reducing):
    """A non-bool reducing must not silently mis-classify direction."""
    led = _led()
    with pytest.raises(ValueError, match="reducing"):
        led.reserve("o1", "BUY", "AAPL", bad_reducing, 1.0, 1.0, qty=1.0, now=_T0)


def test_reserve_rejects_empty_symbol_and_side():
    led = _led()
    with pytest.raises(ValueError):
        led.reserve("o1", "BUY", "  ", False, 1.0, 1.0, qty=1.0, now=_T0)
    with pytest.raises(ValueError):
        led.reserve("o1", "", "AAPL", False, 1.0, 1.0, qty=1.0, now=_T0)


@pytest.mark.parametrize("bad_ttl", [0.0, -1.0, float("nan"), float("inf")])
def test_construct_rejects_bad_ttl(bad_ttl):
    with pytest.raises(ValueError):
        PendingExposureLedger(ttl_seconds=bad_ttl)


def test_zero_amount_is_valid_not_rejected():
    """A legitimately-zero margin (e.g. a flat-reducing order) is usable, not a
    validation error — mirrors the provider's zero-is-real doctrine."""
    led = _led()
    led.reserve("o1", "SELL", "AAPL", True, 0.0, 0.0, qty=0.0, now=_T0)
    assert led.is_reserved("o1") is True
    assert led.total_reserved().margin == 0.0


# ---------------------------------------------------------------------------
# expire(): reconcile vs broker truth, NEVER blind-release
# ---------------------------------------------------------------------------
def test_expire_releases_past_ttl_only_when_broker_confirms_gone():
    led = _led(ttl=30.0)
    _reserve_increasing(led, "o1", now=_T0)
    # Past TTL, broker truth says it is NOT open anymore → released.
    released = led.expire(now=_T0 + 31.0, broker_open_order_ids=set())
    assert released == ["o1"]
    assert led.is_reserved("o1") is False


def test_expire_does_not_blind_release_entry_still_in_broker_truth():
    """The leaked-reservation guard: past TTL but broker says STILL live → keep.

    A missed lifecycle event must not free a reservation the broker still holds
    as a working order."""
    led = _led(ttl=30.0)
    _reserve_increasing(led, "o1", now=_T0)
    released = led.expire(now=_T0 + 31.0, broker_open_order_ids={"o1"})
    assert released == []
    assert led.is_reserved("o1") is True  # NOT blind-released


def test_expire_with_none_truth_releases_nothing():
    """broker truth UNAVAILABLE (None) → release nothing, even past TTL."""
    led = _led(ttl=30.0)
    _reserve_increasing(led, "o1", now=_T0)
    released = led.expire(now=_T0 + 999.0, broker_open_order_ids=None)
    assert released == []
    assert led.is_reserved("o1") is True


def test_expire_empty_set_is_authoritative_zero_open():
    """Empty set != None: an empty set is a valid truth meaning 'zero open
    orders' and DOES release past-TTL entries."""
    led = _led(ttl=30.0)
    _reserve_increasing(led, "o1", now=_T0)
    _reserve_increasing(led, "o2", now=_T0)
    released = led.expire(now=_T0 + 31.0, broker_open_order_ids=frozenset())
    assert released == ["o1", "o2"]
    assert len(led) == 0


def test_expire_keeps_not_yet_past_ttl_even_if_broker_says_gone():
    """Too soon to reconcile: within TTL, keep the reservation regardless of
    broker truth (the terminal event should still arrive)."""
    led = _led(ttl=30.0)
    _reserve_increasing(led, "o1", now=_T0)
    released = led.expire(now=_T0 + 10.0, broker_open_order_ids=set())
    assert released == []
    assert led.is_reserved("o1") is True


def test_expire_exactly_at_ttl_is_not_past():
    led = _led(ttl=30.0)
    _reserve_increasing(led, "o1", now=_T0)
    released = led.expire(now=_T0 + 30.0, broker_open_order_ids=set())
    assert released == []  # age == ttl is not > ttl


def test_expire_mixed_batch_releases_only_confirmed_gone_past_ttl():
    led = _led(ttl=30.0)
    _reserve_increasing(led, "gone", now=_T0)      # past ttl + not in truth
    _reserve_increasing(led, "still", now=_T0)     # past ttl + in truth
    _reserve_increasing(led, "young", now=_T0 + 20.0)  # not past ttl
    released = led.expire(now=_T0 + 31.0, broker_open_order_ids={"still"})
    assert released == ["gone"]
    assert led.is_reserved("gone") is False
    assert led.is_reserved("still") is True
    assert led.is_reserved("young") is True


# ---------------------------------------------------------------------------
# rebuild_from_broker: full replace + restart-rebuild flag
# ---------------------------------------------------------------------------
def test_rebuild_flag_starts_false():
    assert _led().rebuilt is False


def test_rebuild_from_broker_replaces_state():
    led = _led()
    _reserve_increasing(led, "stale1")  # pre-restart junk that must be discarded
    _reserve_increasing(led, "stale2")
    open_orders = [
        OpenOrder("b1", "BUY", "AAPL", False, 500.0, 1000.0, 10.0),
        OpenOrder("b2", "SELL", "MSFT", True, 0.0, 800.0, 100.0),
    ]
    count = led.rebuild_from_broker(open_orders, now=_T0)
    assert count == 2
    assert led.rebuilt is True
    assert led.is_reserved("stale1") is False  # old state gone
    assert led.is_reserved("stale2") is False
    assert led.is_reserved("b1") is True
    assert led.is_reserved("b2") is True
    assert led.pending_reducing_qty("MSFT") == pytest.approx(100.0)
    assert led.total_reserved().notional == pytest.approx(1800.0)


def test_rebuild_from_broker_accepts_mapping_specs():
    led = _led()
    open_orders = [
        {"order_id": "b1", "side": "BUY", "symbol": "AAPL",
         "reducing": False, "margin": 100.0, "notional": 500.0, "qty": 5.0},
    ]
    assert led.rebuild_from_broker(open_orders, now=_T0) == 1
    res = led.get("b1")
    assert res is not None
    assert res.reserved_at_epoch == _T0  # TTL restarts fresh at now


def test_rebuild_stamps_all_entries_at_now_so_ttl_restarts():
    led = _led(ttl=30.0)
    open_orders = [OpenOrder("b1", "BUY", "AAPL", False, 1.0, 2.0, 3.0)]
    led.rebuild_from_broker(open_orders, now=_T0)
    # Immediately after rebuild, nothing is past TTL.
    assert led.expire(now=_T0 + 10.0, broker_open_order_ids=set()) == []
    assert led.is_reserved("b1") is True


def test_rebuild_empty_open_orders_yields_empty_but_rebuilt():
    led = _led()
    _reserve_increasing(led, "stale")
    assert led.rebuild_from_broker([], now=_T0) == 0
    assert len(led) == 0
    assert led.rebuilt is True  # "rebuilt to empty" distinguishable from never


def test_rebuild_rejects_duplicate_order_id_in_input():
    """A valid spec BEFORE the duplicate must NOT survive the aborted rebuild —
    binds against a non-atomic rebuild that would leave the first 'b1' inserted."""
    led = _led()
    dup = [
        OpenOrder("b1", "BUY", "AAPL", False, 1.0, 2.0, 3.0),   # valid, first
        OpenOrder("b1", "SELL", "AAPL", True, 1.0, 2.0, 3.0),   # duplicate id
    ]
    with pytest.raises(ValueError, match="duplicate order_id"):
        led.rebuild_from_broker(dup, now=_T0)
    # No partial insertion of the valid first spec; rebuild never committed.
    assert led.is_reserved("b1") is False
    assert len(led) == 0
    assert led.rebuilt is False


def test_rebuild_rejects_missing_field():
    """A valid spec BEFORE the malformed one must NOT survive — same atomicity
    guard, via a missing required field rather than a duplicate id."""
    led = _led()
    bad = [
        {"order_id": "good", "side": "BUY", "symbol": "AAPL",
         "reducing": False, "margin": 1.0, "notional": 2.0, "qty": 3.0},  # valid
        {"order_id": "b1", "side": "BUY", "symbol": "AAPL",
         "reducing": False, "margin": 1.0, "notional": 2.0},  # no qty
    ]
    with pytest.raises(ValueError):
        led.rebuild_from_broker(bad, now=_T0)
    assert led.is_reserved("good") is False  # valid-first spec discarded
    assert len(led) == 0
    assert led.rebuilt is False


def test_rebuild_failure_is_atomic_leaves_prior_state_intact():
    """A failed rebuild is transactional: last-good state and the rebuilt flag
    survive unchanged, never a partially-reconstructed (under-counting) ledger."""
    led = _led()
    # Establish a good prior state via a successful rebuild.
    led.rebuild_from_broker([OpenOrder("keep", "BUY", "AAPL", False, 500.0, 1234.0, 7.0)], now=_T0)
    assert led.rebuilt is True
    before = led.total_reserved()
    # Now a rebuild that fails partway (valid spec, then a bad amount).
    bad = [
        OpenOrder("n1", "BUY", "MSFT", False, 10.0, 20.0, 1.0),
        OpenOrder("n2", "BUY", "MSFT", False, -1.0, 20.0, 1.0),  # negative margin
    ]
    with pytest.raises(ValueError):
        led.rebuild_from_broker(bad, now=_T0 + 5.0)
    # Prior state fully intact; neither new spec leaked in.
    assert led.is_reserved("keep") is True
    assert led.is_reserved("n1") is False
    assert led.is_reserved("n2") is False
    assert led.total_reserved() == before
    assert led.rebuilt is True  # prior success flag unchanged


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def _apply_ops(led):
    _reserve_increasing(led, "i1", "AAPL", qty=10.0, margin=500.0, notional=1000.0)
    _reserve_reducing(led, "r1", "AAPL", qty=100.0, margin=0.0, notional=900.0)
    _reserve_reducing(led, "r2", "MSFT", qty=50.0, margin=0.0, notional=400.0)
    led.release("i1")
    _reserve_increasing(led, "i2", "TSLA", qty=3.0, margin=200.0, notional=700.0)


def test_same_ops_yield_same_state():
    a = _led()
    b = _led()
    _apply_ops(a)
    _apply_ops(b)
    # Same live ids, same aggregates, same reducing bound → identical state.
    assert a.live_order_ids() == b.live_order_ids()
    assert a.total_reserved() == b.total_reserved()
    assert a.pending_reducing_qty("AAPL") == b.pending_reducing_qty("AAPL")
    assert a.pending_reducing_qty("MSFT") == b.pending_reducing_qty("MSFT")
    assert [r.to_dict() for r in a.live_reservations()] == \
        [r.to_dict() for r in b.live_reservations()]


def test_live_reservations_preserve_insertion_order():
    led = _led()
    _reserve_increasing(led, "first")
    _reserve_increasing(led, "second")
    _reserve_increasing(led, "third")
    assert led.live_order_ids() == ("first", "second", "third")


# ---------------------------------------------------------------------------
# isolation: NO live/execution/broker import, NO order method
# ---------------------------------------------------------------------------
_FORBIDDEN = ("place_order", "placeorder", "submit", "cancel", "allow", "block", "decide")


def _public_names(obj):
    return [n for n in dir(obj) if not n.startswith("_")]


def test_ledger_exposes_no_order_or_decision_method():
    for name in _public_names(PendingExposureLedger):
        low = name.lower()
        assert all(tok not in low for tok in _FORBIDDEN), name
    for attr in ("place_order", "placeOrder", "submit", "cancel", "allow", "block", "decide"):
        assert not hasattr(PendingExposureLedger, attr)


def test_module_does_not_import_live_or_execution_paths():
    import chad.risk.pending_exposure_ledger as mod

    src = mod.__file__
    assert src is not None
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    for banned in ("live_loop", "execution", "ibkr_adapter", "placeOrder",
                   "place_order", "orchestrator", "import ib_async", "import ib_insync"):
        assert banned not in text, banned

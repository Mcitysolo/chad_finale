"""WKF U1 — guard books on CONFIRMED FILL only, never on submission.

Covers the four required invariants:
  * submission (PreSubmitted / Submitted) does NOT create a guard entry;
  * a confirmed fill DOES create a guard entry;
  * duplicate_open_order NEVER creates a guard entry;
  * the one-time boot reconcile clears phantoms and emits GUARD_PHANTOM_RECONCILED.

The live_loop execution path books the guard with exactly two gates, in order:
  1. `_should_persist_paper_evidence(order, ...)` — an unconfirmed/duplicate
     adapter status returns (False, ...) and the loop `continue`s BEFORE the
     guard-open block is reached;
  2. `is_fill_confirmed(open_evidence)` — even if the block is reached, only a
     trusted paper_fill/filled + fill_id (not pnl_untrusted) opens the guard.
These tests exercise both gates plus the real `mark_position_open` action so the
decision→action mapping the loop performs is verified end-to-end, and the pure
`reconcile_boot_phantoms` sweep + its live_loop wrapper.

All disk I/O is redirected into tmp_path (repo write-guard compliant).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chad.core import position_guard
from chad.core.position_guard import (
    is_fill_confirmed,
    mark_position_open,
    reconcile_boot_phantoms,
)
from chad.core.live_loop import _should_persist_paper_evidence


def _order(status: str):
    return SimpleNamespace(status=status, symbol="BAC", side="BUY", quantity=10.0)


def _intent(strategy="alpha", symbol="BAC", side="BUY", qty=10.0):
    return SimpleNamespace(strategy=strategy, symbol=symbol, side=side,
                           quantity=qty, meta={})


def _open_ev(status: str, fill_id: str = "", pnl_untrusted: bool = False):
    return {
        "fill_id": fill_id,
        "status": status,
        "pnl_untrusted": pnl_untrusted,
        "reject": False,
        "tags": [],
        "extra": {},
    }


# ---------------------------------------------------------------------------
# Test 1 — submission does NOT create a guard entry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["presubmitted", "submitted", "pendingsubmit"])
def test_submission_status_does_not_open_guard(status, tmp_path, monkeypatch):
    guard_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)

    # Gate 1: an unconfirmed submission is skipped before the guard-open block.
    persist, reason = _should_persist_paper_evidence(_order(status), {})
    assert persist is False
    assert reason and "unconfirmed_order_status" in reason

    # Gate 2 (defence in depth): even the evidence dict fails is_fill_confirmed.
    assert is_fill_confirmed(_open_ev(status, fill_id="x1")) is False

    # Faithfully replay the loop's decision: skip => no mark_position_open.
    if persist and is_fill_confirmed(_open_ev(status, fill_id="x1")):
        mark_position_open(_intent())
    assert not guard_path.exists() or json.loads(guard_path.read_text()) == {}


# ---------------------------------------------------------------------------
# Test 2 — a confirmed fill DOES create a guard entry
# ---------------------------------------------------------------------------

def test_confirmed_fill_opens_guard(tmp_path, monkeypatch):
    guard_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)

    persist, reason = _should_persist_paper_evidence(_order("paper_fill"), {})
    assert persist is True and reason is None

    ev = _open_ev("paper_fill", fill_id="f-123")
    assert is_fill_confirmed(ev) is True

    # Replay the loop's decision: persist + confirmed => mark_position_open.
    if persist and is_fill_confirmed(ev):
        mark_position_open(_intent())

    state = json.loads(guard_path.read_text())
    assert state["alpha|BAC"]["open"] is True
    assert state["alpha|BAC"]["side"] == "BUY"
    assert state["alpha|BAC"]["quantity"] == 10.0


# ---------------------------------------------------------------------------
# Test 3 — duplicate_open_order NEVER creates a guard entry
# ---------------------------------------------------------------------------

def test_duplicate_open_order_never_opens_guard(tmp_path, monkeypatch):
    guard_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)

    for status in ("duplicate_open_order", "duplicate_blocked",
                   "suppressed_open_orders_cap"):
        persist, reason = _should_persist_paper_evidence(_order(status), {})
        assert persist is False, f"{status} must not persist evidence"
        assert is_fill_confirmed(_open_ev(status, fill_id="d1")) is False
        if persist and is_fill_confirmed(_open_ev(status, fill_id="d1")):
            mark_position_open(_intent())

    assert not guard_path.exists() or json.loads(guard_path.read_text()) == {}


# ---------------------------------------------------------------------------
# Test 4a — reconcile_boot_phantoms (pure) clears phantoms, retains the rest
# ---------------------------------------------------------------------------

def test_reconcile_boot_phantoms_pure():
    state = {
        # phantom: broker holds nothing, no FIFO backing → CLEARED
        "alpha|BAC": {"open": True, "strategy": "alpha", "symbol": "BAC",
                      "side": "BUY", "quantity": 10.0, "last_state": "OPEN"},
        # broker-held: TLT is in broker_symbols → RETAINED
        "beta|TLT": {"open": True, "strategy": "beta", "symbol": "TLT",
                     "side": "BUY", "quantity": 5.0, "last_state": "OPEN"},
        # fill-backed: gamma|SPY has a FIFO lot → RETAINED
        "gamma|SPY": {"open": True, "strategy": "gamma", "symbol": "SPY",
                      "side": "BUY", "quantity": 3.0, "last_state": "OPEN"},
        # broker_sync entries are never touched
        "broker_sync|TLT": {"open": True, "strategy": "broker_sync",
                            "symbol": "TLT", "side": "BUY", "quantity": 420.0},
        "_version": 123, "_written_by": "test",
    }
    new_state, cleared = reconcile_boot_phantoms(
        state,
        broker_symbols={"TLT"},
        fill_backed_keys={"gamma|SPY"},
    )
    assert cleared == ["alpha|BAC"]
    assert new_state["alpha|BAC"]["open"] is False
    assert new_state["alpha|BAC"]["closed_by"] == "phantom_boot_reconcile"
    assert new_state["alpha|BAC"]["last_state"] == "CLOSED"
    assert new_state["beta|TLT"]["open"] is True, "broker-held entry retained"
    assert new_state["gamma|SPY"]["open"] is True, "fill-backed entry retained"
    assert new_state["broker_sync|TLT"]["open"] is True, "broker_sync untouched"
    # purity: input state not mutated
    assert state["alpha|BAC"]["open"] is True


def test_reconcile_boot_phantoms_empty_and_closed():
    assert reconcile_boot_phantoms({}, set(), set()) == ({}, [])
    # an already-closed phantom is not re-touched (no double-close)
    state = {"alpha|BAC": {"open": False, "strategy": "alpha", "symbol": "BAC",
                           "side": "BUY", "quantity": 10.0}}
    new_state, cleared = reconcile_boot_phantoms(state, set(), set())
    assert cleared == []
    assert new_state["alpha|BAC"]["open"] is False


# ---------------------------------------------------------------------------
# Test 4b — the live_loop boot wrapper emits the marker + is one-shot
# ---------------------------------------------------------------------------

def test_boot_reconcile_wrapper_clears_and_marks(tmp_path, monkeypatch, caplog):
    import logging
    from chad.core import live_loop

    guard_path = tmp_path / "position_guard.json"
    tc_path = tmp_path / "trade_closer_state.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(live_loop, "_TRADE_CLOSER_STATE_PATH", tc_path)
    monkeypatch.setattr(live_loop, "_BOOT_PHANTOM_RECONCILE_DONE", False)

    guard_path.write_text(json.dumps({
        "alpha|BAC": {"open": True, "strategy": "alpha", "symbol": "BAC",
                      "side": "BUY", "quantity": 10.0, "last_state": "OPEN"},
        "gamma|SPY": {"open": True, "strategy": "gamma", "symbol": "SPY",
                      "side": "BUY", "quantity": 3.0, "last_state": "OPEN"},
        "broker_sync|TLT": {"open": True, "strategy": "broker_sync",
                            "symbol": "TLT", "side": "BUY", "quantity": 420.0},
        "beta|TLT": {"open": True, "strategy": "beta", "symbol": "TLT",
                     "side": "BUY", "quantity": 5.0, "last_state": "OPEN"},
    }), encoding="utf-8")
    tc_path.write_text(json.dumps({
        "queues": [{"strategy": "gamma", "symbol": "SPY", "lots": [
            {"side": "BUY", "quantity": 3.0, "fill_id": "s1"}]}],
        "processed_fill_ids": ["s1"],
    }), encoding="utf-8")

    logger = logging.getLogger("wkf.u1.test")
    with caplog.at_level("WARNING"):
        live_loop._reconcile_boot_phantoms(logger)

    assert any("GUARD_PHANTOM_RECONCILED key=alpha|BAC" in m for m in caplog.messages)

    state = json.loads(guard_path.read_text())
    assert state["alpha|BAC"]["open"] is False, "phantom cleared"
    assert state["gamma|SPY"]["open"] is True, "fill-backed retained"
    assert state["beta|TLT"]["open"] is True, "broker-held retained"
    assert state["broker_sync|TLT"]["open"] is True, "broker_sync untouched"

    # One-shot: the flag is now set; a second call is a no-op even if a new
    # phantom appears.
    guard_path.write_text(json.dumps({
        "delta|GME": {"open": True, "strategy": "delta", "symbol": "GME",
                      "side": "BUY", "quantity": 1.0, "last_state": "OPEN"},
    }), encoding="utf-8")
    caplog.clear()
    live_loop._reconcile_boot_phantoms(logger)
    assert not any("GUARD_PHANTOM_RECONCILED" in m for m in caplog.messages)
    assert json.loads(guard_path.read_text())["delta|GME"]["open"] is True

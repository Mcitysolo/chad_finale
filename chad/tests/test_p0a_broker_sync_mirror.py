"""P0A-A6 — broker_sync must mirror CURRENT broker truth, never trade history.

INCIDENT-0713 root cause: _rebuild_guard_from_broker marked a broker_sync entry
open=False when the broker went flat but RETAINED its historical quantity/side
(broker_sync|TLT = SELL 1340 against a flat broker). The drift reader
detect_guard_vs_broker_drift_v2 reads broker_sync `quantity` REGARDLESS of the
`open` flag, so it read broker=-1340 — a phantom short — long after the position
was covered flat.

The fix makes broker_sync always mirror the current broker position: held
symbols carry current qty+sign; flat symbols are forced to quantity 0 (even if
the entry was already closed with a stale value).
"""

from __future__ import annotations

import copy
import logging

import pytest

from chad.core import live_loop
from chad.core import position_guard
from chad.core.position_guard import detect_guard_vs_broker_drift_v2


class _Pos:
    def __init__(self, quantity):
        self.quantity = quantity


@pytest.fixture
def guard_env(monkeypatch):
    """In-memory guard state + patchable broker positions; no disk, no broker."""
    state: dict = {}

    # Mirror the real disk semantics: _load_state returns a FRESH object each
    # call, save_state persists a distinct snapshot (not aliased to `state`).
    monkeypatch.setattr(position_guard, "_load_state", lambda: copy.deepcopy(state))

    def _save(s):
        snapshot = copy.deepcopy(s)
        state.clear()
        state.update(snapshot)

    monkeypatch.setattr(position_guard, "save_state", _save)

    def set_broker(positions: dict):
        monkeypatch.setattr(
            live_loop.position_sync, "fetch_positions", lambda: dict(positions)
        )

    logger = logging.getLogger("test.p0a_a6")
    return state, set_broker, logger


def test_broker_sync_mirrors_flip_long_flat_short(guard_env):
    state, set_broker, logger = guard_env

    # (1) broker long 700 TLT
    set_broker({"TLT": _Pos(700)})
    live_loop._rebuild_guard_from_broker(logger)
    bs = state["broker_sync|TLT"]
    assert bs["side"] == "BUY" and bs["quantity"] == 700.0 and bs["open"] is True

    # (2) broker flat -> broker_sync must read flat (qty 0), NOT retain 700
    set_broker({})
    live_loop._rebuild_guard_from_broker(logger)
    bs = state["broker_sync|TLT"]
    assert bs["quantity"] == 0.0 and bs["open"] is False

    # (3) broker short 640 -> sign flips to SELL, current qty
    set_broker({"TLT": _Pos(-640)})
    live_loop._rebuild_guard_from_broker(logger)
    bs = state["broker_sync|TLT"]
    assert bs["side"] == "SELL" and bs["quantity"] == 640.0 and bs["open"] is True


def test_repairs_already_closed_stale_entry_incident_0713(guard_env):
    """The EXACT current-runtime shape: broker_sync|TLT already open=False but
    carrying SELL 1340. A rebuild against a flat broker must zero it."""
    state, set_broker, logger = guard_env
    state["broker_sync|TLT"] = {
        "open": False, "strategy": "broker_sync", "symbol": "TLT",
        "side": "SELL", "quantity": 1340.0, "closed_by": "broker_truth_rebuild",
        "source": "paper_ledger_rebuild",
    }
    set_broker({})  # broker is flat on TLT
    live_loop._rebuild_guard_from_broker(logger)

    bs = state["broker_sync|TLT"]
    assert bs["quantity"] == 0.0
    # And the drift reader now sees flat broker truth — the phantom -1340 gone.
    drift = detect_guard_vs_broker_drift_v2(state)
    tlt = [d for d in drift["drifts"] if d["symbol"] == "TLT"]
    assert tlt == [], "TLT must no longer show as broker truth after zeroing"


def test_drift_reader_read_phantom_before_fix_is_gone_after(guard_env):
    """End-to-end: the -1340 phantom the incident produced is eliminated."""
    state, set_broker, logger = guard_env
    # Seed the incident-era stale entry as the drift reader would have seen it.
    state["broker_sync|TLT"] = {
        "open": False, "strategy": "broker_sync", "symbol": "TLT",
        "side": "SELL", "quantity": 1340.0,
    }
    pre = detect_guard_vs_broker_drift_v2(state)
    tlt_pre = [d for d in pre["drifts"] if d["symbol"] == "TLT"]
    assert tlt_pre and tlt_pre[0]["broker_qty"] == -1340.0  # the phantom short

    set_broker({})
    live_loop._rebuild_guard_from_broker(logger)
    post = detect_guard_vs_broker_drift_v2(state)
    assert not [d for d in post["drifts"] if d["symbol"] == "TLT"]


def test_held_symbol_not_zeroed(guard_env):
    """A symbol the broker still holds must NOT be zeroed — qty/sign refreshed."""
    state, set_broker, logger = guard_env
    set_broker({"IWM": _Pos(200), "TLT": _Pos(-640)})
    live_loop._rebuild_guard_from_broker(logger)
    assert state["broker_sync|IWM"]["quantity"] == 200.0
    assert state["broker_sync|IWM"]["side"] == "BUY"
    assert state["broker_sync|TLT"]["quantity"] == 640.0
    assert state["broker_sync|TLT"]["side"] == "SELL"


def test_strategy_entry_quantity_left_intact(guard_env):
    """Non-broker_sync (attribution) entries keep their quantity; only open flips."""
    state, set_broker, logger = guard_env
    state["gamma|TLT"] = {
        "open": True, "strategy": "gamma", "symbol": "TLT",
        "side": "BUY", "quantity": 640.0,
    }
    set_broker({})  # broker flat
    live_loop._rebuild_guard_from_broker(logger)
    g = state["gamma|TLT"]
    assert g["open"] is False
    assert g["quantity"] == 640.0  # attribution history preserved, not zeroed

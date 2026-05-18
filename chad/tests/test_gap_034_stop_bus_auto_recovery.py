"""GAP-034 / Phase-44 — durable stop-bus auto-clear (clean-streak hysteresis).

Verifies the four Phase-42 fix_spec cases:

  (a) latched + 4 clean cycles            → still active.
  (b) 5th clean cycle (count >= N and
      elapsed >= min_clean_seconds)        → auto-cleared,
                                             cleared_by literal startswith
                                             ``auto_recovery:``.
  (c) latched + 3 clean cycles + 1 trip    → counter resets to 0
                                             (no premature clear when
                                             clean cycles resume).
  (d) operator manual clear                → immediate, independent of
                                             the recovery counter.

Time and snapshots are mocked. No real systemd or broker is involved.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.risk import stop_bus_state as sbs
from chad.risk.stop_bus_state import (
    clear_stop_bus,
    evaluate_and_persist,
    is_stop_bus_active,
    read_stop_bus,
    set_stop_bus,
)


@pytest.fixture()
def bus_path(tmp_path: Path) -> Path:
    return tmp_path / "stop_bus.json"


@pytest.fixture()
def recovery_path(tmp_path: Path) -> Path:
    return tmp_path / "stop_bus_recovery_state.json"


@pytest.fixture()
def clock(monkeypatch):
    """Mockable UTC clock for the auto-recovery elapsed-time check."""
    state = {"now": datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)}

    def _now():
        return state["now"]

    monkeypatch.setattr(sbs, "_utc_now_dt", _now)
    return state


def _advance(clock_state: dict, seconds: float) -> None:
    clock_state["now"] = clock_state["now"] + timedelta(seconds=seconds)


_TRIP_SNAPSHOT = {"avg_latency_ms": 10000.0}  # well above 2000ms default
_CLEAN_SNAPSHOT = {"avg_latency_ms": 50.0}    # comfortably below threshold


# ---------------------------------------------------------------------------
# Case (a): 4 clean cycles after latch → bus still active
# ---------------------------------------------------------------------------


def test_a_four_clean_cycles_keep_bus_latched(
    bus_path: Path, recovery_path: Path, clock
):
    # Latch the bus via the trigger path.
    res = evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    assert res["any_active"] is True
    assert is_stop_bus_active(bus_path) is True

    # 4 consecutive clean cycles, each 60s apart.
    for _ in range(4):
        _advance(clock, 60.0)
        res = evaluate_and_persist(
            snapshot=_CLEAN_SNAPSHOT,
            triggered_by="test",
            path=bus_path,
            recovery_state_path=recovery_path,
        )
        assert res["any_active"] is False

    # Bus must still be active — count (4) < required (5).
    assert is_stop_bus_active(bus_path) is True
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 4
    assert rec["schema_version"] == "stop_bus_recovery_state.v1"
    assert rec["ttl_seconds"] == 600


# ---------------------------------------------------------------------------
# Case (b): 5th clean cycle → auto-clear with auto_recovery: prefix
# ---------------------------------------------------------------------------


def test_b_fifth_clean_cycle_auto_clears_bus(
    bus_path: Path, recovery_path: Path, clock
):
    # Latch.
    evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    assert is_stop_bus_active(bus_path) is True

    # 5 consecutive clean cycles, each 60s apart → total elapsed = 300s
    # which exceeds the 240s default min_clean_seconds.
    for _ in range(5):
        _advance(clock, 60.0)
        evaluate_and_persist(
            snapshot=_CLEAN_SNAPSHOT,
            triggered_by="test",
            path=bus_path,
            recovery_state_path=recovery_path,
        )

    # Bus has been auto-cleared.
    assert is_stop_bus_active(bus_path) is False
    record = read_stop_bus(bus_path)
    assert record["active"] is False
    assert record["cleared_by"].startswith("auto_recovery:")
    # Counter is reset post-clear.
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 0


def test_b_min_clean_seconds_blocks_premature_clear(
    bus_path: Path, recovery_path: Path, clock
):
    # Same count as (b) but each clean is only 1 second apart — elapsed
    # never satisfies the 240s gate, so auto-clear must NOT fire.
    evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    for _ in range(10):
        _advance(clock, 1.0)
        evaluate_and_persist(
            snapshot=_CLEAN_SNAPSHOT,
            triggered_by="test",
            path=bus_path,
            recovery_state_path=recovery_path,
        )
    assert is_stop_bus_active(bus_path) is True


# ---------------------------------------------------------------------------
# Case (c): 3 clean + 1 trip → counter resets, no premature clear later
# ---------------------------------------------------------------------------


def test_c_trip_resets_counter_no_premature_clear(
    bus_path: Path, recovery_path: Path, clock
):
    # Latch.
    evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )

    # 3 consecutive clean cycles → counter == 3.
    for _ in range(3):
        _advance(clock, 60.0)
        evaluate_and_persist(
            snapshot=_CLEAN_SNAPSHOT,
            triggered_by="test",
            path=bus_path,
            recovery_state_path=recovery_path,
        )
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 3

    # Trip again — counter must reset to 0, bus remains active.
    _advance(clock, 60.0)
    res = evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    assert res["any_active"] is True
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 0
    assert is_stop_bus_active(bus_path) is True

    # Now 4 more clean cycles (well past the 240s gate from this point):
    # counter walks 1,2,3,4 and must NOT auto-clear (still < required 5).
    for _ in range(4):
        _advance(clock, 60.0)
        evaluate_and_persist(
            snapshot=_CLEAN_SNAPSHOT,
            triggered_by="test",
            path=bus_path,
            recovery_state_path=recovery_path,
        )
    assert is_stop_bus_active(bus_path) is True
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 4


# ---------------------------------------------------------------------------
# Case (d): operator manual clear is immediate, independent of counter
# ---------------------------------------------------------------------------


def test_d_operator_manual_clear_is_immediate_independent_of_counter(
    bus_path: Path, recovery_path: Path, clock
):
    # Latch and accumulate a partial clean streak (counter == 3).
    evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    for _ in range(3):
        _advance(clock, 60.0)
        evaluate_and_persist(
            snapshot=_CLEAN_SNAPSHOT,
            triggered_by="test",
            path=bus_path,
            recovery_state_path=recovery_path,
        )
    assert is_stop_bus_active(bus_path) is True
    rec_before = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec_before["consecutive_clean_evaluations"] == 3

    # Operator manual clear — must be immediate.
    cleared = clear_stop_bus(cleared_by="operator", path=bus_path)
    assert cleared["active"] is False
    assert cleared["cleared_by"] == "operator"
    assert is_stop_bus_active(bus_path) is False

    # Manual clear does NOT touch the recovery state file — that is the
    # spec's "no counter dependency" assertion. The counter only resets
    # on the next evaluate_and_persist tick (when bus is inactive).
    rec_after = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec_after["consecutive_clean_evaluations"] == 3

    # Additionally: even with a non-zero stale counter, a manual clear at
    # any prior counter value still works (re-latch + clear at counter==1).
    evaluate_and_persist(
        snapshot=_TRIP_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    _advance(clock, 60.0)
    evaluate_and_persist(
        snapshot=_CLEAN_SNAPSHOT,
        triggered_by="test",
        path=bus_path,
        recovery_state_path=recovery_path,
    )
    assert is_stop_bus_active(bus_path) is True
    cleared2 = clear_stop_bus(cleared_by="operator", path=bus_path)
    assert cleared2["active"] is False
    assert is_stop_bus_active(bus_path) is False

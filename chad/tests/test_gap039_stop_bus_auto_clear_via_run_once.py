"""GAP-039 / Phase-59 — stop-bus auto-clear reachability via run_once.

Drives chad.core.live_loop.run_once (the FULL path) to prove the
Phase-58/59 reorder makes the GAP-034 auto-clear hysteresis reachable
from a real cycle. GAP-034 added clean-streak hysteresis to
stop_bus_state.evaluate_and_persist and its helper-level tests pass,
but the bus stayed latched in production because run_once's early-
return at the top of the cycle exited BEFORE evaluate_and_persist ran,
so the counter never incremented and the bus never auto-cleared.

The Phase-58/59 fix relocates the evaluate_and_persist block to BEFORE
the is_stop_bus_active() early-return. This test exercises that exact
path — it imports and invokes chad.core.live_loop.run_once and asserts
the auto-clear pathway is reached.

Required cases:
  A — latched bus + clean snapshot → counter increments → cycle 3
      auto-clears AND falls through to guard rebuild → cycle 4 still
      proceeds past the early-return.
  B — latched bus + still-bad snapshot → bus stays latched every
      cycle, triggered_at preserved (NOT re-stamped), recovery counter
      stays 0, guard rebuild NEVER invoked.
  C — bus INACTIVE at start + trigger-firing snapshot → evaluate sets
      bus active this cycle, then is_stop_bus_active early-return
      halts the cycle BEFORE guard rebuild. Proves the reorder did
      not break the primary safety invariant.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

# Importing live_loop triggers a module-level IB connect; the env var
# short-circuits that for tests (see live_loop.py:114-120).
os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

import chad.core.live_loop as live_loop  # noqa: E402
import chad.risk.stop_bus_state as sbs  # noqa: E402


class _ReachedGuardRebuild(BaseException):
    """Sentinel: run_once reached the guard-rebuild stage past the
    stop-bus halt. BaseException-derived so the live_loop guard-rebuild
    try/except (which catches Exception) does not swallow it, allowing
    the test to detect 'cycle proceeded past the early-return'.
    """


@pytest.fixture()
def bus_paths(tmp_path: Path) -> Dict[str, Path]:
    return {
        "bus": tmp_path / "stop_bus.json",
        "recovery": tmp_path / "stop_bus_recovery_state.json",
    }


@pytest.fixture()
def clock(monkeypatch):
    state = {"now": datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)}

    def _now():
        return state["now"]

    monkeypatch.setattr(sbs, "_utc_now_dt", _now)
    return state


def _advance(clock_state: dict, seconds: float) -> None:
    clock_state["now"] = clock_state["now"] + timedelta(seconds=seconds)


@pytest.fixture()
def runonce_harness(monkeypatch, bus_paths):
    """Wire run_once's stop-bus calls + snapshot to tmp_path-backed state,
    stub guard rebuild to a BaseException sentinel, and supply a
    harness-controlled snapshot.

    live_loop does ``from chad.risk.stop_bus_state import ...`` INSIDE
    run_once, which re-reads the module attribute on each call, so
    monkey-patching the module attribute is sufficient to redirect the
    helper to tmp_path-backed paths.
    """
    bus_path = bus_paths["bus"]
    recovery_path = bus_paths["recovery"]

    harness: Dict[str, Any] = {
        "snapshot": {"avg_latency_ms": 50.0},
        "guard_paper_calls": 0,
        "guard_broker_calls": 0,
    }

    # Lower thresholds for deterministic 3-cycle clearing (vs production 5).
    monkeypatch.setattr(
        sbs, "DEFAULT_AUTO_CLEAR_CONSECUTIVE_CLEAN_REQUIRED", 3, raising=True
    )
    monkeypatch.setattr(
        sbs, "DEFAULT_AUTO_CLEAR_MIN_CLEAN_SECONDS", 0, raising=True
    )

    _orig_eval = sbs.evaluate_and_persist
    _orig_is_active = sbs.is_stop_bus_active
    _orig_read = sbs.read_stop_bus

    def _eval_wrapped(snapshot, config=None, triggered_by="live_loop",
                      path=None, recovery_state_path=None):
        return _orig_eval(
            snapshot=snapshot, config=config, triggered_by=triggered_by,
            path=bus_path, recovery_state_path=recovery_path,
        )

    def _is_active_wrapped(path=None):
        return _orig_is_active(path=bus_path)

    def _read_wrapped(path=None):
        return _orig_read(path=bus_path)

    monkeypatch.setattr(sbs, "evaluate_and_persist", _eval_wrapped)
    monkeypatch.setattr(sbs, "is_stop_bus_active", _is_active_wrapped)
    monkeypatch.setattr(sbs, "read_stop_bus", _read_wrapped)

    def _snap_stub(_logger):
        return dict(harness["snapshot"])

    monkeypatch.setattr(live_loop, "_build_stop_bus_snapshot", _snap_stub)
    monkeypatch.setattr(live_loop, "_is_paper_mode", lambda: True)

    def _stub_rebuild_paper(_logger):
        harness["guard_paper_calls"] += 1
        raise _ReachedGuardRebuild()

    def _stub_rebuild_broker(_logger):
        harness["guard_broker_calls"] += 1
        raise _ReachedGuardRebuild()

    monkeypatch.setattr(
        live_loop, "_rebuild_guard_from_paper_ledger", _stub_rebuild_paper
    )
    monkeypatch.setattr(
        live_loop, "_rebuild_guard_from_broker", _stub_rebuild_broker
    )

    return harness


def _logger() -> logging.Logger:
    return logging.getLogger("gap039_test")


def _pre_latch_bus(bus_path: Path, triggered_at_iso: str) -> None:
    payload = {
        "schema_version": sbs.STOP_BUS_SCHEMA_VERSION,
        "active": True,
        "reason": "broker_latency_spike: avg_latency_ms=10000",
        "triggered_at": triggered_at_iso,
        "triggered_by": "test_preseed",
        "cleared_at": None,
        "cleared_by": "",
    }
    bus_path.parent.mkdir(parents=True, exist_ok=True)
    bus_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Case A — latched + clean snapshot → counter walks 1,2,3 → cycle 3 clears
# AND proceeds past early-return; cycle 4 still proceeds.
# ---------------------------------------------------------------------------


def test_case_a_auto_clear_reachable_via_run_once(
    runonce_harness, bus_paths, clock
):
    bus_path = bus_paths["bus"]
    recovery_path = bus_paths["recovery"]
    runonce_harness["snapshot"] = {"avg_latency_ms": 50.0}

    pre_latch_iso = clock["now"].isoformat()
    _pre_latch_bus(bus_path, triggered_at_iso=pre_latch_iso)
    assert sbs.read_stop_bus(bus_path)["active"] is True

    # Cycle 1: counter→1, bus stays latched.
    _advance(clock, 60.0)
    live_loop.run_once(_logger())
    assert runonce_harness["guard_paper_calls"] == 0
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 1
    assert sbs.read_stop_bus(bus_path)["active"] is True

    # Cycle 2: counter→2, bus stays latched.
    _advance(clock, 60.0)
    live_loop.run_once(_logger())
    assert runonce_harness["guard_paper_calls"] == 0
    rec = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert rec["consecutive_clean_evaluations"] == 2
    assert sbs.read_stop_bus(bus_path)["active"] is True

    # Cycle 3: counter→3 == required, bus auto-clears INSIDE
    # evaluate_and_persist, then is_stop_bus_active=False so run_once
    # falls through to guard rebuild → sentinel raised.
    _advance(clock, 60.0)
    with pytest.raises(_ReachedGuardRebuild):
        live_loop.run_once(_logger())
    assert runonce_harness["guard_paper_calls"] == 1
    cleared = sbs.read_stop_bus(bus_path)
    assert cleared["active"] is False
    assert cleared["cleared_by"].startswith(
        "auto_recovery:broker_latency_clean_streak="
    )

    # Cycle 4: bus already cleared → falls through again → sentinel.
    # Confirms subsequent cycles continue to proceed past the early-return.
    _advance(clock, 60.0)
    with pytest.raises(_ReachedGuardRebuild):
        live_loop.run_once(_logger())
    assert runonce_harness["guard_paper_calls"] == 2


# ---------------------------------------------------------------------------
# Case B — latched + bad snapshot → bus stays latched, triggered_at
# preserved, counter stays 0, guard rebuild NEVER invoked.
# ---------------------------------------------------------------------------


def test_case_b_bad_snapshot_keeps_bus_latched_via_run_once(
    runonce_harness, bus_paths, clock
):
    bus_path = bus_paths["bus"]
    recovery_path = bus_paths["recovery"]
    runonce_harness["snapshot"] = {"avg_latency_ms": 10000.0}

    pre_latch_iso = clock["now"].isoformat()
    _pre_latch_bus(bus_path, triggered_at_iso=pre_latch_iso)
    original_triggered_at = sbs.read_stop_bus(bus_path)["triggered_at"]
    assert original_triggered_at == pre_latch_iso

    for _ in range(4):
        _advance(clock, 60.0)
        live_loop.run_once(_logger())  # No sentinel — cycle halts every time.

    assert runonce_harness["guard_paper_calls"] == 0
    assert runonce_harness["guard_broker_calls"] == 0

    state = sbs.read_stop_bus(bus_path)
    assert state["active"] is True
    # triggered_at preserved — set_stop_bus's idempotent path retains
    # the original timestamp when bus is re-set while already active.
    assert state["triggered_at"] == original_triggered_at

    # any_active=True every cycle → _reset_recovery_counter_if_nonzero
    # keeps the counter at 0 (idempotent reset).
    if recovery_path.is_file():
        rec = json.loads(recovery_path.read_text(encoding="utf-8"))
        assert int(rec["consecutive_clean_evaluations"]) == 0


# ---------------------------------------------------------------------------
# Case C (MANDATORY) — bus INACTIVE + fresh trigger → evaluate sets bus
# active, is_stop_bus_active early-return halts cycle BEFORE guard
# rebuild. Proves the reorder preserves the safety invariant that a
# newly-firing trigger stops the cycle it fires on.
# ---------------------------------------------------------------------------


def test_case_c_fresh_trigger_still_halts_cycle_via_run_once(
    runonce_harness, bus_paths, clock
):
    bus_path = bus_paths["bus"]
    runonce_harness["snapshot"] = {"avg_latency_ms": 10000.0}

    assert not bus_path.exists()
    assert sbs.read_stop_bus(bus_path)["active"] is False

    _advance(clock, 60.0)
    # No sentinel raised — cycle halts at is_stop_bus_active() after
    # evaluate_and_persist sets the bus active in the same cycle.
    live_loop.run_once(_logger())

    assert runonce_harness["guard_paper_calls"] == 0
    assert runonce_harness["guard_broker_calls"] == 0

    state = sbs.read_stop_bus(bus_path)
    assert state["active"] is True
    assert state["triggered_by"] == "live_loop.run_once"
    assert "latency" in state["reason"].lower()

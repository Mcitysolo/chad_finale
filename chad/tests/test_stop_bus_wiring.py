"""Tests for Phase-8 Session 4 STOP bus persistence + wiring (R2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_stop_bus_initially_inactive(bus_path: Path):
    assert is_stop_bus_active(bus_path) is False
    record = read_stop_bus(bus_path)
    assert record["active"] is False


def test_set_stop_bus_writes_active_record(bus_path: Path):
    state = set_stop_bus("daily_loss:pnl=-6000", triggered_by="test", path=bus_path)
    assert state["active"] is True
    assert "daily_loss" in state["reason"]
    assert state["triggered_by"] == "test"
    assert state["triggered_at"]
    assert is_stop_bus_active(bus_path) is True
    # File exists and is valid JSON.
    data = json.loads(bus_path.read_text(encoding="utf-8"))
    assert data["active"] is True


def test_clear_stop_bus_resumes_trading(bus_path: Path):
    set_stop_bus("test_trigger", path=bus_path)
    assert is_stop_bus_active(bus_path) is True
    state = clear_stop_bus(cleared_by="operator", path=bus_path)
    assert state["active"] is False
    assert state["cleared_by"] == "operator"
    assert is_stop_bus_active(bus_path) is False


def test_evaluate_and_persist_writes_on_active_trigger(bus_path: Path):
    snapshot = {
        "realized_pnl": -6000.0,
        "daily_loss_limit": 5000.0,
    }
    result = evaluate_and_persist(snapshot, triggered_by="test", path=bus_path)
    assert result["any_active"] is True
    assert "daily_loss" in result["active_triggers"]
    assert is_stop_bus_active(bus_path) is True


def test_evaluate_and_persist_no_op_when_clean(bus_path: Path):
    snapshot = {"realized_pnl": 100.0, "daily_loss_limit": 5000.0}
    result = evaluate_and_persist(snapshot, path=bus_path)
    assert result["any_active"] is False
    # File not created when no trigger fires and no prior state existed.
    assert is_stop_bus_active(bus_path) is False


def test_read_stop_bus_tolerates_missing_file(bus_path: Path):
    # Path deliberately does not exist.
    record = read_stop_bus(bus_path)
    assert record["active"] is False
    assert record["reason"] == ""


def test_read_stop_bus_tolerates_malformed_file(bus_path: Path):
    bus_path.write_text("{not valid json", encoding="utf-8")
    record = read_stop_bus(bus_path)
    assert record["active"] is False
    # Reason flags the cause so operators can diagnose.
    assert "unreadable" in record["reason"]


def test_set_stop_bus_preserves_triggered_at_on_reentry(bus_path: Path):
    first = set_stop_bus("r1", path=bus_path)
    triggered_at = first["triggered_at"]
    # Second trigger while active should preserve the first triggered_at.
    second = set_stop_bus("r2", path=bus_path)
    assert second["triggered_at"] == triggered_at


def test_execution_pipeline_halts_intent_building_when_bus_active(bus_path: Path, monkeypatch):
    """Pre-submit guard in execution_pipeline must return [] when active."""
    from chad.execution import execution_pipeline as ep

    # Point the module's stop-bus reader at our tmp path.
    monkeypatch.setattr(ep, "_stop_bus_active", lambda: True)

    from chad.execution.execution_pipeline import (
        build_ibkr_intents_from_plan,
        ExecutionPlan,
    )
    plan = ExecutionPlan(orders=[])
    result = build_ibkr_intents_from_plan(plan)
    assert result == []


def test_execution_pipeline_allows_intents_when_bus_inactive(monkeypatch):
    from chad.execution import execution_pipeline as ep
    monkeypatch.setattr(ep, "_stop_bus_active", lambda: False)

    from chad.execution.execution_pipeline import (
        build_ibkr_intents_from_plan,
        ExecutionPlan,
    )
    plan = ExecutionPlan(orders=[])
    # Empty plan in, empty list out — but not short-circuited by STOP.
    result = build_ibkr_intents_from_plan(plan)
    assert result == []


# ---------------------------------------------------------------------------
# Fix A activation: live_loop snapshot/config carry the hysteresis inputs
# ---------------------------------------------------------------------------


def test_live_loop_snapshot_carries_hysteresis_fields(tmp_path: Path):
    import logging
    from chad.core.live_loop import _build_stop_bus_snapshot

    status = tmp_path / "ibkr_status.json"
    status.write_text(json.dumps({
        "latency_ms": 8221.0,
        "consecutive_cycles_above_stop_threshold": 715,
        "last_above_threshold_at": "2026-05-28T17:34:00Z",
        "breach_streak_started_at": "2026-05-28T04:28:00Z",
    }), encoding="utf-8")

    snap = _build_stop_bus_snapshot(logging.getLogger("test"), ibkr_status_path=status)
    assert snap["avg_latency_ms"] == 8221.0
    assert snap["consecutive_cycles_above_stop_threshold"] == 715
    assert snap["last_above_threshold_at"] == "2026-05-28T17:34:00Z"
    assert snap["breach_streak_started_at"] == "2026-05-28T04:28:00Z"


def test_live_loop_stop_bus_config_carries_hysteresis_keys():
    from chad.core.live_loop import _build_stop_bus_config

    cfg = _build_stop_bus_config()
    assert cfg["broker_latency_trip_consecutive_required"] == 5
    assert cfg["broker_latency_trip_min_breach_seconds"] == 60.0
    assert cfg["broker_latency_trip_hysteresis_enabled"] is True


def test_live_loop_stop_bus_config_env_override(monkeypatch):
    from chad.core.live_loop import _build_stop_bus_config

    monkeypatch.setenv("CHAD_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED", "9")
    monkeypatch.setenv("CHAD_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS", "120")
    monkeypatch.setenv("CHAD_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED", "0")
    cfg = _build_stop_bus_config()
    assert cfg["broker_latency_trip_consecutive_required"] == 9
    assert cfg["broker_latency_trip_min_breach_seconds"] == 120.0
    assert cfg["broker_latency_trip_hysteresis_enabled"] is False


def test_live_loop_snapshot_resilient_to_missing_artifact(tmp_path: Path):
    import logging
    from chad.core.live_loop import _build_stop_bus_snapshot

    missing = tmp_path / "does_not_exist.json"
    # Must not raise, and must not carry hysteresis keys -> aggregator falls
    # back to legacy single-cycle behaviour (counter=None).
    snap = _build_stop_bus_snapshot(logging.getLogger("test"), ibkr_status_path=missing)
    assert isinstance(snap, dict)
    assert "avg_latency_ms" not in snap
    assert "consecutive_cycles_above_stop_threshold" not in snap
    assert "breach_streak_started_at" not in snap


def test_live_loop_snapshot_old_format_artifact_falls_back_to_legacy(tmp_path: Path):
    import logging
    from chad.core.live_loop import _build_stop_bus_snapshot

    # Pre-activation artifact: latency present, hysteresis fields absent.
    status = tmp_path / "ibkr_status.json"
    status.write_text(json.dumps({"latency_ms": 3000.0}), encoding="utf-8")
    snap = _build_stop_bus_snapshot(logging.getLogger("test"), ibkr_status_path=status)
    assert snap["avg_latency_ms"] == 3000.0
    assert "consecutive_cycles_above_stop_threshold" not in snap
    assert "breach_streak_started_at" not in snap


def test_live_loop_snapshot_prefers_api_ms_over_connect_polluted_latency(tmp_path: Path):
    import logging
    from chad.core.live_loop import _build_stop_bus_snapshot

    # SSOT: connect_ms != api_ms. latency_ms includes the one-shot connect cost
    # (connect_ms + api_ms) so it false-trips the broker_latency STOP_BUS on a
    # fresh-connect cycle. The snapshot must select api_ms (true round-trip).
    status = tmp_path / "ibkr_status.json"
    status.write_text(json.dumps({
        "api_ms": 0.3,
        "connect_ms": 8259.7,
        "latency_ms": 8260.0,
    }), encoding="utf-8")
    snap = _build_stop_bus_snapshot(logging.getLogger("test"), ibkr_status_path=status)
    assert snap["avg_latency_ms"] == 0.3


def test_live_loop_snapshot_falls_back_to_latency_ms_when_api_ms_absent(tmp_path: Path):
    import logging
    from chad.core.live_loop import _build_stop_bus_snapshot

    # api_ms is None only on connect failure (latency_ms == connect attempt);
    # the fallback chain must degrade to latency_ms so the trigger still sees a
    # value rather than dropping the field entirely.
    status = tmp_path / "ibkr_status.json"
    status.write_text(json.dumps({"latency_ms": 8260.0}), encoding="utf-8")
    snap = _build_stop_bus_snapshot(logging.getLogger("test"), ibkr_status_path=status)
    assert snap["avg_latency_ms"] == 8260.0

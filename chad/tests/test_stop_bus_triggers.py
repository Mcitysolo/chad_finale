"""Tests for Phase-8 Session 3 STOP bus triggers (R2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chad.risk.stop_bus_triggers import (
    DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED,
    DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS,
    check_broker_latency,
    check_daily_loss_limit,
    check_data_staleness,
    check_reject_rate,
    evaluate_all_stop_triggers,
)


def _iso_z_seconds_ago(seconds: float) -> str:
    """Produce a UTC ISO-8601 'Z'-suffixed timestamp ``seconds`` in the past.

    Matches the exact format ibkr_reliability_tracker emits for
    ``last_above_threshold_at`` (strftime '%Y-%m-%dT%H:%M:%SZ').
    """
    dt = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# daily_loss
# ---------------------------------------------------------------------------


def test_daily_loss_triggers_when_pnl_below_negative_limit():
    r = check_daily_loss_limit(realized_pnl=-6000.0, daily_loss_limit=5000.0)
    assert r.active is True
    assert "realized_pnl" in r.reason


def test_daily_loss_no_trigger_when_within_limit():
    r = check_daily_loss_limit(realized_pnl=-4500.0, daily_loss_limit=5000.0)
    assert r.active is False


def test_daily_loss_accepts_positive_or_negative_limit_value():
    # Passing the limit as a negative number should work the same way.
    r = check_daily_loss_limit(realized_pnl=-6000.0, daily_loss_limit=-5000.0)
    assert r.active is True


# ---------------------------------------------------------------------------
# reject_rate
# ---------------------------------------------------------------------------


def test_reject_rate_triggers_above_threshold():
    r = check_reject_rate(reject_count=5, total_count=10, threshold=0.3, min_samples=10)
    assert r.active is True
    assert r.details["rate"] == pytest.approx(0.5)


def test_reject_rate_no_trigger_below_threshold():
    r = check_reject_rate(reject_count=2, total_count=10, threshold=0.3, min_samples=10)
    assert r.active is False


def test_reject_rate_needs_min_samples():
    r = check_reject_rate(reject_count=3, total_count=4, threshold=0.3, min_samples=10)
    assert r.active is False
    assert "insufficient_samples" in r.reason


# ---------------------------------------------------------------------------
# data_staleness
# ---------------------------------------------------------------------------


def test_data_staleness_triggers_when_gate_rejects_dominate():
    r = check_data_staleness(
        gate_reject_count=9, total_intent_count=10,
        threshold=0.8, min_samples=5,
    )
    assert r.active is True


def test_data_staleness_no_trigger_when_below_threshold():
    r = check_data_staleness(
        gate_reject_count=3, total_intent_count=10,
        threshold=0.8, min_samples=5,
    )
    assert r.active is False


# ---------------------------------------------------------------------------
# broker_latency
# ---------------------------------------------------------------------------


def test_broker_latency_triggers_above_threshold():
    r = check_broker_latency(avg_latency_ms=3000.0, threshold_ms=2000.0)
    assert r.active is True


def test_broker_latency_no_trigger_below_threshold():
    r = check_broker_latency(avg_latency_ms=500.0, threshold_ms=2000.0)
    assert r.active is False


# ---------------------------------------------------------------------------
# evaluate_all_stop_triggers (aggregator)
# ---------------------------------------------------------------------------


def test_evaluate_all_returns_active_triggers():
    snap = {
        "realized_pnl": -6000.0,
        "daily_loss_limit": 5000.0,
        "reject_count": 5,
        "total_order_count": 10,
        "gate_reject_count": 1,
        "total_intent_count": 10,
        "avg_latency_ms": 500.0,
    }
    result = evaluate_all_stop_triggers(snap)
    assert result["any_active"] is True
    active_names = {r.name for r in result["active_triggers"]}
    # daily_loss and reject_rate breach; staleness and latency do not.
    assert "daily_loss" in active_names
    assert "reject_rate" in active_names
    assert "data_staleness" not in active_names
    assert "broker_latency" not in active_names


def test_evaluate_all_empty_snapshot_returns_no_active():
    result = evaluate_all_stop_triggers({})
    assert result["any_active"] is False
    assert result["active_triggers"] == []


def test_evaluate_all_never_raises_on_garbage_input():
    # Deliberately busted values.
    snap = {"realized_pnl": "bad", "daily_loss_limit": "also_bad"}
    # Must not raise.
    result = evaluate_all_stop_triggers(snap)
    assert result["any_active"] is False


# ---------------------------------------------------------------------------
# broker_latency — Fix A: symmetric hysteresis on the trip side
# ---------------------------------------------------------------------------


class TestBrokerLatencyHysteresis:
    """Fix A: the trip side requires a SUSTAINED breach (count + wall-clock
    gates), mirroring the auto-clear discipline on the release side. Legacy
    single-cycle behaviour is preserved for un-migrated callers."""

    def test_legacy_single_cycle_when_hysteresis_disabled(self):
        # hysteresis_enabled=False -> single above-threshold cycle trips.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=1,
            last_above_threshold_at=_iso_z_seconds_ago(0),
            hysteresis_enabled=False,
        )
        assert r.active is True
        assert "avg_latency_ms" in r.reason

    def test_legacy_single_cycle_when_counter_not_passed(self):
        # consecutive_cycles_above=None (default) -> legacy behaviour even with
        # hysteresis enabled. Protects callers that have not migrated.
        r = check_broker_latency(avg_latency_ms=3000.0, threshold_ms=2000.0)
        assert r.active is True

    def test_single_cycle_spike_does_not_trip_with_hysteresis(self):
        # Today's 18:12Z incident: one 2962ms-class cycle must NOT halt.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=1,
            last_above_threshold_at=_iso_z_seconds_ago(0),
            hysteresis_enabled=True,
        )
        assert r.active is False
        assert "transient" in r.reason or "below_trip_streak" in r.reason

    def test_4_cycle_breach_does_not_trip(self):
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=4,
            last_above_threshold_at=_iso_z_seconds_ago(120),
            trip_consecutive_required=5,
            hysteresis_enabled=True,
        )
        assert r.active is False

    def test_5_cycle_sustained_breach_trips(self):
        # Count gate (5/5) AND time gate (120s > 60s default) both pass.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(120),
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is True
        assert "sustained_latency" in r.reason
        assert r.details["consecutive_cycles_above"] == 5

    def test_count_met_but_time_gate_fails(self):
        # Count gate passes (5/5) but breach only 10s old < 60s floor.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(10),
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is False

    def test_recovery_during_climb_does_not_trip(self):
        # First: climbing (3 cycles, above). Second: recovered (counter reset).
        first = check_broker_latency(
            avg_latency_ms=2500.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=3,
            last_above_threshold_at=_iso_z_seconds_ago(40),
            trip_consecutive_required=5,
            hysteresis_enabled=True,
        )
        second = check_broker_latency(
            avg_latency_ms=600.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=0,
            last_above_threshold_at=_iso_z_seconds_ago(40),
            trip_consecutive_required=5,
            hysteresis_enabled=True,
        )
        assert first.active is False
        assert second.active is False
        assert second.reason == "ok"

    def test_hard_wedge_scenario_today_morning(self):
        # Today's 04:28-17:34Z hard wedge: 715 consecutive cycles at 8221ms.
        # MUST still trip under Fix A.
        r = check_broker_latency(
            avg_latency_ms=8221.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=715,
            last_above_threshold_at=_iso_z_seconds_ago(3600),
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is True
        assert "sustained_latency" in r.reason

    def test_invalid_latency_returns_inactive(self):
        r = check_broker_latency(
            avg_latency_ms="not-a-number",
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(120),
            hysteresis_enabled=True,
        )
        assert r.active is False
        assert r.reason == "invalid_latency"

    def test_invalid_threshold_returns_inactive(self):
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=0.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(120),
            hysteresis_enabled=True,
        )
        assert r.active is False
        assert r.reason == "invalid_threshold"

    def test_invalid_timestamp_blocks_time_gate(self):
        # Count gate passes but the timestamp is unparseable; the time gate
        # must fail closed so the trigger stays inactive.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at="garbage",
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is False

    def test_defaults_are_symmetric_with_clear_side_count(self):
        # The trip count gate default mirrors the auto-clear clean-streak (5).
        assert DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED == 5
        assert DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS == 60.0

    def test_breach_streak_started_at_preferred_over_last_above(self):
        # breach anchor is old enough (61s); last_above is ~now. If the trigger
        # used last_above it would NOT trip (time gate ~0s). Using the breach
        # anchor it trips — proving breach_streak_started_at takes precedence.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(0),
            breach_streak_started_at=_iso_z_seconds_ago(61),
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is True
        assert "sustained_latency" in r.reason

    def test_falls_back_to_last_above_when_breach_streak_missing(self):
        # breach_streak_started_at absent -> fall back to last_above (f3ab3d8
        # behaviour preserved).
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(61),
            breach_streak_started_at=None,
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is True

    def test_breach_streak_recent_does_not_trip_even_if_last_above_old(self):
        # Inverse precedence check: breach anchor is fresh (10s) so the trip is
        # suppressed even though last_above is stale (120s). Confirms the breach
        # anchor — not last_above — drives the time gate.
        r = check_broker_latency(
            avg_latency_ms=3000.0,
            threshold_ms=2000.0,
            consecutive_cycles_above=5,
            last_above_threshold_at=_iso_z_seconds_ago(120),
            breach_streak_started_at=_iso_z_seconds_ago(10),
            trip_consecutive_required=5,
            trip_min_breach_seconds=60.0,
            hysteresis_enabled=True,
        )
        assert r.active is False

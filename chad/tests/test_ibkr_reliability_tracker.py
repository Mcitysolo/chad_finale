"""Unit tests for chad/ops/ibkr_reliability_tracker.py (STOP-BUS-RECOVERY-1)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from chad.ops import ibkr_reliability_tracker as tr

NOW = datetime(2026, 5, 27, 20, 0, 0, tzinfo=timezone.utc)


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_single_spike_does_not_increment_above_one():
    prev = {"latency_ms": 500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 0, "client_id": 9001}
    cur = {"latency_ms": 2500, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 1
    assert fields.current_recovery_state == tr.RECOVERY_DEGRADING


def test_sustained_latency_increments_counter():
    prev = {"latency_ms": 2500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 3, "client_id": 9001}
    cur = {"latency_ms": 2700, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 4
    assert fields.current_recovery_state == tr.RECOVERY_ABOVE


def test_clean_cycle_resets_counter_to_zero():
    prev = {"latency_ms": 2500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 4, "client_id": 9001}
    cur = {"latency_ms": 600, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 0
    assert fields.current_recovery_state == tr.RECOVERY_RECOVERING


def test_healthy_steady_state():
    prev = {"latency_ms": 500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 0, "client_id": 9001}
    cur = {"latency_ms": 480, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 0
    assert fields.current_recovery_state == tr.RECOVERY_HEALTHY


def test_alert_fires_at_threshold_only():
    payload_below = {"consecutive_cycles_above_stop_threshold": 4}
    payload_at = {"consecutive_cycles_above_stop_threshold": 5}
    payload_well_above = {"consecutive_cycles_above_stop_threshold": 8,
                          "max_latency_observed_in_window": 3500.0,
                          "current_recovery_state": tr.RECOVERY_ABOVE,
                          "last_above_threshold_at": _ts(NOW),
                          "last_gateway_churn_at": None,
                          "ts_utc": _ts(NOW)}

    fire, _ = tr.should_alert(payload_below)
    assert not fire

    fire, alert = tr.should_alert(payload_at)
    assert fire
    assert alert["rule"] == "ibkr_reliability.sustained_latency_above_threshold"
    assert alert["consecutive_cycles_above_stop_threshold"] == 5

    fire, alert = tr.should_alert(payload_well_above)
    assert fire
    assert alert["max_latency_observed_in_window"] == 3500.0
    assert alert["current_recovery_state"] == tr.RECOVERY_ABOVE


def test_max_latency_in_window_tracked_and_decayed_outside_window():
    # Previous max inside the window — kept.
    prev_in_window = {
        "latency_ms": 2500,
        "ts_utc": _ts(NOW - timedelta(seconds=60)),
        "consecutive_cycles_above_stop_threshold": 1,
        "max_latency_observed_in_window": 3200.0,
        "client_id": 9001,
    }
    cur = {"latency_ms": 1800, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev_in_window, cur, now=NOW)
    assert fields.max_latency_observed_in_window == 3200.0

    # Previous max outside the window — reset.
    prev_outside = {
        "latency_ms": 2500,
        "ts_utc": _ts(NOW - timedelta(hours=2)),
        "consecutive_cycles_above_stop_threshold": 0,
        "max_latency_observed_in_window": 3200.0,
        "client_id": 9001,
    }
    fields2 = tr.compute_reliability_fields(prev_outside, cur, now=NOW)
    assert fields2.max_latency_observed_in_window == 1800.0


def test_gateway_churn_detected_on_client_id_flip():
    prev = {"latency_ms": 500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 0, "client_id": 9001,
            "last_gateway_churn_at": None}
    cur = {"latency_ms": 500, "client_id": 9002}  # client_id flipped
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.last_gateway_churn_at is not None


def test_no_churn_when_client_id_unchanged():
    prev = {"latency_ms": 500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 0, "client_id": 9001,
            "last_gateway_churn_at": None}
    cur = {"latency_ms": 500, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.last_gateway_churn_at is None


def test_merge_into_payload_returns_new_dict_does_not_mutate_input(tmp_path):
    prev_path = tmp_path / "ibkr_status.json"
    prev_path.write_text(
        '{"latency_ms": 500, "ts_utc": "2026-05-27T19:59:00Z", '
        '"consecutive_cycles_above_stop_threshold": 0, "client_id": 9001}'
    )
    original = {"latency_ms": 2500, "client_id": 9001}
    merged = tr.merge_reliability_into_payload(dict(original), prev_path, now=NOW)
    assert merged["consecutive_cycles_above_stop_threshold"] == 1
    assert "consecutive_cycles_above_stop_threshold" not in original


def test_alert_payload_contains_required_fields():
    payload = {
        "consecutive_cycles_above_stop_threshold": 5,
        "max_latency_observed_in_window": 3000.0,
        "current_recovery_state": tr.RECOVERY_ABOVE,
        "last_above_threshold_at": _ts(NOW),
        "last_gateway_churn_at": None,
        "ts_utc": _ts(NOW),
    }
    fire, alert = tr.should_alert(payload)
    assert fire
    required = {
        "rule",
        "consecutive_cycles_above_stop_threshold",
        "stop_threshold_ms",
        "alert_at_consecutive",
        "max_latency_observed_in_window",
        "current_recovery_state",
        "last_above_threshold_at",
        "last_gateway_churn_at",
        "ts_utc",
    }
    assert required.issubset(alert.keys())


def test_health_rule_emits_finding_when_counter_crosses_threshold(monkeypatch, tmp_path):
    from chad.ops import health_monitor_rules as hmr

    status_path = tmp_path / "ibkr_status.json"
    status_path.write_text(
        '{"ok": false, "latency_ms": 3000, '
        '"consecutive_cycles_above_stop_threshold": 6, '
        '"current_recovery_state": "above_threshold", '
        '"max_latency_observed_in_window": 3500, '
        '"last_above_threshold_at": "2026-05-27T20:00:00Z", '
        '"last_gateway_churn_at": null, '
        '"ts_utc": "2026-05-27T20:01:00Z"}'
    )
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    findings: list = []
    hmr.rule_ibkr_sustained_latency(findings)
    r19 = [f for f in findings if f.rule_id == "R19"]
    assert r19, "rule did not emit R19 finding"
    assert r19[0].severity == "CRITICAL"


def test_health_rule_no_finding_below_threshold(monkeypatch, tmp_path):
    from chad.ops import health_monitor_rules as hmr

    status_path = tmp_path / "ibkr_status.json"
    status_path.write_text(
        '{"ok": true, "latency_ms": 600, '
        '"consecutive_cycles_above_stop_threshold": 1, '
        '"current_recovery_state": "degrading", '
        '"ts_utc": "2026-05-27T20:01:00Z"}'
    )
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    findings: list = []
    hmr.rule_ibkr_sustained_latency(findings)
    r19 = [f for f in findings if f.rule_id == "R19"]
    assert not r19


# ---------------------------------------------------------------------------
# Fix A activation: breach_streak_started_at (stamp-once streak anchor)
# ---------------------------------------------------------------------------


def test_breach_streak_started_at_stamped_on_first_above():
    # Streak begins (counter 0 -> 1): anchor stamped to now.
    prev = {"latency_ms": 500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 0, "client_id": 9001}
    cur = {"latency_ms": 2500, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 1
    assert fields.breach_streak_started_at == _ts(NOW)
    assert fields.as_payload()["breach_streak_started_at"] == _ts(NOW)


def test_breach_streak_started_at_preserved_across_cycles():
    # Streak continues: anchor preserved, NOT re-stamped.
    anchor = "2026-01-01T00:00:00Z"
    prev = {"latency_ms": 2500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 3,
            "breach_streak_started_at": anchor, "client_id": 9001}
    cur = {"latency_ms": 2700, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 4
    assert fields.breach_streak_started_at == anchor


def test_breach_streak_started_at_cleared_on_recovery():
    # Streak ends (cur below threshold): anchor cleared to "".
    prev = {"latency_ms": 2500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 5,
            "breach_streak_started_at": "2026-01-01T00:00:00Z", "client_id": 9001}
    cur = {"latency_ms": 600, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.consecutive_cycles_above_stop_threshold == 0
    assert fields.breach_streak_started_at == ""


def test_last_above_threshold_at_still_restamps_for_compat():
    # last_above_threshold_at must keep updating every above cycle (compat),
    # while breach_streak_started_at stays pinned to the streak start.
    anchor = "2026-01-01T00:00:00Z"
    prev = {"latency_ms": 2500,
            "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 3,
            "last_above_threshold_at": _ts(NOW - timedelta(seconds=70)),
            "breach_streak_started_at": anchor, "client_id": 9001}
    cur = {"latency_ms": 2700, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.last_above_threshold_at == _ts(NOW)   # re-stamped
    assert fields.breach_streak_started_at == anchor    # preserved


def test_breach_streak_started_at_fallback_when_missing_prev():
    # Mid-streak but the field is absent from prev (e.g. first cycle after the
    # tracker upgrade): fall back to stamping now rather than leaving it blank.
    prev = {"latency_ms": 2500, "ts_utc": _ts(NOW - timedelta(seconds=60)),
            "consecutive_cycles_above_stop_threshold": 2, "client_id": 9001}
    cur = {"latency_ms": 2600, "client_id": 9001}
    fields = tr.compute_reliability_fields(prev, cur, now=NOW)
    assert fields.breach_streak_started_at == _ts(NOW)

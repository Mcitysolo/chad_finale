"""Tests for Phase-8 Session 3 STOP bus triggers (R2)."""

from __future__ import annotations

import pytest

from chad.risk.stop_bus_triggers import (
    check_broker_latency,
    check_daily_loss_limit,
    check_data_staleness,
    check_reject_rate,
    evaluate_all_stop_triggers,
)


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

"""Tests for Phase-8 Session 7 (S5) EventCalendar + event_risk_gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.analytics.event_calendar import (
    ACTION_PASS,
    ACTION_REDUCE,
    ACTION_REJECT,
    EventCalendar,
)
from chad.execution.routing_gates import event_risk_gate, run_all_gates


def _write_calendar(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "event_calendar.v1", "events": events}),
        encoding="utf-8",
    )


def _mk_calendar(tmp_path: Path, events) -> EventCalendar:
    path = tmp_path / "event_calendar.json"
    _write_calendar(path, events)
    return EventCalendar(calendar_path=path)


@dataclass
class _FakeIntent:
    """Mutable stand-in so event_risk_gate can halve its quantity in place."""
    symbol: str = "SPY"
    quantity: float = 100.0
    notional_estimate: float = 50000.0
    order_urgency: str = "normal"


def test_event_within_window_suppresses(tmp_path: Path):
    now = datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc)
    # FOMC at 2026-05-07 18:00 UTC, 24h window — now is 12h before.
    cal = _mk_calendar(
        tmp_path,
        [{"date": "2026-05-07", "time": "18:00", "name": "FOMC", "suppress_hours": 24}],
    )
    verdict = cal.get_suppression(symbol="SPY", urgency="normal", now=now)
    assert verdict["suppress"] is True
    assert verdict["action"] == ACTION_REDUCE
    assert verdict["event_name"] == "FOMC"


def test_event_outside_window_passes(tmp_path: Path):
    now = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    cal = _mk_calendar(
        tmp_path,
        [{"date": "2026-05-07", "time": "18:00", "name": "FOMC", "suppress_hours": 24}],
    )
    verdict = cal.get_suppression(symbol="SPY", urgency="normal", now=now)
    assert verdict["suppress"] is False
    assert verdict["action"] == ACTION_PASS


def test_high_urgency_rejects(tmp_path: Path):
    now = datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc)
    cal = _mk_calendar(
        tmp_path,
        [{"date": "2026-05-07", "time": "18:00", "name": "FOMC", "suppress_hours": 24}],
    )
    verdict = cal.get_suppression(symbol="SPY", urgency="high", now=now)
    assert verdict["action"] == ACTION_REJECT


def test_normal_urgency_reduces_50pct_via_gate(tmp_path: Path):
    now = datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc)
    cal = _mk_calendar(
        tmp_path,
        [{"date": "2026-05-07", "time": "18:00", "name": "FOMC", "suppress_hours": 24}],
    )
    intent = _FakeIntent(symbol="SPY", quantity=100.0, notional_estimate=50000.0,
                         order_urgency="normal")
    passed, reason = event_risk_gate(intent, calendar=cal, now=now)
    assert passed is True
    assert intent.quantity == pytest.approx(50.0)
    assert intent.notional_estimate == pytest.approx(25000.0)
    assert "reduce_50pct" in reason


def test_no_events_passes_all(tmp_path: Path):
    cal = _mk_calendar(tmp_path, [])
    verdict = cal.get_suppression(symbol="SPY", urgency="high")
    assert verdict["suppress"] is False
    assert verdict["action"] == ACTION_PASS


def test_symbol_scoped_event_only_applies_to_symbol(tmp_path: Path):
    now = datetime(2026, 5, 7, 6, 0, tzinfo=timezone.utc)
    cal = _mk_calendar(
        tmp_path,
        [
            {
                "date": "2026-05-07",
                "time": "18:00",
                "name": "AAPL Earnings",
                "type": "earnings",
                "symbol": "AAPL",
                "suppress_hours": 24,
            }
        ],
    )
    v_aapl = cal.get_suppression(symbol="AAPL", urgency="high", now=now)
    v_spy = cal.get_suppression(symbol="SPY", urgency="high", now=now)
    assert v_aapl["suppress"] is True
    assert v_spy["suppress"] is False


def test_past_events_are_ignored(tmp_path: Path):
    now = datetime(2026, 5, 8, 0, 0, tzinfo=timezone.utc)
    cal = _mk_calendar(
        tmp_path,
        [{"date": "2026-05-07", "time": "18:00", "name": "FOMC", "suppress_hours": 24}],
    )
    # now is 6h AFTER the event → not upcoming.
    verdict = cal.get_suppression(symbol="SPY", urgency="high", now=now)
    assert verdict["action"] == ACTION_PASS


def test_event_risk_gate_without_calendar_passes():
    intent = _FakeIntent()
    passed, reason = event_risk_gate(intent, calendar=None)
    assert passed is True
    assert reason == "ok"


def test_run_all_gates_includes_gate_5(tmp_path: Path):
    # High urgency + inside window → gate 5 rejects.
    # `now` is dynamic so `stale_intent_gate` (which uses wall-clock UTC and
    # is not given the test's `now` by run_all_gates) doesn't reject the
    # intent before gate 5 runs. The FOMC event is placed 12h ahead so the
    # 24h suppression window still contains `now`.
    now = datetime.now(timezone.utc).replace(microsecond=0)
    event_dt = now + timedelta(hours=12)
    cal = _mk_calendar(
        tmp_path,
        [{
            "date": event_dt.date().isoformat(),
            "time": event_dt.strftime("%H:%M"),
            "name": "FOMC",
            "suppress_hours": 24,
        }],
    )

    @dataclass
    class FullIntent:
        symbol: str = "SPY"
        strategy: str = "alpha"
        side: str = "BUY"
        order_type: str = "LMT"
        quantity: float = 100.0
        notional_estimate: float = 50000.0
        limit_price: float = 500.0
        expected_pnl: float = 0.0
        created_at: str = ""
        ttl_seconds: int = 300
        order_urgency: str = "high"

    intent = FullIntent(created_at=now.isoformat())
    passed, reason = run_all_gates(
        intent=intent,
        bar_timestamp=None,
        current_price=500.0,
        event_calendar=cal,
        now=now,
    )
    assert passed is False
    assert reason.startswith("event_risk:")


def test_malformed_events_skipped(tmp_path: Path):
    cal = _mk_calendar(
        tmp_path,
        [
            {"date": "not-a-date", "name": "Garbage"},
            "not-even-a-dict",
            {"date": "2026-05-07", "time": "18:00", "name": "FOMC"},
        ],
    )
    now = datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc)
    verdict = cal.get_suppression(symbol="SPY", urgency="normal", now=now)
    assert verdict["event_name"] == "FOMC"

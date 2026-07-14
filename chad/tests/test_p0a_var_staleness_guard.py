"""P0A-A4 — VaR staleness guard for metrics_server + health_monitor freshness.

metrics_server exported chad_var_status_ok=1 purely from var_state.json's
`status` field, ignoring age. runtime/var_state.json has been frozen at
ts_utc=2026-05-07 (publisher ops/var_publisher.py has no systemd runner), so a
~66-day-stale VaR was scraped as a healthy "ok" VaR for months.

These tests assert the TTL guard: past TTL, chad_var_status_ok=0 +
chad_var_stale=1, and var_state.json is present in the health_monitor
freshness map.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from chad.ops import metrics_server as ms


def _lines_to_map(lines):
    return {ml.name: ml.value for ml in lines}


def _write_var_state(tmp_path, ts_utc: str, status: str = "ok"):
    p = tmp_path / "var_state.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "var_state.v1",
                "status": status,
                "ts_utc": ts_utc,
                "var_95_1day_usd": 2325.94,
                "var_99_1day_usd": 3289.63,
                "var_pct_of_equity": 1.2663,
                "symbol_count": 4,
            }
        ),
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Pure staleness helper
# ---------------------------------------------------------------------------

def test_compute_staleness_fresh():
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    obj = {"ts_utc": "2026-07-13T23:00:00Z"}  # 1h old
    age, stale = ms._compute_state_staleness(obj, now=now, ttl_s=86400)
    assert stale is False
    assert 3000 < age < 4000


def test_compute_staleness_stale_66_days():
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    obj = {"ts_utc": "2026-05-07T13:39:30Z"}  # the real frozen value
    age, stale = ms._compute_state_staleness(obj, now=now, ttl_s=86400)
    assert stale is True
    assert age > 60 * 86400  # >60 days


def test_compute_staleness_missing_ts_is_failclosed_stale():
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    age, stale = ms._compute_state_staleness({}, now=now, ttl_s=86400)
    assert age is None and stale is True


# ---------------------------------------------------------------------------
# Metric export end-to-end (66-day fixture)
# ---------------------------------------------------------------------------

def test_stale_var_exports_status_not_ok_and_stale_flag(tmp_path, monkeypatch):
    _write_var_state(tmp_path, "2026-05-07T13:39:30Z", status="ok")
    monkeypatch.setattr(ms, "RUNTIME_VAR_STATE_PATH", tmp_path / "var_state.json")
    monkeypatch.setattr(ms, "RUNTIME_DRAWDOWN_STATE_PATH", tmp_path / "missing_dd.json")

    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    m = _lines_to_map(ms._var_drawdown_lines(now=now))

    # Even though status=="ok", the 66-day age forces status_ok=0 + stale=1.
    assert m["chad_var_status_ok"] == 0.0
    assert m["chad_var_stale"] == 1.0
    assert m["chad_var_age_seconds"] > 60 * 86400
    assert m["chad_var_ttl_seconds"] == 86400.0


def test_fresh_var_exports_status_ok_and_not_stale(tmp_path, monkeypatch):
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    fresh_ts = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_var_state(tmp_path, fresh_ts, status="ok")
    monkeypatch.setattr(ms, "RUNTIME_VAR_STATE_PATH", tmp_path / "var_state.json")
    monkeypatch.setattr(ms, "RUNTIME_DRAWDOWN_STATE_PATH", tmp_path / "missing_dd.json")

    m = _lines_to_map(ms._var_drawdown_lines(now=now))
    assert m["chad_var_status_ok"] == 1.0
    assert m["chad_var_stale"] == 0.0


def test_missing_var_state_not_flagged_stale_but_status_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "RUNTIME_VAR_STATE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(ms, "RUNTIME_DRAWDOWN_STATE_PATH", tmp_path / "missing_dd.json")
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    m = _lines_to_map(ms._var_drawdown_lines(now=now))
    # Missing file: status_ok=0 (reported via that gauge, not as "stale=1").
    assert m["chad_var_status_ok"] == 0.0
    assert m["chad_var_stale"] == 0.0
    assert m["chad_var_age_seconds"] == -1.0


def test_env_override_ttl(tmp_path, monkeypatch):
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    # 2h old — fresh under default 24h, stale under a 1h override.
    two_h_old = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_var_state(tmp_path, two_h_old, status="ok")
    monkeypatch.setattr(ms, "RUNTIME_VAR_STATE_PATH", tmp_path / "var_state.json")
    monkeypatch.setattr(ms, "RUNTIME_DRAWDOWN_STATE_PATH", tmp_path / "missing_dd.json")

    monkeypatch.delenv("CHAD_VAR_STATE_TTL_SECONDS", raising=False)
    m = _lines_to_map(ms._var_drawdown_lines(now=now))
    assert m["chad_var_stale"] == 0.0  # default 24h -> fresh

    monkeypatch.setenv("CHAD_VAR_STATE_TTL_SECONDS", "3600")
    m = _lines_to_map(ms._var_drawdown_lines(now=now))
    assert m["chad_var_stale"] == 1.0  # 1h TTL -> 2h old is stale


# ---------------------------------------------------------------------------
# health_monitor freshness map wiring
# ---------------------------------------------------------------------------

def test_var_state_in_health_monitor_freshness_map():
    import inspect

    from chad.ops import health_monitor_rules as hmr

    # var_state is NOTIFY_ONLY (no scheduled publisher to auto-restart).
    assert "var_state.json" in hmr._FEED_NOTIFY_ONLY
    # and present in the feed-freshness rule's TTL table.
    src = inspect.getsource(hmr.rule_feed_freshness)
    assert "var_state.json" in src


def test_var_state_stale_emits_notify_only_finding(tmp_path, monkeypatch):
    from chad.ops import health_monitor_rules as hmr

    # Point RUNTIME at a tmp dir with a var_state.json whose mtime is ancient.
    (tmp_path / "var_state.json").write_text("{}", encoding="utf-8")
    import os
    old = datetime(2026, 5, 7, tzinfo=timezone.utc).timestamp()
    os.utime(tmp_path / "var_state.json", (old, old))
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)

    findings = []
    hmr.rule_feed_freshness(findings)
    var_findings = [f for f in findings if "var_state.json" in f.title]
    assert var_findings, "expected a staleness finding for var_state.json"
    assert var_findings[0].remedy_type == "NOTIFY_ONLY"

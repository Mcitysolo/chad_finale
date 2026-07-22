"""W3B-2 — drawdown staleness guard for metrics_server (A4 pattern, second leg).

P0A-A4 fixed the VaR stale-as-fresh export but left the drawdown gauges
trusting drawdown_state.json's `status`=="ok" with no age check: a frozen
publisher (the P1-3 era ran 10 days like that) would scrape as healthy
forever. These tests assert the mirrored TTL guard: past TTL,
chad_drawdown_status_ok=0 + chad_drawdown_stale=1.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from chad.ops import metrics_server as ms


def _lines_to_map(lines):
    return {ml.name: ml.value for ml in lines}


def _write_dd_state(tmp_path, ts_utc: str, status: str = "ok"):
    p = tmp_path / "drawdown_state.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": "drawdown_state.v1",
                "status": status,
                "ts_utc": ts_utc,
                "ttl_seconds": 300,
                "drawdown_pct": -0.6086,
                "halt_threshold_pct": -15.0,
                "halt": False,
                "enforcement_active": False,
            }
        ),
        encoding="utf-8",
    )
    return p


def _point_paths(tmp_path, monkeypatch, *, dd: bool = True):
    monkeypatch.setattr(ms, "RUNTIME_VAR_STATE_PATH", tmp_path / "missing_var.json")
    monkeypatch.setattr(
        ms,
        "RUNTIME_DRAWDOWN_STATE_PATH",
        tmp_path / ("drawdown_state.json" if dd else "absent_dd.json"),
    )


NOW = datetime(2026, 7, 22, 21, 0, 0, tzinfo=timezone.utc)


def test_stale_drawdown_exports_status_not_ok_and_stale_flag(tmp_path, monkeypatch):
    # 10 days frozen — the P1-3 scenario.
    _write_dd_state(tmp_path, (NOW - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _point_paths(tmp_path, monkeypatch)

    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_drawdown_status_ok"] == 0.0
    assert m["chad_drawdown_stale"] == 1.0
    assert m["chad_drawdown_age_seconds"] > 9 * 86400
    assert m["chad_drawdown_ttl_seconds"] == 900.0
    # value gauges still export the (stale) numbers for forensics
    assert m["chad_drawdown_pct"] == -0.6086


def test_fresh_drawdown_exports_status_ok(tmp_path, monkeypatch):
    _write_dd_state(tmp_path, (NOW - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _point_paths(tmp_path, monkeypatch)

    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_drawdown_status_ok"] == 1.0
    assert m["chad_drawdown_stale"] == 0.0
    assert 0 <= m["chad_drawdown_age_seconds"] <= 120


def test_fresh_but_not_ok_status_stays_not_ok(tmp_path, monkeypatch):
    # Freshness must never launder a bad status.
    _write_dd_state(
        tmp_path,
        (NOW - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        status="error",
    )
    _point_paths(tmp_path, monkeypatch)

    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_drawdown_status_ok"] == 0.0
    assert m["chad_drawdown_stale"] == 0.0


def test_missing_drawdown_state_not_flagged_stale_but_status_zero(tmp_path, monkeypatch):
    _point_paths(tmp_path, monkeypatch, dd=False)

    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_drawdown_status_ok"] == 0.0
    assert m["chad_drawdown_stale"] == 0.0
    assert m["chad_drawdown_age_seconds"] == -1.0


def test_env_override_ttl(tmp_path, monkeypatch):
    # 10 min old — stale under default 900s? No: 600 < 900 -> fresh.
    # Stale under a 300s override.
    _write_dd_state(tmp_path, (NOW - timedelta(seconds=600)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _point_paths(tmp_path, monkeypatch)

    monkeypatch.delenv("CHAD_DRAWDOWN_STATE_TTL_SECONDS", raising=False)
    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_drawdown_stale"] == 0.0

    monkeypatch.setenv("CHAD_DRAWDOWN_STATE_TTL_SECONDS", "300")
    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_drawdown_stale"] == 1.0
    assert m["chad_drawdown_status_ok"] == 0.0


def test_var_guard_unchanged_by_drawdown_guard(tmp_path, monkeypatch):
    # W3B-2 must not perturb the A4 VaR block: missing var_state still
    # exports status_ok=0 / stale=0 / age=-1 exactly as before.
    _write_dd_state(tmp_path, (NOW - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _point_paths(tmp_path, monkeypatch)

    m = _lines_to_map(ms._var_drawdown_lines(now=NOW))
    assert m["chad_var_status_ok"] == 0.0
    assert m["chad_var_stale"] == 0.0
    assert m["chad_var_age_seconds"] == -1.0

"""W5A-5 — DQ2 clock-health gate (EXS10, warn-only per R3).

Never FAIL — only OK/WARN — so it can never page before thresholds are
ratified. Probes broker-vs-box skew + fill-time/sequence monotonicity.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from chad.ops.exterminator_sentinel import ExterminatorSentinel

NOW = datetime(2026, 7, 23, 18, 0, 0, tzinfo=timezone.utc)
REAL_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[2] / "config" / "exterminator.json"


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make(tmp_path, config=None):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / "config" / "exterminator.json"
    cfg_path.write_text(json.dumps(config if config is not None else {
        "clock_health": {"ttl_verified": False, "warn_skew_ms": 2000,
                         "warn_regression_ms": 1000, "max_fill_rows_scanned": 5000}}))
    return ExterminatorSentinel(
        repo_root=tmp_path, runtime_dir=tmp_path / "runtime",
        data_dir=tmp_path / "data", config_path=cfg_path,
        reports_dir=tmp_path / "runtime" / "reports", clock=lambda: NOW,
    )


def _status(tmp_path, *, box=NOW, server=NOW):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime" / "ibkr_status.json").write_text(json.dumps({
        "ts_utc": _iso(box), "server_time_iso": _iso(server)}))


def _fill(fid, seq, ts):
    return {"payload": {"fill_id": fid, "fill_time_utc": _iso(ts),
                        "status": "paper_fill"}, "sequence_id": seq}


def _fills(tmp_path, rows):
    d = tmp_path / "data" / "fills"
    d.mkdir(parents=True, exist_ok=True)
    (d / "FILLS_20260723.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------------------- #
# Config gating
# --------------------------------------------------------------------------- #

def test_unconfigured_warns(tmp_path):
    s = _make(tmp_path, config={})  # no clock_health block
    r = s.check_clock_health()
    assert r.check_id == "EXS10" and r.status == "warn"
    assert "unconfigured" in r.title.lower()


# --------------------------------------------------------------------------- #
# Probe 1: broker-vs-box skew
# --------------------------------------------------------------------------- #

def test_skew_ok_when_aligned(tmp_path):
    _status(tmp_path, box=NOW, server=NOW)
    r = _make(tmp_path).check_clock_health()
    p = next(p for p in r.evidence["probes"] if p["probe"] == "broker_vs_box_skew")
    assert p["status"] == "ok" and p["skew_ms"] == 0.0


def test_skew_warns_above_threshold(tmp_path):
    _status(tmp_path, box=NOW, server=NOW + timedelta(seconds=5))  # 5000ms > 2000
    r = _make(tmp_path).check_clock_health()
    p = next(p for p in r.evidence["probes"] if p["probe"] == "broker_vs_box_skew")
    assert p["status"] == "warn" and p["skew_ms"] == 5000.0
    assert r.status == "warn"


def test_server_time_absent_is_finding(tmp_path):
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runtime" / "ibkr_status.json").write_text(json.dumps({"ts_utc": _iso(NOW)}))
    r = _make(tmp_path).check_clock_health()
    p = next(p for p in r.evidence["probes"] if p["probe"] == "broker_vs_box_skew")
    assert p["status"] == "warn" and p["reason"] == "server_time_absent"


def test_missing_status_file_warns(tmp_path):
    r = _make(tmp_path).check_clock_health()  # no ibkr_status.json
    p = next(p for p in r.evidence["probes"] if p["probe"] == "broker_vs_box_skew")
    assert p["status"] == "warn"


# --------------------------------------------------------------------------- #
# Probe 2: fill-time vs sequence monotonicity
# --------------------------------------------------------------------------- #

def test_monotonic_fills_ok(tmp_path):
    _status(tmp_path)
    _fills(tmp_path, [_fill("a", 1, NOW), _fill("b", 2, NOW + timedelta(seconds=1)),
                      _fill("c", 3, NOW + timedelta(seconds=2))])
    r = _make(tmp_path).check_clock_health()
    p = next(p for p in r.evidence["probes"] if p["probe"] == "fill_time_sequence_monotonic")
    assert p["status"] == "ok"


def test_regression_warns(tmp_path):
    _status(tmp_path)
    # seq 2 carries an earlier fill_time than seq 1 (batch harvest / replay)
    _fills(tmp_path, [_fill("a", 1, NOW), _fill("b", 2, NOW - timedelta(seconds=10))])
    r = _make(tmp_path).check_clock_health()
    p = next(p for p in r.evidence["probes"] if p["probe"] == "fill_time_sequence_monotonic")
    assert p["status"] == "warn"
    assert p["reason"] == "fill_time_regresses_vs_sequence"
    assert p["regressions"][0]["regression_ms"] == 10000.0


def test_no_fills_today_ok(tmp_path):
    _status(tmp_path)
    r = _make(tmp_path).check_clock_health()
    p = next(p for p in r.evidence["probes"] if p["probe"] == "fill_time_sequence_monotonic")
    assert p["status"] == "ok" and p["reason"] == "no_fills_today"


# --------------------------------------------------------------------------- #
# R3: warn-only — never FAIL
# --------------------------------------------------------------------------- #

def test_never_fails_even_with_both_anomalies(tmp_path):
    _status(tmp_path, server=NOW + timedelta(seconds=30))  # huge skew
    _fills(tmp_path, [_fill("a", 1, NOW), _fill("b", 2, NOW - timedelta(minutes=5))])
    r = _make(tmp_path).check_clock_health()
    assert r.status == "warn"  # never "fail"
    assert r.status != "fail"


def test_registered_as_exs10(tmp_path):
    _status(tmp_path)
    s = _make(tmp_path)
    # fabricate the minimal runtime the other checks tolerate is out of scope;
    # just confirm the method exists and yields EXS10.
    assert s.check_clock_health().check_id == "EXS10"

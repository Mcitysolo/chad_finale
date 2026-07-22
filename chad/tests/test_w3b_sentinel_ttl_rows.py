"""W3B-1: sentinel TTL rows for drawdown_state + ibkr_watchdog_last.

Covers:
- config/exterminator.json declares both feed rows (the P1-3/P1-4 artifacts
  that previously had no freshness net anywhere);
- _parse_ts numeric-epoch support (ibkr_watchdog_last stamps ts_unix, a float);
- EXS1 end-to-end over a ts_unix feed (fresh -> ok, dead -> fail);
- feed_watchdog secondary-net rows present.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.ops.exterminator_sentinel import ExterminatorSentinel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "exterminator.json"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Config rows
# ---------------------------------------------------------------------------


def test_config_declares_drawdown_state_row():
    feeds = _load_config()["feeds"]
    row = feeds["drawdown_state"]
    assert row["path"] == "runtime/drawdown_state.json"
    assert row["ts_field"] == "ts_utc"
    assert row["ttl_seconds"] == 300
    assert row["ttl_verified"] is True
    assert "ops/drawdown_publisher.py" in row["ttl_source"]
    assert row["warn_after_seconds"] == 300
    assert row["fail_after_seconds"] == 900


def test_config_declares_ibkr_watchdog_row():
    feeds = _load_config()["feeds"]
    row = feeds["ibkr_watchdog_last"]
    assert row["path"] == "runtime/ibkr_watchdog_last.json"
    assert row["ts_field"] == "ts_unix"
    assert row["ttl_verified"] is True
    # warn at 3x the 120s cadence so timer jitter cannot flap a 5-min scan;
    # fail must never precede warn.
    assert row["warn_after_seconds"] == 360
    assert row["fail_after_seconds"] >= row["warn_after_seconds"]


# ---------------------------------------------------------------------------
# _parse_ts numeric-epoch support
# ---------------------------------------------------------------------------


def test_parse_ts_accepts_numeric_epoch():
    ts = ExterminatorSentinel._parse_ts(1784753160.107)
    assert ts is not None
    assert ts.tzinfo is not None
    assert ts.year == 2026


def test_parse_ts_rejects_non_timestamp_numerics():
    # counters/durations below the 1e9 floor
    assert ExterminatorSentinel._parse_ts(0) is None
    assert ExterminatorSentinel._parse_ts(42) is None
    assert ExterminatorSentinel._parse_ts(86400.0) is None
    # millisecond epochs above the 4e9 ceiling
    assert ExterminatorSentinel._parse_ts(1784753160107.0) is None
    # bool is an int subclass and must never parse
    assert ExterminatorSentinel._parse_ts(True) is None


def test_parse_ts_iso_still_works():
    ts = ExterminatorSentinel._parse_ts("2026-07-22T20:45:57Z")
    assert ts is not None and ts.year == 2026


# ---------------------------------------------------------------------------
# EXS1 end-to-end over a ts_unix feed
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 7, 22, 21, 0, 0, tzinfo=timezone.utc)


def _sentinel(tmp_path: Path, feed_cfg: dict) -> ExterminatorSentinel:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()
    config_path = tmp_path / "exterminator.json"
    config_path.write_text(json.dumps({"feeds": feed_cfg}), encoding="utf-8")
    return ExterminatorSentinel(
        repo_root=tmp_path,
        runtime_dir=runtime,
        reports_dir=reports,
        config_path=config_path,
        clock=lambda: _NOW,
    )


_WATCHDOG_FEED = {
    "ibkr_watchdog_last": {
        "path": "runtime/ibkr_watchdog_last.json",
        "format": "json",
        "ts_field": "ts_unix",
        "ttl_seconds": 120,
        "ttl_verified": True,
        "ttl_source": "test",
        "warn_after_seconds": 360,
        "fail_after_seconds": 720,
    }
}


def _write_watchdog_artifact(tmp_path: Path, age_seconds: float) -> None:
    payload = {
        "ts_unix": (_NOW - timedelta(seconds=age_seconds)).timestamp(),
        "ok": True,
        "ttl_seconds": 120,
    }
    (tmp_path / "runtime" / "ibkr_watchdog_last.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _feed_row(result, name):
    return next(r for r in result.evidence["feeds"] if r["feed"] == name)


def test_exs1_ts_unix_feed_fresh_is_ok(tmp_path):
    s = _sentinel(tmp_path, _WATCHDOG_FEED)
    _write_watchdog_artifact(tmp_path, age_seconds=30)
    row = _feed_row(s.check_stale_feeds(), "ibkr_watchdog_last")
    assert row["status"] == "ok"
    assert 25 <= row["age_seconds"] <= 35


def test_exs1_ts_unix_feed_dead_is_fail(tmp_path):
    # 44 days stale -- the P1-4 silent-death scenario this row exists for.
    s = _sentinel(tmp_path, _WATCHDOG_FEED)
    _write_watchdog_artifact(tmp_path, age_seconds=44 * 86400)
    row = _feed_row(s.check_stale_feeds(), "ibkr_watchdog_last")
    assert row["status"] == "fail"


def test_exs1_ts_unix_feed_jitter_zone_stays_ok(tmp_path):
    # 130s old: beyond the artifact's 120s ttl but inside the 360s warn --
    # ordinary timer jitter must not flap.
    s = _sentinel(tmp_path, _WATCHDOG_FEED)
    _write_watchdog_artifact(tmp_path, age_seconds=130)
    row = _feed_row(s.check_stale_feeds(), "ibkr_watchdog_last")
    assert row["status"] == "ok"


# ---------------------------------------------------------------------------
# feed_watchdog secondary net
# ---------------------------------------------------------------------------


def test_feed_watchdog_watches_both_artifacts():
    from chad.ops.feed_watchdog import WATCHED_FEEDS

    by_name = {f["name"]: f for f in WATCHED_FEEDS}
    assert by_name["drawdown_state"]["file"] == "drawdown_state.json"
    assert by_name["drawdown_state"]["ttl"] == 300
    assert by_name["ibkr_watchdog"]["file"] == "ibkr_watchdog_last.json"
    assert by_name["ibkr_watchdog"]["ttl"] == 360

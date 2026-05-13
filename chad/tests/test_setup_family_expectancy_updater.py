"""Tests for chad/analytics/setup_family_expectancy_updater.py — Gap-2
of the v9.1 post-implementation audit.

Covers the 12 acceptance cases:
1.  empty ledger → all canonical families exist with NO_DATA
2.  5 ORB trades, 4 wins → LOW_SAMPLE, win_rate=0.8
3.  10 ORB trades, 6 wins → ACTIVE
4.  missing setup_family → bucketed as UNKNOWN
5.  stop_width_usd=0 → no ZeroDivisionError, avg_r safely computed
6.  alpha_intraday trades → ignored
7.  trades older than 90-day lookback → excluded
8.  meta.skip_reason=SKIP_STOP_TOO_WIDE → counted correctly
9.  output file atomic + valid JSON + canonical families exist
10. idempotency: two runs on identical input → identical payload
11. malformed NDJSON line → skipped, warning logged
12. expectancy_r formula: 2 wins +2R / 2 losses -1R → expectancy=0.5
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.analytics.setup_family_expectancy_updater import (
    CANONICAL_FAMILIES,
    SKIP_REASON_STOP_TOO_WIDE,
    SetupFamilyExpectancyUpdater,
    UNKNOWN_FAMILY,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _record(
    *,
    strategy: str = "alpha_intraday_micro",
    pnl: Optional[float] = None,
    setup_family: Optional[str] = "ORB",
    stop_width_usd: Optional[float] = 50.0,
    skip_reason: Optional[str] = None,
    exit_offset_days: int = 0,
    now: Optional[datetime] = None,
    include_meta: bool = True,
) -> Dict[str, Any]:
    now = now or datetime(2026, 5, 13, tzinfo=timezone.utc)
    exit_ts = now - timedelta(days=int(exit_offset_days))
    payload: Dict[str, Any] = {
        "schema_version": "closed_trade.v1",
        "strategy": strategy,
        "symbol": "MES",
        "side": "BUY",
        "exit_time_utc": exit_ts.isoformat().replace("+00:00", "Z"),
        "entry_time_utc": (exit_ts - timedelta(minutes=10))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    if pnl is not None:
        payload["pnl"] = float(pnl)
    if include_meta:
        meta: Dict[str, Any] = {}
        if setup_family is not None:
            meta["setup_family"] = setup_family
        if stop_width_usd is not None:
            meta["stop_width_usd"] = float(stop_width_usd)
        if skip_reason is not None:
            meta["skip_reason"] = skip_reason
        if meta:
            payload["meta"] = meta
    return {
        "payload": payload,
        "timestamp_utc": exit_ts.isoformat().replace("+00:00", "Z"),
    }


def _write_ledger(
    trades_dir: Path,
    records: List[Dict[str, Any]],
    *,
    filename: str = "trade_history_20260513.ndjson",
) -> Path:
    trades_dir.mkdir(parents=True, exist_ok=True)
    path = trades_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _make_updater(tmp_path: Path, **kwargs: Any) -> SetupFamilyExpectancyUpdater:
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    return SetupFamilyExpectancyUpdater(
        trades_dir=trades_dir,
        output_path=tmp_path / "out" / "setup_family_expectancy.json",
        lookback_days=kwargs.pop("lookback_days", 90),
        now=kwargs.pop(
            "now", datetime(2026, 5, 13, 12, tzinfo=timezone.utc)
        ),
        ts_override=kwargs.pop("ts_override", "2026-05-13T12:00:00Z"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_empty_ledger_writes_canonical_families(tmp_path: Path) -> None:
    updater = _make_updater(tmp_path)
    payload = updater.run()
    fams = payload["families"]
    for fam in CANONICAL_FAMILIES:
        assert fam in fams
        assert fams[fam]["status"] == "NO_DATA"
        assert fams[fam]["trades"] == 0
        assert fams[fam]["wins"] == 0
        assert fams[fam]["win_rate"] is None
        assert fams[fam]["avg_r"] is None
        assert fams[fam]["expectancy_r"] is None
    assert payload["summary"]["trades_processed"] == 0
    assert payload["summary"]["last_trade_ts_utc"] is None


def test_2_five_orb_trades_four_wins(tmp_path: Path) -> None:
    records = [
        _record(pnl=+100.0, setup_family="ORB") for _ in range(4)
    ] + [_record(pnl=-50.0, setup_family="ORB")]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    orb = payload["families"]["ORB"]
    assert orb["trades"] == 5
    assert orb["wins"] == 4
    assert orb["win_rate"] == pytest.approx(0.8, abs=1e-6)
    assert orb["status"] == "LOW_SAMPLE"


def test_3_ten_orb_trades_six_wins_active(tmp_path: Path) -> None:
    records = [_record(pnl=+100.0, setup_family="ORB") for _ in range(6)] + [
        _record(pnl=-50.0, setup_family="ORB") for _ in range(4)
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    orb = payload["families"]["ORB"]
    assert orb["trades"] == 10
    assert orb["wins"] == 6
    assert orb["status"] == "ACTIVE"
    assert orb["win_rate"] == pytest.approx(0.6, abs=1e-6)


def test_4_missing_setup_family_goes_to_unknown(tmp_path: Path) -> None:
    records = [
        _record(pnl=+50.0, setup_family=None, stop_width_usd=25.0),
        _record(pnl=-25.0, setup_family=None, stop_width_usd=25.0),
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    assert UNKNOWN_FAMILY in payload["families"]
    unk = payload["families"][UNKNOWN_FAMILY]
    assert unk["trades"] == 2
    assert unk["wins"] == 1


def test_5_zero_stop_width_does_not_crash(tmp_path: Path) -> None:
    records = [
        _record(pnl=+100.0, setup_family="ORB", stop_width_usd=0.0),
        _record(pnl=+100.0, setup_family="ORB", stop_width_usd=50.0),
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()  # must not raise ZeroDivisionError
    orb = payload["families"]["ORB"]
    assert orb["trades"] == 2
    # Only the valid stop_width trade contributes to avg_r.
    assert orb["avg_r"] == pytest.approx(2.0, abs=1e-6)


def test_6_other_strategies_ignored(tmp_path: Path) -> None:
    records = [
        _record(pnl=+100.0, setup_family="ORB", strategy="alpha_intraday"),
        _record(pnl=-100.0, setup_family="ORB", strategy="alpha_intraday"),
        _record(pnl=+100.0, setup_family="ORB", strategy="alpha_intraday_micro"),
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    assert payload["families"]["ORB"]["trades"] == 1
    assert payload["summary"]["total_trades_found"] == 1


def test_7_records_outside_lookback_excluded(tmp_path: Path) -> None:
    records = [
        # 100 days old — outside default 90-day lookback.
        _record(
            pnl=+100.0, setup_family="ORB", exit_offset_days=100
        ),
        # Inside lookback.
        _record(pnl=+50.0, setup_family="ORB", exit_offset_days=10),
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    orb = payload["families"]["ORB"]
    assert orb["trades"] == 1
    assert orb["wins"] == 1


def test_8_skip_stop_too_wide_counted(tmp_path: Path) -> None:
    records = [
        _record(pnl=None, setup_family="ORB", skip_reason=SKIP_REASON_STOP_TOO_WIDE),
        _record(pnl=None, setup_family="ORB", skip_reason=SKIP_REASON_STOP_TOO_WIDE),
        _record(pnl=+50.0, setup_family="ORB"),  # normal trade, not a skip
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    orb = payload["families"]["ORB"]
    assert orb["skip_count_stop_too_wide"] == 2
    assert orb["trades"] == 1


def test_9_output_written_atomically(tmp_path: Path) -> None:
    records = [_record(pnl=+10.0, setup_family="VWAP_RECLAIM")]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    updater.run()

    out_path = tmp_path / "out" / "setup_family_expectancy.json"
    assert out_path.is_file()
    parent = out_path.parent
    leftover = [p for p in parent.iterdir() if p.suffix == ".tmp"]
    assert leftover == []
    data = json.loads(out_path.read_text(encoding="utf-8"))
    for fam in CANONICAL_FAMILIES:
        assert fam in data["families"]
    assert data["schema_version"] == "setup_family_expectancy.v2"
    assert data["strategy"] == "alpha_intraday_micro"


def test_10_idempotent_two_runs(tmp_path: Path) -> None:
    records = [
        _record(pnl=+100.0, setup_family="ORB"),
        _record(pnl=-50.0, setup_family="ORB"),
        _record(pnl=+25.0, setup_family="VWAP_RECLAIM"),
    ]
    _write_ledger(tmp_path / "trades", records)

    first = _make_updater(tmp_path).run()
    second = _make_updater(tmp_path).run()

    assert first == second


def test_11_malformed_ndjson_line_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    good = _record(pnl=+10.0, setup_family="ORB")
    path = trades_dir / "trade_history_20260513.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("this is not valid json\n")
        fh.write(json.dumps(good) + "\n")
        fh.write("{ also broken\n")

    updater = SetupFamilyExpectancyUpdater(
        trades_dir=trades_dir,
        output_path=tmp_path / "out" / "setup_family_expectancy.json",
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        ts_override="2026-05-13T12:00:00Z",
    )
    with caplog.at_level(
        logging.WARNING,
        logger="chad.analytics.setup_family_expectancy_updater",
    ):
        payload = updater.run()

    assert payload["families"]["ORB"]["trades"] == 1
    assert payload["summary"]["trades_skipped_corrupt"] == 2
    assert any(
        "corrupt_ledger_line" in rec.message for rec in caplog.records
    )


def test_12_expectancy_formula(tmp_path: Path) -> None:
    # 2 wins at +2R, 2 losses at -1R.
    # pnl / stop_width = R; with stop_width=50, pnl=+100 gives R=+2,
    # pnl=-50 gives R=-1.
    records = [
        _record(pnl=+100.0, setup_family="ORB", stop_width_usd=50.0),
        _record(pnl=+100.0, setup_family="ORB", stop_width_usd=50.0),
        _record(pnl=-50.0, setup_family="ORB", stop_width_usd=50.0),
        _record(pnl=-50.0, setup_family="ORB", stop_width_usd=50.0),
    ]
    _write_ledger(tmp_path / "trades", records)
    updater = _make_updater(tmp_path)
    payload = updater.run()
    orb = payload["families"]["ORB"]
    assert orb["trades"] == 4
    assert orb["wins"] == 2
    assert orb["win_rate"] == pytest.approx(0.5, abs=1e-6)
    # avg_r over [+2, +2, -1, -1] = 0.5
    assert orb["avg_r"] == pytest.approx(0.5, abs=1e-6)
    # expectancy = 0.5 * 2 - 0.5 * 1 = 0.5
    assert orb["expectancy_r"] == pytest.approx(0.5, abs=1e-6)

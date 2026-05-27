"""Unit tests for chad/validators/position_authority.py (R1 / TRUTH-RECONCILE-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.validators import position_authority as pav

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "position_authority"
HUGE_STALE = 10**12  # effectively disables the stale check


def _run(case: str, stale_seconds: float = HUGE_STALE) -> dict:
    return pav.build_report(FIXTURE_ROOT / case, stale_seconds=stale_seconds)


def test_matching_fixture_returns_ok():
    report = _run("matching")
    assert report["verdict"] == "OK", json.dumps(report["mismatch"], indent=2)
    m = report["mismatch"]
    assert m["missing_symbol"] == []
    assert m["extra_symbol"] == []
    assert m["qty_mismatch"] == []
    assert m["side_mismatch"] == []
    assert m["stale_ts"] == []
    assert m["key_shape_mismatch"] == []


def test_missing_symbol_detected():
    report = _run("missing_symbol")
    assert report["verdict"] == "MISMATCH"
    missing = report["mismatch"]["missing_symbol"]
    assert any(e["symbol"] == "GOOGL" for e in missing), missing


def test_extra_symbol_detected():
    report = _run("extra_symbol")
    assert report["verdict"] == "MISMATCH"
    extra = report["mismatch"]["extra_symbol"]
    assert any(e["symbol"] == "TSLA" for e in extra), extra


def test_qty_mismatch_detected():
    report = _run("qty_mismatch")
    assert report["verdict"] == "MISMATCH"
    diffs = report["mismatch"]["qty_mismatch"]
    aapl = [d for d in diffs if d["symbol"] == "AAPL"]
    assert aapl, diffs
    assert aapl[0]["snapshot_qty"] == 100.0
    assert aapl[0]["ledger_qty"] == 80.0
    assert report["mismatch"]["worst_qty_diff"] == pytest.approx(20.0)


def test_stale_ts_detected_with_strict_threshold():
    # Snapshot fixture uses ts_utc=2026-04-01 (far past any sane threshold).
    report = pav.build_report(FIXTURE_ROOT / "stale_ts", stale_seconds=60)
    assert report["verdict"] == "MISMATCH"
    stales = report["mismatch"]["stale_ts"]
    snap_stales = [s for s in stales if s["path"].endswith("positions_snapshot.json")]
    assert snap_stales, stales


def test_key_shape_ledger_entry_missing_symbol_field():
    # Ledger entry without 'symbol' triggers a load-side warning surfaced via
    # SurfaceLoad.error; this is not a per-symbol mismatch but it must not
    # silently absorb the bad row into "positions agree".
    report = _run("key_shape")
    led_error = report["surfaces"]["ledger"]["error"]
    assert led_error is not None and "missing_symbol" in led_error, led_error


def test_validator_never_writes_to_runtime(tmp_path: Path):
    runtime_dir = FIXTURE_ROOT / "matching"
    before = sorted((p, p.stat().st_mtime) for p in runtime_dir.iterdir())
    pav.build_report(runtime_dir)
    after = sorted((p, p.stat().st_mtime) for p in runtime_dir.iterdir())
    assert before == after, "build_report mutated fixture directory"


def test_main_exit_codes(tmp_path: Path, capsys):
    rc_ok = pav.main(["--runtime", str(FIXTURE_ROOT / "matching"), "--stale-seconds", str(HUGE_STALE)])
    assert rc_ok == 0
    capsys.readouterr()

    rc_bad = pav.main(["--runtime", str(FIXTURE_ROOT / "qty_mismatch"), "--stale-seconds", str(HUGE_STALE)])
    assert rc_bad == 2
    capsys.readouterr()


def test_main_refuses_output_inside_runtime(tmp_path, capsys, monkeypatch):
    # Construct a fake runtime/ subtree inside tmp_path.
    fake_runtime = tmp_path / "runtime"
    fake_runtime.mkdir()
    monkeypatch.chdir(tmp_path)
    out_path = fake_runtime / "report.json"
    rc = pav.main(
        [
            "--runtime",
            str(FIXTURE_ROOT / "matching"),
            "--output",
            str(out_path),
            "--stale-seconds",
            str(HUGE_STALE),
        ]
    )
    err = capsys.readouterr().err
    assert rc == pav.EXIT_READ_ERROR
    assert "refusing to write inside runtime/" in err
    assert not out_path.exists()

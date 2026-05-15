"""Phase B Item 3 — RVOL scanner tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pytest

from chad.market_data import volume_scan_publisher as vsp
from chad.utils.rvol_gate import (
    RVOL_HIGH_BOOST,
    RVOL_LOW_PENALTY,
    RvolGateResult,
    get_rvol_adjustment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_daily_bars(path: Path, volumes: List[float]) -> None:
    """Write a JSON daily bar file with the given closing volumes."""
    bars: List[Dict[str, Any]] = []
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i, v in enumerate(volumes):
        ts = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        bars.append({
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": v, "ts_utc": ts,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "bars": bars,
        "source": "test",
        "symbol": path.stem,
        "timeframe": "1d",
        "ts_utc": "2026-05-15T00:00:00Z",
        "ttl_seconds": 86400,
    }), encoding="utf-8")


def _write_rvol_payload(
    path: Path,
    *,
    ts_utc: str,
    ttl_seconds: int = 600,
    symbols: Dict[str, Dict[str, Any]],
    market_open: bool = True,
) -> None:
    payload = {
        "schema_version": "volume_scan.v1",
        "ts_utc": ts_utc,
        "ttl_seconds": int(ttl_seconds),
        "market_open": bool(market_open),
        "minutes_into_session": 60.0 if market_open else None,
        "fraction_elapsed": 0.1538 if market_open else None,
        "symbols": symbols,
        "summary": {
            "symbols_scanned": len(symbols),
            "high_rvol_count": 0,
            "high_rvol_symbols": [],
        },
        "source": {
            "volume_provider": "polygon_snapshot",
            "avg_volume_source": "daily_bars",
            "provider_status": "real",
        },
        "status": "ok",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_market_now_utc() -> datetime:
    """Return a UTC datetime that maps to a Tuesday 10:30 ET (open market)."""
    et = datetime(2026, 5, 12, 10, 30, tzinfo=ZoneInfo("America/New_York"))
    return et.astimezone(timezone.utc)


def _closed_market_now_utc() -> datetime:
    """Return a UTC datetime that maps to a Saturday (market closed)."""
    et = datetime(2026, 5, 16, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    return et.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Test 1-5 — classify_rvol thresholds
# ---------------------------------------------------------------------------


def test_classify_rvol_high() -> None:
    assert vsp.classify_rvol(3.5) == "high"
    assert vsp.classify_rvol(3.0) == "high"


def test_classify_rvol_above() -> None:
    assert vsp.classify_rvol(2.0) == "above"
    assert vsp.classify_rvol(1.5) == "above"
    assert vsp.classify_rvol(2.99) == "above"


def test_classify_rvol_normal() -> None:
    assert vsp.classify_rvol(1.0) == "normal"
    assert vsp.classify_rvol(0.7) == "normal"
    assert vsp.classify_rvol(1.49) == "normal"


def test_classify_rvol_low() -> None:
    assert vsp.classify_rvol(0.5) == "low"
    assert vsp.classify_rvol(0.0) == "low"
    assert vsp.classify_rvol(0.69) == "low"


def test_classify_rvol_none_unavailable() -> None:
    assert vsp.classify_rvol(None) == "unavailable"


# ---------------------------------------------------------------------------
# Test 6-7 — gate boost/penalty
# ---------------------------------------------------------------------------


def test_rvol_gate_high_boosts(tmp_path: Path) -> None:
    _write_rvol_payload(
        tmp_path / "volume_scan.json",
        ts_utc=_utc_now_z(),
        symbols={"SPY": {
            "rvol": 3.2, "rvol_class": "high", "data_available": True,
            "current_volume": 12000000, "avg_daily_volume": 25000000.0,
            "expected_volume": 3846000.0,
        }},
    )
    r = get_rvol_adjustment("SPY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == pytest.approx(RVOL_HIGH_BOOST)
    assert r.rvol_class == "high"
    assert r.rvol == pytest.approx(3.2)


def test_rvol_gate_low_penalizes(tmp_path: Path) -> None:
    _write_rvol_payload(
        tmp_path / "volume_scan.json",
        ts_utc=_utc_now_z(),
        symbols={"BAC": {
            "rvol": 0.4, "rvol_class": "low", "data_available": True,
            "current_volume": 100000, "avg_daily_volume": 25000000.0,
            "expected_volume": 250000.0,
        }},
    )
    r = get_rvol_adjustment("BAC", runtime_dir=tmp_path)
    assert r.confidence_adjustment == pytest.approx(-RVOL_LOW_PENALTY)
    assert r.rvol_class == "low"


# ---------------------------------------------------------------------------
# Test 8 — normal class -> zero adjustment
# ---------------------------------------------------------------------------


def test_rvol_gate_normal_zero(tmp_path: Path) -> None:
    _write_rvol_payload(
        tmp_path / "volume_scan.json",
        ts_utc=_utc_now_z(),
        symbols={"AAPL": {
            "rvol": 1.0, "rvol_class": "normal", "data_available": True,
            "current_volume": 1000000, "avg_daily_volume": 26000000.0,
            "expected_volume": 1000000.0,
        }},
    )
    r = get_rvol_adjustment("AAPL", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rvol_class == "normal"


# ---------------------------------------------------------------------------
# Test 9 — missing file fails open
# ---------------------------------------------------------------------------


def test_rvol_gate_missing_file_fails_open(tmp_path: Path) -> None:
    r = get_rvol_adjustment("AAPL", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rvol_class == "unavailable"
    assert r.rvol is None


# ---------------------------------------------------------------------------
# Test 10 — stale file fails open
# ---------------------------------------------------------------------------


def test_rvol_gate_stale_file_fails_open(tmp_path: Path) -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_rvol_payload(
        tmp_path / "volume_scan.json",
        ts_utc=old_ts,
        ttl_seconds=1,
        symbols={"SPY": {
            "rvol": 3.5, "rvol_class": "high", "data_available": True,
            "current_volume": 1, "avg_daily_volume": 1.0,
            "expected_volume": 1.0,
        }},
    )
    r = get_rvol_adjustment("SPY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rvol_class == "unavailable"


# ---------------------------------------------------------------------------
# Test 11 — publisher no-fetch-test-mode emits valid schema
# ---------------------------------------------------------------------------


def test_publisher_no_fetch_test_mode_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_daily_bars(data_dir / "AAPL.json", [20_000_000.0] * 20)
    _write_daily_bars(data_dir / "SPY.json", [30_000_000.0] * 20)

    payload = vsp.build_payload(
        data_dir=data_dir,
        extra_symbols=["AAPL", "SPY"],
        test_mode=True,
    )
    assert payload["schema_version"] == "volume_scan.v1"
    assert "ts_utc" in payload
    assert payload["ttl_seconds"] == vsp.DEFAULT_TTL_SECONDS
    assert payload["status"] == "unavailable"
    assert payload["source"]["volume_provider"] == "test_no_fetch"
    assert payload["source"]["provider_status"] == "test_no_fetch"
    for sym in ("AAPL", "SPY"):
        rec = payload["symbols"][sym]
        assert rec["rvol_class"] == "unavailable"
        assert rec["data_available"] is False
        assert rec["rvol"] is None
        assert rec["avg_daily_volume"] is not None


# ---------------------------------------------------------------------------
# Test 12 — publisher outside market hours marks unavailable
# ---------------------------------------------------------------------------


def test_publisher_outside_market_hours(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_daily_bars(data_dir / "AAPL.json", [20_000_000.0] * 20)

    payload = vsp.build_payload(
        data_dir=data_dir,
        extra_symbols=["AAPL"],
        now_utc=_closed_market_now_utc(),
    )
    assert payload["market_open"] is False
    assert payload["fraction_elapsed"] is None
    assert payload["minutes_into_session"] is None
    assert payload["status"] == "unavailable"
    aapl = payload["symbols"]["AAPL"]
    assert aapl["rvol_class"] == "unavailable"
    assert aapl["data_available"] is False


# ---------------------------------------------------------------------------
# Test 13 — publisher market-hours synthetic Polygon snapshot computes rvol
# ---------------------------------------------------------------------------


def test_publisher_market_hours_synthetic_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "bars"
    _write_daily_bars(data_dir / "SPY.json", [30_000_000.0] * 20)
    _write_daily_bars(data_dir / "AAPL.json", [25_000_000.0] * 20)

    # Isolate the universe so the publisher only sees the test symbols.
    monkeypatch.setattr(vsp, "_load_universe_symbols", lambda: [])

    def fake_fetch(api_key: str, tickers: List[str]) -> Dict[str, int]:
        # At 10:30 ET fraction_elapsed = 60/390 ~ 0.1538.
        # SPY expected ~30M * 0.1538 = 4.615M. Provide 4x that for "high".
        # AAPL expected ~25M * 0.1538 = 3.846M. Provide ~1x for "normal".
        return {"SPY": 18_500_000, "AAPL": 3_900_000}

    payload = vsp.build_payload(
        data_dir=data_dir,
        extra_symbols=["SPY", "AAPL"],
        now_utc=_open_market_now_utc(),
        polygon_fetcher=fake_fetch,
        polygon_api_key="UNUSED-TEST",
    )
    assert payload["market_open"] is True
    assert payload["status"] == "ok"
    assert payload["source"]["volume_provider"] == "polygon_snapshot"
    spy = payload["symbols"]["SPY"]
    assert spy["data_available"] is True
    assert spy["rvol_class"] == "high"
    assert spy["rvol"] is not None and spy["rvol"] > 3.0
    aapl = payload["symbols"]["AAPL"]
    assert aapl["data_available"] is True
    assert aapl["rvol_class"] == "normal"
    assert payload["summary"]["high_rvol_count"] == 1
    assert payload["summary"]["high_rvol_symbols"] == ["SPY"]


# ---------------------------------------------------------------------------
# Test 14 — daily bars average volume calculation
# ---------------------------------------------------------------------------


def test_compute_avg_daily_volume() -> None:
    bars = [{"volume": 10.0}, {"volume": 20.0}, {"volume": 30.0},
            {"volume": 40.0}, {"volume": 50.0}]
    avg = vsp.compute_avg_daily_volume(bars)
    assert avg == pytest.approx(30.0)

    # Trailing 20 only: bars older than 20 are ignored.
    bars_long = [{"volume": 1.0}] * 5 + [{"volume": 100.0}] * 20
    avg2 = vsp.compute_avg_daily_volume(bars_long)
    assert avg2 == pytest.approx(100.0)

    # Insufficient valid bars -> None.
    bars_short = [{"volume": 10.0}, {"volume": 20.0}]
    assert vsp.compute_avg_daily_volume(bars_short) is None

    # Invalid volumes are skipped.
    bars_mixed = [{"volume": "bad"}, {"volume": None}, {"volume": -1.0},
                  {"volume": 10.0}, {"volume": 20.0}, {"volume": 30.0},
                  {"volume": 40.0}, {"volume": 50.0}]
    avg3 = vsp.compute_avg_daily_volume(bars_mixed)
    assert avg3 == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Test 15 — alpha_intraday _build_signal meta contains rvol_class
# ---------------------------------------------------------------------------


def test_alpha_intraday_meta_contains_rvol_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chad.strategies import alpha_intraday as ai

    monkeypatch.setattr(
        ai, "get_rvol_adjustment",
        lambda sym: RvolGateResult(0.0, None, "unavailable"),
    )
    sig = ai._build_signal(
        "SPY",
        ai.SignalSide.BUY,
        confidence=0.7,
        trigger="vol_explosion",
        timeframe="1m",
        atr=0.5,
        tier_max_risk_usd=1000.0,
    )
    assert sig is not None
    assert "rvol_class" in sig.meta
    assert "rvol" in sig.meta
    assert "rvol_confidence_adjustment" in sig.meta
    assert sig.meta["rvol_class"] == "unavailable"
    assert sig.meta["rvol"] is None
    assert sig.meta["rvol_confidence_adjustment"] == 0.0


# ---------------------------------------------------------------------------
# Test 16 — alpha_intraday confidence boosted on high RVOL
# ---------------------------------------------------------------------------


def test_alpha_intraday_confidence_boosted_on_high_rvol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chad.strategies import alpha_intraday as ai

    monkeypatch.setattr(
        ai, "get_rvol_adjustment",
        lambda sym: RvolGateResult(RVOL_HIGH_BOOST, 3.5, "high"),
    )
    # Disable RS adjustment so only RVOL affects confidence.
    from chad.utils.rs_gate import RSGateResult as _RS
    monkeypatch.setattr(
        ai, "get_rs_adjustment",
        lambda sym, side: _RS(0.0, "unknown", None, None, "unknown"),
    )
    sig = ai._build_signal(
        "SPY",
        ai.SignalSide.BUY,
        confidence=0.70,
        trigger="vol_explosion",
        timeframe="1m",
        atr=0.5,
        tier_max_risk_usd=1000.0,
    )
    assert sig is not None
    assert sig.meta["rvol_class"] == "high"
    assert sig.meta["rvol_confidence_adjustment"] == pytest.approx(RVOL_HIGH_BOOST)
    assert sig.confidence == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Test 17 — alpha_intraday confidence penalized on low RVOL
# ---------------------------------------------------------------------------


def test_alpha_intraday_confidence_penalized_on_low_rvol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chad.strategies import alpha_intraday as ai

    monkeypatch.setattr(
        ai, "get_rvol_adjustment",
        lambda sym: RvolGateResult(-RVOL_LOW_PENALTY, 0.3, "low"),
    )
    from chad.utils.rs_gate import RSGateResult as _RS
    monkeypatch.setattr(
        ai, "get_rs_adjustment",
        lambda sym, side: _RS(0.0, "unknown", None, None, "unknown"),
    )
    sig = ai._build_signal(
        "SPY",
        ai.SignalSide.BUY,
        confidence=0.70,
        trigger="vol_explosion",
        timeframe="1m",
        atr=0.5,
        tier_max_risk_usd=1000.0,
    )
    assert sig is not None
    assert sig.meta["rvol_class"] == "low"
    assert sig.meta["rvol_confidence_adjustment"] == pytest.approx(-RVOL_LOW_PENALTY)
    assert sig.confidence == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Test 18 — deploy service/timer files present with expected contents
# ---------------------------------------------------------------------------


def test_deploy_volume_scan_service_and_timer_present() -> None:
    repo = Path(__file__).resolve().parents[2]
    service = repo / "deploy" / "chad-volume-scan.service"
    timer = repo / "deploy" / "chad-volume-scan.timer"
    assert service.is_file(), f"missing {service}"
    assert timer.is_file(), f"missing {timer}"

    svc_text = service.read_text(encoding="utf-8")
    assert (
        "ExecStart=/home/ubuntu/chad_finale/venv/bin/python3 "
        "-m chad.market_data.volume_scan_publisher" in svc_text
    )

    tmr_text = timer.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=300" in tmr_text
    assert "Persistent=true" in tmr_text

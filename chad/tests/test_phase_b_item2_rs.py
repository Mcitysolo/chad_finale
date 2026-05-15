"""Phase B Item 2 — relative strength publisher tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from chad.market_data import relative_strength_publisher as rsp
from chad.utils.rs_gate import (
    RS_CONFIDENCE_PENALTY,
    RSGateResult,
    get_rs_adjustment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bar(close: float, ts: str = "2026-05-14T00:00:00Z") -> Dict[str, Any]:
    return {"close": close, "volume": 1_000_000, "ts_utc": ts}


def _write_bars(path: Path, closes: List[float]) -> None:
    """Write a JSON bar file with the given closing series.

    The series is treated chronologically: index 0 is oldest, index -1 newest.
    """
    bars: List[Dict[str, Any]] = []
    base = datetime(2026, 5, 1, tzinfo=timezone.utc)
    for i, c in enumerate(closes):
        ts = (base + timedelta(days=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        bars.append(_bar(c, ts=ts))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bars": bars}), encoding="utf-8")


def _series_for_return(start: float, total_return: float, length: int) -> List[float]:
    """Build a flat series ending with a single jump matching ``total_return``.

    The publisher's 5-day return uses bars[-1] and bars[-(lookback+1)], so the
    earlier prices are irrelevant — they just need to be valid. We always
    pin bars[-(lookback+1)] (== bars[0] when length == lookback+1) to ``start``
    and bars[-1] to ``start * (1 + total_return)``.
    """
    assert length >= 2
    end = start * (1.0 + total_return)
    body = [start] * (length - 1) + [end]
    return body


def _write_rs(
    path: Path,
    *,
    ts_utc: str,
    ttl_seconds: int = 90000,
    market_direction: str = "up",
    symbols: Dict[str, Dict[str, Any]],
) -> None:
    payload = {
        "schema_version": "relative_strength.v1",
        "ts_utc": ts_utc,
        "ttl_seconds": int(ttl_seconds),
        "lookback_days": 5,
        "benchmark_spy_return_5d": 0.04,
        "benchmark_qqq_return_5d": 0.05,
        "market_direction": market_direction,
        "symbols": symbols,
        "summary": {
            "symbols_computed": len(symbols),
            "strong_count": 0,
            "neutral_count": 0,
            "weak_count": 0,
            "unknown_count": 0,
        },
        "source": {
            "provider": "daily_bars",
            "bar_path": "data/bars/1d/",
            "provider_status": "real",
        },
        "status": "ok",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 — strong by ratio and excess
# ---------------------------------------------------------------------------


def test_rs_strong_by_ratio_and_excess(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_bars(data_dir / "SPY.json", _series_for_return(100.0, 0.04, 10))
    _write_bars(data_dir / "QQQ.json", _series_for_return(100.0, 0.05, 10))
    _write_bars(data_dir / "AAPL.json", _series_for_return(100.0, 0.10, 10))

    payload = rsp.build_payload(
        data_dir=data_dir, lookback_days=5, extra_symbols=["AAPL"],
    )
    aapl = payload["symbols"]["AAPL"]
    assert aapl["data_available"] is True
    assert aapl["rs_class"] == "strong"
    assert aapl["rs_vs_spy"] is not None
    assert aapl["rs_vs_spy"] > 2.0
    assert aapl["excess_vs_spy_5d"] is not None
    assert aapl["excess_vs_spy_5d"] > 0.03


# ---------------------------------------------------------------------------
# Test 2 — weak
# ---------------------------------------------------------------------------


def test_rs_weak_by_excess(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_bars(data_dir / "SPY.json", _series_for_return(100.0, 0.04, 10))
    _write_bars(data_dir / "QQQ.json", _series_for_return(100.0, 0.05, 10))
    _write_bars(data_dir / "BAC.json", _series_for_return(100.0, 0.01, 10))

    payload = rsp.build_payload(
        data_dir=data_dir, lookback_days=5, extra_symbols=["BAC"],
    )
    bac = payload["symbols"]["BAC"]
    assert bac["data_available"] is True
    assert bac["rs_class"] == "weak"
    assert bac["excess_vs_spy_5d"] is not None
    assert bac["excess_vs_spy_5d"] <= -0.03


# ---------------------------------------------------------------------------
# Test 3 — neutral
# ---------------------------------------------------------------------------


def test_rs_neutral_by_excess(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_bars(data_dir / "SPY.json", _series_for_return(100.0, 0.03, 10))
    _write_bars(data_dir / "QQQ.json", _series_for_return(100.0, 0.03, 10))
    _write_bars(data_dir / "MSFT.json", _series_for_return(100.0, 0.04, 10))

    payload = rsp.build_payload(
        data_dir=data_dir, lookback_days=5, extra_symbols=["MSFT"],
    )
    msft = payload["symbols"]["MSFT"]
    assert msft["data_available"] is True
    assert msft["rs_class"] == "neutral"


# ---------------------------------------------------------------------------
# Test 4 — SPY flat avoids div/0
# ---------------------------------------------------------------------------


def test_rs_flat_spy_avoids_div_zero(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_bars(data_dir / "SPY.json", _series_for_return(100.0, 0.0, 10))
    _write_bars(data_dir / "QQQ.json", _series_for_return(100.0, 0.0, 10))
    _write_bars(data_dir / "NVDA.json", _series_for_return(100.0, 0.05, 10))

    payload = rsp.build_payload(
        data_dir=data_dir, lookback_days=5, extra_symbols=["NVDA"],
    )
    nvda = payload["symbols"]["NVDA"]
    assert nvda["data_available"] is True
    # Ratio must be None when SPY is flat — no division by zero crash.
    assert nvda["rs_vs_spy"] is None
    # Classification falls back to excess (0.05 > 0.03 -> strong).
    assert nvda["rs_class"] == "strong"
    assert payload["market_direction"] == "flat"


# ---------------------------------------------------------------------------
# Test 5 — negative SPY benchmark does not invert RS
# ---------------------------------------------------------------------------


def test_rs_negative_spy_no_invert(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    _write_bars(data_dir / "SPY.json", _series_for_return(100.0, -0.04, 10))
    _write_bars(data_dir / "QQQ.json", _series_for_return(100.0, -0.04, 10))
    _write_bars(data_dir / "META.json", _series_for_return(100.0, 0.02, 10))

    payload = rsp.build_payload(
        data_dir=data_dir, lookback_days=5, extra_symbols=["META"],
    )
    meta = payload["symbols"]["META"]
    assert meta["data_available"] is True
    # Ratio is None for negative SPY — classification must use excess.
    assert meta["rs_vs_spy"] is None
    assert meta["excess_vs_spy_5d"] is not None
    assert meta["excess_vs_spy_5d"] > 0.03
    assert meta["rs_class"] == "strong"
    assert payload["market_direction"] == "down"


# ---------------------------------------------------------------------------
# Test 6 — publisher dry-run real data includes SPY and QQQ
# ---------------------------------------------------------------------------


def test_publisher_dry_run_includes_spy_qqq() -> None:
    payload = rsp.build_payload(
        data_dir=rsp.DEFAULT_DATA_DIR,
        lookback_days=5,
    )
    assert "SPY" in payload["symbols"]
    assert "QQQ" in payload["symbols"]


# ---------------------------------------------------------------------------
# Test 7 — BUY + weak + UP market -> penalty
# ---------------------------------------------------------------------------


def test_rs_gate_buy_weak_up_penalty(tmp_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_rs(
        tmp_path / "relative_strength.json",
        ts_utc=ts,
        market_direction="up",
        symbols={
            "BAC": {
                "rs_class": "weak",
                "rs_vs_spy": 0.4,
                "excess_vs_spy_5d": -0.05,
            }
        },
    )
    r = get_rs_adjustment("BAC", "BUY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == pytest.approx(-RS_CONFIDENCE_PENALTY)
    assert r.rs_class == "weak"
    assert r.market_direction == "up"


# ---------------------------------------------------------------------------
# Test 8 — BUY + weak + DOWN market -> 0.0
# ---------------------------------------------------------------------------


def test_rs_gate_buy_weak_down_no_penalty(tmp_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_rs(
        tmp_path / "relative_strength.json",
        ts_utc=ts,
        market_direction="down",
        symbols={
            "BAC": {
                "rs_class": "weak",
                "rs_vs_spy": None,
                "excess_vs_spy_5d": -0.05,
            }
        },
    )
    r = get_rs_adjustment("BAC", "BUY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rs_class == "weak"
    assert r.market_direction == "down"


# ---------------------------------------------------------------------------
# Test 9 — SELL + strong + DOWN market -> penalty
# ---------------------------------------------------------------------------


def test_rs_gate_sell_strong_down_penalty(tmp_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_rs(
        tmp_path / "relative_strength.json",
        ts_utc=ts,
        market_direction="down",
        symbols={
            "AAPL": {
                "rs_class": "strong",
                "rs_vs_spy": None,
                "excess_vs_spy_5d": 0.06,
            }
        },
    )
    r = get_rs_adjustment("AAPL", "SELL", runtime_dir=tmp_path)
    assert r.confidence_adjustment == pytest.approx(-RS_CONFIDENCE_PENALTY)
    assert r.rs_class == "strong"
    assert r.market_direction == "down"


# ---------------------------------------------------------------------------
# Test 10 — SELL + strong + UP market -> 0.0
# ---------------------------------------------------------------------------


def test_rs_gate_sell_strong_up_no_penalty(tmp_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_rs(
        tmp_path / "relative_strength.json",
        ts_utc=ts,
        market_direction="up",
        symbols={
            "AAPL": {
                "rs_class": "strong",
                "rs_vs_spy": 2.5,
                "excess_vs_spy_5d": 0.06,
            }
        },
    )
    r = get_rs_adjustment("AAPL", "SELL", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0


# ---------------------------------------------------------------------------
# Test 11 — missing file fails open
# ---------------------------------------------------------------------------


def test_rs_gate_missing_file_fails_open(tmp_path: Path) -> None:
    r = get_rs_adjustment("AAPL", "BUY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rs_class == "unknown"
    assert r.market_direction == "unknown"


# ---------------------------------------------------------------------------
# Test 12 — stale file fails open
# ---------------------------------------------------------------------------


def test_rs_gate_stale_file_fails_open(tmp_path: Path) -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_rs(
        tmp_path / "relative_strength.json",
        ts_utc=old_ts,
        ttl_seconds=1,
        market_direction="up",
        symbols={
            "BAC": {
                "rs_class": "weak",
                "rs_vs_spy": 0.2,
                "excess_vs_spy_5d": -0.08,
            }
        },
    )
    r = get_rs_adjustment("BAC", "BUY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rs_class == "unknown"
    assert r.market_direction == "unknown"


# ---------------------------------------------------------------------------
# Test 13 — unknown rs_class gives no penalty
# ---------------------------------------------------------------------------


def test_rs_gate_unknown_class_no_penalty(tmp_path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_rs(
        tmp_path / "relative_strength.json",
        ts_utc=ts,
        market_direction="up",
        symbols={
            "MEME": {
                "rs_class": "unknown",
                "rs_vs_spy": None,
                "excess_vs_spy_5d": None,
            }
        },
    )
    r = get_rs_adjustment("MEME", "BUY", runtime_dir=tmp_path)
    assert r.confidence_adjustment == 0.0
    assert r.rs_class == "unknown"


# ---------------------------------------------------------------------------
# Test 14 — alpha handler no-raise with missing relative_strength.json
# ---------------------------------------------------------------------------


def test_alpha_handler_missing_rs_no_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chad.strategies.alpha import alpha_handler
    import chad.utils.rs_gate as rs_gate

    monkeypatch.setattr(rs_gate, "DEFAULT_RUNTIME_DIR", tmp_path)
    ctx = SimpleNamespace(
        now=datetime.now(timezone.utc),
        prices={},
        legend=None,
        bars={},
        tier_profile=None,
    )
    result = alpha_handler(ctx)
    assert result == []


# ---------------------------------------------------------------------------
# Test 15 — alpha_intraday MES futures does not use RS adjustment
# ---------------------------------------------------------------------------


def test_alpha_intraday_futures_no_rs_adjustment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chad.strategies import alpha_intraday as ai

    def boom(*args: Any, **kwargs: Any):
        raise AssertionError("rs_gate must not be called for futures")

    monkeypatch.setattr(ai, "get_rs_adjustment", boom)

    sig = ai._build_signal(
        "MES",
        ai.SignalSide.BUY,
        confidence=0.7,
        trigger="vol_explosion",
        timeframe="1m",
        atr=0.5,
        tier_max_risk_usd=1000.0,
    )
    # MES sizing depends on FUTURES_SPECS — only assert no exception fired.
    assert sig is None or sig.symbol == "MES"
    if sig is not None:
        # Futures meta must carry the unknown sentinel and zero adjustment.
        assert sig.meta.get("rs_class") == "unknown"
        assert sig.meta.get("rs_confidence_adjustment") == 0.0
        assert sig.meta.get("rs_market_direction") == "unknown"


# ---------------------------------------------------------------------------
# Test 16 — publisher writes valid schema to temp runtime dir
# ---------------------------------------------------------------------------


def test_publisher_writes_valid_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "bars"
    runtime_dir = tmp_path / "runtime"
    _write_bars(data_dir / "SPY.json", _series_for_return(100.0, 0.04, 10))
    _write_bars(data_dir / "QQQ.json", _series_for_return(100.0, 0.05, 10))
    _write_bars(data_dir / "AAPL.json", _series_for_return(100.0, 0.09, 10))

    payload = rsp.run_publish(
        runtime_dir=runtime_dir,
        data_dir=data_dir,
        lookback_days=5,
        dry_run=False,
        extra_symbols=["AAPL"],
    )

    out_path = runtime_dir / "relative_strength.json"
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "relative_strength.v1"
    assert "ts_utc" in on_disk
    assert on_disk["ttl_seconds"] == 90000
    assert on_disk["lookback_days"] == 5
    assert on_disk["status"] in ("ok", "partial", "error")
    assert on_disk["source"]["provider"] == "daily_bars"
    assert "SPY" in on_disk["symbols"]
    assert "QQQ" in on_disk["symbols"]
    assert on_disk["symbols"]["SPY"]["rs_class"] == "neutral"
    assert payload["status"] == on_disk["status"]


# ---------------------------------------------------------------------------
# Test 17 — deploy service/timer files exist with expected contents
# ---------------------------------------------------------------------------


def test_deploy_service_and_timer_present() -> None:
    repo = Path(__file__).resolve().parents[2]
    service = repo / "deploy" / "chad-rs-refresh.service"
    timer = repo / "deploy" / "chad-rs-refresh.timer"
    assert service.is_file(), f"missing {service}"
    assert timer.is_file(), f"missing {timer}"

    svc_text = service.read_text(encoding="utf-8")
    assert (
        "ExecStart=/home/ubuntu/chad_finale/venv/bin/python3 "
        "-m chad.market_data.relative_strength_publisher" in svc_text
    )

    tmr_text = timer.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=86400" in tmr_text
    assert "Persistent=true" in tmr_text

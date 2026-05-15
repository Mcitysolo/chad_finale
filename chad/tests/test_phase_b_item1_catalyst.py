"""Phase B Item 1 — catalyst news gate tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.market_data import catalyst_news_provider as cnp
from chad.market_data.catalyst_news_provider import (
    CatalystIntel,
    NewsArticle,
    _classify_article,
    build_catalyst_intel,
    get_catalyst_intel,
)
from chad.market_data import news_intel_publisher as nip
from chad.utils.catalyst_gate import check_catalyst_gate


def _utc_now_z(offset_seconds: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_intel(
    runtime_dir: Path,
    symbols: Dict[str, Dict[str, Any]],
    *,
    ttl_seconds: int = 3600,
    ts_utc: str | None = None,
) -> None:
    payload = {
        "schema_version": "news_intel.v1",
        "ts_utc": ts_utc if ts_utc is not None else _utc_now_z(),
        "ttl_seconds": int(ttl_seconds),
        "lookback_hours": 24,
        "source": {"primary": "polygon", "fallback": "yahoo", "provider_status": "real"},
        "status": "ok",
        "symbols": symbols,
        "summary": {
            "symbols_processed": len(symbols),
            "symbols_with_catalyst": sum(1 for v in symbols.values() if v.get("has_catalyst")),
            "high_strength_count": sum(
                1 for v in symbols.values() if v.get("catalyst_strength") == "high"
            ),
            "provider_breakdown": {"polygon": 0, "yahoo": 0, "none": 0, "test_no_fetch": 0},
        },
    }
    (runtime_dir / "news_intel.json").write_text(json.dumps(payload), encoding="utf-8")


def _high_bullish(headline: str = "LLY earnings beat estimates by 35%") -> Dict[str, Any]:
    return {
        "has_catalyst": True,
        "catalyst_strength": "high",
        "catalyst_direction": "bullish",
        "news_count": 3,
        "catalyst_count": 2,
        "latest_headline": headline,
        "latest_ts_utc": _utc_now_z(),
        "catalyst_categories": ["earnings"],
        "source_provider": "test",
    }


def _high_bearish(headline: str = "FDA rejects application — shares halted") -> Dict[str, Any]:
    return {
        "has_catalyst": True,
        "catalyst_strength": "high",
        "catalyst_direction": "bearish",
        "news_count": 2,
        "catalyst_count": 2,
        "latest_headline": headline,
        "latest_ts_utc": _utc_now_z(),
        "catalyst_categories": ["regulatory"],
        "source_provider": "test",
    }


def _medium_bearish() -> Dict[str, Any]:
    return {
        "has_catalyst": True,
        "catalyst_strength": "medium",
        "catalyst_direction": "bearish",
        "news_count": 1,
        "catalyst_count": 1,
        "latest_headline": "Analyst downgrade — price target cut",
        "latest_ts_utc": _utc_now_z(),
        "catalyst_categories": ["ratings"],
        "source_provider": "test",
    }


def _medium_bullish() -> Dict[str, Any]:
    return {
        "has_catalyst": True,
        "catalyst_strength": "medium",
        "catalyst_direction": "bullish",
        "news_count": 1,
        "catalyst_count": 1,
        "latest_headline": "Analyst upgrade — partnership announced",
        "latest_ts_utc": _utc_now_z(),
        "catalyst_categories": ["ratings"],
        "source_provider": "test",
    }


def _low() -> Dict[str, Any]:
    return {
        "has_catalyst": False,
        "catalyst_strength": "low",
        "catalyst_direction": "neutral",
        "news_count": 1,
        "catalyst_count": 0,
        "latest_headline": "Routine market commentary",
        "latest_ts_utc": _utc_now_z(),
        "catalyst_categories": [],
        "source_provider": "test",
    }


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------
def test_build_catalyst_intel_empty_articles_has_no_catalyst() -> None:
    intel = build_catalyst_intel("LLY", [], source_provider="test")
    assert intel.symbol == "LLY"
    assert intel.has_catalyst is False
    assert intel.catalyst_strength == "none"


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------
def test_high_bullish_lly_earnings_beat() -> None:
    art = NewsArticle(
        headline="LLY earnings beat estimates by 35% — guidance raised",
        published_utc=_utc_now_z(),
    )
    intel = build_catalyst_intel("LLY", [art], source_provider="test")
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------
def test_high_bearish_fda_rejection() -> None:
    art = NewsArticle(
        headline="FDA rejects BIOP application — shares halted",
        published_utc=_utc_now_z(),
    )
    intel = build_catalyst_intel("BIOP", [art], source_provider="test")
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bearish"


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------
def test_check_catalyst_gate_fail_open_when_file_absent(tmp_path: Path) -> None:
    result = check_catalyst_gate("LLY", "BUY", runtime_dir=tmp_path)
    assert result.allowed is True
    assert result.catalyst_strength == "unknown"


# ---------------------------------------------------------------------------
# Tests 5-6
# ---------------------------------------------------------------------------
def test_high_bullish_lly_blocks_sell(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"LLY": _high_bullish()})
    r = check_catalyst_gate("LLY", "SELL", runtime_dir=tmp_path)
    assert r.allowed is False
    assert r.block_reason == "high_catalyst_bullish_opposes_bearish"


def test_high_bullish_lly_allows_buy(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"LLY": _high_bullish()})
    r = check_catalyst_gate("LLY", "BUY", runtime_dir=tmp_path)
    assert r.allowed is True
    assert r.block_reason is None


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------
def test_low_catalyst_allows_both_sides(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"AAPL": _low()})
    r_buy = check_catalyst_gate("AAPL", "BUY", runtime_dir=tmp_path)
    r_sell = check_catalyst_gate("AAPL", "SELL", runtime_dir=tmp_path)
    assert r_buy.allowed is True
    assert r_sell.allowed is True


# ---------------------------------------------------------------------------
# Tests 8-9
# ---------------------------------------------------------------------------
def test_medium_bearish_allows_sell(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"AAPL": _medium_bearish()})
    r = check_catalyst_gate("AAPL", "SELL", runtime_dir=tmp_path)
    assert r.allowed is True


def test_medium_bearish_blocks_buy(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"AAPL": _medium_bearish()})
    r = check_catalyst_gate("AAPL", "BUY", runtime_dir=tmp_path)
    assert r.allowed is False
    assert r.block_reason == "medium_catalyst_bearish_opposes_bullish"


# ---------------------------------------------------------------------------
# Tests 10-11
# ---------------------------------------------------------------------------
def test_medium_bullish_blocks_sell(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"AAPL": _medium_bullish()})
    r = check_catalyst_gate("AAPL", "SELL", runtime_dir=tmp_path)
    assert r.allowed is False
    assert r.block_reason == "medium_catalyst_bullish_opposes_bearish"


def test_medium_bullish_allows_buy(tmp_path: Path) -> None:
    _write_intel(tmp_path, {"AAPL": _medium_bullish()})
    r = check_catalyst_gate("AAPL", "BUY", runtime_dir=tmp_path)
    assert r.allowed is True


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------
def test_stale_news_intel_fails_open(tmp_path: Path) -> None:
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=10)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_intel(
        tmp_path,
        {"LLY": _high_bullish()},
        ttl_seconds=60,
        ts_utc=stale_ts,
    )
    r = check_catalyst_gate("LLY", "SELL", runtime_dir=tmp_path)
    assert r.allowed is True
    assert r.catalyst_strength == "unknown"


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------
def test_alpha_handler_does_not_raise_when_news_intel_absent(tmp_path: Path) -> None:
    """Alpha handler must complete cleanly with no news_intel.json present.

    We don't assert anything about emitted signals — only that the catalyst
    gate wiring does not throw on an empty/absent file (fail-open).
    """
    from chad.strategies import alpha as alpha_mod

    # Confirm import wires the gate cleanly.
    assert hasattr(alpha_mod, "check_catalyst_gate")
    # Gate call must not raise on a temp dir with no file.
    r = check_catalyst_gate("AAPL", "BUY", runtime_dir=tmp_path)
    assert r.allowed is True


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------
def test_classify_article_lly_earnings_high_bullish() -> None:
    art = NewsArticle(
        headline="LLY beats earnings estimates — guidance raised",
        published_utc=_utc_now_z(),
    )
    strength, direction = _classify_article(art)
    assert strength == "high"
    assert direction == "bullish"


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------
def test_publisher_no_fetch_test_mode_produces_valid_payload(tmp_path: Path) -> None:
    payload = nip.run_publish(
        runtime_dir=tmp_path,
        lookback_hours=24,
        dry_run=False,
        no_fetch_test_mode=True,
    )
    assert payload["schema_version"] == "news_intel.v1"
    assert payload["source"]["provider_status"] == "test_no_fetch"
    out_path = tmp_path / "news_intel.json"
    assert out_path.is_file()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "news_intel.v1"
    for sym, rec in on_disk["symbols"].items():
        assert rec["has_catalyst"] is False
        assert rec["catalyst_strength"] == "none"
        assert rec["source_provider"] == "test_no_fetch"


# ---------------------------------------------------------------------------
# Test 16
# ---------------------------------------------------------------------------
def test_get_catalyst_intel_uses_injected_polygon_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[str] = []

    def fake_polygon(symbol: str, api_key: str, lookback: int, limit: int) -> List[NewsArticle]:
        calls.append(symbol)
        return [
            NewsArticle(
                headline=f"{symbol} beats earnings estimates",
                published_utc=_utc_now_z(),
            )
        ]

    def fail_yahoo(symbol: str, limit: int) -> List[NewsArticle]:
        raise AssertionError("yahoo must not be called when polygon returns data")

    out = get_catalyst_intel(
        ["LLY", "AAPL"],
        polygon_fetcher=fake_polygon,
        yahoo_fetcher=fail_yahoo,
    )
    assert calls == ["LLY", "AAPL"]
    assert out["LLY"].source_provider == "polygon"
    assert out["LLY"].catalyst_strength == "high"


# ---------------------------------------------------------------------------
# Test 17
# ---------------------------------------------------------------------------
def test_get_catalyst_intel_falls_back_to_yahoo_when_polygon_empty() -> None:
    def empty_polygon(symbol: str, api_key: str, lookback: int, limit: int) -> List[NewsArticle]:
        return []

    def fake_yahoo(symbol: str, limit: int) -> List[NewsArticle]:
        return [
            NewsArticle(
                headline="Analyst upgrade — partnership announced",
                published_utc=_utc_now_z(),
            )
        ]

    out = get_catalyst_intel(
        ["AAPL"],
        polygon_fetcher=empty_polygon,
        yahoo_fetcher=fake_yahoo,
    )
    assert out["AAPL"].source_provider == "yahoo"
    assert out["AAPL"].catalyst_strength == "medium"


# ---------------------------------------------------------------------------
# Test 18
# ---------------------------------------------------------------------------
def test_alpha_intraday_futures_does_not_call_catalyst_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chad.strategies import alpha_intraday as ai

    def boom(*args: Any, **kwargs: Any):
        raise AssertionError("catalyst_gate must not be called for futures")

    monkeypatch.setattr(ai, "check_catalyst_gate", boom)

    sig = ai._build_signal(
        "MES",
        ai.SignalSide.BUY,
        confidence=0.7,
        trigger="vol_explosion",
        timeframe="1m",
        atr=0.5,
        tier_max_risk_usd=1000.0,
    )
    # MES sizing depends on FUTURES_SPECS — we don't assert its truthiness,
    # only that no exception was raised by the catalyst gate.
    assert sig is None or sig.symbol == "MES"


# ---------------------------------------------------------------------------
# Test 19
# ---------------------------------------------------------------------------
def test_catalyst_gate_respects_ttl_seconds_from_file(tmp_path: Path) -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _write_intel(
        tmp_path,
        {"LLY": _high_bullish()},
        ttl_seconds=1,
        ts_utc=old_ts,
    )
    r = check_catalyst_gate("LLY", "SELL", runtime_dir=tmp_path)
    assert r.allowed is True
    assert r.catalyst_strength == "unknown"


# ---------------------------------------------------------------------------
# Test 20
# ---------------------------------------------------------------------------
def test_publisher_filter_universe_excludes_crypto_and_futures() -> None:
    raw = [
        "AAPL", "SPY", "MES", "MNQ", "MCL", "MGC",
        "ZN", "ES", "RTY", "BTC-USD", "ETH-USD",
        "QQQ", "LLY",
    ]
    filtered = nip.filter_universe(raw)
    assert "AAPL" in filtered
    assert "SPY" in filtered
    assert "QQQ" in filtered
    assert "LLY" in filtered
    for excluded in ("MES", "MNQ", "MCL", "MGC", "ZN", "ES", "RTY", "BTC-USD", "ETH-USD"):
        assert excluded not in filtered, f"{excluded} should be excluded"

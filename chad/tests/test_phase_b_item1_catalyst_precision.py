"""Phase B Item 1 precision tests — symbol relevance filtering for catalyst news."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.market_data import news_intel_publisher as nip
from chad.market_data.catalyst_news_provider import (
    CatalystIntel,
    NewsArticle,
    build_catalyst_intel,
    classify_symbol_relevance,
    get_catalyst_intel,
)
from chad.utils.catalyst_gate import check_catalyst_gate


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Test 1 — Direct symbol in article.symbols creates catalyst
# ---------------------------------------------------------------------------
def test_direct_symbol_in_article_symbols_creates_catalyst() -> None:
    art = NewsArticle(
        headline="LLY earnings beat estimates by 35% — guidance raised",
        published_utc=_utc_now_z(),
        symbols=["LLY"],
    )
    intel = build_catalyst_intel("LLY", [art], source_provider="test")
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"
    assert intel.symbol_relevance == "direct"
    assert intel.relevant_news_count == 1


# ---------------------------------------------------------------------------
# Test 2 — Direct alias creates catalyst
# ---------------------------------------------------------------------------
def test_direct_alias_creates_catalyst() -> None:
    art = NewsArticle(
        headline="Eli Lilly earnings beat estimates and guidance raised",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("LLY", [art], source_provider="test")
    assert intel.symbol_relevance == "direct"
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"


# ---------------------------------------------------------------------------
# Test 3 — Broad market headline does not create symbol catalyst
# ---------------------------------------------------------------------------
def test_broad_market_headline_does_not_create_avgo_catalyst() -> None:
    art = NewsArticle(
        headline="Dow Jones Futures: Stocks rise as Nvidia runs and Cerebras IPO soars",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("AVGO", [art], source_provider="test")
    assert intel.has_catalyst is False
    assert intel.catalyst_strength == "none"
    assert intel.catalyst_direction == "none"
    assert intel.symbol_relevance in ("broad_market", "unknown")
    assert intel.catalyst_count == 0


# ---------------------------------------------------------------------------
# Test 4 — Cerebras IPO headline does not create AMZN catalyst
# ---------------------------------------------------------------------------
def test_cerebras_ipo_headline_does_not_create_amzn_catalyst() -> None:
    art = NewsArticle(
        headline="Cerebras Raises $5.55 Billion IPO at $40 Billion Valuation",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("AMZN", [art], source_provider="test")
    assert intel.has_catalyst is False
    assert intel.catalyst_strength == "none"
    assert intel.catalyst_direction == "none"


# ---------------------------------------------------------------------------
# Test 5 — BAC analyst-source headline about Nvidia is not BAC catalyst
# ---------------------------------------------------------------------------
def test_bac_analyst_source_headline_about_nvidia_is_not_bac_catalyst() -> None:
    art = NewsArticle(
        headline="Bank of America resets Nvidia stock price target for 2026",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    rel = classify_symbol_relevance("BAC", art)
    assert rel in ("weak", "broad_market")
    intel = build_catalyst_intel("BAC", [art], source_provider="test")
    assert intel.has_catalyst is False
    assert intel.catalyst_strength == "none"
    assert intel.catalyst_direction == "none"
    assert intel.symbol_relevance in ("weak", "broad_market")


# ---------------------------------------------------------------------------
# Test 6 — BAC direct earnings headline is a BAC catalyst
# ---------------------------------------------------------------------------
def test_bac_direct_earnings_headline_creates_catalyst() -> None:
    art = NewsArticle(
        headline="Bank of America earnings beat estimates as revenue rises",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("BAC", [art], source_provider="test")
    assert intel.symbol_relevance == "direct"
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"


# ---------------------------------------------------------------------------
# Test 7 — article.symbols overrides a broad-looking headline
# ---------------------------------------------------------------------------
def test_article_symbols_override_broad_looking_headline() -> None:
    art = NewsArticle(
        headline="Dow Jones Futures rise as Broadcom earnings beat estimates",
        published_utc=_utc_now_z(),
        symbols=["AVGO"],
    )
    intel = build_catalyst_intel("AVGO", [art], source_provider="test")
    assert intel.symbol_relevance == "direct"
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"


# ---------------------------------------------------------------------------
# Test 8 — Weak/unknown article increments news_count but not catalyst_count
# ---------------------------------------------------------------------------
def test_weak_unknown_article_increments_news_count_only() -> None:
    art = NewsArticle(
        headline="1 Unstoppable Stock to Buy Right Now",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("AAPL", [art], source_provider="test")
    assert intel.news_count == 1
    assert intel.catalyst_count == 0
    assert intel.relevant_news_count == 0
    assert intel.has_catalyst is False


# ---------------------------------------------------------------------------
# Test 9 — Catalyst gate does not block when has_catalyst is False
# ---------------------------------------------------------------------------
def test_catalyst_gate_does_not_block_when_no_catalyst(tmp_path: Path) -> None:
    payload = {
        "schema_version": "news_intel.v1",
        "ts_utc": _utc_now_z(),
        "ttl_seconds": 3600,
        "lookback_hours": 24,
        "source": {"primary": "polygon", "fallback": "yahoo", "provider_status": "real"},
        "status": "ok",
        "symbols": {
            "AAPL": {
                "has_catalyst": False,
                "catalyst_strength": "none",
                "catalyst_direction": "none",
                "news_count": 1,
                "catalyst_count": 0,
                "latest_headline": "FDA halt rejects merger bankruptcy lawsuit",
                "latest_ts_utc": _utc_now_z(),
                "catalyst_categories": [],
                "source_provider": "test",
                "symbol_relevance": "broad_market",
                "relevant_news_count": 0,
            }
        },
        "summary": {
            "symbols_processed": 1,
            "symbols_with_catalyst": 0,
            "high_strength_count": 0,
            "provider_breakdown": {"polygon": 0, "yahoo": 0, "none": 0, "test_no_fetch": 0},
        },
    }
    (tmp_path / "news_intel.json").write_text(json.dumps(payload), encoding="utf-8")
    r_buy = check_catalyst_gate("AAPL", "BUY", runtime_dir=tmp_path)
    r_sell = check_catalyst_gate("AAPL", "SELL", runtime_dir=tmp_path)
    assert r_buy.allowed is True
    assert r_buy.block_reason is None
    assert r_sell.allowed is True
    assert r_sell.block_reason is None


# ---------------------------------------------------------------------------
# Test 10 — Publisher no-fetch payload includes relevance fields
# ---------------------------------------------------------------------------
def test_publisher_no_fetch_payload_includes_relevance_fields(tmp_path: Path) -> None:
    payload = nip.run_publish(
        runtime_dir=tmp_path,
        lookback_hours=24,
        dry_run=False,
        no_fetch_test_mode=True,
    )
    assert payload["schema_version"] == "news_intel.v1"
    for sym, rec in payload["symbols"].items():
        assert "symbol_relevance" in rec, f"missing symbol_relevance for {sym}"
        assert "relevant_news_count" in rec, f"missing relevant_news_count for {sym}"
        assert rec["symbol_relevance"] == "unknown"
        assert rec["relevant_news_count"] == 0


# ---------------------------------------------------------------------------
# Test 11 — Existing Phase B catalyst tests still importable and runnable
# ---------------------------------------------------------------------------
def test_existing_phase_b_catalyst_tests_still_compatible() -> None:
    from chad.tests import test_phase_b_item1_catalyst as legacy

    art = NewsArticle(
        headline="LLY earnings beat estimates by 35% — guidance raised",
        published_utc=_utc_now_z(),
    )
    intel = build_catalyst_intel("LLY", [art], source_provider="test")
    # original behavior on a direct-relevant ticker-token headline must survive
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"
    # the legacy test module must still expose its helpers
    assert hasattr(legacy, "_high_bullish")
    assert hasattr(legacy, "test_high_bullish_lly_blocks_sell")


# ---------------------------------------------------------------------------
# Test 12 — Yahoo dict normalization preserves symbols
# ---------------------------------------------------------------------------
def test_yahoo_dict_normalization_preserves_symbols() -> None:
    def empty_polygon(symbol: str, api_key: str, lookback: int, limit: int) -> List[NewsArticle]:
        return []

    def fake_yahoo(symbol: str, limit: int) -> List[Dict[str, Any]]:
        return [
            {
                "headline": "Generic-looking story — beats estimates and guidance raised",
                "published_utc": _utc_now_z(),
                "symbols": ["LLY"],
                "source": "yahoo_finance",
                "url": "https://example.test/article",
            }
        ]

    out = get_catalyst_intel(
        ["LLY"],
        polygon_fetcher=empty_polygon,
        yahoo_fetcher=fake_yahoo,
    )
    intel = out["LLY"]
    assert intel.source_provider == "yahoo"
    assert intel.symbol_relevance == "direct"
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.relevant_news_count == 1


# ---------------------------------------------------------------------------
# Test 13 — Polygon normalization preserves tickers field
# ---------------------------------------------------------------------------
def test_polygon_normalization_preserves_tickers() -> None:
    art = NewsArticle(
        headline="Generic mixed-market headline beats estimates",
        published_utc=_utc_now_z(),
        symbols=["LLY"],
    )

    def fake_polygon(symbol: str, api_key: str, lookback: int, limit: int) -> List[NewsArticle]:
        return [art]

    def fail_yahoo(symbol: str, limit: int) -> List[NewsArticle]:
        raise AssertionError("yahoo must not be called when polygon returns data")

    out = get_catalyst_intel(
        ["LLY"],
        polygon_fetcher=fake_polygon,
        yahoo_fetcher=fail_yahoo,
    )
    intel = out["LLY"]
    assert intel.source_provider == "polygon"
    assert intel.symbol_relevance == "direct"
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.relevant_news_count == 1

"""Phase B Item 1 safety tests — confirmed catalyst gate relevance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from chad.market_data import news_intel_publisher as nip
from chad.market_data.catalyst_news_provider import (
    NewsArticle,
    build_catalyst_intel,
)
from chad.utils.catalyst_gate import check_catalyst_gate


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_intel(runtime_dir: Path, symbols: Dict[str, Dict[str, Any]]) -> None:
    payload = {
        "schema_version": "news_intel.v1",
        "ts_utc": _utc_now_z(),
        "ttl_seconds": 3600,
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


# ---------------------------------------------------------------------------
# Test 1 — Provider ticker tag alone does not confirm relevance
# ---------------------------------------------------------------------------
def test_bac_chime_headline_with_provider_ticker_is_not_confirmed() -> None:
    art = NewsArticle(
        headline="Chime Finally Turns Profitable—But Risks Remain",
        published_utc=_utc_now_z(),
        symbols=["BAC"],
    )
    intel = build_catalyst_intel("BAC", [art], source_provider="polygon")
    # Provider ticker tag keeps symbol_relevance="direct" and relevant_news_count=1
    # but the gate must NOT block — headline does not confirm BAC.
    assert intel.symbol_relevance == "direct"
    assert intel.relevant_news_count == 1
    assert intel.confirmed_gate_relevant is False
    assert intel.has_catalyst is False
    assert intel.catalyst_strength == "none"


# ---------------------------------------------------------------------------
# Test 2 — Direct alias in headline confirms relevance
# ---------------------------------------------------------------------------
def test_bac_direct_earnings_headline_is_confirmed() -> None:
    art = NewsArticle(
        headline="Bank of America earnings beat estimates as revenue rises",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("BAC", [art], source_provider="test")
    assert intel.confirmed_gate_relevant is True
    assert intel.has_catalyst is True
    assert intel.catalyst_strength == "high"
    assert intel.catalyst_direction == "bullish"


# ---------------------------------------------------------------------------
# Test 3 — Confirmed but neutral direction does not block either side
# ---------------------------------------------------------------------------
def test_googl_unstoppable_alphabet_is_confirmed_neutral(tmp_path: Path) -> None:
    art = NewsArticle(
        headline=(
            "1 Unstoppable Stock to Buy Before It Joins Nvidia, Alphabet, and Apple in the "
            "Trillion-Dollar Club"
        ),
        published_utc=_utc_now_z(),
        symbols=["GOOGL"],
    )
    intel = build_catalyst_intel("GOOGL", [art], source_provider="polygon")
    # Alphabet alias confirms relevance, but no high/medium catalyst keywords
    # apply and no bullish/bearish direction is established → gate stays open.
    assert intel.confirmed_gate_relevant is True
    assert intel.catalyst_direction in ("neutral", "none")

    _write_intel(
        tmp_path,
        {
            "GOOGL": {
                "has_catalyst": intel.has_catalyst,
                "catalyst_strength": intel.catalyst_strength,
                "catalyst_direction": intel.catalyst_direction,
                "news_count": intel.news_count,
                "catalyst_count": intel.catalyst_count,
                "latest_headline": intel.latest_headline,
                "latest_ts_utc": intel.latest_ts_utc,
                "catalyst_categories": list(intel.catalyst_categories),
                "source_provider": intel.source_provider,
                "symbol_relevance": intel.symbol_relevance,
                "relevant_news_count": intel.relevant_news_count,
                "confirmed_gate_relevant": intel.confirmed_gate_relevant,
            }
        },
    )
    r_buy = check_catalyst_gate("GOOGL", "BUY", runtime_dir=tmp_path)
    r_sell = check_catalyst_gate("GOOGL", "SELL", runtime_dir=tmp_path)
    assert r_buy.allowed is True
    assert r_sell.allowed is True


# ---------------------------------------------------------------------------
# Test 4 — AVGO ticker token in headline confirms relevance
# ---------------------------------------------------------------------------
def test_avgo_ticker_in_headline_is_confirmed() -> None:
    art = NewsArticle(
        headline="Jim Cramer Makes Big Claim About Broadcom (AVGO) CEO",
        published_utc=_utc_now_z(),
        symbols=[],
    )
    intel = build_catalyst_intel("AVGO", [art], source_provider="test")
    assert intel.confirmed_gate_relevant is True
    assert intel.symbol_relevance == "direct"


# ---------------------------------------------------------------------------
# Test 5 — Provider tickers alone cannot block trades via the gate
# ---------------------------------------------------------------------------
def test_unconfirmed_high_bearish_does_not_block_buy(tmp_path: Path) -> None:
    _write_intel(
        tmp_path,
        {
            "BAC": {
                "has_catalyst": True,
                "catalyst_strength": "high",
                "catalyst_direction": "bearish",
                "news_count": 10,
                "catalyst_count": 2,
                "latest_headline": "Chime Finally Turns Profitable—But Risks Remain",
                "latest_ts_utc": _utc_now_z(),
                "catalyst_categories": [],
                "source_provider": "polygon",
                "symbol_relevance": "direct",
                "relevant_news_count": 5,
                "confirmed_gate_relevant": False,
            }
        },
    )
    r = check_catalyst_gate("BAC", "BUY", runtime_dir=tmp_path)
    assert r.allowed is True
    assert r.block_reason is None


# ---------------------------------------------------------------------------
# Test 6 — Confirmed high bearish blocks BUY, allows SELL
# ---------------------------------------------------------------------------
def test_confirmed_high_bearish_blocks_buy_allows_sell(tmp_path: Path) -> None:
    _write_intel(
        tmp_path,
        {
            "BAC": {
                "has_catalyst": True,
                "catalyst_strength": "high",
                "catalyst_direction": "bearish",
                "news_count": 3,
                "catalyst_count": 2,
                "latest_headline": "Bank of America misses earnings — shares halted",
                "latest_ts_utc": _utc_now_z(),
                "catalyst_categories": ["earnings"],
                "source_provider": "test",
                "symbol_relevance": "direct",
                "relevant_news_count": 2,
                "confirmed_gate_relevant": True,
            }
        },
    )
    r_buy = check_catalyst_gate("BAC", "BUY", runtime_dir=tmp_path)
    r_sell = check_catalyst_gate("BAC", "SELL", runtime_dir=tmp_path)
    assert r_buy.allowed is False
    assert r_buy.block_reason == "high_catalyst_bearish_opposes_bullish"
    assert r_sell.allowed is True


# ---------------------------------------------------------------------------
# Test 7 — Confirmed high bullish blocks SELL, allows BUY
# ---------------------------------------------------------------------------
def test_confirmed_high_bullish_blocks_sell_allows_buy(tmp_path: Path) -> None:
    _write_intel(
        tmp_path,
        {
            "LLY": {
                "has_catalyst": True,
                "catalyst_strength": "high",
                "catalyst_direction": "bullish",
                "news_count": 3,
                "catalyst_count": 2,
                "latest_headline": "Eli Lilly earnings beat estimates and guidance raised",
                "latest_ts_utc": _utc_now_z(),
                "catalyst_categories": ["earnings"],
                "source_provider": "test",
                "symbol_relevance": "direct",
                "relevant_news_count": 2,
                "confirmed_gate_relevant": True,
            }
        },
    )
    r_buy = check_catalyst_gate("LLY", "BUY", runtime_dir=tmp_path)
    r_sell = check_catalyst_gate("LLY", "SELL", runtime_dir=tmp_path)
    assert r_buy.allowed is True
    assert r_sell.allowed is False
    assert r_sell.block_reason == "high_catalyst_bullish_opposes_bearish"


# ---------------------------------------------------------------------------
# Test 8 — Missing confirmed_gate_relevant field fails open
# ---------------------------------------------------------------------------
def test_missing_confirmed_gate_relevant_fails_open(tmp_path: Path) -> None:
    _write_intel(
        tmp_path,
        {
            "AAPL": {
                "has_catalyst": True,
                "catalyst_strength": "high",
                "catalyst_direction": "bearish",
                "news_count": 3,
                "catalyst_count": 2,
                "latest_headline": "Apple misses earnings",
                "latest_ts_utc": _utc_now_z(),
                "catalyst_categories": ["earnings"],
                "source_provider": "test",
                # confirmed_gate_relevant deliberately absent — old payload shape
            }
        },
    )
    r = check_catalyst_gate("AAPL", "BUY", runtime_dir=tmp_path)
    assert r.allowed is True
    assert r.block_reason is None


# ---------------------------------------------------------------------------
# Test 9 — Publisher no-fetch payload includes confirmed_gate_relevant=False
# ---------------------------------------------------------------------------
def test_publisher_no_fetch_payload_includes_confirmed_gate_relevant(tmp_path: Path) -> None:
    payload = nip.run_publish(
        runtime_dir=tmp_path,
        lookback_hours=24,
        dry_run=False,
        no_fetch_test_mode=True,
    )
    assert payload["schema_version"] == "news_intel.v1"
    for sym, rec in payload["symbols"].items():
        assert "confirmed_gate_relevant" in rec, f"missing confirmed_gate_relevant for {sym}"
        assert rec["confirmed_gate_relevant"] is False


# ---------------------------------------------------------------------------
# Test 10 — GLD broad gold headline is not confirmed
# ---------------------------------------------------------------------------
def test_gld_broad_gold_headline_not_confirmed() -> None:
    art = NewsArticle(
        headline="Stocks rise as gold prices climb to record",
        published_utc=_utc_now_z(),
        symbols=["GLD"],
    )
    intel = build_catalyst_intel("GLD", [art], source_provider="polygon")
    # Provider tagged GLD, but headline mentions neither GLD ticker nor any
    # GLD alias ("SPDR Gold", "Gold ETF") — gate must stay open.
    assert intel.confirmed_gate_relevant is False
    assert intel.has_catalyst is False

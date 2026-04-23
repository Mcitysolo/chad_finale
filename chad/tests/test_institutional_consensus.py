"""Tests for chad.analytics.institutional_consensus."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.analytics.institutional_consensus import (
    InstitutionalConsensus,
    NAME_TO_TICKER,
)


# ---------------------------------------------------------------------------
# Helpers — minimal holdings fixtures that resemble SEC13FFetcher output
# ---------------------------------------------------------------------------

AAPL_CUSIP = "037833100"
MSFT_CUSIP = "594918104"
NVDA_CUSIP = "67066G104"
UNKNOWN_CUSIP = "999999999"


def _h(name: str, cusip: str, value: float, shares: int) -> Dict[str, Any]:
    return {
        "name_of_issuer": name,
        "cusip": cusip,
        "value_usd": value,
        "shares": shares,
    }


def _fund_payload(*holdings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "holdings": list(holdings),
        "report_date": "2026-02-14",
        "accession": "x",
        "source": "cache",
    }


# ---------------------------------------------------------------------------
# Conviction scoring
# ---------------------------------------------------------------------------

def test_conviction_score_higher_for_widely_held() -> None:
    # AAPL held by 3 funds, NVDA by 1 — breadth dominates depth for AAPL
    all_holdings = {
        "fund_a": _fund_payload(
            _h("APPLE INC", AAPL_CUSIP, 10_000_000, 100),
            _h("NVIDIA CORP", NVDA_CUSIP, 10_000_000, 50),
        ),
        "fund_b": _fund_payload(_h("APPLE INC", AAPL_CUSIP, 5_000_000, 50)),
        "fund_c": _fund_payload(_h("APPLE INC", AAPL_CUSIP, 5_000_000, 50)),
    }
    result = InstitutionalConsensus().compute_consensus(all_holdings)
    by_sym = {e.symbol: e for e in result}
    assert "AAPL" in by_sym
    assert "NVDA" in by_sym
    assert by_sym["AAPL"].conviction_score > by_sym["NVDA"].conviction_score
    assert by_sym["AAPL"].fund_count == 3
    assert by_sym["NVDA"].fund_count == 1


def test_weights_sum_to_1() -> None:
    all_holdings = {
        "fund_a": _fund_payload(
            _h("APPLE INC", AAPL_CUSIP, 10_000_000, 100),
            _h("MICROSOFT CORP", MSFT_CUSIP, 8_000_000, 80),
        ),
        "fund_b": _fund_payload(
            _h("NVIDIA CORP", NVDA_CUSIP, 6_000_000, 40),
            _h("APPLE INC", AAPL_CUSIP, 4_000_000, 40),
        ),
    }
    ic = InstitutionalConsensus()
    entries = ic.compute_consensus(all_holdings)
    weights = ic.get_consensus_weights(entries)
    assert weights, "expected non-empty weights"
    assert math.isclose(sum(weights.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_top_25_returns_correct_count() -> None:
    # Build 40 synthetic unique issuers; request top_n=25
    holdings = []
    for i in range(40):
        cusip = f"{i:09d}"
        # Use AAPL as the resolved name — we'll override symbol lookup by
        # pre-seeding NAME_TO_TICKER via the explicit cusips? Simpler:
        # use real-ish issuer names that resolve. Since NAME_TO_TICKER is
        # finite, cycle through known-resolving issuer names.
        names = list(NAME_TO_TICKER.keys())
        name = names[i % len(names)]
        holdings.append(_h(name, cusip, 1_000_000 * (i + 1), 100 * (i + 1)))
    all_holdings = {"fund_a": _fund_payload(*holdings)}
    ic = InstitutionalConsensus()
    entries = ic.compute_consensus(all_holdings, top_n=25)
    assert len(entries) == 25
    # Top entries should be ranked by conviction (desc)
    scores = [e.conviction_score for e in entries]
    assert scores == sorted(scores, reverse=True)


def test_unresolved_dropped_by_default() -> None:
    all_holdings = {
        "fund_a": _fund_payload(
            _h("APPLE INC", AAPL_CUSIP, 10_000_000, 100),
            _h("Mysterious Private Co", UNKNOWN_CUSIP, 50_000_000, 500),
        ),
    }
    ic = InstitutionalConsensus()
    entries = ic.compute_consensus(all_holdings)
    syms = [e.symbol for e in entries]
    assert "AAPL" in syms
    assert not any(s.startswith("UNRESOLVED:") for s in syms)


def test_unresolved_kept_when_requested() -> None:
    all_holdings = {
        "fund_a": _fund_payload(
            _h("Mysterious Private Co", UNKNOWN_CUSIP, 50_000_000, 500),
        ),
    }
    ic = InstitutionalConsensus()
    entries = ic.compute_consensus(all_holdings, include_unresolved=True)
    assert len(entries) == 1
    assert entries[0].symbol.startswith("UNRESOLVED:")


def test_empty_input_returns_empty() -> None:
    ic = InstitutionalConsensus()
    assert ic.compute_consensus({}) == []
    assert ic.get_consensus_weights([]) == {}


def test_googl_goog_deduplication() -> None:
    # Alphabet class A (CUSIP 02079K107) and class C (02079K305) are tracked
    # under different CUSIPs but can resolve to the same ticker once the
    # issuer name normalizes (both commonly labelled "ALPHABET INC").
    # Expect a single GOOGL entry with fund_holders unioned and totals summed.
    goog_a_cusip = "02079K107"
    goog_c_cusip = "02079K305"
    all_holdings = {
        "fund_a": _fund_payload(
            _h("ALPHABET INC", goog_a_cusip, 10_000_000, 100),
        ),
        "fund_b": _fund_payload(
            _h("ALPHABET INC", goog_c_cusip, 6_000_000, 60),
        ),
        "fund_c": _fund_payload(
            _h("ALPHABET INC", goog_a_cusip, 4_000_000, 40),
            _h("ALPHABET INC", goog_c_cusip, 2_000_000, 20),
        ),
    }
    entries = InstitutionalConsensus().compute_consensus(all_holdings)
    googl = [e for e in entries if e.symbol == "GOOGL"]
    assert len(googl) == 1, f"expected 1 GOOGL entry, got {len(googl)}"
    e = googl[0]
    assert set(e.fund_holders) == {"fund_a", "fund_b", "fund_c"}
    assert e.fund_count == 3
    assert e.total_value_usd == 22_000_000
    assert e.total_shares == 220


def test_unresolved_excluded_from_weights() -> None:
    all_holdings = {
        "fund_a": _fund_payload(
            _h("APPLE INC", AAPL_CUSIP, 10_000_000, 100),
            _h("Mysterious Private Co", UNKNOWN_CUSIP, 50_000_000, 500),
        ),
    }
    ic = InstitutionalConsensus()
    entries = ic.compute_consensus(all_holdings, include_unresolved=True)
    weights = ic.get_consensus_weights(entries)
    assert "AAPL" in weights
    assert all(not k.startswith("UNRESOLVED:") for k in weights)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_write_and_load_cache_round_trip(tmp_path: Path) -> None:
    all_holdings = {
        "fund_a": _fund_payload(
            _h("APPLE INC", AAPL_CUSIP, 10_000_000, 100),
            _h("MICROSOFT CORP", MSFT_CUSIP, 8_000_000, 80),
        ),
        "fund_b": _fund_payload(
            _h("APPLE INC", AAPL_CUSIP, 5_000_000, 50),
        ),
    }
    ic = InstitutionalConsensus()
    entries = ic.compute_consensus(all_holdings)
    weights = ic.get_consensus_weights(entries)

    out_path = tmp_path / "institutional_consensus.json"
    ic.write_cache(entries, weights, funds_included=["fund_a", "fund_b"], path=out_path)
    loaded = ic.load_cache(out_path)

    assert loaded is not None
    assert loaded["schema_version"] == "institutional_consensus.v1"
    assert "updated_ts_utc" in loaded
    assert loaded["funds_included"] == ["fund_a", "fund_b"]
    assert loaded["weights"] == weights
    assert len(loaded["top_holdings"]) == len(entries)


def test_load_cache_missing_returns_none(tmp_path: Path) -> None:
    ic = InstitutionalConsensus()
    assert ic.load_cache(tmp_path / "nonexistent.json") is None

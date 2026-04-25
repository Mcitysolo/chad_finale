"""Tests for normalize_paper_fill_evidence — the single chokepoint that
guarantees zero PendingSubmit / unknown / zero-price records ever land in
data/fills/FILLS_*.ndjson in paper mode.
"""

from __future__ import annotations

import json
import pytest

from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
    _strip_futures_month_code,
    _lookup_paper_fill_price,
    PRICE_CACHE_PATH,
)


@pytest.fixture
def fake_price_cache(tmp_path, monkeypatch):
    """Patch PRICE_CACHE_PATH to a tmp file with deterministic prices."""
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(json.dumps({
        "prices": {
            "MES": 7100.0,
            "MGC": 4700.0,
            "MNQ": 26000.0,
            "MCL": 95.0,
            "ZN": 111.0,
            "SPY": 700.0,
            "AAPL": 270.0,
        },
        "ts_utc": "2026-04-25T00:00:00Z",
        "ttl_seconds": 300,
    }))
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.PRICE_CACHE_PATH",
        cache_path,
    )
    return cache_path


def test_pendingsubmit_futures_normalized_to_paper_fill(fake_price_cache):
    """status=PendingSubmit + fill_price=0 + symbol=MES → paper_fill at cache price."""
    ev = PaperExecEvidence(
        symbol="MES",
        side="SELL",
        quantity=2.0,
        fill_price=0.0,
        status="PendingSubmit",
        is_live=False,
        asset_class="",
        expected_price=0.0,
    )
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"
    assert ev.fill_price == pytest.approx(7100.0)
    assert ev.asset_class == "futures"


def test_month_coded_futures_symbol_resolves_via_root(fake_price_cache):
    """symbol=MGCK6 (May 2026 micro gold) should fetch MGC's cached price."""
    ev = PaperExecEvidence(
        symbol="MGCK6",
        side="BUY",
        quantity=1.0,
        fill_price=0.0,
        status="error",
        is_live=False,
        asset_class="unknown",
    )
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"
    assert ev.fill_price == pytest.approx(4700.0)
    assert ev.asset_class == "futures"


def test_etf_unknown_asset_class_resolves(fake_price_cache):
    """status=error + asset_class=unknown + symbol=SPY → paper_fill, etf."""
    ev = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        fill_price=0.0,
        status="error",
        is_live=False,
        asset_class="unknown",
    )
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"
    assert ev.fill_price == pytest.approx(700.0)
    assert ev.asset_class == "etf"


def test_pending_with_no_resolvable_price_raises(fake_price_cache):
    """Unknown symbol + no expected_price + pending status → ValueError."""
    ev = PaperExecEvidence(
        symbol="ZZZNOTREAL",
        side="BUY",
        quantity=1.0,
        fill_price=0.0,
        status="PendingSubmit",
        is_live=False,
        asset_class="",
        expected_price=0.0,
    )
    with pytest.raises(ValueError, match="no positive fill price"):
        normalize_paper_fill_evidence(ev)


def test_live_mode_evidence_passes_through_unchanged(fake_price_cache):
    """Live-mode records must NOT be rewritten — broker is source of truth."""
    ev = PaperExecEvidence(
        symbol="MES",
        side="BUY",
        quantity=2.0,
        fill_price=0.0,
        status="PendingSubmit",
        is_live=True,
        asset_class="",
    )
    normalize_paper_fill_evidence(ev)
    # is_live=True returns early after asset_class fix — status & price untouched.
    assert ev.status == "PendingSubmit"
    assert ev.fill_price == 0.0
    # asset_class is still resolved (it runs before the live-mode early return).
    assert ev.asset_class == "futures"


def test_strip_futures_month_code():
    """Futures contract-month codes strip to their root."""
    assert _strip_futures_month_code("MGCK6") == "MGC"
    assert _strip_futures_month_code("MES2606") == "MES"
    assert _strip_futures_month_code("ZNH6") == "ZN"
    assert _strip_futures_month_code("MES") == "MES"  # already root
    assert _strip_futures_month_code("SPY") == "SPY"  # not a futures root
    assert _strip_futures_month_code("") == ""


def test_lookup_paper_fill_price_uses_root_for_month_code(fake_price_cache):
    """Price cache lookup resolves MGCK6 via the MGC entry."""
    assert _lookup_paper_fill_price("MGCK6") == pytest.approx(4700.0)
    assert _lookup_paper_fill_price("MES") == pytest.approx(7100.0)
    assert _lookup_paper_fill_price("FAKE") == 0.0


def test_expected_price_fallback_when_cache_miss(fake_price_cache):
    """If cache cannot resolve, fall back to expected_price."""
    ev = PaperExecEvidence(
        symbol="MES",
        side="BUY",
        quantity=2.0,
        fill_price=0.0,
        expected_price=7150.0,
        status="PendingSubmit",
        is_live=False,
        asset_class="",
    )
    # MES is in the cache, so price comes from cache (7100), not expected.
    normalize_paper_fill_evidence(ev)
    assert ev.fill_price == pytest.approx(7100.0)

    ev2 = PaperExecEvidence(
        symbol="ZZZNOTREAL",
        side="BUY",
        quantity=1.0,
        fill_price=0.0,
        expected_price=42.0,
        status="error",
        is_live=False,
        asset_class="equity",
    )
    normalize_paper_fill_evidence(ev2)
    # No cache match; falls back to expected_price.
    assert ev2.fill_price == pytest.approx(42.0)
    assert ev2.status == "paper_fill"

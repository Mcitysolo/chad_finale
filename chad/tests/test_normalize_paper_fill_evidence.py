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


# ---------------------------------------------------------------------------
# Pre-Epoch-2 blocker fix tests — price-sanity guard / placeholder rejection
# ---------------------------------------------------------------------------


def test_paper_executor_uses_price_cache_for_spy(fake_price_cache):
    """When fill_price is missing, SPY fill resolves to the cached price."""
    ev = PaperExecEvidence(
        symbol="SPY",
        side="SELL",
        quantity=10.0,
        fill_price=0.0,
        status="filled",
        is_live=False,
        asset_class="equity",
        expected_price=0.0,
    )
    normalize_paper_fill_evidence(ev)
    assert ev.fill_price == pytest.approx(700.0)


def test_paper_executor_rejects_placeholder_100_when_cache_real(fake_price_cache):
    """A proposed fill_price=100.0 for SPY (cache=700.0) is flagged untrusted.
    The 50% deviation guard catches placeholder prices that would otherwise
    silently feed bogus realized PnL into the trade closer."""
    ev = PaperExecEvidence(
        symbol="SPY",
        side="SELL",
        quantity=10.0,
        fill_price=100.0,
        expected_price=100.0,
        status="filled",
        is_live=False,
        asset_class="equity",
    )
    normalize_paper_fill_evidence(ev)
    assert isinstance(ev.extra, dict)
    assert ev.extra.get("pnl_untrusted") is True
    reason = ev.extra.get("pnl_untrusted_reason", "")
    assert "100" in reason and "700" in reason
    assert any(
        str(t).strip().lower() == "pnl_untrusted" for t in (ev.tags or ())
    )


def test_invalid_fill_not_trusted_closed_trade(tmp_path, monkeypatch):
    """trade_closer._extract_fill must skip records flagged pnl_untrusted —
    they are not allowed to feed FIFO matching."""
    from chad.execution.trade_closer import _extract_fill

    bad = {
        "payload": {
            "fill_id": "abc123",
            "strategy": "delta",
            "symbol": "SPY",
            "side": "SELL",
            "quantity": 10.0,
            "fill_price": 100.0,
            "fill_time_utc": "2026-05-03T11:48:55.976Z",
            "extra": {"pnl_untrusted": True, "pnl_untrusted_reason": "placeholder"},
            "tags": ["paper", "filled", "delta", "pnl_untrusted"],
            "status": "filled",
            "reject": False,
        }
    }
    assert _extract_fill(bad) is None

    good = {
        "payload": {
            "fill_id": "def456",
            "strategy": "delta",
            "symbol": "SPY",
            "side": "SELL",
            "quantity": 10.0,
            "fill_price": 700.0,
            "fill_time_utc": "2026-05-03T11:48:55.976Z",
            "extra": {},
            "tags": ["paper", "filled", "delta"],
            "status": "filled",
            "reject": False,
        }
    }
    fill = _extract_fill(good)
    assert fill is not None
    assert fill["fill_price"] == pytest.approx(700.0)


def test_in_range_fill_price_is_trusted(fake_price_cache):
    """Fill within 50% of cache passes through clean — no untrusted flag."""
    ev = PaperExecEvidence(
        symbol="SPY",
        side="SELL",
        quantity=10.0,
        fill_price=720.0,  # cache is 700.0; ~3% deviation
        expected_price=720.0,
        status="filled",
        is_live=False,
        asset_class="equity",
    )
    normalize_paper_fill_evidence(ev)
    extra = ev.extra if isinstance(ev.extra, dict) else {}
    assert not extra.get("pnl_untrusted")
    assert ev.fill_price == pytest.approx(720.0)


def test_paper_executor_never_writes_placeholder_100_as_filled_when_cache_valid(
    fake_price_cache,
):
    """Placeholder fill_price=100 for SPY (cache=700) must never be persisted
    as a trusted ``filled`` record. The normalizer demotes it to ``rejected``
    + reject=True so trade_closer cannot consume it via FIFO matching."""
    ev = PaperExecEvidence(
        symbol="SPY",
        side="SELL",
        quantity=10.0,
        fill_price=100.0,
        expected_price=100.0,
        status="filled",
        is_live=False,
        asset_class="equity",
    )
    normalize_paper_fill_evidence(ev)
    # Status must NOT remain "filled" — it has been demoted.
    assert str(ev.status).lower() != "filled"
    assert str(ev.status).lower() == "rejected"
    assert ev.reject is True
    # Untrusted flag must still be set so multiple guards catch it.
    assert isinstance(ev.extra, dict)
    assert ev.extra.get("pnl_untrusted") is True


def test_untrusted_placeholder_fill_is_not_fifo_matched(
    fake_price_cache, tmp_path, monkeypatch
):
    """End-to-end: a placeholder $100 SPY fill written through the normalizer
    must be filtered by trade_closer._extract_fill before FIFO matching can
    produce a phantom closed trade."""
    import datetime as _dt
    from chad.execution import paper_exec_evidence_writer as pew
    from chad.execution.trade_closer import TradeCloser, _extract_fill

    fills_dir = tmp_path / "fills"
    fees_dir = tmp_path / "fees"
    metrics_dir = tmp_path / "execution_metrics"
    fills_dir.mkdir(parents=True, exist_ok=True)
    fees_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(pew, "FILLS_DIR", fills_dir)
    monkeypatch.setattr(pew, "FEES_DIR", fees_dir)
    monkeypatch.setattr(pew, "EXEC_METRICS_DIR", metrics_dir)

    # 1) Build placeholder fill, normalize, write through the real writer.
    bad = PaperExecEvidence(
        symbol="SPY",
        strategy="delta",
        source_strategies=["delta"],
        side="SELL",
        quantity=10.0,
        fill_price=100.0,
        expected_price=100.0,
        status="filled",
        is_live=False,
        asset_class="equity",
        fill_time_utc="2026-05-04T14:00:00+00:00",
        entry_time_utc="2026-05-04T14:00:00+00:00",
        exit_time_utc="2026-05-04T14:00:00+00:00",
    )
    pew.write_paper_exec_evidence(bad)

    # 2) Run trade_closer over the persisted record on whichever YYYYMMDD
    # the writer used (today's UTC date).
    day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    fills_path = fills_dir / f"FILLS_{day}.ndjson"
    assert fills_path.exists(), f"writer did not produce {fills_path}"

    closer = TradeCloser(
        fills_dir=fills_dir,
        trades_dir=tmp_path / "trades",
        state_path=tmp_path / "state.json",
    )
    closer.load_state()
    closed = closer.process_fills(day)
    assert closed == [], (
        "trade_closer must not produce a closed trade from an untrusted "
        "placeholder fill"
    )

    # 3) Direct _extract_fill check: the on-disk record reads as None.
    raw = json.loads(fills_path.read_text(encoding="utf-8").splitlines()[0])
    assert _extract_fill(raw) is None

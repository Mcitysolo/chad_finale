"""PA-EP2a — honest, fail-open, zero-new-I/O submit_quote stamp.

Covers chad/execution/paper_exec_evidence_writer:
- build_submit_quote_stamp(): cache hit / symbol absent / None cache / garbage
- the normalizer attaching ev.extra['submit_quote'] and it flowing into the
  EXECUTION_METRICS payload's extra
- fail-open: a raising price-cache read does NOT propagate; evidence proceeds
- invariant: bid/ask/spread are null in EVERY code path (no NBBO fabricated)

No IBKR connection or network required — the price cache is supplied in-memory
or monkeypatched, so every test is deterministic.
"""

from __future__ import annotations

import pytest

import chad.execution.paper_exec_evidence_writer as pew
from chad.execution.paper_exec_evidence_writer import (
    EvidencePayloadFactory,
    PaperExecEvidence,
    build_submit_quote_stamp,
    normalize_paper_fill_evidence,
)

_SUBMIT = "2026-06-15T00:05:00+00:00"
_CACHE = {"prices": {"AAPL": 291.52, "MES": 7100.0}, "ts_utc": "2026-06-15T00:02:59Z"}


def _assert_no_nbbo(stamp) -> None:
    assert stamp["bid"] is None
    assert stamp["ask"] is None
    assert stamp["spread"] is None


# 1. cache hit --------------------------------------------------------------

def test_cache_hit_sets_ref_price_no_nbbo() -> None:
    stamp = build_submit_quote_stamp("AAPL", _SUBMIT, _CACHE)
    _assert_no_nbbo(stamp)
    assert stamp["ref_price"] == 291.52
    assert stamp["source"] == "price_cache_mid_or_last"
    assert stamp["confidence"] == "ref_only_no_nbbo"
    assert stamp["quote_ts"] == "2026-06-15T00:02:59Z"
    assert stamp["quote_age_s"] == pytest.approx(121.0)  # 00:05:00 - 00:02:59
    assert stamp["quote_ttl_s"] == 300


def test_cache_hit_uses_futures_root() -> None:
    # MESM6 (contract-month) resolves to the MES root scalar.
    stamp = build_submit_quote_stamp("MESM6", _SUBMIT, _CACHE)
    _assert_no_nbbo(stamp)
    assert stamp["ref_price"] == 7100.0
    assert stamp["confidence"] == "ref_only_no_nbbo"


# 2. symbol absent ----------------------------------------------------------

def test_symbol_absent_is_unavailable() -> None:
    stamp = build_submit_quote_stamp("NVDA", _SUBMIT, _CACHE)
    _assert_no_nbbo(stamp)
    assert stamp["ref_price"] is None
    assert stamp["quote_ts"] is None
    assert stamp["quote_age_s"] is None
    assert stamp["source"] == "unavailable"
    assert stamp["confidence"] == "none"
    assert stamp["quote_ttl_s"] == 300


def test_symbol_absent_evidence_still_built(monkeypatch) -> None:
    # Cache present but missing the symbol → unavailable marker, record intact.
    monkeypatch.setattr(pew, "_read_price_cache_raw", lambda: {"prices": {}, "ts_utc": "x"})
    ev = PaperExecEvidence(
        symbol="NVDA", side="BUY", quantity=1.0, fill_price=500.0,
        expected_price=500.0, status="paper_fill", is_live=False,
        asset_class="equity", fill_time_utc=_SUBMIT,
        strategy="alpha", source_strategies=["alpha"],
    )
    normalize_paper_fill_evidence(ev)
    stamp = ev.extra["submit_quote"]
    _assert_no_nbbo(stamp)
    assert stamp["source"] == "unavailable"
    # the stamp flows into the persisted EXECUTION_METRICS extra
    payload = EvidencePayloadFactory().build_execution_metric_payload(ev)
    assert payload["extra"]["submit_quote"]["source"] == "unavailable"


# 3. read raises (fail-open) ------------------------------------------------

def test_read_raises_is_fail_open(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(pew, "_read_price_cache_raw", _boom)
    ev = PaperExecEvidence(
        symbol="AAPL", side="BUY", quantity=1.0, fill_price=290.0,
        expected_price=290.0, status="paper_fill", is_live=False,
        asset_class="equity", fill_time_utc=_SUBMIT,
    )
    # Must NOT raise — order + evidence proceed untouched.
    normalize_paper_fill_evidence(ev)
    stamp = ev.extra["submit_quote"]
    _assert_no_nbbo(stamp)
    assert stamp["source"] == "unavailable"
    assert stamp["confidence"] == "none"
    assert stamp["ref_price"] is None


# 4. invariant: never a non-null bid/ask/spread -----------------------------

@pytest.mark.parametrize(
    "symbol,cache",
    [
        ("AAPL", _CACHE),                                   # hit
        ("NVDA", _CACHE),                                   # absent
        ("AAPL", None),                                     # no cache
        ("AAPL", {"prices": "not-a-dict"}),               # garbage prices
        ("AAPL", {"prices": {"AAPL": -5.0}}),             # non-positive scalar
        ("", _CACHE),                                       # empty symbol
        ("AAPL", {"prices": {"AAPL": 291.52}}),           # hit, no ts_utc
    ],
)
def test_bid_ask_spread_always_null(symbol, cache) -> None:
    stamp = build_submit_quote_stamp(symbol, _SUBMIT, cache)
    _assert_no_nbbo(stamp)
    # null trio holds even on the happy path
    assert "bid" in stamp and "ask" in stamp and "spread" in stamp


def test_cache_hit_without_ts_has_null_quote_ts() -> None:
    stamp = build_submit_quote_stamp("AAPL", _SUBMIT, {"prices": {"AAPL": 291.52}})
    _assert_no_nbbo(stamp)
    assert stamp["ref_price"] == 291.52
    assert stamp["confidence"] == "ref_only_no_nbbo"
    assert stamp["quote_ts"] is None
    assert stamp["quote_age_s"] is None

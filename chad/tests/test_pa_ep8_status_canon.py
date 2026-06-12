"""PA-EP8 — status canonicalization at the evidence chokepoint.

The paper write path historically persisted multiple literals for the same
event: genuine synchronous fills landed as "FILLED" (IBKR) or "filled", while
the pending→fill translation and the harvester both emit the canonical
"paper_fill". PA-EP8 collapses the genuine-fill literals to "paper_fill"
EXACTLY ONCE inside normalize_paper_fill_evidence, AFTER the existing pending
translation and BEFORE the modeled-commission fee predicate, preserving the
pre-canon value on extra.status_raw. Unknown literals pass through unchanged
and emit a loud STATUS_CANON_UNMAPPED marker.

lifecycle_replay_engine.py / lifecycle_replay_coverage.py are intentionally
OUT of scope (PRE-A proved the replay diagnostic is observability-only and
non-blocking; the FILLED subset freezing historically is accepted).

Test layers:
  * per-literal mapping (FILLED→paper_fill, filled→paper_fill)
  * status_raw provenance on remapped records
  * unknown status passthrough + STATUS_CANON_UNMAPPED journal marker
  * idempotency across the double-normalize (explicit + write() safety net)
  * containment ordering: a $100 placeholder FILLED stays rejected, never canon
  * trade_closer trust-set end-to-end (canonicalized fill feeds FIFO)
  * PA-EP1 fee predicate evaluated post-canon (genuine fill charged; non-fill
    and rejected charged nothing)
  * harvester shares the SAME map (import, not copy) — defensive write guard
"""

from __future__ import annotations

import json
import logging

import pytest

from chad.execution import paper_exec_evidence_writer as wmod
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    EvidencePayloadFactory,
    normalize_paper_fill_evidence,
    _STATUS_CANON,
)
from chad.execution import trade_closer

FEE_TAG = "ibkr_fixed_v1"


# ---------------------------------------------------------------------------
# shared price-cache fixture (mirrors test_normalize_paper_fill_evidence)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_price_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(json.dumps({
        "prices": {"SPY": 700.0, "MES": 7100.0, "AAPL": 270.0},
        "ts_utc": "2026-06-12T00:00:00Z",
        "ttl_seconds": 300,
    }))
    monkeypatch.setattr(wmod, "PRICE_CACHE_PATH", cache_path, raising=True)
    return cache_path


def _equity_fill(status: str, *, symbol: str = "SPY", price: float = 700.0,
                 qty: float = 100.0, strategy: str = "alpha") -> PaperExecEvidence:
    return PaperExecEvidence(
        symbol=symbol,
        side="BUY",
        quantity=qty,
        fill_price=price,
        status=status,
        is_live=False,
        asset_class="equity",
        strategy=strategy,
        expected_price=price,
    )


# ===========================================================================
# Layer 1 — per-literal mapping
# ===========================================================================
def test_uppercase_FILLED_maps_to_paper_fill(fake_price_cache):
    ev = _equity_fill("FILLED")
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"


def test_lowercase_filled_maps_to_paper_fill(fake_price_cache):
    ev = _equity_fill("filled")
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"


def test_pending_translated_fill_is_left_as_canon_target(fake_price_cache):
    """A pending status becomes paper_fill via the pre-existing translation;
    canon must leave it as the canon TARGET and must NOT stamp status_raw
    (status_raw belongs only to records the canon map itself remapped)."""
    ev = _equity_fill("PendingSubmit")
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"
    assert "status_raw" not in (ev.extra or {})


# ===========================================================================
# Layer 2 — status_raw provenance
# ===========================================================================
def test_status_raw_records_original_uppercase(fake_price_cache):
    ev = _equity_fill("FILLED")
    normalize_paper_fill_evidence(ev)
    assert ev.extra.get("status_raw") == "FILLED"


def test_status_raw_records_original_lowercase(fake_price_cache):
    ev = _equity_fill("filled")
    normalize_paper_fill_evidence(ev)
    assert ev.extra.get("status_raw") == "filled"


# ===========================================================================
# Layer 3 — unknown passthrough + loud marker
# ===========================================================================
def test_unknown_status_passes_through_with_marker(fake_price_cache, caplog):
    # Non-liquid symbol + sane non-placeholder price so no guard fires and the
    # record reaches the canon step with an unrecognized literal.
    ev = PaperExecEvidence(
        symbol="ZZTOPCO",
        side="BUY",
        quantity=10.0,
        fill_price=42.0,
        status="WEIRD_STATUS",
        is_live=False,
        asset_class="equity",
        strategy="alpha",
        expected_price=42.0,
    )
    with caplog.at_level(logging.WARNING, logger="chad.execution.paper_exec_evidence_writer"):
        normalize_paper_fill_evidence(ev)
    assert ev.status == "WEIRD_STATUS"          # unchanged
    assert "status_raw" not in (ev.extra or {})  # not remapped
    assert "STATUS_CANON_UNMAPPED" in caplog.text


def test_recognized_nonfill_does_not_emit_marker(fake_price_cache, caplog):
    """dry_run is a recognized literal — it must pass through silently."""
    ev = _equity_fill("dry_run")
    with caplog.at_level(logging.WARNING, logger="chad.execution.paper_exec_evidence_writer"):
        normalize_paper_fill_evidence(ev)
    assert ev.status == "dry_run"
    assert "STATUS_CANON_UNMAPPED" not in caplog.text


# ===========================================================================
# Layer 4 — idempotency (the write() path normalizes twice)
# ===========================================================================
def test_double_normalize_is_idempotent(fake_price_cache):
    ev = _equity_fill("FILLED")
    normalize_paper_fill_evidence(ev)
    normalize_paper_fill_evidence(ev)  # write()-safety-net second pass
    assert ev.status == "paper_fill"
    assert ev.extra.get("status_raw") == "FILLED"  # not clobbered to "paper_fill"


# ===========================================================================
# Layer 5 — containment ordering: canon never resurrects a placeholder
# ===========================================================================
def test_placeholder_filled_stays_rejected_not_canon(fake_price_cache):
    """A $100 placeholder arriving as FILLED on a liquid equity hits the
    price-deviation guard (cache SPY=700) FIRST and is demoted to rejected
    before the canon step — canon must not promote it to a trusted fill."""
    ev = _equity_fill("FILLED", price=100.0)
    ev.expected_price = 100.0
    normalize_paper_fill_evidence(ev)
    assert ev.status == "rejected"
    assert ev.reject is True
    assert "status_raw" not in (ev.extra or {})


# ===========================================================================
# Layer 6 — trade_closer trust-set end-to-end
# ===========================================================================
def test_canonicalized_fill_feeds_trade_closer_fifo(fake_price_cache):
    ev = _equity_fill("FILLED")
    normalize_paper_fill_evidence(ev)
    payload = EvidencePayloadFactory().build_fill_payload(ev)
    assert payload["status"] == "paper_fill"

    record = {"payload": payload, "record_hash": "TESTHASH"}
    extracted = trade_closer._extract_fill(record)
    # _extract_fill returns the flattened fill dict (no "status" key) only when
    # the record clears the trust gate — None means it was filtered out.
    assert extracted is not None, "canonicalized paper_fill must feed FIFO"
    assert extracted["symbol"] == "SPY"
    assert extracted["side"] == "BUY"
    assert extracted["quantity"] == pytest.approx(100.0)


def test_untrusted_status_is_filtered_by_trade_closer(fake_price_cache):
    """Negative control: the same record with an untrusted status is dropped by
    the trust gate, proving it is the (canonicalized) trusted status that lets
    the fill feed FIFO."""
    ev = _equity_fill("FILLED")
    normalize_paper_fill_evidence(ev)
    payload = EvidencePayloadFactory().build_fill_payload(ev)
    payload["status"] = "PendingSubmit"  # force an untrusted shape
    extracted = trade_closer._extract_fill({"payload": payload})
    assert extracted is None


def test_trust_set_is_fifo_neutral_across_canon():
    """Both the pre-canon literal (filled) and the canon target (paper_fill)
    are trusted, so canonicalization changes nothing for FIFO matching."""
    assert "filled" in trade_closer._TRUSTED_FILL_STATUSES
    assert "paper_fill" in trade_closer._TRUSTED_FILL_STATUSES


# ===========================================================================
# Layer 7 — PA-EP1 fee predicate evaluated post-canon
# ===========================================================================
def test_fee_charged_on_canonicalized_filled(fake_price_cache):
    """Genuine fill feeds the fee predicate: FILLED→paper_fill stays
    fee-eligible (_FEE_ELIGIBLE_FILL_STATUSES) so a modeled commission lands."""
    ev = _equity_fill("FILLED", qty=100.0, price=700.0)
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"
    assert ev.fee_amount > 0.0
    assert ev.extra.get("fee_model") == FEE_TAG


def test_no_fee_on_dry_run_post_canon(fake_price_cache):
    """A non-genuine-fill literal is not canonicalized and is not fee-eligible."""
    ev = _equity_fill("dry_run")
    normalize_paper_fill_evidence(ev)
    assert ev.status == "dry_run"
    assert ev.fee_amount == 0.0
    assert "fee_model" not in (ev.extra or {})


def test_no_fee_on_rejected(fake_price_cache):
    """Broker-rejected records return before canon and carry no fee."""
    ev = _equity_fill("rejected")
    normalize_paper_fill_evidence(ev)
    assert ev.status == "rejected"
    assert ev.fee_amount == 0.0
    assert "fee_model" not in (ev.extra or {})


# ===========================================================================
# Layer 8 — harvester shares the SAME map (import, not copy)
# ===========================================================================
def test_harvester_imports_same_status_canon_object():
    from chad.portfolio.ibkr_paper_fill_harvester import _SHARED_STATUS_CANON
    # Identity, not equality: proves "import, not copy".
    assert _SHARED_STATUS_CANON is _STATUS_CANON


def test_shared_map_is_noop_on_paper_fill_but_guards_filled():
    from chad.portfolio.ibkr_paper_fill_harvester import _SHARED_STATUS_CANON
    # No-op today — the harvester already emits the canon target.
    assert _SHARED_STATUS_CANON.get("paper_fill") is None
    # Defensive guard would fire if the harvester status ever regressed.
    assert _SHARED_STATUS_CANON.get("FILLED") == "paper_fill"
    assert _SHARED_STATUS_CANON.get("filled") == "paper_fill"


def test_shared_map_guard_logic_remaps_payload_dict():
    """Mirror the harvester's inline guard against a payload dict to prove the
    shared map + status_raw provenance behave on the harvester's data shape."""
    from chad.portfolio.ibkr_paper_fill_harvester import _SHARED_STATUS_CANON

    def _guard(payload):
        raw = payload.get("status")
        canon = _SHARED_STATUS_CANON.get(raw)
        if canon is not None and canon != raw:
            payload.setdefault("extra", {}).setdefault("status_raw", raw)
            payload["status"] = canon
        return payload

    # No-op on the real harvester payload.
    p1 = _guard({"status": "paper_fill", "extra": {}})
    assert p1["status"] == "paper_fill"
    assert "status_raw" not in p1["extra"]

    # Defensive remap if status ever changes.
    p2 = _guard({"status": "FILLED", "extra": {}})
    assert p2["status"] == "paper_fill"
    assert p2["extra"]["status_raw"] == "FILLED"

"""PA-EP7-T — containment tripwire for $100 placeholder records.

Context: EP-7 emission (delta/reconciler -> RoutedSignal.price=0 -> $100
placeholder) is confirmed LIVE but DEFERRED to backlog PA-EP7v2. The root
emission is not fixed here. This file instead LOCKS the containment invariant
that currently keeps those placeholders out of trusted PnL, so any future
regression that weakens it trips a test.

What normalize_paper_fill_evidence actually does (verified, not assumed):
  * It force-rejects a placeholder ONLY via the live NUMERIC fingerprint —
      5a (writer:~1725-1746): cached<=0 AND asset_class in equity-set AND
          symbol in _LIQUID_PRICED_EQUITIES AND fill_price≈100.0
      5b (writer:~1763-1789): cached>0 AND fill_price>0 AND >50% deviation
    Both then SET extra.trust_state="PLACEHOLDER" (:1741 / :1786) and
    status="rejected".
  * The extra.trust_state / placeholder_fill_price / tag "placeholder" markers
    are OUTPUTS of detection — they are NEVER read back as a rejection input.

Consequence (FINDING for PA-EP7v2, locked by the GAP tests below): a placeholder
that does not match the numeric+symbol detection (pre-stamped marker w/ clean
numerics, or a $100 fill on a symbol NOT in _LIQUID_PRICED_EQUITIES with no
price_cache reference) is NOT force-rejected and — post-PA-EP1 — also receives a
modeled commission (fee_model=ibkr_fixed_v1). See test docstrings for sites.

Repo-only behavior-lock; no production code changed.
"""

from __future__ import annotations

import json

import pytest

from chad.execution import paper_exec_evidence_writer as wmod
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
)

FEE_TAG = "ibkr_fixed_v1"


@pytest.fixture
def price_cache(tmp_path, monkeypatch):
    """Factory: redirect the writer's PRICE_CACHE_PATH to a tmp cache."""
    def _set(prices: dict):
        cache = tmp_path / "price_cache.json"
        cache.write_text(json.dumps({
            "prices": dict(prices),
            "ts_utc": "2026-06-12T00:00:00Z",
            "ttl_seconds": 300,
        }))
        monkeypatch.setattr(wmod, "PRICE_CACHE_PATH", cache, raising=True)
        return cache
    return _set


def _ev(**kw) -> PaperExecEvidence:
    base = dict(
        symbol="IWM", side="SELL", quantity=3.0,
        strategy="delta", source_strategies=["delta"],
        status="paper_fill", is_live=False, asset_class="equity",
    )
    base.update(kw)
    return PaperExecEvidence(**base)


# ===========================================================================
# 1a — CONTAINMENT: the live placeholder pattern is ALWAYS force-rejected
# ===========================================================================
def test_contained_5b_deviation_forces_reject(price_cache):
    """$100 fill on a liquid symbol with a real cache price (the production
    IWM leak) -> 5b deviation guard demotes to rejected."""
    price_cache({"IWM": 282.0})
    ev = _ev(fill_price=100.0, expected_price=100.0)
    normalize_paper_fill_evidence(ev)
    assert ev.status == "rejected"
    assert bool(ev.reject) is True
    assert (ev.extra or {}).get("trust_state") == "PLACEHOLDER"
    assert "placeholder" in (ev.tags or ())


def test_contained_5a_no_cache_forces_reject(price_cache):
    """$100 fill on a liquid symbol with NO cache reference -> 5a
    placeholder-without-cache guard demotes to rejected."""
    price_cache({})  # no IWM entry -> cached<=0
    ev = _ev(fill_price=100.0, expected_price=100.0)
    normalize_paper_fill_evidence(ev)
    assert ev.status == "rejected"
    assert bool(ev.reject) is True
    assert (ev.extra or {}).get("trust_state") == "PLACEHOLDER"


# ===========================================================================
# 1b — NO MODELED COMMISSION on a contained placeholder (PA-EP1 predicate)
# ===========================================================================
@pytest.mark.parametrize("cache", [{"IWM": 282.0}, {}], ids=["5b_deviation", "5a_no_cache"])
def test_contained_placeholder_gets_no_modeled_fee(price_cache, cache):
    price_cache(cache)
    ev = _ev(fill_price=100.0, expected_price=100.0)
    normalize_paper_fill_evidence(ev)
    assert ev.status == "rejected"          # precondition
    assert ev.fee_amount == 0.0             # never modeled
    assert "fee_model" not in (ev.extra or {})


# ===========================================================================
# 1c — LOCKED GAP (FINDING -> PA-EP7v2): placeholders the detector misses are
#      NOT force-rejected and now also receive a modeled commission.
#      These assert CURRENT behavior. When PA-EP7v2 closes the gap, flip them.
# ===========================================================================
def test_GAP_prestamped_marker_not_force_rejected(price_cache):
    """Record pre-stamped extra.trust_state=PLACEHOLDER but with clean numerics
    ($282 == cache) and status=paper_fill. normalize keys rejection on the
    numeric fingerprint (5a/5b), NOT on the marker, so this is NOT rejected and
    is modeled a fee. GAP — see writer detection sites 5a/5b; markers are set,
    never read, at writer:1741/1786."""
    price_cache({"IWM": 282.0})
    ev = _ev(fill_price=282.0, expected_price=282.0,
             extra={"trust_state": "PLACEHOLDER", "placeholder_fill_price": 100.0})
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"        # NOT force-rejected (current/gap)
    assert bool(ev.reject) is False
    assert ev.fee_amount == pytest.approx(1.0)            # fee modeled on a placeholder
    assert (ev.extra or {}).get("fee_model") == FEE_TAG


def test_GAP_nonliquid_dollar100_passes_as_fill(price_cache):
    """Raw $100 placeholder on a symbol NOT in _LIQUID_PRICED_EQUITIES
    (writer:1728) with no price_cache reference (5b needs cached>0, :1763) is
    caught by NEITHER guard -> passes as a $100 fill AND is modeled a fee. GAP."""
    price_cache({})  # no ZZZZ entry
    ev = _ev(symbol="ZZZZ", fill_price=100.0, expected_price=100.0)
    normalize_paper_fill_evidence(ev)
    assert ev.status == "paper_fill"        # NOT rejected (current/gap)
    assert ev.fill_price == pytest.approx(100.0)         # $100 retained as a "fill"
    assert bool(ev.reject) is False
    assert (ev.extra or {}).get("fee_model") == FEE_TAG  # fee modeled on a $100 placeholder


def test_GAP_nonliquid_filled_dollar100_passes(price_cache):
    """Same gap with status=FILLED and a pre-existing 'placeholder' tag — still
    not rejected, $100 retained, fee modeled. GAP."""
    price_cache({})
    ev = _ev(symbol="ZZZZ", fill_price=100.0, expected_price=100.0, status="FILLED",
             tags=["placeholder"],
             extra={"trust_state": "PLACEHOLDER", "placeholder_fill_price": 100.0})
    normalize_paper_fill_evidence(ev)
    assert ev.status == "FILLED"            # NOT rejected (current/gap)
    assert ev.fill_price == pytest.approx(100.0)
    assert bool(ev.reject) is False
    assert (ev.extra or {}).get("fee_model") == FEE_TAG

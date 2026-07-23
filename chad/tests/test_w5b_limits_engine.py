"""
W5B-2 — limits config + marginal evaluation tests.

Covers: the shipped config's declared bindings and stamps, the binding-vs-
report-only split, marginal breach arithmetic against the real book, the
never-trippable unmapped bucket, resize-vs-reject, and the degradation rules
(corrupt config binds nothing; a null cap can never bind).
"""

from __future__ import annotations

import json

import pytest

from chad.risk.allocator_limits import (
    LIMIT_GROSS,
    LIMIT_PER_SECTOR,
    LIMIT_PER_SYMBOL,
    SCHEMA_VERSION,
    WOULD_APPROVE,
    WOULD_REJECT,
    WOULD_RESIZE,
    PortfolioLimits,
    evaluate_marginal,
)
from chad.risk.portfolio_allocator import build_base_book, vector_from_intent
from chad.tests.test_w5b_exposure_core import REAL_BOOK, REAL_PRICES


@pytest.fixture()
def limits():
    return PortfolioLimits.load()


@pytest.fixture()
def sectors():
    from chad.risk.fuse_box import load_sector_map, make_sector_lookup

    return make_sector_lookup(load_sector_map())


@pytest.fixture()
def book(sectors):
    return build_base_book(positions=REAL_BOOK, prices=REAL_PRICES,
                           sector_lookup=sectors)


def _intent(symbol, qty, price, side="BUY", sec_type="STK"):
    return {"symbol": symbol, "side": side, "quantity": qty,
            "sec_type": sec_type, "limit_price": price}


def _vec(sectors, *args, **kwargs):
    return vector_from_intent(_intent(*args, **kwargs), prices=REAL_PRICES,
                              sector_lookup=sectors)


# --------------------------------------------------------------------------- #
# The shipped config
# --------------------------------------------------------------------------- #

def test_config_schema_and_bindings(limits):
    assert limits.raw["schema_version"] == SCHEMA_VERSION

    cap, binds, basis, ratified = limits.cap("gross_exposure")
    assert (cap, binds) == (750000.0, True)
    assert basis == "shadow_derivation_2026-07"
    assert ratified is False

    cap, binds, basis, ratified = limits.cap("per_sector")
    assert (cap, binds) == (375000.0, True)
    assert basis == "shadow_derivation_2026-07"
    assert ratified is False

    cap, binds, basis, ratified = limits.cap("per_symbol_concentration")
    assert (cap, binds, basis, ratified) == (150000.0, True, "sourced", True)

    # No net-exposure cap exists anywhere in the repo.
    cap, binds, _, _ = limits.cap("net_exposure")
    assert cap is None and binds is False


def test_derived_caps_are_stamped_unratified(limits):
    """OPERATOR_VERIFY 1/2: these generate would-reject evidence but are NOT
    ratified limits. If that stamp ever silently flips, this fails."""
    firm = limits.raw["firm"]
    for key in ("gross_exposure", "per_sector"):
        node = firm[key]
        assert node["basis"] == "shadow_derivation_2026-07"
        assert node["ratified"] is False
        assert node["enforce_era_requires_pa"] is True
        assert node["derivation"], f"{key} must carry its derivation"


def test_equity_is_never_used_as_a_divisor(limits):
    """§12.2: the book is USD-priced, the only currency_ok equity is CAD, and
    no FX rate exists. Dividing one by the other would be a unit error."""
    assert limits.equity_basis["currency"] == "CAD"
    assert limits.equity_basis["used_as_divisor"] is False


def test_venue_caps(limits):
    cap, binds, _, _ = limits.venue_cap("KRAKEN")
    assert (cap, binds) == (184.58, True)
    cap, binds, _, _ = limits.venue_cap("IBKR")
    assert cap is None and binds is False


def test_correlation_is_declarative_only(limits):
    """§13.3: W5B computes no correlation. The node may NAME the mode and the
    deferral, and may CITE the existing per-order reducer's threshold — but it
    must carry no number of its own, or a reader could mistake a citation for
    a measurement. Asserted structurally: every numeric leaf in the node must
    live under `existing_per_order_reducer`."""
    corr = limits.correlation_mode
    assert corr["mode"] == "static_sector_buckets"
    assert corr["rolling_deferred_to"] == "R2"

    def numeric_paths(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                yield from numeric_paths(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from numeric_paths(v, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            yield path

    found = list(numeric_paths(limits.raw["correlation"]))
    assert found == [".existing_per_order_reducer.threshold"], found


# --------------------------------------------------------------------------- #
# Marginal evaluation against the real book
# --------------------------------------------------------------------------- #

def test_small_add_is_approved(book, limits, sectors):
    v = _vec(sectors, "AAPL", 10, 321.01)
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict == WOULD_APPROVE
    assert verdict.which_limit is None


def test_lly_add_rejects_on_already_breached_per_symbol(book, limits, sectors):
    """The live finding: LLY is $215,488 against an ENFORCED $150k per-symbol
    cap, because that cap is per-ORDER. Every further LLY add would-rejects,
    and no resize fits because the pre-intent headroom is already negative."""
    v = _vec(sectors, "LLY", 10, 1184.0)
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict == WOULD_REJECT
    assert verdict.which_limit == LIMIT_PER_SYMBOL
    assert verdict.would_resize_to_qty is None
    assert verdict.breach_by_usd == pytest.approx(215_488.0 + 11_840.0 - 150_000.0)


def test_third_correlated_ticket_resizes_on_per_symbol(book, limits, sectors):
    """A 150-share UNH add breaches BOTH per_symbol ($164,853 vs $150k) and
    per_sector (healthcare $380,341 vs $375k). The worse breach names the
    limit, and the resize is the largest whole quantity fitting under both."""
    v = _vec(sectors, "UNH", 150, 422.70)
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict == WOULD_RESIZE
    assert verdict.which_limit == LIMIT_PER_SYMBOL
    assert verdict.would_resize_to_qty == 114.0

    # The resize must genuinely fit under every breached dimension.
    fitted = 114.0 * 422.70
    assert 101_448.0 + fitted <= 150_000.0
    assert 316_936.0 + fitted <= 375_000.0


def test_gross_ceiling_binds_on_a_fresh_symbol(book, limits, sectors):
    """A new mega_tech ticket big enough to breach the DERIVED gross ceiling
    while staying inside per_symbol and per_sector — isolates the gross leg."""
    v = _vec(sectors, "NVDA", 1000, 100.0)
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict == WOULD_RESIZE
    assert verdict.which_limit == LIMIT_GROSS
    assert verdict.breach_by_usd == pytest.approx(671_037.40 + 100_000.0 - 750_000.0, abs=0.5)
    # gross headroom 78,962.60 at $100/share ⇒ 789 whole shares.
    assert verdict.would_resize_to_qty == 789.0


def test_sector_ceiling_binds(book, limits, sectors):
    """Healthcare sits at $316,936 of its $375,000 derived cap — 84.5%, with
    $58,064 of headroom. The next sizeable healthcare ticket is what trips it."""
    approved = evaluate_marginal(_vec(sectors, "JNJ", 100, 400.0), book, limits)
    assert approved.verdict == WOULD_APPROVE

    v = _vec(sectors, "JNJ", 200, 400.0)  # $80,000 ⇒ healthcare $396,936
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict in (WOULD_RESIZE, WOULD_REJECT)
    assert verdict.which_limit == LIMIT_PER_SECTOR
    assert verdict.breach_by_usd == pytest.approx(396_936.0 - 375_000.0, abs=0.5)


def test_reject_when_no_whole_unit_fits(book, limits, sectors):
    """Gross headroom is $78,962.60. A single unit costing more than that
    cannot be resized into the book, so the verdict is REJECT, not RESIZE."""
    v = _vec(sectors, "NVDA", 1, 90_000.0)
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict == WOULD_REJECT
    assert verdict.would_resize_to_qty is None


def test_short_entry_adds_to_gross_but_reduces_net(book, limits, sectors):
    """A short adds to gross (risk) while reducing net (direction). Both legs
    must move the right way or the two dimensions are measuring the same thing."""
    v = _vec(sectors, "QQQ", 100, 500.0, side="SELL")
    verdict = evaluate_marginal(v, book, limits)
    gross = next(c for c in verdict.checks if c.limit == LIMIT_GROSS)
    net = next(c for c in verdict.checks if c.limit == "net")
    assert gross.projected_usd == pytest.approx(671_037.40 + 50_000.0, abs=0.5)
    assert net.projected_usd == pytest.approx(671_037.40 - 50_000.0, abs=0.5)


def test_report_only_dimensions_never_reject(book, limits, sectors):
    """Net has no cap, so however extreme the net becomes it can never produce
    a rejection — only a reported number."""
    v = _vec(sectors, "AAPL", 1, 321.01)
    verdict = evaluate_marginal(v, book, limits)
    net = next(c for c in verdict.checks if c.limit == "net")
    assert net.binds is False
    assert net.cap_usd is None
    assert verdict.which_limit != "net"


# --------------------------------------------------------------------------- #
# The unmapped bucket is never trippable (W4A LC3 idiom)
# --------------------------------------------------------------------------- #

def test_unmapped_sector_never_binds(limits, sectors):
    """Three unmapped tickets at $140k each sum to $420k — over the $375k
    sector cap — while each stays under the $150k per-symbol cap and the total
    stays under the $750k gross cap. It must still approve: a symbol missing
    from the sector map must not create a blockable bucket nobody ratified."""
    from chad.risk.portfolio_allocator import ProvisionalBook

    book = ProvisionalBook()
    for sym in ("ZZ1", "ZZ2"):
        v = _vec(sectors, sym, 1400, 100.0)
        assert v.sector == "unmapped"
        book.add_intent(v)

    v = _vec(sectors, "ZZ3", 1400, 100.0)
    verdict = evaluate_marginal(v, book, limits)
    sector_check = next(c for c in verdict.checks if c.limit == LIMIT_PER_SECTOR)
    assert sector_check.projected_usd == pytest.approx(420_000.0)
    assert sector_check.breached is True      # the arithmetic is still reported
    assert sector_check.binds is False        # ... but it cannot reject
    assert verdict.verdict == WOULD_APPROVE


# --------------------------------------------------------------------------- #
# Non-evaluable input, and degradation
# --------------------------------------------------------------------------- #

def test_uncomputable_intent_is_not_a_verdict(book, limits, sectors):
    """An unmapped futures root has no delta. The allocator must not
    manufacture a verdict from an unknown nor treat it as zero exposure."""
    v = vector_from_intent(
        _intent("ES", 1, 6000.0, sec_type="FUT"),
        prices={}, sector_lookup=sectors,
    )
    verdict = evaluate_marginal(v, book, limits)
    assert verdict.verdict == WOULD_APPROVE
    assert verdict.reason == "not_evaluable:unmapped_futures_root"
    assert verdict.checks == ()


def test_corrupt_config_binds_nothing(tmp_path, book, sectors):
    bad = tmp_path / "limits.json"
    bad.write_text("{not json", encoding="utf-8")
    limits = PortfolioLimits.load(bad)
    for key in ("gross_exposure", "per_sector", "per_symbol_concentration"):
        cap, binds, _, _ = limits.cap(key)
        assert cap is None and binds is False

    verdict = evaluate_marginal(_vec(sectors, "LLY", 1000, 1184.0), book, limits)
    assert verdict.verdict == WOULD_APPROVE


def test_missing_config_binds_nothing(tmp_path):
    limits = PortfolioLimits.load(tmp_path / "nope.json")
    assert limits.cap("gross_exposure") == (None, False, "unknown", False)


def test_null_cap_can_never_bind(tmp_path, book, sectors):
    """A config claiming binds:true against a null cap would mean 'reject
    everything against no number'. The loader must refuse that pairing."""
    p = tmp_path / "limits.json"
    p.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "firm": {"gross_exposure": {"cap_notional_usd": None, "binds": True}},
    }), encoding="utf-8")
    limits = PortfolioLimits.load(p)
    cap, binds, _, _ = limits.cap("gross_exposure")
    assert cap is None and binds is False


def test_reject_streak_default(limits):
    assert limits.reject_streak_n == 3

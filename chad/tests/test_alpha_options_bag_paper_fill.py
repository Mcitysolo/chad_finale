"""V9.0 outstanding item — alpha_options BAG/COMBO real paper-fill simulation.

Before this fix the paper executor stamped SPY's underlying price (~$720)
onto every alpha_options vertical-spread fill, with asset_class=etf and
notional=quantity*underlying_price. That made the position appear to have
a $720 cost basis when the actual debit paid was the per-contract net
debit (e.g. $1.50). Trade close PnL was unusable for SCR / profit_lock /
trade_closer accounting.

simulate_bag_paper_fill rewrites the record from the strategy-supplied
``net_debit_estimate``:

  * fill_price        ← net_debit_estimate (NOT the underlying)
  * notional          ← quantity * net_debit_estimate * 100 (multiplier)
  * asset_class       ← "options" (no ETF downgrade)
  * extra.bag_legs    ← both legs with strike/right/action/ratio
  * extra.sec_type    ← "BAG"
  * extra.net_debit   ← net_debit_estimate
  * status            ← "paper_fill"
  * extra.pnl_untrusted ← False (trustworthy — strategy-reference price)
  * tags              ← include "bag", "spread", "paper_fill"

These tests fail loudly if any of those invariants regress.
"""
from __future__ import annotations

from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
    simulate_bag_paper_fill,
)


def _bag_evidence(**overrides) -> PaperExecEvidence:
    """Build a representative alpha_options BAG paper evidence record."""
    base_extra = {
        "sec_type": "BAG",
        "spread_id": "test-spread-uuid",
        "spread_type": "BULL_CALL",
        "expiry": "20260516",
        "long_strike": 720.0,
        "short_strike": 725.0,
        "long_right": "C",
        "short_right": "C",
        "dte": 10,
        "max_loss_per_contract": 150.0,
        "net_debit_estimate": 1.50,
        "contracts": 1,
        "required_asset_class": "options",
        "engine": "alpha_options.v1",
    }
    base_extra.update(overrides.pop("extra_extra", {}))
    kwargs = dict(
        symbol="SPY",
        side="BUY",
        quantity=1.0,
        fill_price=0.0,
        expected_price=1.50,
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        status="",
        asset_class="",
        is_live=False,
        fill_time_utc="2026-05-06T16:32:00Z",
        extra=base_extra,
    )
    kwargs.update(overrides)
    return PaperExecEvidence(**kwargs)


def test_bag_simulator_records_net_debit_as_fill_price():
    ev = _bag_evidence()
    fired = simulate_bag_paper_fill(ev)
    assert fired is True, "simulate_bag_paper_fill must fire when sec_type=BAG and meta is present"
    assert ev.fill_price == 1.50, (
        f"fill_price must be the per-contract net debit (1.50), not the SPY "
        f"underlying — got {ev.fill_price}"
    )


def test_bag_simulator_sets_asset_class_options_no_etf_downgrade():
    ev = _bag_evidence()
    simulate_bag_paper_fill(ev)
    assert ev.asset_class == "options", (
        f"asset_class must be 'options' for a BAG fill — got {ev.asset_class!r}"
    )


def test_bag_simulator_records_both_legs_in_extra():
    ev = _bag_evidence()
    simulate_bag_paper_fill(ev)
    legs = ev.extra.get("bag_legs")
    assert isinstance(legs, list) and len(legs) == 2, f"must have exactly 2 legs, got {legs!r}"
    long_leg, short_leg = legs[0], legs[1]
    assert long_leg["action"] == "BUY"
    assert long_leg["strike"] == 720.0
    assert long_leg["right"] == "C"
    assert long_leg["expiry"] == "20260516"
    assert short_leg["action"] == "SELL"
    assert short_leg["strike"] == 725.0
    assert short_leg["right"] == "C"
    assert short_leg["expiry"] == "20260516"


def test_bag_simulator_marks_record_trustworthy():
    ev = _bag_evidence()
    simulate_bag_paper_fill(ev)
    # The strategy's net_debit_estimate is the trusted reference price.
    assert ev.extra.get("pnl_untrusted") is False, (
        "BAG paper fills must NOT be flagged pnl_untrusted — net_debit_estimate "
        "is a trusted strategy-supplied reference price."
    )
    assert ev.reject is False, "BAG paper fills must NOT be rejected"
    assert ev.status == "paper_fill", f"status must be paper_fill, got {ev.status!r}"


def test_bag_simulator_computes_notional_with_options_multiplier():
    ev = _bag_evidence(quantity=2.0, extra_extra={"contracts": 2})
    simulate_bag_paper_fill(ev)
    # 2 contracts * $1.50 debit * 100 multiplier = $300
    assert ev.notional == 300.0, (
        f"notional must be quantity * net_debit * 100 (options multiplier) — "
        f"got {ev.notional}"
    )


def test_bag_simulator_tags_include_bag_and_spread():
    ev = _bag_evidence()
    simulate_bag_paper_fill(ev)
    assert "bag" in ev.tags
    assert "spread" in ev.tags
    assert "paper_fill" in ev.tags


def test_normalize_runs_bag_simulator_first_so_full_pipeline_succeeds():
    """End-to-end: an alpha_options BAG record passing through the full
    normalizer (which is what paper_trade_executor and live_loop call)
    must come out with all BAG invariants applied AND must NOT be
    rejected by the strategy↔asset_class consistency check.
    """
    ev = _bag_evidence()
    normalize_paper_fill_evidence(ev)
    assert ev.asset_class == "options"
    assert ev.fill_price == 1.50
    assert ev.notional == 150.0
    assert ev.reject is False, (
        "Full normalizer must not reject a properly-typed BAG record. "
        "Regression here means the strategy↔asset_class guard fired before "
        "simulate_bag_paper_fill could set asset_class='options'."
    )
    assert ev.status == "paper_fill"
    legs = ev.extra.get("bag_legs")
    assert isinstance(legs, list) and len(legs) == 2


def test_normalize_does_not_overwrite_bag_fill_price_with_underlying():
    """Regression guard: the writer's price-cache lookup (step 2 in
    normalize_paper_fill_evidence) MUST NOT overwrite the BAG-simulated
    net debit ($1.50) with SPY's underlying price (~$720). Without the
    short-circuit this test fails with fill_price≈720.
    """
    ev = _bag_evidence()
    normalize_paper_fill_evidence(ev)
    # SPY's underlying is in the hundreds — the BAG debit is single-digit.
    # If fill_price exceeds $50, the cache-lookup branch fired and
    # corrupted the spread's cost basis.
    assert ev.fill_price < 50.0, (
        f"fill_price={ev.fill_price} — the underlying-price cache lookup "
        f"overwrote the BAG net debit. The bag_active short-circuit in "
        f"normalize_paper_fill_evidence is missing or broken."
    )


def test_simulator_skips_when_meta_missing():
    """Records without BAG meta must pass through untouched so non-options
    strategies (alpha equity, beta, futures) keep their existing flow."""
    ev = PaperExecEvidence(
        symbol="ES", side="BUY", quantity=1.0, fill_price=5500.0,
        expected_price=5500.0, strategy="alpha_futures",
        source_strategies=["alpha_futures"], broker="ibkr_paper",
        status="paper_fill", asset_class="futures", is_live=False,
        fill_time_utc="2026-05-06T16:00:00Z",
        extra={"plan_path": "/tmp/plan.json"},
    )
    fired = simulate_bag_paper_fill(ev)
    assert fired is False
    assert ev.fill_price == 5500.0
    assert ev.asset_class == "futures"


def test_simulator_skips_when_meta_incomplete():
    """When sec_type=BAG but required leg fields are missing, the simulator
    must not synthesize a fake fill — it skips and leaves the rest of
    normalize to handle the record."""
    ev = _bag_evidence(extra_extra={})
    # Drop a required field
    ev.extra.pop("net_debit_estimate", None)
    fired = simulate_bag_paper_fill(ev)
    assert fired is False
    assert "bag_simulator_skipped_reason" in ev.extra

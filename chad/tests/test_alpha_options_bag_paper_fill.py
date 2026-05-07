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

import json

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


# ---------------------------------------------------------------------------
# GAP-A001 — BAG SELL close path
# ---------------------------------------------------------------------------

def _bag_sell_close_evidence(**overrides) -> PaperExecEvidence:
    """Build a representative alpha_options BAG SELL close-side evidence.

    Mirrors the meta the alpha_options strategy now threads from the
    opening BAG fill into the max_hold_exit SELL signal — long_strike,
    short_strike, rights, expiry, net_debit_estimate, sec_type=BAG.
    """
    base_extra = {
        "sec_type": "BAG",
        "expiry": "20260516",
        "long_strike": 720.0,
        "short_strike": 725.0,
        "long_right": "C",
        "short_right": "C",
        "net_debit_estimate": 1.50,
        "net_debit": 1.50,
        "contracts": 1,
        "required_asset_class": "options",
        "engine": "alpha_options.v1",
        "reason": "max_hold_exit",
        "exit": True,
    }
    base_extra.update(overrides.pop("extra_extra", {}))
    kwargs = dict(
        symbol="SPY",
        side="SELL",
        quantity=1.0,
        fill_price=0.0,
        expected_price=0.0,
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        status="",
        asset_class="",
        is_live=False,
        fill_time_utc="2026-05-07T16:00:00Z",
        extra=base_extra,
    )
    kwargs.update(overrides)
    return PaperExecEvidence(**kwargs)


def test_bag_paper_fill_sell_closes_position(monkeypatch, tmp_path):
    """End-to-end: a SELL max_hold_exit intent must produce a trusted BAG
    SELL fill (reversed legs, pnl_breakdown, asset_class=options) AND that
    fill must satisfy is_fill_confirmed so mark_position_closed mutates
    the guard from open=True → open=False.
    """
    from chad.core import position_guard

    # Isolate the position guard to a tmp_path so the test never touches
    # the production runtime/position_guard.json.
    state_path = tmp_path / "position_guard.json"
    state_path.write_text(json.dumps({
        "alpha_options|SPY": {
            "open": True,
            "strategy": "alpha_options",
            "symbol": "SPY",
            "side": "BUY",
            "quantity": 1.0,
            "last_state": "OPEN",
        },
    }), encoding="utf-8")
    monkeypatch.setattr(position_guard, "STATE_PATH", state_path)

    ev = _bag_sell_close_evidence()
    fired = simulate_bag_paper_fill(ev)

    assert fired is True, "SELL close path must fire when BAG meta + opening debit are present"
    # 1. Status / asset_class / sec_type / fill_price wired correctly.
    assert ev.status == "paper_fill", f"status must be paper_fill, got {ev.status!r}"
    assert ev.asset_class == "options"
    assert ev.extra["sec_type"] == "BAG"
    # close_credit = original_debit * 0.30 → 1.50 * 0.30 = 0.45
    assert abs(ev.fill_price - 0.45) < 1e-9, (
        f"close fill_price must be conservative close_credit "
        f"(0.30 * net_debit) — got {ev.fill_price}"
    )
    # 2. Bag legs reversed: long BUY → SELL, short SELL → BUY.
    legs = ev.extra.get("bag_legs")
    assert isinstance(legs, list) and len(legs) == 2
    long_leg = next(l for l in legs if l["strike"] == 720.0)
    short_leg = next(l for l in legs if l["strike"] == 725.0)
    assert long_leg["action"] == "SELL", (
        "original long-strike leg must be SELL on the close"
    )
    assert short_leg["action"] == "BUY", (
        "original short-strike leg must be BUY on the close"
    )
    # 3. pnl_untrusted is False because original_debit context exists.
    assert ev.extra.get("pnl_untrusted") is False, (
        "BAG SELL close with opening debit context must NOT be flagged untrusted"
    )
    # 4. pnl_breakdown structured for trade_closer / SCR.
    pnl = ev.extra.get("pnl_breakdown")
    assert isinstance(pnl, dict)
    assert pnl["original_debit"] == 1.50
    assert abs(pnl["close_credit"] - 0.45) < 1e-9
    assert pnl["quantity"] == 1.0
    assert pnl["multiplier"] == 100
    # gross_pnl = (0.45 - 1.50) * 1 * 100 = -105.0
    assert abs(pnl["gross_pnl"] - (-105.0)) < 1e-9
    # 5. Tags include bag/spread/paper_fill (existing) + bag_close (new).
    assert "bag" in ev.tags
    assert "spread" in ev.tags
    assert "paper_fill" in ev.tags
    assert "bag_close" in ev.tags

    # 6. Position-guard close confirmation: build the fill_evidence dict
    # the way live_loop will after write_paper_exec_evidence returns, and
    # verify mark_position_closed flips the guard entry to closed.
    fill_evidence = {
        "fill_id": "bag-close-test-fill-id-1",
        "status": ev.status,
        "pnl_untrusted": bool(ev.extra.get("pnl_untrusted")),
        "reject": bool(ev.reject),
        "tags": list(ev.tags),
        "extra": dict(ev.extra),
    }
    assert position_guard.is_fill_confirmed(fill_evidence) is True, (
        "BAG SELL close fill must satisfy is_fill_confirmed so the guard "
        "can be safely closed"
    )

    class _Intent:
        strategy = "alpha_options"
        symbol = "SPY"
        side = "SELL"
        quantity = 1.0

    closed = position_guard.mark_position_closed(_Intent(), fill_evidence=fill_evidence)
    assert closed is True, "mark_position_closed must mutate guard on confirmed fill"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    entry = state["alpha_options|SPY"]
    assert entry["open"] is False
    assert entry["last_state"] == "CLOSED"
    assert entry["closed_by"] == "fill_confirmed"
    assert entry["closed_fill_id"] == "bag-close-test-fill-id-1"


def test_bag_paper_fill_sell_close_without_debit_context_marks_untrusted(monkeypatch, tmp_path):
    """When no opening debit context exists (no extra meta AND no opening
    fill on disk), the SELL close must still write a fill — but flagged
    pnl_untrusted=True with close_credit=0, so mark_position_closed
    refuses to close the guard (per ISSUE-29: no phantom closes).
    """
    from chad.core import position_guard
    from chad.execution import paper_exec_evidence_writer as writer_mod

    # Empty fills dir → opening lookup returns None.
    monkeypatch.setattr(writer_mod, "FILLS_DIR", tmp_path / "empty_fills")

    ev = PaperExecEvidence(
        symbol="SPY",
        side="SELL",
        quantity=1.0,
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        status="",
        asset_class="",
        is_live=False,
        fill_time_utc="2026-05-07T16:00:00Z",
        extra={
            "engine": "alpha_options.v1",
            "reason": "max_hold_exit",
            "exit": True,
            "required_asset_class": "options",
        },
    )
    fired = simulate_bag_paper_fill(ev)
    assert fired is True, "SELL close must still fire for an options-only strategy"
    assert ev.status == "paper_fill"
    assert ev.fill_price == 0.0, (
        "close_credit must be 0.0 when no original_debit can be recovered"
    )
    pnl = ev.extra.get("pnl_breakdown")
    assert isinstance(pnl, dict)
    assert pnl["original_debit"] == 0.0
    assert pnl["close_credit"] == 0.0
    assert pnl["gross_pnl"] == 0.0
    assert ev.extra.get("pnl_untrusted") is True

    # is_fill_confirmed must reject the untrusted close so the guard
    # remains open until a trusted fill is produced.
    fill_evidence = {
        "fill_id": "untrusted-close",
        "status": ev.status,
        "pnl_untrusted": True,
        "reject": False,
        "tags": list(ev.tags),
        "extra": dict(ev.extra),
    }
    assert position_guard.is_fill_confirmed(fill_evidence) is False


def test_bag_paper_fill_sell_close_recovers_legs_from_disk(monkeypatch, tmp_path):
    """When the SELL signal carries no leg meta, the writer must scan
    data/fills/FILLS_*.ndjson and recover the legs / debit from the most
    recent matching opening BAG fill.
    """
    from chad.execution import paper_exec_evidence_writer as writer_mod

    fills_dir = tmp_path / "fills"
    fills_dir.mkdir()
    monkeypatch.setattr(writer_mod, "FILLS_DIR", fills_dir)

    opening_payload = {
        "schema_version": "paper_exec_fill.v4",
        "strategy": "alpha_options",
        "symbol": "SPY",
        "side": "BUY",
        "status": "paper_fill",
        "fill_price": 1.50,
        "quantity": 1.0,
        "asset_class": "options",
        "reject": False,
        "extra": {
            "sec_type": "BAG",
            "net_debit": 1.50,
            "net_debit_estimate": 1.50,
            "expiry": "20260516",
            "contracts": 1,
            "options_multiplier": 100,
            "bag_legs": [
                {"action": "BUY", "strike": 720.0, "right": "C",
                 "ratio": 1, "expiry": "20260516"},
                {"action": "SELL", "strike": 725.0, "right": "C",
                 "ratio": 1, "expiry": "20260516"},
            ],
        },
    }
    record = {
        "payload": opening_payload,
        "prev_hash": "GENESIS",
        "sequence_id": 1,
        "timestamp_utc": "2026-05-06T18:35:42Z",
        "record_hash": "test-hash-1",
    }
    (fills_dir / "FILLS_20260506.ndjson").write_text(
        json.dumps(record) + "\n", encoding="utf-8",
    )

    ev = PaperExecEvidence(
        symbol="SPY",
        side="SELL",
        quantity=1.0,
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        status="",
        asset_class="",
        is_live=False,
        fill_time_utc="2026-05-07T16:00:00Z",
        extra={
            "engine": "alpha_options.v1",
            "reason": "max_hold_exit",
            "exit": True,
            "required_asset_class": "options",
        },
    )
    fired = simulate_bag_paper_fill(ev)
    assert fired is True
    assert ev.extra.get("pnl_untrusted") is False, (
        "writer must recover original_debit from the disk-backed opening "
        "fill so the close is trusted"
    )
    assert abs(ev.fill_price - 0.45) < 1e-9
    legs = ev.extra.get("bag_legs")
    assert isinstance(legs, list) and len(legs) == 2
    # Reversed legs: 720 strike now SELL (was BUY), 725 strike now BUY (was SELL).
    long_leg = next(l for l in legs if l["strike"] == 720.0)
    short_leg = next(l for l in legs if l["strike"] == 725.0)
    assert long_leg["action"] == "SELL"
    assert short_leg["action"] == "BUY"

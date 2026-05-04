"""Regression tests for the alpha_options BAG combo paper-fill downgrade.

Background
----------
alpha_options emits BAG (combo) TradeSignals with asset_class=OPTIONS for SPY
vertical spreads. Empirically (see data/fills/FILLS_20260502/04.ndjson), recent
fills for strategy=alpha_options were being persisted with asset_class="etf"
and sec_type=null — a silent BAG→equity downgrade that hid as a SPY ETF buy.

These tests pin three structural invariants so the downgrade cannot reappear:

  1. resolve_asset_class("SPY","BAG") and ("SPY","COMBO") map to "options",
     not "etf". BAG/COMBO are IBKR's combo containers — used exclusively for
     options multi-leg orders in this codebase.
  2. The paper-fill normalizer rejects (status=rejected, reject=True,
     pnl_untrusted) any record whose strategy is options-only but whose
     asset_class is non-options, instead of letting it land as a paper_fill
     ETF buy.
  3. The chad_paper_trade_executor refuses to write evidence for plan orders
     where strategy=alpha_options but asset_class!=options.
  4. Equity SPY fills (strategy=alpha, asset_class=etf) continue to write
     normally — the guard targets options-only strategies, not the ETF path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

from chad.execution.ibkr_adapter import resolve_asset_class
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    _OPTIONS_ONLY_STRATEGIES,
    normalize_paper_fill_evidence,
)
from chad.strategies.alpha_options import (
    AlphaOptionsTuning,
    _build_spread_signals,
)
from chad.options.strike_selector import SpreadSpec
from chad.types import AssetClass, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Test 1: BAG metadata preserved through plan / asset-class resolution
# ---------------------------------------------------------------------------

def test_alpha_options_bag_metadata_preserved_through_plan():
    """alpha_options must emit asset_class=OPTIONS, sec_type=BAG, spread legs.

    Also pins resolve_asset_class so a SPY/BAG (or SPY/COMBO) intent never
    falls through to the ETF lookup.
    """
    spread = SpreadSpec(
        symbol="SPY",
        expiry="20260619",
        long_strike=720.0,
        short_strike=727.0,
        right="C",
        spread_type="BULL_CALL",
        max_loss_per_contract=500.0,
        net_debit_estimate=5.0,
        dte=30,
    )
    tuning = AlphaOptionsTuning()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    signals = _build_spread_signals(
        spread=spread,
        confidence=0.8,
        equity=1_000_000.0,
        tuning=tuning,
        now=now,
        source_info={"source_strategy": "alpha", "source_direction": "bullish"},
    )

    assert len(signals) == 1
    sig = signals[0]
    # 1. asset_class on the TradeSignal itself must be OPTIONS
    assert sig.asset_class == AssetClass.OPTIONS
    # 2. meta carries the BAG combo marker and required_asset_class
    assert sig.meta["sec_type"] == "BAG"
    assert sig.meta["required_asset_class"] == "options"
    # 3. spread leg metadata must be preserved (long/short strikes + rights)
    assert sig.meta["long_strike"] == 720.0
    assert sig.meta["short_strike"] == 727.0
    assert sig.meta["long_right"] == "C"
    assert sig.meta["short_right"] == "C"
    assert sig.meta["spread_type"] == "BULL_CALL"
    # 4. The strategy enum must be ALPHA_OPTIONS
    assert sig.strategy == StrategyName.ALPHA_OPTIONS

    # 5. resolve_asset_class must map BAG/COMBO sec_type to "options" rather
    # than falling through to the SPY ETF lookup. This is the structural
    # check that prevents the silent downgrade.
    assert resolve_asset_class("SPY", "BAG") == "options"
    assert resolve_asset_class("SPY", "COMBO") == "options"
    # And the existing OPT path stays correct.
    assert resolve_asset_class("SPY", "OPT") == "options"
    # Unrelated equity path unchanged.
    assert resolve_asset_class("SPY", "STK") == "etf"


# ---------------------------------------------------------------------------
# Test 2: paper executor rejects unsupported BAG without writing as ETF
# ---------------------------------------------------------------------------

def _import_paper_trade_executor():
    """Load ops/bin/chad_paper_trade_executor.py as a module for import-time tests."""
    import importlib.util
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "ops" / "bin" / "chad_paper_trade_executor.py"
    spec = importlib.util.spec_from_file_location("_chad_paper_trade_executor_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paper_executor_rejects_unsupported_bag_without_downgrading_to_etf(tmp_path, monkeypatch, caplog):
    """An alpha_options plan order with non-options asset_class must be skipped.

    The executor must NOT write a paper_fill ETF/equity record in its place.
    """
    pte = _import_paper_trade_executor()

    # Synthetic plan with one alpha_options SPY order whose asset_class has
    # been silently downgraded to "etf" upstream.
    plan = {
        "orders": [
            {
                "primary_strategy": "ALPHA_OPTIONS",
                "symbol": "SPY",
                "side": "BUY",
                "size": 1.0,
                "price": 718.0,
                "notional": 718.0,
                "asset_class": "etf",  # ← the bad shape we want rejected
                "contributors": ["ALPHA_OPTIONS"],
            }
        ],
    }
    plan_path = tmp_path / "full_execution_cycle_last.json"
    plan_path.write_text(json.dumps(plan))
    price_cache = tmp_path / "price_cache.json"
    price_cache.write_text(json.dumps({
        "prices": {"SPY": 718.0},
        "ts_utc": "2099-01-01T00:00:00Z",
        "ttl_seconds": 300,
    }))

    monkeypatch.setattr(pte, "PLAN_PATH", plan_path)
    monkeypatch.setattr(pte, "PRICE_CACHE_PATH", price_cache)
    monkeypatch.setattr(pte, "LEDGER_PATH", tmp_path / "ledger.json")

    wrote_evidence: list = []
    wrote_results: list = []

    def _capture_evidence(ev):
        wrote_evidence.append(ev)
        return {}

    def _capture_result(tr):
        wrote_results.append(tr)
        return tmp_path / "noop_trade_result.ndjson"

    monkeypatch.setattr(pte, "write_paper_exec_evidence", _capture_evidence)
    monkeypatch.setattr(pte, "log_trade_result", _capture_result)
    # No-op normalizer wrapper — guard happens before normalization is reached.
    monkeypatch.setattr(pte, "normalize_paper_fill_evidence", lambda ev: ev)

    # Run the executor non-dry, large notional cap so the only thing that
    # could block the order is the new options-strategy guard.
    import logging
    caplog.set_level(logging.WARNING, logger="chad_paper_trade_executor")
    monkeypatch.setattr(sys, "argv", [
        "chad_paper_trade_executor",
        "--max-orders", "5",
        "--max-notional-usd", "1000000",
    ])
    pte.main()

    assert wrote_evidence == [], (
        "executor must NOT write evidence for an alpha_options plan order "
        f"with asset_class=etf; got {len(wrote_evidence)} evidence writes"
    )
    assert wrote_results == [], (
        "executor must NOT log a TradeResult for the rejected order"
    )
    # The skip path must have logged a clear reason mentioning the strategy
    # and asset_class so an operator can diagnose without a debugger.
    log_text = caplog.text.lower()
    assert "alpha_options" in log_text
    assert "unsupported_options_combo" in log_text


# ---------------------------------------------------------------------------
# Test 3: writer-level guard — alpha_options fill is never written as ETF
# ---------------------------------------------------------------------------

def test_alpha_options_fill_never_written_as_etf_for_bag():
    """normalize_paper_fill_evidence must reject options-strategy ETF fills.

    Even if every upstream layer fails to label the fill correctly, the
    chokepoint normalizer turns the record into status=rejected with a
    pnl_untrusted reason rather than letting it persist as a SPY ETF
    paper_fill.
    """
    ev = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=1.0,
        fill_price=718.0,
        status="paper_fill",
        is_live=False,
        asset_class="etf",          # ← the silent downgrade we are guarding against
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        expected_price=718.0,
    )
    normalize_paper_fill_evidence(ev)

    # 1. Status flipped to rejected — will not be treated as a real fill.
    assert ev.status == "rejected"
    assert ev.reject is True
    # 2. asset_class untouched (still "etf") so the audit record shows what
    #    the upstream layer tried to write — but the rejected status keeps it
    #    out of trade_closer FIFO matching and SCR effective trade counting.
    assert ev.asset_class == "etf"
    # 3. extra carries a clear reason an operator/diagnostic tool can grep.
    extra = ev.extra or {}
    assert extra.get("pnl_untrusted") is True
    assert "options_strategy_asset_class_mismatch" in str(extra.get("pnl_untrusted_reason", ""))
    assert extra.get("rejected_reason") == "unsupported_options_combo"
    # 4. Tags carry both pnl_untrusted (existing trusted-fill detectors) and
    #    unsupported_options_combo (new explicit marker).
    tags = list(ev.tags or ())
    assert "pnl_untrusted" in tags
    assert "unsupported_options_combo" in tags
    # 5. omega_momentum_options follows the same rule — both options-only
    #    strategies are guarded.
    assert "alpha_options" in _OPTIONS_ONLY_STRATEGIES
    assert "omega_momentum_options" in _OPTIONS_ONLY_STRATEGIES


def test_alpha_options_fill_with_correct_options_asset_class_passes_through(tmp_path, monkeypatch):
    """Sanity counterpart: a properly labeled options fill must NOT be rejected.

    Patches PRICE_CACHE_PATH so the existing 50%-deviation sanity guard
    (which compares fill_price vs the cache's SPY spot ~$700) does not
    incorrectly trip on the synthetic options premium of $5. The point of
    this test is the strategy/asset_class consistency check, not the
    placeholder-price guard.
    """
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(json.dumps({
        "prices": {"SPY": 5.0},  # match expected_price so deviation guard passes
        "ts_utc": "2099-01-01T00:00:00Z",
        "ttl_seconds": 300,
    }))
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.PRICE_CACHE_PATH",
        cache_path,
    )

    ev = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=1.0,
        fill_price=5.0,
        status="paper_fill",
        is_live=False,
        asset_class="options",      # ← correct
        strategy="alpha_options",
        source_strategies=["alpha_options"],
        broker="ibkr_paper",
        expected_price=5.0,
    )
    normalize_paper_fill_evidence(ev)
    assert ev.reject is False
    assert ev.status == "paper_fill"
    assert ev.asset_class == "options"


# ---------------------------------------------------------------------------
# Test 4: equity SPY path still works — guard does not over-reach
# ---------------------------------------------------------------------------

def test_equity_spy_fill_still_writes_as_etf_or_equity_normally(tmp_path, monkeypatch):
    """alpha (NOT alpha_options) SPY ETF fills must still normalize cleanly.

    The new guard must only fire on options-only strategies. A regular
    alpha-strategy SPY ETF buy is unchanged: status stays paper_fill,
    asset_class stays etf, and no rejected/unsupported tags are added.
    """
    # Stub the price cache so normalize_paper_fill_evidence can resolve a
    # cached price for SPY without touching the real runtime file.
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(json.dumps({
        "prices": {"SPY": 718.0},
        "ts_utc": "2099-01-01T00:00:00Z",
        "ttl_seconds": 300,
    }))
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.PRICE_CACHE_PATH",
        cache_path,
    )

    ev = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        fill_price=718.0,
        status="paper_fill",
        is_live=False,
        asset_class="etf",
        strategy="alpha",            # ← regular ETF strategy, NOT options
        source_strategies=["alpha"],
        broker="ibkr_paper",
        expected_price=718.0,
    )
    normalize_paper_fill_evidence(ev)

    # 1. Untouched by the new options guard.
    assert ev.reject is False
    assert ev.status == "paper_fill"
    assert ev.asset_class == "etf"
    # 2. No options-mismatch reason was attached.
    extra = ev.extra or {}
    assert "options_strategy_asset_class_mismatch" not in str(
        extra.get("pnl_untrusted_reason", "")
    )
    assert extra.get("rejected_reason") != "unsupported_options_combo"
    tags = list(ev.tags or ())
    assert "unsupported_options_combo" not in tags

    # 3. Pendingsubmit equity path still translates to paper_fill cleanly.
    ev2 = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=1.0,
        fill_price=0.0,
        status="PendingSubmit",
        is_live=False,
        asset_class="",              # blank → resolver fills "etf" for SPY
        strategy="alpha",
        broker="ibkr_paper",
    )
    normalize_paper_fill_evidence(ev2)
    assert ev2.status == "paper_fill"
    assert ev2.asset_class == "etf"
    assert ev2.reject is False

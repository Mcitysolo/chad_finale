"""PA-EP1 — modeled IBKR Fixed commissions at the evidence chokepoint.

Evidence Pipeline EP-1: FEES_*.ndjson fee_amount was always 0.0 because the
paper write path never computes a commission. This PA injects a modeled IBKR
Fixed-schedule commission inside normalize_paper_fill_evidence (the single
chokepoint every paper writer shares), forward-only, stamped
fee_model="ibkr_fixed_v1" so modeled fees are forever distinguishable from
broker-reported fees.

Operator-ratified schedule (2026-06-12):
  equities : $0.005/share, min $1.00/order, cap 1.0% of trade value
  futures  : $0.85/contract (micro table {MES,MNQ,M2K,M6E,MCL}, default $0.85)
  options  : $0.65/contract, min $1.00/order
  crypto   : 0.26% taker on notional
  unknown  : 0.0 + loud FEE_MODEL_UNKNOWN_ASSET_CLASS marker

Test layers:
  * model unit tests per asset class incl. boundaries (floor, cap, per-unit)
  * predicate tests: rejected/duplicate/dry_run -> fee 0, no fee_model stamp
  * provenance: modeled fee carries fee_model; nonzero (broker) fee untouched
  * single-application: repeated normalize() never double-charges
  * end-to-end through normalize_paper_fill_evidence (equity + options BAG)
"""

from __future__ import annotations

import json
import logging

import pytest

from chad.execution import paper_exec_evidence_writer as wmod
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
)

FEE_TAG = "ibkr_fixed_v1"


# ---------------------------------------------------------------------------
# shared price-cache fixture (mirrors test_normalize_paper_fill_evidence)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_price_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(json.dumps({
        "prices": {"SPY": 400.0, "MES": 7100.0},
        "ts_utc": "2026-06-12T00:00:00Z",
        "ttl_seconds": 300,
    }))
    monkeypatch.setattr(wmod, "PRICE_CACHE_PATH", cache_path, raising=True)
    return cache_path


# ===========================================================================
# Layer 1 — _model_ibkr_commission per asset class + boundaries
# ===========================================================================
def test_model_equity_min_floor():
    # 100 sh * $400: per-share 0.50 < $1 floor; cap 1%*40000=$400 -> $1.00 floor
    assert wmod._model_ibkr_commission("equity", 100, 400.0, "SPY") == pytest.approx(1.00)


def test_model_equity_per_share_between_floor_and_cap():
    # 400 sh * $400: per-share 2.00 (>floor); cap 1%*160000=$1600 -> $2.00
    assert wmod._model_ibkr_commission("equity", 400, 400.0, "AAPL") == pytest.approx(2.00)


def test_model_equity_cap_ceiling():
    # 10000 sh * $0.40 = $4000 value: per-share 50.0; cap 1%*4000=$40 -> $40 cap
    assert wmod._model_ibkr_commission("equity", 10000, 0.40, "PENNY") == pytest.approx(40.0)


def test_model_etf_uses_equity_schedule():
    assert wmod._model_ibkr_commission("etf", 400, 400.0, "QQQ") == pytest.approx(2.00)


def test_model_futures_micro_table():
    # 3 MES * $0.85
    assert wmod._model_ibkr_commission("futures", 3, 7100.0, "MES") == pytest.approx(2.55)


def test_model_futures_month_coded_symbol_resolves_root():
    # MESH6 -> MES table entry
    assert wmod._model_ibkr_commission("futures", 2, 7100.0, "MESH6") == pytest.approx(1.70)


def test_model_futures_default_fallback():
    # ZN not in micro table -> default $0.85/contract
    assert wmod._model_ibkr_commission("futures", 4, 111.0, "ZN") == pytest.approx(3.40)


def test_model_options_min_floor():
    # 1 contract * $0.65 = $0.65 -> $1.00 floor
    assert wmod._model_ibkr_commission("options", 1, 4.0, "SPY") == pytest.approx(1.00)


def test_model_options_per_contract_above_floor():
    # 4 contracts * $0.65 = $2.60 (>floor)
    assert wmod._model_ibkr_commission("options", 4, 4.0, "SPY") == pytest.approx(2.60)


def test_model_crypto_taker_bps():
    # 0.5 BTC * $60000 = $30000 notional * 0.26%
    assert wmod._model_ibkr_commission("crypto", 0.5, 60000.0, "BTCUSD") == pytest.approx(78.0)


def test_model_zero_qty_returns_zero():
    assert wmod._model_ibkr_commission("equity", 0, 400.0, "SPY") == 0.0


def test_model_unknown_asset_class_returns_zero_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        fee = wmod._model_ibkr_commission("forex", 100000, 1.1, "EURUSD")
    assert fee == 0.0
    assert "FEE_MODEL_UNKNOWN_ASSET_CLASS" in caplog.text


# ===========================================================================
# Layer 2 — predicate: non-genuine fills carry NO fee and NO fee_model stamp
# ===========================================================================
@pytest.mark.parametrize("status", ["dry_run", "duplicate_blocked", "duplicate_open_order", "PendingSubmit"])
def test_apply_skips_non_genuine_status(status):
    ev = PaperExecEvidence(
        symbol="SPY", side="BUY", quantity=400, fill_price=400.0,
        asset_class="equity", status=status, is_live=False,
    )
    wmod._apply_modeled_commission(ev)
    assert ev.fee_amount == 0.0
    assert "fee_model" not in (ev.extra or {})
    assert "_fee_modeled" not in (ev.extra or {})


def test_apply_skips_rejected_record():
    ev = PaperExecEvidence(
        symbol="SPY", side="BUY", quantity=400, fill_price=400.0,
        asset_class="equity", status="paper_fill", is_live=False, reject=True,
    )
    wmod._apply_modeled_commission(ev)
    assert ev.fee_amount == 0.0
    assert "fee_model" not in (ev.extra or {})


# ===========================================================================
# Layer 3 — provenance + never-overwrite-broker-fee
# ===========================================================================
def test_apply_stamps_provenance_on_modeled_equity():
    ev = PaperExecEvidence(
        symbol="AAPL", side="BUY", quantity=400, fill_price=400.0,
        asset_class="equity", status="paper_fill", is_live=False,
    )
    wmod._apply_modeled_commission(ev)
    assert ev.fee_amount == pytest.approx(2.00)
    assert ev.extra["fee_model"] == FEE_TAG


def test_apply_never_overwrites_broker_reported_fee():
    ev = PaperExecEvidence(
        symbol="AAPL", side="BUY", quantity=400, fill_price=400.0,
        asset_class="equity", status="paper_fill", is_live=False,
        fee_amount=2.50,  # broker-reported
    )
    wmod._apply_modeled_commission(ev)
    assert ev.fee_amount == pytest.approx(2.50)        # untouched
    assert "fee_model" not in (ev.extra or {})         # no model stamp


def test_apply_is_idempotent_double_call():
    ev = PaperExecEvidence(
        symbol="AAPL", side="BUY", quantity=400, fill_price=400.0,
        asset_class="equity", status="paper_fill", is_live=False,
    )
    wmod._apply_modeled_commission(ev)
    first = ev.fee_amount
    wmod._apply_modeled_commission(ev)
    assert ev.fee_amount == first == pytest.approx(2.00)  # not doubled


# ===========================================================================
# Layer 4 — end-to-end through the real normalize chokepoint
# ===========================================================================
def test_normalize_equity_applies_modeled_fee(fake_price_cache):
    ev = PaperExecEvidence(
        symbol="SPY", side="BUY", quantity=400, fill_price=400.0,
        expected_price=400.0, strategy="alpha", source_strategies=["alpha"],
        asset_class="stock", status="paper_fill", is_live=False,
    )
    normalize_paper_fill_evidence(ev)
    assert ev.fee_amount == pytest.approx(2.00)
    assert ev.extra["fee_model"] == FEE_TAG


def test_normalize_double_call_single_fee(fake_price_cache):
    """Reproduces the live_loop / reconciler pattern (explicit normalize +
    write() safety-net normalize). Fee must apply exactly once."""
    ev = PaperExecEvidence(
        symbol="SPY", side="BUY", quantity=400, fill_price=400.0,
        expected_price=400.0, strategy="alpha", source_strategies=["alpha"],
        asset_class="stock", status="paper_fill", is_live=False,
    )
    normalize_paper_fill_evidence(ev)
    normalize_paper_fill_evidence(ev)
    assert ev.fee_amount == pytest.approx(2.00)  # not 4.00


def test_normalize_options_bag_applies_per_leg_fee():
    """BAG vertical (3 spreads x 2 legs = 6 contracts) -> 6 * $0.65 = $3.90."""
    ev = PaperExecEvidence(
        symbol="SPY", side="BUY", quantity=3, fill_price=0.0,
        strategy="alpha_options", source_strategies=["alpha_options"],
        status="paper_fill", is_live=False,
        extra={
            "sec_type": "BAG",
            "required_asset_class": "options",
            "long_strike": 740.0, "short_strike": 746.0,
            "long_right": "C", "short_right": "C",
            "expiry": "20260612", "net_debit_estimate": 4.0,
        },
    )
    normalize_paper_fill_evidence(ev)
    assert ev.asset_class == "options"
    assert ev.fee_amount == pytest.approx(3.90)
    assert ev.extra["fee_model"] == FEE_TAG


def test_normalize_rejected_status_no_fee(fake_price_cache):
    ev = PaperExecEvidence(
        symbol="SPY", side="BUY", quantity=400, fill_price=400.0,
        expected_price=400.0, strategy="alpha", source_strategies=["alpha"],
        asset_class="stock", status="rejected", is_live=False,
    )
    normalize_paper_fill_evidence(ev)
    assert ev.fee_amount == 0.0
    assert "fee_model" not in (ev.extra or {})

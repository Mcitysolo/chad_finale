"""Tests for the v9.1 TierManager and the legacy `_select_tier` shim.

Covers the 10 acceptance cases for the v9.1 tier ladder
(MICRO < STARTER < PRO_GROWTH < SCALE):
    1. SCALE for $182k, risk_profile fields all None.
    2. STARTER caps at $2,600.
    3. MICRO caps at $500.
    4. STARTER held inside its band ($2,480).
    5. Mid-session demotion deferred ($2,100, market open).
    6. Demotion applied off-session ($2,100, market closed).
    7. Promotion applies immediately ($2,600 from MICRO).
    8. SCALE null risk_profile yields all-None caps without raising.
    9. PRO_GROWTH caps at $50,000.
   10. Pending demotion in the warning band ($159,999 with current=SCALE).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import chad.risk.tier_manager as tmmod
from chad.risk.tier_manager import (
    TIER_RANK,
    TierManager,
    TierRiskProfile,
)

REPO_ROOT = Path("/home/ubuntu/chad_finale")
TIERS_CONFIG_PATH = REPO_ROOT / "config" / "tiers.json"


@pytest.fixture(scope="module")
def tiers_config() -> dict:
    return json.loads(TIERS_CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test 1 — SCALE for $182k, risk_profile fields all None.
# ---------------------------------------------------------------------------

def test_1_scale_tier_at_182k(tiers_config):
    tm = TierManager(equity=182_000.0, tiers_config=tiers_config)
    assert tm.tier_name == "SCALE"
    profile = tm.get_risk_profile()
    assert profile.max_contracts_per_trade is None
    assert profile.max_risk_per_trade_usd is None
    assert profile.max_daily_loss_usd is None
    assert profile.max_weekly_loss_usd is None
    assert profile.max_trades_per_day is None
    assert profile.flatten_eod_minutes_before_close is None
    assert tm.demotion_pending is False


# ---------------------------------------------------------------------------
# Test 2 — STARTER caps at $2,600.
# ---------------------------------------------------------------------------

def test_2_starter_caps_at_2600(tiers_config):
    tm = TierManager(equity=2_600.0, tiers_config=tiers_config)
    assert tm.tier_name == "STARTER"
    profile = tm.get_risk_profile()
    assert profile.max_contracts_per_trade == 2
    assert profile.max_daily_loss_usd == 150


# ---------------------------------------------------------------------------
# Test 3 — MICRO caps at $500.
# ---------------------------------------------------------------------------

def test_3_micro_caps_at_500(tiers_config):
    tm = TierManager(equity=500.0, tiers_config=tiers_config)
    assert tm.tier_name == "MICRO"
    profile = tm.get_risk_profile()
    assert profile.max_contracts_per_trade == 1
    assert profile.max_daily_loss_usd == 20


# ---------------------------------------------------------------------------
# Test 4 — STARTER held inside its band ($2,480 > demotion gate $2,250).
# ---------------------------------------------------------------------------

def test_4_starter_held_inside_band(tiers_config):
    tm = TierManager(
        equity=2_480.0,
        current_tier="STARTER",
        market_open=True,
        tiers_config=tiers_config,
    )
    # "Stays STARTER" — tier resolution must not change.
    assert tm.tier_name == "STARTER"


# ---------------------------------------------------------------------------
# Test 5 — Mid-session demotion deferred ($2,100, market open).
# ---------------------------------------------------------------------------

def test_5_demotion_pending_market_open(tiers_config):
    tm = TierManager(
        equity=2_100.0,
        current_tier="STARTER",
        market_open=True,
        tiers_config=tiers_config,
    )
    assert tm.tier_name == "STARTER"
    assert tm.demotion_pending is True
    assert tm.demotion_pending_to == "MICRO"


# ---------------------------------------------------------------------------
# Test 6 — Demotion applied off-session ($2,100, market closed).
# ---------------------------------------------------------------------------

def test_6_demotion_applied_market_closed(tiers_config):
    tm = TierManager(
        equity=2_100.0,
        current_tier="STARTER",
        market_open=False,
        tiers_config=tiers_config,
    )
    assert tm.tier_name == "MICRO"
    assert tm.demotion_pending is False


# ---------------------------------------------------------------------------
# Test 7 — Promotion applies immediately ($2,600 from MICRO, market open).
# ---------------------------------------------------------------------------

def test_7_promotion_immediate(tiers_config):
    tm = TierManager(
        equity=2_600.0,
        current_tier="MICRO",
        market_open=True,
        tiers_config=tiers_config,
    )
    assert tm.tier_name == "STARTER"
    assert tm.demotion_pending is False


# ---------------------------------------------------------------------------
# Test 8 — SCALE null risk_profile fields yield all-None caps without raising.
# ---------------------------------------------------------------------------

def test_8_scale_null_risk_profile_no_raise(tiers_config):
    tm = TierManager(equity=10_000_000.0 - 1, tiers_config=tiers_config)
    assert tm.tier_name == "SCALE"
    profile = tm.get_risk_profile()
    assert isinstance(profile, TierRiskProfile)
    assert profile.max_contracts_per_trade is None
    assert profile.max_risk_per_trade_usd is None
    assert profile.max_daily_loss_usd is None
    assert profile.max_weekly_loss_usd is None
    assert profile.max_trades_per_day is None
    assert profile.flatten_eod_minutes_before_close is None
    # No exception was raised — the resolution and risk-profile read are safe
    # even when every numeric cap is null.


# ---------------------------------------------------------------------------
# Test 9 — PRO_GROWTH caps at $50,000.
# ---------------------------------------------------------------------------

def test_9_pro_growth_at_50k(tiers_config):
    tm = TierManager(equity=50_000.0, tiers_config=tiers_config)
    assert tm.tier_name == "PRO_GROWTH"
    profile = tm.get_risk_profile()
    assert profile.primary_session_only is False
    assert profile.max_risk_per_trade_usd == 200


# ---------------------------------------------------------------------------
# Test 10 — Pending demotion in the warning band ($159,999, current=SCALE).
# ---------------------------------------------------------------------------

def test_10_scale_warning_band(tiers_config):
    tm = TierManager(
        equity=159_999.0,
        current_tier="SCALE",
        market_open=True,
        tiers_config=tiers_config,
    )
    assert tm.tier_name == "SCALE"
    assert tm.demotion_pending is True
    assert tm.demotion_pending_to == "PRO_GROWTH"


# ---------------------------------------------------------------------------
# Bonus invariants — TIER_RANK ordering and legacy-name migration.
# ---------------------------------------------------------------------------

def test_tier_rank_ordering():
    assert TIER_RANK["MICRO"] < TIER_RANK["STARTER"] < TIER_RANK["PRO_GROWTH"] < TIER_RANK["SCALE"]


def test_legacy_pro_alias_migrates_to_scale(tiers_config):
    # renamed PRO -> SCALE in v9.1: a runtime tier_state.json carrying
    # the legacy "PRO" name should be treated as "SCALE" on read.
    tm = TierManager(
        equity=180_000.0,
        current_tier="PRO",
        market_open=True,
        tiers_config=tiers_config,
    )
    assert tm.tier_name == "SCALE"


# ---------------------------------------------------------------------------
# main() — tier band sourced from authoritative USD equity (fail-closed).
#
# The CLI reads runtime/portfolio_snapshot.json and must select the tier from
# `total_equity_usd_authoritative` / `usd_ok` ONLY — never the CAD component
# sum (ibkr_equity + kraken_equity + coinbase_equity).  When usd_ok is false
# (or the USD figure is absent / null) it HOLDS the last persisted tier and
# does not recompute.  Snapshots below deliberately set the CAD components to
# values whose CAD sum would resolve to a *different* tier, proving CAD is
# never consulted.
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, *, snapshot, prior_state=None):
    snap_path = tmp_path / "portfolio_snapshot.json"
    out_path = tmp_path / "tier_state.json"
    cfg_path = tmp_path / "tiers.json"
    snap_path.write_text(json.dumps(snapshot), encoding="utf-8")
    if prior_state is not None:
        out_path.write_text(json.dumps(prior_state), encoding="utf-8")
    # use the real production tier bands
    cfg_path.write_text(TIERS_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(tmmod, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(tmmod, "SNAPSHOT_PATH", snap_path)
    monkeypatch.setattr(tmmod, "OUT_PATH", out_path)
    monkeypatch.setattr(tmmod, "TIERS_CONFIG_PATH", cfg_path)
    rc = tmmod.main()
    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else None
    return rc, data


def test_main_usd_ok_137k_selects_pro_growth(monkeypatch, tmp_path):
    # USD 137k -> PRO_GROWTH band [25k, 160k).  The CAD sum (~198k) would be
    # SCALE; prove the CAD sum is NOT used.
    snapshot = {
        "ibkr_equity": 197_935.4,        # CAD — must be ignored
        "kraken_equity": 258.5,          # CAD — must be ignored
        "coinbase_equity": 0.0,
        "total_equity_usd_authoritative": 137_000.0,
        "usd_ok": True,
    }
    rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot)
    assert rc == 0
    assert data["tier_name"] == "PRO_GROWTH"
    assert data["current_equity_usd"] == 137_000.0   # USD figure, not CAD sum
    cad_sum = 197_935.4 + 258.5 + 0.0                 # ~198k -> would be SCALE
    assert data["current_equity_usd"] != cad_sum
    assert data["tier_name"] != "SCALE"


def test_main_usd_ok_above_160k_selects_scale(monkeypatch, tmp_path):
    # USD >160k -> SCALE.  The tiny CAD sum (100) would be MICRO; prove USD
    # drives the band, not CAD.
    snapshot = {
        "ibkr_equity": 100.0,            # CAD — would be MICRO if used
        "kraken_equity": 0.0,
        "coinbase_equity": 0.0,
        "total_equity_usd_authoritative": 165_000.0,
        "usd_ok": True,
    }
    rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot)
    assert rc == 0
    assert data["tier_name"] == "SCALE"
    assert data["current_equity_usd"] == 165_000.0
    assert data["tier_name"] != "MICRO"


def test_main_usd_not_ok_holds_prior_tier_no_cad_fallback(monkeypatch, tmp_path, caplog):
    # usd_ok false (FX unavailable) -> HOLD the last persisted tier; do NOT
    # recompute and NEVER fall back to the CAD sum.  The CAD sum here (~600k)
    # would resolve to SCALE — proving CAD is never consulted.
    snapshot = {
        "ibkr_equity": 500_000.0,        # CAD — must NEVER be used
        "kraken_equity": 100_000.0,      # CAD — must NEVER be used
        "coinbase_equity": 0.0,
        "total_equity_usd_authoritative": None,
        "usd_ok": False,
    }
    prior = {
        "schema_version": "tier_state.v2",
        "tier_name": "PRO_GROWTH",
        "current_equity_usd": 137_000.0,
    }
    with caplog.at_level(logging.WARNING, logger=tmmod.LOG.name):
        rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot, prior_state=prior)
    assert rc == 0
    # tier_state.json is left untouched (held) — no recompute occurred.
    assert data == prior
    assert data["tier_name"] == "PRO_GROWTH"
    assert data["current_equity_usd"] == 137_000.0   # untouched; NOT the CAD sum
    cad_sum = 500_000.0 + 100_000.0 + 0.0             # ~600k -> would be SCALE
    assert data["current_equity_usd"] != cad_sum
    assert data["tier_name"] != "SCALE"
    assert any("TIER_HELD_NO_USD_RATE" in r.getMessage() for r in caplog.records)


def test_main_usd_field_absent_treated_as_not_ok(monkeypatch, tmp_path, caplog):
    # An old snapshot lacking the authoritative-USD fields entirely must be
    # treated as fail-closed (hold), never CAD-summed.
    snapshot = {
        "ibkr_equity": 500_000.0,        # CAD
        "kraken_equity": 100_000.0,      # CAD
        "coinbase_equity": 0.0,
    }
    prior = {"schema_version": "tier_state.v2", "tier_name": "STARTER",
             "current_equity_usd": 10_000.0}
    with caplog.at_level(logging.WARNING, logger=tmmod.LOG.name):
        rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot, prior_state=prior)
    assert rc == 0
    assert data == prior                              # held, no recompute
    assert any("TIER_HELD_NO_USD_RATE" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Demotion-deferral state machine is preserved unchanged by the USD-source fix:
# a SCALE -> PRO_GROWTH demotion is queued (deferred) while the market is open
# and only applies once the market is closed.
# ---------------------------------------------------------------------------

def test_demotion_deferral_preserved_scale_to_pro_growth(tiers_config):
    # equity 140k is below SCALE's demotion gate (144k) but, with the market
    # open, the SCALE -> PRO_GROWTH demotion must DEFER, not apply instantly.
    tm_open = TierManager(
        equity=140_000.0,
        current_tier="SCALE",
        market_open=True,
        tiers_config=tiers_config,
    )
    assert tm_open.tier_name == "SCALE"              # held, not instant
    assert tm_open.demotion_pending is True
    assert tm_open.demotion_pending_to == "PRO_GROWTH"
    assert tm_open.demotion_pending_reason == "mid_session_demotion_deferred"

    # Off-session, the same equity applies the demotion.
    tm_closed = TierManager(
        equity=140_000.0,
        current_tier="SCALE",
        market_open=False,
        tiers_config=tiers_config,
    )
    assert tm_closed.tier_name == "PRO_GROWTH"
    assert tm_closed.demotion_pending is False

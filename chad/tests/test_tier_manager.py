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
from pathlib import Path

import pytest

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

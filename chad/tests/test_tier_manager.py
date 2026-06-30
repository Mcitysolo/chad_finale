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
from datetime import datetime, timezone
from pathlib import Path

import pytest

import chad.risk.tier_manager as tmmod
from chad.risk.tier_manager import (
    TIER_RANK,
    TierManager,
    TierRiskProfile,
)
from chad.utils.market_hours import market_is_open

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
# main() — tier band sourced from broker-native CAD equity (C-ii, fail-closed).
#
# The CLI reads runtime/portfolio_snapshot.json and selects the tier from
# `ibkr_equity` / `ibkr_equity_currency_ok` (broker-native CAD, no FX call).
# When ibkr_equity_currency_ok is false (or ibkr_equity is absent / null) it
# HOLDS the last persisted tier and does not recompute.  The config below is the
# CAD-repriced tiers config (PA C-ii); the band readers are dual-key, so these
# tests validate the end-state (CAD intake vs CAD bands).
# ---------------------------------------------------------------------------

_USDCAD = 1.4160


def _cad_tiers_config() -> dict:
    """Build the C-ii CAD-repriced tiers config from the live USD config:
    rename ``*_equity_usd`` -> ``*_equity_cad`` and re-price ×1.4160 (whole CAD);
    ``risk_profile.*`` caps are left untouched.  Mirrors PA C-ii."""
    cfg = json.loads(TIERS_CONFIG_PATH.read_text(encoding="utf-8"))
    for spec in cfg.get("tiers", {}).values():
        for base in ("min_equity", "max_equity", "demotion_equity"):
            usd_key = base + "_usd"
            if usd_key in spec:
                v = spec.pop(usd_key)
                spec[base + "_cad"] = None if v is None else round(v * _USDCAD)
    return cfg


def _run_main(monkeypatch, tmp_path, *, snapshot, prior_state=None):
    snap_path = tmp_path / "portfolio_snapshot.json"
    out_path = tmp_path / "tier_state.json"
    cfg_path = tmp_path / "tiers.json"
    snap_path.write_text(json.dumps(snapshot), encoding="utf-8")
    if prior_state is not None:
        out_path.write_text(json.dumps(prior_state), encoding="utf-8")
    # CAD-repriced tier bands (PA C-ii); read dual-key by the publisher.
    cfg_path.write_text(json.dumps(_cad_tiers_config()), encoding="utf-8")
    monkeypatch.setattr(tmmod, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(tmmod, "SNAPSHOT_PATH", snap_path)
    monkeypatch.setattr(tmmod, "OUT_PATH", out_path)
    monkeypatch.setattr(tmmod, "TIERS_CONFIG_PATH", cfg_path)
    rc = tmmod.main()
    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else None
    return rc, data


def test_main_cad_ok_137k_selects_pro_growth(monkeypatch, tmp_path):
    # CAD 137k -> PRO_GROWTH band [35.4k, 226.56k) under the CAD-repriced bands.
    snapshot = {
        "ibkr_equity": 137_000.0,        # CAD — the broker-native tier driver
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "kraken_equity": 258.5,
        "coinbase_equity": 0.0,
    }
    rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot)
    assert rc == 0
    assert data["tier_name"] == "PRO_GROWTH"
    assert data["current_equity_cad"] == 137_000.0   # ibkr_equity drives the band
    assert data["tier_name"] != "SCALE"


def test_main_cad_ok_above_226k_selects_scale(monkeypatch, tmp_path):
    # CAD >= 226.56k -> SCALE under the CAD-repriced bands.
    snapshot = {
        "ibkr_equity": 230_000.0,        # CAD
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": True,
        "kraken_equity": 0.0,
        "coinbase_equity": 0.0,
    }
    rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot)
    assert rc == 0
    assert data["tier_name"] == "SCALE"
    assert data["current_equity_cad"] == 230_000.0
    assert data["tier_name"] != "PRO_GROWTH"


def test_main_cad_not_ok_holds_prior_tier(monkeypatch, tmp_path, caplog):
    # ibkr_equity_currency_ok false -> HOLD the last persisted tier; do NOT
    # recompute.  The ibkr_equity value is present but untrusted.
    snapshot = {
        "ibkr_equity": 500_000.0,        # CAD — present but currency not ok
        "ibkr_equity_currency": "CAD",
        "ibkr_equity_currency_ok": False,
        "kraken_equity": 100_000.0,
        "coinbase_equity": 0.0,
    }
    prior = {
        "schema_version": "tier_state.v3",
        "tier_name": "PRO_GROWTH",
        "current_equity_cad": 137_000.0,
    }
    with caplog.at_level(logging.WARNING, logger=tmmod.LOG.name):
        rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot, prior_state=prior)
    assert rc == 0
    # tier_state.json is left untouched (held) — no recompute occurred.
    assert data == prior
    assert data["tier_name"] == "PRO_GROWTH"
    assert data["current_equity_cad"] == 137_000.0   # untouched
    assert any("TIER_HELD_NO_CAD_EQUITY" in r.getMessage() for r in caplog.records)


def test_main_cad_field_absent_treated_as_not_ok(monkeypatch, tmp_path, caplog):
    # A snapshot lacking the CAD currency-ok marker entirely must be treated as
    # fail-closed (hold).
    snapshot = {
        "ibkr_equity": 500_000.0,        # CAD, but no currency_ok marker
        "kraken_equity": 100_000.0,
        "coinbase_equity": 0.0,
    }
    prior = {"schema_version": "tier_state.v3", "tier_name": "STARTER",
             "current_equity_cad": 10_000.0}
    with caplog.at_level(logging.WARNING, logger=tmmod.LOG.name):
        rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot, prior_state=prior)
    assert rc == 0
    assert data == prior                              # held, no recompute
    assert any("TIER_HELD_NO_CAD_EQUITY" in r.getMessage() for r in caplog.records)


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


# ---------------------------------------------------------------------------
# market_is_open helper — pure RTH boundary checks.
#
# RTH window = weekday (Mon-Fri) AND 14:30 <= UTC time-of-day < 21:00.
# 2026-06-15 is a Monday; 2026-06-13 is a Saturday.
# ---------------------------------------------------------------------------

def test_market_is_open_just_before_open_closed():
    # 14:29 UTC on a weekday — one minute before the 14:30 open.
    assert market_is_open(datetime(2026, 6, 15, 14, 29, tzinfo=timezone.utc)) is False


def test_market_is_open_at_open_is_open():
    # 14:30 UTC exactly — the open boundary is inclusive.
    assert market_is_open(datetime(2026, 6, 15, 14, 30, tzinfo=timezone.utc)) is True


def test_market_is_open_mid_session_is_open():
    assert market_is_open(datetime(2026, 6, 15, 17, 0, tzinfo=timezone.utc)) is True


def test_market_is_open_just_before_close_is_open():
    # 20:59 UTC — still inside the session (close is exclusive at 21:00).
    assert market_is_open(datetime(2026, 6, 15, 20, 59, tzinfo=timezone.utc)) is True


def test_market_is_open_at_close_is_closed():
    # 21:00 UTC exactly — the close boundary is exclusive.
    assert market_is_open(datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc)) is False


def test_market_is_open_after_close_is_closed():
    # 02:00 UTC on a weekday — deep off-session.
    assert market_is_open(datetime(2026, 6, 15, 2, 0, tzinfo=timezone.utc)) is False


def test_market_is_open_weekend_is_closed():
    # Saturday mid-session-time-of-day — weekends are always closed.
    assert market_is_open(datetime(2026, 6, 13, 17, 0, tzinfo=timezone.utc)) is False


# ---------------------------------------------------------------------------
# main() now passes the REAL US-equity market-open state (was hardcoded False).
# These exercise the CLI through _run_main with tmmod.market_is_open stubbed so
# the wiring — not a fixed clock — is under test.  Snapshot equity 140k is below
# SCALE's demotion gate (144k) and resolves naively to PRO_GROWTH; prior tier is
# SCALE so a SCALE -> PRO_GROWTH demotion is the candidate.
# ---------------------------------------------------------------------------

_DEMOTE_SNAPSHOT = {
    "ibkr_equity": 200_000.0,            # CAD: below SCALE demotion gate (203 904),
    "ibkr_equity_currency": "CAD",       # naive PRO_GROWTH (< 226 560)
    "ibkr_equity_currency_ok": True,
}
_PRIOR_SCALE = {"schema_version": "tier_state.v3", "tier_name": "SCALE",
                "current_equity_cad": 240_000.0}


def test_main_demotion_defers_when_market_open(monkeypatch, tmp_path):
    # With the real helper reporting OPEN, main() must DEFER the demotion:
    # tier held at SCALE, demotion_pending True, applies at next market open.
    monkeypatch.setattr(tmmod, "market_is_open", lambda _now: True)
    rc, data = _run_main(
        monkeypatch, tmp_path, snapshot=_DEMOTE_SNAPSHOT, prior_state=_PRIOR_SCALE
    )
    assert rc == 0
    assert data["tier_name"] == "SCALE"                       # UNCHANGED
    assert data["demotion_pending"] is True
    assert data["demotion_pending_to"] == "PRO_GROWTH"        # lower tier
    assert data["demotion_pending_reason"] == "mid_session_demotion_deferred"
    assert data["demotion_applies_at"] == "next_market_open"


def test_main_demotion_applies_when_market_closed(monkeypatch, tmp_path):
    # Same equity, but the real helper reports CLOSED -> demotion APPLIES.
    monkeypatch.setattr(tmmod, "market_is_open", lambda _now: False)
    rc, data = _run_main(
        monkeypatch, tmp_path, snapshot=_DEMOTE_SNAPSHOT, prior_state=_PRIOR_SCALE
    )
    assert rc == 0
    assert data["tier_name"] == "PRO_GROWTH"                  # tier CHANGED
    assert data["demotion_pending"] is False


def test_main_promotion_unaffected_by_market_open(monkeypatch, tmp_path):
    # A promotion applies regardless of market_open — confirms only the
    # *demotion* deferral path consults the helper.  Prior PRO_GROWTH, CAD 230k
    # promotes to SCALE even with the market reported OPEN.
    monkeypatch.setattr(tmmod, "market_is_open", lambda _now: True)
    snapshot = {"ibkr_equity": 230_000.0, "ibkr_equity_currency": "CAD",
                "ibkr_equity_currency_ok": True}
    prior = {"schema_version": "tier_state.v3", "tier_name": "PRO_GROWTH",
             "current_equity_cad": 200_000.0}
    rc, data = _run_main(monkeypatch, tmp_path, snapshot=snapshot, prior_state=prior)
    assert rc == 0
    assert data["tier_name"] == "SCALE"                       # promotion applied
    assert data["demotion_pending"] is False

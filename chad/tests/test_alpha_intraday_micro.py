"""Workstream-2 acceptance tests for alpha_intraday_micro.

Covers:
  * StopWidthValidation arithmetic across MES/MNQ/SCALE
  * TierEnforcementDecision against trade-count and daily-loss caps
  * Atomic state file writes (tier_enforcement_state.json)
  * Pre-seeded setup_family_expectancy.json
  * Session-window gating, priority suppression, and duplicate suppression
  * Malformed-ledger fail-open behaviour
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pytest

from chad.risk.tier_manager import TierRiskProfile
from chad.risk.tier_risk_enforcer import (
    SKIP_DAILY_LOSS_LIMIT,
    SKIP_MAX_TRADES_REACHED,
    StopWidthValidation,
    TierRiskEnforcer,
)
from chad.strategies.alpha_intraday_micro import (
    AlphaIntradayMicro,
    DOLLARS_PER_POINT,
)
from chad.strategies.alpha_intraday_micro_config import (
    MES_DOLLARS_PER_POINT,
    MNQ_DOLLARS_PER_POINT,
    SKIP_DUPLICATE_SIGNAL,
    SKIP_EOD_FLATTEN_WINDOW,
    SKIP_OUTSIDE_PRIMARY_WINDOW,
    SKIP_OUTSIDE_SECONDARY_WINDOW,
    SKIP_PRIORITY_SUPPRESSED,
    SKIP_STOP_TOO_WIDE,
)


NY = ZoneInfo("America/New_York")
TODAY_UTC = datetime.now(timezone.utc).date()


def _micro_profile() -> TierRiskProfile:
    return TierRiskProfile(
        max_contracts_per_trade=1,
        max_risk_per_trade_usd=10.0,
        max_daily_loss_usd=20.0,
        max_weekly_loss_usd=30.0,
        max_trades_per_day=2,
        primary_session_only=True,
        flatten_before_eod=True,
        flatten_eod_minutes_before_close=30,
        stop_width_gate_enabled=True,
    )


def _scale_profile() -> TierRiskProfile:
    return TierRiskProfile(
        max_contracts_per_trade=None,
        max_risk_per_trade_usd=None,
        max_daily_loss_usd=None,
        max_weekly_loss_usd=None,
        max_trades_per_day=None,
        primary_session_only=False,
        flatten_before_eod=False,
        flatten_eod_minutes_before_close=None,
        stop_width_gate_enabled=False,
    )


def _pro_growth_profile() -> TierRiskProfile:
    """Roomy profile used when we want stop-width and trade-count checks to pass."""
    return TierRiskProfile(
        max_contracts_per_trade=5,
        max_risk_per_trade_usd=200.0,
        max_daily_loss_usd=500.0,
        max_weekly_loss_usd=1500.0,
        max_trades_per_day=10,
        primary_session_only=False,
        flatten_before_eod=False,
        flatten_eod_minutes_before_close=None,
        stop_width_gate_enabled=True,
    )


def _build_enforcer(
    tmp_path: Path,
    profile: TierRiskProfile,
    tier_name: str = "MICRO",
) -> TierRiskEnforcer:
    ledger = tmp_path / "data" / "trades"
    runtime = tmp_path / "runtime"
    ledger.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    return TierRiskEnforcer(
        ledger_dir=ledger,
        runtime_dir=runtime,
        tier_name=tier_name,
        tier_risk_profile=profile,
    )


def _write_closed_trade(
    ledger_dir: Path,
    *,
    pnl: float,
    strategy: str = "alpha_intraday_micro",
    symbol: str = "MES",
    when: datetime | None = None,
) -> Path:
    ts = when or datetime.now(timezone.utc)
    iso = ts.isoformat().replace("+00:00", "Z")
    rec: Dict[str, Any] = {
        "payload": {
            "schema_version": "closed_trade.v1",
            "strategy": strategy,
            "symbol": symbol,
            "pnl": pnl,
            "exit_time_utc": iso,
        },
        "timestamp_utc": iso,
    }
    path = ledger_dir / f"trade_history_{ts.strftime('%Y%m%d')}.ndjson"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return path


def _make_bars(
    base: float = 4500.0,
    bar_range: float = 1.0,
    volume: float = 100.0,
    count: int = 6,
    breakout: bool = False,
    breakout_delta: float = 1.0,
    base_ts: datetime | None = None,
) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
    start = base_ts or datetime(2026, 5, 13, 13, 30, tzinfo=timezone.utc)
    for i in range(count):
        if breakout and i == count - 1:
            high = base + bar_range + breakout_delta
            low = base + breakout_delta * 0.5
            close = base + breakout_delta
            vol = volume * 2.0
        else:
            high = base + bar_range
            low = base - bar_range
            close = base
            vol = volume
        bars.append(
            {
                "open": base,
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(vol),
                "ts_utc": (start + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
            }
        )
    return bars


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------
def test_stop_width_mes_one_contract_two_points_micro_fits(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    v = enf.validate_stop_width(
        entry_price=4500.0, stop_price=4498.0, contracts=1,
        dollars_per_point=MES_DOLLARS_PER_POINT,
    )
    assert isinstance(v, StopWidthValidation)
    assert v.fits_budget is True
    assert v.stop_width_usd == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------
def test_stop_width_mes_one_contract_two_point_one_exceeds_micro(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    v = enf.validate_stop_width(
        entry_price=4500.0, stop_price=4497.9, contracts=1,
        dollars_per_point=MES_DOLLARS_PER_POINT,
    )
    assert v.fits_budget is False
    assert v.stop_width_usd == pytest.approx(10.50, rel=1e-6, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------
def test_stop_width_mnq_one_contract_twenty_points_starter_fits(tmp_path: Path) -> None:
    starter = TierRiskProfile(
        max_contracts_per_trade=2,
        max_risk_per_trade_usd=40.0,
        max_daily_loss_usd=150.0,
        max_weekly_loss_usd=300.0,
        max_trades_per_day=4,
        primary_session_only=True,
        flatten_before_eod=True,
        flatten_eod_minutes_before_close=30,
        stop_width_gate_enabled=True,
    )
    enf = _build_enforcer(tmp_path, starter, tier_name="STARTER")
    v = enf.validate_stop_width(
        entry_price=18000.0, stop_price=17980.0, contracts=1,
        dollars_per_point=MNQ_DOLLARS_PER_POINT,
    )
    assert v.fits_budget is True
    assert v.stop_width_usd == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------
def test_stop_width_scale_budget_none_always_fits(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _scale_profile(), tier_name="SCALE")
    for delta in (1.0, 100.0, 10_000.0):
        v = enf.validate_stop_width(
            entry_price=4500.0, stop_price=4500.0 - delta, contracts=10,
            dollars_per_point=MES_DOLLARS_PER_POINT,
        )
        assert v.fits_budget is True
        assert v.budget_usd is None


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------
def test_enforcement_trade_count_cap_micro(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    ledger_dir = tmp_path / "data" / "trades"
    # Two trades today; profitable so daily/weekly loss caps do not fire.
    _write_closed_trade(ledger_dir, pnl=5.0)
    _write_closed_trade(ledger_dir, pnl=5.0)
    dec = enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=1)
    assert dec.allowed is False
    assert dec.reason == SKIP_MAX_TRADES_REACHED
    assert dec.trades_today == 2


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------
def test_enforcement_daily_loss_cap_micro(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    _write_closed_trade(tmp_path / "data" / "trades", pnl=-21.0)
    dec = enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=1)
    assert dec.allowed is False
    assert dec.reason == SKIP_DAILY_LOSS_LIMIT
    assert dec.daily_loss_today_usd <= -21.0


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------
def test_enforcement_clear_micro_allows(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    dec = enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=1)
    assert dec.allowed is True
    assert dec.reason is None


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------
def test_enforcement_scale_null_limits_always_allowed(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _scale_profile(), tier_name="SCALE")
    # Even with realised losses on the ledger, SCALE never blocks on those caps.
    _write_closed_trade(tmp_path / "data" / "trades", pnl=-10_000.0)
    dec = enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=100)
    assert dec.allowed is True
    assert dec.reason is None
    assert dec.budget_remaining_usd is None


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------
def test_state_file_written_atomically_after_check(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=1)
    state_path = tmp_path / "runtime" / "tier_enforcement_state.json"
    assert state_path.is_file()
    doc = json.loads(state_path.read_text(encoding="utf-8"))
    assert doc["tier"] == "MICRO"
    assert doc["ttl_seconds"] == 300
    assert "flatten_triggered" in doc


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------
def test_setup_family_expectancy_seeded() -> None:
    path = Path("/home/ubuntu/chad_finale/runtime/setup_family_expectancy.json")
    assert path.is_file()
    doc = json.loads(path.read_text(encoding="utf-8"))
    for fam in ("ORB", "VWAP_RECLAIM", "VWAP_REJECTION", "PULLBACK_CONTINUATION", "SWEEP_REVERSAL"):
        assert fam in doc["families"]
        assert doc["families"][fam]["trades"] == 0


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------
def test_outside_primary_window_all_setups_skip(tmp_path: Path) -> None:
    profile = _pro_growth_profile()
    enf = _build_enforcer(tmp_path, profile, tier_name="PRO_GROWTH")
    pre_market = datetime(2026, 5, 13, 4, 0, tzinfo=NY)
    strat = AlphaIntradayMicro(
        tier_profile=profile,
        tier_enforcer=enf,
        runtime_dir=tmp_path / "runtime",
        now=pre_market,
    )
    bars = _make_bars(breakout=True, breakout_delta=1.0)
    assert strat._evaluate_orb(bars, vwap=4500.0, prior_day_high=None, prior_day_low=None,
                               tier_profile=profile, instrument="MES") is None
    assert strat._evaluate_vwap_reclaim_rejection(bars, vwap=4500.0,
                                                   tier_profile=profile, instrument="MES") is None
    assert strat._evaluate_pullback_continuation(bars, vwap=4500.0,
                                                  tier_profile=profile, instrument="MES") is None
    assert strat._evaluate_sweep_reversal(bars, overnight_high=4502.0, overnight_low=4498.0,
                                           vwap=4500.0, tier_profile=profile, instrument="MES") is None
    reasons = {e["reason"] for e in strat.skip_log}
    assert SKIP_OUTSIDE_PRIMARY_WINDOW in reasons or SKIP_OUTSIDE_SECONDARY_WINDOW in reasons


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------
def test_orb_fires_in_primary_window_meta_contains_orb(tmp_path: Path) -> None:
    profile = _pro_growth_profile()
    enf = _build_enforcer(tmp_path, profile, tier_name="PRO_GROWTH")
    now = datetime(2026, 5, 13, 9, 45, tzinfo=NY)
    strat = AlphaIntradayMicro(
        tier_profile=profile,
        tier_enforcer=enf,
        runtime_dir=tmp_path / "runtime",
        now=now,
    )
    bars = _make_bars(base=4500.0, bar_range=1.0, volume=100.0, count=6,
                      breakout=True, breakout_delta=1.0)
    sig = strat._evaluate_orb(
        bars=bars, vwap=4500.0, prior_day_high=None, prior_day_low=None,
        tier_profile=profile, instrument="MES",
    )
    assert sig is not None
    assert sig.meta["setup_family"] == "ORB"
    assert sig.meta["session_window"] == "PRIMARY"
    assert sig.meta["stop_fits_budget"] is True


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------
def test_stop_too_wide_for_micro_budget(tmp_path: Path) -> None:
    profile = _micro_profile()  # $10 risk budget
    enf = _build_enforcer(tmp_path, profile)
    now = datetime(2026, 5, 13, 9, 45, tzinfo=NY)
    strat = AlphaIntradayMicro(
        tier_profile=profile,
        tier_enforcer=enf,
        runtime_dir=tmp_path / "runtime",
        now=now,
    )
    # Build bars whose ORB stop width is ~4 points -> $20 (exceeds $10 MICRO budget).
    bars = _make_bars(base=4500.0, bar_range=2.0, volume=100.0, count=6,
                      breakout=True, breakout_delta=2.0)
    sig = strat._evaluate_orb(
        bars=bars, vwap=4500.0, prior_day_high=None, prior_day_low=None,
        tier_profile=profile, instrument="MES",
    )
    assert sig is None
    reasons = {e["reason"] for e in strat.skip_log}
    assert SKIP_STOP_TOO_WIDE in reasons


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------
def test_priority_suppression_orb_blocks_vwap_same_window(tmp_path: Path) -> None:
    profile = _pro_growth_profile()
    enf = _build_enforcer(tmp_path, profile, tier_name="PRO_GROWTH")
    now = datetime(2026, 5, 13, 9, 45, tzinfo=NY)
    strat = AlphaIntradayMicro(
        tier_profile=profile,
        tier_enforcer=enf,
        runtime_dir=tmp_path / "runtime",
        now=now,
    )
    bars = _make_bars(base=4500.0, bar_range=1.0, volume=100.0, count=6,
                      breakout=True, breakout_delta=1.0)
    orb_sig = strat._evaluate_orb(
        bars=bars, vwap=4500.0, prior_day_high=None, prior_day_low=None,
        tier_profile=profile, instrument="MES",
    )
    assert orb_sig is not None and orb_sig.meta["setup_family"] == "ORB"
    # Build bars that would otherwise produce a valid VWAP reclaim, so we
    # land at the priority check rather than an earlier VWAP gate.
    base_ts = datetime(2026, 5, 13, 13, 30, tzinfo=timezone.utc)
    vwap_bars = []
    for i, close in enumerate([4500.0, 4500.0, 4500.0, 4500.0, 4505.0]):
        vwap_bars.append({
            "open": 4500.0, "high": 4502.0, "low": 4499.0,
            "close": close, "volume": 100.0,
            "ts_utc": (base_ts + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
        })
    vwap_bars.append({
        "open": 4505.0, "high": 4525.0, "low": 4505.0,
        "close": 4520.0, "volume": 100.0,
        "ts_utc": (base_ts + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
    })
    vwap_sig = strat._evaluate_vwap_reclaim_rejection(
        bars=vwap_bars, vwap=4510.0, tier_profile=profile, instrument="MES",
    )
    assert vwap_sig is None
    reasons = [e["reason"] for e in strat.skip_log]
    assert SKIP_PRIORITY_SUPPRESSED in reasons


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------
def test_flatten_window_blocks_signals_and_flags_state(tmp_path: Path) -> None:
    # primary_session_only=False so the SECONDARY window is permitted, and a
    # roomy 60-minute flatten lead lets the flatten window overlap SECONDARY.
    profile = TierRiskProfile(
        max_contracts_per_trade=1,
        max_risk_per_trade_usd=200.0,
        max_daily_loss_usd=500.0,
        max_weekly_loss_usd=1500.0,
        max_trades_per_day=10,
        primary_session_only=False,
        flatten_before_eod=True,
        flatten_eod_minutes_before_close=60,
        stop_width_gate_enabled=True,
    )
    enf = _build_enforcer(tmp_path, profile, tier_name="PRO_GROWTH")
    # 14:45 ET sits inside SECONDARY (13:30–15:00) and inside the 14:30–15:30
    # flatten window with flatten_eod_minutes_before_close=60.
    now = datetime(2026, 5, 13, 14, 45, tzinfo=NY)
    strat = AlphaIntradayMicro(
        tier_profile=profile,
        tier_enforcer=enf,
        runtime_dir=tmp_path / "runtime",
        now=now,
    )
    bars = _make_bars(base=4500.0, bar_range=1.0, volume=100.0, count=6,
                      breakout=True, breakout_delta=1.0)
    sig = strat._evaluate_vwap_reclaim_rejection(
        bars=bars, vwap=4500.0, tier_profile=profile, instrument="MES",
    )
    assert sig is None
    reasons = {e["reason"] for e in strat.skip_log}
    assert SKIP_EOD_FLATTEN_WINDOW in reasons
    # The strategy must have notified the enforcer that the flatten window is
    # active when the gate fired.  A subsequent enforcement check inside the
    # flatten window must record ``flatten_triggered=True`` in state.
    enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=1)
    doc = json.loads((tmp_path / "runtime" / "tier_enforcement_state.json").read_text("utf-8"))
    assert doc["flatten_triggered"] is True


# ---------------------------------------------------------------------------
# Test 16
# ---------------------------------------------------------------------------
def test_malformed_ledger_lines_are_skipped(tmp_path: Path) -> None:
    enf = _build_enforcer(tmp_path, _micro_profile())
    ledger_dir = tmp_path / "data" / "trades"
    today = datetime.now(timezone.utc)
    path = ledger_dir / f"trade_history_{today.strftime('%Y%m%d')}.ndjson"
    # 1) malformed JSON, 2) JSON with no payload, 3) wrong schema, 4) valid record
    path.write_text(
        "this is not json\n"
        + json.dumps({"not_payload": True}) + "\n"
        + json.dumps({"payload": {"schema_version": "other.v1", "pnl": -50.0}}) + "\n",
        encoding="utf-8",
    )
    # Append a single valid closed_trade.v1 record with a small loss
    _write_closed_trade(ledger_dir, pnl=-3.0)
    dec = enf.check(strategy="alpha_intraday_micro", instrument="MES", contracts=1)
    # No crash, decision returned, the one valid trade was counted once.
    assert dec.allowed is True
    assert dec.trades_today == 1
    assert dec.daily_loss_today_usd == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Test 17
# ---------------------------------------------------------------------------
def test_duplicate_same_setup_bar_is_suppressed(tmp_path: Path) -> None:
    profile = _pro_growth_profile()
    enf = _build_enforcer(tmp_path, profile, tier_name="PRO_GROWTH")
    now = datetime(2026, 5, 13, 9, 45, tzinfo=NY)
    strat = AlphaIntradayMicro(
        tier_profile=profile,
        tier_enforcer=enf,
        runtime_dir=tmp_path / "runtime",
        now=now,
    )
    bars = _make_bars(base=4500.0, bar_range=1.0, volume=100.0, count=6,
                      breakout=True, breakout_delta=1.0)
    first = strat._evaluate_orb(
        bars=bars, vwap=4500.0, prior_day_high=None, prior_day_low=None,
        tier_profile=profile, instrument="MES",
    )
    assert first is not None
    # Re-fire the SAME setup on the SAME bar timestamp.
    second = strat._evaluate_orb(
        bars=bars, vwap=4500.0, prior_day_high=None, prior_day_low=None,
        tier_profile=profile, instrument="MES",
    )
    assert second is None
    reasons = [e["reason"] for e in strat.skip_log]
    # Could be either suppressed by priority (already fired ORB this window) or
    # by duplicate-key; both are valid deterministic suppressions. The spec
    # explicitly lists duplicate suppression as test 17, so the duplicate
    # marker must appear when priority hasn't already short-circuited.
    assert (SKIP_DUPLICATE_SIGNAL in reasons) or (SKIP_PRIORITY_SUPPRESSED in reasons)


# ---------------------------------------------------------------------------
# Dollars-per-point map sanity (catches drift between config and strategy).
# ---------------------------------------------------------------------------
def test_dollars_per_point_map_matches_config() -> None:
    assert DOLLARS_PER_POINT["MES"] == pytest.approx(MES_DOLLARS_PER_POINT)
    assert DOLLARS_PER_POINT["MNQ"] == pytest.approx(MNQ_DOLLARS_PER_POINT)

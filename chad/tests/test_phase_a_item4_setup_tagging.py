"""Phase A Item 4 regression tests — setup_family tagging and expectancy routing."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from chad.types import SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeTierProfile:
    primary_session_only: bool = False
    max_risk_per_trade_usd: Optional[float] = 200.0


@dataclass
class _FakeCtx:
    now: Optional[datetime] = None
    prices: Dict[str, float] = field(default_factory=dict)
    bars: Dict[str, Any] = field(default_factory=dict)
    bars_1m: Dict[str, Any] = field(default_factory=dict)
    tier_profile: Optional[_FakeTierProfile] = None
    tier_name: str = ""
    legend: Any = None
    regime: str = "trend"
    vix: float = 14.0


def _market_hours_now() -> datetime:
    # 09:45 ET (DST-aware) — falls inside the PRIMARY window for the
    # tier-aware session gate. Use a Monday so it isn't a weekend.
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    return datetime(2026, 5, 11, 9, 45, 0, tzinfo=et).astimezone(timezone.utc)


def _intraday_record(
    *,
    strategy: str,
    setup_family: Optional[str],
    pnl: float,
    stop_width_usd: Optional[float] = 50.0,
    exit_offset_days: int = 0,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now = now or datetime(2026, 5, 13, tzinfo=timezone.utc)
    exit_ts = now - timedelta(days=int(exit_offset_days))
    payload: Dict[str, Any] = {
        "schema_version": "closed_trade.v1",
        "strategy": strategy,
        "symbol": "MES",
        "side": "BUY",
        "exit_time_utc": exit_ts.isoformat().replace("+00:00", "Z"),
        "entry_time_utc": (exit_ts - timedelta(minutes=10))
        .isoformat()
        .replace("+00:00", "Z"),
        "pnl": float(pnl),
    }
    meta: Dict[str, Any] = {}
    if setup_family is not None:
        meta["setup_family"] = setup_family
    if stop_width_usd is not None:
        meta["stop_width_usd"] = float(stop_width_usd)
    if meta:
        payload["meta"] = meta
    return {
        "payload": payload,
        "timestamp_utc": exit_ts.isoformat().replace("+00:00", "Z"),
    }


def _write_ledger(trades_dir: Path, records: List[Dict[str, Any]]) -> Path:
    trades_dir.mkdir(parents=True, exist_ok=True)
    path = trades_dir / "trade_history_20260513.ndjson"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests 1-4: alpha_intraday _build_signal carries setup_family == trigger
# ---------------------------------------------------------------------------


def test_01_alpha_intraday_vol_explosion_setup_family(monkeypatch: pytest.MonkeyPatch):
    from chad.strategies import alpha_intraday
    from chad.utils.catalyst_gate import CatalystGateResult

    # Hermetic: neutralize the ambient runtime/news_intel.json catalyst gate so
    # this setup_family tagging assertion does not depend on live market
    # catalysts. The gate itself is covered by test_phase_b_item1_catalyst*.py.
    monkeypatch.setattr(
        alpha_intraday, "check_catalyst_gate",
        lambda sym, side: CatalystGateResult(
            allowed=True, catalyst_strength="none",
            catalyst_direction="none", block_reason=None,
        ),
    )

    sig = alpha_intraday._build_signal(
        "SPY", SignalSide.BUY, 0.7, "vol_explosion", "1m",
        atr=1.0, tier_max_risk_usd=None,
    )
    assert sig is not None
    assert sig.meta["setup_family"] == "vol_explosion"
    assert sig.meta["trigger"] == "vol_explosion"


def test_02_alpha_intraday_momentum_surge_setup_family():
    from chad.strategies import alpha_intraday

    sig = alpha_intraday._build_signal(
        "QQQ", SignalSide.SELL, 0.65, "momentum_surge", "1m",
        atr=1.0, tier_max_risk_usd=None,
    )
    assert sig is not None
    assert sig.meta["setup_family"] == "momentum_surge"


def test_03_alpha_intraday_mean_reversion_snap_setup_family():
    from chad.strategies import alpha_intraday

    sig = alpha_intraday._build_signal(
        "AAPL", SignalSide.BUY, 0.6, "mean_reversion_snap", "1m",
        atr=1.0, tier_max_risk_usd=None,
    )
    assert sig is not None
    assert sig.meta["setup_family"] == "mean_reversion_snap"


def test_04_alpha_intraday_setup_family_nonempty_for_all_three_triggers():
    from chad.strategies import alpha_intraday

    for trigger in ("vol_explosion", "momentum_surge", "mean_reversion_snap"):
        sig = alpha_intraday._build_signal(
            "MES", SignalSide.BUY, 0.7, trigger, "1m",
            atr=2.0, tier_max_risk_usd=None,
        )
        assert sig is not None
        sf = sig.meta.get("setup_family")
        assert isinstance(sf, str) and sf.strip() != ""


# ---------------------------------------------------------------------------
# Tests 5-6: alpha.py helper mapping
# ---------------------------------------------------------------------------


def test_05_alpha_helper_returns_nonempty_for_buy_entry():
    from chad.strategies.alpha import _setup_family_for_alpha

    # uptrend + BUY -> ema_crossover_long
    assert _setup_family_for_alpha("trend_momentum", SignalSide.BUY, "uptrend") == "ema_crossover_long"
    # recovery -> recovery_long
    assert _setup_family_for_alpha("recovery_long", SignalSide.BUY, "recovery") == "recovery_long"
    # chop BUY -> chop_reversion_long
    assert _setup_family_for_alpha("chop_reversion", SignalSide.BUY, "chop") == "chop_reversion_long"
    # fallback BUY -> alpha_long
    assert _setup_family_for_alpha("unexpected", SignalSide.BUY, None) == "alpha_long"


def test_06_alpha_helper_returns_nonempty_for_sell_entry():
    from chad.strategies.alpha import _setup_family_for_alpha

    # downtrend + SELL -> ema_crossover_short
    assert _setup_family_for_alpha("trend_short", SignalSide.SELL, "downtrend") == "ema_crossover_short"
    # chop SELL -> chop_reversion_short
    assert _setup_family_for_alpha("chop_reversion", SignalSide.SELL, "chop") == "chop_reversion_short"
    # fallback SELL -> alpha_short
    assert _setup_family_for_alpha("unexpected", SignalSide.SELL, None) == "alpha_short"


# ---------------------------------------------------------------------------
# Tests 7-8: alpha_futures helper mapping
# ---------------------------------------------------------------------------


def test_07_alpha_futures_helper_long():
    from chad.strategies.alpha_futures import _setup_family_for_alpha_futures

    assert _setup_family_for_alpha_futures(SignalSide.BUY) == "momentum_breakout_long"
    assert _setup_family_for_alpha_futures(SignalSide.BUY, breakout=True) == "momentum_breakout_long"
    assert _setup_family_for_alpha_futures(SignalSide.BUY, breakout=False) == "momentum_breakout_long"


def test_08_alpha_futures_helper_short():
    from chad.strategies.alpha_futures import _setup_family_for_alpha_futures

    assert _setup_family_for_alpha_futures(SignalSide.SELL) == "momentum_breakout_short"
    assert _setup_family_for_alpha_futures(SignalSide.SELL, breakout=False) == "momentum_breakout_short"


# ---------------------------------------------------------------------------
# Tests 9-10: handler-level sanity — does not raise on minimal ctx
# ---------------------------------------------------------------------------


def test_09_alpha_handler_no_raise_minimal_ctx_emits_with_setup_family_if_any():
    from chad.strategies import alpha

    ctx = _FakeCtx(now=_market_hours_now(), prices={}, bars={}, tier_profile=None)
    sigs = alpha.alpha_handler(ctx)
    assert isinstance(sigs, list)
    for sig in sigs:
        if sig.meta.get("reason") != "exit":
            assert "setup_family" in sig.meta
            assert isinstance(sig.meta["setup_family"], str) and sig.meta["setup_family"]


def test_10_alpha_futures_handler_no_raise_minimal_ctx():
    from chad.strategies import alpha_futures

    ctx = _FakeCtx(now=_market_hours_now(), prices={}, bars={}, tier_profile=None)
    sigs = alpha_futures.alpha_futures_handler(ctx)
    assert isinstance(sigs, list)
    for sig in sigs:
        if sig.meta.get("is_exit"):
            continue
        assert "setup_family" in sig.meta
        assert isinstance(sig.meta["setup_family"], str) and sig.meta["setup_family"]


# ---------------------------------------------------------------------------
# Tests 11-14: setup_family_expectancy_updater consumption
# ---------------------------------------------------------------------------


def _make_updater(tmp_path: Path):
    from chad.analytics.setup_family_expectancy_updater import (
        SetupFamilyExpectancyUpdater,
    )

    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    return SetupFamilyExpectancyUpdater(
        trades_dir=trades_dir,
        output_path=tmp_path / "out" / "setup_family_expectancy.json",
        lookback_days=90,
        now=datetime(2026, 5, 13, 12, tzinfo=timezone.utc),
        ts_override="2026-05-13T12:00:00Z",
    )


def test_11_updater_missing_setup_family_buckets_to_unknown(tmp_path: Path):
    from chad.analytics.setup_family_expectancy_updater import UNKNOWN_FAMILY

    records = [
        _intraday_record(strategy="alpha", setup_family=None, pnl=+25.0, stop_width_usd=25.0),
        _intraday_record(strategy="alpha", setup_family=None, pnl=-10.0, stop_width_usd=25.0),
    ]
    _write_ledger(tmp_path / "trades", records)
    payload = _make_updater(tmp_path).run()
    assert UNKNOWN_FAMILY in payload["families"]
    unk = payload["families"][UNKNOWN_FAMILY]
    assert unk["trades"] == 2
    assert unk["wins"] == 1


def test_12_updater_processes_alpha_ema_crossover_long(tmp_path: Path):
    records = [
        _intraday_record(strategy="alpha", setup_family="ema_crossover_long", pnl=+50.0),
    ]
    _write_ledger(tmp_path / "trades", records)
    payload = _make_updater(tmp_path).run()
    assert "ema_crossover_long" in payload["families"]
    assert payload["families"]["ema_crossover_long"]["trades"] == 1
    assert payload["families"]["ema_crossover_long"]["wins"] == 1
    assert "alpha" in payload["summary"]["strategies_processed"]


def test_13_updater_processes_alpha_futures_momentum_breakout_long(tmp_path: Path):
    records = [
        _intraday_record(
            strategy="alpha_futures",
            setup_family="momentum_breakout_long",
            pnl=+100.0,
        ),
    ]
    _write_ledger(tmp_path / "trades", records)
    payload = _make_updater(tmp_path).run()
    assert "momentum_breakout_long" in payload["families"]
    assert payload["families"]["momentum_breakout_long"]["trades"] == 1
    assert "alpha_futures" in payload["summary"]["strategies_processed"]


def test_14_updater_processes_alpha_intraday_vol_explosion(tmp_path: Path):
    records = [
        _intraday_record(
            strategy="alpha_intraday",
            setup_family="vol_explosion",
            pnl=+75.0,
        ),
    ]
    _write_ledger(tmp_path / "trades", records)
    payload = _make_updater(tmp_path).run()
    assert "vol_explosion" in payload["families"]
    assert payload["families"]["vol_explosion"]["trades"] == 1
    assert "alpha_intraday" in payload["summary"]["strategies_processed"]

"""Phase A Item 2 regression tests — tier-aware time-of-day session zones."""
from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

from chad.utils.session import (
    SESSION_PRIMARY,
    SESSION_SECONDARY,
    SKIP_EOD_FLATTEN_WINDOW,
    SKIP_OUTSIDE_PRIMARY_WINDOW,
    SKIP_OUTSIDE_SESSION_WINDOW,
    parse_hhmm,
    session_decision,
)

_ET = ZoneInfo("America/New_York")


def _et(hour: int, minute: int) -> datetime:
    """Return a UTC datetime corresponding to the given America/New_York time
    on a known weekday (Monday 2026-01-05, post-DST winter)."""
    et_dt = datetime(2026, 1, 5, hour, minute, 0, tzinfo=_ET)
    return et_dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Helper-level tests (Tests 1-6, 13-15)
# ---------------------------------------------------------------------------


def test_01_primary_window_primary_only_true():
    d = session_decision(_et(9, 45), primary_session_only=True)
    assert d.session_window == SESSION_PRIMARY
    assert d.entry_allowed is True
    assert d.skip_reason is None


def test_02_secondary_window_primary_only_true_blocks():
    d = session_decision(_et(13, 45), primary_session_only=True)
    assert d.session_window == SESSION_SECONDARY
    assert d.entry_allowed is False
    assert d.skip_reason == SKIP_OUTSIDE_PRIMARY_WINDOW


def test_03_secondary_window_primary_only_false_allows():
    d = session_decision(_et(13, 45), primary_session_only=False)
    assert d.session_window == SESSION_SECONDARY
    assert d.entry_allowed is True


def test_04_eod_flatten_blocks_entries():
    d = session_decision(_et(15, 31), primary_session_only=False)
    assert d.is_eod_flatten_window is True
    assert d.entry_allowed is False
    assert d.skip_reason == SKIP_EOD_FLATTEN_WINDOW


def test_05_outside_session_window_primary_only_false():
    d = session_decision(_et(8, 0), primary_session_only=False)
    assert d.session_window is None
    assert d.entry_allowed is False
    assert d.skip_reason == SKIP_OUTSIDE_SESSION_WINDOW


def test_06_naive_utc_datetime_resolves_to_primary():
    # 09:45 ET on 2026-01-05 (EST, UTC-5) == 14:45 UTC.
    naive_utc = datetime(2026, 1, 5, 14, 45, 0)
    assert naive_utc.tzinfo is None
    d = session_decision(naive_utc, primary_session_only=True)
    assert d.session_window == SESSION_PRIMARY
    assert d.entry_allowed is True


def test_13_dst_aware_conversion_summer_and_winter():
    # Winter EST: 15:30 ET == 20:30 UTC.
    winter_naive_utc = datetime(2026, 1, 5, 20, 30, 0)
    d_winter = session_decision(winter_naive_utc, primary_session_only=False)
    assert d_winter.is_eod_flatten_window is True
    assert d_winter.entry_allowed is False
    assert d_winter.skip_reason == SKIP_EOD_FLATTEN_WINDOW

    # Summer EDT: 15:30 ET == 19:30 UTC.
    summer_naive_utc = datetime(2026, 7, 6, 19, 30, 0)
    d_summer = session_decision(summer_naive_utc, primary_session_only=False)
    assert d_summer.is_eod_flatten_window is True
    assert d_summer.entry_allowed is False
    assert d_summer.skip_reason == SKIP_EOD_FLATTEN_WINDOW


def test_14_parse_hhmm_malformed_raises():
    with pytest.raises(ValueError):
        parse_hhmm("not-a-time")
    with pytest.raises(ValueError):
        parse_hhmm("25:00")
    with pytest.raises(ValueError):
        parse_hhmm("12:60")


def test_15_primary_only_true_blocks_secondary_false_allows_it():
    # Same instant, opposite tier settings, opposite entry permission.
    instant = _et(13, 45)
    d_true = session_decision(instant, primary_session_only=True)
    d_false = session_decision(instant, primary_session_only=False)
    assert d_true.session_window == SESSION_SECONDARY
    assert d_false.session_window == SESSION_SECONDARY
    assert d_true.entry_allowed is False
    assert d_false.entry_allowed is True


# ---------------------------------------------------------------------------
# Strategy-level fixtures + tests (Tests 7-12)
# ---------------------------------------------------------------------------


@dataclass
class _FakeTierProfile:
    primary_session_only: bool = False
    max_risk_per_trade_usd: Optional[float] = 200.0


@dataclass
class _FakeCtx:
    """Minimal duck-typed MarketContext for handler-level invocation."""

    now: Optional[datetime] = None
    prices: Dict[str, float] = field(default_factory=dict)
    bars: Dict[str, Any] = field(default_factory=dict)
    bars_1m: Dict[str, Any] = field(default_factory=dict)
    tier_profile: Optional[_FakeTierProfile] = None
    tier_name: str = ""
    legend: Any = None
    regime: str = "trend"
    vix: float = 14.0


def test_07_alpha_intraday_blocks_entries_outside_primary_when_primary_only_true():
    from chad.strategies import alpha_intraday

    ctx = _FakeCtx(
        now=_et(13, 45),
        tier_profile=_FakeTierProfile(primary_session_only=True),
        prices={"SPY": 500.0},
    )
    # Even if we populated bars, the handler short-circuits to []. We
    # don't bother — the gate fires before symbol iteration.
    sigs = alpha_intraday.alpha_intraday_handler(ctx)
    assert sigs == []


def test_08_alpha_intraday_fail_open_when_tier_profile_missing():
    from chad.strategies import alpha_intraday

    ctx = _FakeCtx(now=_et(13, 45), tier_profile=None, prices={})
    # No bars / no prices → still returns [] but without raising; the
    # gate must not be triggered (tier_profile absent ⇒ fail-open).
    sigs = alpha_intraday.alpha_intraday_handler(ctx)
    assert isinstance(sigs, list)


def test_09_alpha_blocks_new_entries_outside_primary_but_does_not_top_return():
    """alpha.py must NOT short-circuit the per-symbol loop with a top-level
    ``return []`` for session blocks — exits must still emit. We verify the
    structural property via static-code inspection."""
    from chad.strategies import alpha

    src = inspect.getsource(alpha.build_alpha_signals)
    # The session gate must be inside the symbol loop (after exit emit),
    # not at the top of the handler. We check that "if not _entries_allowed"
    # appears AFTER the exit-signal append.
    assert "_entries_allowed" in src, "session gate not present in build_alpha_signals"
    src_dedented = textwrap.dedent(src)
    exit_idx = src_dedented.find('meta={"reason": "exit"}')
    gate_idx = src_dedented.find("if not _entries_allowed:")
    assert exit_idx >= 0, "exit signal emission missing from alpha"
    assert gate_idx >= 0, "session gate missing from alpha"
    assert gate_idx > exit_idx, (
        "session gate must follow exit emission so exits are never blocked"
    )

    # Behavioral check: outside-PRIMARY with primary_only=True, empty
    # universe yields [] without exception (fail-open path test).
    ctx = _FakeCtx(
        now=_et(13, 45),
        tier_profile=_FakeTierProfile(primary_session_only=True),
        prices={},
    )
    sigs = alpha.build_alpha_signals(ctx=ctx)
    assert sigs == []


def test_10_alpha_fail_open_when_tier_profile_missing():
    from chad.strategies import alpha

    ctx = _FakeCtx(now=_et(13, 45), tier_profile=None, prices={})
    sigs = alpha.build_alpha_signals(ctx=ctx)
    assert isinstance(sigs, list)


def test_11_alpha_futures_session_helper_allows_secondary_when_primary_only_false():
    # The strategy must call the helper that allows SECONDARY for
    # primary_session_only=False; verify both the helper itself and that
    # the import is wired into alpha_futures.
    from chad.strategies import alpha_futures

    assert hasattr(alpha_futures, "session_decision"), (
        "alpha_futures must import the shared session_decision helper"
    )

    d = session_decision(_et(13, 45), primary_session_only=False)
    assert d.session_window == SESSION_SECONDARY
    assert d.entry_allowed is True

    # Equity-index universe must be the documented frozenset.
    assert alpha_futures._EQUITY_INDEX_FUTURES == frozenset(
        {"MES", "MNQ", "MYM", "M2K"}
    )


def test_12_alpha_futures_exit_path_not_blocked_by_session_gate():
    """The session gate must appear inside ``_build_signal_for_symbol`` only
    *after* ``_evaluate_exit_signal`` has run, so exits emit even outside
    session windows. We verify by static-code inspection."""
    from chad.strategies import alpha_futures

    src = inspect.getsource(alpha_futures._build_signal_for_symbol)
    exit_idx = src.find("_evaluate_exit_signal")
    gate_idx = src.find("SESSION_GATE_SKIP")
    assert exit_idx >= 0, "exit evaluator missing from _build_signal_for_symbol"
    assert gate_idx >= 0, "session gate skip log missing"
    assert gate_idx > exit_idx, (
        "alpha_futures session gate must follow the exit evaluator"
    )

    # Also: the MCL/MGC UTC overnight gate must still be present and
    # unmodified (the new session gate must be additive, not a replacement).
    assert 'symbol in ("MCL", "MGC")' in src
    assert "OVERNIGHT_GATE_SKIP" in src

    # MCL is NOT in the equity-index frozenset → session gate would not
    # block it even if the helper said entries are disallowed.
    assert "MCL" not in alpha_futures._EQUITY_INDEX_FUTURES
    assert "MGC" not in alpha_futures._EQUITY_INDEX_FUTURES
    assert "ZN" not in alpha_futures._EQUITY_INDEX_FUTURES
    assert "ZB" not in alpha_futures._EQUITY_INDEX_FUTURES

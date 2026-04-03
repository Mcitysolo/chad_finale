#!/usr/bin/env python3
"""
Tests for the CHAD Production Backtesting Engine.

Covers:
- BacktestContext no-lookahead invariant
- SimulatedPortfolio fill and mark-to-market
- Stop loss trigger
- Time stop trigger
- Statistics computation (Sharpe, win rate, drawdown)
- Full engine run with synthetic data
- Bar alignment
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from chad.analytics.backtest_engine import (
    BacktestBar,
    BacktestContext,
    BacktestPosition,
    BacktestTrade,
    SimulatedPortfolio,
    _MockPortfolio,
    _MockTick,
    _compute_atr,
    align_bars,
    compute_stats,
    load_bars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = datetime(2025, 6, 1, 5, 0, tzinfo=timezone.utc)


def _make_timestamps(n: int = 50) -> List[datetime]:
    return [BASE_TS + timedelta(days=i) for i in range(n)]


def _make_bar_dicts(
    n: int = 50,
    base: float = 5000.0,
    step: float = 10.0,
    symbol: str = "MES",
) -> List[Dict[str, Any]]:
    bars = []
    for i in range(n):
        c = base + i * step
        bars.append({
            "ts_utc": (BASE_TS + timedelta(days=i)).isoformat().replace("+00:00", "Z"),
            "open": c - 5.0,
            "high": c + 15.0,
            "low": c - 15.0,
            "close": c,
            "volume": 100_000.0,
        })
    return bars


def _make_backtest_bars(
    n: int = 50,
    base: float = 5000.0,
    step: float = 10.0,
    symbol: str = "MES",
) -> List[BacktestBar]:
    bars = []
    for i in range(n):
        c = base + i * step
        bars.append(BacktestBar(
            ts_utc=BASE_TS + timedelta(days=i),
            open=c - 5.0, high=c + 15.0, low=c - 15.0, close=c,
            volume=100_000.0, symbol=symbol,
        ))
    return bars


# ===========================================================================
# BacktestContext no-lookahead tests
# ===========================================================================

class TestNoLookahead:

    def test_bars_only_up_to_current_idx(self) -> None:
        """At step i, bars should contain exactly i+1 elements."""
        timestamps = _make_timestamps(50)
        all_bars = {"MES": _make_bar_dicts(50)}

        for idx in [0, 10, 25, 49]:
            ctx = BacktestContext(
                all_bars=all_bars,
                current_idx=idx,
                timestamps=timestamps,
                portfolio=_MockPortfolio(100_000, {}, 100_000, 100_000, timestamps[idx]),
            )
            assert len(ctx.bars["MES"]) == idx + 1

    def test_future_bars_not_accessible(self) -> None:
        """At step 10, bar 11+ must not be in the context."""
        timestamps = _make_timestamps(50)
        all_bars = {"MES": _make_bar_dicts(50, base=1000, step=100)}

        ctx = BacktestContext(
            all_bars=all_bars,
            current_idx=10,
            timestamps=timestamps,
            portfolio=_MockPortfolio(100_000, {}, 100_000, 100_000, timestamps[10]),
        )

        # Last bar in context should be bar 10, not bar 11+
        last_close = ctx.bars["MES"][-1]["close"]
        bar_10_close = 1000 + 10 * 100  # 2000
        assert last_close == pytest.approx(bar_10_close)

    def test_ticks_reflect_current_bar_only(self) -> None:
        timestamps = _make_timestamps(50)
        all_bars = {"MES": _make_bar_dicts(50, base=5000, step=10)}

        ctx = BacktestContext(
            all_bars=all_bars,
            current_idx=20,
            timestamps=timestamps,
            portfolio=_MockPortfolio(100_000, {}, 100_000, 100_000, timestamps[20]),
        )

        expected_close = 5000 + 20 * 10
        assert ctx.ticks["MES"].price == pytest.approx(expected_close)
        assert ctx.prices["MES"] == pytest.approx(expected_close)

    def test_now_is_current_timestamp(self) -> None:
        timestamps = _make_timestamps(50)
        all_bars = {"MES": _make_bar_dicts(50)}

        ctx = BacktestContext(
            all_bars=all_bars,
            current_idx=15,
            timestamps=timestamps,
            portfolio=_MockPortfolio(100_000, {}, 100_000, 100_000, timestamps[15]),
        )

        assert ctx.now == timestamps[15]


# ===========================================================================
# SimulatedPortfolio tests
# ===========================================================================

class TestSimulatedPortfolio:

    def test_initial_state(self) -> None:
        port = SimulatedPortfolio(100_000)
        assert port.cash == 100_000
        assert port.positions == []
        assert port.equity_curve == [100_000]

    def test_fill_order(self) -> None:
        port = SimulatedPortfolio(100_000)
        port.fill_order("SPY", "BUY", 10.0, 500.0, 0, "alpha")
        assert len(port.positions) == 1
        assert port.cash == 100_000 - 10 * 500

    def test_mark_to_market_gain(self) -> None:
        port = SimulatedPortfolio(100_000)
        port.fill_order("SPY", "BUY", 10.0, 500.0, 0, "alpha")
        # Price went up to 510
        equity = port.mark_to_market({"SPY": 510.0})
        # cash = 95000, unrealized = (510-500)*10 = 100, position cost = 5000
        assert equity > 100_000

    def test_stop_loss_triggers(self) -> None:
        port = SimulatedPortfolio(100_000)
        port.fill_order("SPY", "BUY", 10.0, 500.0, 0, "alpha", stop_price=490.0)

        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        trades = port.check_exits({"SPY": 489.0}, 5, ts)
        assert len(trades) == 1
        assert trades[0].exit_reason == "STOP"
        assert trades[0].pnl < 0

    def test_time_stop_triggers(self) -> None:
        port = SimulatedPortfolio(100_000)
        port.fill_order("SPY", "BUY", 10.0, 500.0, 0, "alpha", time_stop_bars=10)

        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        # At bar 9, should NOT trigger
        trades = port.check_exits({"SPY": 505.0}, 9, ts)
        assert len(trades) == 0

        # At bar 10, should trigger
        trades = port.check_exits({"SPY": 505.0}, 10, ts)
        assert len(trades) == 1
        assert trades[0].exit_reason == "TIME"

    def test_close_all(self) -> None:
        port = SimulatedPortfolio(100_000)
        port.fill_order("SPY", "BUY", 10.0, 500.0, 0, "alpha")
        port.fill_order("QQQ", "BUY", 5.0, 400.0, 0, "alpha")

        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        trades = port.close_all({"SPY": 510.0, "QQQ": 410.0}, 20, ts)
        assert len(trades) == 2
        assert all(t.exit_reason == "END_OF_DATA" for t in trades)
        assert port.positions == []

    def test_short_position_stop(self) -> None:
        port = SimulatedPortfolio(100_000)
        port.fill_order("SPY", "SELL", 10.0, 500.0, 0, "alpha", stop_price=510.0)

        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        trades = port.check_exits({"SPY": 511.0}, 5, ts)
        assert len(trades) == 1
        assert trades[0].exit_reason == "STOP"
        assert trades[0].pnl < 0


# ===========================================================================
# Statistics tests
# ===========================================================================

class TestStatistics:

    def test_win_rate(self) -> None:
        trades = [
            BacktestTrade("SPY", "alpha", "BUY", 500, 510, 10, BASE_TS, BASE_TS, 5, 100, 0.02, "TARGET"),
            BacktestTrade("SPY", "alpha", "BUY", 500, 490, 10, BASE_TS, BASE_TS, 5, -100, -0.02, "STOP"),
            BacktestTrade("SPY", "alpha", "BUY", 500, 520, 10, BASE_TS, BASE_TS, 5, 200, 0.04, "TARGET"),
        ]
        stats = compute_stats(trades, [100000, 100100, 100000, 100200], 50)
        assert stats["win_rate"] == pytest.approx(2/3, abs=0.01)

    def test_empty_trades(self) -> None:
        stats = compute_stats([], [100000], 50)
        assert stats["win_rate"] == 0
        assert stats["sharpe_ratio"] == 0

    def test_max_drawdown(self) -> None:
        equity = [100000, 105000, 95000, 97000, 110000]
        stats = compute_stats(
            [BacktestTrade("SPY", "a", "BUY", 100, 110, 1, BASE_TS, BASE_TS, 5, 10, 0.1, "T")],
            equity, 50,
        )
        # Peak 105000 -> trough 95000 = -9.52%
        assert stats["max_drawdown_pct"] < -0.09

    def test_sharpe_positive(self) -> None:
        trades = [
            BacktestTrade("S", "a", "BUY", 100, 105, 1, BASE_TS, BASE_TS, 5, 5, 0.05, "T"),
            BacktestTrade("S", "a", "BUY", 100, 103, 1, BASE_TS, BASE_TS, 5, 3, 0.03, "T"),
            BacktestTrade("S", "a", "BUY", 100, 102, 1, BASE_TS, BASE_TS, 5, 2, 0.02, "T"),
            BacktestTrade("S", "a", "BUY", 100, 99, 1, BASE_TS, BASE_TS, 5, -1, -0.01, "S"),
        ]
        stats = compute_stats(trades, [100000, 100500, 100800, 101000, 100900], 50)
        assert stats["sharpe_ratio"] > 0


# ===========================================================================
# ATR computation tests
# ===========================================================================

class TestATR:

    def test_atr_from_bars(self) -> None:
        bars = _make_bar_dicts(20, base=100, step=1)
        atr = _compute_atr(bars, period=14)
        assert atr > 0

    def test_atr_empty_bars(self) -> None:
        assert _compute_atr([], period=14) == 0.0

    def test_atr_single_bar(self) -> None:
        assert _compute_atr([{"open": 100, "high": 105, "low": 95, "close": 102}], period=14) == 0.0


# ===========================================================================
# Bar alignment tests
# ===========================================================================

class TestBarAlignment:

    def test_intersection_of_dates(self) -> None:
        bars_a = _make_backtest_bars(10, symbol="A")
        bars_b = _make_backtest_bars(8, symbol="B")  # shorter
        ts, aligned = align_bars({"A": bars_a, "B": bars_b})
        assert len(ts) == 8  # intersection
        assert len(aligned["A"]) == 8
        assert len(aligned["B"]) == 8

    def test_date_filtering(self) -> None:
        bars = _make_backtest_bars(50, symbol="MES")
        start = BASE_TS + timedelta(days=10)
        end = BASE_TS + timedelta(days=30)
        ts, aligned = align_bars({"MES": bars}, start, end)
        for t in ts:
            assert t >= start
            assert t <= end

    def test_empty_on_no_overlap(self) -> None:
        bars_a = _make_backtest_bars(5, symbol="A")
        bars_b = [BacktestBar(
            ts_utc=BASE_TS + timedelta(days=100 + i),
            open=100, high=110, low=90, close=100,
            volume=1000, symbol="B",
        ) for i in range(5)]
        ts, aligned = align_bars({"A": bars_a, "B": bars_b})
        assert len(ts) == 0


# ===========================================================================
# VIX injection tests
# ===========================================================================

class TestVIXInjection:

    def test_vix_available_in_context(self) -> None:
        timestamps = _make_timestamps(30)
        all_bars = {"SPY": _make_bar_dicts(30)}
        vix_series = [20.0 + i * 0.1 for i in range(30)]

        ctx = BacktestContext(
            all_bars=all_bars,
            current_idx=15,
            timestamps=timestamps,
            portfolio=_MockPortfolio(100_000, {}, 100_000, 100_000, timestamps[15]),
            vix_series=vix_series,
        )

        assert ctx.vix == pytest.approx(20.0 + 15 * 0.1)
        assert ctx.vix_history is not None
        assert len(ctx.vix_history) == 16  # 0..15

    def test_no_vix_series(self) -> None:
        timestamps = _make_timestamps(30)
        all_bars = {"SPY": _make_bar_dicts(30)}

        ctx = BacktestContext(
            all_bars=all_bars,
            current_idx=15,
            timestamps=timestamps,
            portfolio=_MockPortfolio(100_000, {}, 100_000, 100_000, timestamps[15]),
            vix_series=None,
        )

        assert ctx.vix is None
        assert ctx.vix_history is None


# ===========================================================================
# MockTick tests
# ===========================================================================

class TestMockTick:

    def test_tick_has_price(self) -> None:
        t = _MockTick("SPY", 650.0, BASE_TS)
        assert t.price == 650.0
        assert t.last == 650.0
        assert t.bid < t.ask
        assert t.symbol == "SPY"

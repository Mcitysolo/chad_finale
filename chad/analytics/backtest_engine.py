#!/usr/bin/env python3
"""
chad/analytics/backtest_engine.py

Production Backtesting Engine for CHAD.

Runs walk-forward backtests using the EXACT SAME strategy handler functions
that run in production. No reimplementations. No lookahead bias.

Architecture
------------
1. Load historical bars from data/bars/1d/<SYMBOL>.json
2. Align bars by timestamp (intersection of trading days)
3. Walk forward bar-by-bar:
   a. Build BacktestContext (duck-type compatible with MarketContext)
   b. Call strategy_handler(ctx) -> List[TradeSignal]
   c. Process signals -> fill simulated positions
   d. Check exits (stop, target, time)
   e. Record equity curve and trades
4. Compute statistics (Sharpe, win rate, drawdown, etc.)

Key invariant: bars at step i contain ONLY bars[0..i]. Never future data.

Usage:
    python3 -m chad.analytics.backtest_engine \\
        --strategy alpha_futures \\
        --symbols MES MNQ MCL MGC \\
        --start 2025-06-01 --end 2026-04-01 \\
        --equity 100000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BARS_DIR = REPO_ROOT / "data" / "bars" / "1d"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BacktestBar:
    ts_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str


@dataclass
class BacktestPosition:
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    entry_price: float
    entry_bar_idx: int
    strategy: str
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    time_stop_bars: Optional[int] = None


@dataclass
class BacktestTrade:
    symbol: str
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_ts: datetime
    exit_ts: datetime
    bars_held: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # SIGNAL, STOP, TARGET, TIME, END_OF_DATA


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    total_bars: int
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    trades_per_year: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _parse_bar_ts(ts_str: str) -> datetime:
    s = ts_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Mock objects for BacktestContext
# ---------------------------------------------------------------------------

class _MockTick:
    """Synthetic tick from bar close price."""
    __slots__ = ("symbol", "price", "last", "bid", "ask", "size", "exchange", "timestamp", "source")

    def __init__(self, symbol: str, close: float, ts: datetime) -> None:
        self.symbol = symbol
        self.price = close
        self.last = close
        self.bid = close * 0.9999
        self.ask = close * 1.0001
        self.size = 0.0
        self.exchange = None
        self.timestamp = ts
        self.source = "backtest"


class _MockPortfolio:
    """Simulated portfolio snapshot for backtest context."""
    __slots__ = ("timestamp", "cash", "positions", "extra", "total_equity", "equity", "net_liq")

    def __init__(self, cash: float, positions: Dict[str, Any], equity: float, equity_peak: float, ts: datetime) -> None:
        self.timestamp = ts
        self.cash = cash
        self.positions = positions
        self.total_equity = equity
        self.equity = equity
        self.net_liq = equity
        self.extra = {"equity": equity, "equity_peak": equity_peak}


# ---------------------------------------------------------------------------
# BacktestContext — duck-type compatible with MarketContext
# ---------------------------------------------------------------------------

class BacktestContext:
    """
    Walk-forward backtest context. Duck-type compatible with MarketContext
    and all strategy handler expectations.

    Key invariant: self.bars[symbol] contains ONLY bars up to current_idx.
    No future data is ever exposed.
    """

    def __init__(
        self,
        *,
        all_bars: Dict[str, List[Dict[str, Any]]],
        current_idx: int,
        timestamps: List[datetime],
        portfolio: _MockPortfolio,
        vix_series: Optional[List[float]] = None,
    ) -> None:
        self._all_bars = all_bars
        self._idx = current_idx
        self._timestamps = timestamps

        # Core fields strategies expect
        self.now = timestamps[current_idx]
        self.legend = None
        self.portfolio = portfolio

        # Build bars window: only bars[0..current_idx+1] (no lookahead)
        self.bars: Dict[str, list] = {}
        for sym, bars in all_bars.items():
            self.bars[sym] = bars[:current_idx + 1]

        # Build ticks and prices from latest bar close
        self.ticks: Dict[str, _MockTick] = {}
        self.prices: Dict[str, float] = {}
        for sym, bars in self.bars.items():
            if bars:
                close = _safe_float(bars[-1].get("close"), 0.0)
                if close > 0:
                    self.ticks[sym] = _MockTick(sym, close, self.now)
                    self.prices[sym] = close

        # VIX injection
        self.vix: Optional[float] = None
        self.vix_history: Optional[List[float]] = None
        self.vol_index: Optional[float] = None
        self.volatility_index: Optional[float] = None

        if vix_series is not None and current_idx < len(vix_series):
            self.vix = vix_series[current_idx]
            self.vol_index = self.vix
            self.vix_history = vix_series[:current_idx + 1]

        # Additional fields some strategies check
        self.spread_bps = None
        self.dollar_volume = None
        self.volume_usd = None
        self.liquidity_usd = None
        self.volatility = None
        self.strategy_signals = None
        self.market_data: Dict[str, Any] = {}
        if self.vix is not None:
            self.market_data["VIX"] = self.vix


# ---------------------------------------------------------------------------
# SimulatedPortfolio
# ---------------------------------------------------------------------------

class SimulatedPortfolio:
    """Tracks simulated cash, positions, and equity through the backtest."""

    def __init__(self, initial_equity: float) -> None:
        self.initial_equity = initial_equity
        self.cash = initial_equity
        self.positions: List[BacktestPosition] = []
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_equity]
        self.equity_peak = initial_equity

    @property
    def current_equity(self) -> float:
        return self.cash + sum(
            p.quantity * p.entry_price for p in self.positions
        )

    def fill_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        bar_idx: int,
        strategy: str,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        time_stop_bars: Optional[int] = None,
    ) -> None:
        """Open a new simulated position."""
        cost = quantity * fill_price
        self.cash -= cost
        self.positions.append(BacktestPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=fill_price,
            entry_bar_idx=bar_idx,
            strategy=strategy,
            stop_price=stop_price,
            target_price=target_price,
            time_stop_bars=time_stop_bars,
        ))

    def check_exits(
        self,
        current_prices: Dict[str, float],
        bar_idx: int,
        bar_ts: datetime,
    ) -> List[BacktestTrade]:
        """Check all open positions for exit conditions."""
        trades: List[BacktestTrade] = []
        remaining: List[BacktestPosition] = []

        for pos in self.positions:
            price = current_prices.get(pos.symbol, 0.0)
            if price <= 0:
                remaining.append(pos)
                continue

            bars_held = bar_idx - pos.entry_bar_idx
            exit_reason: Optional[str] = None

            if pos.side == "BUY":
                pnl_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
                # Stop loss
                if pos.stop_price is not None and price <= pos.stop_price:
                    exit_reason = "STOP"
                # Target
                elif pos.target_price is not None and price >= pos.target_price:
                    exit_reason = "TARGET"
            else:  # SELL (short)
                pnl_pct = (pos.entry_price - price) / pos.entry_price if pos.entry_price > 0 else 0.0
                if pos.stop_price is not None and price >= pos.stop_price:
                    exit_reason = "STOP"
                elif pos.target_price is not None and price <= pos.target_price:
                    exit_reason = "TARGET"

            # Time stop
            if exit_reason is None and pos.time_stop_bars is not None and bars_held >= pos.time_stop_bars:
                exit_reason = "TIME"

            if exit_reason is not None:
                pnl = (price - pos.entry_price) * pos.quantity if pos.side == "BUY" else (pos.entry_price - price) * pos.quantity
                trade = BacktestTrade(
                    symbol=pos.symbol,
                    strategy=pos.strategy,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    quantity=pos.quantity,
                    entry_ts=datetime.min.replace(tzinfo=timezone.utc),  # populated later
                    exit_ts=bar_ts,
                    bars_held=bars_held,
                    pnl=round(pnl, 4),
                    pnl_pct=round(pnl_pct, 6),
                    exit_reason=exit_reason,
                )
                trades.append(trade)
                self.cash += pos.quantity * price  # return capital
            else:
                remaining.append(pos)

        self.positions = remaining
        self.closed_trades.extend(trades)
        return trades

    def close_all(
        self,
        current_prices: Dict[str, float],
        bar_idx: int,
        bar_ts: datetime,
    ) -> List[BacktestTrade]:
        """Force-close all positions at end of data."""
        trades: List[BacktestTrade] = []
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            bars_held = bar_idx - pos.entry_bar_idx
            if pos.side == "BUY":
                pnl = (price - pos.entry_price) * pos.quantity
                pnl_pct = (price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            else:
                pnl = (pos.entry_price - price) * pos.quantity
                pnl_pct = (pos.entry_price - price) / pos.entry_price if pos.entry_price > 0 else 0.0
            trades.append(BacktestTrade(
                symbol=pos.symbol, strategy=pos.strategy, side=pos.side,
                entry_price=pos.entry_price, exit_price=price, quantity=pos.quantity,
                entry_ts=datetime.min.replace(tzinfo=timezone.utc), exit_ts=bar_ts,
                bars_held=bars_held, pnl=round(pnl, 4), pnl_pct=round(pnl_pct, 6),
                exit_reason="END_OF_DATA",
            ))
            self.cash += pos.quantity * price
        self.closed_trades.extend(trades)
        self.positions = []
        return trades

    def mark_to_market(self, current_prices: Dict[str, float]) -> float:
        """Compute current equity and update curve."""
        unrealized = 0.0
        for pos in self.positions:
            price = current_prices.get(pos.symbol, pos.entry_price)
            if pos.side == "BUY":
                unrealized += (price - pos.entry_price) * pos.quantity
            else:
                unrealized += (pos.entry_price - price) * pos.quantity
        equity = self.cash + sum(p.quantity * p.entry_price for p in self.positions) + unrealized
        self.equity_curve.append(equity)
        if equity > self.equity_peak:
            self.equity_peak = equity
        return equity

    def to_mock_portfolio(self, ts: datetime) -> _MockPortfolio:
        """Build a mock portfolio for BacktestContext."""
        equity = self.equity_curve[-1] if self.equity_curve else self.initial_equity
        positions: Dict[str, Any] = {}
        for pos in self.positions:
            positions[pos.symbol] = type("Pos", (), {
                "symbol": pos.symbol, "quantity": pos.quantity,
                "avg_price": pos.entry_price, "asset_class": "equity",
            })()
        return _MockPortfolio(
            cash=self.cash, positions=positions,
            equity=equity, equity_peak=self.equity_peak, ts=ts,
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(
    trades: List[BacktestTrade],
    equity_curve: List[float],
    total_bars: int,
) -> Dict[str, float]:
    """Compute backtest statistics from trade list and equity curve."""
    stats: Dict[str, float] = {}

    if not trades:
        return {"win_rate": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
                "profit_factor": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0,
                "total_return_pct": 0, "trades_per_year": 0}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    stats["win_rate"] = round(len(wins) / len(trades), 4) if trades else 0.0
    stats["avg_win_pct"] = round(sum(t.pnl_pct for t in wins) / len(wins), 6) if wins else 0.0
    stats["avg_loss_pct"] = round(sum(t.pnl_pct for t in losses) / len(losses), 6) if losses else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    stats["profit_factor"] = round(gross_profit / gross_loss, 4) if gross_loss > 0 else 999.0

    # Sharpe from trade returns
    returns = [t.pnl_pct for t in trades]
    if len(returns) >= 2:
        mean_r = sum(returns) / len(returns)
        var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
        avg_held = sum(t.bars_held for t in trades) / len(trades)
        trades_per_year = 252.0 / max(avg_held, 1.0)
        stats["sharpe_ratio"] = round((mean_r / std_r) * math.sqrt(trades_per_year), 4)
        stats["trades_per_year"] = round(trades_per_year, 2)
    else:
        stats["sharpe_ratio"] = 0.0
        stats["trades_per_year"] = 0.0

    # Max drawdown from equity curve
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        stats["max_drawdown_pct"] = round(max_dd, 6)
        stats["total_return_pct"] = round((equity_curve[-1] / equity_curve[0]) - 1.0, 6) if equity_curve[0] > 0 else 0.0
    else:
        stats["max_drawdown_pct"] = 0.0
        stats["total_return_pct"] = 0.0

    return stats


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

def load_bars(symbol: str, bars_dir: Optional[Path] = None) -> List[BacktestBar]:
    """Load daily bars from JSON file."""
    d = bars_dir or BARS_DIR
    path = d / f"{symbol.strip().upper()}.json"
    if not path.is_file():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    bars_raw = raw.get("bars", raw) if isinstance(raw, dict) else raw
    if not isinstance(bars_raw, list):
        return []

    bars: List[BacktestBar] = []
    for b in bars_raw:
        if not isinstance(b, dict):
            continue
        try:
            o = float(b["open"])
            h = float(b["high"])
            l = float(b["low"])
            c = float(b["close"])
            v = float(b.get("volume", 0))
            # Handle both "ts_utc" and "time" timestamp keys
            ts_raw = b.get("ts_utc") or b.get("time") or b.get("timestamp") or b.get("date")
            if ts_raw is None:
                continue
            ts = _parse_bar_ts(str(ts_raw))
        except (KeyError, ValueError, TypeError):
            continue
        if min(o, h, l, c) <= 0 or h < l:
            continue
        bars.append(BacktestBar(ts_utc=ts, open=o, high=h, low=l, close=c, volume=max(v, 0), symbol=symbol.upper()))

    bars.sort(key=lambda b: b.ts_utc)
    return bars


def align_bars(
    bars_dict: Dict[str, List[BacktestBar]],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[List[datetime], Dict[str, List[Dict[str, Any]]]]:
    """
    Align multiple symbol bar series by timestamp intersection.

    Returns (aligned_timestamps, bars_as_dicts_per_symbol).
    """
    if not bars_dict:
        return [], {}

    # Get timestamp sets per symbol
    ts_sets = []
    for sym, bars in bars_dict.items():
        ts_set = set()
        for b in bars:
            if start_date and b.ts_utc < start_date:
                continue
            if end_date and b.ts_utc > end_date:
                continue
            ts_set.add(b.ts_utc)
        ts_sets.append(ts_set)

    # Intersection of all timestamp sets
    if not ts_sets:
        return [], {}
    common_ts = sorted(ts_sets[0].intersection(*ts_sets[1:]) if len(ts_sets) > 1 else ts_sets[0])

    if not common_ts:
        return [], {}

    # Build aligned dict-format bars
    aligned: Dict[str, List[Dict[str, Any]]] = {}
    for sym, bars in bars_dict.items():
        ts_to_bar = {b.ts_utc: b for b in bars}
        aligned[sym] = []
        for ts in common_ts:
            b = ts_to_bar.get(ts)
            if b:
                aligned[sym].append({
                    "ts_utc": b.ts_utc.isoformat().replace("+00:00", "Z"),
                    "open": b.open, "high": b.high, "low": b.low,
                    "close": b.close, "volume": b.volume,
                })

    return common_ts, aligned


# ---------------------------------------------------------------------------
# ATR calculation for stop placement
# ---------------------------------------------------------------------------

def _compute_atr(bars: List[Dict[str, Any]], period: int = 14) -> float:
    """Compute ATR from the last `period` bars."""
    if len(bars) < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, len(bars)):
        h = _safe_float(bars[i].get("high"), 0)
        l = _safe_float(bars[i].get("low"), 0)
        prev_c = _safe_float(bars[i - 1].get("close"), 0)
        if h > 0 and l > 0 and prev_c > 0:
            trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    if not trs:
        return 0.0
    window = trs[-period:] if len(trs) >= period else trs
    return sum(window) / len(window)


# ---------------------------------------------------------------------------
# Strategy handler lookup
# ---------------------------------------------------------------------------

def _get_handler(strategy_name: str) -> Callable:
    """Look up strategy handler from the registry."""
    from chad.strategies import iter_strategy_registrations
    from chad.types import StrategyName

    for reg in iter_strategy_registrations():
        if reg.name.value == strategy_name:
            return reg.handler

    raise ValueError(f"Strategy '{strategy_name}' not found in registry")


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward backtesting engine that uses live strategy handlers.

    No reimplementations. No lookahead bias.
    """

    def __init__(self, bars_dir: Optional[Path] = None) -> None:
        self.bars_dir = bars_dir or BARS_DIR

    def run(
        self,
        strategy_name: str,
        universe: Sequence[str],
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_equity: float = 100_000.0,
        atr_stop_mult: float = 2.5,
        time_stop_bars: int = 15,
    ) -> Dict[str, BacktestResult]:
        """
        Run walk-forward backtest for a single strategy across symbols.

        Returns per-symbol BacktestResult dict.
        """
        handler = _get_handler(strategy_name)

        # Load bars
        raw_bars: Dict[str, List[BacktestBar]] = {}
        for sym in universe:
            bars = load_bars(sym.strip().upper(), self.bars_dir)
            if bars:
                raw_bars[sym.strip().upper()] = bars

        if not raw_bars:
            return {}

        start_dt = _parse_date(start_date) if start_date else None
        end_dt = _parse_date(end_date) if end_date else None

        timestamps, aligned_bars = align_bars(raw_bars, start_dt, end_dt)
        if not timestamps:
            return {}

        # Extract VIX series if available
        vix_series: Optional[List[float]] = None
        vix_bars = load_bars("VIX", self.bars_dir)
        if not vix_bars:
            # Try constructing from aligned bars if VIX is in universe
            pass
        else:
            vix_ts_map = {b.ts_utc: b.close for b in vix_bars}
            vix_series = [vix_ts_map.get(ts, 0.0) for ts in timestamps]
            # Replace 0.0 with None-like behavior: keep last known
            last_vix = 20.0
            for i in range(len(vix_series)):
                if vix_series[i] > 0:
                    last_vix = vix_series[i]
                else:
                    vix_series[i] = last_vix

        # Initialize portfolio
        portfolio = SimulatedPortfolio(initial_equity)

        # Warmup: skip first 40 bars for indicator warmup
        warmup = min(40, len(timestamps) // 4)

        # Walk forward
        for i in range(warmup, len(timestamps)):
            ts = timestamps[i]

            # Current prices
            current_prices: Dict[str, float] = {}
            for sym, bars in aligned_bars.items():
                if i < len(bars):
                    current_prices[sym] = _safe_float(bars[i].get("close"), 0.0)

            # Check exits first
            portfolio.check_exits(current_prices, i, ts)

            # Build context
            equity = portfolio.mark_to_market(current_prices)
            mock_port = portfolio.to_mock_portfolio(ts)

            ctx = BacktestContext(
                all_bars=aligned_bars,
                current_idx=i,
                timestamps=timestamps,
                portfolio=mock_port,
                vix_series=vix_series,
            )

            # Call strategy handler
            try:
                signals = handler(ctx)
            except Exception:
                signals = []

            if not isinstance(signals, (list, tuple)):
                signals = []

            # Process signals
            for sig in signals:
                sym = str(getattr(sig, "symbol", "")).strip().upper()
                side = str(getattr(sig, "side", "")).upper()
                size = _safe_float(getattr(sig, "size", 0), 0.0)
                confidence = _safe_float(getattr(sig, "confidence", 0), 0.0)

                if not sym or side not in ("BUY", "SELL") or size <= 0:
                    continue

                price = current_prices.get(sym, 0.0)
                if price <= 0:
                    continue

                # Skip if already have position in this symbol
                if any(p.symbol == sym for p in portfolio.positions):
                    continue

                # Compute ATR-based stop
                sym_bars = aligned_bars.get(sym, [])
                bars_window = sym_bars[:i + 1]
                atr = _compute_atr(bars_window)
                if atr <= 0:
                    atr = price * 0.02  # fallback 2%

                if side == "BUY":
                    stop = price - atr_stop_mult * atr
                    target = None  # Let strategy exits handle
                else:
                    stop = price + atr_stop_mult * atr
                    target = None

                # Extract meta stops if available
                meta = getattr(sig, "meta", {}) or {}
                if "stop_price" in meta:
                    stop = _safe_float(meta["stop_price"], stop)
                if "target_price" in meta:
                    target = _safe_float(meta["target_price"], None)

                portfolio.fill_order(
                    symbol=sym, side=side, quantity=size,
                    fill_price=price, bar_idx=i,
                    strategy=strategy_name,
                    stop_price=stop, target_price=target,
                    time_stop_bars=time_stop_bars,
                )

        # Close remaining positions
        final_prices = {}
        for sym, bars in aligned_bars.items():
            if bars:
                final_prices[sym] = _safe_float(bars[-1].get("close"), 0.0)
        portfolio.close_all(final_prices, len(timestamps) - 1, timestamps[-1])

        # Build per-symbol results
        results: Dict[str, BacktestResult] = {}
        for sym in universe:
            sym = sym.strip().upper()
            sym_trades = [t for t in portfolio.closed_trades if t.symbol == sym]

            stats = compute_stats(sym_trades, portfolio.equity_curve, len(timestamps))

            results[sym] = BacktestResult(
                strategy_name=strategy_name,
                symbol=sym,
                start_date=timestamps[warmup] if warmup < len(timestamps) else timestamps[0],
                end_date=timestamps[-1],
                total_bars=len(timestamps) - warmup,
                trades=sym_trades,
                equity_curve=portfolio.equity_curve,
                **stats,
            )

        # Add aggregate result
        all_trades = portfolio.closed_trades
        agg_stats = compute_stats(all_trades, portfolio.equity_curve, len(timestamps))
        results["_AGGREGATE"] = BacktestResult(
            strategy_name=strategy_name,
            symbol="_AGGREGATE",
            start_date=timestamps[warmup] if warmup < len(timestamps) else timestamps[0],
            end_date=timestamps[-1],
            total_bars=len(timestamps) - warmup,
            trades=all_trades,
            equity_curve=portfolio.equity_curve,
            **agg_stats,
        )

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="CHAD Production Backtesting Engine")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g., alpha_futures)")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols to backtest")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--equity", type=float, default=100_000.0, help="Initial equity")
    parser.add_argument("--output", default=None, help="Output directory for reports")
    args = parser.parse_args()

    engine = BacktestEngine()
    results = engine.run(
        strategy_name=args.strategy,
        universe=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_equity=args.equity,
    )

    if not results:
        print("No results — check symbol availability and date range.")
        return 1

    # Import and use report generator
    from chad.analytics.backtest_report import print_summary, write_json_report

    print_summary(results)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.strategy}_backtest.json"
        write_json_report(results, out_path)
        print(f"\nReport written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
chad/analytics/reversion_backtest.py

Standalone mean-reversion backtest for GAMMA_REVERSION signal validation.

Reads daily bars from data/bars/1d/<SYMBOL>.json, walks forward bar-by-bar
computing RSI / Bollinger / Z-score / ROC indicators, generates entry signals
on multi-indicator confluence, tracks hypothetical positions, and reports
per-symbol and aggregate statistics.

No external dependencies beyond stdlib. No broker calls. Deterministic.

Usage:
    python3 -m chad.analytics.reversion_backtest --symbols SPY QQQ IWM GLD TLT
    python3 -m chad.analytics.reversion_backtest --symbols SPY QQQ IWM GLD TLT \\
        --output reports/backtest/gamma_reversion_backtest.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
BARS_DIR = REPO_ROOT / "data" / "bars" / "1d"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bars(symbol: str) -> List[Dict[str, Any]]:
    """
    Load daily OHLCV bars for a symbol from data/bars/1d/<SYMBOL>.json.

    Returns a list of bar dicts with keys: ts_utc, open, high, low, close, volume.
    Filters out invalid bars (missing fields, non-positive prices, high < low).
    """
    path = BARS_DIR / f"{symbol.strip().upper()}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No bar data for {symbol}: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    # Handle both {bars: [...]} and [{...}, ...] formats
    if isinstance(raw, dict):
        bars_raw = raw.get("bars", [])
    elif isinstance(raw, list):
        bars_raw = raw
    else:
        bars_raw = []

    bars: List[Dict[str, Any]] = []
    for b in bars_raw:
        if not isinstance(b, dict):
            continue
        try:
            o = float(b.get("open", 0))
            h = float(b.get("high", 0))
            l = float(b.get("low", 0))
            c = float(b.get("close", 0))
            v = float(b.get("volume", 0))
        except (TypeError, ValueError):
            continue
        if min(o, h, l, c) <= 0 or h < l or v <= 0:
            continue
        bars.append({
            "ts_utc": str(b.get("ts_utc", "")),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": max(v, 0.0),
        })
    return bars


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _rsi(closes: Sequence[float], period: int = 14) -> List[float]:
    """
    Compute RSI (Relative Strength Index) using Wilder's smoothing.

    Returns a list of RSI values aligned to the input, with the first
    `period` values set to 50.0 (neutral) as warmup.
    """
    n = len(closes)
    out = [50.0] * n
    if n < period + 1:
        return out

    gains = []
    losses = []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, n):
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

        if i < n - 1:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return out


def _sma(values: Sequence[float], period: int) -> List[float]:
    """Simple moving average. First `period-1` values use expanding window."""
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        window = values[start:i + 1]
        out.append(sum(window) / len(window))
    return out


def _std(values: Sequence[float], period: int) -> List[float]:
    """Rolling standard deviation. First `period-1` values use expanding window."""
    sma_vals = _sma(values, period)
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        window = values[start:i + 1]
        mean = sma_vals[i]
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        out.append(math.sqrt(variance))
    return out


def _bollinger(
    closes: Sequence[float],
    period: int = 20,
    width: float = 2.0,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute Bollinger Bands.

    Returns (upper, middle, lower) as lists aligned to input.
    """
    middle = _sma(closes, period)
    sd = _std(closes, period)
    upper = [m + width * s for m, s in zip(middle, sd)]
    lower = [m - width * s for m, s in zip(middle, sd)]
    return upper, middle, lower


def _zscore(closes: Sequence[float], period: int = 20) -> List[float]:
    """
    Compute Z-score of close price relative to SMA(period).

    Z = (close - SMA) / StdDev. Returns 0.0 when StdDev is near zero.
    """
    sma_vals = _sma(closes, period)
    sd = _std(closes, period)
    out: List[float] = []
    for i in range(len(closes)):
        if sd[i] < 1e-10:
            out.append(0.0)
        else:
            out.append((closes[i] - sma_vals[i]) / sd[i])
    return out


def _roc(closes: Sequence[float], period: int = 5) -> List[float]:
    """
    Rate of change: (close - close[n-period]) / close[n-period].

    Returns 0.0 for the first `period` bars.
    """
    out: List[float] = []
    for i in range(len(closes)):
        if i < period or closes[i - period] <= 0:
            out.append(0.0)
        else:
            out.append((closes[i] - closes[i - period]) / closes[i - period])
    return out


def _atr(bars: Sequence[Dict[str, Any]], period: int = 14) -> List[float]:
    """Compute ATR. Returns list aligned to bars, with warmup using expanding window."""
    n = len(bars)
    trs: List[float] = []
    for i in range(n):
        h = bars[i]["high"]
        l = bars[i]["low"]
        if i == 0:
            trs.append(h - l)
        else:
            prev_c = bars[i - 1]["close"]
            trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))

    out: List[float] = []
    for i in range(n):
        start = max(0, i - period + 1)
        window = trs[start:i + 1]
        out.append(sum(window) / len(window))
    return out


# ---------------------------------------------------------------------------
# Trade and result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    entry_bar: int
    exit_bar: int
    side: str  # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    pnl_pct: float
    bars_held: int
    exit_reason: str = ""


@dataclass
class BacktestResult:
    symbol: str
    trades: List[BacktestTrade] = field(default_factory=list)
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    total_bars: int = 0


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReversionTuning:
    rsi_period: int = 14
    rsi_overbought: float = 72.0
    rsi_oversold: float = 28.0
    bb_period: int = 20
    bb_width: float = 2.0
    zscore_period: int = 20
    zscore_threshold: float = 1.8
    roc_period: int = 5
    atr_period: int = 14
    atr_stop_mult: float = 2.5
    time_stop_bars: int = 15
    min_warmup: int = 40


DEFAULT_TUNING = ReversionTuning()


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_reversion_backtest(
    symbol: str,
    bars: List[Dict[str, Any]],
    tuning: ReversionTuning = DEFAULT_TUNING,
) -> BacktestResult:
    """
    Walk-forward backtest of a mean-reversion strategy on daily bars.

    Entry logic (requires 2+ of 3 confluence signals):
      SELL: RSI > 72 AND (price > bb_upper OR zscore > 1.8) AND roc > 0
      BUY:  RSI < 28 AND (price < bb_lower OR zscore < -1.8) AND roc < 0

    Exit logic:
      Target: price crosses SMA-20 (mean reversion achieved)
      Stop:   2.5 * ATR from entry
      Time:   15 bars max hold
    """
    result = BacktestResult(symbol=symbol, total_bars=len(bars))

    if len(bars) < tuning.min_warmup:
        return result

    closes = [b["close"] for b in bars]

    # Pre-compute all indicators
    rsi_vals = _rsi(closes, tuning.rsi_period)
    bb_upper, bb_middle, bb_lower = _bollinger(closes, tuning.bb_period, tuning.bb_width)
    zs_vals = _zscore(closes, tuning.zscore_period)
    roc_vals = _roc(closes, tuning.roc_period)
    atr_vals = _atr(bars, tuning.atr_period)
    sma20 = _sma(closes, tuning.bb_period)

    trades: List[BacktestTrade] = []
    equity_curve: List[float] = [1.0]

    # Walk-forward simulation
    in_trade = False
    trade_side = ""
    entry_price = 0.0
    entry_bar = 0
    stop_price = 0.0

    for i in range(tuning.min_warmup, len(bars)):
        price = closes[i]

        if in_trade:
            # Check exits
            bars_held = i - entry_bar
            exit_reason = ""

            if trade_side == "BUY":
                pnl_pct = (price - entry_price) / entry_price
                # Target: price crosses above SMA-20
                if price >= sma20[i]:
                    exit_reason = "target_mean"
                # Stop loss
                elif price <= stop_price:
                    exit_reason = "stop_loss"
                # Time stop
                elif bars_held >= tuning.time_stop_bars:
                    exit_reason = "time_stop"

            else:  # SELL (short)
                pnl_pct = (entry_price - price) / entry_price
                # Target: price crosses below SMA-20
                if price <= sma20[i]:
                    exit_reason = "target_mean"
                # Stop loss
                elif price >= stop_price:
                    exit_reason = "stop_loss"
                # Time stop
                elif bars_held >= tuning.time_stop_bars:
                    exit_reason = "time_stop"

            if exit_reason:
                trades.append(BacktestTrade(
                    entry_bar=entry_bar,
                    exit_bar=i,
                    side=trade_side,
                    entry_price=round(entry_price, 4),
                    exit_price=round(price, 4),
                    pnl_pct=round(pnl_pct, 6),
                    bars_held=bars_held,
                    exit_reason=exit_reason,
                ))
                equity_curve.append(equity_curve[-1] * (1.0 + pnl_pct))
                in_trade = False
                continue

        else:
            # Check entries
            rsi_v = rsi_vals[i]
            zs_v = zs_vals[i]
            roc_v = roc_vals[i]
            atr_v = atr_vals[i]

            # SELL signal: overbought reversion
            sell_confluence = 0
            if rsi_v > tuning.rsi_overbought:
                sell_confluence += 1
            if price > bb_upper[i] or zs_v > tuning.zscore_threshold:
                sell_confluence += 1
            if roc_v > 0:
                sell_confluence += 1

            # BUY signal: oversold reversion
            buy_confluence = 0
            if rsi_v < tuning.rsi_oversold:
                buy_confluence += 1
            if price < bb_lower[i] or zs_v < -tuning.zscore_threshold:
                buy_confluence += 1
            if roc_v < 0:
                buy_confluence += 1

            if sell_confluence >= 3:
                in_trade = True
                trade_side = "SELL"
                entry_price = price
                entry_bar = i
                stop_price = price + tuning.atr_stop_mult * atr_v
            elif buy_confluence >= 3:
                in_trade = True
                trade_side = "BUY"
                entry_price = price
                entry_bar = i
                stop_price = price - tuning.atr_stop_mult * atr_v

    # Close any open trade at last bar
    if in_trade:
        price = closes[-1]
        bars_held = len(bars) - 1 - entry_bar
        if trade_side == "BUY":
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price
        trades.append(BacktestTrade(
            entry_bar=entry_bar,
            exit_bar=len(bars) - 1,
            side=trade_side,
            entry_price=round(entry_price, 4),
            exit_price=round(price, 4),
            pnl_pct=round(pnl_pct, 6),
            bars_held=bars_held,
            exit_reason="end_of_data",
        ))
        equity_curve.append(equity_curve[-1] * (1.0 + pnl_pct))

    # Compute statistics
    result.trades = trades
    result.total_return = round((equity_curve[-1] / equity_curve[0]) - 1.0, 6) if equity_curve else 0.0

    if trades:
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]
        result.win_rate = round(len(wins) / len(trades), 4)
        result.avg_win = round(sum(t.pnl_pct for t in wins) / len(wins), 6) if wins else 0.0
        result.avg_loss = round(sum(t.pnl_pct for t in losses) / len(losses), 6) if losses else 0.0

        # Sharpe: annualized (assume daily returns, ~252 trading days)
        returns = [t.pnl_pct for t in trades]
        if len(returns) >= 2:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = math.sqrt(var_r) if var_r > 0 else 1e-10
            # Annualize: assume avg bars_held per trade
            avg_held = sum(t.bars_held for t in trades) / len(trades)
            trades_per_year = 252.0 / max(avg_held, 1.0)
            result.sharpe = round((mean_r / std_r) * math.sqrt(trades_per_year), 4)

    # Max drawdown from equity curve
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    result.max_drawdown = round(max_dd, 6)

    return result


# ---------------------------------------------------------------------------
# Multi-symbol runner
# ---------------------------------------------------------------------------

def run_all(
    symbols: Sequence[str],
    output_json: Optional[str] = None,
) -> Dict[str, BacktestResult]:
    """Run backtest across all symbols and print summary."""
    results: Dict[str, BacktestResult] = {}

    print(f"\n{'='*80}")
    print(f"  GAMMA_REVERSION Backtest — {len(symbols)} symbols")
    print(f"{'='*80}\n")

    for sym in symbols:
        sym = sym.strip().upper()
        try:
            bars = load_bars(sym)
            result = run_reversion_backtest(sym, bars)
            results[sym] = result
        except FileNotFoundError as e:
            print(f"  SKIP {sym}: {e}")
            continue

    # Print per-symbol summary
    header = f"{'Symbol':<8} {'Trades':>6} {'Win%':>7} {'AvgWin':>8} {'AvgLoss':>8} {'Sharpe':>7} {'MaxDD':>8} {'Return':>8}"
    print(header)
    print("-" * len(header))

    total_trades = 0
    total_wins = 0
    all_returns: List[float] = []

    for sym, r in results.items():
        wins = sum(1 for t in r.trades if t.pnl_pct > 0)
        total_trades += len(r.trades)
        total_wins += wins
        all_returns.extend(t.pnl_pct for t in r.trades)

        print(
            f"{sym:<8} {len(r.trades):>6} {r.win_rate:>6.1%} "
            f"{r.avg_win:>7.3%} {r.avg_loss:>7.3%} "
            f"{r.sharpe:>7.2f} {r.max_drawdown:>7.3%} {r.total_return:>7.3%}"
        )

    # Aggregate
    print("-" * len(header))
    if total_trades > 0:
        agg_wr = total_wins / total_trades
        agg_mean = sum(all_returns) / len(all_returns)
        agg_var = sum((r - agg_mean) ** 2 for r in all_returns) / max(len(all_returns) - 1, 1)
        agg_std = math.sqrt(agg_var) if agg_var > 0 else 1e-10
        agg_sharpe = (agg_mean / agg_std) * math.sqrt(252.0 / 5.0)  # ~5 day avg hold
        print(
            f"{'TOTAL':<8} {total_trades:>6} {agg_wr:>6.1%} "
            f"{'':>8} {'':>8} "
            f"{agg_sharpe:>7.2f} {'':>8} {'':>8}"
        )
    else:
        print(f"{'TOTAL':<8} {'0':>6} {'N/A':>7}")

    print(f"\n{'='*80}\n")

    # Trade detail
    for sym, r in results.items():
        if r.trades:
            print(f"  {sym} trades:")
            for i, t in enumerate(r.trades):
                print(
                    f"    [{i+1:>2}] {t.side:<4} bar {t.entry_bar:>3}->{t.exit_bar:>3} "
                    f"${t.entry_price:.2f}->${t.exit_price:.2f} "
                    f"pnl={t.pnl_pct:>+7.3%} held={t.bars_held}d "
                    f"exit={t.exit_reason}"
                )
            print()

    # Write JSON report
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "symbols": list(results.keys()),
            "total_trades": total_trades,
            "aggregate_win_rate": round(total_wins / total_trades, 4) if total_trades else 0,
            "results": {
                sym: {
                    "trades": len(r.trades),
                    "win_rate": r.win_rate,
                    "avg_win": r.avg_win,
                    "avg_loss": r.avg_loss,
                    "sharpe": r.sharpe,
                    "max_drawdown": r.max_drawdown,
                    "total_return": r.total_return,
                    "trade_detail": [asdict(t) for t in r.trades],
                }
                for sym, r in results.items()
            },
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"  Report written to: {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="GAMMA_REVERSION backtest — RSI+Bollinger+Z-score mean reversion"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "IWM", "GLD", "TLT"],
        help="Symbols to backtest (default: SPY QQQ IWM GLD TLT)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON report (optional)",
    )
    args = parser.parse_args()

    results = run_all(args.symbols, output_json=args.output)

    # Exit 0 if we got trades, 1 if no signals at all
    total = sum(len(r.trades) for r in results.values())
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
chad/analytics/backtest_report.py

Report generation for CHAD backtesting engine.

Formats results into summary tables and JSON reports.
Highlights strong strategies (Sharpe > 1.0) and flags weak ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from chad.analytics.backtest_engine import BacktestResult, BacktestTrade


def print_summary(results: Dict[str, BacktestResult]) -> None:
    """Print a formatted summary table of backtest results."""
    agg = results.get("_AGGREGATE")

    print(f"\n{'='*90}")
    strategy = agg.strategy_name if agg else "unknown"
    print(f"  Backtest Results: {strategy}")
    if agg:
        print(f"  Period: {agg.start_date.strftime('%Y-%m-%d')} to {agg.end_date.strftime('%Y-%m-%d')} ({agg.total_bars} bars)")
    print(f"{'='*90}\n")

    header = f"{'Symbol':<12} {'Trades':>6} {'Win%':>7} {'AvgWin':>8} {'AvgLoss':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8} {'Return':>8} {'Status'}"
    print(header)
    print("-" * len(header))

    for sym, r in sorted(results.items()):
        if sym == "_AGGREGATE":
            continue

        status = _assess_status(r)
        print(
            f"{sym:<12} {len(r.trades):>6} {r.win_rate:>6.1%} "
            f"{r.avg_win_pct:>7.3%} {r.avg_loss_pct:>7.3%} "
            f"{r.profit_factor:>5.1f} {r.sharpe_ratio:>7.2f} "
            f"{r.max_drawdown_pct:>7.3%} {r.total_return_pct:>7.3%} {status}"
        )

    # Aggregate line
    if agg:
        print("-" * len(header))
        status = _assess_status(agg)
        print(
            f"{'AGGREGATE':<12} {len(agg.trades):>6} {agg.win_rate:>6.1%} "
            f"{agg.avg_win_pct:>7.3%} {agg.avg_loss_pct:>7.3%} "
            f"{agg.profit_factor:>5.1f} {agg.sharpe_ratio:>7.2f} "
            f"{agg.max_drawdown_pct:>7.3%} {agg.total_return_pct:>7.3%} {status}"
        )

    print(f"\n{'='*90}")

    # Trade detail
    for sym, r in sorted(results.items()):
        if sym == "_AGGREGATE" or not r.trades:
            continue
        print(f"\n  {sym} trades ({len(r.trades)}):")
        for i, t in enumerate(r.trades):
            print(
                f"    [{i+1:>2}] {t.side:<4} {t.symbol:<6} "
                f"${t.entry_price:.2f}->${t.exit_price:.2f} "
                f"pnl={t.pnl_pct:>+7.3%} held={t.bars_held}d "
                f"exit={t.exit_reason}"
            )

    print()


def _assess_status(r: BacktestResult) -> str:
    """Assess strategy health from backtest results."""
    if not r.trades:
        return "NO_TRADES"
    if r.sharpe_ratio > 1.0 and r.win_rate > 0.55:
        return "STRONG"
    if r.sharpe_ratio > 0.5 and r.win_rate > 0.45:
        return "OK"
    if r.sharpe_ratio < 0 or r.win_rate < 0.40:
        return "REVIEW_NEEDED"
    return "MARGINAL"


def write_json_report(results: Dict[str, BacktestResult], path: Path) -> None:
    """Write backtest results to a JSON file."""
    report: Dict[str, Any] = {}
    for sym, r in results.items():
        report[sym] = {
            "strategy_name": r.strategy_name,
            "symbol": r.symbol,
            "start_date": r.start_date.isoformat(),
            "end_date": r.end_date.isoformat(),
            "total_bars": r.total_bars,
            "num_trades": len(r.trades),
            "win_rate": r.win_rate,
            "avg_win_pct": r.avg_win_pct,
            "avg_loss_pct": r.avg_loss_pct,
            "profit_factor": r.profit_factor,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown_pct": r.max_drawdown_pct,
            "total_return_pct": r.total_return_pct,
            "trades_per_year": r.trades_per_year,
            "trades": [
                {
                    "symbol": t.symbol, "side": t.side,
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "quantity": t.quantity, "bars_held": t.bars_held,
                    "pnl": t.pnl, "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                }
                for t in r.trades
            ],
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

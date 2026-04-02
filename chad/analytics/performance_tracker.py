#!/usr/bin/env python3
"""
chad/analytics/performance_tracker.py

Deterministic strategy performance tracker for CHAD.

Purpose
-------
Track per-strategy trade outcomes and compute summary metrics used by:
- expectancy engine
- capital allocator
- reporting
- risk throttling
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, timezone


@dataclass
class TradeRecord:
    strategy: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    opened_at_utc: str
    closed_at_utc: str


@dataclass
class StrategyPerformance:
    strategy: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    expectancy: float


class PerformanceTracker:
    def __init__(self, path: str = "/home/ubuntu/chad_finale/runtime/performance_tracker.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, List[dict]]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
            return {}
        except Exception:
            return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def record_trade(self, record: TradeRecord) -> None:
        bucket = self._data.setdefault(record.strategy, [])
        bucket.append(asdict(record))
        self._save()

    def summarize_strategy(self, strategy: str) -> StrategyPerformance:
        rows = self._data.get(strategy, [])

        pnls = [float(r.get("pnl", 0.0)) for r in rows]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]

        trades = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = (win_count / trades) if trades else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (abs(sum(losses)) / len(losses)) if losses else 0.0
        total_pnl = sum(pnls)
        loss_rate = (loss_count / trades) if trades else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return StrategyPerformance(
            strategy=strategy,
            trades=trades,
            wins=win_count,
            losses=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_pnl=total_pnl,
            expectancy=expectancy,
        )

    def summarize_all(self) -> Dict[str, StrategyPerformance]:
        return {
            strategy: self.summarize_strategy(strategy)
            for strategy in sorted(self._data.keys())
        }


if __name__ == "__main__":
    tracker = PerformanceTracker()
    summaries = tracker.summarize_all()
    for strategy, summary in summaries.items():
        print(strategy, asdict(summary))

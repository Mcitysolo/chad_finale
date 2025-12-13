#!/usr/bin/env python3
"""
chad/execution/logger.py

Dry-run execution logger for CHAD.

This module provides a production-safe sink that records what CHAD *would*
trade on each decision cycle, without sending any orders to brokers.

Output format:
    - NDJSON (one JSON object per line)
    - Path: <root>/reports/execution_log.ndjson

Each logged record includes:
    - cycle_timestamp: ISO 8601 time of the decision cycle
    - symbol, side, net_size
    - strategies: list of contributing strategy names
    - price: best-known price at the time of decision (if available)
    - legend_weight: legend consensus weight for the symbol (if available)
    - context_ts: MarketContext.now
    - meta: optional extra fields for future extensions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

from chad.types import LegendConsensus, MarketContext
from chad.utils.signal_router import RoutedSignal


@dataclass(frozen=True)
class ExecutionLoggerConfig:
    """
    Configuration for the ExecutionLogger.

    Phase 3:
        - root_dir: project root (containing reports/).
        - reports_rel_path: relative directory path for reports.
        - filename: file name for the NDJSON log.
    """

    root_dir: Path = Path(__file__).resolve().parents[2]
    reports_rel_path: Path = Path("reports")
    filename: str = "execution_log.ndjson"

    @property
    def log_path(self) -> Path:
        return (self.root_dir / self.reports_rel_path / self.filename).resolve()


class ExecutionLogger:
    """
    Dry-run execution logger.

    This class appends one NDJSON line per RoutedSignal. It is safe to call
    from the orchestrator at every cycle; log files can be rotated externally
    by logrotate or any other log-management process.
    """

    def __init__(self, config: ExecutionLoggerConfig | None = None) -> None:
        self._config = config or ExecutionLoggerConfig()
        # Ensure reports directory exists.
        self._config.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_cycle(
        self,
        context: MarketContext,
        routed_signals: Iterable[RoutedSignal],
    ) -> None:
        """
        Log all routed signals for a single decision cycle.

        This function is best-effort: failures to write are raised to the caller
        so that the orchestrator can decide how to handle them (e.g. log and
        continue, or escalate).
        """
        cycle_ts = datetime.now(timezone.utc).isoformat()
        prices: Mapping[str, float] = {
            sym: tick.price for sym, tick in context.ticks.items()
        }
        legend: Optional[LegendConsensus] = context.legend

        path = self._config.log_path
        with path.open("a", encoding="utf-8") as f:
            for r in routed_signals:
                symbol = r.symbol
                price = prices.get(symbol)
                legend_weight: Optional[float] = None
                if legend is not None:
                    legend_weight = legend.weights.get(symbol)

                record = {
                    "cycle_timestamp": cycle_ts,
                    "context_timestamp": context.now.isoformat(),
                    "symbol": symbol,
                    "side": r.side.value,
                    "net_size": r.net_size,
                    "strategies": [s.value for s in r.source_strategies],
                    "price": price,
                    "legend_weight": legend_weight,
                    "meta": {},
                }
                f.write(json.dumps(record) + "\n")


_default_logger: Optional[ExecutionLogger] = None


def get_execution_logger() -> ExecutionLogger:
    """
    Return the process-global ExecutionLogger singleton.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = ExecutionLogger()
    return _default_logger

#!/usr/bin/env python3
"""
chad/utils/context_builder.py

Context builder for CHAD Phase 3.

This module assembles all the inputs required for a single decision cycle:

    - Market ticks from the latest Polygon NDJSON feed
    - Legend consensus from data/legend_top_stocks.json
    - A portfolio snapshot (Phase 3: conservative, zero-position baseline)
    - Derived prices and notional exposure views

It returns:
    - MarketContext
    - prices (symbol -> last price)
    - current_symbol_notional
    - current_total_notional

Execution, real portfolio state, and PnL tracking are wired in later phases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Tuple

from chad.types import (
    AssetClass,
    LegendConsensus,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
)
from chad.utils.legend_loader import load_legend, LegendLoaderError


@dataclass(frozen=True)
class ContextBuilderConfig:
    """
    Configuration for ContextBuilder.

    Phase 3:
    - root_dir: project root containing data/, control/, etc.
    - feeds_rel_path: relative path to NDJSON feed directory.
    """

    root_dir: Path = Path(__file__).resolve().parents[2]
    feeds_rel_path: Path = Path("data/feeds")

    @property
    def feeds_dir(self) -> Path:
        return (self.root_dir / self.feeds_rel_path).resolve()


@dataclass(frozen=True)
class ContextBuildResult:
    """
    Full result for a single context build operation.
    """

    context: MarketContext
    prices: Dict[str, float]
    current_symbol_notional: Dict[str, float]
    current_total_notional: float


class ContextBuilder:
    """
    Build MarketContext + risk inputs from on-disk data.

    Responsibilities:
      - Locate latest Polygon NDJSON file.
      - Parse per-symbol last trade ticks.
      - Load LegendConsensus via legend_loader.
      - Build a conservative PortfolioSnapshot (Phase 3 baseline).
      - Derive prices and notional exposures.

    Assumptions for Phase 3:
      - Portfolio is flat (no open positions), all cash baseline.
        This is the safest starting point; live portfolio integration is added
        in later phases when execution/accounting are wired.
    """

    def __init__(self, config: ContextBuilderConfig | None = None) -> None:
        self._config = config or ContextBuilderConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(self) -> ContextBuildResult:
        """
        Build a full decision context from the latest available data.

        Raises:
            RuntimeError if critical inputs are missing (no feed, no legend, etc).
        """
        now = datetime.now(timezone.utc)

        ticks = self._load_latest_ticks()
        legend = self._load_legend()
        portfolio = self._build_portfolio_snapshot(now)

        prices = {sym: tick.price for sym, tick in ticks.items()}

        # Phase 3 baseline: zero notional (no positions yet).
        current_symbol_notional: Dict[str, float] = {}
        current_total_notional = 0.0

        ctx = MarketContext(
            now=now,
            ticks=ticks,
            legend=legend,
            portfolio=portfolio,
        )

        return ContextBuildResult(
            context=ctx,
            prices=prices,
            current_symbol_notional=current_symbol_notional,
            current_total_notional=current_total_notional,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_latest_ticks(self) -> Dict[str, MarketTick]:
        """
        Load last trade ticks for each symbol from the latest NDJSON file.

        We treat the most recent file by modification time as the latest feed.

        Raises:
            RuntimeError if no suitable NDJSON file is found or parsing fails.
        """
        feeds_dir = self._config.feeds_dir
        if not feeds_dir.exists():
            raise RuntimeError(f"Feeds directory not found: {feeds_dir}")

        candidates = sorted(
            feeds_dir.glob("polygon_stocks_*.ndjson"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not candidates:
            raise RuntimeError(f"No polygon_stocks_*.ndjson files in {feeds_dir}")

        latest = candidates[0]

        last_per_symbol: Dict[str, MarketTick] = {}
        try:
            with latest.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue

                    symbol = payload.get("ticker")
                    price = payload.get("price")
                    size = payload.get("size")
                    exchange = payload.get("exchange")
                    ts_raw = payload.get("timestamp_utc")

                    if not isinstance(symbol, str):
                        continue
                    try:
                        price_f = float(price)
                        size_f = float(size)
                    except Exception:
                        continue
                    if price_f <= 0.0 or size_f <= 0.0:
                        continue

                    timestamp = self._parse_timestamp(ts_raw)

                    last_per_symbol[symbol] = MarketTick(
                        symbol=symbol,
                        price=price_f,
                        size=size_f,
                        exchange=int(exchange) if isinstance(exchange, int) else None,
                        timestamp=timestamp,
                        source="polygon_ndjson",
                    )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to read NDJSON feed {latest}: {exc}") from exc

        if not last_per_symbol:
            raise RuntimeError(f"No valid ticks found in latest feed: {latest}")

        return last_per_symbol

    @staticmethod
    def _parse_timestamp(raw: object) -> datetime:
        """
        Parse an ISO 8601 timestamp string into an aware datetime.

        If parsing fails or input is missing, fall back to current UTC time.
        """
        if isinstance(raw, str):
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
        return datetime.now(timezone.utc)

    def _load_legend(self) -> LegendConsensus:
        """
        Load LegendConsensus from the standard path using the LegendLoader.

        Raises:
            RuntimeError if the legend cannot be loaded.
        """
        try:
            return load_legend()
        except LegendLoaderError as exc:
            raise RuntimeError(f"Failed to load legend consensus: {exc}") from exc

    @staticmethod
    def _build_portfolio_snapshot(now: datetime) -> PortfolioSnapshot:
        """
        Build a conservative portfolio snapshot for Phase 3:

        - All cash, no positions.

        Live portfolios will be wired once execution/accounting is restored.
        """
        return PortfolioSnapshot(
            timestamp=now,
            cash=100_000.0,
            positions={},
        )

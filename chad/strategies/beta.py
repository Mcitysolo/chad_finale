#!/usr/bin/env python3
"""
chad/strategies/beta.py

BetaBrain (Phase 3) — legend / long-term allocator.

Uses the Legend consensus file (generated from data/legend_top_stocks.json)
to propose small, incremental entries into high-conviction symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from chad.types import (
    AssetClass,
    LegendConsensus,
    MarketContext,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)


@dataclass(frozen=True)
class BetaParams:
    """
    Tunable parameters for Beta.

    Very conservative profile for Phase 3; real sizing and cash controls are
    handled by the execution/risk layer in later phases.
    """

    min_cash: float = 10_000.0
    min_weight: float = 0.05  # minimum legend weight to consider
    max_symbols: int = 10     # focus on top N legend names
    base_size: float = 3.0    # nominal units per entry


DEFAULT_PARAMS = BetaParams()


def _sorted_legend_weights(legend: LegendConsensus, params: BetaParams) -> List[tuple[str, float]]:
    """
    Return top N legend weights above min_weight, sorted descending.
    """
    items = [
        (sym, w)
        for sym, w in legend.weights.items()
        if w >= params.min_weight
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[: params.max_symbols]


def build_beta_config() -> StrategyConfig:
    """
    StrategyConfig for BetaBrain.
    """
    return StrategyConfig(
        name=StrategyName.BETA,
        enabled=True,
        target_universe=None,  # derived dynamically from legend
        max_gross_exposure=None,
        notes="Phase-3 BetaBrain (Legend-driven long-term entries).",
    )


def beta_handler(ctx: MarketContext, params: BetaParams | None = None) -> Sequence[TradeSignal]:
    """
    Public handler for Beta, suitable for registration with StrategyEngine.
    """
    p = params or DEFAULT_PARAMS
    legend = ctx.legend
    if legend is None:
        return []

    if ctx.portfolio.cash < p.min_cash:
        return []

    portfolio: PortfolioSnapshot = ctx.portfolio
    positions: dict[str, Position] = dict(portfolio.positions)

    signals: List[TradeSignal] = []

    for symbol, weight in _sorted_legend_weights(legend, p):
        pos = positions.get(symbol)
        qty = pos.quantity if pos is not None else 0.0

        # For Phase 3, Beta only proposes entries when we have zero exposure.
        if qty > 0:
            continue

        signals.append(
            TradeSignal(
                strategy=StrategyName.BETA,
                symbol=symbol,
                side=SignalSide.BUY,
                size=p.base_size,
                confidence=min(1.0, max(0.5, weight * 2.0)),  # simple mapping of legend weight → confidence
                asset_class=AssetClass.EQUITY,
                created_at=ctx.now,
                meta={
                    "reason": "legend_weight_entry",
                    "legend_weight": weight,
                    "legend_as_of": legend.as_of.isoformat(),
                },
            )
        )

    return signals

#!/usr/bin/env python3
"""
chad/strategies/gamma.py

GammaBrain — swing/mean-reversion brain.

Phase 9 upgrade:
- Universe is no longer hardcoded.
- Uses UniverseProvider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from chad.types import (
    AssetClass,
    MarketContext,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)
from chad.utils.universe_provider import get_trade_universe


@dataclass(frozen=True)
class GammaParams:
    min_cash: float = 10_000.0
    inner_band_pct: float = 0.005
    outer_band_pct: float = 0.02
    base_size: float = 5.0
    max_position_units: float = 50.0


DEFAULT_PARAMS = GammaParams()


def _get_anchor_price(symbol: str, ctx: MarketContext) -> float | None:
    portfolio: PortfolioSnapshot = ctx.portfolio
    if symbol in portfolio.positions:
        pos: Position = portfolio.positions[symbol]
        if pos.avg_price > 0:
            return float(pos.avg_price)

    tick = ctx.ticks.get(symbol)
    if tick is not None and tick.price > 0:
        return float(tick.price)
    return None


def _propose_for_symbol(symbol: str, ctx: MarketContext, params: GammaParams) -> Sequence[TradeSignal]:
    portfolio: PortfolioSnapshot = ctx.portfolio
    if portfolio.cash < params.min_cash:
        return []

    tick = ctx.ticks.get(symbol)
    if tick is None or tick.price <= 0:
        return []

    anchor = _get_anchor_price(symbol, ctx)
    if anchor is None or anchor <= 0:
        return []

    price = float(tick.price)
    deviation = (price - anchor) / anchor

    if abs(deviation) < params.inner_band_pct:
        return []

    pos = portfolio.positions.get(symbol)
    qty = pos.quantity if pos is not None else 0.0
    if abs(qty) >= params.max_position_units:
        return []

    signals: List[TradeSignal] = []

    if deviation <= -params.outer_band_pct and qty <= 0:
        signals.append(
            TradeSignal(
                strategy=StrategyName.GAMMA,
                symbol=symbol,
                side=SignalSide.BUY,
                size=params.base_size,
                confidence=0.65,
                asset_class=AssetClass.ETF,
                created_at=ctx.now,
                meta={
                    "reason": "mean_reversion_buy",
                    "anchor": anchor,
                    "price": price,
                    "deviation_pct": deviation,
                    "band": f"{params.outer_band_pct:.3%}",
                },
            )
        )

    if deviation >= params.outer_band_pct and qty > 0:
        size = min(params.base_size, qty)
        signals.append(
            TradeSignal(
                strategy=StrategyName.GAMMA,
                symbol=symbol,
                side=SignalSide.SELL,
                size=size,
                confidence=0.6,
                asset_class=AssetClass.ETF,
                created_at=ctx.now,
                meta={
                    "reason": "mean_reversion_sell",
                    "anchor": anchor,
                    "price": price,
                    "deviation_pct": deviation,
                    "band": f"{params.outer_band_pct:.3%}",
                },
            )
        )

    return signals


def build_gamma_config() -> StrategyConfig:
    universe = get_trade_universe()
    return StrategyConfig(
        name=StrategyName.GAMMA,
        enabled=True,
        target_universe=universe,
        max_gross_exposure=None,
        notes="GammaBrain (configurable universe; safe-by-default).",
    )


def gamma_handler(ctx: MarketContext, params: GammaParams | None = None) -> Sequence[TradeSignal]:
    p = params or DEFAULT_PARAMS
    universe = get_trade_universe()
    signals: List[TradeSignal] = []
    for symbol in universe:
        signals.extend(_propose_for_symbol(symbol, ctx, p))
    return signals


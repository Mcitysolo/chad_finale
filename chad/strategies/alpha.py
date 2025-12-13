#!/usr/bin/env python3
"""
chad/strategies/alpha.py

AlphaBrain (Phase 3) — intraday / tactical stock & ETF brain.

Design goals for this phase:
- Keep logic simple, deterministic, and explainable.
- Operate only on a small, known universe (SPY/QQQ).
- Use MarketContext (ticks + portfolio) to propose modest trades.
- Be safe by default: zero signals if required data is missing.

Sizing / risk is *not* handled here; that comes in the execution/risk engine.
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


ALPHA_UNIVERSE = ("SPY", "QQQ")


@dataclass(frozen=True)
class AlphaParams:
    """
    Tunable parameters for Alpha.

    For now these are fixed constants; later phases can move this to control/
    files or a database.
    """

    min_cash: float = 5_000.0
    buy_discount_pct: float = 0.005  # 0.5% below anchor → consider buy
    sell_premium_pct: float = 0.005  # 0.5% above anchor → consider sell
    base_size: float = 5.0  # nominal trade size in units


DEFAULT_PARAMS = AlphaParams()


def _get_anchor_price(symbol: str, ctx: MarketContext) -> float | None:
    """
    Anchor logic for Phase 3:

    - If the symbol exists in the current portfolio, use avg_price as anchor.
    - Otherwise, if there's a tick, use its price as anchor.
    - If neither is present, return None (no trade).
    """
    portfolio: PortfolioSnapshot = ctx.portfolio
    if symbol in portfolio.positions:
        pos: Position = portfolio.positions[symbol]
        if pos.avg_price > 0:
            return float(pos.avg_price)

    tick = ctx.ticks.get(symbol)
    if tick is not None and tick.price > 0:
        return float(tick.price)

    return None


def _propose_for_symbol(
    symbol: str,
    ctx: MarketContext,
    params: AlphaParams,
) -> Sequence[TradeSignal]:
    """
    Generate zero or more trade signals for a single symbol.
    """
    if ctx.portfolio.cash < params.min_cash:
        # Not enough idle cash; skip.
        return []

    tick = ctx.ticks.get(symbol)
    if tick is None or tick.price <= 0:
        # No live price; skip this symbol.
        return []

    anchor = _get_anchor_price(symbol, ctx)
    if anchor is None or anchor <= 0:
        return []

    price = float(tick.price)
    discount_threshold = anchor * (1.0 - params.buy_discount_pct)
    premium_threshold = anchor * (1.0 + params.sell_premium_pct)

    signals: List[TradeSignal] = []

    # Buy if price is sufficiently cheap vs anchor and we have no / small position.
    pos = ctx.portfolio.positions.get(symbol)
    qty = pos.quantity if pos is not None else 0.0

    if price <= discount_threshold and qty <= 0:
        signals.append(
            TradeSignal(
                strategy=StrategyName.ALPHA,
                symbol=symbol,
                side=SignalSide.BUY,
                size=params.base_size,
                confidence=0.7,
                asset_class=AssetClass.ETF,
                created_at=ctx.now,
                meta={
                    "reason": "discount_vs_anchor",
                    "anchor": anchor,
                    "price": price,
                },
            )
        )

    # Sell (scale-out) if price is rich vs anchor and we hold some.
    if price >= premium_threshold and qty > 0:
        size = min(params.base_size, qty)
        signals.append(
            TradeSignal(
                strategy=StrategyName.ALPHA,
                symbol=symbol,
                side=SignalSide.SELL,
                size=size,
                confidence=0.6,
                asset_class=AssetClass.ETF,
                created_at=ctx.now,
                meta={
                    "reason": "premium_vs_anchor",
                    "anchor": anchor,
                    "price": price,
                },
            )
        )

    return signals


def build_alpha_config() -> StrategyConfig:
    """
    StrategyConfig for AlphaBrain.
    """
    return StrategyConfig(
        name=StrategyName.ALPHA,
        enabled=True,
        target_universe=list(ALPHA_UNIVERSE),
        max_gross_exposure=None,
        notes="Phase-3 AlphaBrain (SPY/QQQ anchor-based tactical entries/exits).",
    )


def alpha_handler(ctx: MarketContext, params: AlphaParams | None = None) -> Sequence[TradeSignal]:
    """
    Public handler for Alpha, suitable for registration with StrategyEngine.
    """
    p = params or DEFAULT_PARAMS
    signals: List[TradeSignal] = []
    for symbol in ALPHA_UNIVERSE:
        signals.extend(_propose_for_symbol(symbol, ctx, p))
    return signals

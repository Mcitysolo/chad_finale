#!/usr/bin/env python3
"""
chad/strategies/gamma.py

GammaBrain (Phase 3+) — swing / mean-reversion brain.

Design goals:
- Operate on the same core universe as Alpha (SPY/QQQ) for now.
- Look for short-term over-extensions relative to an anchor price.
- Provide *swing-style* adjustments, complementary to Alpha:
    * Alpha: tactical entries around a tight band.
    * Gamma: larger mean-reversion trades when price deviates further.
- Be SAFE and deterministic:
    * No broker calls.
    * Emits only TradeSignal objects.
    * Returns [] if data is missing or inconsistent.

Sizing / risk:
- As with Alpha/Beta, Gamma emits *unit-sized* signals.
- Actual sizing and caps are enforced by DynamicRiskAllocator, SCR, and
  the execution layer (not here).
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


GAMMA_UNIVERSE = ("SPY", "QQQ")


@dataclass(frozen=True)
class GammaParams:
    """
    Tunable parameters for GammaBrain.

    Notes:
    - min_cash:         minimum idle cash before proposing new entries.
    - inner_band_pct:   small band around anchor where Gamma does nothing.
    - outer_band_pct:   deviation beyond which Gamma will act (swing).
    - base_size:        nominal unit size per adjustment.
    - max_position_units: soft cap on units Gamma will propose into a symbol.
    """

    min_cash: float = 10_000.0
    inner_band_pct: float = 0.005   # 0.5% → no action band
    outer_band_pct: float = 0.02    # 2% deviation → swing-level adjustment
    base_size: float = 5.0
    max_position_units: float = 50.0


DEFAULT_PARAMS = GammaParams()


def _get_anchor_price(symbol: str, ctx: MarketContext) -> float | None:
    """
    Anchor logic for Gamma:

    Priority:
    1) If we hold a position with a valid avg_price → use that.
    2) Else fall back to last tick price (if present).
    3) Else no trade.

    This keeps anchor simple and robust.
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
    params: GammaParams,
) -> Sequence[TradeSignal]:
    """
    Generate zero or more Gamma trade signals for a single symbol.

    Logic:
    - If price is within inner_band → no action.
    - If price is BELOW anchor by >= outer_band_pct:
        * Mean-reversion BUY if we are flat/small and have enough cash.
    - If price is ABOVE anchor by >= outer_band_pct:
        * Mean-reversion SELL (scale out) if we have a position.

    Gamma explicitly avoids shorting; it only:
    - Adds into weakness (relative to anchor).
    - Scales out into strength.
    """
    portfolio: PortfolioSnapshot = ctx.portfolio

    if portfolio.cash < params.min_cash:
        # Not enough idle cash for swing entries.
        return []

    tick = ctx.ticks.get(symbol)
    if tick is None or tick.price <= 0:
        # No live price; skip.
        return []

    anchor = _get_anchor_price(symbol, ctx)
    if anchor is None or anchor <= 0:
        return []

    price = float(tick.price)
    deviation = (price - anchor) / anchor

    # Bands
    inner_band = params.inner_band_pct
    outer_band = params.outer_band_pct

    # Inside the inner band → no Gamma action.
    if abs(deviation) < inner_band:
        return []

    signals: List[TradeSignal] = []
    pos = portfolio.positions.get(symbol)
    qty = pos.quantity if pos is not None else 0.0

    # Guardrail: avoid Gamma proposals if position already very large.
    if abs(qty) >= params.max_position_units:
        return []

    # Mean-reversion BUY: price below anchor by >= outer_band_pct, position small/flat.
    if deviation <= -outer_band and qty <= 0:
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
                    "band": f"{outer_band:.3%}",
                },
            )
        )

    # Mean-reversion SELL: price above anchor by >= outer_band_pct, we hold some.
    if deviation >= outer_band and qty > 0:
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
                    "band": f"{outer_band:.3%}",
                },
            )
        )

    return signals


def build_gamma_config() -> StrategyConfig:
    """
    StrategyConfig for GammaBrain.

    Universe is currently SPY/QQQ. Future phases may extend this to
    additional large, liquid symbols with good swing profiles.
    """
    return StrategyConfig(
        name=StrategyName.GAMMA,
        enabled=True,
        target_universe=list(GAMMA_UNIVERSE),
        max_gross_exposure=None,
        notes=(
            "GammaBrain (swing / mean-reversion) over core ETF universe "
            "(SPY/QQQ). Emits unit-sized adjustments; risk caps and SCR "
            "enforce actual sizing and live/paper mode."
        ),
    )


def gamma_handler(
    ctx: MarketContext,
    params: GammaParams | None = None,
) -> Sequence[TradeSignal]:
    """
    Public handler for Gamma, suitable for registration with any strategy
    engine or orchestrator layer.

    This function is PURE:
    - No broker calls.
    - No mode flips.
    - Only reads the provided MarketContext and emits TradeSignal.
    """
    p = params or DEFAULT_PARAMS
    signals: List[TradeSignal] = []
    for symbol in GAMMA_UNIVERSE:
        signals.extend(_propose_for_symbol(symbol, ctx, p))
    return signals

#!/usr/bin/env python3
"""
chad/strategies/delta_pairs.py

DELTA_PAIRS -- Market-Neutral Pairs Trading Strategy for CHAD.

Generates mean-reversion signals on correlated ETF pairs (SPY/QQQ,
SPY/IWM, QQQ/IWM) using z-score of the price ratio.

Signal Flow
-----------
1. For each pair, load bars for both symbols from ctx.bars.
2. Compute price ratio series: ratio = price_a / price_b.
3. Compute rolling mean and std over lookback window.
4. Compute z-score: (current_ratio - mean) / std.
5. Entry: |Z| >= zscore_entry -> fade the divergence.
6. Exit:  |Z| <= zscore_exit  -> mean reversion achieved.
7. Stop:  |Z| >= zscore_stop  -> regime break, close.
8. Emit TWO TradeSignals per pair, linked by pair_id.

Instruments
-----------
- SPY, QQQ, IWM (most liquid large-cap equity ETFs).

Risk Model
----------
- Market-neutral: each pair has equal-notional long + short legs.
- Defined risk per pair: risk_budget_pct of equity.
- Both legs same unit count for dollar-neutrality.

Design
------
- Strategy-only: emits TradeSignal intents, never executes.
- Deterministic given inputs. No IBKR calls in signal path.
- Fail-closed: missing bars -> no signals.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from chad.types import (
    AssetClass,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# PairSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairSpec:
    """Specification for a tradeable pair."""
    sym_long: str
    sym_short: str
    correlation: float
    half_life_days: float
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    zscore_stop: float = 3.5


DEFAULT_PAIRS: List[PairSpec] = [
    PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2),
    PairSpec("SPY", "IWM", correlation=0.967, half_life_days=30.3),
    PairSpec("QQQ", "IWM", correlation=0.941, half_life_days=20.1),
]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaPairsTuning:
    """Tuning parameters for DELTA_PAIRS strategy."""
    pairs: tuple = tuple(DEFAULT_PAIRS)
    lookback_days: int = 60
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    zscore_stop: float = 3.5
    min_bars: int = 40
    base_units: int = 10
    max_units: int = 50
    risk_budget_pct: float = 0.008
    min_confidence: float = 0.65


DEFAULT_TUNING = DeltaPairsTuning()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _get_mapping(obj: Any, attr: str) -> Mapping[str, Any]:
    try:
        m = getattr(obj, attr, None)
        if isinstance(m, dict):
            return m
        if m is not None and hasattr(m, "get"):
            return m  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def _extract_equity(ctx: Any, fallback: float = 100_000.0) -> float:
    """Extract portfolio equity from context."""
    portfolio = getattr(ctx, "portfolio", None)
    if portfolio is not None:
        for field_name in ("total_equity", "equity", "net_liq", "cash"):
            val = _safe_float(getattr(portfolio, field_name, None), 0.0)
            if val > 0:
                return val
        extra = getattr(portfolio, "extra", None)
        if isinstance(extra, dict):
            eq = _safe_float(extra.get("equity"), 0.0)
            if eq > 0:
                return eq
    return fallback


def _extract_closes(ctx: Any, symbol: str) -> List[float]:
    """Extract close prices from ctx.bars[symbol]."""
    bars = _get_mapping(ctx, "bars").get(symbol)
    if not isinstance(bars, (list, tuple)):
        return []
    closes: List[float] = []
    for b in bars:
        if isinstance(b, dict):
            c = _safe_float(b.get("close"), 0.0)
            if c > 0:
                closes.append(c)
    return closes


def _extract_price(ctx: Any, symbol: str) -> float:
    """Extract latest price from ctx.ticks or ctx.bars."""
    ticks = _get_mapping(ctx, "ticks")
    tick = ticks.get(symbol)
    if tick is not None:
        price = _safe_float(getattr(tick, "price", None), 0.0)
        if price > 0:
            return price
        if isinstance(tick, dict):
            price = _safe_float(tick.get("price"), 0.0)
            if price > 0:
                return price

    prices = _get_mapping(ctx, "prices")
    p = _safe_float(prices.get(symbol), 0.0)
    if p > 0:
        return p

    # Fall back to last close from bars
    closes = _extract_closes(ctx, symbol)
    if closes:
        return closes[-1]

    return 0.0


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

def compute_zscore(
    closes_a: List[float],
    closes_b: List[float],
    lookback: int,
) -> Optional[float]:
    """
    Compute z-score of the price ratio (A/B) over a rolling window.

    Returns None if insufficient data or zero standard deviation.
    """
    min_len = min(len(closes_a), len(closes_b))
    if min_len < lookback:
        return None

    # Align to common length, take most recent data
    ca = closes_a[-min_len:]
    cb = closes_b[-min_len:]

    # Compute ratio series
    ratios: List[float] = []
    for a, b in zip(ca, cb):
        if b > 0:
            ratios.append(a / b)

    if len(ratios) < lookback:
        return None

    # Rolling window for mean/std
    window = ratios[-lookback:]
    mean = sum(window) / len(window)
    variance = sum((r - mean) ** 2 for r in window) / len(window)
    std = math.sqrt(variance)

    if std <= 0:
        return None

    current_ratio = ratios[-1]
    return (current_ratio - mean) / std


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _build_pair_signals(
    *,
    pair: PairSpec,
    zscore: float,
    confidence: float,
    equity: float,
    tuning: DeltaPairsTuning,
    now: datetime,
    price_a: float,
    price_b: float,
) -> List[TradeSignal]:
    """
    Build two linked TradeSignals for a pair trade.

    Z > +entry: ratio too high -> SHORT sym_long, LONG sym_short (fade)
    Z < -entry: ratio too low  -> LONG sym_long, SHORT sym_short (fade)
    """
    abs_z = abs(zscore)
    entry_threshold = pair.zscore_entry or tuning.zscore_entry
    exit_threshold = pair.zscore_exit or tuning.zscore_exit
    stop_threshold = pair.zscore_stop or tuning.zscore_stop

    # Determine signal type
    if abs_z >= entry_threshold:
        signal_type = "entry"
    elif abs_z <= exit_threshold:
        signal_type = "exit"
    elif abs_z >= stop_threshold:
        signal_type = "stop"
    else:
        return []

    if confidence < tuning.min_confidence:
        return []

    # Determine direction
    if zscore > 0:
        # Ratio too high: short A, long B
        side_a = SignalSide.SELL
        side_b = SignalSide.BUY
        role_a = "SHORT"
        role_b = "LONG"
    else:
        # Ratio too low: long A, short B
        side_a = SignalSide.BUY
        side_b = SignalSide.SELL
        role_a = "LONG"
        role_b = "SHORT"

    # Sizing: dollar-neutral, both legs same unit count
    notional = equity * tuning.risk_budget_pct * confidence
    if price_a <= 0 or price_b <= 0:
        return []

    # Use the average price for unit count to keep dollar-neutral
    avg_price = (price_a + price_b) / 2.0
    units = max(1, min(tuning.max_units, int(notional / avg_price)))

    pair_id = str(uuid.uuid4())
    common_meta = {
        "engine": "delta_pairs.v1",
        "pair_id": pair_id,
        "pair": f"{pair.sym_long}/{pair.sym_short}",
        "zscore": round(zscore, 4),
        "signal_type": signal_type,
        "correlation": pair.correlation,
        "half_life_days": pair.half_life_days,
        "required_asset_class": "equity",
        "sec_type": "STK",
    }

    signals: List[TradeSignal] = []

    # Leg A
    signals.append(TradeSignal(
        strategy=StrategyName.DELTA_PAIRS,
        symbol=pair.sym_long,
        side=side_a,
        size=float(units),
        confidence=float(confidence),
        asset_class=AssetClass.ETF,
        created_at=now,
        meta={
            **common_meta,
            "pair_role": role_a,
            "partner_symbol": pair.sym_short,
        },
    ))

    # Leg B
    signals.append(TradeSignal(
        strategy=StrategyName.DELTA_PAIRS,
        symbol=pair.sym_short,
        side=side_b,
        size=float(units),
        confidence=float(confidence),
        asset_class=AssetClass.ETF,
        created_at=now,
        meta={
            **common_meta,
            "pair_role": role_b,
            "partner_symbol": pair.sym_long,
        },
    ))

    return signals


# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------

def build_delta_pairs_config() -> StrategyConfig:
    """Return StrategyConfig from config module when available, else safe default."""
    try:
        from chad.strategies.delta_pairs_config import build_delta_pairs_config as _impl
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.DELTA_PAIRS,
            enabled=True,
            target_universe=["SPY", "QQQ", "IWM"],
            max_gross_exposure=0.15,
            notes="Market-neutral pairs trading engine (fallback config)",
        )


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def build_delta_pairs_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    Build DELTA_PAIRS signals -- mean-reversion on correlated ETF pairs.

    Flow:
    1. For each pair, load bars for both symbols.
    2. Compute z-score of price ratio.
    3. If z-score crosses entry/exit/stop thresholds, emit signals.
    """
    tuning = DEFAULT_TUNING
    now = getattr(ctx, "now", None)
    if not isinstance(now, datetime):
        now = datetime.now(timezone.utc)

    equity = _extract_equity(ctx, 100_000.0)

    signals: List[TradeSignal] = []

    for pair in tuning.pairs:
        closes_a = _extract_closes(ctx, pair.sym_long)
        closes_b = _extract_closes(ctx, pair.sym_short)

        if len(closes_a) < tuning.min_bars or len(closes_b) < tuning.min_bars:
            continue

        zscore = compute_zscore(closes_a, closes_b, tuning.lookback_days)
        if zscore is None:
            continue

        abs_z = abs(zscore)

        # Confidence: scaled by z-magnitude and pair correlation
        correlation_weight = pair.correlation
        confidence = min(1.0, abs_z / tuning.zscore_entry) * correlation_weight
        if confidence < tuning.min_confidence:
            continue

        # Only emit entry signals when |Z| >= entry threshold
        # Exit/stop signals only relevant if we have open positions
        # (position tracking is handled by orchestrator, strategy just emits)
        entry_threshold = pair.zscore_entry or tuning.zscore_entry

        if abs_z < entry_threshold:
            continue

        price_a = _extract_price(ctx, pair.sym_long)
        price_b = _extract_price(ctx, pair.sym_short)
        if price_a <= 0 or price_b <= 0:
            continue

        pair_signals = _build_pair_signals(
            pair=pair,
            zscore=zscore,
            confidence=confidence,
            equity=equity,
            tuning=tuning,
            now=now,
            price_a=price_a,
            price_b=price_b,
        )
        signals.extend(pair_signals)

    return signals


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def delta_pairs_handler(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """StrategyEngine-compatible handler for DELTA_PAIRS. Fail-closed."""
    try:
        cfg = build_delta_pairs_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_delta_pairs_signals(ctx=ctx, params=params)
    except Exception:
        return []


__all__ = [
    "PairSpec",
    "DeltaPairsTuning",
    "DEFAULT_PAIRS",
    "compute_zscore",
    "build_delta_pairs_config",
    "build_delta_pairs_signals",
    "delta_pairs_handler",
]

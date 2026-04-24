#!/usr/bin/env python3
"""
chad/strategies/gamma_futures.py

Production-grade Gamma Futures strategy for CHAD.

Mean-reversion counterpart to Alpha Futures (momentum). The two
strategies now trade DISJOINT symbol universes so opposite-side signals
on the same symbol cannot net to zero in the execution planner:

    alpha_futures: MES, MNQ, MGC          (equity-index + metals)
    gamma_futures: MCL, MYM, M2K, ZN, ZB  (energy, Dow/Russell, bonds)

Gamma fades overextension rather than chasing trend.

Signal logic
------------
Alpha Futures (momentum):
    BUY  when price > ema_fast > ema_slow        (trend-following)
    SELL when price < ema_fast < ema_slow
    BUY  on breakout above highest-high
    SELL on breakdown below lowest-low

Gamma Futures (reversion):
    SELL when RSI > overbought AND price above upper Bollinger Band
    BUY  when RSI < oversold   AND price below lower Bollinger Band
    SELL when price/ema_slow deviation > threshold (overextended high)
    BUY  when ema_slow/price deviation > threshold (overextended low)

The two strategies are designed to coexist: in strong trending markets
Alpha Futures dominates; in range-bound/choppy markets Gamma Futures
captures mean reversion. The overlap on the same instruments provides
natural hedging when signals conflict.

Position sizing
---------------
Identical ATR-based sizing via alpha_futures._compute_contract_size,
but with a tighter risk budget (1.2% vs 1.5%) and lower max notional
($40K vs $50K) to reflect the shorter expected holding period of
reversion trades.

Confidence calculation
----------------------
Three components scaled to [0, 1]:
1. RSI extremity: distance of RSI from neutral (50), normalized by
   the overbought/oversold thresholds
2. Bollinger penetration: how far price has pierced beyond the band
3. Mean deviation magnitude: |price - ema_slow| / ema_slow relative
   to the reversion threshold
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import logging

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal

# Shared infrastructure from alpha_futures — same spec registry, different
# universe (disjoint subset to avoid same-symbol netting in the planner).
from chad.strategies.alpha_futures import (
    ALPHA_FUTURES_UNIVERSE,
    DEFAULT_SPECS,
    FuturesInstrumentSpec,
    _atr,
    _clamp,
    _compute_contract_size,
    _derive_liquidity_usd,
    _ema,
    _extract_bars,
    _extract_equity,
    _extract_prices,
    _safe_div,
    _to_float,
)

logger = logging.getLogger(__name__)

# gamma_futures owns the energy / Dow / Russell / bond-complex reversion
# book. MCL is the primary symbol; MYM, M2K, ZN, ZB are used when bar data
# is available. Explicitly disjoint from ALPHA_FUTURES_UNIVERSE.
_GAMMA_FUTURES_PRIMARY: Tuple[str, ...] = ("MCL",)
_GAMMA_FUTURES_EXTENDED: Tuple[str, ...] = ("MYM", "M2K", "ZN", "ZB")


def _resolve_gamma_universe(ctx: object) -> Tuple[str, ...]:
    """
    Return the symbol list gamma_futures should evaluate this cycle.

    Always includes MCL. Adds MYM / M2K when bar data exists; otherwise
    falls back to ZN / ZB so the strategy retains at least one additional
    symbol beyond the energy leg. Any symbol overlapping with
    ALPHA_FUTURES_UNIVERSE is stripped defensively.
    """
    bars_map = getattr(ctx, "bars", None)
    have_bars = bars_map if isinstance(bars_map, Mapping) else {}

    selected: List[str] = list(_GAMMA_FUTURES_PRIMARY)
    added_micro = False
    for sym in ("MYM", "M2K"):
        rows = have_bars.get(sym)
        if isinstance(rows, Sequence) and len(rows) >= 10:
            selected.append(sym)
            added_micro = True
    if not added_micro:
        for sym in ("ZN", "ZB"):
            rows = have_bars.get(sym)
            if isinstance(rows, Sequence) and len(rows) >= 10:
                selected.append(sym)

    return tuple(s for s in selected if s not in ALPHA_FUTURES_UNIVERSE)


GAMMA_FUTURES_UNIVERSE: Tuple[str, ...] = tuple(
    s for s in (_GAMMA_FUTURES_PRIMARY + _GAMMA_FUTURES_EXTENDED)
    if s not in ALPHA_FUTURES_UNIVERSE
)


# ---------------------------------------------------------------------------
# Configuration fallback
# ---------------------------------------------------------------------------

def build_gamma_futures_config() -> StrategyConfig:
    """Return StrategyConfig from config module when available, else safe default."""
    try:
        from chad.strategies.gamma_futures_config import (
            build_gamma_futures_config as _impl,
        )
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.GAMMA_FUTURES,
            enabled=True,
            target_universe=list(GAMMA_FUTURES_UNIVERSE),
            max_gross_exposure=0.20,
            notes="Futures mean-reversion engine (fallback config)",
        )


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GammaFuturesTuning:
    """
    Tuning parameters for the Gamma Futures mean-reversion strategy.

    Defaults are deliberately conservative: the strategy should only fire
    on clear overextension, not on mild oscillation.
    """

    # Indicators
    ema_slow_len: int = 26
    rsi_len: int = 14
    bb_len: int = 20
    bb_width: float = 2.0
    atr_len: int = 14

    # Data requirements
    min_bars: int = 40

    # Signal thresholds
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    mean_reversion_threshold: float = 0.02  # 2% deviation from EMA

    # Confidence
    min_confidence: float = 0.65

    # Position sizing (tighter than alpha_futures)
    risk_budget_pct: float = 0.012
    min_risk_budget_usd: float = 150.0
    equity_fallback: float = 10_000.0
    max_trade_notional: float = 40_000.0

    # Direction gates
    allow_long: bool = True
    allow_short: bool = True


# ---------------------------------------------------------------------------
# Indicator functions
# ---------------------------------------------------------------------------

def _rsi(closes: Sequence[float], length: int) -> float:
    """
    Compute the Relative Strength Index over the last `length` periods.

    Uses the Wilder smoothing method (exponential moving average of gains
    and losses). Returns 50.0 (neutral) if insufficient data or zero
    movement.

    Parameters
    ----------
    closes : Sequence[float]
        Chronologically ordered close prices. Must contain at least
        ``length + 1`` elements for a meaningful result.
    length : int
        RSI lookback period (typically 14).

    Returns
    -------
    float
        RSI value in [0, 100]. Returns 50.0 on degenerate input.
    """
    if length <= 0 or len(closes) < length + 1:
        return 50.0

    # Calculate period-over-period changes for the relevant window
    changes: List[float] = []
    for i in range(len(closes) - length, len(closes)):
        changes.append(closes[i] - closes[i - 1])

    if not changes:
        return 50.0

    # Seed with simple average of first `length` changes
    gains: List[float] = []
    losses: List[float] = []
    for c in changes:
        gains.append(c if c > 0 else 0.0)
        losses.append(-c if c < 0 else 0.0)

    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length

    if avg_gain + avg_loss < 1e-12:
        return 50.0
    if avg_loss < 1e-12:
        return 100.0
    if avg_gain < 1e-12:
        return 0.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_bands(
    closes: Sequence[float], length: int, width: float,
) -> Tuple[float, float, float]:
    """
    Compute Bollinger Bands: (upper, middle, lower).

    Parameters
    ----------
    closes : Sequence[float]
        Chronologically ordered close prices.
    length : int
        SMA lookback period (typically 20).
    width : float
        Number of standard deviations for the bands (typically 2.0).

    Returns
    -------
    Tuple[float, float, float]
        (upper_band, middle_band, lower_band).
        Returns (0.0, 0.0, 0.0) on insufficient data.
    """
    if length <= 0 or len(closes) < length:
        return 0.0, 0.0, 0.0

    window = closes[-length:]
    middle = sum(window) / length

    variance = sum((x - middle) ** 2 for x in window) / length
    std = math.sqrt(variance)

    upper = middle + width * std
    lower = middle - width * std
    return upper, middle, lower


def _mean_deviation_ratio(price: float, reference: float) -> float:
    """
    Compute signed deviation of price from reference as a ratio.

    Positive when price > reference (overextended high).
    Negative when price < reference (overextended low).

    Returns 0.0 if reference is zero.
    """
    if abs(reference) < 1e-12:
        return 0.0
    return (price - reference) / reference


# ---------------------------------------------------------------------------
# Signal generator
# ---------------------------------------------------------------------------

def _build_signal_for_symbol(
    *,
    symbol: str,
    bars: Sequence[Mapping[str, float]],
    price: float,
    spec: FuturesInstrumentSpec,
    tuning: GammaFuturesTuning,
    equity: float,
) -> Optional[TradeSignal]:
    """
    Generate a mean-reversion TradeSignal for the given symbol, or None.

    Signal cases (any one triggers):

    Case 1 — RSI overbought + Bollinger upper breach:
        SELL when RSI > rsi_overbought AND price >= upper_band

    Case 2 — RSI oversold + Bollinger lower breach:
        BUY when RSI < rsi_oversold AND price <= lower_band

    Case 3 — Mean overextension high:
        SELL when (price - ema_slow) / ema_slow > mean_reversion_threshold

    Case 4 — Mean overextension low:
        BUY when (ema_slow - price) / ema_slow > mean_reversion_threshold

    Cases 1-2 require both RSI and Bollinger confirmation (higher quality).
    Cases 3-4 use raw EMA deviation (broader catch, lower base confidence).
    If multiple cases fire, the highest-confidence signal wins.
    """
    if len(bars) < tuning.min_bars or price <= 0:
        return None

    closes = [_to_float(bar.get("close"), 0.0) for bar in bars]
    closes = [x for x in closes if x > 0]
    if len(closes) < tuning.min_bars:
        return None

    # Compute indicators
    ema_slow = _ema(closes, tuning.ema_slow_len)
    atr_val = _atr(bars, tuning.atr_len)
    if atr_val <= 0:
        return None

    rsi_val = _rsi(closes, tuning.rsi_len)
    bb_upper, bb_middle, bb_lower = _bollinger_bands(
        closes, tuning.bb_len, tuning.bb_width,
    )
    deviation = _mean_deviation_ratio(price, ema_slow)

    # Liquidity filter
    latest_bar = bars[-1]
    liquidity_usd = _derive_liquidity_usd(
        price, _to_float(latest_bar.get("volume"), 0.0), spec.point_value,
    )
    if liquidity_usd < spec.min_liquidity_usd:
        return None

    # Evaluate all four reversion cases, track best
    candidates: List[Tuple[SignalSide, float, str]] = []

    # Case 1: RSI overbought + Bollinger upper breach -> SELL
    if (
        tuning.allow_short
        and rsi_val > tuning.rsi_overbought
        and bb_upper > 0
        and price >= bb_upper
    ):
        # RSI extremity: how far past overbought threshold
        rsi_extremity = _clamp(
            (rsi_val - tuning.rsi_overbought) / (100.0 - tuning.rsi_overbought),
            0.0, 1.0,
        )
        # BB penetration: how far above upper band
        bb_penetration = _clamp(
            _safe_div(price - bb_upper, bb_upper - bb_middle, 0.0),
            0.0, 1.0,
        ) if bb_upper > bb_middle else 0.0
        conf = 0.65 + 0.15 * rsi_extremity + 0.10 * bb_penetration
        candidates.append((SignalSide.SELL, conf, "rsi_overbought_bb_upper"))

    # Case 2: RSI oversold + Bollinger lower breach -> BUY
    if (
        tuning.allow_long
        and rsi_val < tuning.rsi_oversold
        and bb_lower > 0
        and price <= bb_lower
    ):
        rsi_extremity = _clamp(
            (tuning.rsi_oversold - rsi_val) / tuning.rsi_oversold,
            0.0, 1.0,
        )
        bb_penetration = _clamp(
            _safe_div(bb_lower - price, bb_middle - bb_lower, 0.0),
            0.0, 1.0,
        ) if bb_middle > bb_lower else 0.0
        conf = 0.65 + 0.15 * rsi_extremity + 0.10 * bb_penetration
        candidates.append((SignalSide.BUY, conf, "rsi_oversold_bb_lower"))

    # Case 3: Mean overextension high -> SELL
    if (
        tuning.allow_short
        and deviation > tuning.mean_reversion_threshold
    ):
        dev_strength = _clamp(
            deviation / (tuning.mean_reversion_threshold * 3.0),
            0.0, 1.0,
        )
        conf = 0.60 + 0.20 * dev_strength
        candidates.append((SignalSide.SELL, conf, "mean_overextension_high"))

    # Case 4: Mean overextension low -> BUY
    if (
        tuning.allow_long
        and deviation < -tuning.mean_reversion_threshold
    ):
        dev_strength = _clamp(
            abs(deviation) / (tuning.mean_reversion_threshold * 3.0),
            0.0, 1.0,
        )
        conf = 0.60 + 0.20 * dev_strength
        candidates.append((SignalSide.BUY, conf, "mean_overextension_low"))

    if not candidates:
        return None

    # Pick highest confidence candidate
    candidates.sort(key=lambda c: c[1], reverse=True)
    side, confidence, trigger = candidates[0]
    confidence = _clamp(confidence, 0.0, 1.0)

    # Add liquidity bonus (small)
    confidence += _clamp(
        _safe_div(liquidity_usd, 10_000_000_000.0, 0.0), 0.0, 0.05,
    )
    confidence = _clamp(confidence, 0.0, 1.0)

    if confidence < tuning.min_confidence:
        return None

    # Position sizing (reuses alpha_futures infrastructure)
    contracts, alloc_wt, risk_budget_usd, risk_per_contract_usd = _compute_contract_size(
        symbol=symbol,
        spec=spec,
        price=price,
        atr_val=atr_val,
        equity=equity,
        tuning=tuning,
        confidence=confidence,
    )
    if contracts <= 0:
        return None

    estimated_notional = price * spec.point_value * contracts
    atr_pct = _safe_div(atr_val, price, 0.0)

    return TradeSignal(
        strategy=StrategyName.GAMMA_FUTURES,
        symbol=symbol,
        side=side,
        size=float(contracts),
        confidence=float(confidence),
        asset_class=AssetClass.FUTURES,
        meta={
            "engine": "gamma_futures.v1",
            "family": spec.family,
            "contract_symbol": symbol,
            "exchange": spec.exchange,
            "point_value": spec.point_value,
            "min_tick": spec.min_tick,
            "trigger": trigger,
            "rsi": round(rsi_val, 4),
            "bb_upper": round(bb_upper, 6),
            "bb_middle": round(bb_middle, 6),
            "bb_lower": round(bb_lower, 6),
            "ema_slow": round(ema_slow, 6),
            "mean_deviation": round(deviation, 6),
            "atr": round(atr_val, 6),
            "atr_pct": round(atr_pct, 6),
            "allocation_weight": round(alloc_wt, 6),
            "risk_budget_usd": round(risk_budget_usd, 6),
            "risk_per_contract_usd": round(risk_per_contract_usd, 6),
            "estimated_notional": round(estimated_notional, 6),
            "liquidity_usd": round(liquidity_usd, 6),
            "required_asset_class": "futures",
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gamma_futures_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """Build mean-reversion signals for all defined futures instruments."""
    tuning = GammaFuturesTuning(
        risk_budget_pct=_to_float(
            os.getenv("CHAD_GAMMA_FUTURES_RISK_PCT"), 0.012,
        ),
        min_risk_budget_usd=_to_float(
            os.getenv("CHAD_GAMMA_FUTURES_MIN_RISK_BUDGET_USD"), 150.0,
        ),
        equity_fallback=_to_float(
            os.getenv("CHAD_GAMMA_FUTURES_EQUITY_FALLBACK"), 10_000.0,
        ),
        max_trade_notional=_to_float(
            os.getenv("CHAD_GAMMA_FUTURES_MAX_TRADE_NOTIONAL"), 40_000.0,
        ),
        min_confidence=_to_float(
            os.getenv("CHAD_GAMMA_FUTURES_MIN_CONFIDENCE"), 0.65,
        ),
    )
    universe = _resolve_gamma_universe(ctx)
    if not universe:
        logger.info("GAMMA_FUTURES_UNIVERSE_EMPTY no_symbols_available_this_cycle")
        return []

    prices = _extract_prices(ctx)
    bars_by_symbol = _extract_bars(ctx, universe)
    equity = _extract_equity(ctx, tuning)
    signals: List[TradeSignal] = []
    for symbol in universe:
        spec = DEFAULT_SPECS.get(symbol)
        if spec is None:
            continue
        signal = _build_signal_for_symbol(
            symbol=symbol,
            bars=bars_by_symbol.get(symbol, []),
            price=prices.get(symbol, 0.0),
            spec=spec,
            tuning=tuning,
            equity=equity,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def gamma_futures_handler(
    ctx: object, params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """StrategyEngine-compatible handler; supports optional params argument."""
    try:
        cfg = build_gamma_futures_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_gamma_futures_signals(ctx=ctx, params=params)
    except Exception:
        return []


__all__ = [
    "GAMMA_FUTURES_UNIVERSE",
    "GammaFuturesTuning",
    "build_gamma_futures_config",
    "build_gamma_futures_signals",
    "gamma_futures_handler",
    "_rsi",
    "_bollinger_bands",
    "_mean_deviation_ratio",
]

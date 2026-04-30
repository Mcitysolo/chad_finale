#!/usr/bin/env python3
"""
chad/strategies/gamma_reversion.py

GAMMA_REVERSION — ETF Mean Reversion Strategy for CHAD.

Fades overextended moves on liquid ETFs using multi-indicator confluence:
RSI + Bollinger Bands + Z-score + Rate of Change. Both long and short.

Instruments
-----------
- SPY : S&P 500 ETF
- QQQ : Nasdaq 100 ETF
- IWM : Russell 2000 ETF (removed: negative backtest Sharpe)
- GLD : Gold ETF
- TLT : 20+ Year Treasury ETF

Signal Logic
------------
Entry (requires 3/3 confluence):
  SHORT: RSI > 72 AND (price > BB_upper OR zscore > 1.8) AND ROC > 0
  LONG:  RSI < 28 AND (price < BB_lower OR zscore < -1.8) AND ROC < 0

Exit:
  Target: price crosses SMA-20 (mean reversion achieved)
  Stop:   2.5 * ATR from entry
  Time:   15 bars max hold

Complementary to GAMMA
-----------------------
GAMMA is a swing/momentum engine (trend continuation + EMA-based reversion).
GAMMA_REVERSION uses statistical indicators (RSI, Bollinger, Z-score) and
trades both directions. No overlap in signal logic.

Design
------
- Strategy-only: emits TradeSignal intents, never executes.
- Deterministic, no I/O (except env fallbacks in config).
- Fail-closed: insufficient data -> no signals.
- Both long AND short.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GammaReversionTuning:
    """Tuning parameters for GAMMA_REVERSION strategy."""
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 72.0
    rsi_oversold: float = 28.0

    # Bollinger Bands
    bb_period: int = 20
    bb_width: float = 2.0

    # Z-score
    zscore_period: int = 20
    zscore_threshold: float = 1.8

    # Rate of change
    roc_period: int = 5

    # ATR
    atr_period: int = 14
    atr_stop_mult: float = 2.5

    # Timing
    time_stop_bars: int = 15
    min_bars: int = 40

    # Confidence
    min_confidence: float = 0.60

    # Sizing
    base_size: float = 5.0
    max_size: float = 15.0

    # GLD strict confluence: require 3/3 indicators for GLD
    # (backtest showed 38% win rate with 2/3 confluence)
    gld_strict_confluence: bool = True


DEFAULT_TUNING = GammaReversionTuning()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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


def _bar_field(bar: Any, key: str) -> float:
    if isinstance(bar, dict):
        return _safe_float(bar.get(key), 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _rsi(closes: Sequence[float], period: int = 14) -> List[float]:
    """RSI using Wilder's smoothing. Warmup values set to 50.0."""
    n = len(closes)
    out = [50.0] * n
    if n < period + 1:
        return out

    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, n):
        if avg_loss < 1e-12:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

        if i < n - 1:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return out


def _sma(values: Sequence[float], period: int) -> List[float]:
    """Simple moving average with expanding window warmup."""
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        window = values[start:i + 1]
        out.append(sum(window) / len(window))
    return out


def _std(values: Sequence[float], period: int) -> List[float]:
    """Rolling standard deviation with expanding window warmup."""
    sma_vals = _sma(values, period)
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        window = values[start:i + 1]
        mean = sma_vals[i]
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        out.append(math.sqrt(variance))
    return out


def _bollinger(
    closes: Sequence[float],
    period: int = 20,
    width: float = 2.0,
) -> tuple[list[float], list[float], list[float]]:
    """Bollinger Bands: (upper, middle, lower)."""
    middle = _sma(closes, period)
    sd = _std(closes, period)
    upper = [m + width * s for m, s in zip(middle, sd)]
    lower = [m - width * s for m, s in zip(middle, sd)]
    return upper, middle, lower


def _zscore(closes: Sequence[float], period: int = 20) -> List[float]:
    """Z-score of close vs SMA(period)."""
    sma_vals = _sma(closes, period)
    sd = _std(closes, period)
    out: List[float] = []
    for i in range(len(closes)):
        if sd[i] < 1e-10:
            out.append(0.0)
        else:
            out.append((closes[i] - sma_vals[i]) / sd[i])
    return out


def _roc(closes: Sequence[float], period: int = 5) -> List[float]:
    """Rate of change: (close - close[n-period]) / close[n-period]."""
    out: List[float] = []
    for i in range(len(closes)):
        if i < period or closes[i - period] <= 0:
            out.append(0.0)
        else:
            out.append((closes[i] - closes[i - period]) / closes[i - period])
    return out


def _atr_series(bars: Sequence[Mapping[str, Any]], period: int = 14) -> List[float]:
    """ATR series aligned to bars."""
    n = len(bars)
    trs: List[float] = []
    for i in range(n):
        h = _bar_field(bars[i], "high")
        l = _bar_field(bars[i], "low")
        if i == 0:
            trs.append(h - l if h > l else 0.0)
        else:
            prev_c = _bar_field(bars[i - 1], "close")
            trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    out: List[float] = []
    for i in range(n):
        start = max(0, i - period + 1)
        window = trs[start:i + 1]
        out.append(sum(window) / len(window))
    return out


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _compute_confidence(
    rsi_val: float,
    zscore_val: float,
    side: SignalSide,
    tuning: GammaReversionTuning,
) -> float:
    """
    Compute signal confidence based on indicator extremity.

    Base: 0.55
    +0.10 if RSI is very extreme (>80 for short, <20 for long)
    +0.10 if Z-score magnitude > 2.2
    +0.05 if Z-score magnitude > 2.5
    Clamped to [0.0, 1.0].
    """
    conf = 0.55

    if side == SignalSide.SELL:
        if rsi_val > 80.0:
            conf += 0.10
        if zscore_val > 2.2:
            conf += 0.10
        if zscore_val > 2.5:
            conf += 0.05
    else:
        if rsi_val < 20.0:
            conf += 0.10
        if zscore_val < -2.2:
            conf += 0.10
        if zscore_val < -2.5:
            conf += 0.05

    return _clamp(conf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def _compute_size(
    confidence: float,
    zscore_mag: float,
    tuning: GammaReversionTuning,
) -> float:
    """
    Size based on confidence and Z-score magnitude.

    Higher confidence and more extreme deviation -> larger size.
    """
    scale = 1.0 + 0.5 * max(0.0, zscore_mag - 1.5)
    size = tuning.base_size * scale * (confidence / 0.7)
    return _clamp(size, 1.0, tuning.max_size)


# ---------------------------------------------------------------------------
# Per-symbol signal generator
# ---------------------------------------------------------------------------

def _build_signal_for_symbol(
    *,
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    price: float,
    tuning: GammaReversionTuning,
    now: datetime,
) -> Optional[TradeSignal]:
    """
    Generate a reversion TradeSignal for the given symbol, or None.

    Entry requires 3/3 confluence:
      SHORT: RSI > overbought AND (price > BB_upper OR zscore > threshold) AND ROC > 0
      LONG:  RSI < oversold   AND (price < BB_lower OR zscore < -threshold) AND ROC < 0
    """
    if len(bars) < tuning.min_bars or price <= 0:
        return None

    closes: List[float] = []
    cleaned_bars: List[Mapping[str, Any]] = []
    for b in bars:
        c = _bar_field(b, "close")
        h = _bar_field(b, "high")
        l = _bar_field(b, "low")
        if c <= 0 or h <= 0 or l <= 0 or h < l:
            continue
        closes.append(c)
        cleaned_bars.append(b)

    if len(closes) < tuning.min_bars:
        return None

    # Compute indicators on full history
    rsi_vals = _rsi(closes, tuning.rsi_period)
    bb_upper, bb_middle, bb_lower = _bollinger(closes, tuning.bb_period, tuning.bb_width)
    zs_vals = _zscore(closes, tuning.zscore_period)
    roc_vals = _roc(closes, tuning.roc_period)
    atr_vals = _atr_series(cleaned_bars, tuning.atr_period)

    # Use latest values
    idx = len(closes) - 1
    rsi_v = rsi_vals[idx]
    zs_v = zs_vals[idx]
    roc_v = roc_vals[idx]
    atr_v = atr_vals[idx]
    sma20 = bb_middle[idx]

    # GLD strict confluence: require BOTH Bollinger AND Z-score (not OR)
    strict = tuning.gld_strict_confluence and symbol.upper() == "GLD"

    # Check SHORT confluence (3/3 required)
    short_confluence = 0
    if rsi_v > tuning.rsi_overbought:
        short_confluence += 1
    if strict:
        # GLD strict: require both Bollinger AND Z-score
        if price > bb_upper[idx] and zs_v > tuning.zscore_threshold:
            short_confluence += 1
    else:
        if price > bb_upper[idx] or zs_v > tuning.zscore_threshold:
            short_confluence += 1
    if roc_v > 0:
        short_confluence += 1

    # Check LONG confluence (3/3 required)
    long_confluence = 0
    if rsi_v < tuning.rsi_oversold:
        long_confluence += 1
    if strict:
        # GLD strict: require both Bollinger AND Z-score
        if price < bb_lower[idx] and zs_v < -tuning.zscore_threshold:
            long_confluence += 1
    else:
        if price < bb_lower[idx] or zs_v < -tuning.zscore_threshold:
            long_confluence += 1
    if roc_v < 0:
        long_confluence += 1

    side: Optional[SignalSide] = None
    if short_confluence >= 3:
        side = SignalSide.SELL
    elif long_confluence >= 3:
        side = SignalSide.BUY

    if side is None:
        LOG.debug(
            "gamma_reversion: sym=%s no 3/3 confluence "
            "(rsi=%.1f z=%.2f roc=%.4f) — no signal",
            symbol, rsi_v, zs_v, roc_v,
        )
        return None

    confidence = _compute_confidence(rsi_v, zs_v, side, tuning)
    if confidence < tuning.min_confidence:
        return None

    size = _compute_size(confidence, abs(zs_v), tuning)

    # Compute stop and target for meta
    if side == SignalSide.SELL:
        stop_price = price + tuning.atr_stop_mult * atr_v
    else:
        stop_price = price - tuning.atr_stop_mult * atr_v

    return TradeSignal(
        strategy=StrategyName.GAMMA_REVERSION,
        symbol=symbol,
        side=side,
        size=float(size),
        confidence=float(confidence),
        asset_class=AssetClass.ETF,
        created_at=now,
        meta={
            "engine": "gamma_reversion.v1",
            "rsi": round(rsi_v, 4),
            "zscore": round(zs_v, 4),
            "roc": round(roc_v, 6),
            "bb_upper": round(bb_upper[idx], 4),
            "bb_lower": round(bb_lower[idx], 4),
            "sma20": round(sma20, 4),
            "atr": round(atr_v, 4),
            "stop_price": round(stop_price, 4),
            "target_price": round(sma20, 4),  # mean reversion target
            "confluence": short_confluence if side == SignalSide.SELL else long_confluence,
        },
    )


# ---------------------------------------------------------------------------
# Configuration fallback
# ---------------------------------------------------------------------------

def build_gamma_reversion_config() -> StrategyConfig:
    """Return StrategyConfig from config module when available, else safe default."""
    try:
        from chad.strategies.gamma_reversion_config import build_gamma_reversion_config as _impl
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.GAMMA_REVERSION,
            enabled=True,
            target_universe=["SPY", "QQQ", "GLD", "TLT"],
            max_gross_exposure=0.18,
            notes="ETF mean reversion engine (fallback config)",
        )


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def build_gamma_reversion_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    Build GAMMA_REVERSION signals for all target ETFs.

    Reads bars from ctx.bars, latest price from ctx.ticks or ctx.bars.
    Returns valid signals or empty list.
    """
    tuning = GammaReversionTuning()
    now = getattr(ctx, "now", None)
    if not isinstance(now, datetime):
        now = datetime.now(timezone.utc)

    bars_map = _get_mapping(ctx, "bars")
    ticks_map = _get_mapping(ctx, "ticks")

    cfg = build_gamma_reversion_config()
    universe = list(cfg.target_universe or ["SPY", "QQQ", "GLD", "TLT"])

    signals: List[TradeSignal] = []
    for symbol in universe:
        sym = symbol.strip().upper()

        # Get bars
        raw_bars = bars_map.get(sym, [])
        if not isinstance(raw_bars, (list, tuple)):
            continue

        # Get latest price from ticks or last bar
        price = 0.0
        tick = ticks_map.get(sym)
        if tick is not None:
            price = _safe_float(getattr(tick, "price", None), 0.0)
            if price <= 0 and isinstance(tick, dict):
                price = _safe_float(tick.get("price"), 0.0)
        if price <= 0 and raw_bars:
            last_bar = raw_bars[-1]
            if isinstance(last_bar, dict):
                price = _safe_float(last_bar.get("close"), 0.0)

        signal = _build_signal_for_symbol(
            symbol=sym,
            bars=raw_bars,
            price=price,
            tuning=tuning,
            now=now,
        )
        if signal is not None:
            signals.append(signal)

    if not signals:
        LOG.info(
            "gamma_reversion: zero signals — "
            "no 3/3 confluence on any of %d universe symbols "
            "(regime=%s vix=%.2f)",
            len(universe),
            getattr(ctx, "regime", "?"),
            getattr(ctx, "vix", 0.0),
        )
    return signals


# ---------------------------------------------------------------------------
# Engine-compatible handler
# ---------------------------------------------------------------------------

def gamma_reversion_handler(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    StrategyEngine-compatible handler for GAMMA_REVERSION.

    Fail-closed on any exception.
    """
    try:
        cfg = build_gamma_reversion_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_gamma_reversion_signals(ctx=ctx, params=params)
    except Exception:
        return []


__all__ = [
    "GammaReversionTuning",
    "build_gamma_reversion_config",
    "build_gamma_reversion_signals",
    "gamma_reversion_handler",
    "_rsi",
    "_bollinger",
    "_zscore",
    "_roc",
    "_build_signal_for_symbol",
    "_compute_confidence",
]

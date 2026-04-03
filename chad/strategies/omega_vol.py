#!/usr/bin/env python3
"""
chad/strategies/omega_vol.py

OMEGA_VOL — Volatility Regime Alpha Strategy for CHAD.

Takes directional positions on volatility itself via VIX-linked ETPs.
Complements OMEGA (crash insurance via inverse equity ETFs) by exploiting
the structural properties of VIX products:

- Short vol (SVXY): profits from contango roll yield in calm markets.
- Long vol (UVXY): profits from VIX spikes in crises.

OMEGA vs OMEGA_VOL
------------------
OMEGA:     Insurance. Always long inverse equity ETFs when danger detected.
           Instruments: SH, PSQ. Trigger: 2+ danger sensors. Binary hedge.

OMEGA_VOL: Alpha. Directional vol bets based on 5-state regime classifier.
           Instruments: SVXY, UVXY. Both long AND short vol.
           Active in most regimes (except ELEVATED where direction is ambiguous).

Decay Warning
-------------
- UVXY loses ~50-70% per year in contango. Long UVXY positions must be
  short-duration and high-confidence only. Time stop: 10 bars.
- SVXY has positive structural drift but can crash -50% in a single day
  during a VIX spike. Sizing is capped at 3% of equity.

Signal Model
------------
Three-factor:
  1. VIX Level → regime classification
  2. VIX Momentum (5-day ROC) → direction confirmation
  3. VIX Z-score (vs 20-day SMA) → extremity / mean-reversion signal

Fail-closed: if VIX data unavailable, no signals emitted.

Design
------
- Strategy-only: emits TradeSignal intents, never executes.
- Deterministic given inputs, no I/O.
- Requires VIX data in context (ctx.vix or ctx.bars["VIX"]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from chad.types import (
    AssetClass,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# VolRegime enum
# ---------------------------------------------------------------------------

class VolRegime(str, Enum):
    """
    Five-state volatility regime classification.

    LOW_VOL:      VIX < 15 — complacency, steep contango, short vol opportunity.
    NORMAL_VOL:   15 <= VIX < 22 — typical market, mild short vol.
    ELEVATED_VOL: 22 <= VIX < 30 — uncertainty, ambiguous direction.
    CRISIS_VOL:   VIX >= 30 — spike/crash, long vol momentum.
    VOL_CRUSH:    VIX falling >20% from recent peak — post-spike reversion.
    """
    LOW_VOL = "low_vol"
    NORMAL_VOL = "normal_vol"
    ELEVATED_VOL = "elevated_vol"
    CRISIS_VOL = "crisis_vol"
    VOL_CRUSH = "vol_crush"


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OmegaVolTuning:
    """Tuning parameters for OMEGA_VOL strategy."""
    # Regime thresholds
    vix_low: float = 15.0
    vix_normal_high: float = 22.0
    vix_crisis: float = 30.0
    vol_crush_pct: float = 0.20

    # VIX indicators
    vix_momentum_period: int = 5
    vix_zscore_period: int = 20
    vix_zscore_extreme: float = 2.0
    vix_zscore_caution: float = 1.5

    # Risk controls
    atr_len: int = 14
    atr_stop_multiple: float = 2.0
    time_stop_bars: int = 10
    max_position_pct: float = 0.03  # 3% of equity per instrument
    min_confidence: float = 0.65

    # Sizing
    base_units: int = 3
    max_units: int = 8
    equity_fallback: float = 100_000.0

    # Universe
    short_vol_symbol: str = "SVXY"
    long_vol_symbol: str = "UVXY"


DEFAULT_TUNING = OmegaVolTuning()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
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


# ---------------------------------------------------------------------------
# VIX data extraction
# ---------------------------------------------------------------------------

def _extract_vix_current(ctx: Any) -> Optional[float]:
    """
    Extract current VIX level from context.

    Searches: ctx.vix, ctx.vol_index, ctx.volatility_index,
    ctx.market_data["VIX"], price_cache.
    Returns None if unavailable (fail-closed).
    """
    for attr in ("vix", "vol_index", "volatility_index"):
        raw = getattr(ctx, attr, None)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            v = _safe_float(raw)
            if v > 0:
                return v
        elif isinstance(raw, dict):
            for key in ("value", "last", "close"):
                if key in raw:
                    v = _safe_float(raw[key])
                    if v > 0:
                        return v

    # Try market_data dict
    md = _get_mapping(ctx, "market_data")
    vix_raw = md.get("VIX") or md.get("vix")
    if vix_raw is not None:
        v = _safe_float(vix_raw)
        if v > 0:
            return v

    return None


def _extract_vix_history(ctx: Any, min_bars: int = 20) -> Optional[List[float]]:
    """
    Extract VIX history from bars data.

    Looks for VIX bars in ctx.bars or ctx.vix_history.
    Returns list of VIX close values, or None if insufficient data.
    """
    # Try ctx.vix_history directly
    hist = getattr(ctx, "vix_history", None)
    if isinstance(hist, (list, tuple)) and len(hist) >= min_bars:
        values = [_safe_float(v) for v in hist if _safe_float(v) > 0]
        if len(values) >= min_bars:
            return values

    # Try bars["VIX"]
    bars = _get_mapping(ctx, "bars").get("VIX")
    if isinstance(bars, (list, tuple)) and len(bars) >= min_bars:
        values = []
        for b in bars:
            if isinstance(b, dict):
                c = _safe_float(b.get("close"), 0.0)
                if c > 0:
                    values.append(c)
            elif isinstance(b, (int, float)):
                v = _safe_float(b)
                if v > 0:
                    values.append(v)
        if len(values) >= min_bars:
            return values

    return None


def _extract_equity(ctx: Any, fallback: float = 100_000.0) -> float:
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


def _extract_price(ctx: Any, symbol: str) -> float:
    ticks = _get_mapping(ctx, "ticks")
    tick = ticks.get(symbol)
    if tick is not None:
        p = _safe_float(getattr(tick, "price", None), 0.0)
        if p > 0:
            return p
        if isinstance(tick, dict):
            p = _safe_float(tick.get("price"), 0.0)
            if p > 0:
                return p
    prices = _get_mapping(ctx, "prices")
    p = _safe_float(prices.get(symbol), 0.0)
    if p > 0:
        return p
    bars = _get_mapping(ctx, "bars").get(symbol)
    if isinstance(bars, (list, tuple)) and bars:
        last = bars[-1]
        if isinstance(last, dict):
            return _safe_float(last.get("close"), 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# VIX indicators
# ---------------------------------------------------------------------------

def _vix_momentum(vix_history: List[float], period: int = 5) -> Optional[float]:
    """
    Compute VIX rate of change over `period` bars.

    Returns fractional change (e.g., 0.15 = VIX up 15%).
    """
    if len(vix_history) < period + 1:
        return None
    current = vix_history[-1]
    past = vix_history[-(period + 1)]
    if past <= 0:
        return None
    return (current - past) / past


def _vix_zscore(vix_history: List[float], period: int = 20) -> Optional[float]:
    """
    Compute Z-score of current VIX vs its `period`-bar SMA.
    """
    if len(vix_history) < period:
        return None
    window = vix_history[-period:]
    mean = sum(window) / len(window)
    variance = sum((v - mean) ** 2 for v in window) / len(window)
    std = math.sqrt(variance) if variance > 0 else 0.0
    if std < 1e-10:
        return 0.0
    return (vix_history[-1] - mean) / std


def _vix_recent_peak(vix_history: List[float], lookback: int = 5) -> float:
    """Return the peak VIX value in the last `lookback` bars."""
    if len(vix_history) < lookback:
        return max(vix_history) if vix_history else 0.0
    return max(vix_history[-lookback:])


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

def classify_vol_regime(
    vix_current: float,
    vix_history: List[float],
    tuning: OmegaVolTuning = DEFAULT_TUNING,
) -> VolRegime:
    """
    Classify the current volatility regime from VIX data.

    Classification rules (evaluated in priority order):

    1. VOL_CRUSH: VIX has fallen >20% from 5-bar peak AND VIX < 30.
       Post-spike normalization — high conviction short vol.

    2. CRISIS_VOL: VIX >= 30.
       Active crisis, vol spike in progress.

    3. ELEVATED_VOL: VIX >= 22.
       Uncertainty, ambiguous direction.

    4. LOW_VOL: VIX < 15.
       Complacency, steep contango.

    5. NORMAL_VOL: Everything else (15 <= VIX < 22).

    Parameters
    ----------
    vix_current : float
        Current VIX level.
    vix_history : List[float]
        Recent VIX values (at least 5 for peak detection).
    tuning : OmegaVolTuning
        Regime thresholds.

    Returns
    -------
    VolRegime
        Classified volatility regime.
    """
    # VOL_CRUSH: VIX falling sharply from recent peak
    if len(vix_history) >= 5:
        peak = _vix_recent_peak(vix_history, lookback=5)
        if peak > 0 and vix_current < peak * (1.0 - tuning.vol_crush_pct) and vix_current < tuning.vix_crisis:
            return VolRegime.VOL_CRUSH

    # CRISIS_VOL
    if vix_current >= tuning.vix_crisis:
        return VolRegime.CRISIS_VOL

    # ELEVATED_VOL
    if vix_current >= tuning.vix_normal_high:
        return VolRegime.ELEVATED_VOL

    # LOW_VOL
    if vix_current < tuning.vix_low:
        return VolRegime.LOW_VOL

    # NORMAL_VOL
    return VolRegime.NORMAL_VOL


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _compute_confidence(
    regime: VolRegime,
    zscore: Optional[float],
    momentum: Optional[float],
    tuning: OmegaVolTuning,
) -> float:
    """
    Compute signal confidence based on regime and indicator extremity.

    Base confidence per regime:
      LOW_VOL:    0.75  (well-understood structural edge)
      NORMAL_VOL: 0.65  (mild edge)
      VOL_CRUSH:  0.80  (highest conviction — mean reversion)
      CRISIS_VOL: 0.70  (spike momentum)

    Bonuses:
      +0.05 if Z-score confirms extremity (>2.0 for crisis, <-1.5 for low)
      +0.05 if momentum strongly aligns
    """
    if regime == VolRegime.LOW_VOL:
        conf = 0.75
        if zscore is not None and zscore < -tuning.vix_zscore_caution:
            conf += 0.05
    elif regime == VolRegime.NORMAL_VOL:
        conf = 0.65
    elif regime == VolRegime.VOL_CRUSH:
        conf = 0.80
        if momentum is not None and momentum < -0.10:
            conf += 0.05
    elif regime == VolRegime.CRISIS_VOL:
        conf = 0.70
        if zscore is not None and zscore > tuning.vix_zscore_extreme:
            conf += 0.05
        if momentum is not None and momentum > 0.15:
            conf += 0.05
    else:
        conf = 0.50  # ELEVATED — should not generate signals

    return _clamp(conf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def _compute_units(
    confidence: float,
    equity: float,
    price: float,
    is_long_vol: bool,
    tuning: OmegaVolTuning,
) -> int:
    """
    Compute position size in units (shares).

    Confidence-scaled between base_units and max_units.
    Capped by max_position_pct of equity / price.
    UVXY (long vol) positions halved due to decay risk.
    """
    raw = tuning.base_units + int(confidence * (tuning.max_units - tuning.base_units))

    # Cap by equity allocation
    if price > 0:
        max_by_equity = int((equity * tuning.max_position_pct) / price)
        raw = min(raw, max_by_equity)

    # Halve UVXY positions (structural decay risk)
    if is_long_vol:
        raw = max(1, raw // 2)

    return max(0, raw)


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------

def _make_signal(
    *,
    symbol: str,
    side: SignalSide,
    size: float,
    confidence: float,
    regime: VolRegime,
    now: datetime,
    meta_extra: Dict[str, Any],
) -> TradeSignal:
    return TradeSignal(
        strategy=StrategyName.OMEGA_VOL,
        symbol=symbol,
        side=side,
        size=size,
        confidence=confidence,
        asset_class=AssetClass.ETF,
        created_at=now,
        meta={
            "engine": "omega_vol.v1",
            "regime": regime.value,
            **meta_extra,
        },
    )


# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------

def build_omega_vol_config() -> StrategyConfig:
    try:
        from chad.strategies.omega_vol_config import build_omega_vol_config as _impl
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.OMEGA_VOL,
            enabled=True,
            target_universe=["SVXY", "UVXY"],
            max_gross_exposure=0.06,
            notes="Volatility regime alpha engine (fallback config)",
        )


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def build_omega_vol_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    Build OMEGA_VOL signals based on volatility regime classification.

    Flow:
    1. Extract VIX current + history from context.
    2. Classify vol regime (5 states).
    3. Compute VIX momentum and Z-score.
    4. Map regime to instrument + direction.
    5. Size and emit signal.

    Fail-closed: no VIX data -> no signals.
    """
    tuning = DEFAULT_TUNING
    now = getattr(ctx, "now", None)
    if not isinstance(now, datetime):
        now = datetime.now(timezone.utc)

    # Extract VIX
    vix_current = _extract_vix_current(ctx)
    if vix_current is None:
        return []

    vix_history = _extract_vix_history(ctx, min_bars=tuning.vix_zscore_period)

    # If no history, build minimal from current
    if vix_history is None:
        # Cannot compute z-score or momentum — only regime from level
        vix_history = [vix_current]

    # Ensure current is last element
    if vix_history[-1] != vix_current:
        vix_history = list(vix_history) + [vix_current]

    # Classify regime
    regime = classify_vol_regime(vix_current, vix_history, tuning)

    # Compute indicators
    momentum = _vix_momentum(vix_history, tuning.vix_momentum_period)
    zscore = _vix_zscore(vix_history, tuning.vix_zscore_period)

    # Determine signal based on regime + momentum
    equity = _extract_equity(ctx, tuning.equity_fallback)

    signals: List[TradeSignal] = []

    if regime == VolRegime.LOW_VOL:
        # Short vol via SVXY — but only if momentum confirms (not rising)
        if momentum is None or momentum <= 0:
            conf = _compute_confidence(regime, zscore, momentum, tuning)
            if conf >= tuning.min_confidence:
                price = _extract_price(ctx, tuning.short_vol_symbol)
                units = _compute_units(conf, equity, price, is_long_vol=False, tuning=tuning)
                if units > 0:
                    signals.append(_make_signal(
                        symbol=tuning.short_vol_symbol,
                        side=SignalSide.BUY,
                        size=float(units),
                        confidence=conf,
                        regime=regime,
                        now=now,
                        meta_extra={
                            "vix": vix_current,
                            "vix_momentum": round(momentum, 6) if momentum is not None else None,
                            "vix_zscore": round(zscore, 4) if zscore is not None else None,
                            "position_type": "short_vol",
                            "reason": "low_vol_contango_harvest",
                        },
                    ))

    elif regime == VolRegime.NORMAL_VOL:
        # Mild short vol — only if VIX momentum is negative (contracting)
        if momentum is not None and momentum < 0:
            conf = _compute_confidence(regime, zscore, momentum, tuning)
            if conf >= tuning.min_confidence:
                price = _extract_price(ctx, tuning.short_vol_symbol)
                units = _compute_units(conf, equity, price, is_long_vol=False, tuning=tuning)
                if units > 0:
                    signals.append(_make_signal(
                        symbol=tuning.short_vol_symbol,
                        side=SignalSide.BUY,
                        size=float(units),
                        confidence=conf,
                        regime=regime,
                        now=now,
                        meta_extra={
                            "vix": vix_current,
                            "vix_momentum": round(momentum, 6),
                            "vix_zscore": round(zscore, 4) if zscore is not None else None,
                            "position_type": "short_vol",
                            "reason": "normal_vol_mild_short",
                        },
                    ))

    elif regime == VolRegime.ELEVATED_VOL:
        # NO SIGNAL — ambiguous regime, direction unclear
        pass

    elif regime == VolRegime.CRISIS_VOL:
        # Long vol via UVXY — chase the spike if momentum confirms
        if momentum is not None and momentum > 0:
            conf = _compute_confidence(regime, zscore, momentum, tuning)
            if conf >= tuning.min_confidence:
                price = _extract_price(ctx, tuning.long_vol_symbol)
                units = _compute_units(conf, equity, price, is_long_vol=True, tuning=tuning)
                if units > 0:
                    signals.append(_make_signal(
                        symbol=tuning.long_vol_symbol,
                        side=SignalSide.BUY,
                        size=float(units),
                        confidence=conf,
                        regime=regime,
                        now=now,
                        meta_extra={
                            "vix": vix_current,
                            "vix_momentum": round(momentum, 6),
                            "vix_zscore": round(zscore, 4) if zscore is not None else None,
                            "position_type": "long_vol",
                            "reason": "crisis_vol_spike_chase",
                        },
                    ))

    elif regime == VolRegime.VOL_CRUSH:
        # Short vol via SVXY — highest conviction (post-spike reversion)
        conf = _compute_confidence(regime, zscore, momentum, tuning)
        if conf >= tuning.min_confidence:
            price = _extract_price(ctx, tuning.short_vol_symbol)
            units = _compute_units(conf, equity, price, is_long_vol=False, tuning=tuning)
            if units > 0:
                signals.append(_make_signal(
                    symbol=tuning.short_vol_symbol,
                    side=SignalSide.BUY,
                    size=float(units),
                    confidence=conf,
                    regime=regime,
                    now=now,
                    meta_extra={
                        "vix": vix_current,
                        "vix_momentum": round(momentum, 6) if momentum is not None else None,
                        "vix_zscore": round(zscore, 4) if zscore is not None else None,
                        "position_type": "short_vol",
                        "reason": "vol_crush_reversion",
                    },
                ))

    return signals


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def omega_vol_handler(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """StrategyEngine-compatible handler for OMEGA_VOL. Fail-closed."""
    try:
        cfg = build_omega_vol_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_omega_vol_signals(ctx=ctx, params=params)
    except Exception:
        return []


__all__ = [
    "VolRegime",
    "OmegaVolTuning",
    "classify_vol_regime",
    "build_omega_vol_config",
    "build_omega_vol_signals",
    "omega_vol_handler",
]

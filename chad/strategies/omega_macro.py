#!/usr/bin/env python3
"""
chad/strategies/omega_macro.py

OMEGA_MACRO — Macro Regime Futures Strategy for CHAD.

Trades macro futures instruments (bonds, FX, metals) based on a 4-state
macro regime classifier. Complements OMEGA (VIX-aware ETF hedging) by
adding futures-based directional macro exposure.

Instruments
-----------
- ZN  : 10-Year Treasury Note (CBOT)  — interest rate / duration exposure
- ZB  : 30-Year Treasury Bond (CBOT)  — long-duration rate exposure
- M6E : Micro Euro FX (CME)           — USD strength / global risk proxy
- SIL : Micro Silver (COMEX)          — industrial metal / inflation proxy

Signal Logic
------------
Regime-driven directional positioning. Each instrument has a defined
side per regime:

    RISK_OFF:     ZN BUY, ZB BUY, M6E SELL, SIL SELL
    RISK_ON:      ZN SELL, ZB SELL, M6E BUY, SIL BUY
    STAGFLATION:  ZN BUY, ZB BUY, M6E SELL, SIL BUY
    NEUTRAL:      No signals (flat)

Sizing
------
ATR-based position sizing via alpha_futures._compute_contract_size,
with OMEGA_MACRO's own risk budget (1.0% vs alpha's 1.5%).

Design
------
- Strategy-only: emits TradeSignal intents, never executes.
- Deterministic, no I/O (except env fallbacks in config).
- Fail-closed: insufficient data -> no signals.
- Bounded: max contract limits per instrument.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal

from chad.strategies.alpha_futures import (
    FuturesInstrumentSpec,
    StrategyTuning,
    _atr,
    _clamp,
    _compute_contract_size,
    _extract_bars,
    _extract_equity,
    _extract_prices,
    _safe_div,
    _to_float,
)

from chad.strategies.macro_sensors import (
    MacroRegime,
    _atr_pct_from_bars,
    _ema_slope,
    _portfolio_drawdown_pct,
    _vix_value,
    classify_macro_regime,
)


# ---------------------------------------------------------------------------
# Instrument specs for OMEGA_MACRO universe
# ---------------------------------------------------------------------------

OMEGA_MACRO_SPECS: Dict[str, FuturesInstrumentSpec] = {
    "ZN": FuturesInstrumentSpec(
        symbol="ZN",
        family="ZN",
        exchange="CBOT",
        point_value=1000.0,
        min_tick=0.015625,
        max_contracts=3,
    ),
    "ZB": FuturesInstrumentSpec(
        symbol="ZB",
        family="ZB",
        exchange="CBOT",
        point_value=1000.0,
        min_tick=0.03125,
        max_contracts=2,
    ),
    "M6E": FuturesInstrumentSpec(
        symbol="M6E",
        family="6E",
        exchange="CME",
        point_value=12500.0,
        min_tick=0.0001,
        max_contracts=3,
    ),
    "SIL": FuturesInstrumentSpec(
        symbol="SIL",
        family="SI",
        exchange="COMEX",
        point_value=1000.0,
        min_tick=0.001,
        max_contracts=2,
    ),
}


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OmegaMacroTuning:
    """
    Tuning parameters for OMEGA_MACRO strategy.

    These control regime thresholds, indicator periods, confidence floors,
    and risk sizing. All values have safe defaults for paper trading.
    """
    # Regime classification thresholds
    vix_risk_off: float = 25.0
    vix_risk_on: float = 18.0
    drawdown_risk_off: float = -0.05
    drawdown_risk_on: float = -0.02

    # Indicator periods
    ema_slope_period: int = 20
    atr_len: int = 14
    min_bars: int = 40

    # Confidence floor
    min_confidence: float = 0.60

    # Risk sizing (tighter than alpha_futures)
    risk_budget_pct: float = 0.010
    max_trade_notional: float = 35000.0

    # Equity fallback
    equity_fallback: float = 10_000.0
    min_risk_budget_usd: float = 100.0


DEFAULT_TUNING = OmegaMacroTuning()


# ---------------------------------------------------------------------------
# Regime -> signal direction mapping
# ---------------------------------------------------------------------------

# Each entry: {symbol: SignalSide}
# NEUTRAL maps to empty dict (no signals).
REGIME_SIGNAL_MAP: Dict[MacroRegime, Dict[str, SignalSide]] = {
    MacroRegime.RISK_OFF: {
        "ZN": SignalSide.BUY,     # Flight to safety, bonds rally
        "ZB": SignalSide.BUY,     # Long duration rally
        "M6E": SignalSide.SELL,   # USD strength, euro weakness
        "SIL": SignalSide.SELL,   # Industrial demand falls
    },
    MacroRegime.RISK_ON: {
        "ZN": SignalSide.SELL,    # Yields rise, bonds fall
        "ZB": SignalSide.SELL,    # Duration sells off
        "M6E": SignalSide.BUY,   # Risk appetite, euro strengthens
        "SIL": SignalSide.BUY,   # Industrial demand, growth
    },
    MacroRegime.STAGFLATION: {
        "ZN": SignalSide.BUY,     # Rate uncertainty, flight to quality
        "ZB": SignalSide.BUY,     # Convexity hedge
        "M6E": SignalSide.SELL,   # USD safe haven
        "SIL": SignalSide.BUY,   # Commodity inflation hedge
    },
    MacroRegime.NEUTRAL: {},       # No signals — stay flat
}


# ---------------------------------------------------------------------------
# Configuration fallback
# ---------------------------------------------------------------------------

def build_omega_macro_config() -> StrategyConfig:
    """Return StrategyConfig from config module when available, else safe default."""
    try:
        from chad.strategies.omega_macro_config import build_omega_macro_config as _impl
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.OMEGA_MACRO,
            enabled=True,
            target_universe=["ZN", "ZB", "M6E", "SIL"],
            max_gross_exposure=0.18,
            notes="Macro regime futures engine (fallback config)",
        )


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _compute_confidence(
    regime: MacroRegime,
    vix: Optional[float],
    drawdown_pct: Optional[float],
    bond_trend: Optional[float],
    commodity_trend: Optional[float],
    symbol: str,
) -> float:
    """
    Compute signal confidence for a given instrument under the current regime.

    Base confidence: 0.55
    Bonuses:
    - +0.10 if VIX signal is strong (>30 for risk-off, <15 for risk-on)
    - +0.10 if drawdown signal aligns with VIX direction
    - +0.10 if bond trend aligns with regime expectation
    - +0.05 if commodity trend aligns with regime expectation
    Clamped to [0.0, 1.0].

    Parameters
    ----------
    regime : MacroRegime
        Currently classified macro regime.
    vix : Optional[float]
        Current VIX level.
    drawdown_pct : Optional[float]
        Portfolio drawdown as negative fraction.
    bond_trend : Optional[float]
        Normalized EMA slope for bonds.
    commodity_trend : Optional[float]
        Normalized EMA slope for commodities.
    symbol : str
        Instrument symbol (for regime-specific alignment checks).

    Returns
    -------
    float
        Confidence value clamped to [0.0, 1.0].
    """
    conf = 0.55

    vix_val = _to_float(vix, 20.0) if vix is not None else 20.0
    dd_val = _to_float(drawdown_pct, 0.0) if drawdown_pct is not None else 0.0

    # Strong VIX signal bonus
    if regime == MacroRegime.RISK_OFF and vix_val > 30.0:
        conf += 0.10
    elif regime == MacroRegime.RISK_ON and vix_val < 15.0:
        conf += 0.10

    # Drawdown alignment bonus
    if regime == MacroRegime.RISK_OFF and dd_val <= -0.05:
        conf += 0.10
    elif regime == MacroRegime.RISK_ON and dd_val > -0.01:
        conf += 0.10

    # Bond trend alignment bonus
    if bond_trend is not None:
        if regime in (MacroRegime.RISK_OFF, MacroRegime.STAGFLATION) and bond_trend > 0:
            # Bonds rising (prices up) aligns with risk-off / stagflation
            conf += 0.10
        elif regime == MacroRegime.RISK_ON and bond_trend < 0:
            # Bonds falling (prices down) aligns with risk-on
            conf += 0.10

    # Commodity trend alignment bonus
    if commodity_trend is not None:
        if regime == MacroRegime.STAGFLATION and commodity_trend > 0:
            conf += 0.05
        elif regime == MacroRegime.RISK_ON and commodity_trend > 0:
            conf += 0.05
        elif regime == MacroRegime.RISK_OFF and commodity_trend < 0:
            conf += 0.05

    return _clamp(conf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------

def _build_signal_for_symbol(
    *,
    symbol: str,
    bars: Sequence[Mapping[str, float]],
    price: float,
    spec: FuturesInstrumentSpec,
    side: SignalSide,
    confidence: float,
    regime: MacroRegime,
    tuning: OmegaMacroTuning,
    equity: float,
) -> Optional[TradeSignal]:
    """
    Build a TradeSignal for a single OMEGA_MACRO instrument.

    Returns None if:
    - Insufficient bars
    - Price invalid
    - ATR invalid
    - Confidence below minimum
    - Sizing yields zero contracts
    - Notional exceeds cap

    Parameters
    ----------
    symbol : str
        Futures symbol (e.g., "ZN").
    bars : Sequence[Mapping[str, float]]
        OHLCV bar history for this symbol.
    price : float
        Latest price for this symbol.
    spec : FuturesInstrumentSpec
        Instrument specification.
    side : SignalSide
        BUY or SELL direction from regime mapping.
    confidence : float
        Pre-computed confidence score.
    regime : MacroRegime
        Current macro regime.
    tuning : OmegaMacroTuning
        Strategy tuning parameters.
    equity : float
        Portfolio equity for sizing.

    Returns
    -------
    Optional[TradeSignal]
        TradeSignal if all conditions pass, else None.
    """
    if len(bars) < tuning.min_bars or price <= 0:
        return None

    atr_val = _atr(bars, tuning.atr_len)
    if atr_val <= 0:
        return None

    if confidence < tuning.min_confidence:
        return None

    # Build a StrategyTuning adapter for _compute_contract_size
    sizing_tuning = StrategyTuning(
        risk_budget_pct=tuning.risk_budget_pct,
        min_risk_budget_usd=tuning.min_risk_budget_usd,
        equity_fallback=tuning.equity_fallback,
        max_trade_notional=tuning.max_trade_notional,
        min_confidence=tuning.min_confidence,
    )

    contracts, alloc_wt, risk_budget_usd, risk_per_contract_usd = _compute_contract_size(
        symbol=symbol,
        spec=spec,
        price=price,
        atr_val=atr_val,
        equity=equity,
        tuning=sizing_tuning,
        confidence=confidence,
    )
    if contracts <= 0:
        return None

    estimated_notional = price * spec.point_value * contracts
    atr_pct = _safe_div(atr_val, price, 0.0)

    return TradeSignal(
        strategy=StrategyName.OMEGA_MACRO,
        symbol=symbol,
        side=side,
        size=float(contracts),
        confidence=float(confidence),
        asset_class=AssetClass.FUTURES,
        meta={
            "engine": "omega_macro.v1",
            "family": spec.family,
            "contract_symbol": symbol,
            "exchange": spec.exchange,
            "point_value": spec.point_value,
            "min_tick": spec.min_tick,
            "regime": regime.value,
            "atr": round(atr_val, 6),
            "atr_pct": round(atr_pct, 6),
            "allocation_weight": round(alloc_wt, 6),
            "risk_budget_usd": round(risk_budget_usd, 6),
            "risk_per_contract_usd": round(risk_per_contract_usd, 6),
            "estimated_notional": round(estimated_notional, 6),
            "required_asset_class": "futures",
        },
    )


# ---------------------------------------------------------------------------
# Main signal generator
# ---------------------------------------------------------------------------

def build_omega_macro_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    Build OMEGA_MACRO signals for all instruments based on macro regime.

    Flow:
    1. Extract VIX, drawdown, and trend data from context.
    2. Classify macro regime.
    3. Map regime to per-instrument directions.
    4. Size each position via ATR-based contract sizing.
    5. Return valid signals (or empty list if NEUTRAL / insufficient data).

    Parameters
    ----------
    ctx : object
        Market context (MarketContext or compatible).
    params : Optional[Mapping[str, Any]]
        Optional parameter overrides (reserved for future use).

    Returns
    -------
    List[TradeSignal]
        List of TradeSignal intents for the OMEGA_MACRO strategy.
    """
    tuning = OmegaMacroTuning(
        risk_budget_pct=_to_float(
            os.getenv("CHAD_OMEGA_MACRO_RISK_PCT"), DEFAULT_TUNING.risk_budget_pct
        ),
        max_trade_notional=_to_float(
            os.getenv("CHAD_OMEGA_MACRO_MAX_TRADE_NOTIONAL"), DEFAULT_TUNING.max_trade_notional
        ),
        min_confidence=_to_float(
            os.getenv("CHAD_OMEGA_MACRO_MIN_CONFIDENCE"), DEFAULT_TUNING.min_confidence
        ),
    )

    # Extract macro inputs
    vix = _vix_value(ctx)
    drawdown_pct = _portfolio_drawdown_pct(ctx)

    # Extract prices and bars for all instruments
    prices = _extract_prices(ctx)
    all_symbols = list(OMEGA_MACRO_SPECS.keys())
    bars_by_symbol = _extract_bars(ctx, all_symbols)
    equity = _extract_equity(ctx, StrategyTuning(equity_fallback=tuning.equity_fallback))

    # Compute trend signals from bond and commodity bars
    zn_bars = bars_by_symbol.get("ZN", [])
    sil_bars = bars_by_symbol.get("SIL", [])

    zn_closes = [_to_float(b.get("close"), 0.0) for b in zn_bars if _to_float(b.get("close"), 0.0) > 0]
    sil_closes = [_to_float(b.get("close"), 0.0) for b in sil_bars if _to_float(b.get("close"), 0.0) > 0]

    bond_trend = _ema_slope(zn_closes, tuning.ema_slope_period)
    commodity_trend = _ema_slope(sil_closes, tuning.ema_slope_period)

    # Classify regime
    regime = classify_macro_regime(vix, drawdown_pct, bond_trend, commodity_trend)

    # Get signal directions for this regime
    direction_map = REGIME_SIGNAL_MAP.get(regime, {})
    if not direction_map:
        return []

    # Build signals
    signals: List[TradeSignal] = []
    for symbol, spec in OMEGA_MACRO_SPECS.items():
        side = direction_map.get(symbol)
        if side is None:
            continue

        confidence = _compute_confidence(
            regime=regime,
            vix=vix,
            drawdown_pct=drawdown_pct,
            bond_trend=bond_trend,
            commodity_trend=commodity_trend,
            symbol=symbol,
        )

        signal = _build_signal_for_symbol(
            symbol=symbol,
            bars=bars_by_symbol.get(symbol, []),
            price=prices.get(symbol, 0.0),
            spec=spec,
            side=side,
            confidence=confidence,
            regime=regime,
            tuning=tuning,
            equity=equity,
        )
        if signal is not None:
            signals.append(signal)

    return signals


# ---------------------------------------------------------------------------
# Engine-compatible handler
# ---------------------------------------------------------------------------

def omega_macro_handler(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """
    StrategyEngine-compatible handler for OMEGA_MACRO.

    Matches the alpha_futures_handler / gamma_futures_handler signature.
    Checks config enabled flag before generating signals. Fail-closed
    on any unexpected exception.

    Parameters
    ----------
    ctx : object
        Market context.
    params : Optional[Mapping[str, Any]]
        Optional parameter overrides.

    Returns
    -------
    List[TradeSignal]
        Signals or empty list.
    """
    try:
        cfg = build_omega_macro_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_omega_macro_signals(ctx=ctx, params=params)
    except Exception:
        return []


__all__ = [
    "OMEGA_MACRO_SPECS",
    "OmegaMacroTuning",
    "REGIME_SIGNAL_MAP",
    "build_omega_macro_config",
    "build_omega_macro_signals",
    "omega_macro_handler",
]

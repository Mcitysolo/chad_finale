#!/usr/bin/env python3
"""
chad/strategies/alpha_futures.py

Production-grade Alpha Futures strategy for CHAD.
Generates deterministic futures TradeSignal objects for MES, MNQ, MCL, and MGC,
with dynamic sizing based on ATR and capital-allocation weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal

# ---------------------------------------------------------------------------
# Configuration fallback (for robustness if config module missing)
def build_alpha_futures_config() -> StrategyConfig:
    """Return StrategyConfig from config module when available, else use a safe default."""
    try:
        from chad.strategies.alpha_futures_config import build_alpha_futures_config as _impl
        return _impl()
    except Exception:
        return StrategyConfig(
            name=StrategyName.ALPHA_FUTURES,
            enabled=True,
            target_universe=["MES", "MNQ", "MCL", "MGC"],
            max_gross_exposure=0.25,
            notes="Futures momentum engine (fallback config)",
        )

# ---------------------------------------------------------------------------
# Instrument and tuning specs
@dataclass(frozen=True)
class FuturesInstrumentSpec:
    symbol: str
    family: str
    exchange: str
    point_value: float
    min_tick: float
    risk_multiple: float = 1.25
    max_contracts: int = 5
    min_liquidity_usd: float = 1_000_000.0

@dataclass(frozen=True)
class StrategyTuning:
    ema_fast_len: int = 12
    ema_slow_len: int = 26
    atr_len: int = 14
    breakout_lookback: int = 20
    min_bars: int = 40
    min_confidence: float = 0.65
    risk_budget_pct: float = 0.015
    min_risk_budget_usd: float = 150.0
    equity_fallback: float = 10_000.0
    max_trade_notional: float = 50_000.0
    confidence_trend_weight: float = 20.0
    allow_long: bool = True
    allow_short: bool = True

# Default spec definitions
DEFAULT_SPECS: Dict[str, FuturesInstrumentSpec] = {
    "MES": FuturesInstrumentSpec("MES", "ES", "CME", 5.0, 0.25, max_contracts=5),
    "MNQ": FuturesInstrumentSpec("MNQ", "NQ", "CME", 2.0, 0.25, max_contracts=5),
    "MCL": FuturesInstrumentSpec("MCL", "CL", "NYMEX", 100.0, 0.01, max_contracts=2),
    "MGC": FuturesInstrumentSpec("MGC", "GC", "COMEX", 10.0, 0.1, max_contracts=5),
}

# ---------------------------------------------------------------------------
# Utility functions
def _to_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float; return default on failure or NaN."""
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if abs(denominator) < 1e-12 else numerator / denominator

def _ctx_mapping(ctx: object, attr: str) -> Mapping[str, Any]:
    """Return a mapping from context or an empty dict."""
    value = getattr(ctx, attr, None)
    return value if isinstance(value, Mapping) else {}

def _extract_equity(ctx: object, tuning: StrategyTuning) -> float:
    """Derive total portfolio equity from context or fallback."""
    override = os.getenv("CHAD_ALPHA_FUTURES_EQUITY_OVERRIDE")
    if override:
        val = _to_float(override, 0.0)
        if val > 0:
            return val
    portfolio = getattr(ctx, "portfolio", None)
    if portfolio is not None:
        for field_name in ("total_equity", "equity", "net_liq", "cash"):
            val = _to_float(getattr(portfolio, field_name, None), 0.0)
            if val > 0:
                return val
    return tuning.equity_fallback

def _extract_prices(ctx: object) -> Dict[str, float]:
    """Extract latest prices from ticks, direct prices, or last bar close."""
    prices: Dict[str, float] = {}
    # 1) direct price map
    for sym, px in _ctx_mapping(ctx, "prices").items():
        p = _to_float(px, 0.0)
        if p > 0:
            prices[str(sym).strip().upper()] = p
    # 2) tick map
    for sym, tick in _ctx_mapping(ctx, "ticks").items():
        symbol = str(sym).strip().upper()
        price = _to_float(getattr(tick, "price", None), 0.0)
        if price <= 0 and isinstance(tick, Mapping):
            price = _to_float(tick.get("price"), 0.0)
        if price > 0:
            prices[symbol] = price
    # 3) fallback to latest bar close
    for sym, bars in _ctx_mapping(ctx, "bars").items():
        symbol = str(sym).strip().upper()
        if symbol in prices:
            continue
        if isinstance(bars, Sequence) and bars:
            last = bars[-1]
            if isinstance(last, Mapping):
                close = _to_float(last.get("close"), 0.0)
                if close > 0:
                    prices[symbol] = close
    return prices

def _extract_bars(ctx: object, symbols: Iterable[str]) -> Dict[str, List[Mapping[str, float]]]:
    """Clean and return bar histories."""
    bars_map = _ctx_mapping(ctx, "bars")
    out: Dict[str, List[Mapping[str, float]]] = {}
    for symbol in symbols:
        rows = bars_map.get(symbol, [])
        cleaned: List[Mapping[str, float]] = []
        if isinstance(rows, Sequence):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                o = _to_float(row.get("open"), 0.0)
                h = _to_float(row.get("high"), 0.0)
                l = _to_float(row.get("low"), 0.0)
                c = _to_float(row.get("close"), 0.0)
                v = _to_float(row.get("volume"), 0.0)
                # ensure valid bars
                if min(o, h, l, c) <= 0.0 or h < l:
                    continue
                cleaned.append({"open": o, "high": h, "low": l, "close": c, "volume": max(v, 0.0)})
        out[symbol] = cleaned
    return out

def _ema(values: Sequence[float], length: int) -> float:
    """Compute exponential moving average."""
    if not values:
        return 0.0
    if length <= 1:
        return float(values[-1])
    alpha = 2.0 / (length + 1.0)
    ema_val = float(values[0])
    for x in values[1:]:
        ema_val = alpha * float(x) + (1.0 - alpha) * ema_val
    return ema_val

def _atr(bars: Sequence[Mapping[str, float]], length: int) -> float:
    """Compute average true range."""
    if len(bars) < 2:
        return 0.0
    trs: List[float] = []
    prev_close = _to_float(bars[0].get("close"), 0.0)
    for bar in bars[1:]:
        high = _to_float(bar.get("high"), 0.0)
        low = _to_float(bar.get("low"), 0.0)
        close = _to_float(bar.get("close"), 0.0)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if tr > 0:
            trs.append(tr)
        prev_close = close
    if not trs:
        return 0.0
    window = trs[-length:] if length > 0 else trs
    return sum(window) / len(window)

def _highest_high(bars: Sequence[Mapping[str, float]], lookback: int) -> float:
    sample = bars[-lookback:] if lookback > 0 else bars
    values = [_to_float(bar.get("high"), 0.0) for bar in sample]
    values = [v for v in values if v > 0]
    return max(values) if values else 0.0

def _lowest_low(bars: Sequence[Mapping[str, float]], lookback: int) -> float:
    sample = bars[-lookback:] if lookback > 0 else bars
    values = [_to_float(bar.get("low"), 0.0) for bar in sample]
    values = [v for v in values if v > 0]
    return min(values) if values else 0.0

# ---------------------------------------------------------------------------
# Capital allocator interface
def _allocation_weight() -> float:
    try:
        from chad.portfolio.capital_allocator import get_strategy_weight
        return max(_to_float(get_strategy_weight("alpha_futures", default=1.0), 1.0), 0.0)
    except Exception:
        return 1.0

def _derive_liquidity_usd(price: float, volume: float, point_value: float) -> float:
    return max(price, 0.0) * max(volume, 0.0) * max(point_value, 0.0)

def _compute_contract_size(
    *,
    symbol: str,
    spec: FuturesInstrumentSpec,
    price: float,
    atr_val: float,
    equity: float,
    tuning: StrategyTuning,
    confidence: float,
) -> Tuple[int, float, float, float]:
    """
    Return (contracts, allocation_weight, risk_budget_usd, risk_per_contract_usd).
    If risk budget cannot cover one contract, returns 0 contracts.
    """
    alloc_weight = _allocation_weight()
    effective_risk_pct = tuning.risk_budget_pct * alloc_weight
    risk_budget = max(equity * effective_risk_pct * _clamp(confidence, 0.5, 1.25),
                      tuning.min_risk_budget_usd)
    risk_per_contract = atr_val * spec.point_value * spec.risk_multiple
    # If budget cannot support one contract, return 0
    if risk_per_contract <= 0 or risk_budget < risk_per_contract:
        return 0, alloc_weight, risk_budget, risk_per_contract
    raw_contracts = int(risk_budget // risk_per_contract)
    contracts = min(raw_contracts, spec.max_contracts)
    if contracts <= 0:
        return 0, alloc_weight, risk_budget, risk_per_contract
    # apply trade notional cap
    estimated_notional = price * spec.point_value * contracts
    while contracts > 0 and estimated_notional > tuning.max_trade_notional:
        contracts -= 1
        estimated_notional = price * spec.point_value * contracts
    if contracts <= 0:
        return 0, alloc_weight, risk_budget, risk_per_contract
    return contracts, alloc_weight, risk_budget, risk_per_contract

# ---------------------------------------------------------------------------
# Signal generator
def _build_signal_for_symbol(
    *,
    symbol: str,
    bars: Sequence[Mapping[str, float]],
    price: float,
    spec: FuturesInstrumentSpec,
    tuning: StrategyTuning,
    equity: float,
) -> Optional[TradeSignal]:
    """Generate a TradeSignal for the given symbol, or None if conditions fail."""
    if len(bars) < tuning.min_bars or price <= 0:
        return None
    closes = [_to_float(bar.get("close"), 0.0) for bar in bars]
    closes = [x for x in closes if x > 0]
    if len(closes) < tuning.min_bars:
        return None
    ema_fast = _ema(closes, tuning.ema_fast_len)
    ema_slow = _ema(closes, tuning.ema_slow_len)
    atr_val = _atr(bars, tuning.atr_len)
    if atr_val <= 0:
        return None
    highest_high = _highest_high(bars[:-1], tuning.breakout_lookback) if len(bars) > 1 else 0.0
    lowest_low = _lowest_low(bars[:-1], tuning.breakout_lookback) if len(bars) > 1 else 0.0
    latest_bar = bars[-1]
    liquidity_usd = _derive_liquidity_usd(price, _to_float(latest_bar.get("volume"), 0.0), spec.point_value)
    if liquidity_usd < spec.min_liquidity_usd:
        return None
    # Trend regime: fast/slow EMA alignment
    side: Optional[SignalSide] = None
    if tuning.allow_short and price < ema_fast < ema_slow:
        side = SignalSide.SELL
    if tuning.allow_long and price > ema_fast > ema_slow:
        side = SignalSide.BUY
    # Breakout overrides
    if tuning.allow_short and lowest_low > 0 and price <= lowest_low:
        side = SignalSide.SELL
    if tuning.allow_long and highest_high > 0 and price >= highest_high:
        side = SignalSide.BUY
    if side is None:
        return None
    # Confidence calculation
    trend_strength = abs(ema_fast - ema_slow) / max(price, 1e-12)
    confidence = 0.60
    confidence += _clamp(trend_strength * tuning.confidence_trend_weight, 0.0, 0.25)
    confidence += 0.10 if (price >= highest_high or price <= lowest_low) else 0.0
    confidence += _clamp(_safe_div(liquidity_usd, 10_000_000_000.0, 0.0), 0.0, 0.10)
    confidence = _clamp(confidence, 0.0, 1.0)
    if confidence < tuning.min_confidence:
        return None
    contracts, alloc_wt, risk_budget_usd, risk_per_contract_usd = _compute_contract_size(
        symbol=symbol, spec=spec, price=price, atr_val=atr_val,
        equity=equity, tuning=tuning, confidence=confidence)
    if contracts <= 0:
        return None
    estimated_notional = price * spec.point_value * contracts
    atr_pct = _safe_div(atr_val, price, 0.0)
    return TradeSignal(
        strategy=StrategyName.ALPHA_FUTURES,
        symbol=symbol,
        side=side,
        size=float(contracts),
        confidence=float(confidence),
        asset_class=AssetClass.FUTURES,
        meta={
            "engine": "alpha_futures.v4",
            "family": spec.family,
            "contract_symbol": symbol,
            "exchange": spec.exchange,
            "point_value": spec.point_value,
            "min_tick": spec.min_tick,
            "ema_fast": round(ema_fast, 6),
            "ema_slow": round(ema_slow, 6),
            "atr": round(atr_val, 6),
            "atr_pct": round(atr_pct, 6),
            "highest_high": round(highest_high, 6),
            "lowest_low": round(lowest_low, 6),
            "allocation_weight": round(alloc_wt, 6),
            "risk_budget_usd": round(risk_budget_usd, 6),
            "risk_per_contract_usd": round(risk_per_contract_usd, 6),
            "estimated_notional": round(estimated_notional, 6),
            "liquidity_usd": round(liquidity_usd, 6),
            "spread_bps": 0.0,
            "required_asset_class": "futures",
        },
    )

def build_alpha_futures_signals(
    ctx: object,
    params: Optional[Mapping[str, Any]] = None,
) -> List[TradeSignal]:
    """Build signals for all defined futures instruments."""
    tuning = StrategyTuning(
        risk_budget_pct=_to_float(os.getenv("CHAD_ALPHA_FUTURES_RISK_PCT"), 0.015),
        min_risk_budget_usd=_to_float(os.getenv("CHAD_ALPHA_FUTURES_MIN_RISK_BUDGET_USD"), 150.0),
        equity_fallback=_to_float(os.getenv("CHAD_ALPHA_FUTURES_EQUITY_FALLBACK"), 10_000.0),
        max_trade_notional=_to_float(os.getenv("CHAD_ALPHA_FUTURES_MAX_TRADE_NOTIONAL"), 50_000.0),
        min_confidence=_to_float(os.getenv("CHAD_ALPHA_FUTURES_MIN_CONFIDENCE"), 0.65),
    )
    prices = _extract_prices(ctx)
    bars_by_symbol = _extract_bars(ctx, DEFAULT_SPECS.keys())
    equity = _extract_equity(ctx, tuning)
    signals: List[TradeSignal] = []
    for symbol, spec in DEFAULT_SPECS.items():
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

def alpha_futures_handler(ctx: object, params: Optional[Mapping[str, Any]] = None) -> List[TradeSignal]:
    """StrategyEngine-compatible handler; supports optional params argument."""
    try:
        cfg = build_alpha_futures_config()
        if not getattr(cfg, "enabled", True):
            return []
        return build_alpha_futures_signals(ctx=ctx, params=params)
    except Exception:
        return []

__all__ = ["build_alpha_futures_config", "build_alpha_futures_signals", "alpha_futures_handler"]

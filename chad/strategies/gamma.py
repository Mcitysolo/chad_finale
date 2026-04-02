#!/usr/bin/env python3
"""
chad/strategies/gamma.py

GammaBrain — activated swing engine (Phase-4 Edge Upgrade)

Why this replacement
--------------------
The prior Gamma was "mean reversion around entry anchor" and rarely triggers unless a position
already exists and price deviates > outer_band_pct. Phase 11 dev note: "Gamma barely trades".

This Gamma V2:
- Trades with or without existing positions (no anchor dependency)
- Uses regime switching:
    * Trend regime: momentum / trend-follow continuation
    * Range regime: mean-reversion to EMA
- Adds volatility + liquidity sanity gates (best-effort)
- Adds deterministic exits (time stop, ATR trail, trend break, vol spike)
- Keeps StrategyEngine contract: build_gamma_config + gamma_handler -> Sequence[TradeSignal]
- Deterministic, no I/O, no broker calls

Inputs expected (best-effort)
-----------------------------
Required:
- ctx.ticks[symbol].price
- ctx.portfolio.cash (for min_cash guard)
Optional (for full edge):
- ctx.bars[symbol] : list of dict bars with keys: open/high/low/close (or tuple-like)
- ctx.dollar_volume / ctx.volume_usd / ctx.liquidity_usd for liquidity gates
- ctx.spread_bps for liquidity/spread sanity (if provided by feed)

If bars are missing, strategy becomes conservative and mostly holds/exits only.
"""

from __future__ import annotations


def _safe_int(x, default: int = 0) -> int:
    """Best-effort int conversion. Never raises."""
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)

from dataclasses import dataclass
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
from chad.utils.universe_provider import get_trade_universe


Number = float


# -------------------------
# Params
# -------------------------

@dataclass(frozen=True)
class GammaParams:
    # Safety
    min_cash: float = 5_000.0
    max_position_units: float = 50.0

    # Indicators
    ema_fast: int = 20
    ema_slow: int = 80
    atr_period: int = 14

    # Regime filters
    min_atr_pct: float = 0.0015
    max_atr_pct: float = 0.0400
    vol_spike_atr_pct: float = 0.0550

    # Range vs trend classifier
    # If EMA slope small and price oscillates around slow EMA -> range regime
    slope_bars: int = 12
    max_slope_atr: float = 0.06  # slope (in ATR units per bar) below this => range-ish

    # Entries
    momentum_atr: float = 0.35         # trend entries
    reversion_atr: float = 0.75        # range entries when deviated from EMA slow

    # Position sizing
    base_size: float = 5.0
    max_size: float = 12.0

    # Exits
    time_stop_bars: int = 60
    min_favor_move_atr: float = 0.5
    atr_trail_mult: float = 2.4

    # Anti-churn
    min_delta_size: float = 1.0


DEFAULT_PARAMS = GammaParams()


# -------------------------
# Helpers
# -------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _get_mapping(obj: Any, attr: str) -> Mapping[str, Any]:
    try:
        m = getattr(obj, attr, None)
        if isinstance(m, dict):
            return m
        if m is None:
            return {}
        if hasattr(m, "get"):
            return m  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def _bar_close(bar: Any) -> float:
    if isinstance(bar, dict):
        return _safe_float(bar.get("close") or bar.get("c") or bar.get("Close") or bar.get("C"), 0.0)
    if isinstance(bar, (list, tuple)) and len(bar) >= 4:
        # (o,h,l,c,...) or (ts,o,h,l,c,...)
        return _safe_float(bar[-2], 0.0) if len(bar) >= 5 else _safe_float(bar[-1], 0.0)
    return 0.0


def _bar_high(bar: Any) -> float:
    if isinstance(bar, dict):
        return _safe_float(bar.get("high") or bar.get("h") or bar.get("High") or bar.get("H"), 0.0)
    if isinstance(bar, (list, tuple)) and len(bar) >= 5:
        return _safe_float(bar[1] if len(bar) == 5 else bar[2], 0.0)
    return 0.0


def _bar_low(bar: Any) -> float:
    if isinstance(bar, dict):
        return _safe_float(bar.get("low") or bar.get("l") or bar.get("Low") or bar.get("L"), 0.0)
    if isinstance(bar, (list, tuple)) and len(bar) >= 5:
        return _safe_float(bar[2] if len(bar) == 5 else bar[3], 0.0)
    return 0.0


def _ema(values: Sequence[float], period: int) -> List[float]:
    alpha = 2.0 / (period + 1.0)
    out: List[float] = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
        out.append(e)
    return out


def _true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> List[float]:
    out = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        out.append(max(hl, hc, lc))
    return out


def _atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> List[float]:
    tr = _true_range(highs, lows, closes)
    return _ema(tr, period)


def _asset_class(sym: str) -> AssetClass:
    # Gamma is typically ETFs in your system; keep ETF default, fallback to STOCK
    if sym in {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "LQD", "VWO", "IEMG"}:
        return AssetClass.ETF
    return AssetClass.STOCK


# -------------------------
# In-memory position state
# -------------------------

class _GammaState:
    def __init__(self) -> None:
        self.pos: Dict[str, Dict[str, Any]] = {}

    def get(self, sym: str) -> Dict[str, Any]:
        return dict(self.pos.get(sym, {}))

    def set(self, sym: str, st: Dict[str, Any]) -> None:
        self.pos[sym] = dict(st)


_STATE = _GammaState()


# -------------------------
# Core per-symbol logic
# -------------------------

def _propose_for_symbol(symbol: str, ctx: MarketContext, p: GammaParams) -> Sequence[TradeSignal]:
    portfolio: PortfolioSnapshot = ctx.portfolio
    if portfolio.cash < p.min_cash:
        return []

    tick = ctx.ticks.get(symbol)
    if tick is None or tick.price <= 0:
        return []

    price = float(tick.price)

    # Optional liquidity gates (best-effort)
    dv_map = _get_mapping(ctx, "dollar_volume") or _get_mapping(ctx, "volume_usd") or _get_mapping(ctx, "liquidity_usd")
    if dv_map and symbol in dv_map:
        dv = _safe_float(dv_map.get(symbol), 0.0)
        # very light gate; you can tighten later
        if dv > 0 and dv < 2_000_000.0:
            return []

    # Optional spread gate
    sp_map = _get_mapping(ctx, "spread_bps")
    if sp_map and symbol in sp_map:
        sp = _safe_float(sp_map.get(symbol), 0.0)
        if sp > 8.0:
            return []

    bars = _get_mapping(ctx, "bars").get(symbol)
    if not isinstance(bars, (list, tuple)) or len(bars) < max(p.ema_slow + 2, p.atr_period + 2):
        # No history => only allow exits if currently holding (fail-closed)
        pos = portfolio.positions.get(symbol)
        if pos is not None and pos.quantity > 0:
            return [
                TradeSignal(
                    strategy=StrategyName.GAMMA,
                    symbol=symbol,
                    side=SignalSide.SELL,
                    size=min(p.base_size, float(pos.quantity)),
                    confidence=0.55,
                    asset_class=_asset_class(symbol),
                    created_at=ctx.now,
                    meta={"reason": "no_history_exit_only"},
                )
            ]
        return []

    closes = []
    highs = []
    lows = []
    tail = list(bars)[-max(120, p.ema_slow + 10, p.atr_period + 10) :]
    for b in tail:
        c = _bar_close(b)
        h = _bar_high(b)
        l = _bar_low(b)
        if c <= 0 or h <= 0 or l <= 0 or h < l:
            continue
        closes.append(c)
        highs.append(h)
        lows.append(l)

    if len(closes) < max(p.ema_slow + 2, p.atr_period + 2):
        return []

    ef = _ema(closes, p.ema_fast)[-1]
    es_series = _ema(closes, p.ema_slow)
    es = es_series[-1]
    a = _atr(highs, lows, closes, p.atr_period)[-1]
    atr_pct = (a / price) if price else 0.0

    # Regime gate
    if atr_pct < p.min_atr_pct or atr_pct > p.max_atr_pct:
        return []

    # Classify regime: range vs trend based on slow EMA slope in ATR units
    if len(es_series) < p.slope_bars + 1:
        return []
    slope = (es_series[-1] - es_series[-(p.slope_bars + 1)]) / float(p.slope_bars)
    slope_atr = (slope / a) if a else 0.0
    range_regime = abs(slope_atr) <= p.max_slope_atr

    # Position info
    pos = portfolio.positions.get(symbol)
    qty = float(pos.quantity) if pos is not None else 0.0
    if abs(qty) >= p.max_position_units:
        return []

    # In-memory state for exits
    st = _STATE.get(symbol)
    entry = _safe_float(st.get("entry", 0.0), 0.0)
    held = _safe_int(st.get("held", 0), 0)
    peak = _safe_float(st.get("peak", 0.0), 0.0)

    # Exits (if holding)
    if qty > 0:
        if atr_pct > p.vol_spike_atr_pct:
            size = min(p.base_size, qty)
            _STATE.set(symbol, {"entry": 0.0, "held": 0, "peak": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.GAMMA,
                    symbol=symbol,
                    side=SignalSide.SELL,
                    size=size,
                    confidence=0.75,
                    asset_class=_asset_class(symbol),
                    created_at=ctx.now,
                    meta={"reason": "vol_spike_exit", "atr_pct": atr_pct},
                )
            ]

        if ef <= es:
            size = min(p.base_size, qty)
            _STATE.set(symbol, {"entry": 0.0, "held": 0, "peak": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.GAMMA,
                    symbol=symbol,
                    side=SignalSide.SELL,
                    size=size,
                    confidence=0.70,
                    asset_class=_asset_class(symbol),
                    created_at=ctx.now,
                    meta={"reason": "trend_break_exit"},
                )
            ]

        if entry > 0 and held >= p.time_stop_bars:
            if (price - entry) < (p.min_favor_move_atr * a):
                size = min(p.base_size, qty)
                _STATE.set(symbol, {"entry": 0.0, "held": 0, "peak": 0.0})
                return [
                    TradeSignal(
                        strategy=StrategyName.GAMMA,
                        symbol=symbol,
                        side=SignalSide.SELL,
                        size=size,
                        confidence=0.65,
                        asset_class=_asset_class(symbol),
                        created_at=ctx.now,
                        meta={"reason": "time_stop_exit"},
                    )
                ]

        if peak > 0 and (peak - price) > (p.atr_trail_mult * a):
            size = min(p.base_size, qty)
            _STATE.set(symbol, {"entry": 0.0, "held": 0, "peak": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.GAMMA,
                    symbol=symbol,
                    side=SignalSide.SELL,
                    size=size,
                    confidence=0.65,
                    asset_class=_asset_class(symbol),
                    created_at=ctx.now,
                    meta={"reason": "atr_trail_exit"},
                )
            ]

        # Update hold state
        _STATE.set(symbol, {"entry": entry or price, "held": held + 1, "peak": max(peak or price, price)})

    # Entries
    signals: List[TradeSignal] = []

    if qty <= 0:
        if range_regime:
            # Mean reversion to slow EMA: buy when price is sufficiently below slow EMA
            dev_atr = (es - price) / a if a else 0.0
            if dev_atr >= p.reversion_atr:
                size = min(p.max_size, p.base_size * (1.0 + 0.4 * dev_atr))
                signals.append(
                    TradeSignal(
                        strategy=StrategyName.GAMMA,
                        symbol=symbol,
                        side=SignalSide.BUY,
                        size=float(size),
                        confidence=min(0.85, 0.55 + 0.12 * dev_atr),
                        asset_class=_asset_class(symbol),
                        created_at=ctx.now,
                        meta={"reason": "range_mean_reversion_buy", "dev_atr": dev_atr, "slope_atr": slope_atr},
                    )
                )
                _STATE.set(symbol, {"entry": price, "held": 1, "peak": price})
        else:
            # Trend continuation: buy when price above fast EMA with ATR-normalized momentum
            mom_atr = (price - ef) / a if a else 0.0
            if (ef > es) and (price > ef) and (mom_atr >= p.momentum_atr):
                size = min(p.max_size, p.base_size * (1.0 + 0.6 * mom_atr))
                signals.append(
                    TradeSignal(
                        strategy=StrategyName.GAMMA,
                        symbol=symbol,
                        side=SignalSide.BUY,
                        size=float(size),
                        confidence=min(0.90, 0.58 + 0.15 * mom_atr),
                        asset_class=_asset_class(symbol),
                        created_at=ctx.now,
                        meta={"reason": "trend_momentum_buy", "mom_atr": mom_atr, "slope_atr": slope_atr},
                    )
                )
                _STATE.set(symbol, {"entry": price, "held": 1, "peak": price})

    return signals


# -------------------------
# Engine wiring
# -------------------------

def build_gamma_config() -> StrategyConfig:
    universe = get_trade_universe()
    return StrategyConfig(
        name=StrategyName.GAMMA,
        enabled=True,
        target_universe=universe,
        max_gross_exposure=None,
        notes="GammaBrain (activated swing engine; regime switch + exits).",
    )


def gamma_handler(ctx: MarketContext, params: GammaParams | None = None) -> Sequence[TradeSignal]:
    p = params or DEFAULT_PARAMS
    universe = get_trade_universe()
    signals: List[TradeSignal] = []
    for symbol in universe:
        sym = _norm_sym(symbol)
        if not sym:
            continue
        signals.extend(_propose_for_symbol(sym, ctx, p))
    return signals

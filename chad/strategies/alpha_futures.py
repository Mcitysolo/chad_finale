#!/usr/bin/env python3
"""
chad/strategies/alpha_futures.py

Production-grade Alpha Futures strategy for CHAD.
Generates deterministic futures TradeSignal objects for MES, MNQ, and MGC,
with dynamic sizing based on ATR and capital-allocation weights.

Universe is deliberately non-overlapping with gamma_futures (MCL-led)
so opposite-side signals on the same symbol cannot net to zero and
silently drop in the execution planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal
from chad.utils.risk_reward import passes_rr_gate
from chad.utils.session import session_decision

# Equity-index futures eligible for tier-aware session gating. MCL/MGC keep
# their pre-existing UTC overnight gate; ZN/ZB and other instruments stay
# unaffected by Phase A Item 2.
_EQUITY_INDEX_FUTURES = frozenset({"MES", "MNQ", "MYM", "M2K"})

_TRADE_CLOSER_STATE_PATH = "/home/ubuntu/chad_finale/runtime/trade_closer_state.json"

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
            target_universe=["MES", "MNQ", "MGC"],
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
    # Exit rules — checked before entry logic when a position is open
    stop_loss_atr_multiple: float = 2.0
    target_atr_multiple: float = 3.0
    trend_exit_on_ema_slow: bool = True
    time_stop_bars: int = 20

# Default spec definitions — shared across alpha_futures and gamma_futures.
# Each strategy selects a DISJOINT subset via ALPHA_FUTURES_UNIVERSE /
# GAMMA_FUTURES_UNIVERSE to prevent opposite-side cancellation in the
# execution planner (alpha momentum BUY vs gamma reversion SELL on MES
# historically netted to zero and dropped both signals every cycle).
DEFAULT_SPECS: Dict[str, FuturesInstrumentSpec] = {
    "MES": FuturesInstrumentSpec("MES", "ES", "CME", 5.0, 0.25, max_contracts=5),
    "MNQ": FuturesInstrumentSpec("MNQ", "NQ", "CME", 2.0, 0.25, max_contracts=5),
    "MCL": FuturesInstrumentSpec("MCL", "CL", "NYMEX", 100.0, 0.01, max_contracts=2),
    "MGC": FuturesInstrumentSpec("MGC", "GC", "COMEX", 10.0, 0.1, max_contracts=5),
    "MYM": FuturesInstrumentSpec("MYM", "YM", "CBOT", 0.5, 1.0, max_contracts=5),
    "M2K": FuturesInstrumentSpec("M2K", "RTY", "CME", 5.0, 0.1, max_contracts=5),
    "ZN": FuturesInstrumentSpec("ZN", "ZN", "CBOT", 1000.0, 0.015625, max_contracts=3),
    "ZB": FuturesInstrumentSpec("ZB", "ZB", "CBOT", 1000.0, 0.03125, max_contracts=3),
}

# alpha_futures owns the equity-index momentum + metals leg. Explicitly
# excludes MCL / MYM / M2K / bonds — those belong to gamma_futures.
ALPHA_FUTURES_UNIVERSE: Tuple[str, ...] = ("MES", "MNQ", "MGC")

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


def _setup_family_for_alpha_futures(side: SignalSide, *, breakout: bool = True) -> str:
    if side == SignalSide.BUY:
        return "momentum_breakout_long"
    if side == SignalSide.SELL:
        return "momentum_breakout_short"
    return f"alpha_futures_{side.value.lower()}"

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
    tier_max_risk_usd: Optional[float] = None,
) -> Tuple[int, float, float, float]:
    """
    Return (contracts, allocation_weight, risk_budget_usd, risk_per_contract_usd).
    If risk budget cannot cover one contract, returns 0 contracts.
    """
    alloc_weight = _allocation_weight()
    effective_risk_pct = tuning.risk_budget_pct * alloc_weight
    risk_budget = max(equity * effective_risk_pct * _clamp(confidence, 0.5, 1.25),
                      tuning.min_risk_budget_usd)
    # Option A:
    # Preserve legacy sizing when no tier budget is active.
    # Activate stop-aligned sizing only when tier_max_risk_usd is present.
    if tier_max_risk_usd is not None:
        stop_atr_mult_for_sizing = getattr(
            tuning,
            "stop_loss_atr_multiple",
            spec.risk_multiple,
        )
    else:
        stop_atr_mult_for_sizing = spec.risk_multiple
    sizing_distance_pts = atr_val * stop_atr_mult_for_sizing
    risk_per_contract = sizing_distance_pts * spec.point_value
    # If budget cannot support one contract, return 0
    if risk_per_contract <= 0 or risk_budget < risk_per_contract:
        return 0, alloc_weight, risk_budget, risk_per_contract
    raw_contracts = int(risk_budget // risk_per_contract)
    if tier_max_risk_usd is not None and risk_per_contract > 0:
        tier_contracts = int(float(tier_max_risk_usd) // risk_per_contract)
        raw_contracts = min(raw_contracts, tier_contracts)
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
def _load_alpha_futures_open_positions() -> Dict[str, Dict[str, Any]]:
    """
    Read open alpha_futures positions out of runtime/trade_closer_state.json.

    Returns: { symbol: {side, quantity, avg_entry_price, earliest_ts} }
    """
    out: Dict[str, Dict[str, Any]] = {}
    data: Optional[Mapping[str, Any]] = None
    last_err: Optional[Exception] = None
    # Atomic read with retry — protects against concurrent writer races where
    # the trade_closer state file may be partially written or briefly absent.
    for _attempt in range(3):
        try:
            with open(_TRADE_CLOSER_STATE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            break
        except Exception as e:
            last_err = e
            time.sleep(0.05)
    if data is None:
        logger.warning(
            "ALPHA_FUTURES_POSITION_LOAD_FAILED fail_closed symbol_guard_active: %s",
            last_err,
        )
        # Fail CLOSED — if we cannot read position state, assume positions exist
        # for every symbol this strategy trades to prevent stacking on a
        # state-read failure.
        return {
            sym: {"side": "UNKNOWN", "quantity": 1, "fail_closed": True}
            for sym in ALPHA_FUTURES_UNIVERSE
        }
    for entry in (data.get("queues") or []):
        if str(entry.get("strategy", "")).strip().lower() != "alpha_futures":
            continue
        symbol = str(entry.get("symbol", "")).strip().upper()
        if symbol not in ALPHA_FUTURES_UNIVERSE:
            continue
        lots = [
            lot for lot in (entry.get("lots") or [])
            if _to_float(lot.get("quantity"), 0.0) > 0
        ]
        if not symbol or not lots:
            continue
        head_side = str(lots[0].get("side", "")).strip().upper()
        total_qty = sum(_to_float(lot.get("quantity"), 0.0) for lot in lots)
        if total_qty <= 0:
            continue
        notional = sum(
            _to_float(lot.get("quantity"), 0.0) * _to_float(lot.get("fill_price"), 0.0)
            for lot in lots
        )
        avg_px = notional / total_qty if total_qty > 0 else 0.0
        earliest: Optional[str] = None
        for lot in lots:
            ts = lot.get("ts_utc")
            if ts and (earliest is None or str(ts) < str(earliest)):
                earliest = str(ts)
        out[symbol] = {
            "side": head_side,
            "quantity": total_qty,
            "avg_entry_price": avg_px,
            "earliest_ts": earliest,
        }
    return out


def _evaluate_exit_signal(
    *,
    symbol: str,
    spec: FuturesInstrumentSpec,
    price: float,
    ema_slow: float,
    atr_val: float,
    open_position: Mapping[str, Any],
    tuning: StrategyTuning,
) -> Optional[TradeSignal]:
    """
    If the open position satisfies any exit condition, return a closing
    TradeSignal (opposite side, sized to flatten). Otherwise return None.

    Order of checks:
      1. ATR stop loss (adverse move > stop_loss_atr_multiple * ATR)
      2. Trend exit vs ema_slow
      3. Time stop (held > time_stop_bars minutes ≈ bars on a 1m loop)
    """
    pos_side = str(open_position.get("side", "")).upper()
    pos_qty = _to_float(open_position.get("quantity"), 0.0)
    avg_entry = _to_float(open_position.get("avg_entry_price"), 0.0)
    if pos_side not in ("BUY", "SELL") or pos_qty <= 0:
        return None

    exit_side: Optional[SignalSide] = None
    exit_reason: Optional[str] = None

    # 1. ATR stop loss
    if avg_entry > 0 and atr_val > 0 and tuning.stop_loss_atr_multiple > 0:
        if pos_side == "BUY":
            adverse = avg_entry - price
            if adverse > tuning.stop_loss_atr_multiple * atr_val:
                exit_side = SignalSide.SELL
                exit_reason = "stop_loss_atr"
        else:  # SELL
            adverse = price - avg_entry
            if adverse > tuning.stop_loss_atr_multiple * atr_val:
                exit_side = SignalSide.BUY
                exit_reason = "stop_loss_atr"

    # 2. Trend exit vs ema_slow
    if exit_side is None and tuning.trend_exit_on_ema_slow and ema_slow > 0:
        if pos_side == "BUY" and price < ema_slow:
            exit_side = SignalSide.SELL
            exit_reason = "trend_exit_ema_slow"
        elif pos_side == "SELL" and price > ema_slow:
            exit_side = SignalSide.BUY
            exit_reason = "trend_exit_ema_slow"

    # 3. Time stop
    if exit_side is None and tuning.time_stop_bars > 0:
        earliest_ts = open_position.get("earliest_ts")
        if isinstance(earliest_ts, str) and earliest_ts:
            try:
                ts_clean = earliest_ts.replace("Z", "+00:00")
                entry_dt = datetime.fromisoformat(ts_clean)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                age_min = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 60.0
                if age_min >= float(tuning.time_stop_bars):
                    exit_side = SignalSide.SELL if pos_side == "BUY" else SignalSide.BUY
                    exit_reason = "time_stop"
            except Exception:
                pass

    if exit_side is None:
        return None

    return TradeSignal(
        strategy=StrategyName.ALPHA_FUTURES,
        symbol=symbol,
        side=exit_side,
        size=float(pos_qty),
        confidence=1.0,
        asset_class=AssetClass.FUTURES,
        meta={
            "engine": "alpha_futures.v4",
            "family": spec.family,
            "contract_symbol": symbol,
            "exchange": spec.exchange,
            "point_value": spec.point_value,
            "min_tick": spec.min_tick,
            "exit_reason": exit_reason,
            "open_side": pos_side,
            "avg_entry_price": round(avg_entry, 6),
            "atr": round(atr_val, 6),
            "ema_slow": round(ema_slow, 6),
            "estimated_notional": round(price * spec.point_value * pos_qty, 6),
            "spread_bps": 0.0,
            "required_asset_class": "futures",
            "is_exit": True,
        },
    )


def _build_signal_for_symbol(
    *,
    symbol: str,
    bars: Sequence[Mapping[str, float]],
    price: float,
    spec: FuturesInstrumentSpec,
    tuning: StrategyTuning,
    equity: float,
    open_position: Optional[Mapping[str, Any]] = None,
    tier_max_risk_usd: Optional[float] = None,
    session_entry_allowed: bool = True,
    session_window: Optional[str] = None,
    primary_session_only: Optional[bool] = None,
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

    # ---- Exit logic (position-aware) — runs before entry signals ----
    if open_position:
        exit_signal = _evaluate_exit_signal(
            symbol=symbol,
            spec=spec,
            price=price,
            ema_slow=ema_slow,
            atr_val=atr_val,
            open_position=open_position,
            tuning=tuning,
        )
        if exit_signal is not None:
            return exit_signal
        # Position open and no exit triggered → suppress re-entry on this symbol.
        return None

    # ---- Overnight illiquidity gate (entry-only) ----
    # MCL and MGC have thin overnight liquidity and have produced poor fills
    # outside of regular trading hours. Block new entries when current UTC time
    # is outside the 13:30–20:00 UTC window. Exits are unaffected (handled above).
    # MES and MNQ are intentionally exempt — they have shown edge overnight.
    if symbol in ("MCL", "MGC"):
        _now_utc = datetime.now(timezone.utc)
        _minutes_of_day = _now_utc.hour * 60 + _now_utc.minute
        # Active RTH window: 13:30 (810) inclusive → 20:00 (1200) exclusive
        if _minutes_of_day < 810 or _minutes_of_day >= 1200:
            logger.info(
                "OVERNIGHT_GATE_SKIP symbol=%s hour=%d minute=%d",
                symbol, _now_utc.hour, _now_utc.minute,
            )
            return None

    # ---- Tier-aware session gate (entry-only, equity-index futures only) ----
    # Applies only when a TierRiskProfile is present on the context. The
    # caller resolves the SessionDecision once per cycle and passes it in;
    # exits have already been emitted above. MCL/MGC keep their existing
    # UTC overnight gate; bonds and other instruments are not affected.
    if (
        primary_session_only is not None
        and symbol in _EQUITY_INDEX_FUTURES
        and not session_entry_allowed
    ):
        logger.info(
            "SESSION_GATE_SKIP symbol=%s primary_session_only=%s window=%s",
            symbol, bool(primary_session_only), session_window,
        )
        return None

    # ---- Pre-entry R:R gate (entry-only, fail-open) ----
    # Block new entries whose reward/risk ratio falls below the floor. Exits
    # above are unaffected. Degenerate inputs (atr_val <= 0 or zero multiples)
    # fail open via passes_rr_gate.
    _stop_mult = getattr(tuning, "stop_loss_atr_multiple", spec.risk_multiple)
    _target_mult = getattr(tuning, "target_atr_multiple", 3.0)
    _stop_pts = atr_val * float(_stop_mult)
    _target_pts = atr_val * float(_target_mult)
    if not passes_rr_gate(_target_pts, _stop_pts):
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
        equity=equity, tuning=tuning, confidence=confidence,
        tier_max_risk_usd=tier_max_risk_usd)
    if contracts <= 0:
        return None
    estimated_notional = price * spec.point_value * contracts
    atr_pct = _safe_div(atr_val, price, 0.0)
    _breakout_entry = (
        (highest_high > 0 and price >= highest_high)
        or (lowest_low > 0 and price <= lowest_low)
    )
    _entry_meta: Dict[str, Any] = {
            "engine": "alpha_futures.v4",
            "family": spec.family,
            "setup_family": _setup_family_for_alpha_futures(
                side, breakout=_breakout_entry,
            ),
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
            "stop_distance_pts": round(
                atr_val * getattr(tuning, "stop_loss_atr_multiple", spec.risk_multiple),
                6,
            ),
            "stop_distance_usd": round(
                atr_val * getattr(tuning, "stop_loss_atr_multiple", spec.risk_multiple) * spec.point_value,
                6,
            ),
            "sizing_distance_pts": round(
                risk_per_contract_usd / spec.point_value,
                6,
            ) if spec.point_value else 0.0,
            "sizing_risk_multiple": round(
                (risk_per_contract_usd / spec.point_value / atr_val),
                6,
            ) if spec.point_value and atr_val else 0.0,
            "tier_max_risk_usd": tier_max_risk_usd,
            "rr_ratio": round(float(_target_mult) / float(_stop_mult), 4)
                if float(_stop_mult) > 0 else None,
            "rr_gate": "PASSED",
    }
    if primary_session_only is not None and symbol in _EQUITY_INDEX_FUTURES:
        _entry_meta["session_window"] = session_window
        _entry_meta["session_gate"] = "PASSED"
        _entry_meta["primary_session_only"] = bool(primary_session_only)
    return TradeSignal(
        strategy=StrategyName.ALPHA_FUTURES,
        symbol=symbol,
        side=side,
        size=float(contracts),
        confidence=float(confidence),
        asset_class=AssetClass.FUTURES,
        meta=_entry_meta,
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
    bars_by_symbol = _extract_bars(ctx, ALPHA_FUTURES_UNIVERSE)
    equity = _extract_equity(ctx, tuning)
    # Position-aware: prefer ctx.paper_positions if provided, otherwise read
    # the trade_closer ledger directly so the strategy can emit exit signals.
    ctx_positions = getattr(ctx, "paper_positions", None)
    if isinstance(ctx_positions, Mapping):
        open_positions: Dict[str, Dict[str, Any]] = {
            str(k).strip().upper(): dict(v)
            for k, v in ctx_positions.items()
            if isinstance(v, Mapping)
        }
    else:
        open_positions = _load_alpha_futures_open_positions()
    _tier_profile = getattr(ctx, "tier_profile", None)
    tier_max_risk_usd = getattr(
        _tier_profile,
        "max_risk_per_trade_usd",
        None,
    )
    # Resolve the tier-aware session decision once per cycle. Fail-open:
    # if tier_profile is absent, or session_decision raises, the strategy
    # preserves its pre-existing behavior (no equity-index session gate).
    _primary_only = getattr(_tier_profile, "primary_session_only", None)
    _session_entry_allowed: bool = True
    _session_window: Optional[str] = None
    if _primary_only is not None:
        try:
            _decision = session_decision(
                getattr(ctx, "now", None),
                primary_session_only=bool(_primary_only),
            )
            _session_entry_allowed = _decision.entry_allowed
            _session_window = _decision.session_window
        except Exception as _exc:
            logger.warning(
                "alpha_futures: session_decision failed=%s — fail-open", _exc,
            )
    signals: List[TradeSignal] = []
    for symbol in ALPHA_FUTURES_UNIVERSE:
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
            open_position=open_positions.get(symbol),
            tier_max_risk_usd=tier_max_risk_usd,
            session_entry_allowed=_session_entry_allowed,
            session_window=_session_window,
            primary_session_only=(
                bool(_primary_only) if _primary_only is not None else None
            ),
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

__all__ = [
    "ALPHA_FUTURES_UNIVERSE",
    "build_alpha_futures_config",
    "build_alpha_futures_signals",
    "alpha_futures_handler",
]

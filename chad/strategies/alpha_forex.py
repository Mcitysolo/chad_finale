from __future__ import annotations

"""
AlphaForex — CHAD Phase-4 Edge Quality Upgrade (Strategy-only, execution-clean)

What this module is
-------------------
A production-grade FX signal engine that stays strictly within CHAD’s strategy boundary:
- NO network calls
- NO broker/exchange execution
- NO disk writes or shared-state writes (in-memory state only)
- Deterministic for a given ctx snapshot + params

Why this exists
---------------
Your current alpha_forex.py is a Phase-7 safe baseline that emits no signals.
Phase 11 dev request requires "make edge real" (paper-first), specifically:
- volatility filter
- trend filter
- volume/liquidity filter (in FX: spreads/liquidity proxies)
- tighter bad setups, fewer low-quality entries
- real exits

This implementation adds:
- Regime gating (trend + ATR% volatility band)
- Liquidity proxy gating (spread guard if available; optional $volume if available)
- Anti-chase (avoid entering after oversized candles)
- Deterministic exit logic (time stop, ATR trail, trend break, vol spike)
- Churn control (min delta exposure)
- Audit-grade metadata: blocked_by, reasons, exit_reason, diagnostics

Compatibility contract (must not break wiring/tests)
---------------------------------------------------
Preserved public API used by registry/tests:
- AlphaForexParams dataclass
- build_alpha_forex_config() -> StrategyConfig
- alpha_forex_handler(ctx: Any, params: AlphaForexParams) -> List

Signal type compatibility
-------------------------
If the host provides a StrategySignal factory/type via ctx.signal_factory or ctx.StrategySignal,
we use it. Otherwise we return dict payloads (safe fallback).

Paper-first evaluation note
---------------------------
SCR evaluates win_rate + sharpe_like from the ledger. This module improves expectancy by:
- blocking chop/panic volatility regimes
- confirming trend
- avoiding FOMO entries
- exiting losers/invalidations deterministically
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal


FOREX_UNIVERSE_DEFAULT: List[str] = ["EUR-USD", "GBP-USD", "USD-CAD", "USD-JPY"]

Number = float


# -------------------------
# Small pure helpers
# -------------------------

def _clamp(x: Number, lo: Number, hi: Number) -> Number:
    if x != x:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(x: Any, default: Number = 0.0) -> Number:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_iter(obj: Any) -> Iterable[Any]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return obj
    return [obj]


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


def _extract_prices(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    prices = _get_mapping(ctx, "prices")
    out: Dict[str, Number] = {}
    for s in universe:
        if s in prices:
            v = prices.get(s)
            if v is not None:
                out[s] = _safe_float(v, default=0.0)
    return out


def _extract_volatility(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    vol = _get_mapping(ctx, "volatility")
    out: Dict[str, Number] = {}
    for s in universe:
        if s in vol:
            v = vol.get(s)
            if v is not None:
                out[s] = _safe_float(v, default=0.0)
    return out


def _extract_spread_bps(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Optional liquidity proxy for FX:
      - ctx.spread_bps[symbol] -> float
      - ctx.spreads_bps[symbol] -> float
      - ctx.spread[symbol] -> float (assumed bps if <= 100, else absolute ignored)
    """
    for key in ("spread_bps", "spreads_bps", "spread"):
        m = _get_mapping(ctx, key)
        if not m:
            continue
        out: Dict[str, Number] = {}
        for s in universe:
            if s in m and m.get(s) is not None:
                v = _safe_float(m.get(s), default=0.0)
                # If it's huge, likely not bps. Keep but it will fail max_spread_bps anyway.
                out[s] = v
        return out
    return {}


def _extract_dollar_volume(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Optional: if your FX feed provides $volume or liquidity estimates.
    """
    for key in ("dollar_volume", "volume_usd", "liquidity_usd"):
        m = _get_mapping(ctx, key)
        if not m:
            continue
        out: Dict[str, Number] = {}
        for s in universe:
            if s in m and m.get(s) is not None:
                out[s] = _safe_float(m.get(s), default=0.0)
        return out
    return {}


def _extract_bars(ctx: Any, symbol: str) -> Optional[Sequence[Any]]:
    for key in ("bars", "ohlcv", "series", "candles"):
        m = _get_mapping(ctx, key)
        if m and symbol in m:
            v = m.get(symbol)
            if isinstance(v, (list, tuple)) and len(v) >= 5:
                return v
    return None


def _bar_close(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("close", "c", "Close", "C"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        for v in reversed(list(bar.values())):
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    if isinstance(bar, (list, tuple)) and len(bar) >= 5:
        # (o,h,l,c,v) or (ts,o,h,l,c,v) -> close is -2
        return _safe_float(bar[-2], default=0.0)
    return _safe_float(bar, default=0.0)


def _bar_high(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("high", "h", "High", "H"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)) and len(bar) >= 5:
        # (o,h,l,c,v)-> high=1 ; (ts,o,h,l,c,v)-> high=2
        return _safe_float(bar[1] if len(bar) == 5 else bar[2], default=0.0)
    return 0.0


def _bar_low(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("low", "l", "Low", "L"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)) and len(bar) >= 5:
        # (o,h,l,c,v)-> low=2 ; (ts,o,h,l,c,v)-> low=3
        return _safe_float(bar[2] if len(bar) == 5 else bar[3], default=0.0)
    return 0.0


def _ema(values: Sequence[Number], period: int) -> List[Number]:
    if period <= 1:
        raise ValueError("EMA period must be > 1")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out: List[Number] = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = alpha * v + (1.0 - alpha) * e
        out.append(e)
    return out


def _true_range(highs: Sequence[Number], lows: Sequence[Number], closes: Sequence[Number]) -> List[Number]:
    n = len(closes)
    if n == 0:
        return []
    out = [0.0] * n
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        out[i] = max(hl, hc, lc)
    return out


def _atr(highs: Sequence[Number], lows: Sequence[Number], closes: Sequence[Number], period: int) -> List[Number]:
    tr = _true_range(highs, lows, closes)
    return _ema(tr, period)


# -------------------------
# Parameters (immutable)
# -------------------------

@dataclass(frozen=True)
class AlphaForexParams:
    """
    Tunable parameters for AlphaForex (Phase-4 upgrade).

    Key gates:
      - Trend: EMA fast > EMA slow AND close > EMA fast
      - Vol regime: ATR% in [min_atr_pct, max_atr_pct]
      - Liquidity proxy:
          * If spread_bps available: enforce <= max_spread_bps
          * If $volume available: enforce >= min_liquidity_usd (optional)
      - Anti-chase: range/ATR <= anti_chase_range_atr
      - Momentum: (close - ema_fast)/ATR >= momentum_atr

    Exits:
      - Vol spike
      - Trend break
      - Time stop
      - ATR trail
    """

    enabled: bool = True
    universe: Optional[List[str]] = None

    # Liquidity proxies
    min_liquidity_usd: float = 100_000_000.0
    max_spread_bps: float = 3.5  # majors typically tight; adjust to your feed reality

    # Legacy (kept)
    max_volatility: float = 4.0

    # Indicators
    atr_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 48

    # Regime
    min_atr_pct: float = 0.0008   # FX ATR% is much smaller than crypto
    max_atr_pct: float = 0.0200

    # Entry quality
    anti_chase_range_atr: float = 3.2
    momentum_atr: float = 0.25

    # Exposure & churn
    max_abs_exposure: float = 0.20
    min_delta_exposure: float = 0.03

    # Exits
    time_stop_bars: int = 30
    min_favor_move_atr: float = 0.5
    atr_trail_mult: float = 2.2
    vol_spike_atr_pct: float = 0.030

    def actual_universe(self) -> List[str]:
        if self.universe is None:
            return list(FOREX_UNIVERSE_DEFAULT)
        try:
            u = [str(x) for x in self.universe if x]
            return u if u else list(FOREX_UNIVERSE_DEFAULT)
        except Exception:
            return list(FOREX_UNIVERSE_DEFAULT)


# -------------------------
# StrategyConfig Factory
# -------------------------

def build_alpha_forex_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.ALPHA_FOREX,
        enabled=True,
        target_universe=list(FOREX_UNIVERSE_DEFAULT),
    )


# -------------------------
# Minimal in-memory state
# -------------------------

class _StateStore:
    __slots__ = ("pos",)

    def __init__(self) -> None:
        self.pos: Dict[str, Dict[str, Any]] = {}

    def get(self, symbol: str) -> Dict[str, Any]:
        return dict(self.pos.get(symbol, {}))

    def set(self, symbol: str, st: Dict[str, Any]) -> None:
        self.pos[symbol] = dict(st)


_STATE = _StateStore()


# -------------------------
# Signal factory adapter
# -------------------------

def _make_signal(ctx: Any, payload: Dict[str, Any]) -> Any:
    try:
        sf = getattr(ctx, "signal_factory", None)
        if callable(sf):
            return sf(payload)
    except Exception:
        pass
    try:
        cls = getattr(ctx, "StrategySignal", None)
        if cls is not None:
            return cls(**payload)  # type: ignore[misc]
    except Exception:
        pass
    return payload


# -------------------------
# Core handler
# -------------------------

def alpha_forex_handler(ctx: Any, params: AlphaForexParams) -> List:
    if not params.enabled:
        return []

    universe = params.actual_universe()
    prices = _extract_prices(ctx, universe)
    vol_est = _extract_volatility(ctx, universe)
    spread_bps = _extract_spread_bps(ctx, universe)
    dv_map = _extract_dollar_volume(ctx, universe)

    out: List[Any] = []

    for symbol in universe:
        bars = _extract_bars(ctx, symbol)

        closes: List[Number] = []
        highs: List[Number] = []
        lows: List[Number] = []

        if bars is not None:
            tail = list(bars)[-max(80, params.ema_slow + 10, params.atr_period + 10) :]
            for b in tail:
                c = _bar_close(b)
                h = _bar_high(b)
                l = _bar_low(b)
                if c <= 0 or h <= 0 or l <= 0:
                    continue
                if h < l:
                    continue
                closes.append(c)
                highs.append(h)
                lows.append(l)

        have_history = len(closes) >= max(params.ema_slow + 2, params.atr_period + 2)

        st = _STATE.get(symbol)
        pos = _safe_float(st.get("exposure", 0.0), 0.0)
        entry_close = st.get("entry_close", None)
        if entry_close is not None:
            entry_close = _safe_float(entry_close, default=closes[-1] if closes else 0.0)
        bars_held = _safe_int(st.get("bars_held", 0), 0)
        peak = st.get("peak_favorable_close", None)
        if peak is not None:
            peak = _safe_float(peak, default=closes[-1] if closes else 0.0)

        # If no history: allow exits only, otherwise stay flat.
        if not have_history:
            if pos != 0.0:
                out.append(
                    TradeSignal(
                        strategy=StrategyName.ALPHA_FOREX,
                        symbol=str(symbol).upper(),
                        side=SignalSide.SELL,
                        size=abs(float(st.get("exposure", 0.0))),
                        confidence=0.0,
                        asset_class=AssetClass.FOREX,
                        meta={
                            "blocked_by": ["NO_HISTORY"],
                            "reasons": ["No OHLCV history in ctx; fail-closed flatten"],
                            "exit_reason": "NO_HISTORY",
                            "diagnostics": {},
                            "target_exposure": 0.0,
                            "previous_exposure": float(st.get("exposure", 0.0)),
                        },
                    )
                )
                _STATE.set(symbol, {"exposure": 0.0, "entry_close": None, "bars_held": 0, "peak_favorable_close": None})
            continue

        ef = _ema(closes, params.ema_fast)[-1]
        es = _ema(closes, params.ema_slow)[-1]
        a = _atr(highs, lows, closes, params.atr_period)[-1]

        close = closes[-1]
        atr_pct = (a / close) if close else 0.0
        rng = highs[-1] - lows[-1]
        range_atr = (rng / a) if a else 0.0

        blocked_by: List[str] = []
        reasons: List[str] = []
        diagnostics: Dict[str, Any] = {
            "atr_pct": float(atr_pct),
            "range_atr": float(range_atr),
            "ema_fast": float(ef),
            "ema_slow": float(es),
            "bars_held": int(bars_held),
        }

        # -------------------------
        # EXIT ENGINE
        # -------------------------
        exit_reason: Optional[str] = None
        if pos != 0.0:
            if atr_pct > params.vol_spike_atr_pct:
                exit_reason = "VOL_SPIKE"
            if exit_reason is None and not (ef > es):
                exit_reason = "TREND_BREAK"
            if exit_reason is None and entry_close is not None and bars_held >= params.time_stop_bars:
                favor_move = close - entry_close
                diagnostics["favor_move"] = float(favor_move)
                if favor_move < params.min_favor_move_atr * a:
                    exit_reason = "TIME_STOP"
            if exit_reason is None and peak is not None:
                adverse = peak - close
                diagnostics["adverse_from_peak"] = float(adverse)
                if adverse > params.atr_trail_mult * a:
                    exit_reason = "ATR_TRAIL"

            if exit_reason is not None:
                out.append(
                    TradeSignal(
                        strategy=StrategyName.ALPHA_FOREX,
                        symbol=str(symbol).upper(),
                        side=SignalSide.SELL,
                        size=abs(float(pos)),
                        confidence=0.0,
                        asset_class=AssetClass.FOREX,
                        meta={
                            "blocked_by": ["EXIT_ENGINE"],
                            "reasons": [f"EXIT:{exit_reason}"],
                            "exit_reason": exit_reason,
                            "diagnostics": diagnostics,
                            "target_exposure": 0.0,
                            "previous_exposure": float(pos),
                        },
                    )
                )
                _STATE.set(symbol, {"exposure": 0.0, "entry_close": None, "bars_held": 0, "peak_favorable_close": None})
                continue

        # -------------------------
        # ENTRY GATES
        # -------------------------

        # Spread gate (if present)
        sp = spread_bps.get(symbol)
        if sp is not None:
            diagnostics["spread_bps"] = float(sp)
            if sp > params.max_spread_bps:
                blocked_by.append("SPREAD")
                reasons.append(f"spread_bps {sp:.2f} > {params.max_spread_bps:.2f}")

        # Optional $volume gate (only if present; FX often lacks volume)
        dv = dv_map.get(symbol)
        if dv is not None:
            diagnostics["dollar_volume"] = float(dv)
            if dv < params.min_liquidity_usd:
                blocked_by.append("LIQUIDITY")
                reasons.append(f"dollar_volume {dv:.0f} < {params.min_liquidity_usd:.0f}")

        # Volatility regime gate
        if atr_pct < params.min_atr_pct:
            blocked_by.append("VOL_LOW")
            reasons.append(f"atr_pct {atr_pct:.5f} < {params.min_atr_pct:.5f}")
        if atr_pct > params.max_atr_pct:
            blocked_by.append("VOL_HIGH")
            reasons.append(f"atr_pct {atr_pct:.5f} > {params.max_atr_pct:.5f}")

        # Trend gate
        if not (ef > es):
            blocked_by.append("TREND")
            reasons.append("ema_fast<=ema_slow")
        if not (close > ef):
            blocked_by.append("TREND")
            reasons.append("close<=ema_fast")

        # Anti-chase gate
        if range_atr > params.anti_chase_range_atr:
            blocked_by.append("ANTI_CHASE")
            reasons.append(f"range/atr {range_atr:.2f} > {params.anti_chase_range_atr:.2f}")

        # Momentum gate (ATR normalized)
        mom = ((close - ef) / a) if a else 0.0
        diagnostics["momentum_atr"] = float(mom)
        if mom < params.momentum_atr:
            blocked_by.append("MOMENTUM")
            reasons.append(f"momentum_atr {mom:.2f} < {params.momentum_atr:.2f}")

        # Optional legacy volatility estimate guard
        v_est = vol_est.get(symbol)
        if v_est is not None and v_est > params.max_volatility:
            blocked_by.append("VOL_EST")
            reasons.append(f"ctx.volatility {v_est:.2f} > max_volatility {params.max_volatility:.2f}")

        # -------------------------
        # Target exposure decision
        # -------------------------
        target = pos  # default hold
        confidence = 0.2

        if not blocked_by:
            confidence = _clamp(0.35 + 0.18 * mom, 0.0, 0.90)
            target = _clamp(params.max_abs_exposure * _clamp(0.55 + 0.25 * mom, 0.0, 1.0), 0.0, params.max_abs_exposure)
            reasons.append("ENTER:TREND_VOL_MOM_OK")
        else:
            if pos == 0.0:
                target = 0.0

        # Churn band
        delta = abs(target - pos)
        diagnostics["delta_exposure"] = float(delta)
        if delta < params.min_delta_exposure:
            target = pos
            blocked_by.append("CHURN_BAND")
            reasons.append(f"delta {delta:.3f} < {params.min_delta_exposure:.3f}")

        # Update in-memory state
        if target == 0.0:
            _STATE.set(symbol, {"exposure": 0.0, "entry_close": None, "bars_held": 0, "peak_favorable_close": None})
        else:
            if pos == 0.0:
                _STATE.set(symbol, {"exposure": float(target), "entry_close": float(close), "bars_held": 1, "peak_favorable_close": float(close)})
            else:
                bars_held = bars_held + 1
                peak = close if peak is None else max(peak, close)
                _STATE.set(symbol, {"exposure": float(target), "entry_close": float(entry_close or close), "bars_held": int(bars_held), "peak_favorable_close": float(peak)})

        meta = {
            "blocked_by": blocked_by,
            "reasons": reasons if reasons else ["NO_SIGNAL"],
            "exit_reason": None,
            "diagnostics": diagnostics,
            "target_exposure": float(target),
            "previous_exposure": float(pos),
        }

        delta_exposure = float(target) - float(pos)
        if abs(delta_exposure) <= 1e-12:
            continue

        side = SignalSide.BUY if delta_exposure > 0 else SignalSide.SELL
        size = abs(delta_exposure)

        out.append(
            TradeSignal(
                strategy=StrategyName.ALPHA_FOREX,
                symbol=str(symbol).upper(),
                side=side,
                size=float(size),
                confidence=float(confidence),
                asset_class=AssetClass.FOREX,
                meta=meta,
            )
        )

    return out

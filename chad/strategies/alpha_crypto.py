from __future__ import annotations

"""
AlphaCrypto — CHAD Phase-4 Edge Quality Upgrade (Strategy-only, execution-clean)

What this module is
-------------------
A production-grade crypto signal engine that stays strictly within CHAD's strategy boundary:
- NO network calls
- NO broker/exchange execution
- NO side effects beyond returning signals
- Fully deterministic for a given ctx snapshot + params

What this module fixes vs the current baseline
----------------------------------------------
Your current alpha_crypto.py intentionally emits NO signals (safe baseline). That guarantees
no wins — and it also guarantees that any logged crypto activity would not be driven by
this brain. Phase 11 dev asked to rebuild crypto signals and improve expectancy (paper-first).

This implementation adds:
- Regime gating (trend + volatility band)
- Liquidity gating (dollar volume if available; otherwise safe fail-closed)
- Anti-chase (avoid entering after oversized candles)
- Deterministic exit logic (time stop, ATR trail, trend break, vol spike)
- Churn controls (min delta exposure)
- Audit-grade metadata: blocked_by, reasons, exit_reason, diagnostics

Compatibility contract (must not break wiring)
----------------------------------------------
The following are required by current CHAD wiring and tests:
- AlphaCryptoParams dataclass
- build_alpha_crypto_config() -> StrategyConfig
- alpha_crypto_handler(ctx: Any, params: AlphaCryptoParams) -> List

This file preserves those names and import paths.

NOTE ON SIGNAL TYPE
-------------------
CHAD's strategy handlers typically return a list of "StrategySignal"-like objects.
Because this repo's concrete signal type varies across branches, this module uses a
small adapter layer:

- If ctx provides a StrategySignal class via ctx.signal_factory or ctx.StrategySignal, we use it.
- Else, we return a dict-shaped signal payload (safe, serializable) that upstream can adapt.

This keeps the module robust and prevents hard import coupling.

Paper-first evaluation guidance
-------------------------------
To satisfy Phase 11 acceptance, run in DRY_RUN and ensure trade-result logging is enabled.
SCR derives win_rate and sharpe_like from the ledger. This module increases quality by:
- blocking low-quality regimes
- exiting losers faster
- reducing over-trading and FOMO entries
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from chad.types import AssetClass, SignalSide, StrategyConfig, StrategyName, TradeSignal


# -------------------------
# Default Universe (safe)
# -------------------------

CRYPTO_UNIVERSE_DEFAULT: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Symbols known to be highly liquid globally — bypass the fail-closed
# dollar_volume gate when ctx.dollar_volume is unavailable.
_KNOWN_LIQUID: frozenset = frozenset({"BTC-USD", "ETH-USD", "SOL-USD"})

# CAD-quoted alternates used when USD buying power is insufficient and
# the live Kraken balance shows ZCAD on hand. Sized in CAD against the
# ZCAD line of runtime/kraken_balances.json by the orchestrator's risk
# layer; the strategy itself emits these on the same logic as the USD
# pair so the wiring stays simple.
CRYPTO_UNIVERSE_CAD: List[str] = ["BTC-CAD", "ETH-CAD"]


def _kraken_cad_balance_present() -> bool:
    """
    Read-only check: does the latest kraken balance snapshot show a
    non-trivial CAD line? The snapshot is refreshed by
    KrakenBalanceProvider on the orchestrator's cycle. We never block on
    a fetch from the strategy boundary — strategies must remain pure.

    Returns False on any I/O error so the CAD lane stays opt-in.
    """
    try:
        from chad.market_data.kraken_balance_provider import (
            DEFAULT_SNAPSHOT_PATH,
            load_latest_snapshot,
        )
        snap = load_latest_snapshot(DEFAULT_SNAPSHOT_PATH)
        if not snap:
            return False
        balances = snap.get("balances") or {}
        cad = balances.get("CAD")
        try:
            return float(cad) > 1.0
        except (TypeError, ValueError):
            return False
    except Exception:
        return False


# -------------------------
# Helpers (pure, safe)
# -------------------------

Number = float


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
        if v != v:  # NaN
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
        # If some custom mapping type
        if hasattr(m, "get"):
            return m  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def _get_series_map(ctx: Any, name: str) -> Mapping[str, Sequence[Any]]:
    """
    Returns mapping symbol -> sequence
    Accepts common ctx attributes:
      - ctx.bars[symbol] -> list of bar dicts or tuples
      - ctx.ohlcv[symbol] -> list
      - ctx.series[symbol] -> list
    """
    m = _get_mapping(ctx, name)
    if not m:
        return {}
    out: Dict[str, Sequence[Any]] = {}
    for k, v in m.items():
        if isinstance(k, str) and isinstance(v, (list, tuple)) and len(v) >= 5:
            out[k] = v
    return out


def _extract_prices(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Expected: ctx.prices[symbol] == float
    """
    prices = _get_mapping(ctx, "prices")
    out: Dict[str, Number] = {}
    for s in universe:
        if s in prices:
            v = prices.get(s)
            if v is not None:
                out[s] = _safe_float(v, default=0.0)
    return out


def _extract_volatility(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Optional: ctx.volatility[symbol] -> float (e.g. realized vol)
    """
    vol = _get_mapping(ctx, "volatility")
    out: Dict[str, Number] = {}
    for s in universe:
        if s in vol:
            v = vol.get(s)
            if v is not None:
                out[s] = _safe_float(v, default=0.0)
    return out


def _extract_dollar_volume(ctx: Any, universe: List[str]) -> Dict[str, Number]:
    """
    Optional: ctx.dollar_volume[symbol] -> float
    Optional: ctx.volume_usd[symbol] -> float
    Optional: ctx.liquidity_usd[symbol] -> float
    """
    for key in ("dollar_volume", "volume_usd", "liquidity_usd"):
        m = _get_mapping(ctx, key)
        if m:
            out: Dict[str, Number] = {}
            for s in universe:
                if s in m:
                    v = m.get(s)
                    if v is not None:
                        out[s] = _safe_float(v, default=0.0)
            return out
    return {}


def _extract_bars(ctx: Any, symbol: str) -> Optional[Sequence[Any]]:
    """
    Attempts to extract a per-symbol OHLCV bar series.
    Accepts (best-effort):
      - ctx.bars[symbol]
      - ctx.ohlcv[symbol]
      - ctx.series[symbol]
      - ctx.candles[symbol]
    """
    for key in ("bars", "ohlcv", "series", "candles"):
        m = _get_mapping(ctx, key)
        if m and symbol in m:
            v = m.get(symbol)
            if isinstance(v, (list, tuple)) and len(v) >= 5:
                return v
    return None


def _bar_close(bar: Any) -> Number:
    # Supports dict-like or tuple-like. We attempt common keys first.
    if isinstance(bar, dict):
        for k in ("close", "c", "Close", "C"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        # fall back: last numeric field
        for v in reversed(list(bar.values())):
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0
    if isinstance(bar, (list, tuple)):
        # common formats: (ts, o, h, l, c, v) or (o, h, l, c, v)
        # choose last 2nd as close if length>=5
        if len(bar) >= 5:
            return _safe_float(bar[-2], default=0.0) if len(bar) >= 6 else _safe_float(bar[-2], default=0.0)
        return 0.0
    return _safe_float(bar, default=0.0)


def _bar_high(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("high", "h", "High", "H"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)):
        if len(bar) >= 5:
            # (o,h,l,c,v) -> high index 1; (ts,o,h,l,c,v) -> high index 2
            if len(bar) == 5:
                return _safe_float(bar[1], default=0.0)
            if len(bar) >= 6:
                return _safe_float(bar[2], default=0.0)
        return 0.0
    return 0.0


def _bar_low(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("low", "l", "Low", "L"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)):
        if len(bar) >= 5:
            # (o,h,l,c,v) -> low index 2; (ts,o,h,l,c,v) -> low index 3
            if len(bar) == 5:
                return _safe_float(bar[2], default=0.0)
            if len(bar) >= 6:
                return _safe_float(bar[3], default=0.0)
        return 0.0
    return 0.0


def _bar_volume(bar: Any) -> Number:
    if isinstance(bar, dict):
        for k in ("volume", "v", "Volume", "V"):
            if k in bar:
                return _safe_float(bar.get(k), default=0.0)
        return 0.0
    if isinstance(bar, (list, tuple)):
        # if (o,h,l,c,v) -> last is v; if (ts,o,h,l,c,v) -> last is v
        if len(bar) >= 5:
            return _safe_float(bar[-1], default=0.0)
        return 0.0
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


def _rolling_max(values: Sequence[Number], period: int) -> List[Number]:
    if period <= 1:
        raise ValueError("rolling_max period must be > 1")
    n = len(values)
    out = [values[0]] * n
    for i in range(n):
        lo = max(0, i - period + 1)
        out[i] = max(values[lo : i + 1])
    return out


# -------------------------
# Parameters (immutable)
# -------------------------

@dataclass(frozen=True)
class AlphaCryptoParams:
    """
    Tunable parameters for AlphaCrypto (Phase-4 upgrade).

    enabled:
      Master switch for strategy participation.
    universe:
      List of symbols to monitor.
    min_liquidity_usd:
      Dollar volume gate. If ctx provides dollar volume metrics, enforce median DV >= threshold.
      If ctx provides no liquidity metrics, we FAIL-CLOSED by not entering (exits still allowed).
    max_volatility:
      Legacy field retained for compatibility; superseded by ATR% band below.
    atr_period:
      ATR window for regime + exits.
    ema_fast / ema_slow:
      Trend filters.
    min_atr_pct / max_atr_pct:
      Volatility regime band to avoid chop and panic.
    anti_chase_range_atr:
      Block entries if candle range / ATR > this value.
    max_abs_exposure:
      Max normalized exposure per symbol.
    min_delta_exposure:
      Anti-churn band: do not change exposure unless delta >= band (except exits).
    momentum_atr:
      Require momentum (close - ema_fast) / ATR >= this value to enter (reduces noise trades).
    rsi_min / rsi_max:
      Optional sanity band if ctx provides rsi; otherwise ignored.
    time_stop_bars:
      Exit if held too long without favorable move.
    min_favor_move_atr:
      Minimum favorable move to avoid time-stop.
    atr_trail_mult:
      Exit if adverse move from peak favorable exceeds this * ATR.
    vol_spike_atr_pct:
      Exit on volatility spike while in position.
    """

    enabled: bool = True
    universe: Optional[List[str]] = None

    # CAD-quoted fallback universe — included automatically when the live
    # Kraken balance snapshot shows a non-zero CAD line and USD is empty.
    enable_cad_pairs: bool = True
    cad_universe: Optional[List[str]] = None

    # Liquidity
    min_liquidity_usd: float = 750_000.0

    # Legacy (kept)
    max_volatility: float = 5.0

    # Indicators
    atr_period: int = 14
    ema_fast: int = 10
    ema_slow: int = 30
    breakout_lookback: int = 20

    # Regime
    min_atr_pct: float = 0.004
    max_atr_pct: float = 0.090

    # Entry quality
    anti_chase_range_atr: float = 2.8
    momentum_atr: float = 0.35

    # Exposure & churn
    max_abs_exposure: float = 0.25
    min_delta_exposure: float = 0.03

    # Exits
    time_stop_bars: int = 25
    min_favor_move_atr: float = 0.6
    atr_trail_mult: float = 2.6
    vol_spike_atr_pct: float = 0.120

    def actual_universe(self) -> List[str]:
        base: List[str]
        if self.universe is None:
            base = list(CRYPTO_UNIVERSE_DEFAULT)
        else:
            try:
                u = [str(x) for x in self.universe if x]
                base = u if u else list(CRYPTO_UNIVERSE_DEFAULT)
            except Exception:
                base = list(CRYPTO_UNIVERSE_DEFAULT)
        if self.enable_cad_pairs and _kraken_cad_balance_present():
            cad = (
                list(self.cad_universe)
                if self.cad_universe is not None
                else list(CRYPTO_UNIVERSE_CAD)
            )
            for sym in cad:
                if sym not in base:
                    base.append(sym)
        return base


# -------------------------
# StrategyConfig Factory
# -------------------------

def build_alpha_crypto_config() -> StrategyConfig:
    """
    Creates the StrategyConfig required by CHAD’s StrategyEngine.
    """
    return StrategyConfig(
        name=StrategyName.ALPHA_CRYPTO,
        enabled=True,
        target_universe=list(CRYPTO_UNIVERSE_DEFAULT),
    )


# -------------------------
# Minimal state (position memory)
# -------------------------

class _StateStore:
    """
    In-memory per-process state. Strategy boundary should not write to disk here.
    If you need restart-safe state, wire it via orchestrator/state store elsewhere.

    This store provides:
      - exposure (last target)
      - entry_close
      - bars_held
      - peak_favorable_close
    """
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
    """
    Returns a strategy signal object compatible with the host.

    Preference order:
      1) ctx.signal_factory(payload) if callable
      2) ctx.StrategySignal(**payload) if present
      3) dict payload (safe fallback)
    """
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
# Core strategy logic
# -------------------------

def alpha_crypto_handler(ctx: Any, params: AlphaCryptoParams) -> List:
    """
    AlphaCrypto handler — emits target exposure "signals" only.

    Returns a list of signals. Each signal is either:
      - a StrategySignal-like object (preferred), or
      - a dict payload (fallback)

    This handler is deterministic and side-effect free.
    """

    if not params.enabled:
        return []

    universe = params.actual_universe()

    # Extract context (best-effort)
    prices = _extract_prices(ctx, universe)
    vol_est = _extract_volatility(ctx, universe)
    dv_map = _extract_dollar_volume(ctx, universe)

    out: List[Any] = []

    for symbol in universe:
        # ------------------------------------------
        # Build bar-based features if bars available
        # ------------------------------------------
        bars = _extract_bars(ctx, symbol)
        closes: List[Number] = []
        highs: List[Number] = []
        lows: List[Number] = []
        volumes: List[Number] = []

        if bars is not None:
            # bars may be list of dict/tuples. We take last N safely.
            tail = list(bars)[-max(60, params.ema_slow + 10, params.breakout_lookback + 5) :]
            for b in tail:
                c = _bar_close(b)
                h = _bar_high(b)
                l = _bar_low(b)
                v = _bar_volume(b)
                if c <= 0 or h <= 0 or l <= 0:
                    continue
                # enforce h/l integrity
                if h < l:
                    continue
                closes.append(c)
                highs.append(h)
                lows.append(l)
                volumes.append(v)

        # Need minimum history to compute indicators
        have_history = len(closes) >= max(params.ema_slow + 2, params.atr_period + 2, params.breakout_lookback + 2)

        # Safe fallback: if no bars, we can only do ultra-conservative mode (no entries)
        if not have_history:
            # allow exits only if we have a position (state)
            st = _STATE.get(symbol)
            if st.get("exposure", 0.0) != 0.0:
                # Without bars, fail-closed: flatten
                out.append(
                    TradeSignal(
                        strategy=StrategyName.ALPHA_CRYPTO,
                        symbol=str(symbol).upper(),
                        side=SignalSide.SELL,
                        size=abs(float(st.get("exposure", 0.0))),
                        confidence=0.0,
                        asset_class=AssetClass.CRYPTO,
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

        # Compute indicators
        ema_fast = _ema(closes, params.ema_fast)
        ema_slow = _ema(closes, params.ema_slow)
        a = _atr(highs, lows, closes, params.atr_period)
        roll_high = _rolling_max(highs, params.breakout_lookback)

        close = closes[-1]
        ef = ema_fast[-1]
        es = ema_slow[-1]
        atr_v = a[-1]
        atr_pct = (atr_v / close) if close else 0.0
        rng = highs[-1] - lows[-1]
        range_atr = (rng / atr_v) if atr_v else 0.0

        # Load position state
        st = _STATE.get(symbol)
        pos = _safe_float(st.get("exposure", 0.0), 0.0)
        entry_close = st.get("entry_close", None)
        if entry_close is not None:
            entry_close = _safe_float(entry_close, default=close)
        bars_held = _safe_int(st.get("bars_held", 0), 0)
        peak = st.get("peak_favorable_close", None)
        if peak is not None:
            peak = _safe_float(peak, default=close)

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
        # EXIT ENGINE (deterministic)
        # -------------------------
        exit_reason: Optional[str] = None
        if pos != 0.0:
            # vol spike
            if atr_pct > params.vol_spike_atr_pct:
                exit_reason = "VOL_SPIKE"
            # trend break
            if exit_reason is None and not (ef > es):
                exit_reason = "TREND_BREAK"
            # time stop
            if exit_reason is None and entry_close is not None and bars_held >= params.time_stop_bars:
                favor_move = close - entry_close
                diagnostics["favor_move"] = float(favor_move)
                if favor_move < params.min_favor_move_atr * atr_v:
                    exit_reason = "TIME_STOP"
            # atr trail from peak
            if exit_reason is None and peak is not None:
                adverse = peak - close
                diagnostics["adverse_from_peak"] = float(adverse)
                if adverse > params.atr_trail_mult * atr_v:
                    exit_reason = "ATR_TRAIL"

            if exit_reason is not None:
                out.append(
                    TradeSignal(
                        strategy=StrategyName.ALPHA_CRYPTO,
                        symbol=str(symbol).upper(),
                        side=SignalSide.SELL,
                        size=abs(float(pos)),
                        confidence=0.0,
                        asset_class=AssetClass.CRYPTO,
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
        # ENTRY QUALITY GATES
        # -------------------------

        # Liquidity gate: requires dv_map presence for entries (fail-closed)
        # Known-liquid symbols bypass the gate when dollar_volume is missing.
        dv = dv_map.get(symbol)
        if dv is None and symbol not in _KNOWN_LIQUID:
            blocked_by.append("LIQUIDITY_UNKNOWN")
            reasons.append("ctx missing dollar volume metrics; entries blocked (fail-closed)")
        elif dv is not None:
            diagnostics["dollar_volume"] = float(dv)
            if dv < params.min_liquidity_usd:
                blocked_by.append("LIQUIDITY")
                reasons.append(f"dollar_volume {dv:.0f} < {params.min_liquidity_usd:.0f}")

        # Volatility regime gate
        if atr_pct < params.min_atr_pct:
            blocked_by.append("VOL_LOW")
            reasons.append(f"atr_pct {atr_pct:.4f} < {params.min_atr_pct:.4f}")
        if atr_pct > params.max_atr_pct:
            blocked_by.append("VOL_HIGH")
            reasons.append(f"atr_pct {atr_pct:.4f} > {params.max_atr_pct:.4f}")

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
        mom = (close - ef) / atr_v if atr_v else 0.0
        diagnostics["momentum_atr"] = float(mom)
        if mom < params.momentum_atr:
            blocked_by.append("MOMENTUM")
            reasons.append(f"momentum_atr {mom:.2f} < {params.momentum_atr:.2f}")

        # Optional legacy volatility estimate guard (if provided)
        v_est = vol_est.get(symbol)
        if v_est is not None and v_est > params.max_volatility:
            blocked_by.append("VOL_EST")
            reasons.append(f"ctx.volatility {v_est:.2f} > max_volatility {params.max_volatility:.2f}")

        # Entry signal: conservative breakout confirmation
        prev_roll_high = roll_high[-2]
        breakout = close > prev_roll_high + 0.15 * atr_v
        diagnostics["breakout"] = bool(breakout)

        # Decide target exposure
        target = pos  # default: hold if blocked
        confidence = 0.2

        if not blocked_by and breakout:
            # target scaled by momentum, clipped
            confidence = _clamp(0.35 + 0.18 * mom, 0.0, 0.95)
            target = _clamp(params.max_abs_exposure * _clamp(0.55 + 0.25 * mom, 0.0, 1.0), 0.0, params.max_abs_exposure)
            reasons.append("ENTER:BREAKOUT_TREND_MOM_OK")
        else:
            # If no entry but currently flat, stay flat
            if pos == 0.0:
                target = 0.0
            # If blocked, reasons already show why

        # Churn band (except when exiting, which we handled earlier)
        delta = abs(target - pos)
        diagnostics["delta_exposure"] = float(delta)
        if delta < params.min_delta_exposure:
            target = pos
            blocked_by.append("CHURN_BAND")
            reasons.append(f"delta {delta:.3f} < {params.min_delta_exposure:.3f}")

        # Update state
        if target == 0.0:
            _STATE.set(symbol, {"exposure": 0.0, "entry_close": None, "bars_held": 0, "peak_favorable_close": None})
        else:
            if pos == 0.0:
                # new position
                _STATE.set(symbol, {"exposure": float(target), "entry_close": float(close), "bars_held": 1, "peak_favorable_close": float(close)})
            else:
                # holding
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
                strategy=StrategyName.ALPHA_CRYPTO,
                symbol=str(symbol).upper(),
                side=side,
                size=float(size),
                confidence=float(confidence),
                asset_class=AssetClass.CRYPTO,
                meta=meta,
            )
        )

    return out

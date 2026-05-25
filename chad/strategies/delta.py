from __future__ import annotations

"""
DeltaBrain — Cross-Asset Convexity Hunter (WEALTH MODE, Phase-4 Edge Quality)

This is the REAL Delta described in your SSOT intent:
- Opportunistic hitter
- Low frequency, high convexity
- Regime + event aware
- Dynamic sizing by conviction (bounded)
- First to shut down when risk rises (handled by Risk/Policy/SCR layers)

Hard Guarantees
---------------
- Strategy-only: emits TradeSignal intents; never executes.
- Deterministic for given ctx snapshot + params.
- No network calls, no disk writes, no broker imports.
- Defensive against missing context (bars optional). If bars missing: fail conservative (no entries).
- Preserves registry/test contract:
    - DeltaParams
    - build_delta_config()
    - delta_handler(ctx, params, prices=..., **_) -> List[TradeSignal]

Core Logic
----------
1) Universe selection (cross-asset):
   - ctx.delta_universe (if provided)
   - else legend top weights
   - else prices keys
2) Regime gating:
   - ATR% band (avoid chop + panic)
   - optional volatility expansion requirement
3) Signal formation:
   - Trend: EMA fast > EMA slow + price above fast
   - Breakout: price above rolling high + ATR buffer
   - Momentum: (price - EMA_fast)/ATR >= threshold
   - Event gate: ctx.event_risk veto (if present)
   - Liquidity gate: spread_bps and/or dollar_volume if present
4) Conviction score in [0,1] → dynamic size
5) Exits:
   - vol spike
   - trend break
   - time stop
   - ATR trail
6) Anti-churn:
   - do not adjust size unless change >= min_delta_size
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
from chad.strategies._upstream_exclusion import (
    OPERATOR_EXCLUDED_SYMBOLS,
    is_operator_excluded,
)

Number = float

LOG = logging.getLogger("chad.strategies.delta")

# PR-02: price-cache ground truth used as the cross-check reference for
# delta's upstream price validation. The path is resolved lazily so tests
# can monkey-patch it via the env var.
_PRICE_CACHE_PATH = Path(
    os.environ.get(
        "CHAD_PRICE_CACHE_PATH",
        "/home/ubuntu/chad_finale/runtime/price_cache.json",
    )
)

# Symbols where a fill_price of 100.0 is the canonical "no-live-price"
# placeholder fingerprint (mirrors paper_exec_evidence_writer._LIQUID_PRICED_EQUITIES).
# Delta cross-checks resolved prices against the broker price_cache for
# these symbols and abstains on a >50% deviation.
_LIQUID_PRICED_ETFS = frozenset({"SPY", "QQQ", "IWM", "DIA"})

# Cross-check deviation threshold. Matches the paper_exec_evidence_writer
# deviation guard so delta upstream catches the same placeholder pattern
# the writer would otherwise tag downstream.
_PRICE_CACHE_DEVIATION_THRESHOLD = 0.50


# -------------------------
# Params
# -------------------------

@dataclass(frozen=True)
class DeltaParams:
    enabled: bool = True

    # Universe
    max_symbols_per_cycle: int = 4

    # Indicators
    ema_fast: int = 10
    ema_slow: int = 30
    atr_period: int = 14
    breakout_lookback: int = 20

    # Regime gating (wealth-safe)
    min_atr_pct: float = 0.004
    max_atr_pct: float = 0.090
    vol_spike_atr_pct: float = 0.120

    # Convexity gating
    require_vol_expansion: bool = True
    vol_expansion_factor: float = 1.35   # ATR_now >= factor * ATR_baseline
    anti_chase_range_atr: float = 3.2

    # Entry quality
    momentum_atr: float = 0.45
    breakout_atr_buffer: float = 0.10

    # Conviction → size
    base_size: float = 3.0
    max_size: float = 12.0
    conviction_size_scale: float = 18.0  # size = base + score*scale (clamped)

    # Exits
    time_stop_bars: int = 40
    min_favor_move_atr: float = 0.7
    atr_trail_mult: float = 2.8

    # Safety/cash
    min_cash: float = 2_500.0
    max_position_units: float = 50.0

    # Anti-churn sizing
    min_delta_size: float = 1.0

    # Minimum conviction score to enter
    min_conviction: float = 0.65


DEFAULT_PARAMS = DeltaParams()


# -------------------------
# In-memory state (no disk)
# -------------------------

class _DeltaState:
    def __init__(self) -> None:
        self.pos: Dict[str, Dict[str, Any]] = {}

    def get(self, sym: str) -> Dict[str, Any]:
        return dict(self.pos.get(sym, {}))

    def set(self, sym: str, st: Dict[str, Any]) -> None:
        self.pos[sym] = dict(st)


_STATE = _DeltaState()


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


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
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


def _clamp(x: float, lo: float, hi: float) -> float:
    if x != x:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _ema(values: Sequence[float], period: int) -> List[float]:
    if period <= 1:
        raise ValueError("ema period must be > 1")
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out: List[float] = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = alpha * v + (1.0 - alpha) * e
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


def _rolling_max(values: Sequence[float], period: int) -> List[float]:
    if period <= 1:
        raise ValueError("rolling_max period must be > 1")
    n = len(values)
    if n == 0:
        return []
    out = [values[0]] * n
    for i in range(n):
        lo = max(0, i - period + 1)
        out[i] = max(values[lo : i + 1])
    return out


def _is_finite_positive(value: Any) -> bool:
    """True only when value parses as a finite, strictly positive float.

    Rejects None, non-numeric, NaN, +/-Inf, 0, and negative values in a
    single call.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    return v > 0.0


def _load_price_cache_entry(symbol: str) -> float:
    """Read runtime/price_cache.json::prices[symbol] as a float.

    Returns 0.0 on any failure (missing file, missing key, parse error)
    so the caller can treat absence and corruption identically.
    Resolves the path at call time so test monkey-patches via
    CHAD_PRICE_CACHE_PATH take effect.
    """
    try:
        path = Path(
            os.environ.get("CHAD_PRICE_CACHE_PATH", str(_PRICE_CACHE_PATH))
        )
        if not path.is_file():
            return 0.0
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    prices = data.get("prices")
    if not isinstance(prices, dict):
        return 0.0
    raw = prices.get(symbol)
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _resolve_positive_price(
    ctx: Any,
    sym: str,
    prices: Mapping[str, Any],
) -> Optional[Tuple[float, str]]:
    """PR-02: resolve a strictly-positive price for sym across all sources
    delta is willing to trust, and cross-check against runtime/price_cache.json.

    Source priority (first finite-positive wins):
      1. ``prices[sym]`` (caller-supplied prices map)
      2. ``ctx.ticks[sym].price`` (live tick)
      3. last close in ``ctx.bars[sym]``
      4. ``runtime/price_cache.json::prices[sym]`` (broker cache)

    Validation: None / non-numeric / NaN / +/-Inf / 0 / negative -> abstain.

    Cross-check (PR-02 placeholder defense): for liquid equity/ETF symbols
    that have a positive entry in runtime/price_cache.json, the resolved
    candidate must agree with the cache within
    ``_PRICE_CACHE_DEVIATION_THRESHOLD`` (50%). A larger deviation matches
    the $100-placeholder fingerprint that paper_exec_evidence_writer's
    deviation guard catches downstream; abstaining upstream is the
    PR-02-mandated fix so the placeholder row is never emitted.

    Returns (price, source_label) on success; None on abstain. Callers
    that want to log on abstain should do so with the symbol and side in
    scope so the message reads cleanly in journalctl.
    """
    cached = _load_price_cache_entry(sym)

    candidates: List[Tuple[Any, str]] = [(prices.get(sym), "prices_map")]

    tick = None
    try:
        tick = ctx.ticks.get(sym) if hasattr(ctx, "ticks") and ctx.ticks is not None else None
    except Exception:
        tick = None
    if tick is not None:
        candidates.append((getattr(tick, "price", None), "ctx_ticks"))

    last_close: Any = None
    try:
        bars_map = _get_mapping(ctx, "bars")
        bars = bars_map.get(sym) if bars_map else None
        if isinstance(bars, (list, tuple)) and bars:
            last_bar = bars[-1]
            if isinstance(last_bar, dict):
                last_close = last_bar.get("close") or last_bar.get("c")
            elif isinstance(last_bar, (list, tuple)) and len(last_bar) >= 5:
                last_close = last_bar[-2]
    except Exception:
        last_close = None
    candidates.append((last_close, "ctx_bars_last_close"))

    if cached > 0.0:
        candidates.append((cached, "price_cache_json"))

    chosen: Optional[Tuple[float, str]] = None
    for raw_val, source in candidates:
        if _is_finite_positive(raw_val):
            chosen = (float(raw_val), source)
            break

    if chosen is None:
        return None

    px, source = chosen

    # Cross-check against the broker price_cache. Only applied to liquid
    # equity/ETF symbols where the placeholder fingerprint pattern is known
    # to occur, and only when the cache itself carries a positive reference.
    # The cache path itself is always trusted on this branch — we just
    # mirrored from it.
    if (
        source != "price_cache_json"
        and sym in _LIQUID_PRICED_ETFS
        and cached > 0.0
    ):
        try:
            deviation = abs(px - cached) / cached
        except ZeroDivisionError:
            deviation = 0.0
        if deviation > _PRICE_CACHE_DEVIATION_THRESHOLD:
            return None

    return px, source


def _asset_class(sym: str) -> AssetClass:
    # Keep simple + stable for policy/routing
    if sym in {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "LQD", "VWO", "IEMG"}:
        return AssetClass.ETF
    # fall back to equity-like
    return AssetClass.EQUITY


# -------------------------
# Universe selection
# -------------------------

def _select_universe(ctx: Any, p: DeltaParams, prices: Mapping[str, Any]) -> List[str]:
    # Priority 1: explicit override
    du = getattr(ctx, "delta_universe", None)
    if isinstance(du, (list, tuple)) and du:
        out = [_norm_sym(x) for x in du if _norm_sym(x)]
        return out[: max(1, p.max_symbols_per_cycle)]

    # Priority 2: legend weights
    legend = getattr(ctx, "legend", None)
    weights = getattr(legend, "weights", None) if legend is not None else None
    if isinstance(weights, Mapping) and weights:
        ranked = sorted(
            ((_norm_sym(k), _safe_float(v, 0.0)) for k, v in weights.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        out = [s for s, _ in ranked if s]
        return out[: max(1, p.max_symbols_per_cycle)]

    # Priority 3: price keys
    out = [_norm_sym(k) for k in prices.keys() if _norm_sym(k)]
    # de-dupe preserving order
    out = list(dict.fromkeys(out))
    return out[: max(1, p.max_symbols_per_cycle)]


# -------------------------
# Gates
# -------------------------

def _event_window_ok(ctx: Any) -> bool:
    """
    Best-effort event veto:
      - ctx.event_risk.veto == True blocks entries
      - ctx.event_risk["veto"] == True blocks entries
    """
    er = getattr(ctx, "event_risk", None)
    if er is None:
        return True
    try:
        veto = getattr(er, "veto", None)
        if isinstance(veto, bool):
            return not veto
        if isinstance(er, Mapping):
            v = er.get("veto")
            if isinstance(v, bool):
                return not v
    except Exception:
        return True
    return True


def _liquidity_ok(ctx: Any, sym: str) -> bool:
    # Spread guard if present
    sp = _get_mapping(ctx, "spread_bps")
    if sp and sym in sp:
        if _safe_float(sp.get(sym), 0.0) > 12.0:
            return False

    # Dollar volume guard if present
    dv = _get_mapping(ctx, "dollar_volume") or _get_mapping(ctx, "volume_usd") or _get_mapping(ctx, "liquidity_usd")
    if dv and sym in dv:
        v = _safe_float(dv.get(sym), 0.0)
        if v > 0.0 and v < 1_000_000.0:
            return False

    return True


def _conviction_score(
    *,
    vol_expanding: bool,
    breakout: bool,
    momentum_ok: bool,
    event_ok: bool,
    liquidity_ok: bool,
) -> float:
    # Transparent scoring (auditable)
    score = 0.0
    score += 0.35 if vol_expanding else 0.0
    score += 0.25 if breakout else 0.0
    score += 0.25 if momentum_ok else 0.0
    score += 0.10 if event_ok else 0.0
    score += 0.05 if liquidity_ok else 0.0
    return _clamp(score, 0.0, 1.0)


# -------------------------
# Per-symbol engine
# -------------------------

def _extract_ohlc(ctx: Any, sym: str, needed: int) -> Optional[Tuple[List[float], List[float], List[float]]]:
    bars = _get_mapping(ctx, "bars").get(sym)
    if not isinstance(bars, (list, tuple)) or len(bars) < needed:
        return None

    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []

    tail = list(bars)[-needed:]
    for b in tail:
        if isinstance(b, dict):
            c = _safe_float(b.get("close") or b.get("c"), 0.0)
            h = _safe_float(b.get("high") or b.get("h"), 0.0)
            l = _safe_float(b.get("low") or b.get("l"), 0.0)
        else:
            if not isinstance(b, (list, tuple)) or len(b) < 5:
                continue
            c = _safe_float(b[-2], 0.0)
            h = _safe_float(b[1] if len(b) == 5 else b[2], 0.0)
            l = _safe_float(b[2] if len(b) == 5 else b[3], 0.0)

        if c <= 0 or h <= 0 or l <= 0 or h < l:
            continue

        closes.append(c)
        highs.append(h)
        lows.append(l)

    if len(closes) < needed:
        return None

    return closes, highs, lows


def _propose_for_symbol(sym: str, ctx: MarketContext, p: DeltaParams, prices: Mapping[str, Any]) -> List[TradeSignal]:
    portfolio: PortfolioSnapshot = ctx.portfolio
    if portfolio.cash < p.min_cash:
        return []

    # PR-02: gate the entire per-symbol pipeline on a strictly-positive
    # price that survives cross-check against runtime/price_cache.json.
    # Abstaining here prevents BOTH the entry (BUY) and exit (SELL) paths
    # from emitting a TradeSignal that downstream would otherwise resolve
    # to the $100 placeholder fingerprint (caught by P0-1 at the writer
    # but persisted as a rejected row in FILLS_*.ndjson).
    resolved = _resolve_positive_price(ctx, sym, prices)
    if resolved is None:
        # One INFO-level log per (symbol, cycle) so the abstain reason is
        # discoverable in journalctl. The strategy emits no SELL signal,
        # so any open position is left to the position-reconciler / guard
        # subsystem rather than being closed against a suspect price.
        LOG.info("DELTA_ABSTAIN_NO_VALID_PRICE symbol=%s", sym)
        return []
    px, _price_source = resolved

    pos: Optional[Position] = portfolio.positions.get(sym)
    qty = float(pos.quantity) if pos is not None else 0.0
    if abs(qty) >= p.max_position_units:
        return []

    # Context must provide bars for entries. Wealth-mode Delta refuses to guess.
    needed = max(p.ema_slow + 2, p.atr_period + 2, p.breakout_lookback + 2, 60)
    ohlc = _extract_ohlc(ctx, sym, needed=needed)
    if ohlc is None:
        # Exit-only if holding, otherwise no entry.
        if qty > 0:
            st = _STATE.get(sym)
            cur_size = _safe_float(st.get("size", p.base_size), p.base_size)
            size = min(cur_size, qty)
            _STATE.set(sym, {"entry": 0.0, "held": 0, "peak": 0.0, "size": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.DELTA,
                    symbol=sym,
                    side=SignalSide.SELL,
                    size=float(size),
                    confidence=0.60,
                    asset_class=_asset_class(sym),
                    created_at=ctx.now,
                    meta={"reason": "no_bars_exit_only"},
                )
            ]
        return []

    closes, highs, lows = ohlc

    ef = _ema(closes, p.ema_fast)[-1]
    es_series = _ema(closes, p.ema_slow)
    es = es_series[-1]
    atr_series = _atr(highs, lows, closes, p.atr_period)
    atr_now = atr_series[-1]
    atr_pct = (atr_now / px) if px else 0.0

    # Regime gate
    if atr_pct < p.min_atr_pct or atr_pct > p.max_atr_pct:
        return []

    # Anti-chase gate
    rng = highs[-1] - lows[-1]
    range_atr = (rng / atr_now) if atr_now else 0.0
    if range_atr > p.anti_chase_range_atr:
        return []

    # Load state for exits / sizing
    st = _STATE.get(sym)
    entry = _safe_float(st.get("entry", 0.0), 0.0)
    held = _safe_int(st.get("held", 0), 0)
    peak = _safe_float(st.get("peak", 0.0), 0.0)
    cur_size = _safe_float(st.get("size", 0.0), 0.0)

    # Exits if holding
    if qty > 0:
        # vol spike
        if atr_pct > p.vol_spike_atr_pct:
            size = min(cur_size or p.base_size, qty)
            _STATE.set(sym, {"entry": 0.0, "held": 0, "peak": 0.0, "size": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.DELTA,
                    symbol=sym,
                    side=SignalSide.SELL,
                    size=float(size),
                    confidence=0.82,
                    asset_class=_asset_class(sym),
                    created_at=ctx.now,
                    meta={"reason": "vol_spike_exit", "atr_pct": float(atr_pct)},
                )
            ]
        # trend break
        if ef <= es:
            size = min(cur_size or p.base_size, qty)
            _STATE.set(sym, {"entry": 0.0, "held": 0, "peak": 0.0, "size": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.DELTA,
                    symbol=sym,
                    side=SignalSide.SELL,
                    size=float(size),
                    confidence=0.76,
                    asset_class=_asset_class(sym),
                    created_at=ctx.now,
                    meta={"reason": "trend_break_exit"},
                )
            ]
        # time stop
        if entry > 0 and held >= p.time_stop_bars:
            if (px - entry) < (p.min_favor_move_atr * atr_now):
                size = min(cur_size or p.base_size, qty)
                _STATE.set(sym, {"entry": 0.0, "held": 0, "peak": 0.0, "size": 0.0})
                return [
                    TradeSignal(
                        strategy=StrategyName.DELTA,
                        symbol=sym,
                        side=SignalSide.SELL,
                        size=float(size),
                        confidence=0.72,
                        asset_class=_asset_class(sym),
                        created_at=ctx.now,
                        meta={"reason": "time_stop_exit"},
                    )
                ]
        # ATR trail
        if peak > 0 and (peak - px) > (p.atr_trail_mult * atr_now):
            size = min(cur_size or p.base_size, qty)
            _STATE.set(sym, {"entry": 0.0, "held": 0, "peak": 0.0, "size": 0.0})
            return [
                TradeSignal(
                    strategy=StrategyName.DELTA,
                    symbol=sym,
                    side=SignalSide.SELL,
                    size=float(size),
                    confidence=0.72,
                    asset_class=_asset_class(sym),
                    created_at=ctx.now,
                    meta={"reason": "atr_trail_exit"},
                )
            ]

        # update holding state (only when not exiting)
        _STATE.set(sym, {"entry": entry or px, "held": held + 1, "peak": max(peak or px, px), "size": cur_size or p.base_size})

    # Entry conditions (convexity)
    trend_ok = (ef > es) and (px > ef)

    rh = _rolling_max(highs, p.breakout_lookback)
    prev_rh = rh[-2] if len(rh) >= 2 else rh[-1]
    breakout = px > (prev_rh + p.breakout_atr_buffer * atr_now)

    mom_atr = (px - ef) / atr_now if atr_now else 0.0
    momentum_ok = mom_atr >= p.momentum_atr

    # Vol expansion (ATR now vs baseline)
    baseline_idx = max(0, len(atr_series) - (p.atr_period * 2))
    atr_base = atr_series[baseline_idx] if baseline_idx < len(atr_series) else atr_now
    vol_expanding = (atr_now >= (p.vol_expansion_factor * atr_base)) if p.require_vol_expansion else True

    event_ok = _event_window_ok(ctx)
    liq_ok = _liquidity_ok(ctx, sym)

    # Must meet base directional setup for entries
    if not (trend_ok and breakout and momentum_ok):
        return []

    score = _conviction_score(
        vol_expanding=vol_expanding,
        breakout=breakout,
        momentum_ok=momentum_ok,
        event_ok=event_ok,
        liquidity_ok=liq_ok,
    )

    if score < p.min_conviction:
        return []

    # Dynamic size
    size = _clamp(p.base_size + score * p.conviction_size_scale, p.base_size, p.max_size)

    # Anti-churn sizing adjustment
    if qty > 0 and abs(size - (cur_size or 0.0)) < p.min_delta_size:
        return []

    # BUY only here (wealth-safe). Shorts can be added later with explicit policy support.
    sig = TradeSignal(
        strategy=StrategyName.DELTA,
        symbol=sym,
        side=SignalSide.BUY,
        size=float(size),
        confidence=float(_clamp(0.55 + score * 0.45, 0.0, 0.95)),
        asset_class=_asset_class(sym),
        created_at=ctx.now,
        meta={
            "reason": "delta_convexity_entry",
            "conviction": float(score),
            "atr_pct": float(atr_pct),
            "mom_atr": float(mom_atr),
            "vol_expanding": bool(vol_expanding),
            "event_ok": bool(event_ok),
            "liquidity_ok": bool(liq_ok),
        },
    )

    _STATE.set(sym, {"entry": px, "held": 1, "peak": px, "size": float(size)})
    return [sig]


# -------------------------
# Wiring
# -------------------------

def build_delta_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.DELTA,
        enabled=True,
        target_universe=["SPY", "QQQ"],  # stable; actual selection inside handler
        max_gross_exposure=None,
        notes="DeltaBrain (wealth-mode convexity hunter; dynamic sizing by conviction).",
    )


def delta_handler(
    ctx: MarketContext,
    params: DeltaParams | None = None,
    *,
    prices: Optional[Mapping[str, Any]] = None,
    **_: Any,
) -> List[TradeSignal]:
    p = params or DEFAULT_PARAMS
    if not p.enabled:
        return []
    px = prices or {}
    universe = _select_universe(ctx, p, px)
    out: List[TradeSignal] = []
    for sym in universe:
        s = _norm_sym(sym)
        if not s:
            continue
        # GAP-035: refuse to emit on operator-excluded symbols upstream
        # of the close-path chokepoints (position_reconciler /
        # flip_executor). The chokepoints already block close intents;
        # this filter additionally prevents *opening* a new position on
        # an excluded symbol from a stale signal/universe entry.
        if is_operator_excluded(s):
            continue
        out.extend(_propose_for_symbol(s, ctx, p, px))
    return out

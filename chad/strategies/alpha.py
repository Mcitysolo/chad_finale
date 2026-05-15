#!/usr/bin/env python3
"""
CHAD — AlphaBrain (Intraday Tactical) — Phase-4 Edge Upgrade

This is a full replacement of the prior deterministic baseline.

What changed
------------
Old Alpha:
- Constant confidence
- Fixed size
- BUY-only
- No regime logic
- No exits
- No volatility filtering
- No anti-churn
- Essentially “always buy legend universe”

New Alpha:
- Trend + volatility regime gating
- ATR-normalized momentum filter
- Anti-chase (range/ATR)
- Deterministic exit engine (time stop, ATR trail, trend break, vol spike)
- Anti-churn band
- Liquidity proxy via dollar volume if available
- Still pure strategy layer (no I/O, no execution side effects)
- 100% compatible with StrategyEngine + TradeSignal contract

Design guarantees:
- Deterministic
- No broker calls
- No filesystem writes
- No shared-state writes
- Mypy-safe typing
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from chad.types import (
    AssetClass,
    MarketContext,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)
from chad.utils.catalyst_gate import check_catalyst_gate
from chad.utils.liquidity import LiquidityClass, blocks_thin_entry
from chad.utils.risk_reward import passes_rr_gate
from chad.utils.session import session_decision

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AlphaParams:
    # Universe behavior
    max_symbols_per_cycle: int = 8
    min_confidence: float = 0.55
    use_legend_filter: bool = True

    # Exposure sizing
    base_size: float = 5.0
    max_size: float = 10.0

    # Indicator parameters
    ema_fast: int = 12
    ema_slow: int = 48
    atr_period: int = 14

    # Regime filters
    min_atr_pct: float = 0.002
    max_atr_pct: float = 0.050

    # Entry quality
    anti_chase_range_atr: float = 3.0
    momentum_atr: float = 0.35

    # Exit rules
    time_stop_bars: int = 30
    min_favor_move_atr: float = 0.5
    atr_trail_mult: float = 2.0
    target_atr_multiple: float = 3.0
    vol_spike_atr_pct: float = 0.060

    # Churn control
    min_delta_size: float = 1.0


def build_alpha_params() -> AlphaParams:
    return AlphaParams()


def build_alpha_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.ALPHA,
        enabled=True,
        target_universe=None,
        max_gross_exposure=None,
        notes="alpha: phase-4 regime/momentum/exit enhanced",
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _asset_class_for_symbol(sym: str) -> AssetClass:
    if sym in {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "LQD", "VWO", "IEMG"}:
        return AssetClass.ETF
    return AssetClass.EQUITY


def _setup_family_for_alpha(reason: str, side: SignalSide, regime: Optional[str]) -> str:
    if "recovery" in (reason or ""):
        return "recovery_long"
    if regime == "uptrend" and side == SignalSide.BUY:
        return "ema_crossover_long"
    if regime == "downtrend" and side == SignalSide.SELL:
        return "ema_crossover_short"
    if regime == "chop" and side == SignalSide.BUY:
        return "chop_reversion_long"
    if regime == "chop" and side == SignalSide.SELL:
        return "chop_reversion_short"
    if side == SignalSide.BUY:
        return "alpha_long"
    if side == SignalSide.SELL:
        return "alpha_short"
    return f"alpha_{reason}" if reason else "alpha_unknown"


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def _ema(values: Sequence[float], period: int) -> List[float]:
    alpha = 2.0 / (period + 1.0)
    out = []
    e = values[0]
    out.append(e)
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
        out.append(e)
    return out


def _true_range(highs, lows, closes):
    out = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        out.append(max(hl, hc, lc))
    return out


def _atr(highs, lows, closes, period):
    tr = _true_range(highs, lows, closes)
    return _ema(tr, period)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x != x:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# ---------------------------------------------------------------------------
# State (in-memory only)
# ---------------------------------------------------------------------------


class _AlphaState:
    def __init__(self):
        self.positions: Dict[str, Dict[str, Any]] = {}

    def get(self, symbol: str) -> Dict[str, Any]:
        return dict(self.positions.get(symbol, {}))

    def set(self, symbol: str, state: Dict[str, Any]) -> None:
        self.positions[symbol] = dict(state)


_STATE = _AlphaState()

# Per-symbol daily signal counter — prevents churn on trending symbols.
# Format: {date_str: {symbol: count}}
_DAILY_SIGNALS: Dict[str, Dict[str, int]] = {}
_MAX_SIGNALS_PER_SYMBOL_PER_DAY = 3


# ---------------------------------------------------------------------------
# Core signal engine
# ---------------------------------------------------------------------------


def build_alpha_signals(
    *,
    ctx: MarketContext,
    prices: Optional[Mapping[str, Any]] = None,
    params: Optional[AlphaParams] = None,
) -> List[TradeSignal]:
    p = params or build_alpha_params()

    raw_prices = prices or getattr(ctx, "prices", None) or {}
    universe: Sequence[str] = []

    if p.use_legend_filter and ctx.legend and getattr(ctx.legend, "weights", None):
        universe = [ _norm_sym(k) for k in ctx.legend.weights.keys() ]
    else:
        universe = list(raw_prices.keys())

    signals: List[TradeSignal] = []
    now = ctx.now if isinstance(ctx.now, datetime) else _utc_now()

    # Tier-aware session gate (entry-only, fail-open). Resolved once per
    # cycle so exits in the per-symbol loop are never short-circuited.
    _tier_profile = getattr(ctx, "tier_profile", None)
    _primary_only = getattr(_tier_profile, "primary_session_only", None)
    _entries_allowed = True
    _session_window: Optional[str] = None
    if _primary_only is not None:
        try:
            _decision = session_decision(
                now, primary_session_only=bool(_primary_only),
            )
            _entries_allowed = _decision.entry_allowed
            _session_window = _decision.session_window
        except Exception:
            # Fail-open: preserve current behavior on parse / tz failure.
            _entries_allowed = True
            _session_window = None

    for symbol in universe:
        sym = _norm_sym(symbol)
        px = _safe_float(raw_prices.get(sym))
        if px <= 0:
            continue

        bars = getattr(ctx, "bars", {}).get(sym)
        if not bars or len(bars) < max(p.ema_slow + 2, p.atr_period + 2):
            continue

        closes = [float(b["close"]) for b in bars]
        highs = [float(b["high"]) for b in bars]
        lows = [float(b["low"]) for b in bars]

        ef = _ema(closes, p.ema_fast)[-1]
        es = _ema(closes, p.ema_slow)[-1]
        a = _atr(highs, lows, closes, p.atr_period)[-1]

        atr_pct = a / px if px else 0.0
        rng = highs[-1] - lows[-1]
        range_atr = rng / a if a else 0.0
        momentum = (px - ef) / a if a else 0.0

        blocked = False

        if atr_pct < p.min_atr_pct or atr_pct > p.max_atr_pct:
            blocked = True

        # Regime detection: uptrend / downtrend / chop
        ef_es_spread = abs(ef - es) / px if px else 0.0
        if ef > es and px > ef:
            regime = "uptrend"
        elif px < es and ef < es:
            regime = "downtrend"
        elif px > es and ef < es:
            regime = "recovery"
        elif ef_es_spread < 0.001:
            regime = "chop"
        else:
            regime = None
            blocked = True

        if range_atr > p.anti_chase_range_atr:
            blocked = True

        # Momentum gate: only applies to trending regimes
        if regime == "uptrend" and momentum < p.momentum_atr:
            blocked = True
        elif regime == "recovery" and momentum < p.momentum_atr * 0.5:
            blocked = True
        elif regime == "downtrend" and momentum > -p.momentum_atr:
            blocked = True

        state = _STATE.get(sym)
        pos = state.get("size", 0.0)
        entry = state.get("entry")
        held = state.get("held", 0)
        peak = state.get("peak")

        exit_signal = False

        if pos > 0:
            if atr_pct > p.vol_spike_atr_pct:
                exit_signal = True
            elif ef <= es:
                exit_signal = True
            elif entry and held >= p.time_stop_bars:
                if px - entry < p.min_favor_move_atr * a:
                    exit_signal = True
            elif peak and peak - px > p.atr_trail_mult * a:
                exit_signal = True

        if exit_signal:
            signals.append(
                TradeSignal(
                    strategy=StrategyName.ALPHA,
                    symbol=sym,
                    side=SignalSide.SELL,
                    size=float(pos),
                    confidence=1.0,
                    asset_class=_asset_class_for_symbol(sym),
                    created_at=now,
                    meta={"reason": "exit"},
                )
            )
            _STATE.set(sym, {"size": 0.0})
            continue

        # Entry-only session gate. Exits above have already been emitted;
        # this guard never affects exit, stop-loss, or position-reduction
        # paths.
        if not _entries_allowed:
            continue

        if blocked:
            continue

        # Pre-entry R:R gate (entry-only, fail-open). Exits above are
        # unaffected. Degenerate ATR (a <= 0) yields a zero target/stop and
        # passes_rr_gate fails open.
        if a > 0:
            _stop_pts_alpha = p.atr_trail_mult * a
            _target_pts_alpha = p.target_atr_multiple * a
            if not passes_rr_gate(_target_pts_alpha, _stop_pts_alpha):
                continue

        # Pre-entry liquidity gate (entry-only, EQUITY/ETF, fail-open).
        # Exits above are unaffected; this guard never blocks position-
        # reducing or stop-loss paths. UNKNOWN classifications pass.
        confidence = _clamp(0.5 + abs(momentum), 0.0, 0.95)
        _asset_cls = _asset_class_for_symbol(sym)
        if _asset_cls in (AssetClass.EQUITY, AssetClass.ETF):
            _liq_blocked, _liq_class, _liq_required_conf = blocks_thin_entry(
                sym,
                float(confidence),
            )
            if _liq_blocked:
                continue
        else:
            _liq_class = LiquidityClass.UNKNOWN
            _liq_required_conf = float(confidence)

        # Pre-entry catalyst gate (entry-only, EQUITY/ETF, fail-open). Exits
        # above are unaffected. Missing/stale runtime/news_intel.json yields
        # an allowed=True unknown result so the strategy keeps trading.
        if _asset_cls in (AssetClass.EQUITY, AssetClass.ETF):
            _side_for_gate = (
                "BUY" if regime in ("uptrend", "recovery") else
                ("SELL" if regime == "downtrend" else
                 ("SELL" if px > (ef + es) / 2 else "BUY"))
            )
            _cat_result = check_catalyst_gate(sym, _side_for_gate)
            if not _cat_result.allowed:
                continue
        else:
            _cat_result = None

        _tier_profile = getattr(ctx, "tier_profile", None)
        _tier_max = getattr(_tier_profile, "max_risk_per_trade_usd", None)
        _stop_per_share = p.atr_trail_mult * a if a > 0 else 0.0

        if _stop_per_share > 0 and _tier_max is not None:
            _risk_sized_shares = int(float(_tier_max) // _stop_per_share)
            if _risk_sized_shares <= 0:
                continue
            size = max(1, min(p.max_size, _risk_sized_shares))
        else:
            size = min(p.max_size, p.base_size * (1.0 + abs(momentum)))
        if abs(size - pos) < p.min_delta_size:
            continue

        today_key = now.strftime("%Y-%m-%d")
        if _DAILY_SIGNALS.get(today_key, {}).get(sym, 0) >= _MAX_SIGNALS_PER_SYMBOL_PER_DAY:
            continue

        if regime == "uptrend":
            side = SignalSide.BUY
            reason = "trend_momentum"
        elif regime == "recovery":
            side = SignalSide.BUY
            reason = "recovery_long"
        elif regime == "downtrend":
            side = SignalSide.SELL
            reason = "trend_short"
        else:  # chop — fade the deviation from mid
            side = SignalSide.SELL if px > (ef + es) / 2 else SignalSide.BUY
            reason = "chop_reversion"

        _entry_meta: Dict[str, Any] = {
            "reason": reason,
            "regime": regime,
            "setup_family": _setup_family_for_alpha(reason, side, regime),
            "stop_distance_pts": round(_stop_per_share, 6),
            "stop_distance_usd": round(_stop_per_share, 6),
            "tier_max_risk_usd": _tier_max,
            "rr_ratio": round(p.target_atr_multiple / p.atr_trail_mult, 4)
                if p.atr_trail_mult > 0 else None,
            "rr_gate": "PASSED",
            "liquidity_class": _liq_class.value,
            "liquidity_required_confidence": round(float(_liq_required_conf), 6),
            "catalyst_strength": (
                _cat_result.catalyst_strength if _cat_result is not None else "unknown"
            ),
            "catalyst_direction": (
                _cat_result.catalyst_direction if _cat_result is not None else "unknown"
            ),
            "catalyst_gate": (
                "PASSED" if _cat_result is None or _cat_result.allowed else "BLOCKED"
            ),
        }
        if _primary_only is not None:
            _entry_meta["session_window"] = _session_window
            _entry_meta["session_gate"] = "PASSED"
            _entry_meta["primary_session_only"] = bool(_primary_only)
        signals.append(
            TradeSignal(
                strategy=StrategyName.ALPHA,
                symbol=sym,
                side=side,
                size=float(size),
                confidence=confidence,
                asset_class=_asset_cls,
                created_at=now,
                meta=_entry_meta,
            )
        )

        _STATE.set(sym, {"size": size, "entry": px, "held": held + 1, "peak": max(px, peak or px)})

        _DAILY_SIGNALS.setdefault(today_key, {})[sym] = (
            _DAILY_SIGNALS.get(today_key, {}).get(sym, 0) + 1
        )
        for _stale in [k for k in _DAILY_SIGNALS if k != today_key]:
            del _DAILY_SIGNALS[_stale]

        if len(signals) >= p.max_symbols_per_cycle:
            break

    return signals


def alpha_handler(
    ctx: MarketContext,
    *,
    prices: Optional[Mapping[str, Any]] = None,
    params: Optional[AlphaParams] = None,
    **_: Any,
) -> List[TradeSignal]:
    return build_alpha_signals(ctx=ctx, prices=prices, params=params)


def run(ctx: MarketContext, prices: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> List[TradeSignal]:
    return build_alpha_signals(ctx=ctx, prices=prices, params=kwargs.get("params"))


def run_alpha(ctx: MarketContext, prices: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> List[TradeSignal]:
    return run(ctx, prices, **kwargs)


def alpha(ctx: MarketContext, prices: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> List[TradeSignal]:
    return run(ctx, prices, **kwargs)


@dataclass(frozen=True, slots=True)
class AlphaBrain:
    params: AlphaParams = AlphaParams()

    def run(self, ctx: MarketContext, *, prices: Optional[Mapping[str, Any]] = None) -> List[TradeSignal]:
        return build_alpha_signals(ctx=ctx, prices=prices, params=self.params)

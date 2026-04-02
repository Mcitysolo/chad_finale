#!/usr/bin/env python3
"""
chad/strategies/omega.py

OmegaBrain — Wealth-Safe Hedge Sleeve (Crash/Volatility Guard)

Omega is NOT a profit engine. Omega is insurance:
- It reduces drawdown during bad regimes so compounding survives.
- It should activate rarely, and deactivate quickly once danger passes.

Design (SSOT-aligned)
---------------------
- Strategy-only: emits TradeSignal intents, never executes.
- Deterministic, no I/O.
- Fail-closed: if key context is missing, Omega stays conservative.
- Bounded: strict hedge caps and cooldowns prevent over-hedging.

Hedge instruments (configurable)
--------------------------------
Default hedge universe focuses on broad-market hedges:
- SH  (1x inverse S&P 500)
- PSQ (1x inverse Nasdaq 100)
- Optional: VIXY / UVXY (vol ETPs) only if your pipeline supports them safely.

Activation signals (best-effort)
--------------------------------
Omega evaluates multiple “danger sensors”:
1) Portfolio drawdown (if ctx.portfolio has equity history or unrealized PnL)
2) Market volatility proxy:
   - SPY/QQQ ATR% spike if ctx.bars available
   - VIX value/spike if ctx.vix or ctx.volatility_index available
3) Regime flag if ctx.macro_state / ctx.regime_state exists

If 2+ sensors agree -> activate hedge.
If danger clears -> unwind hedge.

Compatibility
-------------
Preserves registry expectations:
- OmegaParams
- build_omega_config()
- omega_handler(ctx, params) -> List[TradeSignal]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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

Number = float


# -------------------------
# Params
# -------------------------

@dataclass(frozen=True)
class OmegaParams:
    enabled: bool = True

    # Hedge instruments (broad, simple)
    hedge_spy_symbol: str = "SH"
    hedge_qqq_symbol: str = "PSQ"

    # Max hedge sizing (units), bounded
    max_hedge_units_per_symbol: float = 25.0
    base_hedge_units: float = 5.0

    # Cooldown to prevent churn (minutes)
    cooldown_minutes: int = 60

    # Activation thresholds (wealth-safe)
    # If portfolio drawdown estimate available:
    drawdown_activate: float = 0.06   # -6% activates
    drawdown_deactivate: float = 0.03 # recover to -3% deactivates

    # Volatility proxy (ATR% on SPY/QQQ if bars available)
    atr_pct_activate: float = 0.030   # 3% ATR% signals danger
    atr_pct_deactivate: float = 0.020 # 2% ATR% to relax

    # VIX proxy if available
    vix_activate: float = 25.0
    vix_deactivate: float = 20.0

    # Require N danger sensors to activate hedge
    min_sensors_to_activate: int = 2


DEFAULT_PARAMS = OmegaParams()


# -------------------------
# In-memory state (no disk)
# -------------------------

class _OmegaState:
    def __init__(self) -> None:
        self.last_action_iso: Optional[str] = None
        self.hedge_on: bool = False

    def _parse_iso(self, s: str) -> Optional[datetime]:
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def cooldown_ok(self, now: datetime, cooldown_minutes: int) -> bool:
        if not self.last_action_iso:
            return True
        dt = self._parse_iso(self.last_action_iso)
        if dt is None:
            return True
        age_s = max(0.0, (now - dt).total_seconds())
        return age_s >= float(max(0, cooldown_minutes)) * 60.0

    def mark_action(self, now: datetime) -> None:
        self.last_action_iso = now.astimezone(timezone.utc).isoformat()


_STATE = _OmegaState()


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


def _ema(values: Sequence[float], period: int) -> List[float]:
    if period <= 1:
        return list(values)
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


def _atr_pct_from_bars(ctx: Any, symbol: str, atr_period: int = 14) -> Optional[float]:
    bars = _get_mapping(ctx, "bars").get(symbol)
    if not isinstance(bars, (list, tuple)) or len(bars) < atr_period + 2:
        return None

    closes: List[float] = []
    highs: List[float] = []
    lows: List[float] = []

    tail = list(bars)[-(atr_period + 30):]
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

    if len(closes) < atr_period + 2:
        return None

    a = _atr(highs, lows, closes, atr_period)[-1]
    px = closes[-1]
    if px <= 0:
        return None
    return float(a / px)


def _portfolio_drawdown_estimate(ctx: MarketContext) -> Optional[float]:
    """
    Best-effort drawdown estimate.
    If ctx.portfolio has equity or NAV fields in extra, use them.
    Otherwise return None (fail-closed).
    """
    p: PortfolioSnapshot = ctx.portfolio
    extra = getattr(p, "extra", None)
    if isinstance(extra, dict):
        peak = extra.get("equity_peak")
        cur = extra.get("equity")
        if peak is not None and cur is not None:
            peak_f = _safe_float(peak, 0.0)
            cur_f = _safe_float(cur, 0.0)
            if peak_f > 0.0:
                dd = (cur_f - peak_f) / peak_f
                return float(dd)  # negative during drawdown
    return None


def _vix_value(ctx: Any) -> Optional[float]:
    for k in ("vix", "vol_index", "volatility_index"):
        v = getattr(ctx, k, None)
        if v is None:
            continue
        # allow dict-like
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict) and "value" in v:
            return _safe_float(v.get("value"), None)  # type: ignore[arg-type]
    return None


def _make_signal(now: datetime, symbol: str, side: SignalSide, size: float, reason: str, meta: Dict[str, Any]) -> TradeSignal:
    return TradeSignal(
        strategy=StrategyName.OMEGA,
        symbol=symbol,
        side=side,
        size=float(size),
        confidence=0.85 if side == SignalSide.BUY else 0.75,
        asset_class=AssetClass.ETF,
        created_at=now,
        meta={"reason": reason, **meta},
    )


# -------------------------
# Config factory
# -------------------------

def build_omega_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.OMEGA,
        enabled=True,
        target_universe=["SPY", "QQQ", "SH", "PSQ"],
        max_gross_exposure=None,
        notes="OmegaBrain (wealth-safe crash hedge sleeve).",
    )


# -------------------------
# Handler
# -------------------------

def omega_handler(ctx: MarketContext, params: OmegaParams | None = None, **_: Any) -> List[TradeSignal]:
    p = params or DEFAULT_PARAMS
    if not p.enabled:
        return []

    now = ctx.now if isinstance(ctx.now, datetime) else datetime.now(timezone.utc)

    # Cooldown prevents hedge thrash
    if not _STATE.cooldown_ok(now, p.cooldown_minutes):
        return []

    danger_sensors = 0
    meta: Dict[str, Any] = {}

    # Sensor 1: drawdown (if available)
    dd = _portfolio_drawdown_estimate(ctx)
    if dd is not None:
        meta["drawdown"] = dd
        if dd <= -abs(p.drawdown_activate):
            danger_sensors += 1
    else:
        meta["drawdown"] = None

    # Sensor 2: ATR% spike on SPY/QQQ (if bars available)
    atr_spy = _atr_pct_from_bars(ctx, "SPY")
    atr_qqq = _atr_pct_from_bars(ctx, "QQQ")
    meta["atr_pct_spy"] = atr_spy
    meta["atr_pct_qqq"] = atr_qqq
    if (atr_spy is not None and atr_spy >= p.atr_pct_activate) or (atr_qqq is not None and atr_qqq >= p.atr_pct_activate):
        danger_sensors += 1

    # Sensor 3: VIX (if available)
    vix = _vix_value(ctx)
    meta["vix"] = vix
    if vix is not None and vix >= p.vix_activate:
        danger_sensors += 1

    # Determine desired hedge state
    want_hedge_on = danger_sensors >= max(1, int(p.min_sensors_to_activate))

    # Deactivation logic (if hedge already on)
    if _STATE.hedge_on:
        calm = 0
        if dd is not None and dd > -abs(p.drawdown_deactivate):
            calm += 1
        if (atr_spy is not None and atr_spy <= p.atr_pct_deactivate) and (atr_qqq is not None and atr_qqq <= p.atr_pct_deactivate):
            calm += 1
        if vix is not None and vix <= p.vix_deactivate:
            calm += 1
        if calm >= 2:
            want_hedge_on = False

    # Build signals
    portfolio: PortfolioSnapshot = ctx.portfolio
    positions: Dict[str, Position] = dict(portfolio.positions)

    def pos_qty(sym: str) -> float:
        pos = positions.get(sym)
        return float(pos.quantity) if pos is not None else 0.0

    sigs: List[TradeSignal] = []

    if want_hedge_on and not _STATE.hedge_on:
        # Turn hedge ON: buy inverse ETFs (bounded)
        sh_qty = pos_qty(p.hedge_spy_symbol)
        psq_qty = pos_qty(p.hedge_qqq_symbol)

        buy_sh = max(0.0, min(p.max_hedge_units_per_symbol - sh_qty, p.base_hedge_units))
        buy_psq = max(0.0, min(p.max_hedge_units_per_symbol - psq_qty, p.base_hedge_units))

        if buy_sh > 0:
            sigs.append(_make_signal(now, p.hedge_spy_symbol, SignalSide.BUY, buy_sh, "omega_hedge_on", meta))
        if buy_psq > 0:
            sigs.append(_make_signal(now, p.hedge_qqq_symbol, SignalSide.BUY, buy_psq, "omega_hedge_on", meta))

        if sigs:
            _STATE.hedge_on = True
            _STATE.mark_action(now)

    elif (not want_hedge_on) and _STATE.hedge_on:
        # Turn hedge OFF: sell down current hedge positions (bounded)
        sh_qty = pos_qty(p.hedge_spy_symbol)
        psq_qty = pos_qty(p.hedge_qqq_symbol)

        if sh_qty > 0:
            sigs.append(_make_signal(now, p.hedge_spy_symbol, SignalSide.SELL, min(sh_qty, p.max_hedge_units_per_symbol), "omega_hedge_off", meta))
        if psq_qty > 0:
            sigs.append(_make_signal(now, p.hedge_qqq_symbol, SignalSide.SELL, min(psq_qty, p.max_hedge_units_per_symbol), "omega_hedge_off", meta))

        if sigs:
            _STATE.hedge_on = False
            _STATE.mark_action(now)

    return sigs

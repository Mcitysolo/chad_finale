#!/usr/bin/env python3
"""
chad/strategies/beta_trend.py

BetaTrend — Legend-driven long-term ETF/equity allocator
(WEALTH MODE, low-churn, once-per-day)

Originally shipped as "beta"; renamed to beta_trend when the Beta slot
was reassigned to the institutional-consensus compounder in
chad/strategies/beta.py. Behavior is unchanged from the previous beta.

Throttles:
- Once per UTC day per symbol (hard gate).
- Max signals per UTC day (hard cap).

Guarantees
----------
- Deterministic, no I/O, no disk writes.
- Uses in-memory state only.
- Works with only ctx.legend + ctx.portfolio + ctx.now.
- Still allows conservative add-ons, but also once-per-day gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from chad.types import (
    AssetClass,
    LegendConsensus,
    MarketContext,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)


# -------------------------
# Params
# -------------------------

@dataclass(frozen=True)
class BetaTrendParams:
    # Safety / cash gating
    min_cash: float = 10_000.0

    # Legend selection
    min_weight: float = 0.05
    max_symbols: int = 10

    # WEALTH MODE throttles (critical)
    max_signals_per_day: int = 20         # hard cap: total Beta signals/day
    once_per_day_per_symbol: bool = True  # hard gate

    # Churn controls
    min_hold_days: int = 21
    cooldown_days_after_exit: int = 14

    # Smoothing
    smoothing_alpha: float = 0.35

    # Sizing
    base_size: float = 3.0
    max_size: float = 8.0
    size_scale: float = 10.0


DEFAULT_PARAMS = BetaTrendParams()


# -------------------------
# In-memory state (no disk)
# -------------------------

# -------------------------
# Symbol classification helper
# -------------------------

def _asset_class_for_symbol(sym: str) -> AssetClass:
    if sym in {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "LQD", "VWO", "IEMG"}:
        return AssetClass.ETF
    return AssetClass.EQUITY

class _BetaTrendState:
    def __init__(self) -> None:
        self.ewma: Dict[str, float] = {}
        self.last_entered_iso: Dict[str, str] = {}
        self.last_exited_iso: Dict[str, str] = {}

        # Throttle state (UTC day keys)
        self.last_proposed_day: Dict[str, str] = {}  # sym -> YYYY-MM-DD
        self.day_counts: Dict[str, int] = {}         # YYYY-MM-DD -> count

    def _parse_iso(self, s: str) -> datetime | None:
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    def _day_key(self, now: datetime) -> str:
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        return now.date().isoformat()  # YYYY-MM-DD

    def days_since(self, *, now: datetime, iso: str | None) -> int:
        if not iso:
            return 999999
        dt = self._parse_iso(iso)
        if dt is None:
            return 999999
        delta = now - dt
        return max(0, int(delta.total_seconds() // 86400))

    def mark_enter(self, sym: str, now: datetime) -> None:
        self.last_entered_iso[sym] = now.isoformat()

    def mark_exit(self, sym: str, now: datetime) -> None:
        self.last_exited_iso[sym] = now.isoformat()

    # Throttle
    def can_propose(self, *, sym: str, now: datetime, max_signals_per_day: int, once_per_day_per_symbol: bool) -> bool:
        day = self._day_key(now)

        # daily total cap
        c = int(self.day_counts.get(day, 0))
        if c >= int(max(1, max_signals_per_day)):
            return False

        if once_per_day_per_symbol:
            last_day = self.last_proposed_day.get(sym)
            if last_day == day:
                return False

        return True

    def mark_proposed(self, *, sym: str, now: datetime) -> None:
        day = self._day_key(now)
        self.last_proposed_day[sym] = day
        self.day_counts[day] = int(self.day_counts.get(day, 0)) + 1


_STATE = _BetaTrendState()


# -------------------------
# Helpers
# -------------------------

def _norm_sym(x: Any) -> str:
    return str(x or "").strip().upper()


def _clamp(x: float, lo: float, hi: float) -> float:
    if x != x:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _sorted_legend_weights(legend: LegendConsensus, p: BetaTrendParams) -> List[Tuple[str, float]]:
    items = [(_norm_sym(sym), float(w)) for sym, w in legend.weights.items() if float(w) >= p.min_weight]
    items = [(s, w) for s, w in items if s]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return items[: p.max_symbols]


def _ewma_update(sym: str, weight: float, alpha: float) -> float:
    prev = _STATE.ewma.get(sym)
    if prev is None:
        _STATE.ewma[sym] = weight
        return weight
    new = alpha * weight + (1.0 - alpha) * prev
    _STATE.ewma[sym] = new
    return new


def _size_for_weight(p: BetaTrendParams, w: float) -> float:
    return _clamp(p.base_size + (w * p.size_scale), p.base_size, p.max_size)


# -------------------------
# Config
# -------------------------

def build_beta_trend_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.BETA_TREND,
        enabled=True,
        target_universe=None,
        max_gross_exposure=None,
        notes="BetaTrend (legend-driven long-term allocator; once-per-day throttle).",
    )


# -------------------------
# Handler
# -------------------------

def beta_trend_handler(ctx: MarketContext, params: BetaTrendParams | None = None) -> Sequence[TradeSignal]:
    p = params or DEFAULT_PARAMS

    legend = ctx.legend
    if legend is None:
        return []

    portfolio: PortfolioSnapshot = ctx.portfolio
    if portfolio.cash < p.min_cash:
        return []

    now = ctx.now if isinstance(ctx.now, datetime) else datetime.now(timezone.utc)
    positions: Dict[str, Position] = dict(portfolio.positions)

    ranked = _sorted_legend_weights(legend, p)
    if not ranked:
        return []

    alpha = _clamp(p.smoothing_alpha, 0.01, 0.95)

    scored: List[Tuple[str, float]] = []
    for sym, w in ranked:
        sm = _ewma_update(sym, w, alpha=alpha)
        scored.append((sym, _clamp(sm, 0.0, 1.0)))

    scored.sort(key=lambda kv: kv[1], reverse=True)

    signals: List[TradeSignal] = []

    for sym, w in scored:
        pos = positions.get(sym)
        qty = float(pos.quantity) if pos is not None else 0.0

        # cooldown after exit (only matters if flat)
        if qty <= 0:
            if _STATE.days_since(now=now, iso=_STATE.last_exited_iso.get(sym)) < p.cooldown_days_after_exit:
                continue

        # Throttle gate: once/day per symbol + max/day
        if not _STATE.can_propose(
            sym=sym,
            now=now,
            max_signals_per_day=p.max_signals_per_day,
            once_per_day_per_symbol=p.once_per_day_per_symbol,
        ):
            continue

        # Entry if flat, conservative add-on if holding and hold period satisfied
        size = _size_for_weight(p, w)
        confidence = _clamp(0.50 + (w * 2.0), 0.50, 0.95)

        if qty <= 0:
            signals.append(
                TradeSignal(
                    strategy=StrategyName.BETA_TREND,
                    symbol=sym,
                    side=SignalSide.BUY,
                    size=float(size),
                    confidence=float(confidence),
                    asset_class=_asset_class_for_symbol(sym),
                    created_at=now,
                    meta={
                        "reason": "legend_entry_once_per_day",
                        "legend_weight_raw": float(legend.weights.get(sym, w)),
                        "legend_weight_ewma": float(w),
                        "legend_as_of": legend.as_of.isoformat(),
                    },
                )
            )
            _STATE.mark_enter(sym, now)
            _STATE.mark_proposed(sym=sym, now=now)

        else:
            # Add-ons: only after min_hold_days, and also once-per-day gated
            if _STATE.days_since(now=now, iso=_STATE.last_entered_iso.get(sym)) < p.min_hold_days:
                continue

            signals.append(
                TradeSignal(
                    strategy=StrategyName.BETA_TREND,
                    symbol=sym,
                    side=SignalSide.BUY,
                    size=float(_clamp(size * 0.5, 0.0, p.max_size)),
                    confidence=float(_clamp(confidence * 0.95, 0.50, 0.90)),
                    asset_class=_asset_class_for_symbol(sym),
                    created_at=now,
                    meta={
                        "reason": "legend_topup_once_per_day",
                        "legend_weight_ewma": float(w),
                    },
                )
            )
            _STATE.mark_proposed(sym=sym, now=now)

        if len(signals) >= int(max(1, p.max_signals_per_day)):
            break

    return signals

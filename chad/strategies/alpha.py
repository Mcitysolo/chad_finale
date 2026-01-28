#!/usr/bin/env python3
"""
chad/strategies/alpha.py

AlphaBrain — production-grade, filesystem-SSOT, Phase 9-safe.

This version solves the Phase 9 problem where Alpha never emits entries because
flat-mode anchors collapse to the current price.

Core behaviors
--------------
- Universe: config-driven via chad/utils/universe_provider.py (no hardcoding).
- Anchor:
    * If holding position: anchor = position.avg_price.
    * If flat: anchor = EWMA anchor from runtime/alpha_anchor_book.json.
      If missing, deterministic bootstrap anchor = last_price * (1 + buy_discount_pct).
- BUY entries:
    * Require fresh tick (guard stale snapshot / market closed).
    * Require cash > min_cash and affordability.
- SELL exits:
    * Never blocked by cash (risk-reducing).
- State:
    * Anchor book persisted to runtime with atomic write, includes ts_utc + ttl_seconds.
    * Corrupt/missing file => safe fallback.

Public API
----------
- build_alpha_config()
- alpha_handler(ctx, params=None)

No broker calls, no network I/O. Deterministic and auditable.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "runtime"
ANCHOR_PATH = RUNTIME_DIR / "alpha_anchor_book.json"


# -------------------------
# Time / IO helpers
# -------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_iso_utc(s: Any) -> Optional[datetime]:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


# -------------------------
# Numeric helpers
# -------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    if not math.isfinite(x):
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _norm_symbol(sym: Any) -> str:
    return str(sym).strip().upper()


def _tick_is_fresh(tick: Any, *, now: datetime, max_age_seconds: int) -> bool:
    """
    If tick has a timestamp, enforce freshness.
    If missing timestamp, allow (do not hard-fail).
    """
    ts = getattr(tick, "timestamp", None)
    if not isinstance(ts, datetime):
        return True
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = (now - ts.astimezone(timezone.utc)).total_seconds()
    return age <= float(max_age_seconds)


# -------------------------
# Params
# -------------------------


@dataclass(frozen=True)
class AlphaParams:
    min_cash: float = 5_000.0
    buy_discount_pct: float = 0.0015
    sell_premium_pct: float = 0.005
    base_size: float = 5.0
    min_notional: float = 25.0

    ewma_alpha: float = 0.06
    anchor_ttl_seconds: int = 86400  # 1 day

    tick_max_age_seconds: int = 180  # 3 minutes

    confidence_floor: float = 0.55
    confidence_ceiling: float = 0.85
    confidence_k: float = 12.0

    max_symbols_per_cycle: int = 200
    max_signals_per_cycle: int = 50

    asset_class: AssetClass = AssetClass.ETF

    def validate(self) -> "AlphaParams":
        if not (math.isfinite(self.min_cash) and self.min_cash >= 0.0):
            raise ValueError("min_cash must be finite and >=0")
        for name, pct in (("buy_discount_pct", self.buy_discount_pct), ("sell_premium_pct", self.sell_premium_pct)):
            if not (math.isfinite(pct) and 0.0 < pct < 0.50):
                raise ValueError(f"{name} must be in (0,0.5)")
        if not (math.isfinite(self.base_size) and self.base_size > 0.0):
            raise ValueError("base_size must be >0")
        if not (math.isfinite(self.min_notional) and self.min_notional >= 0.0):
            raise ValueError("min_notional must be >=0")
        if not (math.isfinite(self.ewma_alpha) and 0.0 < self.ewma_alpha <= 1.0):
            raise ValueError("ewma_alpha must be in (0,1]")
        if int(self.anchor_ttl_seconds) <= 0:
            raise ValueError("anchor_ttl_seconds must be positive")
        if int(self.tick_max_age_seconds) <= 0:
            raise ValueError("tick_max_age_seconds must be positive")
        if not (0.0 <= self.confidence_floor <= 1.0):
            raise ValueError("confidence_floor must be in [0,1]")
        if not (0.0 <= self.confidence_ceiling <= 1.0):
            raise ValueError("confidence_ceiling must be in [0,1]")
        if self.confidence_floor > self.confidence_ceiling:
            raise ValueError("confidence_floor must be <= confidence_ceiling")
        if not (math.isfinite(self.confidence_k) and self.confidence_k >= 0.0):
            raise ValueError("confidence_k must be >=0")
        if int(self.max_symbols_per_cycle) <= 0:
            raise ValueError("max_symbols_per_cycle must be positive")
        if int(self.max_signals_per_cycle) <= 0:
            raise ValueError("max_signals_per_cycle must be positive")
        return self


DEFAULT_PARAMS = AlphaParams().validate()


# -------------------------
# Anchor book (SSOT)
# -------------------------


class _AnchorBook:
    def __init__(self, *, alpha: float, ttl_seconds: int, path: Path = ANCHOR_PATH) -> None:
        a = float(alpha)
        if not (0.0 < a <= 1.0) or not math.isfinite(a):
            raise ValueError(f"alpha must be in (0,1], got {alpha!r}")
        self._alpha = a
        self._ttl_seconds = int(ttl_seconds)
        self._path = path
        self._anchors: Dict[str, float] = {}
        self._file_ts: Optional[datetime] = None

    def load(self) -> None:
        self._anchors = {}
        self._file_ts = None
        if not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            self._file_ts = _parse_iso_utc(raw.get("ts_utc"))
            anchors = raw.get("anchors", {})
            if not isinstance(anchors, dict):
                return
            for k, v in anchors.items():
                sym = _norm_symbol(k)
                fv = _safe_float(v, default=float("nan"))
                if sym and math.isfinite(fv) and fv > 0.0:
                    self._anchors[sym] = float(fv)
        except Exception:
            self._anchors = {}
            self._file_ts = None

    def is_stale(self, now: datetime) -> bool:
        if self._file_ts is None:
            return True
        age = (now - self._file_ts).total_seconds()
        return age > float(self._ttl_seconds)

    def get(self, symbol: str) -> Optional[float]:
        sym = _norm_symbol(symbol)
        v = self._anchors.get(sym)
        if v is None or not (math.isfinite(v) and v > 0.0):
            return None
        return float(v)

    def update(self, symbol: str, *, last_price: float) -> None:
        sym = _norm_symbol(symbol)
        px = float(last_price)
        if not sym or not (math.isfinite(px) and px > 0.0):
            return
        prev = self._anchors.get(sym)
        if prev is None or not (math.isfinite(prev) and prev > 0.0):
            self._anchors[sym] = px
        else:
            self._anchors[sym] = (self._alpha * px) + ((1.0 - self._alpha) * prev)

    def flush(self, now: datetime) -> None:
        try:
            payload = {
                "ts_utc": _utc_iso(now),
                "ttl_seconds": int(self._ttl_seconds),
                "ewma_alpha": float(self._alpha),
                "anchors": dict(sorted(self._anchors.items())),
            }
            _atomic_write_json(self._path, payload)
        except Exception:
            pass


def _anchor_price(symbol: str, ctx: MarketContext, book: _AnchorBook, *, last_price: float, params: AlphaParams) -> Optional[float]:
    """
    Anchor selection:
    - If holding: position avg_price
    - Else: EWMA anchor book
    - Else: deterministic bootstrap above price so buy band can trigger
    """
    portfolio: PortfolioSnapshot = ctx.portfolio
    pos: Optional[Position] = portfolio.positions.get(symbol)
    if pos is not None:
        avg = _safe_float(getattr(pos, "avg_price", 0.0), default=0.0)
        if avg > 0.0:
            return float(avg)

    v = book.get(symbol)
    if v is not None:
        return v

    px = float(last_price)
    if not (math.isfinite(px) and px > 0.0):
        return None

    # Bootstrap anchor slightly above price so buy_level ~= price when flat.
    # Deterministic, bounded, avoids anchor==price deadlock.
    return (px / (1.0 - float(params.buy_discount_pct))) * (1.0 + 1e-6)


def _confidence(params: AlphaParams, edge: float, base: float) -> float:
    return _clamp(base + params.confidence_k * max(0.0, edge), params.confidence_floor, params.confidence_ceiling)


# -------------------------
# Public API
# -------------------------


def build_alpha_config() -> StrategyConfig:
    universe = get_trade_universe()
    return StrategyConfig(
        name=StrategyName.ALPHA,
        enabled=True,
        target_universe=list(universe),
        max_gross_exposure=None,
        notes="AlphaBrain (SSOT EWMA anchor; deterministic bootstrap; Phase 9 safe).",
    )


def alpha_handler(ctx: MarketContext, params: AlphaParams | None = None) -> Sequence[TradeSignal]:
    p = (params or DEFAULT_PARAMS).validate()
    now = ctx.now if isinstance(ctx.now, datetime) else _utc_now()

    universe = [s for s in get_trade_universe() if s]
    universe = universe[: int(p.max_symbols_per_cycle)]

    book = _AnchorBook(alpha=p.ewma_alpha, ttl_seconds=int(p.anchor_ttl_seconds))
    book.load()

    signals: List[TradeSignal] = []

    for sym_raw in universe:
        if len(signals) >= int(p.max_signals_per_cycle):
            break

        sym = _norm_symbol(sym_raw)
        if not sym:
            continue

        tick = ctx.ticks.get(sym)
        if tick is None:
            continue

        price = _safe_float(getattr(tick, "price", None), default=0.0)
        if price <= 0.0:
            continue

        portfolio: PortfolioSnapshot = ctx.portfolio
        pos: Optional[Position] = portfolio.positions.get(sym)
        qty = _safe_float(getattr(pos, "quantity", 0.0), default=0.0) if pos is not None else 0.0

        tick_fresh = _tick_is_fresh(tick, now=now, max_age_seconds=int(p.tick_max_age_seconds))

        anchor = _anchor_price(sym, ctx, book, last_price=price, params=p)
        if anchor is None or anchor <= 0.0:
            book.update(sym, last_price=price)
            continue

        buy_level = anchor * (1.0 - p.buy_discount_pct)
        sell_level = anchor * (1.0 + p.sell_premium_pct)

        # BUY entry (flat only): requires fresh tick and cash
        if qty <= 0.0 and price <= buy_level and tick_fresh:
            cash = _safe_float(portfolio.cash, default=0.0)
            if cash > p.min_cash:
                max_affordable = (cash - p.min_cash) / price
                size = min(p.base_size, max_affordable)
                notional = float(size) * float(price)
                if size > 0.0 and math.isfinite(notional) and notional >= p.min_notional:
                    edge = (buy_level - price) / anchor
                    conf = _confidence(p, edge=edge, base=0.65)
                    signals.append(
                        TradeSignal(
                            strategy=StrategyName.ALPHA,
                            symbol=sym,
                            side=SignalSide.BUY,
                            size=float(size),
                            confidence=float(conf),
                            asset_class=p.asset_class,
                            created_at=ctx.now,
                            meta={
                                "reason": "discount_vs_anchor_entry",
                                "anchor": float(anchor),
                                "price": float(price),
                                "buy_level": float(buy_level),
                                "sell_level": float(sell_level),
                                "edge": float(max(0.0, edge)),
                                "tick_fresh": bool(tick_fresh),
                                "anchor_stale": bool(book.is_stale(now)),
                            },
                        )
                    )

        # SELL exit (scale out): never cash-gated
        if qty > 0.0 and price >= sell_level:
            size = min(p.base_size, qty)
            edge = (price - sell_level) / anchor
            conf = _confidence(p, edge=edge, base=0.60)
            signals.append(
                TradeSignal(
                    strategy=StrategyName.ALPHA,
                    symbol=sym,
                    side=SignalSide.SELL,
                    size=float(size),
                    confidence=float(conf),
                    asset_class=p.asset_class,
                    created_at=ctx.now,
                    meta={
                        "reason": "premium_vs_anchor_exit",
                        "anchor": float(anchor),
                        "price": float(price),
                        "buy_level": float(buy_level),
                        "sell_level": float(sell_level),
                        "edge": float(max(0.0, edge)),
                        "tick_fresh": bool(tick_fresh),
                        "anchor_stale": bool(book.is_stale(now)),
                    },
                )
            )

        # Update EWMA
        book.update(sym, last_price=price)

    # Persist anchors (best-effort, never raises)
    book.flush(now)
    return signals


#!/usr/bin/env python3
"""
CHAD — IBKR Paper TradeResult Logger (Production)

Purpose
-------
Record CHAD strategy-originated IBKR PAPER execution events into the same
tamper-evident TradeResult NDJSON ledger used by SCR.

Key upgrade (this patch)
------------------------
This logger now supports logging a *real fill price* and *filled quantity*
when the caller has them (e.g., Paper Shadow Runner after a fill). This allows
TradeResult entries to carry real pricing instead of 0.0.

Notes / Guarantees
------------------
- This module does NOT place orders.
- It logs already-known execution metadata for traceability + attribution.
- PnL can remain 0.0 at entry time (realized PnL typically happens on close).
  Downstream enrichment / close-event logic can update PnL later.
- Strategy must NOT be "manual" (manual trades are excluded from SCR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chad.analytics.trade_result_logger import TradeResult, log_trade_result


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_side(side: str) -> str:
    s = (side or "").strip().upper()
    if s in {"BUY", "B"}:
        return "BUY"
    if s in {"SELL", "S"}:
        return "SELL"
    return s or "UNKNOWN"


def _finite_pos(x: Optional[float]) -> Optional[float]:
    """Return x if it is finite and > 0, else None."""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    if v <= 0.0:
        return None
    return v


@dataclass(frozen=True)
class IBKRPaperOrderEvent:
    """
    Normalized order execution event for logging.

    Required
    --------
    - strategy: CHAD strategy name (e.g., "beta")
    - symbol: trading symbol (e.g., "AAPL")
    - side: BUY/SELL
    - quantity: intended quantity (absolute)

    Optional (recommended when known)
    --------------------------------
    - filled_quantity: actual filled quantity (if known)
    - fill_price: average/actual fill price (if known)

    Notes
    -----
    - notional_estimate is used as a fallback when fill info isn't available.
    - order_id / perm_id help correlate with other IBKR/PnL watchers.
    """

    strategy: str
    symbol: str
    side: str
    quantity: float
    notional_estimate: float

    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    order_type: str = "MKT"

    # Optional execution identifiers
    order_id: Optional[int] = None
    perm_id: Optional[int] = None

    # Optional fill data (if caller has it)
    filled_quantity: Optional[float] = None
    fill_price: Optional[float] = None

    # Raw payload for auditability
    raw_intent: Optional[Dict[str, Any]] = None


def log_ibkr_paper_order_event(
    event: IBKRPaperOrderEvent,
    *,
    account_id: Optional[str] = None,
) -> str:
    """
    Write a CHAD TradeResult record for an IBKR paper execution event.

    Returns
    -------
    str
        log_path returned by log_trade_result()

    Validation rules
    ----------------
    - strategy required and must not be 'manual'
    - quantity must be > 0 (uses filled_quantity if provided and valid)
    - notional must be > 0 (prefers computed from fill if available)
    """
    now = _utc_now_iso()

    strategy = (event.strategy or "").strip()
    if not strategy:
        raise ValueError("strategy is required")
    if strategy.lower() == "manual":
        raise ValueError("refusing to log strategy='manual' as effective trade")

    symbol = (event.symbol or "").strip().upper() or "UNKNOWN"
    side = _norm_side(event.side)

    qty = _finite_pos(event.filled_quantity) or _finite_pos(event.quantity)
    if qty is None:
        raise ValueError(f"quantity must be > 0 (got quantity={event.quantity!r}, filled_quantity={event.filled_quantity!r})")

    fp = _finite_pos(event.fill_price)
    notional_from_fill: Optional[float] = None
    if fp is not None:
        notional_from_fill = float(qty) * float(fp)

    notional = _finite_pos(notional_from_fill) or _finite_pos(event.notional_estimate)
    if notional is None:
        raise ValueError(
            "notional must be > 0 (got notional_estimate={!r}, fill_price={!r}, filled_qty={!r})".format(
                event.notional_estimate, event.fill_price, qty
            )
        )

    tags = [
        "ibkr_paper",
        strategy.lower(),
        side.lower(),
        (event.order_type or "MKT").lower(),
    ]
    if fp is not None:
        tags.append("filled")
    else:
        tags.append("fill_unknown")

    extra: Dict[str, Any] = {
        "source": "paper_shadow_runner",
        "sec_type": event.sec_type,
        "exchange": event.exchange,
        "currency": event.currency,
        "order_type": event.order_type,
        "notional_estimate": float(event.notional_estimate),
        "notional_used": float(notional),
        "filled_quantity_used": float(qty),
        "fill_price_used": float(fp) if fp is not None else 0.0,
    }
    if event.order_id is not None:
        extra["order_id"] = int(event.order_id)
    if event.perm_id is not None:
        extra["perm_id"] = int(event.perm_id)
    if event.raw_intent is not None:
        extra["raw_intent"] = dict(event.raw_intent)

    tr = TradeResult(
        strategy=strategy.lower(),
        symbol=symbol,
        side=side,
        quantity=float(qty),
        fill_price=float(fp) if fp is not None else 0.0,
        notional=float(notional),
        pnl=0.0,  # realized PnL typically arrives on close/enrichment
        entry_time_utc=now,
        exit_time_utc=now,
        is_live=False,
        broker="ibkr",
        account_id=account_id,
        regime=None,
        tags=tags,
        extra=extra,
    )

    path = log_trade_result(tr)
    return str(path)

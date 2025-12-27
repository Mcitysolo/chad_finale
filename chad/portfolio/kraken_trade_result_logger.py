#!/usr/bin/env python3
"""
CHAD — Kraken TradeResult Logger (Production)

Goal
----
Persist Kraken execution events into CHAD's tamper-evident TradeResult NDJSON ledger.

Why
---
Today, Kraken orders execute successfully but are not recorded in CHAD's trade ledger.
SCR / stats can only reason about what is in the ledger, so Kraken activity must be logged.

Design
------
- Writes a CHAD TradeResult record using the existing ledger logger:
    chad.analytics.trade_result_logger.log_trade_result
- Uses minimal required fields so it's safe and deterministic.
- PnL is set to 0.0 for now (PnL tracking requires fills + position snapshots;
  that's a Phase 8+ enhancement). This still counts as an "effective trade"
  because it is non-manual and trusted.

Safety
------
- Does not place any orders.
- Does not read secrets.
- Pure logging: takes already-known execution metadata (txid, pair, side, volume, etc).

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from chad.analytics.trade_result_logger import TradeResult, log_trade_result


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class KrakenTradeEvent:
    """
    Normalized Kraken execution event for CHAD logging.

    Fields:
      strategy: CHAD strategy name (e.g. "alpha_crypto")
      pair: Kraken pair code (e.g. "XXBTZCAD")
      side: "buy" or "sell"
      ordertype: "market" or "limit"
      volume: base asset size
      notional_estimate: approximate fiat notional (CAD in your setup)
      txid: Kraken transaction id
      raw: raw Kraken response dict (stored in extra)
    """

    strategy: str
    pair: str
    side: str
    ordertype: str
    volume: float
    notional_estimate: float
    txid: str
    raw: Dict[str, Any]


def log_kraken_trade_event(
    event: KrakenTradeEvent,
    *,
    account_id: Optional[str] = None,
) -> str:
    """
    Convert a KrakenTradeEvent into a CHAD TradeResult record and append to ledger.

    Returns:
      log_path (string path) returned by log_trade_result
    """
    now = _utc_now_iso()

    # Derive a display symbol. For Kraken pairs like XXBTZCAD -> XXBT.
    # If format unexpected, keep full pair.
    sym = event.pair
    if sym.startswith("X") and "Z" in sym:
        # Common Kraken format: XXBTZCAD (base=XXBT)
        sym = sym.split("Z", 1)[0]

    tags = [
        "kraken_live",
        event.strategy,
        event.side.lower(),
        event.ordertype.lower(),
    ]

    extra: Dict[str, Any] = {
        "source": "kraken_executor",
        "txid": event.txid,
        "pair": event.pair,
        "ordertype": event.ordertype,
        "raw": event.raw,
    }

    tr = TradeResult(
        strategy=event.strategy,
        symbol=sym,
        side=event.side.upper(),
        quantity=float(event.volume),
        fill_price=0.0,  # unknown at this layer; fill details come later
        notional=float(event.notional_estimate),
        pnl=0.0,  # unknown for now; treated as trusted but neutral
        entry_time_utc=now,
        exit_time_utc=now,
        is_live=True,
        broker="kraken",
        account_id=account_id,
        regime=None,
        tags=tags,
        extra=extra,
    )

    path = log_trade_result(tr)
    return str(path)

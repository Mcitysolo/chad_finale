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
from chad.core.regime_tag import resolve_regime_label
from datetime import datetime, timezone


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


# CHAD canonical reverse map for Kraken altnames. Used so paper evidence
# records carry the same canonical symbol form as the rest of the lane
# (SOLUSD -> SOL-USD, XBTUSD -> BTC-USD, etc.) instead of the broker
# altname. Live records keep their existing derivation for backward
# compatibility.
_KRAKEN_ALTNAME_TO_CANONICAL: Dict[str, str] = {
    "XBTUSD": "BTC-USD",
    "XXBTZUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "XETHZUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "XBTCAD": "BTC-CAD",
    "XXBTZCAD": "BTC-CAD",
    "ETHCAD": "ETH-CAD",
    "XETHZCAD": "ETH-CAD",
}


def _canonical_symbol_for_pair(pair: str) -> str:
    p = (pair or "").strip().upper()
    if p in _KRAKEN_ALTNAME_TO_CANONICAL:
        return _KRAKEN_ALTNAME_TO_CANONICAL[p]
    if p.startswith("X") and "Z" in p:
        return p.split("Z", 1)[0]
    return p


def log_kraken_trade_event(
    event: KrakenTradeEvent,
    *,
    account_id: Optional[str] = None,
    paper: bool = False,
    fill_price: float = 0.0,
    expected_price: float = 0.0,
) -> str:
    """
    Convert a KrakenTradeEvent into a CHAD TradeResult record and append to ledger.

    paper=False (default) preserves the live wiring: broker="kraken",
    is_live=True, and the legacy altname-stripped display symbol.

    paper=True writes a paper-mode TradeResult with broker="kraken_paper",
    is_live=False, the canonical symbol form (SOLUSD -> SOL-USD), and
    tags identifying the validate-only origin. PnL stays 0.0 / untrusted —
    we never invent realized PnL on a validate-only response.

    Returns:
      log_path (string path) returned by log_trade_result
    """
    now = _utc_now_iso()

    if paper:
        sym = _canonical_symbol_for_pair(event.pair)
        broker = "kraken_paper"
        is_live = False
        tags = [
            "kraken_paper",
            "pnl_untrusted",
            "validate_only",
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
            "pnl_untrusted": True,
            "pnl_untrusted_reason": "kraken_paper_validate_only_no_realized_fill",
            "validate_only": True,
            "expected_price": float(expected_price or 0.0),
        }
    else:
        # Derive a display symbol. For Kraken pairs like XXBTZCAD -> XXBT.
        # If format unexpected, keep full pair.
        sym = event.pair
        if sym.startswith("X") and "Z" in sym:
            sym = sym.split("Z", 1)[0]

        broker = "kraken"
        is_live = True
        tags = [
            "kraken_live",
            "pnl_untrusted",
            event.strategy,
            event.side.lower(),
            event.ordertype.lower(),
        ]
        extra = {
            "source": "kraken_executor",
            "txid": event.txid,
            "pair": event.pair,
            "ordertype": event.ordertype,
            "raw": event.raw,
            "pnl_untrusted": True,
            "pnl_untrusted_reason": "kraken_fill_price_and_realized_pnl_not_available_yet",
        }

    tr = TradeResult(
        strategy=event.strategy,
        symbol=sym,
        side=event.side.upper(),
        quantity=float(event.volume),
        fill_price=float(fill_price or 0.0),
        notional=float(event.notional_estimate),
        pnl=0.0,
        entry_time_utc=now,
        exit_time_utc=now,
        is_live=is_live,
        broker=broker,
        account_id=account_id,
        regime=resolve_regime_label(now_utc=datetime.now(timezone.utc)),
        tags=tags,
        extra=extra,
    )

    path = log_trade_result(tr)
    return str(path)

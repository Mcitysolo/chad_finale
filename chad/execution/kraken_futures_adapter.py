"""
CHAD — Kraken Futures Execution Adapter Scaffold (Phase C Item 1B).

Scaffold only. Not authorized for live trading until futures credentials,
endpoint validation, and execution routing approval exist.

Purpose
-------
Translate CHAD-style trade intents into Kraken Futures order requests via
``KrakenFuturesClient``. This module is intentionally NOT wired into:

    - chad.core.live_loop
    - chad.execution.execution_pipeline
    - any strategy code

and it does not auto-register with any router. Default behavior is
``dry_run=True`` with no network activity. Live use requires explicit
operator opt-in, credentials, an authenticated smoke test, and execution
pipeline integration review (see docs/PHASE_C_C1B_KRAKEN_FUTURES_ADAPTER_SCAFFOLD.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from chad.exchanges.kraken_futures_client import (
    KrakenFuturesClient,
    KrakenFuturesOrderRequest,
    KrakenFuturesOrderResult,
)

_CANONICAL_TO_KRAKEN: Dict[str, str] = {
    "BTC-USD": "PF_XBTUSD",
    "XBT-USD": "PF_XBTUSD",
    "ETH-USD": "PF_ETHUSD",
    "SOL-USD": "PF_SOLUSD",
}


@dataclass(frozen=True)
class KrakenFuturesIntent:
    """CHAD-style trade intent for a Kraken Futures perpetual."""

    symbol: str
    side: str
    quantity: float
    order_type: str = "mkt"
    limit_price: Optional[float] = None
    reduce_only: bool = False
    strategy: str = "unknown"
    reason: str = ""


class KrakenFuturesAdapter:
    """Scaffold adapter from CHAD intent -> Kraken Futures order request.

    Scaffold only. Not authorized for live trading until futures credentials,
    endpoint validation, and execution routing approval exist.
    """

    def __init__(
        self,
        client: Optional[KrakenFuturesClient] = None,
        dry_run: bool = True,
    ) -> None:
        self._dry_run = bool(dry_run)
        if client is None:
            client = KrakenFuturesClient(credentials=None, dry_run=self._dry_run)
        self._client = client

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise ValueError(f"kraken_futures_symbol_invalid: {symbol!r}")
        s = symbol.strip()
        if not s:
            raise ValueError("kraken_futures_symbol_empty")
        if s.startswith("PF_"):
            return s
        mapped = _CANONICAL_TO_KRAKEN.get(s.upper())
        if mapped is None:
            raise ValueError(f"kraken_futures_symbol_unmapped: {symbol!r}")
        return mapped

    def build_order_request(self, intent: KrakenFuturesIntent) -> KrakenFuturesOrderRequest:
        kraken_symbol = self.normalize_symbol(intent.symbol)
        return KrakenFuturesOrderRequest(
            symbol=kraken_symbol,
            side=(intent.side or "").lower(),
            order_type=(intent.order_type or "mkt").lower(),
            size=float(intent.quantity),
            limit_price=(float(intent.limit_price) if intent.limit_price is not None else None),
            reduce_only=bool(intent.reduce_only),
            client_order_id=None,
        )

    def submit_intent(self, intent: KrakenFuturesIntent) -> KrakenFuturesOrderResult:
        try:
            request = self.build_order_request(intent)
        except ValueError as exc:
            return KrakenFuturesOrderResult(
                ok=False,
                dry_run=self._dry_run,
                status="intent_invalid",
                request={
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "quantity": intent.quantity,
                    "order_type": intent.order_type,
                    "strategy": intent.strategy,
                },
                response={},
                error=str(exc),
            )
        return self._client.submit_order(request)


__all__ = [
    "KrakenFuturesIntent",
    "KrakenFuturesAdapter",
]

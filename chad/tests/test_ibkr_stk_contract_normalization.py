#!/usr/bin/env python3
"""
chad/tests/test_ibkr_stk_contract_normalization.py

IBKR Error 321 fix: every STK contract sent to IBKR must carry an explicit
exchange and currency. Some intent producers (notably reconciler close
intents) build minimal objects that omit these fields. Without defaults the
resulting Stock(symbol='BAC') gets cancelled by IBKR with:
    Error 321 ... cause - Missing order exchange.

Covers:
  1. NormalizedIntent without exchange/currency builds Stock("BAC", "SMART", "USD").
  2. LLY behaves the same way.
  3. Explicit exchange/currency on the intent are preserved verbatim.
  4. Explicit exchange/currency in intent.meta are preserved verbatim.
  5. _intent_from_trade_intent fills SMART/USD on STK objects that omit them.
  6. Futures missing contract_month still raises ContractResolutionError
     (preserves the P0-1 hot-path safety contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from chad.execution.ibkr_adapter import (
    ContractResolutionError,
    IbkrAdapter,
    IbkrConfig,
    NormalizedIntent,
    _ContractResolver,
)


def _make_stk_intent(
    symbol: str = "BAC",
    exchange: str = "",
    currency: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy="alpha_intraday",
        symbol=symbol,
        sec_type="STK",
        exchange=exchange,
        currency=currency,
        side="BUY",
        order_type="MKT",
        quantity=10.0,
        notional_estimate=0.0,
        asset_class="equity",
        source_strategies=("alpha_intraday",),
        created_at=datetime.now(timezone.utc),
        meta=meta or {},
    )


def _make_resolver() -> _ContractResolver:
    return _ContractResolver(IbkrConfig(dry_run=True), lambda: datetime.now(timezone.utc))


def test_resolve_stock_defaults_to_smart_usd_for_bac() -> None:
    intent = _make_stk_intent(symbol="BAC")
    resolved = _make_resolver()._resolve_stock(intent)
    contract = resolved.contract
    assert contract.symbol == "BAC"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"
    assert resolved.summary["exchange"] == "SMART"
    assert resolved.summary["currency"] == "USD"


def test_resolve_stock_defaults_to_smart_usd_for_lly() -> None:
    intent = _make_stk_intent(symbol="LLY")
    resolved = _make_resolver()._resolve_stock(intent)
    assert resolved.contract.symbol == "LLY"
    assert resolved.contract.exchange == "SMART"
    assert resolved.contract.currency == "USD"


def test_resolve_stock_preserves_explicit_exchange_and_currency() -> None:
    intent = _make_stk_intent(symbol="BAC", exchange="NYSE", currency="USD")
    resolved = _make_resolver()._resolve_stock(intent)
    assert resolved.contract.exchange == "NYSE"
    assert resolved.contract.currency == "USD"
    assert resolved.summary["exchange"] == "NYSE"


def test_resolve_stock_meta_overrides_apply_when_intent_fields_blank() -> None:
    intent = _make_stk_intent(
        symbol="ASML",
        exchange="",
        currency="",
        meta={"exchange": "AEB", "currency": "EUR"},
    )
    resolved = _make_resolver()._resolve_stock(intent)
    assert resolved.contract.exchange == "AEB"
    assert resolved.contract.currency == "EUR"


# ---------------------------------------------------------------------------
# StrategyTradeIntent normalization (mirrors reconciler-style minimal intents)
# ---------------------------------------------------------------------------


@dataclass
class _MinimalReconcilerIntent:
    """Mirrors chad.core.position_reconciler._ReconcilerIntent shape."""
    symbol: str
    side: str
    quantity: float
    sec_type: str = "STK"
    strategy: str = "reconciler"
    order_type: str = "MKT"
    confidence: float = 1.0
    # Note: NO exchange / currency / meta — exactly like the reconciler.


def test_intent_from_trade_intent_fills_stk_defaults_when_missing() -> None:
    raw = _MinimalReconcilerIntent(symbol="BAC", side="BUY", quantity=10.0)
    adapter = IbkrAdapter(IbkrConfig(dry_run=True))
    normalized = adapter._intent_from_trade_intent(raw)
    assert normalized.sec_type == "STK"
    assert normalized.exchange == "SMART"
    assert normalized.currency == "USD"


def test_intent_from_trade_intent_preserves_explicit_stk_routing() -> None:
    @dataclass
    class _Intent:
        symbol: str = "BAC"
        side: str = "BUY"
        quantity: float = 10.0
        sec_type: str = "STK"
        exchange: str = "NYSE"
        currency: str = "USD"
        strategy: str = "alpha_intraday"
        order_type: str = "MKT"
        notional_estimate: float = 0.0
        limit_price: Optional[float] = None
        meta: Dict[str, Any] = field(default_factory=dict)

    adapter = IbkrAdapter(IbkrConfig(dry_run=True))
    normalized = adapter._intent_from_trade_intent(_Intent())
    assert normalized.exchange == "NYSE"
    assert normalized.currency == "USD"


# ---------------------------------------------------------------------------
# Futures safety contract — must still raise on missing contract_month
# ---------------------------------------------------------------------------


class _FakeIB:
    def isConnected(self) -> bool:
        return True


def test_futures_missing_contract_month_still_raises() -> None:
    intent = NormalizedIntent(
        strategy="alpha_futures",
        symbol="MES",
        sec_type="FUT",
        exchange="CME",
        currency="USD",
        side="BUY",
        order_type="LMT",
        quantity=1.0,
        notional_estimate=29000.0,
        asset_class="futures",
        source_strategies=("alpha_futures",),
        created_at=datetime.now(timezone.utc),
        limit_price=5800.0,
        meta={},
    )
    resolver = _make_resolver()
    with pytest.raises(ContractResolutionError):
        resolver._resolve_future(_FakeIB(), intent)

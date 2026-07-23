"""GAP-037: reconciler-emitted close intents must build proper futures contracts.

Pins the 2026-05-19 incident in which `_close_intent_to_ibkr` hardcoded
`sec_type="STK"` for every reconciler close intent. Closing MES/MNQ shorts
in the paper account caused IBKR to receive
`Stock(symbol='MES', exchange='SMART', currency='USD')` and reply with
Error 200 "No security definition has been found for the request".

The patch (chad/core/position_reconciler.py::_close_intent_to_ibkr) now
detects futures symbols via `resolve_asset_class` and stamps:
  - sec_type = "FUT"
  - asset_class = AssetClass.FUTURES
  - meta.contract_month from `chad.market_data.futures_contract_resolver`
  - meta.contract_month_source
  - meta.contract_month_resolved_at_utc

For equities/ETFs the legacy STK shape is preserved so existing close
behavior on AAPL/MSFT/GLD/etc. does not regress.
"""
from __future__ import annotations

import pytest

from chad.core.position_reconciler import _close_intent_to_ibkr
from chad.types import AssetClass


def _close(symbol: str, side: str = "BUY", qty: float = 1.0) -> dict:
    return {
        "symbol": symbol,
        "close_side": side,
        "open_side": "SELL" if side == "BUY" else "BUY",
        "quantity": qty,
        "reason": "reconciler_flip_test",
        "position_key": f"broker_sync|{symbol}",
        "strategy": "reconciler",
    }


# ---------------------------------------------------------------------------
# Futures branch — the GAP-037 fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    ["MES", "MNQ", "MCL", "MGC", "M6E", "M2K", "MYM", "ZN", "ZB"],
)
def test_reconciler_close_intent_futures_carries_fut_sectype(symbol):
    intent = _close_intent_to_ibkr(_close(symbol, side="BUY", qty=2.0))
    assert intent.sec_type == "FUT", (
        f"GAP-037 regression: reconciler close intent for {symbol} "
        f"emitted sec_type={intent.sec_type!r}; expected 'FUT'."
    )
    assert intent.asset_class == AssetClass.FUTURES


@pytest.mark.parametrize(
    "symbol,expected_month_prefix",
    [
        ("MES", "2026"),
        ("MNQ", "2026"),
        ("MCL", "2026"),
        ("MGC", "2026"),
        ("M6E", "2026"),
    ],
)
def test_reconciler_close_intent_futures_carries_contract_month(
    symbol, expected_month_prefix
):
    intent = _close_intent_to_ibkr(_close(symbol))
    month = intent.meta.get("contract_month")
    assert month, (
        f"GAP-037 regression: reconciler close intent for {symbol} "
        f"lacks meta.contract_month — adapter._resolve_future will reject."
    )
    assert month.startswith(expected_month_prefix), month
    # Source + resolved_at_utc are required so audit trails can attribute
    # the choice of expiry back to the resolver and the cycle clock.
    assert intent.meta.get("contract_month_source") == (
        "chad.market_data.futures_contract_resolver"
    )
    assert intent.meta.get("contract_month_resolved_at_utc")


def test_reconciler_close_intent_futures_quantity_preserved():
    intent = _close_intent_to_ibkr(_close("MES", side="BUY", qty=41.0))
    assert intent.quantity == pytest.approx(41.0)
    assert intent.side == "BUY"


def test_reconciler_close_intent_futures_meta_is_isolated_per_call():
    """Mutating one returned intent's meta must not bleed into the next."""
    a = _close_intent_to_ibkr(_close("MES"))
    a.meta["contract_month"] = "MUTATED"
    b = _close_intent_to_ibkr(_close("MES"))
    assert b.meta["contract_month"] != "MUTATED"


# ---------------------------------------------------------------------------
# Equity / ETF branches — no regression on the legacy STK path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "NVDA", "GOOGL"])
def test_reconciler_close_intent_equity_remains_stk(symbol):
    intent = _close_intent_to_ibkr(_close(symbol, side="SELL", qty=5.0))
    assert intent.sec_type == "STK"
    assert intent.asset_class == AssetClass.EQUITY
    # GAP-037 contract: no futures contract keys on the STK branch. (W4B-2
    # added close-provenance stamps to ALL branches, so meta is no longer
    # empty — the pin is the absence of contract keys, not emptiness.)
    assert "contract_month" not in intent.meta, (
        "equity close intents must not carry futures contract_month meta"
    )
    assert intent.meta["action"] == "CLOSE"            # W4B-2 stamp present


@pytest.mark.parametrize("symbol", ["GLD", "TLT", "IEMG", "VWO"])
def test_reconciler_close_intent_etf_remains_stk(symbol):
    intent = _close_intent_to_ibkr(_close(symbol, side="BUY", qty=10.0))
    assert intent.sec_type == "STK"
    assert intent.asset_class == AssetClass.ETF
    assert "contract_month" not in intent.meta         # W4B-2: stamps only
    assert intent.meta["action"] == "CLOSE"


# ---------------------------------------------------------------------------
# Defensive: empty / unknown symbol
# ---------------------------------------------------------------------------


def test_reconciler_close_intent_empty_symbol_falls_back_to_stk():
    intent = _close_intent_to_ibkr(_close(""))
    assert intent.sec_type == "STK"
    assert intent.asset_class == AssetClass.EQUITY
    assert "contract_month" not in intent.meta         # W4B-2: stamps only


def test_reconciler_close_intent_unknown_symbol_falls_back_to_stk():
    intent = _close_intent_to_ibkr(_close("ZZTOP"))
    assert intent.sec_type == "STK"
    assert intent.asset_class == AssetClass.EQUITY
    assert "contract_month" not in intent.meta         # W4B-2: stamps only


# ---------------------------------------------------------------------------
# Interface contract — fields the IBKR adapter relies on
# ---------------------------------------------------------------------------


def test_reconciler_close_intent_exposes_adapter_required_fields():
    """`_intent_from_trade_intent` reads: symbol, sec_type, side, order_type,
    quantity, exchange, currency, strategy, limit_price (optional), meta,
    notional_estimate. Pin that the reconciler intent exposes the required
    surface so adapter changes break here loudly rather than at runtime."""
    intent = _close_intent_to_ibkr(_close("MES"))
    for attr in ("symbol", "side", "quantity", "sec_type", "asset_class",
                 "strategy", "order_type", "meta", "exchange", "currency"):
        assert hasattr(intent, attr), f"missing required field: {attr}"

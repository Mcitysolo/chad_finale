"""W4B-2 (J16 rider / Lane-A D8 absorption): close-provenance stamps.

``_close_intent_to_ibkr`` previously dropped the close dict's
``action/reason/position_key/open_side/close_side`` before submission — an
overlay/reconciler close was indistinguishable from a strategy SELL at the
adapter boundary (PLAN_W4A P9 / PLAN_W4B P-rider). Pins:

  - the wrapped intent's meta carries the stamps;
  - the futures branch keeps its contract_month keys alongside the stamps;
  - the stamps are visible at a fake adapter through apply_close_intents;
  - the operator-exclusion backstop is NOT bypassed by stamped closes;
  - the adapter idempotency fingerprint is unchanged by the stamps (meta is
    excluded from the stable payload).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, List

import chad.core.position_reconciler as pr

_LOG = logging.getLogger("test_w4b_close_stamp")


def _close(symbol="UNH", **over):
    base = {
        "symbol": symbol,
        "action": "CLOSE",
        "open_side": "BUY",
        "close_side": "SELL",
        "quantity": 5.0,
        "reason": "exit_overlay_atr_trailing_stop",
        "position_key": f"gamma|{symbol}",
        "strategy": "gamma",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# stamp content
# --------------------------------------------------------------------------- #

def test_equity_close_intent_carries_stamps():
    intent = pr._close_intent_to_ibkr(_close())
    assert intent.symbol == "UNH"
    assert intent.side == "SELL"
    assert intent.sec_type == "STK"
    m = intent.meta
    assert m["action"] == "CLOSE"
    assert m["close_origin"] == "apply_close_intents"
    assert m["reason"] == "exit_overlay_atr_trailing_stop"
    assert m["position_key"] == "gamma|UNH"
    assert m["open_side"] == "BUY"
    assert m["close_side"] == "SELL"


def test_futures_close_keeps_contract_month_alongside_stamps(monkeypatch):
    monkeypatch.setattr(
        "chad.market_data.futures_contract_resolver.resolve_contract_month",
        lambda symbol: "20260918",
    )
    intent = pr._close_intent_to_ibkr(_close(symbol="MES", strategy="alpha_futures",
                                             position_key="alpha_futures|MES"))
    assert intent.sec_type == "FUT"
    m = intent.meta
    assert m["contract_month"] == "20260918"          # GAP-037 branch intact
    assert m["contract_month_source"]
    assert m["action"] == "CLOSE"
    assert m["reason"] == "exit_overlay_atr_trailing_stop"
    assert m["position_key"] == "alpha_futures|MES"


def test_absent_fields_do_not_mint_stamps():
    intent = pr._close_intent_to_ibkr({
        "symbol": "IWM", "close_side": "SELL", "quantity": 1.0,
    })
    m = intent.meta
    assert m["action"] == "CLOSE"                      # always stamped
    assert m["close_origin"] == "apply_close_intents"
    assert "reason" not in m                           # nothing invented
    assert "position_key" not in m
    assert "open_side" not in m


# --------------------------------------------------------------------------- #
# adapter-boundary visibility + exclusion backstop
# --------------------------------------------------------------------------- #

class _FakeAdapter:
    def __init__(self):
        self.intents: List[Any] = []

    def submit_strategy_trade_intents(self, intents):
        self.intents.extend(intents)
        return [SimpleNamespace(
            symbol=i.symbol, side=i.side, quantity=i.quantity,
            status="rejected", dry_run=True, ib_order_id=None,
            raw={}, error="fake",
        ) for i in intents]


def test_stamps_visible_through_apply_close_intents():
    fake = _FakeAdapter()
    pr.apply_close_intents([_close()], fake)
    assert len(fake.intents) == 1
    m = fake.intents[0].meta
    assert m["action"] == "CLOSE"
    assert m["reason"] == "exit_overlay_atr_trailing_stop"
    assert m["position_key"] == "gamma|UNH"


def test_excluded_symbol_still_skipped_despite_stamps():
    """The GAP-001 chokepoint backstop must not be bypassed by stamped closes."""
    fake = _FakeAdapter()
    excluded = sorted(pr._EFFECTIVE_NON_CHAD_SYMBOLS)[0]  # e.g. AAPL
    pr.apply_close_intents([_close(symbol=excluded)], fake)
    assert fake.intents == []                          # never reached the adapter


# --------------------------------------------------------------------------- #
# idempotency fingerprint invariance
# --------------------------------------------------------------------------- #

def test_stamps_do_not_change_idempotency_fingerprint():
    from chad.execution.ibkr_adapter import IbkrAdapter, NormalizedIntent

    from datetime import datetime, timezone

    def _norm(meta):
        return NormalizedIntent(
            symbol="UNH", side="SELL", quantity=5.0, sec_type="STK",
            exchange="SMART", currency="USD", order_type="MKT",
            strategy="gamma", asset_class="equity", limit_price=None,
            notional_estimate=0.0, source_strategies=("gamma",),
            created_at=datetime(2026, 7, 23, tzinfo=timezone.utc),
            meta=dict(meta),
        )

    bare = IbkrAdapter._stable_idempotency_payload(_norm({}))
    stamped = IbkrAdapter._stable_idempotency_payload(_norm({
        "action": "CLOSE", "close_origin": "apply_close_intents",
        "reason": "exit_overlay_atr_trailing_stop",
        "position_key": "gamma|UNH", "open_side": "BUY", "close_side": "SELL",
    }))
    assert bare == stamped

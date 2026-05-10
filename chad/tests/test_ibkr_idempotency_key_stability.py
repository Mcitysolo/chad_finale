#!/usr/bin/env python3
"""
chad/tests/test_ibkr_idempotency_key_stability.py

Stability tests for the IBKR adapter's idempotency key.

The key MUST be derived only from fields that uniquely identify an order
intent. Cycle-to-cycle noise (created_at, source_strategies attribution,
notional_estimate, free-form meta) MUST NOT change the key — otherwise the
SQLite idempotency store sees a fresh row each cycle, the duplicate-submit
gate becomes a no-op, and IBKR rejects the redundant working orders with
Error 201 ("minimum of 15 orders working ...").

These tests pin the stable-key contract so a regression in
``_stable_idempotency_payload`` is caught before it reaches the broker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from chad.execution.ibkr_adapter import (
    IbkrAdapter,
    IbkrConfig,
    NormalizedIntent,
)
from chad.core.live_loop import _UNCONFIRMED_BROKER_STATUSES


def _make_intent(
    *,
    strategy: str = "alpha",
    symbol: str = "SPY",
    sec_type: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD",
    side: str = "BUY",
    order_type: str = "LMT",
    quantity: float = 10.0,
    limit_price: float = 425.50,
    asset_class: str = "etf",
    notional_estimate: float = 4255.0,
    source_strategies: tuple = ("alpha",),
    created_at: datetime | None = None,
    meta: Mapping[str, Any] | None = None,
) -> NormalizedIntent:
    return NormalizedIntent(
        strategy=strategy,
        symbol=symbol,
        sec_type=sec_type,
        exchange=exchange,
        currency=currency,
        side=side,
        order_type=order_type,
        quantity=quantity,
        notional_estimate=notional_estimate,
        asset_class=asset_class,
        source_strategies=source_strategies,
        created_at=created_at or datetime(2026, 5, 10, 13, 0, 0, tzinfo=timezone.utc),
        limit_price=limit_price,
        meta=dict(meta or {}),
    )


def _adapter(tmp_db: Path | None = None) -> IbkrAdapter:
    cfg = IbkrConfig(
        dry_run=True,
        enable_idempotency=tmp_db is not None,
        state_db_path=tmp_db,
    )
    return IbkrAdapter(config=cfg)


# ---------------------------------------------------------------------------
# Stability across cycles
# ---------------------------------------------------------------------------


def test_same_intent_different_created_at_yields_same_key() -> None:
    """The whole point of the patch: created_at must NOT enter the hash."""
    adapter = _adapter()
    base_ts = datetime(2026, 5, 10, 13, 0, 0, tzinfo=timezone.utc)

    a = _make_intent(created_at=base_ts)
    b = _make_intent(created_at=base_ts + timedelta(seconds=37))
    c = _make_intent(created_at=base_ts + timedelta(minutes=15))

    ka = adapter._compute_idempotency_key(a)
    kb = adapter._compute_idempotency_key(b)
    kc = adapter._compute_idempotency_key(c)

    assert ka == kb == kc, "created_at drift must not change idempotency key"


def test_source_strategies_attribution_does_not_change_key() -> None:
    """Same logical order routed through different attribution must dedupe."""
    adapter = _adapter()

    a = _make_intent(source_strategies=("alpha",))
    b = _make_intent(source_strategies=("alpha", "beta"))

    assert adapter._compute_idempotency_key(a) == adapter._compute_idempotency_key(b)


def test_notional_estimate_drift_does_not_change_key() -> None:
    """Notional drifts with mark-to-market; it must not poison the key."""
    adapter = _adapter()
    a = _make_intent(notional_estimate=4255.00)
    b = _make_intent(notional_estimate=4259.75)

    assert adapter._compute_idempotency_key(a) == adapter._compute_idempotency_key(b)


def test_changed_limit_price_changes_key() -> None:
    adapter = _adapter()
    a = _make_intent(limit_price=425.50)
    b = _make_intent(limit_price=426.00)

    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_changed_quantity_changes_key() -> None:
    adapter = _adapter()
    a = _make_intent(quantity=10.0)
    b = _make_intent(quantity=11.0)

    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_changed_side_changes_key() -> None:
    adapter = _adapter()
    a = _make_intent(side="BUY")
    b = _make_intent(side="SELL")

    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


# ---------------------------------------------------------------------------
# BAG (vertical spread) stability
# ---------------------------------------------------------------------------


def _bag_meta(**overrides: Any) -> Dict[str, Any]:
    base = {
        "expiry": "20260619",
        "long_strike": 425.0,
        "short_strike": 430.0,
        "long_right": "C",
        "short_right": "C",
        "net_debit_estimate": 1.85,
    }
    base.update(overrides)
    return base


def test_bag_same_spread_same_key_across_cycles() -> None:
    adapter = _adapter()
    base_ts = datetime(2026, 5, 10, 13, 0, 0, tzinfo=timezone.utc)

    a = _make_intent(
        sec_type="BAG",
        asset_class="options_spread",
        exchange="SMART",
        meta=_bag_meta(net_debit_estimate=1.85),
        created_at=base_ts,
    )
    b = _make_intent(
        sec_type="BAG",
        asset_class="options_spread",
        exchange="SMART",
        meta=_bag_meta(net_debit_estimate=1.92),  # mark drifted
        created_at=base_ts + timedelta(minutes=10),
    )

    assert adapter._compute_idempotency_key(a) == adapter._compute_idempotency_key(b)


def test_bag_changed_long_strike_changes_key() -> None:
    adapter = _adapter()
    a = _make_intent(sec_type="BAG", asset_class="options_spread", meta=_bag_meta())
    b = _make_intent(
        sec_type="BAG",
        asset_class="options_spread",
        meta=_bag_meta(long_strike=420.0),
    )
    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_bag_changed_expiry_changes_key() -> None:
    adapter = _adapter()
    a = _make_intent(sec_type="BAG", asset_class="options_spread", meta=_bag_meta())
    b = _make_intent(
        sec_type="BAG",
        asset_class="options_spread",
        meta=_bag_meta(expiry="20260717"),
    )
    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


def test_bag_changed_short_right_changes_key() -> None:
    adapter = _adapter()
    a = _make_intent(sec_type="BAG", asset_class="options_spread", meta=_bag_meta())
    b = _make_intent(
        sec_type="BAG",
        asset_class="options_spread",
        meta=_bag_meta(short_right="P"),
    )
    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


# ---------------------------------------------------------------------------
# Futures contract month stability
# ---------------------------------------------------------------------------


def test_futures_contract_month_participates_in_key() -> None:
    adapter = _adapter()
    a = _make_intent(
        symbol="MES",
        sec_type="FUT",
        exchange="CME",
        asset_class="futures",
        meta={"contract_month": "202606"},
    )
    b = _make_intent(
        symbol="MES",
        sec_type="FUT",
        exchange="CME",
        asset_class="futures",
        meta={"contract_month": "202609"},
    )

    assert adapter._compute_idempotency_key(a) != adapter._compute_idempotency_key(b)


# ---------------------------------------------------------------------------
# Duplicate submit suppression
# ---------------------------------------------------------------------------


class _PlaceOrderProbe:
    """IB stub that records whether placeOrder / qualifyContracts was called."""

    def __init__(self) -> None:
        self.place_calls: int = 0
        self.qualify_calls: int = 0

    def isConnected(self) -> bool:
        return True

    def connect(self, *a: Any, **k: Any) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def managedAccounts(self):  # noqa: ANN201
        return ["DU0000000"]

    def qualifyContracts(self, *contracts: Any):  # noqa: ANN201
        self.qualify_calls += 1
        return list(contracts)

    def whatIfOrder(self, contract: Any, order: Any) -> Any:
        return order

    def placeOrder(self, contract: Any, order: Any) -> Any:
        self.place_calls += 1

        class _Trade:
            orderStatus = type("S", (), {"status": "Submitted"})()
            order = None
            fills: list = []
            commissionReport: list = []

        return _Trade()

    def sleep(self, seconds: float) -> None:
        return None


def test_duplicate_intent_blocked_does_not_call_place_order(tmp_path: Path) -> None:
    """
    Submitting the same dry-run intent twice must produce a duplicate_blocked
    result on the second call. dry_run guarantees placeOrder is never invoked
    on either pass — what we are pinning is that the *suppression branch*
    fires (status="duplicate_blocked") on cycle 2.
    """
    db = tmp_path / "ibkr_state.sqlite"
    adapter = _adapter(tmp_db=db)

    intent = _make_intent()

    first = adapter._submit_intent(intent)
    second = adapter._submit_intent(intent)

    assert first is not None and second is not None
    assert first.status == "dry_run"
    assert second.status == "duplicate_blocked"
    assert first.idempotency_key == second.idempotency_key


def test_duplicate_blocked_status_is_in_unconfirmed_broker_statuses() -> None:
    """
    The live-loop paper-evidence gate (commit e9c9954) MUST treat
    duplicate_blocked as an unconfirmed broker status — otherwise the
    suppression branch would synthesize a fake paper fill.
    """
    assert "duplicate_blocked" in _UNCONFIRMED_BROKER_STATUSES

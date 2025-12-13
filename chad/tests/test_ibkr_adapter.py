#!/usr/bin/env python3
"""
chad/tests/test_ibkr_adapter.py

Unit tests for the IBKR execution adapter in DRY_RUN mode.
These tests ensure:

- No external IBKR connection is attempted
- RoutedSignal → SubmittedOrder mapping is correct
- Unsupported asset classes are ignored safely
- Side, size, and strategy attribution is preserved
- Adapter is robust against exceptions for individual signals
"""

from __future__ import annotations

from datetime import datetime, timezone
import math

import pytest

from chad.execution.ibkr_adapter import (
    IbkrAdapter,
    IbkrConfig,
    SubmittedOrder,
)
from chad.types import (
    RoutedSignal,
    AssetClass,
    SignalSide,
    StrategyName,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_routed(
    symbol: str,
    side: SignalSide,
    qty: float,
    strategies: tuple[StrategyName, ...],
    asset_class: AssetClass = AssetClass.EQUITY,
) -> RoutedSignal:
    """Helper to build a RoutedSignal for tests."""
    return RoutedSignal(
        symbol=symbol,
        side=side,
        net_size=qty,
        source_strategies=strategies,
        confidence=0.75,
        asset_class=asset_class,
        created_at=_now(),
        meta={},
    )


def test_dry_run_connection_is_skipped() -> None:
    """
    DRY_RUN=True → ensure_connected() must not raise or attempt real connections.
    """
    cfg = IbkrConfig(
        host="127.0.0.1",
        port=4002,
        client_id=123,
        dry_run=True,
    )
    adapter = IbkrAdapter(config=cfg)

    # Should NOT raise even if IBGW is offline
    adapter.ensure_connected()


def test_dry_run_order_submission_creates_submittedorder() -> None:
    """
    DRY_RUN=True → submit_routed_signals should return a SubmittedOrder
    with dry_run=True and ib_order_id=None.
    """
    r = _make_routed("SPY", SignalSide.BUY, 5.0, (StrategyName.BETA,), AssetClass.ETF)

    cfg = IbkrConfig(dry_run=True)
    adapter = IbkrAdapter(config=cfg)

    submitted = adapter.submit_routed_signals([r])
    assert len(submitted) == 1

    so = submitted[0]
    assert isinstance(so, SubmittedOrder)
    assert so.symbol == "SPY"
    assert so.side == "buy"
    assert math.isclose(so.quantity, 5.0)
    assert so.strategy == ["beta"]
    assert so.dry_run is True
    assert so.ib_order_id is None
    assert so.submitted_at is not None


def test_unsupported_asset_classes_are_skipped() -> None:
    """
    Only EQUITY and ETF are supported in this Phase.
    Anything else should be logged + skipped without raising.
    """
    unsupported = [
        AssetClass.CRYPTO,
        AssetClass.FOREX,
        AssetClass.CASH,
    ]

    signals = [
        _make_routed("BTCUSD", SignalSide.BUY, 1.0, (StrategyName.ALPHA_CRYPTO,), ac)
        for ac in unsupported
    ]

    adapter = IbkrAdapter(IbkrConfig(dry_run=True))
    submitted = adapter.submit_routed_signals(signals)

    # Should be empty → skipped
    assert submitted == []


def test_zero_or_negative_size_is_skipped() -> None:
    """
    net_size <= 0 → ignore safely (no order).
    """
    invalid = [
        _make_routed("SPY", SignalSide.BUY, 0.0, (StrategyName.ALPHA,)),
        _make_routed("SPY", SignalSide.BUY, -5.0, (StrategyName.ALPHA,)),
    ]

    adapter = IbkrAdapter(IbkrConfig(dry_run=True))
    submitted = adapter.submit_routed_signals(invalid)
    assert submitted == []


def test_multiple_routed_signals_independent_handling() -> None:
    """
    Adapter must process each signal independently, even if one fails.
    To simulate failure: temporarily monkeypatch _submit_single_routed.
    """
    s1 = _make_routed("SPY", SignalSide.BUY, 3.0, (StrategyName.BETA,))
    s2 = _make_routed("AAPL", SignalSide.BUY, 2.0, (StrategyName.ALPHA,))

    adapter = IbkrAdapter(IbkrConfig(dry_run=True))

    # Save real method
    real = adapter._submit_single_routed  # type: ignore[attr-defined]

    # Monkeypatch to force s2 to fail
    def fake(self: IbkrAdapter, routed: RoutedSignal):
        if routed.symbol == "AAPL":
            raise RuntimeError("Simulated failure")
        return real(routed)

    adapter._submit_single_routed = fake.__get__(adapter, IbkrAdapter)  # type: ignore[assignment]

    submitted = adapter.submit_routed_signals([s1, s2])

    # s1 should succeed, s2 should be skipped but not crash
    assert len(submitted) == 1
    assert submitted[0].symbol == "SPY"
    assert submitted[0].dry_run is True


def test_shutdown_safe_to_call_multiple_times() -> None:
    """
    shutdown() should never raise, regardless of connection state.
    """
    adapter = IbkrAdapter(IbkrConfig(dry_run=True))
    adapter.shutdown()
    adapter.shutdown()  # second call must also be safe

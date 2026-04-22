"""Phase-8 Session 10 (A3): backtest_engine routed through EMS/gates/SimulatedOMS."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chad.analytics.backtest_engine import BacktestEngine
from chad.execution.oms import SimulatedFillLedger, SimulatedOMS


def _bar_ts() -> datetime:
    return datetime(2025, 6, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Unified-path wiring
# ---------------------------------------------------------------------------


def test_backtest_unified_fill_price_includes_slippage():
    """10 bps slippage on a $200 BUY → fill at $200.20."""
    engine = BacktestEngine(unified_execution=True, slippage_bps=10.0)
    fp, qty, ok = engine._unified_execute_signal(
        symbol="AAPL", side="BUY", size=100, price=200.0,
        strategy_name="alpha", bar_ts=_bar_ts(), confidence=0.7,
    )
    assert ok is True
    assert fp == pytest.approx(200.20, abs=1e-6)
    assert qty == pytest.approx(100.0)


def test_backtest_unified_sell_slippage_reduces_fill():
    engine = BacktestEngine(unified_execution=True, slippage_bps=10.0)
    fp, qty, ok = engine._unified_execute_signal(
        symbol="SPY", side="SELL", size=50, price=500.0,
        strategy_name="alpha", bar_ts=_bar_ts(), confidence=0.5,
    )
    assert ok is True
    assert fp == pytest.approx(499.50, abs=1e-6)


def test_backtest_unified_records_fill_to_ledger():
    ledger = SimulatedFillLedger()
    oms = SimulatedOMS(ledger=ledger, slippage_bps=0.0)
    engine = BacktestEngine(
        unified_execution=True, slippage_bps=0.0,
        simulated_oms=oms, simulated_ledger=ledger,
    )
    fp, qty, ok = engine._unified_execute_signal(
        symbol="QQQ", side="BUY", size=25, price=400.0,
        strategy_name="alpha", bar_ts=_bar_ts(), confidence=0.6,
    )
    assert ok is True
    assert ledger.fill_count == 1
    fill = ledger.get_fills()[0]
    assert fill["symbol"] == "QQQ"
    assert fill["side"] == "BUY"
    assert fill["quantity"] == 25
    assert fill["strategy"] == "alpha"
    assert fill["confidence"] == pytest.approx(0.6)
    assert fill["venue"] == "simulated"


def test_backtest_unified_preserves_existing_pnl_run_completes():
    """Full backtest with unified path still completes and returns results
    for every symbol in the universe (no regression on the run() surface)."""
    engine = BacktestEngine(unified_execution=True, slippage_bps=0.0)
    results = engine.run(
        strategy_name="alpha",
        universe=["AAPL", "MSFT"],
        initial_equity=100_000.0,
    )
    assert set(results.keys()) >= {"AAPL", "MSFT", "_AGGREGATE"}


def test_backtest_legacy_path_preserves_zero_slippage():
    """unified_execution=False is the baseline escape hatch — fills come
    from SimulatedPortfolio directly and no ledger is created."""
    engine = BacktestEngine(unified_execution=False)
    results = engine.run(
        strategy_name="alpha",
        universe=["AAPL"],
        initial_equity=100_000.0,
    )
    assert "AAPL" in results
    assert engine.fill_ledger is None  # legacy path doesn't instantiate one


# ---------------------------------------------------------------------------
# Gate integration
# ---------------------------------------------------------------------------


def test_backtest_unified_accepts_normal_signal():
    """With default config, a typical BUY signal passes all gates."""
    engine = BacktestEngine(unified_execution=True, slippage_bps=0.0)
    _, _, ok = engine._unified_execute_signal(
        symbol="SPY", side="BUY", size=10, price=500.0,
        strategy_name="alpha", bar_ts=_bar_ts(), confidence=0.7,
    )
    assert ok is True


def test_backtest_rejected_signal_recorded_on_ledger():
    """Zero-quantity synthetic signal passes to SimulatedOMS which rejects
    with status='error', and the rejection lands on the ledger."""
    ledger = SimulatedFillLedger()
    oms = SimulatedOMS(ledger=ledger, slippage_bps=0.0)
    # Directly drive the OMS with a bad request to observe rejection path.
    from chad.execution.oms import OrderRequest
    from dataclasses import dataclass

    @dataclass
    class _I:
        symbol: str = "X"
        side: str = "BUY"
        strategy: str = "alpha"
        confidence: float = 0.5
    bad = OrderRequest(intent=_I(), venue="simulated", limit_price=0.0, quantity=0)
    result = oms.submit(bad)
    assert result.status == "error"
    assert ledger.rejection_count == 1


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def test_backtest_accepts_injected_oms_and_ledger():
    ledger = SimulatedFillLedger()
    oms = SimulatedOMS(ledger=ledger, slippage_bps=3.0)
    engine = BacktestEngine(
        unified_execution=True,
        simulated_oms=oms,
        simulated_ledger=ledger,
    )
    # Ensure the injected instances are what _ensure_simulated_oms returns.
    assert engine._ensure_simulated_oms() is oms
    assert engine.fill_ledger is ledger


def test_backtest_default_constructor_lazy_inits_ledger():
    engine = BacktestEngine(unified_execution=True, slippage_bps=0.0)
    # Before any signal: ledger is None (lazy).
    assert engine.fill_ledger is None
    # After triggering unified path: ledger gets created.
    engine._unified_execute_signal(
        symbol="SPY", side="BUY", size=5, price=500.0,
        strategy_name="alpha", bar_ts=_bar_ts(), confidence=0.5,
    )
    assert engine.fill_ledger is not None
    assert engine.fill_ledger.fill_count == 1

"""Phase-8 Session 10 (A3): SimulatedOMS + SimulatedFillLedger tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chad.execution.oms import (
    OMSInterface,
    OrderRequest,
    OrderResult,
    SimulatedFillLedger,
    SimulatedOMS,
    STATUS_ERROR,
    STATUS_SUBMITTED,
    compare_backtest_to_paper,
)


@dataclass
class _FakeIntent:
    symbol: str = "SPY"
    side: str = "BUY"
    strategy: str = "alpha"
    confidence: float = 0.7
    order_urgency: str = "normal"


def _mk_request(
    symbol: str = "SPY",
    side: str = "BUY",
    quantity: int = 100,
    limit_price: float = 500.0,
) -> OrderRequest:
    return OrderRequest(
        intent=_FakeIntent(symbol=symbol, side=side),
        venue="simulated",
        order_type="LMT",
        limit_price=limit_price,
        quantity=quantity,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_simulated_oms_satisfies_interface():
    oms = SimulatedOMS()
    assert isinstance(oms, OMSInterface)


# ---------------------------------------------------------------------------
# Slippage model
# ---------------------------------------------------------------------------


def test_simulated_fill_applies_slippage_buy():
    oms = SimulatedOMS(slippage_bps=10.0)  # 10bps
    req = _mk_request(side="BUY", limit_price=100.0)
    result = oms.submit(req)
    # 10bps of 100 = 0.10; BUY pays up.
    assert result.fill_price == pytest.approx(100.10, abs=1e-9)
    assert result.status == STATUS_SUBMITTED


def test_simulated_fill_applies_slippage_sell():
    oms = SimulatedOMS(slippage_bps=10.0)
    req = _mk_request(side="SELL", limit_price=100.0)
    result = oms.submit(req)
    # SELL gives up 10bps.
    assert result.fill_price == pytest.approx(99.90, abs=1e-9)


def test_simulated_fill_zero_slippage_is_limit_price():
    oms = SimulatedOMS(slippage_bps=0.0)
    req = _mk_request(limit_price=250.0)
    result = oms.submit(req)
    assert result.fill_price == pytest.approx(250.0, abs=1e-9)


def test_simulated_fill_unknown_side_no_adjustment():
    oms = SimulatedOMS(slippage_bps=10.0)
    req = OrderRequest(
        intent=_FakeIntent(side="WEIRD"),
        venue="simulated",
        limit_price=100.0,
        quantity=1,
    )
    result = oms.submit(req)
    # Unknown side → no slippage applied.
    assert result.fill_price == 100.0


# ---------------------------------------------------------------------------
# Ledger recording
# ---------------------------------------------------------------------------


def test_fill_ledger_records_correctly():
    ledger = SimulatedFillLedger()
    oms = SimulatedOMS(ledger=ledger, slippage_bps=5.0)
    oms.submit(_mk_request(symbol="SPY", side="BUY", quantity=100, limit_price=500.0))
    oms.submit(_mk_request(symbol="QQQ", side="SELL", quantity=50, limit_price=400.0))
    fills = ledger.get_fills()
    assert len(fills) == 2
    assert fills[0]["symbol"] == "SPY"
    assert fills[0]["quantity"] == 100
    assert fills[1]["symbol"] == "QQQ"
    assert fills[1]["quantity"] == 50
    assert ledger.fill_count == 2
    assert ledger.rejection_count == 0


def test_ledger_records_rejections_separately():
    ledger = SimulatedFillLedger()
    oms = SimulatedOMS(ledger=ledger)
    # Zero quantity → rejected with status='error'.
    bad = OrderRequest(intent=_FakeIntent(), venue="simulated", limit_price=100.0, quantity=0)
    result = oms.submit(bad)
    assert result.status == STATUS_ERROR
    assert ledger.fill_count == 0
    assert ledger.rejection_count == 1
    assert ledger.get_rejections()[0]["reason"] == "simulated_non_positive_qty_or_price"


def test_ledger_clear_empties_both_lists():
    ledger = SimulatedFillLedger()
    oms = SimulatedOMS(ledger=ledger)
    oms.submit(_mk_request())
    oms.submit(OrderRequest(intent=_FakeIntent(), venue="simulated", quantity=0))
    assert ledger.fill_count == 1 and ledger.rejection_count == 1
    ledger.clear()
    assert ledger.fill_count == 0 and ledger.rejection_count == 0


# ---------------------------------------------------------------------------
# Order-id + cancel contract
# ---------------------------------------------------------------------------


def test_cancel_always_succeeds():
    oms = SimulatedOMS()
    assert oms.cancel("SIM_000001") is True
    assert oms.cancel("nonexistent") is True


def test_order_ids_unique_sequential():
    oms = SimulatedOMS()
    ids = [oms.submit(_mk_request()).order_id for _ in range(5)]
    assert ids == ["SIM_000001", "SIM_000002", "SIM_000003", "SIM_000004", "SIM_000005"]


def test_get_status_returns_submitted_for_simulated():
    oms = SimulatedOMS()
    submitted = oms.submit(_mk_request())
    status = oms.get_status(submitted.order_id)
    # The simulator has no open-order state; status defaults to 'submitted'.
    assert status.status == STATUS_SUBMITTED


# ---------------------------------------------------------------------------
# compare_backtest_to_paper utility
# ---------------------------------------------------------------------------


def test_compare_backtest_to_paper_summary_fields():
    bt = [
        {"symbol": "SPY", "side": "BUY", "fill_price": 500.10},
        {"symbol": "QQQ", "side": "SELL", "fill_price": 399.90},
    ]
    pp = [
        {"symbol": "SPY", "side": "BUY", "fill_price": 500.20},
    ]
    summary = compare_backtest_to_paper(bt, pp)
    assert summary["n_backtest"] == 2
    assert summary["n_paper"] == 1
    # Overlap: (SPY,BUY) appears in both → 1/2 = 0.5.
    assert summary["signal_overlap_pct"] == pytest.approx(0.5)
    assert summary["mean_backtest_fill_price"] == pytest.approx(450.0, abs=1e-6)
    assert summary["mean_paper_fill_price"] == pytest.approx(500.20, abs=1e-6)
    assert summary["mean_slippage_diff_bps"] is not None


def test_compare_backtest_to_paper_empty_inputs():
    summary = compare_backtest_to_paper([], [])
    assert summary["n_backtest"] == 0
    assert summary["n_paper"] == 0
    assert summary["mean_fill_price_diff"] is None

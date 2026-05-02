"""
Tests for chad/execution/net_exposure_gate.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from chad.execution.net_exposure_gate import (
    GateAction,
    GateDecision,
    evaluate_signal,
    run_gate,
    REVERSAL_CONFIDENCE_THRESHOLD,
    STRATEGY_PRIORITY,
)


@dataclass
class MockSignal:
    """Minimal signal stand-in matching TradeSignal-like duck-typed surface."""
    strategy: str = "alpha"
    symbol: str = "SPY"
    side: str = "BUY"
    confidence: float = 0.8
    asset_class: str = "equity"
    size: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


def _open_pos(strategy: str, symbol: str, side: str, asset_class: str = "equity") -> dict:
    return {
        "open": True,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "asset_class": asset_class,
        "quantity": 10.0,
        "size": 10.0,
    }


def test_no_conflict_returns_allow():
    sig = MockSignal(strategy="alpha", symbol="SPY", side="BUY", confidence=0.8)
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions={},
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.ALLOW
    assert decision.reason == "no_conflict"


def test_same_direction_returns_merge():
    sig = MockSignal(strategy="alpha", symbol="SPY", side="BUY", confidence=0.8)
    open_positions = {
        "gamma|SPY": _open_pos("gamma", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.MERGE
    assert decision.conflicting_strategy == "gamma"


def test_weak_opposite_signal_blocked():
    sig = MockSignal(strategy="alpha", symbol="SPY", side="SELL", confidence=0.55)
    open_positions = {
        "gamma|SPY": _open_pos("gamma", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.BLOCK
    assert "below_reversal_threshold" in decision.reason


def test_strong_high_priority_opposite_flips():
    # alpha priority=8 > gamma priority=5; confidence 0.85 > 0.70 threshold
    sig = MockSignal(strategy="alpha", symbol="SPY", side="SELL", confidence=0.85)
    open_positions = {
        "gamma|SPY": _open_pos("gamma", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.FLIP_ALLOWED
    assert decision.conflicting_strategy == "gamma"


def test_exit_signal_always_allowed():
    sig = MockSignal(
        strategy="gamma",
        symbol="SPY",
        side="SELL",
        confidence=0.3,
        meta={"exit": True, "reason": "max_hold_exit"},
    )
    # Even with low confidence + opposite direction conflict, exit passes
    open_positions = {
        "alpha|SPY": _open_pos("alpha", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="RED",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.CLOSE_ONLY


def test_hedge_signal_always_allowed():
    sig = MockSignal(
        strategy="omega_vol",
        symbol="VIX",
        side="BUY",
        confidence=0.5,
        tags=["hedge"],
    )
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions={},
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.ALLOW
    assert decision.reason == "hedge_tagged_within_budget"


def test_reconciliation_red_blocks_opposite():
    sig = MockSignal(strategy="alpha", symbol="SPY", side="SELL", confidence=0.85)
    open_positions = {
        "gamma|SPY": _open_pos("gamma", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="RED",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.BLOCK
    assert "reconciliation_RED_blocks_opposite_exposure" in decision.reason


def test_lower_priority_strategy_blocked():
    # omega_macro priority=2 < alpha priority=8; strong confidence still blocks
    sig = MockSignal(strategy="omega_macro", symbol="SPY", side="SELL", confidence=0.85)
    open_positions = {
        "alpha|SPY": _open_pos("alpha", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.BLOCK
    assert "lower_priority" in decision.reason


def test_equal_priority_returns_reduce():
    # gamma_reversion priority=4 == gamma_futures priority=4
    sig = MockSignal(
        strategy="gamma_reversion", symbol="SPY", side="SELL", confidence=0.85
    )
    open_positions = {
        "gamma_futures|SPY": _open_pos("gamma_futures", "SPY", "BUY"),
    }
    decision = evaluate_signal(
        signal=sig,
        signal_index=0,
        open_positions=open_positions,
        reconciliation_status="GREEN",
        portfolio_equity=200000.0,
    )
    assert decision.action == GateAction.REDUCE
    assert "equal_priority_reduce" in decision.reason


def test_gate_failure_allows_all():
    """If gate context-load raises, run_gate must return all signals."""
    sigs = [MockSignal(strategy="alpha", symbol="SPY", side="BUY")]

    with patch(
        "chad.execution.net_exposure_gate._get_open_positions",
        side_effect=RuntimeError("boom"),
    ):
        allowed, decisions = run_gate(sigs)

    assert allowed == sigs
    assert decisions == []


def test_symbol_daily_loss_limit_blocks_when_exceeded():
    """If alpha SELL LLY has lost more than the limit today, block new SELL entries."""
    sig = MockSignal(strategy="alpha", symbol="LLY", side="SELL", confidence=0.85)
    with patch(
        "chad.execution.net_exposure_gate._compute_symbol_daily_pnl",
        return_value={("alpha", "LLY", "SELL"): -500.0},
    ):
        decision = evaluate_signal(
            signal=sig,
            signal_index=0,
            open_positions={},
            reconciliation_status="GREEN",
            portfolio_equity=200000.0,
        )
    assert decision.action == GateAction.BLOCK
    assert "symbol_daily_loss_limit" in decision.reason


def test_symbol_daily_loss_limit_allows_when_under_threshold():
    """Loss under $300 limit should not block."""
    sig = MockSignal(strategy="alpha", symbol="LLY", side="SELL", confidence=0.85)
    with patch(
        "chad.execution.net_exposure_gate._compute_symbol_daily_pnl",
        return_value={("alpha", "LLY", "SELL"): -100.0},
    ):
        decision = evaluate_signal(
            signal=sig,
            signal_index=0,
            open_positions={},
            reconciliation_status="GREEN",
            portfolio_equity=200000.0,
        )
    assert decision.action == GateAction.ALLOW


def test_symbol_daily_loss_limit_never_blocks_exit():
    """Even when daily loss exceeds limit, an exit-tagged signal must pass."""
    sig = MockSignal(
        strategy="alpha",
        symbol="LLY",
        side="SELL",
        confidence=0.85,
        meta={"exit": True, "reason": "max_hold_exit"},
    )
    with patch(
        "chad.execution.net_exposure_gate._compute_symbol_daily_pnl",
        return_value={("alpha", "LLY", "SELL"): -500.0},
    ):
        decision = evaluate_signal(
            signal=sig,
            signal_index=0,
            open_positions={},
            reconciliation_status="GREEN",
            portfolio_equity=200000.0,
        )
    assert decision.action == GateAction.CLOSE_ONLY


def test_symbol_daily_loss_limit_allows_opposite_direction():
    """Loss key is (strategy, symbol, side) — opposite side is a different key."""
    sig = MockSignal(strategy="alpha", symbol="LLY", side="BUY", confidence=0.85)
    with patch(
        "chad.execution.net_exposure_gate._compute_symbol_daily_pnl",
        return_value={("alpha", "LLY", "SELL"): -500.0},
    ):
        decision = evaluate_signal(
            signal=sig,
            signal_index=0,
            open_positions={},
            reconciliation_status="GREEN",
            portfolio_equity=200000.0,
        )
    assert decision.action == GateAction.ALLOW


def test_run_gate_filters_blocked_returns_decisions():
    """End-to-end run_gate filters BLOCK and returns full decision log."""
    sigs = [
        MockSignal(strategy="alpha", symbol="SPY", side="BUY", confidence=0.8),
        MockSignal(strategy="omega_macro", symbol="QQQ", side="SELL", confidence=0.85),
    ]
    fake_positions = {
        "alpha|QQQ": _open_pos("alpha", "QQQ", "BUY"),
    }

    with patch(
        "chad.execution.net_exposure_gate._get_open_positions",
        return_value=fake_positions,
    ), patch(
        "chad.execution.net_exposure_gate._get_reconciliation_status",
        return_value="GREEN",
    ), patch(
        "chad.execution.net_exposure_gate._get_portfolio_equity",
        return_value=200000.0,
    ):
        allowed, decisions = run_gate(sigs)

    # First signal allowed (no conflict), second blocked (lower priority)
    assert len(allowed) == 1
    assert allowed[0].symbol == "SPY"
    assert len(decisions) == 2
    assert decisions[0].action == GateAction.ALLOW
    assert decisions[1].action == GateAction.BLOCK

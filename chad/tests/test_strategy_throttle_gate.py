"""
Tests for chad/execution/strategy_throttle_gate.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from chad.execution import strategy_throttle_gate as gate_mod
from chad.execution.strategy_throttle_gate import (
    LOSS_STREAK_FOR_PAUSE,
    EDGE_DECAY_LOSS_STREAK,
    StrategyStats,
    ThrottleDecision,
    ThrottleLevel,
    evaluate_signal,
    run_throttle_gate,
)


@dataclass
class MockSignal:
    strategy: str = "alpha"
    symbol: str = "SPY"
    side: str = "BUY"
    confidence: float = 0.6
    asset_class: str = "equity"
    size: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset in-memory state between tests."""
    gate_mod._ENTRY_TIMESTAMPS.clear()
    gate_mod._PAUSED_UNTIL.clear()
    yield
    gate_mod._ENTRY_TIMESTAMPS.clear()
    gate_mod._PAUSED_UNTIL.clear()


def _winning_stats(strategy: str = "alpha") -> Dict[str, StrategyStats]:
    s = StrategyStats(
        trades_today=20, wins_today=14, losses_today=6,
        pnl_today=500.0, loss_streak=0, win_rate_today=0.7,
    )
    return {strategy: s}


def _losing_stats(
    strategy: str = "alpha",
    win_rate: float = 0.30,
    pnl: float = -300.0,
    loss_streak: int = 0,
    trades: int = 20,
) -> Dict[str, StrategyStats]:
    wins = int(round(trades * win_rate))
    losses = trades - wins
    s = StrategyStats(
        trades_today=trades, wins_today=wins, losses_today=losses,
        pnl_today=pnl, loss_streak=loss_streak, win_rate_today=win_rate,
    )
    return {strategy: s}


def test_winning_strategy_allowed_unconditionally():
    sig = MockSignal(strategy="alpha", confidence=0.55)
    decision = evaluate_signal(
        signal=sig,
        strategy_stats=_winning_stats(),
        recon_ok=True,
        profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.ALLOW
    assert decision.reason == "winning_strategy_unrestricted"


def test_insufficient_data_allowed():
    sig = MockSignal(strategy="alpha", confidence=0.3)
    stats = {"alpha": StrategyStats(trades_today=3)}
    decision = evaluate_signal(
        signal=sig, strategy_stats=stats,
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.ALLOW
    assert "insufficient_data" in decision.reason


def test_exit_signal_never_throttled():
    # Strategy is in deep trouble but signal is an exit
    sig = MockSignal(
        strategy="alpha", side="SELL",
        meta={"exit": True, "reason": "max_hold_exit"},
    )
    losing = _losing_stats(loss_streak=10, win_rate=0.1, pnl=-9999)
    decision = evaluate_signal(
        signal=sig, strategy_stats=losing,
        recon_ok=False, profit_lock_ok=False,
    )
    assert decision.level == ThrottleLevel.ALLOW
    assert "exit_or_protective" in decision.reason


def test_hedge_signal_never_throttled():
    sig = MockSignal(
        strategy="alpha", side="BUY",
        tags=["hedge"],
    )
    losing = _losing_stats(loss_streak=10, win_rate=0.1, pnl=-9999)
    decision = evaluate_signal(
        signal=sig, strategy_stats=losing,
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.ALLOW
    assert "exit_or_protective" in decision.reason


def test_losing_strategy_throttled_in_time_window():
    """Win rate 0.42 (≤0.45) AND already 3+ entries in last 15 min → THROTTLE."""
    # Pre-populate entry timestamps to trigger the time window check
    now = time.time()
    gate_mod._ENTRY_TIMESTAMPS["alpha"] = [now - 60, now - 120, now - 180]

    sig = MockSignal(strategy="alpha", confidence=0.7)
    # Use win_rate above CONFIDENCE_UPSHIFT (0.40) and PAUSE (0.35)
    # but below THROTTLE (0.45)
    stats = _losing_stats(win_rate=0.42, pnl=-100.0, loss_streak=2)
    decision = evaluate_signal(
        signal=sig, strategy_stats=stats,
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.THROTTLE
    assert "time_window_exceeded" in decision.reason


def test_losing_strategy_confidence_upshift():
    """Win rate 0.40 (≤0.40), low confidence signal → CONFIDENCE_UPSHIFT."""
    # Use loss_streak=2 to avoid pause trigger; pnl positive avoids any path issue
    sig = MockSignal(strategy="alpha", confidence=0.55)  # below 0.65 floor
    stats = _losing_stats(win_rate=0.40, pnl=-50.0, loss_streak=2)
    decision = evaluate_signal(
        signal=sig, strategy_stats=stats,
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.CONFIDENCE_UPSHIFT
    assert decision.confidence_floor == pytest.approx(0.65, abs=1e-6)


def test_loss_streak_triggers_pause():
    """Loss streak ≥4 (and < EDGE_DECAY_LOSS_STREAK=5) triggers PAUSE."""
    sig = MockSignal(strategy="alpha", confidence=0.8)
    stats = _losing_stats(
        win_rate=0.42, pnl=-200.0,
        loss_streak=LOSS_STREAK_FOR_PAUSE,  # 4
    )
    decision = evaluate_signal(
        signal=sig, strategy_stats=stats,
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.PAUSE_TEMPORARILY
    assert decision.pause_until_utc is not None


def test_pause_expires_and_strategy_resumes():
    """An expired pause window does not block."""
    # Set pause to a time in the past
    gate_mod._PAUSED_UNTIL["alpha"] = time.time() - 60

    sig = MockSignal(strategy="alpha", confidence=0.8)
    # Use winning stats so it should ALLOW once pause is expired
    decision = evaluate_signal(
        signal=sig, strategy_stats=_winning_stats(),
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.ALLOW


def test_edge_decay_threshold_defers():
    """Loss streak ≥ EDGE_DECAY_LOSS_STREAK → HALT_DEFER_TO_EDGE_DECAY."""
    sig = MockSignal(strategy="alpha", confidence=0.8)
    stats = _losing_stats(
        win_rate=0.40, pnl=-500.0,
        loss_streak=EDGE_DECAY_LOSS_STREAK,  # 5
    )
    decision = evaluate_signal(
        signal=sig, strategy_stats=stats,
        recon_ok=True, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.HALT_DEFER_TO_EDGE_DECAY


def test_reconciliation_not_green_still_allows_exits():
    """Even when recon is not GREEN, exits pass through."""
    sig = MockSignal(
        strategy="alpha", side="SELL",
        meta={"tags": ["exit"]},
    )
    decision = evaluate_signal(
        signal=sig, strategy_stats=_winning_stats(),
        recon_ok=False, profit_lock_ok=True,
    )
    assert decision.level == ThrottleLevel.ALLOW


def test_gate_failure_allows_all():
    """If gate context-load raises, run_throttle_gate must return all signals."""
    sigs = [MockSignal(strategy="alpha")]

    with patch.object(
        gate_mod, "_load_strategy_stats_today",
        side_effect=RuntimeError("boom"),
    ):
        allowed, decisions = run_throttle_gate(sigs)

    assert allowed == sigs
    assert decisions == []


def test_winning_strategy_after_losses_resets():
    """A strategy that was losing but is now winning should ALLOW."""
    # Even with prior pause set, winning stats win because they short-circuit
    # the pause check (winning is checked before pause).
    gate_mod._PAUSED_UNTIL["alpha"] = time.time() + 600  # paused

    sig = MockSignal(strategy="alpha", confidence=0.6)
    decision = evaluate_signal(
        signal=sig, strategy_stats=_winning_stats(),
        recon_ok=True, profit_lock_ok=True,
    )
    # Winning short-circuit should bypass the pause flag
    assert decision.level == ThrottleLevel.ALLOW
    assert decision.reason == "winning_strategy_unrestricted"


def test_run_throttle_gate_records_entries_for_allowed():
    """run_throttle_gate should record entry timestamps for allowed signals."""
    sigs = [MockSignal(strategy="alpha")]
    with patch.object(
        gate_mod, "_load_strategy_stats_today", return_value={},
    ), patch.object(
        gate_mod, "_check_reconciliation_ok", return_value=True,
    ), patch.object(
        gate_mod, "_check_profit_lock_ok", return_value=True,
    ):
        allowed, decisions = run_throttle_gate(sigs)

    assert len(allowed) == 1
    assert decisions[0].level == ThrottleLevel.ALLOW
    # Entry should have been recorded
    assert "alpha" in gate_mod._ENTRY_TIMESTAMPS
    assert len(gate_mod._ENTRY_TIMESTAMPS["alpha"]) == 1

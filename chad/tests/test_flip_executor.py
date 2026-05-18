"""
BG11 — flip executor enforces close-first / open-second.

These tests exercise chad/core/flip_executor.enforce_flip_close_first
without touching real broker / position_guard state. _load_state and
save_state are monkeypatched per-test so the on-disk state file is
never read or written.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

from chad.core import flip_executor as fx  # noqa: E402
from chad.execution.net_exposure_gate import (  # noqa: E402
    GateAction,
    GateDecision,
)


def _flip_decision(
    *,
    new_strategy: str,
    symbol: str,
    existing_strategy: str,
    existing_side: str,
) -> GateDecision:
    return GateDecision(
        action=GateAction.FLIP_ALLOWED,
        reason="test_flip",
        signal_index=0,
        symbol=symbol,
        strategy=new_strategy,
        conflicting_strategy=existing_strategy,
        conflicting_side=existing_side,
        flip_close_strategy=existing_strategy,
        flip_close_symbol=symbol,
        flip_close_side=existing_side,
    )


def _flipped_signal(strategy: str, symbol: str, new_side: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        symbol=symbol,
        side=new_side,
        confidence=0.85,
        meta={},
    )


def _open_position(strategy: str, symbol: str, side: str, qty: float = 10.0):
    return {
        "open": True,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": qty,
        "last_state": "OPEN",
    }


def _make_submitted(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="IWM",
        side="SELL",
        quantity=10.0,
        status=status,
        submitted_at=datetime.now(timezone.utc),
        limit_price=100.0,
        asset_class="EQUITY",
    )


class _RecordingAdapter:
    """Captures the order in which close intents and would-be opens
    arrive. Returns a configurable status to drive the close path."""

    def __init__(self, status: str = "filled", raise_exc: bool = False) -> None:
        self.status = status
        self.raise_exc = raise_exc
        self.calls: List[List[Any]] = []

    def submit_strategy_trade_intents(self, intents):
        self.calls.append(list(intents))
        if self.raise_exc:
            raise RuntimeError("simulated broker outage")
        return [
            SimpleNamespace(
                symbol=getattr(it, "symbol", "?"),
                side=getattr(it, "side", "?"),
                quantity=getattr(it, "quantity", 0.0),
                status=self.status,
                submitted_at=datetime.now(timezone.utc),
                limit_price=100.0,
                asset_class="EQUITY",
            )
            for it in intents
        ]


@pytest.fixture
def patched_guard(monkeypatch):
    """Provides a writable in-memory position guard that flip_executor
    sees instead of the on-disk file."""
    state: Dict[str, Dict[str, Any]] = {
        "delta|IWM": _open_position("delta", "IWM", "BUY"),
    }
    saved: List[Dict[str, Any]] = []

    def _fake_load_state():
        return {k: dict(v) for k, v in state.items()}

    def _fake_save_state(s):
        saved.append({k: dict(v) for k, v in s.items() if isinstance(v, dict)})
        state.clear()
        state.update(s)

    monkeypatch.setattr(fx, "_load_state", _fake_load_state)
    monkeypatch.setattr(fx, "save_state", _fake_save_state)
    return SimpleNamespace(state=state, saved=saved)


def test_flip_allowed_requires_close_before_open(patched_guard):
    """Phase A submits the close intent for the existing delta|IWM BUY
    BEFORE the flipped alpha SELL is allowed through.

    Symbol switched from SPY → IWM in GAP-001 Phase-48 because SPY now
    routes through the BG11_FLIP_SKIP_EXCLUDED operator-exclusion guard;
    the BG11 close-first flow is still exercised here against a
    non-excluded equity.
    """
    decisions = [
        _flip_decision(
            new_strategy="alpha",
            symbol="IWM",
            existing_strategy="delta",
            existing_side="BUY",
        ),
    ]
    flipped = _flipped_signal("alpha", "IWM", "SELL")
    adapter = _RecordingAdapter(status="filled")

    out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    # Adapter was called exactly once — for the close.
    assert len(adapter.calls) == 1, "close intent must be submitted exactly once"
    close_intent = adapter.calls[0][0]
    assert close_intent.symbol == "IWM"
    # Close side must be the OPPOSITE of the existing (open) side.
    assert close_intent.side == "SELL"
    assert close_intent.strategy == "delta"
    # Flipped entry survives because close was confirmed.
    assert flipped in out
    assert any(r["event"] == "BG11_FLIP_CLOSE_CONFIRMED" for r in audit)


def test_flip_close_failure_blocks_new_entry(patched_guard):
    """Broker rejected close → new flipped entry is BLOCKED and
    position_guard is NOT mutated to a false-flat state."""
    decisions = [
        _flip_decision(
            new_strategy="alpha",
            symbol="IWM",
            existing_strategy="delta",
            existing_side="BUY",
        ),
    ]
    flipped = _flipped_signal("alpha", "IWM", "SELL")
    adapter = _RecordingAdapter(status="rejected")

    out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    assert flipped not in out, "flipped entry must be dropped on close failure"
    assert any(
        r["event"] == "BG11_FLIP_BLOCKED_CLOSE_NOT_CONFIRMED"
        and r["reason"].startswith("broker_status=")
        for r in audit
    )
    # Existing position is still OPEN — no false flat.
    assert patched_guard.state["delta|IWM"]["open"] is True
    # No save_state was issued.
    assert not patched_guard.saved


def test_flip_close_success_allows_new_entry(patched_guard):
    """Broker-confirmed close → flipped entry proceeds, position_guard
    is mutated to closed by 'flip_executor'."""
    decisions = [
        _flip_decision(
            new_strategy="alpha",
            symbol="IWM",
            existing_strategy="delta",
            existing_side="BUY",
        ),
    ]
    flipped = _flipped_signal("alpha", "IWM", "SELL")
    adapter = _RecordingAdapter(status="filled")

    out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    assert flipped in out
    assert patched_guard.state["delta|IWM"]["open"] is False
    assert patched_guard.state["delta|IWM"]["closed_by"] == "flip_executor"
    assert any(r["event"] == "BG11_FLIP_CLOSE_CONFIRMED" for r in audit)


def test_flip_no_close_confirmation_does_not_mutate_position_guard(
    patched_guard,
):
    """Adapter raised an exception (close status unknown). Flipped entry
    is dropped AND position_guard is unchanged — no false flat."""
    decisions = [
        _flip_decision(
            new_strategy="alpha",
            symbol="IWM",
            existing_strategy="delta",
            existing_side="BUY",
        ),
    ]
    flipped = _flipped_signal("alpha", "IWM", "SELL")
    adapter = _RecordingAdapter(raise_exc=True)

    out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    assert flipped not in out
    assert any(
        r["event"] == "BG11_FLIP_BLOCKED_CLOSE_NOT_CONFIRMED"
        and r["reason"].startswith("close_submit_exception:")
        for r in audit
    )
    # Existing position is still OPEN — no false flat.
    assert patched_guard.state["delta|IWM"]["open"] is True
    assert not patched_guard.saved


def test_pending_status_treated_as_unconfirmed(patched_guard):
    """PendingSubmit / PreSubmitted are not broker confirmations."""
    decisions = [
        _flip_decision(
            new_strategy="alpha",
            symbol="IWM",
            existing_strategy="delta",
            existing_side="BUY",
        ),
    ]
    flipped = _flipped_signal("alpha", "IWM", "SELL")
    adapter = _RecordingAdapter(status="PendingSubmit")

    out, _audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    assert flipped not in out
    assert patched_guard.state["delta|IWM"]["open"] is True


def test_flip_with_no_open_position_passes_through(patched_guard):
    """If the conflict referenced in the decision no longer exists in
    position_guard, the flipped entry is allowed without any close — the
    'flip' is moot."""
    # remove the open position we set up in the fixture
    patched_guard.state.clear()
    decisions = [
        _flip_decision(
            new_strategy="alpha",
            symbol="IWM",
            existing_strategy="delta",
            existing_side="BUY",
        ),
    ]
    flipped = _flipped_signal("alpha", "IWM", "SELL")
    adapter = _RecordingAdapter(status="filled")

    out, audit = fx.enforce_flip_close_first([flipped], decisions, adapter)

    assert flipped in out
    assert adapter.calls == [], "no close should be submitted when no open exists"
    assert any(r["event"] == "BG11_FLIP_NO_OPEN_POSITION" for r in audit)

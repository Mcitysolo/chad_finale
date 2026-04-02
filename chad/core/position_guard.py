#!/usr/bin/env python3
"""
chad/core/position_guard.py

Position-state memory for CHAD.

Purpose
-------
Track currently open strategy/symbol positions and allow:
- same-side duplicate blocking
- opposite-side flip detection
- explicit open / close / replace state updates
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional
import json

STATE_PATH = Path("/home/ubuntu/chad_finale/runtime/position_guard.json")


class PositionState(str, Enum):
    """Formal position lifecycle states (SSOT v6.4)."""
    OPEN = "OPEN"                                          # Fresh position opened
    MAINTAINED = "MAINTAINED"                              # Same-side signal on existing open — no change
    FLIPPED = "FLIPPED"                                    # Opposite-side replaced existing position
    CLOSED = "CLOSED"                                      # Position explicitly closed
    RESET_FROM_BROKER_TRUTH = "RESET_FROM_BROKER_TRUTH"    # Reconciled from broker state


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state() -> Dict[str, dict]:
    if not STATE_PATH.is_file():
        return {}
    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_state(state: Dict[str, dict]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _intent_strategy(intent) -> str:
    return str(getattr(intent, "strategy", "") or "")


def _intent_symbol(intent) -> str:
    return str(getattr(intent, "symbol", "") or "")


def _intent_side(intent) -> str:
    return str(getattr(intent, "side", "") or "")


def _position_key(strategy: str, symbol: str) -> str:
    return f"{strategy}|{symbol}"


def get_open_position(intent) -> Optional[dict]:
    state = _load_state()
    key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
    record = state.get(key)
    if record and record.get("open") is True:
        return record
    return None


def has_open_position(intent) -> bool:
    return get_open_position(intent) is not None


def is_same_side_open(intent) -> bool:
    record = get_open_position(intent)
    if not record:
        return False
    match = str(record.get("side", "")) == _intent_side(intent)
    if match:
        # Record MAINTAINED state — position unchanged by this signal
        state = _load_state()
        key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
        if key in state:
            state[key]["last_state"] = PositionState.MAINTAINED.value
            _save_state(state)
    return match


def is_flip_signal(intent) -> bool:
    record = get_open_position(intent)
    if not record:
        return False
    open_side = str(record.get("side", ""))
    new_side = _intent_side(intent)
    return open_side != "" and new_side != "" and open_side != new_side


def mark_position_open(intent) -> None:
    state = _load_state()
    key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
    state[key] = {
        "open": True,
        "updated_at_utc": _utc_now_iso(),
        "strategy": _intent_strategy(intent),
        "symbol": _intent_symbol(intent),
        "side": _intent_side(intent),
        "quantity": float(getattr(intent, "quantity", 0.0) or 0.0),
        "last_state": PositionState.OPEN.value,
    }
    _save_state(state)


def mark_position_closed(intent) -> None:
    state = _load_state()
    key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
    if key in state:
        state[key]["open"] = False
        state[key]["updated_at_utc"] = _utc_now_iso()
        state[key]["last_state"] = PositionState.CLOSED.value
        _save_state(state)


def replace_position(intent) -> None:
    """
    Used for a flip:
    close whatever side was open for strategy+symbol,
    then open the new side.
    """
    state = _load_state()
    key = _position_key(_intent_strategy(intent), _intent_symbol(intent))
    state[key] = {
        "open": True,
        "updated_at_utc": _utc_now_iso(),
        "strategy": _intent_strategy(intent),
        "symbol": _intent_symbol(intent),
        "side": _intent_side(intent),
        "quantity": float(getattr(intent, "quantity", 0.0) or 0.0),
        "last_state": PositionState.FLIPPED.value,
    }
    _save_state(state)


def reset_from_broker(strategy: str, symbol: str) -> None:
    """Reconcile a position from broker truth — marks entry as reset."""
    state = _load_state()
    key = _position_key(strategy, symbol)
    if key in state:
        state[key]["open"] = False
        state[key]["updated_at_utc"] = _utc_now_iso()
        state[key]["last_state"] = PositionState.RESET_FROM_BROKER_TRUTH.value
    else:
        state[key] = {
            "open": False,
            "updated_at_utc": _utc_now_iso(),
            "strategy": strategy,
            "symbol": symbol,
            "side": "",
            "quantity": 0.0,
            "last_state": PositionState.RESET_FROM_BROKER_TRUTH.value,
        }
    _save_state(state)


def reset_all_positions() -> None:
    _save_state({})

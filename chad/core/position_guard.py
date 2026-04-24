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


def _reduce_or_close_broker_sync(
    state: Dict[str, dict],
    broker_sync_key: str,
    strategy: str,
    side: str,
    quantity: float,
) -> None:
    """ISSUE-56 v2: reduce-not-close for partial broker_sync attribution.

    Same side, residual > 0  → reduce quantity, keep open=True.
    Same side, residual <= 0 → soft-close (full attribution).
    Opposite side            → untouched (flip intent; reconciliation surfaces drift).
    """
    if strategy == "broker_sync":
        return
    bs_entry = state.get(broker_sync_key)
    if not bs_entry or not bs_entry.get("open"):
        return
    bs_side = str(bs_entry.get("side", "") or "").upper()
    incoming_side = str(side or "").upper()
    if bs_side != incoming_side:
        return
    bs_qty = abs(float(bs_entry.get("quantity", 0) or 0))
    residual = bs_qty - abs(float(quantity or 0))
    bs_entry["updated_at_utc"] = _utc_now_iso()
    if residual <= 0:
        bs_entry["open"] = False
        bs_entry["closed_by"] = "strategy_ownership_assumed"
    else:
        bs_entry["quantity"] = residual
        bs_entry["open"] = True
        bs_entry["closed_by"] = "partial_attribution_residual"


def mark_position_open(intent) -> None:
    state = _load_state()
    strategy = _intent_strategy(intent)
    symbol = _intent_symbol(intent)
    side = _intent_side(intent)
    quantity = float(getattr(intent, "quantity", 0.0) or 0.0)
    key = _position_key(strategy, symbol)
    broker_sync_key = _position_key("broker_sync", symbol)
    _reduce_or_close_broker_sync(state, broker_sync_key, strategy, side, quantity)
    now_iso = _utc_now_iso()
    prior = state.get(key) or {}
    prior_opened = None
    if prior.get("open") is True and str(prior.get("side")) == str(side):
        prior_opened = prior.get("opened_at")
    state[key] = {
        "open": True,
        "opened_at": prior_opened or now_iso,
        "updated_at_utc": now_iso,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
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
    strategy = _intent_strategy(intent)
    symbol = _intent_symbol(intent)
    side = _intent_side(intent)
    quantity = float(getattr(intent, "quantity", 0.0) or 0.0)
    key = _position_key(strategy, symbol)
    broker_sync_key = _position_key("broker_sync", symbol)
    _reduce_or_close_broker_sync(state, broker_sync_key, strategy, side, quantity)
    now_iso = _utc_now_iso()
    state[key] = {
        "open": True,
        "opened_at": now_iso,
        "updated_at_utc": now_iso,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
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

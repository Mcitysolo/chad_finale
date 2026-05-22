#!/usr/bin/env python3
"""
chad/core/signal_guard.py

Signal deduplication and cooldown guard for CHAD.

Cooldown scope contract (SSOT v6.4)
------------------------------------
- Fingerprint key: (strategy, symbol, side, size).
- Size-sensitive by design: changing quantity produces a distinct
  fingerprint, so a resized order is treated as a new signal.
- Opposite-side signals have distinct fingerprints and are never
  cross-blocked (BUY AAPL 100 vs SELL AAPL 100 are independent).
- Cooldown window: 10 minutes (COOLDOWN_MINUTES = 10, i.e. 600 s).
  Purely time-based; evaluated against updated_at_utc of the last
  emission for the same fingerprint.
- Known gap: broker-confirmed position closes do NOT reset cooldown.
  clear_signal() sets active=False / EXPIRED, but no broker
  confirmation path currently calls it.  Cooldown expires only by
  the passage of time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict
import json

STATE_PATH = Path("/home/ubuntu/chad_finale/runtime/signal_guard.json")
COOLDOWN_MINUTES = 10


class SignalState(str, Enum):
    """Formal signal lifecycle states (SSOT v6.4)."""
    NEW = "NEW"                                        # First occurrence of this fingerprint
    DUPLICATE = "DUPLICATE"                            # Active entry past cooldown — re-emitted
    COOLDOWN_BLOCKED = "COOLDOWN_BLOCKED"              # Active entry within cooldown — blocked
    EXPIRED = "EXPIRED"                                # Entry cleared (active=False)
    REACTIVATABLE = "REACTIVATABLE"                    # Expired entry seen again — will re-emit
    SUPPRESSED_BY_POSITION = "SUPPRESSED_BY_POSITION"  # Blocked by position guard (assigned by caller)
    EXECUTABLE = "EXECUTABLE"                          # Passed all guards, ready to execute


@dataclass(frozen=True)
class SignalFingerprint:
    strategy: str
    symbol: str
    side: str
    size: float

    def key(self) -> str:
        """Cooldown identity: strategy|symbol|side|size — all four components must match."""
        return f"{self.strategy}|{self.symbol}|{self.side}|{self.size}"


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


def fingerprint_signal(signal) -> SignalFingerprint:
    strategy = getattr(getattr(signal, "strategy", None), "value", getattr(signal, "strategy", None))
    side = getattr(getattr(signal, "side", None), "value", getattr(signal, "side", None))

    return SignalFingerprint(
        strategy=str(strategy),
        symbol=str(getattr(signal, "symbol", "")),
        side=str(side),
        size=float(getattr(signal, "size", 0.0)),
    )


def classify_signal(signal) -> SignalState:
    """Classify a signal's lifecycle state without side effects (pure read)."""
    fp = fingerprint_signal(signal)
    state = _load_state()
    key = fp.key()
    existing = state.get(key)

    if existing is None:
        return SignalState.NEW

    if existing.get("active") is not True:
        return SignalState.REACTIVATABLE

    last_ts = existing.get("updated_at_utc")
    if last_ts:
        try:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            if now_dt - last_dt < timedelta(minutes=COOLDOWN_MINUTES):
                return SignalState.COOLDOWN_BLOCKED
        except Exception:
            return SignalState.COOLDOWN_BLOCKED

    return SignalState.DUPLICATE


def should_emit_signal(signal) -> bool:
    fp = fingerprint_signal(signal)
    state = _load_state()
    key = fp.key()

    existing = state.get(key)

    # 🔥 COOLDOWN LOGIC (correct placement)
    if existing and existing.get("active") is True:
        last_ts = existing.get("updated_at_utc")

        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)

                if now_dt - last_dt < timedelta(minutes=COOLDOWN_MINUTES):
                    existing["last_state"] = SignalState.COOLDOWN_BLOCKED.value
                    _save_state(state)
                    return False
            except Exception:
                existing["last_state"] = SignalState.COOLDOWN_BLOCKED.value
                _save_state(state)
                return False

    # Determine lifecycle state for the record
    if existing is None:
        sig_state = SignalState.NEW
    elif existing.get("active") is not True:
        sig_state = SignalState.REACTIVATABLE
    else:
        sig_state = SignalState.DUPLICATE

    # store/update signal
    state[key] = {
        "active": True,
        "updated_at_utc": _utc_now_iso(),
        "strategy": fp.strategy,
        "symbol": fp.symbol,
        "side": fp.side,
        "size": fp.size,
        "last_state": sig_state.value,
    }

    _save_state(state)
    return True


def clear_signal(signal) -> None:
    fp = fingerprint_signal(signal)
    state = _load_state()
    key = fp.key()

    if key in state:
        state[key]["active"] = False
        state[key]["updated_at_utc"] = _utc_now_iso()
        state[key]["last_state"] = SignalState.EXPIRED.value
        _save_state(state)


def revert_emission_for_unconfirmed(signal) -> bool:
    """Undo the cooldown-arming write performed by ``should_emit_signal`` when
    the downstream submission was never accepted by the broker.

    Background — OPS-OMEGA-01 (Pattern C):
      ``should_emit_signal`` writes ``updated_at_utc=now`` BEFORE the IBKR
      adapter is consulted. When the adapter returns an unconfirmed/duplicate
      status (e.g. ``duplicate_blocked``, ``duplicate_open_order``,
      ``suppressed_open_orders_cap``, ``rejected``, ``error``), the 10-minute
      cooldown was armed for a submission that never reached the broker,
      producing a self-perpetuating "duplicate_blocked → cooldown_active for
      10 min → duplicate_blocked" loop.

    This function deletes the entry just written so the cooldown is not
    consumed by an unconfirmed result. Returns True when an entry was removed
    (used for log-line distinction).
    """
    fp = fingerprint_signal(signal)
    state = _load_state()
    key = fp.key()
    if key in state:
        del state[key]
        _save_state(state)
        return True
    return False


def reset_all_signals() -> None:
    _save_state({})

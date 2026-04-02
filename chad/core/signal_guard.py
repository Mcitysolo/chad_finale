#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict
import json

STATE_PATH = Path("/home/ubuntu/chad_finale/runtime/signal_guard.json")
COOLDOWN_MINUTES = 10


@dataclass(frozen=True)
class SignalFingerprint:
    strategy: str
    symbol: str
    side: str
    size: float

    def key(self) -> str:
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
                    return False
            except Exception:
                return False

    # store/update signal
    state[key] = {
        "active": True,
        "updated_at_utc": _utc_now_iso(),
        "strategy": fp.strategy,
        "symbol": fp.symbol,
        "side": fp.side,
        "size": fp.size,
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
        _save_state(state)


def reset_all_signals() -> None:
    _save_state({})

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final


RUNTIME_DIR: Final[Path] = Path("/home/ubuntu/chad_finale/runtime")
STOP_STATE_PATH: Final[Path] = RUNTIME_DIR / "stop_state.json"


@dataclass(frozen=True)
class StopState:
    """
    Authoritative STOP state.

    STOP is stronger than DRY_RUN/LIVE intent: it is an emergency freeze.

    Contract:
      - stop=True  => DENY_ALL (no live, no paper, no exits/cancels)
      - stop=False => normal evaluation continues

    This file is safe to be missing; missing/invalid => stop=False (safe default).
    """
    stop: bool
    reason: str
    updated_at_utc: str


def _default() -> StopState:
    return StopState(stop=False, reason="default_off", updated_at_utc="")


def load_stop_state() -> StopState:
    try:
        if not STOP_STATE_PATH.exists():
            return _default()
        raw = STOP_STATE_PATH.read_text(encoding="utf-8")
        data: Dict[str, Any] = json.loads(raw)
        stop = bool(data.get("stop", False))
        reason = str(data.get("reason", "unknown"))
        updated = str(data.get("updated_at_utc", ""))
        return StopState(stop=stop, reason=reason, updated_at_utc=updated)
    except Exception:
        # Fail closed toward "not stopped" to avoid bricking the system due to JSON corruption.
        # (STOP activation must be an explicit operator action.)
        return _default()


def save_stop_state(state: StopState) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stop": bool(state.stop),
        "reason": state.reason,
        "updated_at_utc": state.updated_at_utc,
    }
    tmp = STOP_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, STOP_STATE_PATH)

"""
CHAD Live Mode State Helper (Phase 8 – Go Live / Stop Framework)

This module provides a single, safe interface for reading and writing
CHAD's live-mode switch, stored in:

    /home/ubuntu/chad_finale/runtime/live_mode.json

The file controls whether CHAD *intends* to operate in live mode.
It does NOT, by itself, enable live trading. All of the following must
still pass before any real order is allowed in future phases:

    - ExecutionConfig allows live (not in this Phase 7 build)
    - SCR (Shadow Confidence Router) approves
    - Live Gate approves
    - Caps and risk checks approve
    - Global STOP flag is not set

In this Phase 7 / early Phase 8 framework, this module is used to:

    - Persist operator intent (live vs stop)
    - Show current live state via CLI
    - Provide a clean API for other components (LiveGate, API gateway)

The effective live behaviour remains DRY_RUN until Phase 8 wiring is complete.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final

# Base paths are known and fixed on this box.
RUNTIME_DIR: Final[Path] = Path("/home/ubuntu/chad_finale/runtime")
LIVE_MODE_PATH: Final[Path] = RUNTIME_DIR / "live_mode.json"


@dataclass(frozen=True)
class LiveModeState:
    """
    Immutable representation of CHAD's live-mode state.

    Attributes
    ----------
    live:
        True  -> operator intends that live trading may be allowed
        False -> operator intends that system must remain in DRY_RUN

        NOTE: In this Phase 7 build, even if live=True, adapters and
        execution remain hard-locked to DRY_RUN. This flag is strictly
        preparatory until Phase 8 wiring is complete.

    reason:
        Short human-readable explanation of why this state is set
        (e.g. "initial_default", "operator_stop", "go_live_approved").
    """

    live: bool
    reason: str


def _default_state() -> LiveModeState:
    """
    Default live-mode state if the file is missing or invalid.

    Returns
    -------
    LiveModeState
        A state representing STOP / DRY_RUN.
    """
    return LiveModeState(live=False, reason="initial_default")


def load_live_mode() -> LiveModeState:
    """
    Load the live-mode state from LIVE_MODE_PATH.

    If the file does not exist, or is invalid, a safe default is returned.
    Errors are not propagated; callers always receive a LiveModeState.

    Returns
    -------
    LiveModeState
        The current live-mode state, or a safe default.
    """
    try:
        if not LIVE_MODE_PATH.exists():
            return _default_state()

        raw = LIVE_MODE_PATH.read_text(encoding="utf-8")
        data: Dict[str, Any] = json.loads(raw)

        live = bool(data.get("live", False))
        reason_raw = data.get("reason", "unknown")
        reason = str(reason_raw) if reason_raw is not None else "unknown"

        return LiveModeState(live=live, reason=reason)
    except Exception:
        # Any failure to read/parse should result in STOP / DRY_RUN.
        return _default_state()


def save_live_mode(state: LiveModeState) -> None:
    """
    Persist the given live-mode state to LIVE_MODE_PATH atomically.

    The write is done via a temporary file + rename to avoid partial
    writes in the event of a crash.

    Parameters
    ----------
    state:
        The LiveModeState to write.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "live": bool(state.live),
        "reason": state.reason,
    }

    tmp_path = LIVE_MODE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, LIVE_MODE_PATH)


def set_live_mode(live: bool, reason: str) -> LiveModeState:
    """
    Convenience helper to update live-mode state.

    Parameters
    ----------
    live:
        The desired live flag. True indicates the operator intends to
        allow live trading (subject to all other gates). False indicates
        STOP / DRY_RUN.

    reason:
        Short explanation for audit/logging.

    Returns
    -------
    LiveModeState
        The state that was written.
    """
    state = LiveModeState(live=bool(live), reason=str(reason))
    save_live_mode(state)
    return state


def main() -> int:
    """
    CLI entrypoint for inspecting the current live-mode state.

    Usage
    -----
    From the CHAD runtime root:

        PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.live_mode

    This prints the current contents of runtime/live_mode.json (or the
    safe default if it does not exist). It does NOT modify any state.
    """
    state = load_live_mode()
    print("=== CHAD Live Mode State ===")
    print(f"path   : {LIVE_MODE_PATH}")
    print(f"live   : {state.live}")
    print(f"reason : {state.reason}")
    print()
    print("Notes:")
    print("  • In this Phase 7 build, adapters and execution remain in DRY_RUN.")
    print("  • This file reflects operator intent only, preparing for Phase 8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

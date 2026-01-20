"""
chad.core.stop_state

World-class STOP state SSOT for CHAD.

Purpose
-------
STOP is the highest-authority kill switch in CHAD. When STOP is enabled:
- DENY_ALL: no live, no paper, no entries, no exits/cancels (unless you later design an
  explicit "cancel-only" carve-out; default is total freeze).

SSOT contract
-------------
This module owns the canonical runtime file:
  runtime/stop_state.json

That file MUST be:
- written atomically
- TTL-stamped (ts_utc + ttl_seconds)
- human-inspectable

Consumers (LiveGate, API surfaces, runners) must fail-closed if stop_state is missing/corrupt/stale.

CLI
---
python -m chad.core.stop_state show
python -m chad.core.stop_state set --enable --reason "panic"
python -m chad.core.stop_state set --disable --reason "resume"

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from chad.utils.runtime_json import read_runtime_state_json, write_runtime_state_json


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
STOP_PATH = RUNTIME_DIR / "stop_state.json"

# STOP should remain authoritative even if timers jitter.
# TTL means "this file is still intended to be trusted as the operator's current intent".
STOP_TTL_SECONDS_DEFAULT = 24 * 60 * 60  # 24h


@dataclass(frozen=True)
class StopState:
    stop: bool
    reason: str


def _normalize_reason(reason: Optional[str]) -> str:
    r = (reason or "").strip()
    if not r:
        return "unspecified"
    # Keep it bounded for logs/UI
    return r[:240]


def load_stop_state() -> StopState:
    """
    Load STOP state with freshness check.

    Fail-closed policy (consumer side) should treat stale/missing as STOP=True.
    This loader returns the raw state + freshness metadata separately.
    """
    obj, fr = read_runtime_state_json(STOP_PATH)

    # If missing/corrupt/stale, we return stop=True and an explicit reason.
    if obj is None or not fr.ok:
        reason = f"stop_state_{fr.reason}"
        return StopState(stop=True, reason=reason)

    stop = bool(obj.get("stop", False))
    reason = _normalize_reason(obj.get("reason"))
    return StopState(stop=stop, reason=reason)


def write_stop_state(*, stop: bool, reason: str, ttl_seconds: int = STOP_TTL_SECONDS_DEFAULT) -> Dict[str, Any]:
    """
    Write STOP state as TTL-stamped runtime SSOT.
    """
    state = {
        "stop": bool(stop),
        "reason": _normalize_reason(reason),
    }
    return write_runtime_state_json(STOP_PATH, state, ttl_seconds=int(ttl_seconds), inject_ts=True)


def _cmd_show() -> int:
    st = load_stop_state()
    print(json.dumps({"stop": st.stop, "reason": st.reason}, sort_keys=True))
    return 0


def _cmd_set(enable: bool, reason: str, ttl_seconds: int) -> int:
    write_stop_state(stop=enable, reason=reason, ttl_seconds=ttl_seconds)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CHAD STOP state SSOT (runtime/stop_state.json)")
    sub = p.add_subparsers(dest="cmd", required=True)

    show = sub.add_parser("show", help="Show current STOP state (fail-closed on stale/missing).")

    setp = sub.add_parser("set", help="Set STOP state.")
    g = setp.add_mutually_exclusive_group(required=True)
    g.add_argument("--enable", action="store_true", help="Enable STOP (DENY_ALL).")
    g.add_argument("--disable", action="store_true", help="Disable STOP.")
    setp.add_argument("--reason", required=True, help="Human reason (required).")
    setp.add_argument("--ttl-seconds", type=int, default=STOP_TTL_SECONDS_DEFAULT, help="TTL for stop_state.json (default 24h).")

    args = p.parse_args()

    if args.cmd == "show":
        return _cmd_show()

    if args.cmd == "set":
        enable = bool(args.enable)
        return _cmd_set(enable=enable, reason=str(args.reason), ttl_seconds=int(args.ttl_seconds))

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

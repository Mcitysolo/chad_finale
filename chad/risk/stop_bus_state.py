"""
chad/risk/stop_bus_state.py

Phase-8 Session 4 (R2 wiring): persistent STOP-bus state + helpers.

stop_bus_triggers.py (Session 3) provides pure-function halt detectors.
This module binds them to a small on-disk state file and exposes three
callables:

    is_stop_bus_active()  -> bool
    set_stop_bus(reason, triggered_by=...)
    clear_stop_bus(cleared_by=...)

State lives at runtime/stop_bus.json with this schema:

    {
      "active":       bool,
      "reason":       str,                 # concatenated trigger reasons
      "triggered_at": ISO8601 UTC | "",
      "triggered_by": str,                 # module/component that fired
      "cleared_at":   ISO8601 UTC | null,
      "cleared_by":   str | ""
    }

Write is atomic (tmp + rename). Read is tolerant to a missing/malformed
file (returns active=False — fail-open so a broken bus file never halts
trading silently).

The evaluate_and_persist() helper composes everything: it runs the four
Session-3 triggers against a supplied snapshot, and when any trigger is
active writes the bus file and returns a structured result. Callers
(live_loop) act on the returned flag.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from chad.risk.stop_bus_triggers import evaluate_all_stop_triggers

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
STOP_BUS_PATH = ROOT / "runtime" / "stop_bus.json"

STOP_BUS_SCHEMA_VERSION = "stop_bus.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_stop_bus(path: Path = STOP_BUS_PATH) -> Dict[str, Any]:
    """Return the current stop_bus record, or an inactive default if missing.

    Never raises — a missing or unreadable file means 'not active' so a
    busted bus file cannot silently keep trading halted forever.
    """
    if not path.is_file():
        return {
            "schema_version": STOP_BUS_SCHEMA_VERSION,
            "active": False,
            "reason": "",
            "triggered_at": "",
            "triggered_by": "",
            "cleared_at": None,
            "cleared_by": "",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": STOP_BUS_SCHEMA_VERSION,
            "active": False,
            "reason": "unreadable_state_file",
            "triggered_at": "",
            "triggered_by": "",
            "cleared_at": None,
            "cleared_by": "",
        }
    if not isinstance(data, dict):
        return {
            "schema_version": STOP_BUS_SCHEMA_VERSION,
            "active": False,
            "reason": "malformed_state_file",
            "triggered_at": "",
            "triggered_by": "",
            "cleared_at": None,
            "cleared_by": "",
        }
    data.setdefault("schema_version", STOP_BUS_SCHEMA_VERSION)
    data.setdefault("active", False)
    data.setdefault("reason", "")
    data.setdefault("triggered_at", "")
    data.setdefault("triggered_by", "")
    data.setdefault("cleared_at", None)
    data.setdefault("cleared_by", "")
    return data


def is_stop_bus_active(path: Path = STOP_BUS_PATH) -> bool:
    """Return True iff the persisted bus record has active=true."""
    return bool(read_stop_bus(path).get("active", False))


def set_stop_bus(
    reason: str,
    triggered_by: str = "live_loop",
    path: Path = STOP_BUS_PATH,
) -> Dict[str, Any]:
    """Write a STOP record. Idempotent re-trigger preserves triggered_at."""
    current = read_stop_bus(path)
    triggered_at = current.get("triggered_at") or _utc_now_iso()
    # If the bus is being freshly triggered (was cleared or missing),
    # reset the triggered_at timestamp.
    if not current.get("active", False):
        triggered_at = _utc_now_iso()

    payload = {
        "schema_version": STOP_BUS_SCHEMA_VERSION,
        "active": True,
        "reason": str(reason or ""),
        "triggered_at": triggered_at,
        "triggered_by": str(triggered_by or ""),
        "cleared_at": None,
        "cleared_by": "",
    }
    _write_atomic(path, payload)
    LOG.warning(
        "STOP_BUS_TRIGGERED reason=%s triggered_by=%s at=%s",
        payload["reason"], payload["triggered_by"], payload["triggered_at"],
    )
    return payload


def clear_stop_bus(
    cleared_by: str = "operator",
    path: Path = STOP_BUS_PATH,
) -> Dict[str, Any]:
    """Clear the STOP bus. Safe to call even when already inactive."""
    payload = {
        "schema_version": STOP_BUS_SCHEMA_VERSION,
        "active": False,
        "reason": "",
        "triggered_at": "",
        "triggered_by": "",
        "cleared_at": _utc_now_iso(),
        "cleared_by": str(cleared_by or ""),
    }
    _write_atomic(path, payload)
    LOG.warning(
        "STOP_BUS_CLEARED cleared_by=%s at=%s",
        payload["cleared_by"], payload["cleared_at"],
    )
    return payload


def evaluate_and_persist(
    snapshot: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    triggered_by: str = "live_loop",
    path: Path = STOP_BUS_PATH,
) -> Dict[str, Any]:
    """Evaluate Session-3 triggers against snapshot; persist if any active.

    Returns a summary dict:
        {
          "any_active": bool,
          "reason": str,                  # '' when no trigger active
          "active_triggers": [names],
          "bus_state": <current stop_bus record>,
        }

    When any_active becomes True the bus file is written. When no trigger
    is active the file is left untouched — callers that want to clear the
    bus on recovery should call clear_stop_bus() explicitly.
    """
    result = evaluate_all_stop_triggers(snapshot=snapshot, config=config)
    active = result.get("active_triggers", [])
    any_active = bool(result.get("any_active", False))

    if any_active:
        reason = "; ".join(result.get("reasons", [])) or "triggered"
        bus_state = set_stop_bus(reason=reason, triggered_by=triggered_by, path=path)
    else:
        bus_state = read_stop_bus(path)

    return {
        "any_active": any_active,
        "reason": bus_state.get("reason", ""),
        "active_triggers": [r.name for r in active],
        "bus_state": bus_state,
    }


__all__ = [
    "STOP_BUS_PATH",
    "STOP_BUS_SCHEMA_VERSION",
    "read_stop_bus",
    "is_stop_bus_active",
    "set_stop_bus",
    "clear_stop_bus",
    "evaluate_and_persist",
]

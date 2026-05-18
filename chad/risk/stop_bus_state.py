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
STOP_BUS_RECOVERY_STATE_PATH = ROOT / "runtime" / "stop_bus_recovery_state.json"

STOP_BUS_SCHEMA_VERSION = "stop_bus.v1"
STOP_BUS_RECOVERY_SCHEMA_VERSION = "stop_bus_recovery_state.v1"

# GAP-034 / Phase-42 auto-recovery defaults.
DEFAULT_AUTO_CLEAR_ENABLED: bool = True
DEFAULT_AUTO_CLEAR_CONSECUTIVE_CLEAN_REQUIRED: int = 5
DEFAULT_AUTO_CLEAR_MIN_CLEAN_SECONDS: int = 240
DEFAULT_AUTO_CLEAR_TTL_SECONDS: int = 600


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_dt() -> datetime:
    # Separate from _utc_now_iso so tests can monkeypatch only the auto-
    # recovery clock without touching set_stop_bus / clear_stop_bus.
    return datetime.now(timezone.utc)


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


# ---------------------------------------------------------------------------
# GAP-034 / Phase-42: durable stop-bus auto-clear (clean-streak hysteresis)
# ---------------------------------------------------------------------------


def _read_recovery_state(
    path: Path = STOP_BUS_RECOVERY_STATE_PATH,
) -> Dict[str, Any]:
    """Return the persisted auto-recovery counter state, tolerant of missing/malformed files."""
    default = {
        "schema_version": STOP_BUS_RECOVERY_SCHEMA_VERSION,
        "consecutive_clean_evaluations": 0,
        "last_eval_ts": "",
        "last_clean_metric_ts": "",
        "ttl_seconds": DEFAULT_AUTO_CLEAR_TTL_SECONDS,
    }
    if not path.is_file():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(default)
    except (OSError, json.JSONDecodeError):
        return dict(default)
    for k, v in default.items():
        data.setdefault(k, v)
    return data


def _write_recovery_state(
    counter: int,
    last_eval_ts: str,
    last_clean_metric_ts: str,
    path: Path = STOP_BUS_RECOVERY_STATE_PATH,
    ttl_seconds: int = DEFAULT_AUTO_CLEAR_TTL_SECONDS,
) -> Dict[str, Any]:
    payload = {
        "schema_version": STOP_BUS_RECOVERY_SCHEMA_VERSION,
        "consecutive_clean_evaluations": int(counter),
        "last_eval_ts": str(last_eval_ts or ""),
        "last_clean_metric_ts": str(last_clean_metric_ts or ""),
        "ttl_seconds": int(ttl_seconds),
    }
    _write_atomic(path, payload)
    return payload


def _reset_recovery_counter_if_nonzero(
    path: Path = STOP_BUS_RECOVERY_STATE_PATH,
) -> None:
    """Idempotent: write counter=0 only if a prior non-zero counter exists."""
    if not path.is_file():
        return
    rec = _read_recovery_state(path)
    if int(rec.get("consecutive_clean_evaluations", 0) or 0) == 0:
        return
    _write_recovery_state(
        counter=0,
        last_eval_ts=_utc_now_iso(),
        last_clean_metric_ts="",
        path=path,
    )


def _maybe_auto_clear_on_clean_streak(
    current_state: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    path: Path = STOP_BUS_PATH,
    recovery_state_path: Path = STOP_BUS_RECOVERY_STATE_PATH,
) -> Optional[Dict[str, Any]]:
    """Increment the clean-streak counter and, when both the count and the
    elapsed-time threshold are met, atomically clear the stop bus with a
    ``cleared_by`` label that begins with ``auto_recovery:``.

    Returns the cleared payload when auto-clear fires; otherwise None.

    Invoked ONLY from ``evaluate_and_persist`` when ``any_active`` is False
    AND the bus is currently latched active. The set_stop_bus path and the
    operator-manual clear path are independent of this helper.
    """
    cfg = dict(config or {})
    enabled = bool(cfg.get("auto_clear_enabled", DEFAULT_AUTO_CLEAR_ENABLED))
    if not enabled:
        return None
    if not bool(current_state.get("active", False)):
        return None

    required = int(cfg.get(
        "auto_clear_consecutive_clean_required",
        DEFAULT_AUTO_CLEAR_CONSECUTIVE_CLEAN_REQUIRED,
    ))
    min_seconds = float(cfg.get(
        "auto_clear_min_clean_seconds",
        DEFAULT_AUTO_CLEAR_MIN_CLEAN_SECONDS,
    ))

    now_dt = _utc_now_dt()
    now_iso = now_dt.isoformat()

    rec = _read_recovery_state(recovery_state_path)
    prior_counter = int(rec.get("consecutive_clean_evaluations", 0) or 0)
    prior_streak_start = rec.get("last_clean_metric_ts") or ""

    if prior_counter <= 0 or not prior_streak_start:
        new_counter = 1
        streak_start_iso = now_iso
    else:
        new_counter = prior_counter + 1
        streak_start_iso = prior_streak_start

    _write_recovery_state(
        counter=new_counter,
        last_eval_ts=now_iso,
        last_clean_metric_ts=streak_start_iso,
        path=recovery_state_path,
    )

    if new_counter < required:
        return None

    try:
        streak_start_dt = datetime.fromisoformat(streak_start_iso)
        elapsed = (now_dt - streak_start_dt).total_seconds()
    except (TypeError, ValueError):
        elapsed = float("inf")

    if elapsed < min_seconds:
        return None

    cleared = clear_stop_bus(
        cleared_by=f"auto_recovery:broker_latency_clean_streak={required}",
        path=path,
    )
    _write_recovery_state(
        counter=0,
        last_eval_ts=now_iso,
        last_clean_metric_ts="",
        path=recovery_state_path,
    )
    LOG.warning(
        "STOP_BUS_AUTO_CLEARED counter=%d elapsed=%.1fs required=%d min_seconds=%.1f",
        new_counter, elapsed, required, min_seconds,
    )
    return cleared


def evaluate_and_persist(
    snapshot: Mapping[str, Any],
    config: Optional[Mapping[str, Any]] = None,
    triggered_by: str = "live_loop",
    path: Path = STOP_BUS_PATH,
    recovery_state_path: Optional[Path] = None,
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
    is active, the GAP-034 / Phase-42 hysteresis helper may auto-clear a
    latched bus once enough consecutive clean cycles AND the
    ``auto_clear_min_clean_seconds`` elapsed-time threshold have both been
    satisfied. Operator-manual clear via ``clear_stop_bus()`` is immediate
    and independent of the counter.
    """
    if recovery_state_path is None:
        if path == STOP_BUS_PATH:
            recovery_state_path = STOP_BUS_RECOVERY_STATE_PATH
        else:
            recovery_state_path = path.parent / "stop_bus_recovery_state.json"

    result = evaluate_all_stop_triggers(snapshot=snapshot, config=config)
    active = result.get("active_triggers", [])
    any_active = bool(result.get("any_active", False))

    if any_active:
        reason = "; ".join(result.get("reasons", [])) or "triggered"
        bus_state = set_stop_bus(reason=reason, triggered_by=triggered_by, path=path)
        _reset_recovery_counter_if_nonzero(recovery_state_path)
    else:
        bus_state = read_stop_bus(path)
        if bool(bus_state.get("active", False)):
            cleared = _maybe_auto_clear_on_clean_streak(
                current_state=bus_state,
                snapshot=snapshot,
                config=config,
                path=path,
                recovery_state_path=recovery_state_path,
            )
            if cleared is not None:
                bus_state = cleared
        else:
            _reset_recovery_counter_if_nonzero(recovery_state_path)

    return {
        "any_active": any_active,
        "reason": bus_state.get("reason", ""),
        "active_triggers": [r.name for r in active],
        "bus_state": bus_state,
    }


__all__ = [
    "STOP_BUS_PATH",
    "STOP_BUS_RECOVERY_STATE_PATH",
    "STOP_BUS_SCHEMA_VERSION",
    "STOP_BUS_RECOVERY_SCHEMA_VERSION",
    "DEFAULT_AUTO_CLEAR_ENABLED",
    "DEFAULT_AUTO_CLEAR_CONSECUTIVE_CLEAN_REQUIRED",
    "DEFAULT_AUTO_CLEAR_MIN_CLEAN_SECONDS",
    "DEFAULT_AUTO_CLEAR_TTL_SECONDS",
    "read_stop_bus",
    "is_stop_bus_active",
    "set_stop_bus",
    "clear_stop_bus",
    "evaluate_and_persist",
]

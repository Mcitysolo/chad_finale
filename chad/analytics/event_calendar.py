"""
chad/analytics/event_calendar.py

Phase-8 Session 7 (S5): event-risk suppression calendar.

Reads ``config/event_calendar.json`` and tells the routing layer
whether a trade should be suppressed, reduced, or rejected because a
scheduled macro catalyst is close. The calendar is operator-maintained
— the module is a read-only consumer.

Contract
--------

``EventCalendar.get_suppression(symbol, urgency, now)`` returns::

    {
        "suppress": bool,           # True iff we're inside the window
        "action":   str,            # "reject" | "reduce_50pct" | "pass"
        "reason":   str,            # human-readable
        "hours_to_event": float | None,
        "event_name":     str,
    }

Resolution rules:

  * ``action == "reject"`` — fire when intent urgency is "high". The
    routing-gate caller drops the intent entirely.
  * ``action == "reduce_50pct"`` — fire when urgency is "normal".
    The routing-gate caller halves intent.size in place and returns
    True (the intent still ships, just smaller).
  * ``action == "pass"`` — no event close enough to act on.

Edge cases
----------

  * Missing config → everything passes.
  * Malformed entry → entry is skipped; remaining entries still count.
  * Past events (beyond their suppression window on the far side) are
    ignored — a catalyst 25 hours after `now` with a 24h window passes.
  * Symbol-scoped events (type == "earnings" with a symbol key) only
    suppress that symbol. Macro events suppress all symbols.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALENDAR_PATH = ROOT / "config" / "event_calendar.json"

DEFAULT_SUPPRESS_HOURS: float = 24.0

ACTION_REJECT = "reject"
ACTION_REDUCE = "reduce_50pct"
ACTION_PASS = "pass"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_event_time(date_s: str, time_s: str = "") -> Optional[datetime]:
    """Combine a YYYY-MM-DD date and optional HH:MM time into a UTC datetime.

    Missing time defaults to 00:00 UTC. Malformed strings return None.
    """
    if not date_s:
        return None
    try:
        d = datetime.strptime(str(date_s)[:10], "%Y-%m-%d")
    except ValueError:
        return None
    if time_s:
        try:
            t = datetime.strptime(str(time_s)[:5], "%H:%M").time()
            d = datetime.combine(d.date(), t)
        except ValueError:
            pass
    return d.replace(tzinfo=timezone.utc)


def _load_events(path: Path = DEFAULT_CALENDAR_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    events = data.get("events")
    if not isinstance(events, list):
        return []
    out: List[Dict[str, Any]] = []
    for raw in events:
        if not isinstance(raw, dict):
            continue
        dt = _parse_event_time(str(raw.get("date", "")), str(raw.get("time", "")))
        if dt is None:
            continue
        out.append(
            {
                "when": dt,
                "name": str(raw.get("name") or "unnamed_event"),
                "type": str(raw.get("type") or "macro"),
                "symbol": str(raw.get("symbol") or "").upper() or None,
                "suppress_hours": float(raw.get("suppress_hours") or DEFAULT_SUPPRESS_HOURS),
            }
        )
    return out


class EventCalendar:
    """Read-only view of scheduled catalysts with a suppression lookup."""

    def __init__(
        self,
        calendar_path: Path = DEFAULT_CALENDAR_PATH,
        default_suppress_hours: float = DEFAULT_SUPPRESS_HOURS,
    ) -> None:
        self._path = Path(calendar_path)
        self._default_suppress_hours = float(max(0.0, default_suppress_hours))
        self._events: Optional[List[Dict[str, Any]]] = None

    def reload(self) -> None:
        self._events = _load_events(self._path)

    def _all_events(self) -> List[Dict[str, Any]]:
        if self._events is None:
            self._events = _load_events(self._path)
        return self._events

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_nearest_upcoming(
        self,
        symbol: str = "",
        now: Optional[datetime] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Return (event, hours_until) for the soonest applicable upcoming event.

        "Applicable" means macro events always apply; symbol-scoped events
        apply only when their symbol matches. Past events are skipped.
        """
        current = now if now is not None else _utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        sym = str(symbol or "").upper()

        best: Optional[Dict[str, Any]] = None
        best_hours: Optional[float] = None
        for ev in self._all_events():
            if ev["symbol"] and sym and ev["symbol"] != sym:
                continue
            delta = (ev["when"] - current).total_seconds() / 3600.0
            if delta < 0:
                continue
            if best_hours is None or delta < best_hours:
                best_hours = delta
                best = ev
        return best, best_hours

    def get_suppression(
        self,
        symbol: str = "",
        urgency: str = "normal",
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Return the suppression verdict for the given intent context."""
        event, hours = self.get_nearest_upcoming(symbol=symbol, now=now)
        if event is None or hours is None:
            return {
                "suppress": False,
                "action": ACTION_PASS,
                "reason": "no_upcoming_events",
                "hours_to_event": None,
                "event_name": "",
            }
        window = float(event.get("suppress_hours") or self._default_suppress_hours)
        if hours > window:
            return {
                "suppress": False,
                "action": ACTION_PASS,
                "reason": "outside_window",
                "hours_to_event": round(hours, 2),
                "event_name": event["name"],
            }
        u = (urgency or "normal").strip().lower()
        if u == "high":
            return {
                "suppress": True,
                "action": ACTION_REJECT,
                "reason": f"event_risk:{event['name']}_in_{hours:.1f}h",
                "hours_to_event": round(hours, 2),
                "event_name": event["name"],
            }
        return {
            "suppress": True,
            "action": ACTION_REDUCE,
            "reason": f"event_risk:{event['name']}_in_{hours:.1f}h",
            "hours_to_event": round(hours, 2),
            "event_name": event["name"],
        }


__all__ = [
    "ACTION_PASS",
    "ACTION_REDUCE",
    "ACTION_REJECT",
    "DEFAULT_CALENDAR_PATH",
    "DEFAULT_SUPPRESS_HOURS",
    "EventCalendar",
]

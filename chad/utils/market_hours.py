"""CHAD shared market-hours utility — US-equity regular-session (RTH) open check.

Single source of truth for the question "is the US equity regular session open
right now?", shared by the dashboard (:mod:`chad.dashboard.api`) and the tier
manager (:mod:`chad.risk.tier_manager`).

This module is intentionally:
  * pure — no broker, adapter, request, or global-state dependencies
  * stdlib-only — operates on a timezone-aware ``datetime`` expressed in UTC
  * test-injectable — the caller supplies ``now_utc``

RTH window (matches the historical dashboard helper exactly):
  a weekday (Mon-Fri) AND ``14:30 <= UTC-time-of-day < 21:00``.

That is the standard US equity regular session (09:30-16:00 America/New_York)
expressed in UTC anchored to Eastern Standard Time.  It deliberately does NOT
model daylight saving or exchange holidays — this preserves the prior dashboard
behaviour and is sufficient for the tier demotion-deferral gate, where the safe
default is to *under*-report "open" (a missed deferral merely applies a demotion
slightly early off-hours, never mid-session).
"""
from __future__ import annotations

from datetime import datetime

# US equity regular session in minutes-since-midnight UTC, anchored to Eastern
# Standard Time: 09:30 ET = 14:30 UTC (open); 16:00 ET = 21:00 UTC (close).
RTH_OPEN_MINUTE_UTC: int = 14 * 60 + 30
RTH_CLOSE_MINUTE_UTC: int = 21 * 60


def market_is_open(now_utc: datetime) -> bool:
    """Return ``True`` if the US equity regular session is open at ``now_utc``.

    ``now_utc`` is interpreted as UTC wall-clock (its weekday and time-of-day);
    weekends are always closed.  Pure and deterministic — performs no I/O and
    holds no state.
    """
    if now_utc.weekday() >= 5:
        return False
    minutes = now_utc.hour * 60 + now_utc.minute
    return RTH_OPEN_MINUTE_UTC <= minutes < RTH_CLOSE_MINUTE_UTC

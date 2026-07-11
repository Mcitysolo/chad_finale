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

from datetime import datetime, time as _dtime, timezone
from zoneinfo import ZoneInfo

# US equity regular session in minutes-since-midnight UTC, anchored to Eastern
# Standard Time: 09:30 ET = 14:30 UTC (open); 16:00 ET = 21:00 UTC (close).
RTH_OPEN_MINUTE_UTC: int = 14 * 60 + 30
RTH_CLOSE_MINUTE_UTC: int = 21 * 60


def market_is_open(now_utc: datetime) -> bool:
    """Return ``True`` if the US equity regular session is open at ``now_utc``.

    ``now_utc`` is interpreted as UTC wall-clock (its weekday and time-of-day);
    weekends are always closed.  Pure and deterministic — performs no I/O and
    holds no state.

    NOTE: this fixed-offset helper does NOT model daylight saving (it assumes
    EST year-round) and exists for the tier-manager demotion-deferral gate,
    where under-reporting "open" is the safe default. For the execution-path
    RTH gate (WKF-U2), which must be accurate in the current EDT regime, use
    :func:`equity_rth_is_open` below.
    """
    if now_utc.weekday() >= 5:
        return False
    minutes = now_utc.hour * 60 + now_utc.minute
    return RTH_OPEN_MINUTE_UTC <= minutes < RTH_CLOSE_MINUTE_UTC


# US equity regular session in America/New_York local time (DST-aware).
_NY_TZ = ZoneInfo("America/New_York")
_RTH_LOCAL_OPEN: _dtime = _dtime(9, 30)   # 09:30 ET
_RTH_LOCAL_CLOSE: _dtime = _dtime(16, 0)  # 16:00 ET


def equity_rth_is_open(now_utc: datetime) -> bool:
    """Return ``True`` iff the US equity regular session is open at ``now_utc``.

    DST-aware: computes the session in ``America/New_York`` local time
    (09:30-16:00 ET on a weekday). In the current EDT regime this is
    13:30-20:00 UTC exactly as specified by WKF-U2; in EST it is 14:30-21:00
    UTC. Unlike :func:`market_is_open`, it is correct across daylight-saving
    transitions, which is why the execution-path RTH gate uses it.

    ``now_utc`` may be timezone-aware or naive; a naive value is interpreted as
    UTC. Pure and deterministic — no I/O, no state.

    TODO(WKF-U2-HALFDAY): exchange holidays and early-close (half-day) sessions
    are NOT modeled — a half-day still reads as open until 16:00 ET. A named
    follow-up will add an exchange calendar.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local = now_utc.astimezone(_NY_TZ)
    if local.weekday() >= 5:
        return False
    return _RTH_LOCAL_OPEN <= local.timetz().replace(tzinfo=None) < _RTH_LOCAL_CLOSE

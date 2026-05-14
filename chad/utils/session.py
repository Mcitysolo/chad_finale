"""CHAD shared session utility — tier-aware time-of-day session zones.

Canonical America/New_York session model, audited against
``chad.strategies.alpha_intraday_micro`` (the existing source of truth).

This module is intentionally:
  * pure — no broker, adapter, execution, or strategy imports
  * stdlib-only — ``zoneinfo.ZoneInfo``, never fixed UTC offsets
  * test-injectable — every entry point accepts a ``now`` override

Strategies consume :func:`session_decision` to gate **entry** signals only.
Exits, position reductions, and stop-loss emissions must not consult this
module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Constants — mirror chad/strategies/alpha_intraday_micro_config.py exactly.
# ---------------------------------------------------------------------------

SESSION_TIMEZONE: str = "America/New_York"

PRIMARY_SESSION_START: str = "09:35"
PRIMARY_SESSION_END: str = "11:00"

SECONDARY_SESSION_START: str = "13:30"
SECONDARY_SESSION_END: str = "15:00"

HARD_EOD_EXIT_TIME: str = "15:30"

# Deterministic skip reason codes.
SKIP_OUTSIDE_PRIMARY_WINDOW: str = "SKIP_OUTSIDE_PRIMARY_WINDOW"
SKIP_OUTSIDE_SESSION_WINDOW: str = "SKIP_OUTSIDE_SESSION_WINDOW"
SKIP_EOD_FLATTEN_WINDOW: str = "SKIP_EOD_FLATTEN_WINDOW"

# Public session-window labels.
SESSION_PRIMARY: str = "PRIMARY"
SESSION_SECONDARY: str = "SECONDARY"


@dataclass(frozen=True)
class SessionDecision:
    """Outcome of a session-zone evaluation for a single instant in time."""

    session_window: Optional[str]
    is_primary: bool
    is_secondary: bool
    is_eod_flatten_window: bool
    entry_allowed: bool
    skip_reason: Optional[str]


def parse_hhmm(value: str) -> time:
    """Parse an ``"HH:MM"`` string into a naive ``datetime.time``.

    Raises ``ValueError`` for malformed input.
    """
    if not isinstance(value, str):
        raise ValueError(f"parse_hhmm: expected str, got {type(value).__name__}")
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"parse_hhmm: expected 'HH:MM', got {value!r}")
    hh_s, mm_s = parts
    hh = int(hh_s)
    mm = int(mm_s)
    if not (0 <= hh <= 23) or not (0 <= mm <= 59):
        raise ValueError(f"parse_hhmm: out-of-range value {value!r}")
    return time(hour=hh, minute=mm)


def now_local(
    now: Optional[datetime] = None,
    timezone_name: str = SESSION_TIMEZONE,
) -> datetime:
    """Return ``now`` projected into ``timezone_name``.

    Naive datetimes are interpreted as UTC, never as the host timezone.
    """
    tz = ZoneInfo(timezone_name)
    if now is None:
        return datetime.now(timezone.utc).astimezone(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(tz)


def session_decision(
    now: Optional[datetime] = None,
    *,
    timezone_name: str = SESSION_TIMEZONE,
    primary_start: str = PRIMARY_SESSION_START,
    primary_end: str = PRIMARY_SESSION_END,
    secondary_start: str = SECONDARY_SESSION_START,
    secondary_end: str = SECONDARY_SESSION_END,
    hard_eod_exit_time: str = HARD_EOD_EXIT_TIME,
    primary_session_only: bool = False,
) -> SessionDecision:
    """Resolve session classification + entry permission for one instant.

    The function is the single chokepoint for tier-aware entry gating
    across CHAD strategies; it never inspects positions, fills, or
    intent. Callers must apply the result to **entry** logic only.
    """
    local_dt = now_local(now, timezone_name=timezone_name)
    local_t = local_dt.time()

    p_start = parse_hhmm(primary_start)
    p_end = parse_hhmm(primary_end)
    s_start = parse_hhmm(secondary_start)
    s_end = parse_hhmm(secondary_end)
    eod_t = parse_hhmm(hard_eod_exit_time)

    is_primary = p_start <= local_t < p_end
    is_secondary = s_start <= local_t < s_end
    is_eod_flatten = local_t >= eod_t

    if is_primary:
        window: Optional[str] = SESSION_PRIMARY
    elif is_secondary:
        window = SESSION_SECONDARY
    else:
        window = None

    if is_eod_flatten:
        return SessionDecision(
            session_window=window,
            is_primary=is_primary,
            is_secondary=is_secondary,
            is_eod_flatten_window=True,
            entry_allowed=False,
            skip_reason=SKIP_EOD_FLATTEN_WINDOW,
        )

    if primary_session_only:
        if is_primary:
            return SessionDecision(
                session_window=window,
                is_primary=True,
                is_secondary=False,
                is_eod_flatten_window=False,
                entry_allowed=True,
                skip_reason=None,
            )
        return SessionDecision(
            session_window=window,
            is_primary=False,
            is_secondary=is_secondary,
            is_eod_flatten_window=False,
            entry_allowed=False,
            skip_reason=SKIP_OUTSIDE_PRIMARY_WINDOW,
        )

    if is_primary or is_secondary:
        return SessionDecision(
            session_window=window,
            is_primary=is_primary,
            is_secondary=is_secondary,
            is_eod_flatten_window=False,
            entry_allowed=True,
            skip_reason=None,
        )

    return SessionDecision(
        session_window=None,
        is_primary=False,
        is_secondary=False,
        is_eod_flatten_window=False,
        entry_allowed=False,
        skip_reason=SKIP_OUTSIDE_SESSION_WINDOW,
    )


__all__ = [
    "SESSION_TIMEZONE",
    "PRIMARY_SESSION_START",
    "PRIMARY_SESSION_END",
    "SECONDARY_SESSION_START",
    "SECONDARY_SESSION_END",
    "HARD_EOD_EXIT_TIME",
    "SKIP_OUTSIDE_PRIMARY_WINDOW",
    "SKIP_OUTSIDE_SESSION_WINDOW",
    "SKIP_EOD_FLATTEN_WINDOW",
    "SESSION_PRIMARY",
    "SESSION_SECONDARY",
    "SessionDecision",
    "parse_hhmm",
    "now_local",
    "session_decision",
]

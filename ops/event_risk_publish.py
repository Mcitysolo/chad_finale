#!/usr/bin/env python3
"""
CHAD Market Radar — event_risk publisher (Production)

File: ops/event_risk_publish.py

Outputs:
  /home/ubuntu/chad_finale/runtime/event_risk.json

Purpose
-------
Publish a fail-closed, schema-locked snapshot of upcoming *event risk* windows
that can tighten execution or force caution elsewhere in the system.

This is an advisory radar only. It places no trades and touches no brokers.

Design Guarantees
-----------------
- Always writes a SINGLE valid JSON object (no concatenation, no literal '\\n')
- Atomic write with fsync(file) + fsync(dir)
- Fail-closed: on any error, writes severity="unknown"
- Deterministic structure (schema_version pinned)
- TTL enforced
- Provider abstraction for future upgrades (economic calendars, paid feeds)

Bootstrap Strategy (no external APIs)
-------------------------------------
Until you wire real calendars (CPI/FOMC/NFP):
- Use market-hours derived “risk windows”
- Label severity conservatively
- Emit placeholders with explicit notes

This keeps SSOT contracts alive without lying.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

# -----------------------------
# Paths / Config
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
OUT_PATH = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))) / "event_risk.json"

DEFAULT_CONFIG_DIR = Path("/home/ubuntu/chad_finale/config")
EVENT_CALENDAR_PATH = Path(
    os.environ.get("CHAD_EVENT_CALENDAR_PATH", str(DEFAULT_CONFIG_DIR / "event_calendar.json"))
)

TTL_SECONDS = int(os.environ.get("CHAD_EVENT_RISK_TTL_SECONDS", "1800"))  # 30 min
CACHE_RESPECT_TTL = os.environ.get("CHAD_EVENT_RISK_RESPECT_TTL", "1").lower() in ("1", "true", "yes", "on")

# How far ahead to project events when computing elevated_risk
LOOKAHEAD_HOURS = int(os.environ.get("CHAD_EVENT_RISK_LOOKAHEAD_HOURS", "48"))
# How far ahead to project events for the rule-based schedule
SCHEDULE_HORIZON_DAYS = int(os.environ.get("CHAD_EVENT_RISK_HORIZON_DAYS", "120"))


# -----------------------------
# Models
# -----------------------------

@dataclass(frozen=True)
class RiskWindow:
    label: str
    start_utc: str
    end_utc: str
    severity: str   # low | medium | high | unknown
    notes: str


@dataclass(frozen=True)
class EventRiskState:
    ts_utc: str
    ttl_seconds: int
    severity: str
    windows: List[RiskWindow]
    notes: str
    source: Dict[str, Any]
    elevated_risk: bool = False
    risk_score: float = 0.0
    risk_description: str = ""
    next_event: Optional[Dict[str, Any]] = None
    schema_version: str = "event_risk.v1"


class EventRiskProvider(Protocol):
    def build(self) -> EventRiskState: ...


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def read_existing_if_fresh(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        ts = obj.get("ts_utc")
        ttl = int(obj.get("ttl_seconds", 0))
        if not ts or ttl <= 0:
            return None

        y, mo, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
        hh, mm, ss = int(ts[11:13]), int(ts[14:16]), int(ts[17:19])
        epoch = int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone

        if int(time.time()) <= epoch + ttl:
            return obj
        return None
    except Exception:
        return None


# -----------------------------
# Provider (bootstrap)
# -----------------------------

class MarketHoursRiskProvider:
    """
    Conservative placeholder provider.
    Treats market open / close windows as elevated operational risk.
    """

    def build(self) -> EventRiskState:
        ts = utc_now_iso()

        # US equity market windows (UTC approximation)
        today = time.strftime("%Y-%m-%d", time.gmtime())
        windows = [
            RiskWindow(
                label="US_MARKET_OPEN",
                start_utc=f"{today}T13:30:00Z",
                end_utc=f"{today}T14:00:00Z",
                severity="medium",
                notes="market open volatility window",
            ),
            RiskWindow(
                label="US_MARKET_CLOSE",
                start_utc=f"{today}T20:00:00Z",
                end_utc=f"{today}T20:30:00Z",
                severity="medium",
                notes="market close liquidity window",
            ),
        ]

        return EventRiskState(
            ts_utc=ts,
            ttl_seconds=TTL_SECONDS,
            severity="medium",
            windows=windows,
            notes="bootstrap_provider=market_hours; replace with CPI/FOMC/NFP calendar",
            source={"provider": "MarketHoursRiskProvider"},
        )


# -----------------------------
# Economic-calendar provider (real)
# -----------------------------

# Per-event severity / risk-score table. Keys must match generated event names.
EVENT_RISK_TABLE: Dict[str, Tuple[str, float]] = {
    "FOMC Rate Decision": ("high", 0.9),
    "CPI Release": ("high", 0.8),
    "Non-Farm Payrolls": ("high", 0.8),
    "GDP Release": ("medium", 0.6),
}

# Hardcoded FOMC announcement dates (Fed publishes a year ahead). Encoded as
# ISO date + announcement time (UTC). These are second-day announcements at
# 18:00 UTC (14:00 ET). Update annually as Fed publishes new calendars.
FOMC_DATES_UTC: List[str] = [
    "2026-01-28T18:00:00Z",
    "2026-03-18T18:00:00Z",
    "2026-04-29T18:00:00Z",
    "2026-06-17T18:00:00Z",
    "2026-07-29T18:00:00Z",
    "2026-09-16T18:00:00Z",
    "2026-10-28T18:00:00Z",
    "2026-12-09T18:00:00Z",
    "2027-01-27T18:00:00Z",
    "2027-03-17T18:00:00Z",
]


def _parse_iso_to_utc(s: str) -> Optional[_dt.datetime]:
    """Parse an ISO-ish string ('YYYY-MM-DDTHH:MM:SSZ' or 'YYYY-MM-DD HH:MM') to UTC datetime."""
    if not s:
        return None
    try:
        s2 = s.strip()
        if s2.endswith("Z"):
            s2 = s2[:-1] + "+00:00"
        return _dt.datetime.fromisoformat(s2).astimezone(_dt.timezone.utc)
    except Exception:
        try:
            return _dt.datetime.strptime(s.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
        except Exception:
            return None


def _first_friday_of_month(year: int, month: int) -> _dt.date:
    d = _dt.date(year, month, 1)
    # weekday(): Mon=0..Sun=6 ; Friday=4
    return d + _dt.timedelta(days=(4 - d.weekday()) % 7)


def _last_day_of_month(year: int, month: int) -> _dt.date:
    if month == 12:
        nxt = _dt.date(year + 1, 1, 1)
    else:
        nxt = _dt.date(year, month + 1, 1)
    return nxt - _dt.timedelta(days=1)


def _generate_rule_based_events(
    *, now_utc: _dt.datetime, horizon_days: int
) -> List[Dict[str, Any]]:
    """
    Generate FOMC / CPI / NFP / GDP events based on hardcoded patterns.
    Each event is a dict {name, ts_utc (iso), source}.
    """
    horizon = now_utc + _dt.timedelta(days=horizon_days)
    out: List[Dict[str, Any]] = []

    # FOMC: take from FOMC_DATES_UTC list within window
    for s in FOMC_DATES_UTC:
        ts = _parse_iso_to_utc(s)
        if ts is None:
            continue
        if now_utc - _dt.timedelta(days=1) <= ts <= horizon:
            out.append({"name": "FOMC Rate Decision", "ts_utc": s, "source": "rule:fomc"})

    # Walk forward month-by-month for CPI, NFP, GDP
    cur_date = now_utc.date().replace(day=1)
    end_date = horizon.date()
    while cur_date <= end_date:
        y, m = cur_date.year, cur_date.month

        # CPI: ~15th of each month at 12:30 UTC (08:30 ET)
        cpi_d = _dt.date(y, m, 15)
        cpi_ts = _dt.datetime(cpi_d.year, cpi_d.month, cpi_d.day, 12, 30, tzinfo=_dt.timezone.utc)
        if now_utc - _dt.timedelta(hours=2) <= cpi_ts <= horizon:
            out.append({
                "name": "CPI Release",
                "ts_utc": cpi_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "rule:cpi_15th",
            })

        # NFP: first Friday of each month at 12:30 UTC
        nfp_d = _first_friday_of_month(y, m)
        nfp_ts = _dt.datetime(nfp_d.year, nfp_d.month, nfp_d.day, 12, 30, tzinfo=_dt.timezone.utc)
        if now_utc - _dt.timedelta(hours=2) <= nfp_ts <= horizon:
            out.append({
                "name": "Non-Farm Payrolls",
                "ts_utc": nfp_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": "rule:nfp_first_friday",
            })

        # GDP: last day of month following quarter end (Apr/Jul/Oct/Jan) at 12:30 UTC
        if m in (1, 4, 7, 10):
            gdp_d = _last_day_of_month(y, m)
            gdp_ts = _dt.datetime(gdp_d.year, gdp_d.month, gdp_d.day, 12, 30, tzinfo=_dt.timezone.utc)
            if now_utc - _dt.timedelta(hours=2) <= gdp_ts <= horizon:
                out.append({
                    "name": "GDP Release",
                    "ts_utc": gdp_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": "rule:gdp_last_day_post_quarter",
                })

        # advance one month
        if m == 12:
            cur_date = _dt.date(y + 1, 1, 1)
        else:
            cur_date = _dt.date(y, m + 1, 1)

    return out


def _read_operator_calendar(path: Path) -> List[Dict[str, Any]]:
    """Read config/event_calendar.json. Never raises — returns [] on failure."""
    try:
        if not path.is_file():
            return []
        obj = json.loads(path.read_text(encoding="utf-8"))
        events = obj.get("events") if isinstance(obj, dict) else None
        if not isinstance(events, list):
            return []
        out: List[Dict[str, Any]] = []
        for e in events:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            date_s = str(e.get("date") or "").strip()
            time_s = str(e.get("time") or "00:00").strip()
            if not name or not date_s:
                continue
            ts_iso = f"{date_s}T{time_s}:00Z" if len(time_s) == 5 else f"{date_s}T{time_s}Z"
            out.append({
                "name": name,
                "ts_utc": ts_iso,
                "source": "operator_calendar",
                "type": str(e.get("type") or "macro"),
                "suppress_hours": int(e.get("suppress_hours") or 0),
            })
        return out
    except Exception:
        return []


def _merge_events(
    rule_events: List[Dict[str, Any]],
    operator_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge: operator entries take priority over rule-based entries when they
    refer to the same event (matched by name + same calendar day).
    """
    op_keys = set()
    for e in operator_events:
        ts = _parse_iso_to_utc(e["ts_utc"])
        if ts is None:
            continue
        op_keys.add((e["name"], ts.date()))

    merged: List[Dict[str, Any]] = list(operator_events)
    for e in rule_events:
        ts = _parse_iso_to_utc(e["ts_utc"])
        if ts is None:
            continue
        if (e["name"], ts.date()) in op_keys:
            continue  # operator entry wins
        merged.append(e)

    # sort by ts_utc
    def _key(ev: Dict[str, Any]) -> _dt.datetime:
        t = _parse_iso_to_utc(ev["ts_utc"])
        return t or _dt.datetime.max.replace(tzinfo=_dt.timezone.utc)

    merged.sort(key=_key)
    return merged


def _event_severity_score(name: str) -> Tuple[str, float]:
    return EVENT_RISK_TABLE.get(name, ("medium", 0.5))


class EconomicCalendarRiskProvider:
    """
    Real economic-calendar provider.

    Combines:
      - rule-based generation (hardcoded FOMC schedule + CPI/NFP/GDP patterns)
      - operator-maintained config/event_calendar.json (operator wins on
        same name+day collisions)

    Sets elevated_risk + risk_score based on whether any event falls within
    LOOKAHEAD_HOURS. Each upcoming event is emitted as a RiskWindow.
    """

    def __init__(
        self,
        *,
        operator_calendar_path: Path = EVENT_CALENDAR_PATH,
        lookahead_hours: int = LOOKAHEAD_HOURS,
        horizon_days: int = SCHEDULE_HORIZON_DAYS,
    ) -> None:
        self.operator_calendar_path = operator_calendar_path
        self.lookahead_hours = lookahead_hours
        self.horizon_days = horizon_days

    def build(self) -> EventRiskState:
        ts = utc_now_iso()
        now_utc = _dt.datetime.now(_dt.timezone.utc)

        rule_events = _generate_rule_based_events(
            now_utc=now_utc, horizon_days=self.horizon_days
        )
        operator_events = _read_operator_calendar(self.operator_calendar_path)
        merged = _merge_events(rule_events, operator_events)

        # Build RiskWindow list
        windows: List[RiskWindow] = []
        elevated = False
        max_score = 0.0
        elevated_desc = ""
        next_event: Optional[Dict[str, Any]] = None

        lookahead_end = now_utc + _dt.timedelta(hours=self.lookahead_hours)

        for ev in merged:
            ev_ts = _parse_iso_to_utc(ev["ts_utc"])
            if ev_ts is None:
                continue

            sev, score = _event_severity_score(ev["name"])

            # Window: 1 hour before to 1 hour after the event
            start = ev_ts - _dt.timedelta(hours=1)
            end = ev_ts + _dt.timedelta(hours=1)
            windows.append(
                RiskWindow(
                    label=ev["name"],
                    start_utc=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end_utc=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    severity=sev,
                    notes=f"source={ev.get('source', 'unknown')}",
                )
            )

            # Track next future event (first by sorted order)
            if next_event is None and ev_ts >= now_utc:
                next_event = {
                    "name": ev["name"],
                    "ts_utc": ev["ts_utc"],
                    "source": ev.get("source"),
                    "hours_until": round((ev_ts - now_utc).total_seconds() / 3600.0, 2),
                }

            # Elevated risk if this event lies inside lookahead window
            if now_utc <= ev_ts <= lookahead_end:
                if score > max_score:
                    max_score = score
                    elevated_desc = ev["name"]
                elevated = True

        overall_severity = "high" if max_score >= 0.8 else ("medium" if max_score >= 0.5 else "low")
        if not elevated:
            overall_severity = "low"

        return EventRiskState(
            ts_utc=ts,
            ttl_seconds=TTL_SECONDS,
            severity=overall_severity,
            windows=windows,
            elevated_risk=elevated,
            risk_score=max_score,
            risk_description=elevated_desc,
            next_event=next_event,
            notes=(
                f"provider=economic_calendar; rule_events={len(rule_events)}; "
                f"operator_events={len(operator_events)}; merged={len(windows)}"
            ),
            source={
                "provider": "EconomicCalendarRiskProvider",
                "operator_calendar_path": str(self.operator_calendar_path),
                "operator_calendar_present": self.operator_calendar_path.is_file(),
                "lookahead_hours": self.lookahead_hours,
                "horizon_days": self.horizon_days,
            },
        )


# -----------------------------
# Orchestration
# -----------------------------

def state_to_dict(s: EventRiskState) -> Dict[str, Any]:
    return {
        "ts_utc": s.ts_utc,
        "ttl_seconds": s.ttl_seconds,
        "severity": s.severity,
        "elevated_risk": bool(s.elevated_risk),
        "risk_score": float(s.risk_score),
        "risk_description": s.risk_description,
        "next_event": s.next_event,
        "windows": [w.__dict__ for w in s.windows],
        "notes": s.notes,
        "source": s.source,
        "schema_version": s.schema_version,
    }


def fail_closed_state(ts: str, reason: str) -> Dict[str, Any]:
    return {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "severity": "unknown",
        "windows": [],
        "notes": reason,
        "source": {"provider": "error"},
        "schema_version": "event_risk.v1",
    }


def main() -> int:
    ts = utc_now_iso()

    if CACHE_RESPECT_TTL:
        cached = read_existing_if_fresh(OUT_PATH)
        if cached is not None:
            print(json.dumps({"ok": True, "cached": True, "out": str(OUT_PATH), "ts_utc": cached.get("ts_utc")}, sort_keys=True))
            return 0

    # Primary: real economic calendar; fallback: market-hours stub.
    try:
        provider: EventRiskProvider = EconomicCalendarRiskProvider()
        state = provider.build()
        payload = state_to_dict(state)
        atomic_write_json(OUT_PATH, payload)
        print(json.dumps({
            "ok": True,
            "out": str(OUT_PATH),
            "ts_utc": state.ts_utc,
            "provider": "EconomicCalendarRiskProvider",
            "elevated_risk": state.elevated_risk,
            "risk_score": state.risk_score,
        }, sort_keys=True))
        return 0
    except Exception as primary_exc:
        try:
            provider = MarketHoursRiskProvider()
            state = provider.build()
            payload = state_to_dict(state)
            payload["notes"] = (
                f"primary_failed:{type(primary_exc).__name__}; fallback=market_hours"
            )
            atomic_write_json(OUT_PATH, payload)
            print(json.dumps({
                "ok": False,
                "out": str(OUT_PATH),
                "ts_utc": state.ts_utc,
                "provider": "MarketHoursRiskProvider(fallback)",
                "primary_error": str(primary_exc),
            }, sort_keys=True))
            return 0
        except Exception as exc:
            payload = fail_closed_state(ts, f"publish_error:{type(exc).__name__}")
            atomic_write_json(OUT_PATH, payload)
            print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "error": str(exc)}, sort_keys=True))
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

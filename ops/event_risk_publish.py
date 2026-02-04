#!/usr/bin/env python3
"""
CHAD Market Radar — event_risk publisher (Production)

File: ops/event_risk_publish.py

Outputs:
  /home/ubuntu/CHAD FINALE/runtime/event_risk.json

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

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

# -----------------------------
# Paths / Config
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/CHAD FINALE/runtime")
OUT_PATH = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))) / "event_risk.json"

TTL_SECONDS = int(os.environ.get("CHAD_EVENT_RISK_TTL_SECONDS", "1800"))  # 30 min
CACHE_RESPECT_TTL = os.environ.get("CHAD_EVENT_RISK_RESPECT_TTL", "1").lower() in ("1", "true", "yes", "on")


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
# Orchestration
# -----------------------------

def state_to_dict(s: EventRiskState) -> Dict[str, Any]:
    return {
        "ts_utc": s.ts_utc,
        "ttl_seconds": s.ttl_seconds,
        "severity": s.severity,
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

    try:
        provider: EventRiskProvider = MarketHoursRiskProvider()
        state = provider.build()
        payload = state_to_dict(state)
        atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": True, "out": str(OUT_PATH), "ts_utc": state.ts_utc}, sort_keys=True))
        return 0
    except Exception as exc:
        payload = fail_closed_state(ts, f"publish_error:{type(exc).__name__}")
        atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "error": str(exc)}, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

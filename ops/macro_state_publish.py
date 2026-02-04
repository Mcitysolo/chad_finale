#!/usr/bin/env python3
"""
CHAD Market Radar — macro_state publisher (Production)

File: ops/macro_state_publish.py

Outputs:
  /home/ubuntu/CHAD FINALE/runtime/macro_state.json

SSOT contract (v4.2):
  runtime/macro_state.json must exist, be valid JSON, include ts_utc + ttl_seconds,
  and be safe to read while the system runs (atomic write). :contentReference[oaicite:1]{index=1}

Design Goals
------------
- Always writes VALID single JSON object (never concatenated, never literal '\\n')
- Atomic write: tmp -> fsync(file) -> rename -> fsync(dir)
- Fail-closed: if provider fetch fails, write risk_label="unknown"
- Provider interface: swap data source without touching publisher logic
- Async fetch: concurrent sensor retrieval (fast, resilient)
- Cache-aware: if existing macro_state.json is fresh, optionally no-op
- systemd-safe: no interactive prompts, deterministic exit behavior

No broker calls. No secrets are written to runtime.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import urllib.request


# -----------------------------
# Constants / Paths
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/CHAD FINALE/runtime")
OUT_PATH = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))) / "macro_state.json"

# TTL (seconds). SSOT suggests 1800 as a sane baseline for macro. :contentReference[oaicite:2]{index=2}
TTL_SECONDS = int(os.environ.get("CHAD_MACRO_TTL_SECONDS", "1800"))

# If true, publisher will skip network work if current file is fresh.
CACHE_RESPECT_TTL = os.environ.get("CHAD_MACRO_RESPECT_TTL", "1").strip().lower() in ("1", "true", "yes", "on")

# Network controls
HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "6.0"))
HTTP_UA = os.environ.get("CHAD_HTTP_USER_AGENT", "chad-macro-state/2.0")

# FRED public CSV endpoints (no key). If FRED changes policy, we fail closed.
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class MacroState:
    ts_utc: str
    ttl_seconds: int
    risk_label: str  # risk_on | risk_off | neutral | unknown
    yields: Dict[str, float]  # optional keys: us_2y, us_10y
    curve_slope_10y_2y: float
    notes: str
    source: Dict[str, Any]
    schema_version: str = "macro_state.v1"


class MacroProvider(Protocol):
    async def fetch(self) -> MacroState: ...


# -----------------------------
# Helpers (safe)
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return v
    except Exception:
        return 0.0


def sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def canonical_json_bytes(obj: Dict[str, Any]) -> bytes:
    # Deterministic JSON for hashing + stable diffs
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> str:
    """
    Atomic JSON write (crash-safe):
      tmp -> fsync(file) -> os.replace -> fsync(dir)
    Returns sha256 over canonical json bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(obj)
    digest = sha256_hex(data)

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)

    # best-effort directory fsync for durability
    try:
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

    return digest


def read_existing_if_fresh(path: Path) -> Optional[Dict[str, Any]]:
    """
    If file exists and ts_utc + ttl_seconds indicate it is still fresh, return it.
    Otherwise return None. Never raises.
    """
    try:
        if not path.is_file():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        ts = str(obj.get("ts_utc") or "")
        ttl = int(obj.get("ttl_seconds") or 0)
        if not ts or ttl <= 0:
            return None

        # parse ts_utc (YYYY-MM-DDTHH:MM:SSZ)
        # This is stable format we produce; fail-safe on parse errors.
        try:
            # manual parse to avoid dateutil dependency
            y = int(ts[0:4])
            mo = int(ts[5:7])
            d = int(ts[8:10])
            hh = int(ts[11:13])
            mm = int(ts[14:16])
            ss = int(ts[17:19])
            # convert to epoch (UTC) using time.gmtime-like tuple
            epoch = int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone
        except Exception:
            return None

        now = int(time.time())
        if now <= epoch + ttl:
            return obj
        return None
    except Exception:
        return None


def _fred_parse_last_value(csv_text: str) -> Tuple[bool, float]:
    """
    Parse the last valid numeric row from FRED csv.
    Expected header: DATE,VALUE
    """
    lines = csv_text.splitlines()
    for line in reversed(lines):
        if not line or line.startswith("DATE"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        v = parts[1].strip()
        if v in ("", "."):
            continue
        return True, safe_float(v)
    return False, 0.0


async def _http_get_text(url: str) -> str:
    """
    Async wrapper around urllib.request (threaded) to avoid adding new deps.
    """
    def _blocking() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": HTTP_UA})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8", errors="replace")

    return await asyncio.to_thread(_blocking)


# -----------------------------
# Provider implementations
# -----------------------------

class FredYieldProvider:
    """
    Bootstrap macro provider using public FRED CSV for yields.
    - 2y: DGS2
    - 10y: DGS10

    Produces a deterministic risk_label heuristic based on curve slope.
    """

    async def fetch(self) -> MacroState:
        ts = utc_now_iso()

        url2 = FRED_CSV.format(series="DGS2")
        url10 = FRED_CSV.format(series="DGS10")

        t2, t10 = await asyncio.gather(
            _http_get_text(url2),
            _http_get_text(url10),
        )

        ok2, y2 = _fred_parse_last_value(t2)
        ok10, y10 = _fred_parse_last_value(t10)

        yields: Dict[str, float] = {}
        if ok2:
            yields["us_2y"] = float(y2)
        if ok10:
            yields["us_10y"] = float(y10)

        slope = 0.0
        risk = "unknown"
        if ok2 and ok10:
            slope = float(y10 - y2)
            if slope < -0.25:
                risk = "risk_off"
            elif slope > 0.25:
                risk = "risk_on"
            else:
                risk = "neutral"

        return MacroState(
            ts_utc=ts,
            ttl_seconds=TTL_SECONDS,
            risk_label=risk,
            yields=yields,
            curve_slope_10y_2y=slope,
            notes="bootstrap_provider=fed_fred_csv; summary only; no secrets",
            source={"provider": "FredYieldProvider", "series": ["DGS2", "DGS10"]},
        )


# -----------------------------
# Publisher / Orchestration
# -----------------------------

def state_to_dict(s: MacroState) -> Dict[str, Any]:
    return {
        "ts_utc": s.ts_utc,
        "ttl_seconds": int(s.ttl_seconds),
        "yields": dict(s.yields),
        "curve_slope_10y_2y": float(s.curve_slope_10y_2y),
        "risk_label": s.risk_label,
        "notes": s.notes,
        "source": dict(s.source),
        "schema_version": s.schema_version,
    }


async def build_state(provider: MacroProvider) -> MacroState:
    return await provider.fetch()


def fail_closed_state(ts: str, *, reason: str) -> Dict[str, Any]:
    return {
        "ts_utc": ts,
        "ttl_seconds": int(TTL_SECONDS),
        "yields": {},
        "curve_slope_10y_2y": 0.0,
        "risk_label": "unknown",
        "notes": reason,
        "source": {"provider": "error"},
        "schema_version": "macro_state.v1",
    }


async def main_async() -> int:
    ts = utc_now_iso()

    if CACHE_RESPECT_TTL:
        existing = read_existing_if_fresh(OUT_PATH)
        if existing is not None:
            # No-op, but still exit 0 (healthy).
            print(json.dumps({"ok": True, "out": str(OUT_PATH), "ts_utc": existing.get("ts_utc"), "cached": True}, sort_keys=True))
            return 0

    provider: MacroProvider = FredYieldProvider()

    try:
        st = await build_state(provider)
        payload = state_to_dict(st)
        digest = atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": True, "out": str(OUT_PATH), "ts_utc": st.ts_utc, "sha256": digest, "cached": False}, sort_keys=True))
        return 0
    except Exception as exc:
        payload = fail_closed_state(ts, reason=f"publish_error:{type(exc).__name__}")
        digest = atomic_write_json(OUT_PATH, payload)
        # Fail-soft (exit 0) so timer doesn't become a restart storm. Runtime file still indicates unknown.
        print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "sha256": digest, "error": str(exc)}, sort_keys=True))
        return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())


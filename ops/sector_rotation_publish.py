#!/usr/bin/env python3
"""
CHAD Market Radar — sector_rotation publisher (Production, bootstrap)

File: ops/sector_rotation_publish.py

Outputs:
  /home/ubuntu/CHAD FINALE/runtime/sector_rotation.json

SSOT contract (v4.2):
- runtime/sector_rotation.json exists, includes ts_utc + ttl_seconds, safe to read while running (atomic write)
- advisory: suggests relative strength / tilts (never direct execution) :contentReference[oaicite:0]{index=0}

Bootstrap Rule (no lying)
-------------------------
If you don't yet have sector relative strength data feeds:
- emit a valid file with:
  - sectors list + placeholders
  - ranks=[] (explicitly empty)
  - notes explain bootstrap status
- Do not invent rankings.

No broker calls. No secrets. Advisory-only.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# Paths / Config
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/CHAD FINALE/runtime")
RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))

OUT_PATH = RUNTIME_DIR / "sector_rotation.json"
TTL_SECONDS = int(os.environ.get("CHAD_SECTOR_ROTATION_TTL_SECONDS", "3600"))  # 1 hour
CACHE_RESPECT_TTL = os.environ.get("CHAD_SECTOR_ROTATION_RESPECT_TTL", "1").lower() in ("1", "true", "yes", "on")

# Canonical sector set (US equities) as placeholders
SECTORS = [
    {"code": "XLK", "name": "Technology"},
    {"code": "XLF", "name": "Financials"},
    {"code": "XLV", "name": "Health Care"},
    {"code": "XLY", "name": "Consumer Discretionary"},
    {"code": "XLP", "name": "Consumer Staples"},
    {"code": "XLE", "name": "Energy"},
    {"code": "XLI", "name": "Industrials"},
    {"code": "XLB", "name": "Materials"},
    {"code": "XLU", "name": "Utilities"},
    {"code": "XLRE", "name": "Real Estate"},
    {"code": "XLC", "name": "Communication Services"},
]


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


def read_existing_if_fresh(path: Path) -> Optional[Dict[str, Any]]:
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

        y, mo, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
        hh, mm, ss = int(ts[11:13]), int(ts[14:16]), int(ts[17:19])
        epoch = int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone

        if int(time.time()) <= epoch + ttl:
            return obj
        return None
    except Exception:
        return None


# -----------------------------
# Publisher
# -----------------------------

def build_bootstrap_state(ts: str) -> Dict[str, Any]:
    return {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "schema_version": "sector_rotation.v1",
        "provider_status": "bootstrap_no_provider",
        "sectors": SECTORS,
        "ranks": [],  # explicit: we are not inventing relative strength rankings
        "tilt_suggestions": [],  # explicit: none until provider exists
        "notes": "bootstrap publisher: no sector RS provider wired yet; ranks intentionally empty",
        "source": {"provider": "bootstrap"},
    }


def main() -> int:
    ts = utc_now_iso()

    if CACHE_RESPECT_TTL:
        cached = read_existing_if_fresh(OUT_PATH)
        if cached is not None:
            print(json.dumps({"ok": True, "cached": True, "out": str(OUT_PATH), "ts_utc": cached.get("ts_utc")}, sort_keys=True))
            return 0

    try:
        payload = build_bootstrap_state(ts)
        atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": True, "cached": False, "out": str(OUT_PATH), "ts_utc": ts, "sectors_count": len(SECTORS)}, sort_keys=True))
        return 0
    except Exception as exc:
        payload = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "schema_version": "sector_rotation.v1",
            "provider_status": "error",
            "sectors": SECTORS,
            "ranks": [],
            "tilt_suggestions": [],
            "notes": f"publish_error:{type(exc).__name__}",
            "source": {"provider": "error"},
        }
        atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "error": str(exc)}, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

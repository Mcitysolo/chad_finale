#!/usr/bin/env python3
"""
CHAD Market Radar — earnings_state publisher (Production, bootstrap)

File: ops/earnings_state_publish.py

Outputs:
  /home/ubuntu/CHAD FINALE/runtime/earnings_state.json

SSOT contract (v4.2):
- runtime/earnings_state.json exists, includes ts_utc + ttl_seconds, is safe to read while running (atomic write).
- Used to tighten policy around earnings windows. :contentReference[oaicite:0]{index=0}

Bootstrap Rule (no lying)
-------------------------
If you do not have a real earnings calendar provider wired, this publisher MUST:
- still write a valid file
- include the symbols it considered (from universe/positions)
- emit events=[] and notes explaining provider is not yet installed

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

OUT_PATH = RUNTIME_DIR / "earnings_state.json"
TTL_SECONDS = int(os.environ.get("CHAD_EARNINGS_TTL_SECONDS", "3600"))  # 1 hour default
CACHE_RESPECT_TTL = os.environ.get("CHAD_EARNINGS_RESPECT_TTL", "1").lower() in ("1", "true", "yes", "on")

UNIVERSE_PATH = RUNTIME_DIR / "universe.json"
POSITIONS_PATH = RUNTIME_DIR / "positions_snapshot.json"


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

    # best-effort dir fsync
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


def _load_universe_symbols() -> Tuple[List[str], str]:
    """
    Returns (symbols, source_note). Never raises.
    """
    try:
        if not UNIVERSE_PATH.is_file():
            return [], "universe_missing"
        obj = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return [], "universe_invalid_shape"
        syms = obj.get("symbols") or []
        if not isinstance(syms, list):
            return [], "universe_symbols_not_list"
        out = []
        for s in syms:
            ss = str(s).strip().upper()
            if ss:
                out.append(ss)
        return out, "universe"
    except Exception:
        return [], "universe_error"


def _load_position_symbols() -> Tuple[List[str], str]:
    """
    Returns (symbols, source_note). Never raises.
    """
    try:
        if not POSITIONS_PATH.is_file():
            return [], "positions_missing"
        obj = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return [], "positions_invalid_shape"
        by = obj.get("positions_by_conid") or {}
        if not isinstance(by, dict):
            return [], "positions_by_conid_invalid"
        out = []
        for v in by.values():
            if not isinstance(v, dict):
                continue
            ss = str(v.get("symbol") or "").strip().upper()
            if ss:
                out.append(ss)
        return sorted(set(out)), "positions_snapshot"
    except Exception:
        return [], "positions_error"


def _choose_symbols(max_symbols: int = 50) -> Tuple[List[str], Dict[str, Any]]:
    """
    Priority:
      1) universe.json symbols (if present)
      2) positions_snapshot symbols
    """
    uni, uni_src = _load_universe_symbols()
    pos, pos_src = _load_position_symbols()

    chosen: List[str] = []
    source: Dict[str, Any] = {
        "universe": {"count": len(uni), "source": uni_src, "path": str(UNIVERSE_PATH)},
        "positions": {"count": len(pos), "source": pos_src, "path": str(POSITIONS_PATH)},
    }

    if uni:
        chosen = uni[:max_symbols]
        source["selected"] = "universe"
    elif pos:
        chosen = pos[:max_symbols]
        source["selected"] = "positions"
    else:
        chosen = []
        source["selected"] = "none"

    return chosen, source


# -----------------------------
# Publisher
# -----------------------------

def build_bootstrap_state(ts: str) -> Dict[str, Any]:
    syms, src = _choose_symbols(max_symbols=50)
    return {
        "ts_utc": ts,
        "ttl_seconds": TTL_SECONDS,
        "schema_version": "earnings_state.v1",
        "provider_status": "bootstrap_no_provider",
        "symbols_considered": syms,
        "events": [],  # explicit: we are not inventing earnings events
        "notes": "bootstrap publisher: no earnings provider wired yet; events intentionally empty",
        "source": src,
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
        print(json.dumps({"ok": True, "cached": False, "out": str(OUT_PATH), "ts_utc": ts, "symbols_count": len(payload.get("symbols_considered") or [])}, sort_keys=True))
        return 0
    except Exception as exc:
        # fail-closed payload
        payload = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "schema_version": "earnings_state.v1",
            "provider_status": "error",
            "symbols_considered": [],
            "events": [],
            "notes": f"publish_error:{type(exc).__name__}",
            "source": {"provider": "error"},
        }
        atomic_write_json(OUT_PATH, payload)
        print(json.dumps({"ok": False, "out": str(OUT_PATH), "ts_utc": ts, "error": str(exc)}, sort_keys=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

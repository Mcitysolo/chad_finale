#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from urllib.request import Request, urlopen

RUNTIME = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
OUT = RUNTIME / "scr_state.json"
SHADOW_URL = os.environ.get("CHAD_SHADOW_URL", "http://127.0.0.1:9618/shadow")
TTL_SECONDS = int(os.environ.get("CHAD_SCR_TTL_SECONDS", "180"))

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
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

def _fetch_shadow() -> Dict[str, Any]:
    req = Request(SHADOW_URL, headers={"User-Agent": "chad-scr-sync/1.1"})
    with urlopen(req, timeout=3) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or "shadow" not in obj:
        raise RuntimeError("invalid_shadow_payload")
    return obj

def main() -> int:
    ts = _utc_now_iso()
    try:
        obj = _fetch_shadow()
        shadow = obj.get("shadow") or {}
        payload: Dict[str, Any] = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "state": shadow.get("state"),
            "paper_only": shadow.get("paper_only"),
            "sizing_factor": shadow.get("sizing_factor"),
            "reasons": shadow.get("reasons") or [],
            "stats": shadow.get("stats") or {},
            "source": {"url": SHADOW_URL},
            "schema_version": "scr_state.v1",
        }
        _atomic_write_json(OUT, payload)
        print(json.dumps({"ok": True, "out": str(OUT), "ts_utc": ts}, sort_keys=True))
        return 0
    except Exception as e:
        payload = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "state": "UNKNOWN",
            "paper_only": True,
            "sizing_factor": 0.0,
            "reasons": [f"scr_sync_error:{type(e).__name__}"],
            "stats": {},
            "source": {"url": SHADOW_URL},
            "schema_version": "scr_state.v1",
        }
        _atomic_write_json(OUT, payload)
        print(json.dumps({"ok": False, "error": str(e), "out": str(OUT), "ts_utc": ts}, sort_keys=True))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

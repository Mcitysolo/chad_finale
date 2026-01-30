#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict
from urllib.request import Request, urlopen

RUNTIME = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
OUT = RUNTIME / "tier_state.json"
RISK_URL = os.environ.get("CHAD_RISK_URL", "http://127.0.0.1:9618/risk-state")
TTL_SECONDS = int(os.environ.get("CHAD_TIER_TTL_SECONDS", "300"))

# SSOT v4.2 tier ladder thresholds (account equity, USD)
# MICRO:   $100–$1,000
# STARTER: $1k–$10k
# PRO:     $10k–$50k
# SCALE:   $50k+
def _tier_for_equity(e: float) -> str:
    if e < 100.0:
        return "BELOW_MICRO"
    if e < 1_000.0:
        return "MICRO"
    if e < 10_000.0:
        return "STARTER"
    if e < 50_000.0:
        return "PRO"
    return "SCALE"

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

def _fetch_json(url: str) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "chad-tier-sync/1.0"})
    with urlopen(req, timeout=3) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("invalid_json_payload")
    return obj

def main() -> int:
    ts = _utc_now_iso()
    try:
        rs = _fetch_json(RISK_URL)
        dyn = rs.get("dynamic_caps") or {}
        total_equity = float(dyn.get("total_equity", 0.0) or 0.0)

        tier = _tier_for_equity(total_equity)

        payload: Dict[str, Any] = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "total_equity": total_equity,
            "tier": tier,
            "thresholds": {
                "micro_min": 100.0,
                "micro_max": 1000.0,
                "starter_max": 10000.0,
                "pro_max": 50000.0,
            },
            "source": {"risk_state_url": RISK_URL},
            "schema_version": "tier_state.v1",
        }
        _atomic_write_json(OUT, payload)
        print(json.dumps({"ok": True, "out": str(OUT), "tier": tier, "ts_utc": ts}, sort_keys=True))
        return 0
    except Exception as e:
        payload = {
            "ts_utc": ts,
            "ttl_seconds": TTL_SECONDS,
            "total_equity": 0.0,
            "tier": "UNKNOWN",
            "thresholds": {},
            "source": {"risk_state_url": RISK_URL},
            "schema_version": "tier_state.v1",
            "reasons": [f"tier_sync_error:{type(e).__name__}"],
        }
        _atomic_write_json(OUT, payload)
        print(json.dumps({"ok": False, "error": str(e), "out": str(OUT), "ts_utc": ts}, sort_keys=True))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

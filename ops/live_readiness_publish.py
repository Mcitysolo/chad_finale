#!/usr/bin/env python3
"""
CHAD Phase 12 — Live Readiness Publisher (Scaffolding, NO live changes)

File: ops/live_readiness_publish.py

Purpose
-------
Create a deterministic, auditable "go/no-go" artifact for live autonomy.

This script does NOT enable live trading.
It only inspects current system gates and writes:
- runtime/live_readiness.json (pointer)
- reports/live_readiness/LIVE_READINESS_<ts>.json (snapshot)

It is designed so that "ready_for_live" is only True when ALL gates are green:
- STOP is false
- CHAD_MODE == LIVE and live_enabled == true
- LiveGate allow_ibkr_live == true
- SCR paper_only == false and state != PAUSED
- feed_state and reconciliation_state are GREEN/fresh
- operator intent is not EXIT_ONLY/DENY_ALL
- (optional) bounded autonomy approvals are enabled

Right now your system is DRY_RUN + SCR PAUSED/paper_only, so this will be fail-closed.

No broker calls. No secrets.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.request import Request, urlopen

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime")).resolve()
REPORTS_DIR = Path(os.environ.get("CHAD_REPORTS_DIR", "/home/ubuntu/CHAD FINALE/reports")).resolve()
OUT_DIR = REPORTS_DIR / "live_readiness"

POINTER_PATH = RUNTIME_DIR / "live_readiness.json"

LIVE_GATE_URL = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")
STATUS_URL = os.environ.get("CHAD_STATUS_URL", "http://127.0.0.1:9618/status")

HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "4.0"))

STOP_PATH = RUNTIME_DIR / "stop_state.json"
FEED_PATH = RUNTIME_DIR / "feed_state.json"
RECON_PATH = RUNTIME_DIR / "reconciliation_state.json"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_hex(data)

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

    return digest


def read_json_dict(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_dict:{path}")
    return obj


def http_get_json(url: str) -> Dict[str, Any]:
    req = Request(url, headers={"User-Agent": "chad-live-readiness/1.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("http_json_not_dict")
    return obj


def stop_ok() -> Tuple[bool, str]:
    try:
        s = read_json_dict(STOP_PATH)
        if bool(s.get("stop", False)):
            return False, f"STOP=true reason={s.get('reason')!r}"
        return True, "STOP=false"
    except Exception as exc:
        return False, f"STOP_unreadable:{type(exc).__name__}"


def reconciliation_ok() -> Tuple[bool, str]:
    try:
        r = read_json_dict(RECON_PATH)
        status = str(r.get("status") or "").upper()
        if status != "GREEN":
            return False, f"reconciliation_status={status}"
        return True, "reconciliation=GREEN"
    except Exception as exc:
        return False, f"reconciliation_unreadable:{type(exc).__name__}"


def feed_ok() -> Tuple[bool, str]:
    try:
        f = read_json_dict(FEED_PATH)
        feeds = f.get("feeds") if isinstance(f.get("feeds"), dict) else {}
        poly = feeds.get("polygon_stocks") if isinstance(feeds.get("polygon_stocks"), dict) else None
        if not isinstance(poly, dict):
            return False, "feed_missing_polygon_stocks"
        fs = float(poly.get("freshness_seconds", 1e9))
        # If feed freshness is NaN/unparsable -> fail
        if not (fs == fs) or fs > 180:
            return False, f"feed_stale:{fs}"
        return True, f"feed_fresh:{fs}"
    except Exception as exc:
        return False, f"feed_unreadable:{type(exc).__name__}"


def main() -> int:
    ts = utc_now_iso()
    ts_compact = utc_now_compact()

    # Pull API state
    status = {}
    live_gate = {}
    try:
        status = http_get_json(STATUS_URL)
    except Exception as exc:
        status = {"error": f"status_fetch_failed:{type(exc).__name__}"}

    try:
        live_gate = http_get_json(LIVE_GATE_URL)
    except Exception as exc:
        live_gate = {"error": f"live_gate_fetch_failed:{type(exc).__name__}"}

    # Local checks
    ok_stop, stop_reason = stop_ok()
    ok_feed, feed_reason = feed_ok()
    ok_recon, recon_reason = reconciliation_ok()

    # Gate interpretation (fail-closed)
    mode = status.get("mode") if isinstance(status.get("mode"), dict) else {}
    chad_mode = str(mode.get("chad_mode") or "")
    live_enabled = bool(mode.get("live_enabled", False))

    operator_mode = str(live_gate.get("operator_mode") or "")
    allow_ibkr_live = bool(live_gate.get("allow_ibkr_live", False))

    shadow = status.get("shadow") if isinstance(status.get("shadow"), dict) else {}
    shadow_state = str(shadow.get("state") or "")
    paper_only = bool(shadow.get("paper_only", True))

    checks = {
        "stop": {"ok": ok_stop, "reason": stop_reason},
        "feed": {"ok": ok_feed, "reason": feed_reason},
        "reconciliation": {"ok": ok_recon, "reason": recon_reason},
        "chad_mode": {"ok": (chad_mode == "LIVE" and live_enabled), "reason": f"chad_mode={chad_mode} live_enabled={live_enabled}"},
        "operator_intent": {"ok": (operator_mode == "ALLOW_LIVE"), "reason": f"operator_mode={operator_mode}"},
        "scr": {"ok": (not paper_only and shadow_state != "PAUSED"), "reason": f"paper_only={paper_only} state={shadow_state}"},
        "live_gate": {"ok": bool(allow_ibkr_live), "reason": f"allow_ibkr_live={allow_ibkr_live}"},
    }

    ready = all(bool(v.get("ok")) for v in checks.values())

    payload = {
        "ts_utc": ts,
        "schema_version": "live_readiness.v1",
        "ready_for_live": bool(ready),
        "checks": checks,
        "status_snapshot": status,
        "live_gate_snapshot": live_gate,
        "notes": "scaffolding only; does not change any live settings",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"LIVE_READINESS_{ts_compact}.json"
    digest = atomic_write_json(report_path, payload)

    pointer = {
        "ts_utc": ts,
        "schema_version": "live_readiness_state.v1",
        "latest_report_path": str(report_path),
        "latest_report_sha256": f"sha256:{digest}",
        "ready_for_live": bool(ready),
    }
    atomic_write_json(POINTER_PATH, pointer)

    print(json.dumps({"ok": True, "ready_for_live": bool(ready), "report_path": str(report_path), "pointer_path": str(POINTER_PATH), "ts_utc": ts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

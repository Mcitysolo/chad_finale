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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.request import Request, urlopen

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/chad_finale/runtime")).resolve()
REPORTS_DIR = Path(os.environ.get("CHAD_REPORTS_DIR", "/home/ubuntu/chad_finale/reports")).resolve()
OUT_DIR = REPORTS_DIR / "live_readiness"

POINTER_PATH = RUNTIME_DIR / "live_readiness.json"

# Pointer cadence is weekly (next_evaluation_cadence="weekly"); declare TTL
# to match so freshness consumers (e.g. LiveGate) don't classify it as stale.
POINTER_TTL_SECONDS = 7 * 24 * 60 * 60

STATUS_URL = os.environ.get("CHAD_STATUS_URL", "http://127.0.0.1:9618/status")

HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "4.0"))

STOP_PATH = RUNTIME_DIR / "stop_state.json"
FEED_PATH = RUNTIME_DIR / "feed_state.json"
RECON_PATH = RUNTIME_DIR / "reconciliation_state.json"
TRUTH_PATH = RUNTIME_DIR / "positions_truth.json"
LIFECYCLE_PATH = RUNTIME_DIR / "trade_lifecycle_state.json"
EXECQ_PATH = RUNTIME_DIR / "execution_quality.json"
ACTION_PATH = RUNTIME_DIR / "action_state.json"
CANARY_PATH = RUNTIME_DIR / "change_canary_state.json"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_hex(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def parse_utc_iso(s: str) -> datetime:
    ss = (s or "").strip()
    if not ss:
        raise ValueError("empty_ts")
    if ss.endswith("Z"):
        ss = ss[:-1] + "+00:00"
    dt = datetime.fromisoformat(ss)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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

        # Prefer ibkr_stocks; fall back to polygon_stocks (legacy)
        feed_entry = None
        feed_label = "unknown"
        for key in ("ibkr_stocks", "polygon_stocks"):
            candidate = feeds.get(key)
            if isinstance(candidate, dict):
                feed_entry = candidate
                feed_label = key
                break

        if feed_entry is None:
            return False, "feed_missing_ibkr_stocks_and_polygon_stocks"

        fs = float(feed_entry.get("freshness_seconds", 1e9))
        # If feed freshness is NaN/unparsable -> fail
        if not (fs == fs) or fs > 180:
            return False, f"feed_stale:{feed_label}:{fs}"
        return True, f"feed_fresh:{feed_label}:{fs}"
    except Exception as exc:
        return False, f"feed_unreadable:{type(exc).__name__}"


def lifecycle_truth_ok() -> Tuple[bool, str]:
    try:
        truth = read_json_dict(TRUTH_PATH)
        lifecycle = read_json_dict(LIFECYCLE_PATH)
        if not bool(truth.get("truth_ok", False)):
            return False, f"truth_ok={truth.get('truth_ok', False)} source={truth.get('truth_source')}"
        if bool(lifecycle.get("gap_flag", True)):
            return False, "lifecycle_gap_flag=true"
        if bool(lifecycle.get("backlog_flag", True)):
            return False, "lifecycle_backlog_flag=true"
        return True, "lifecycle_truth=GREEN"
    except Exception as exc:
        return False, f"lifecycle_truth_unreadable:{type(exc).__name__}"


def execution_quality_ok() -> Tuple[bool, str]:
    try:
        eq = read_json_dict(EXECQ_PATH)
        env_label = str(eq.get("env_label") or "unknown").lower()
        if env_label in ("dangerous", "unknown"):
            return False, f"execution_env={env_label}"
        return True, f"execution_env={env_label}"
    except Exception as exc:
        return False, f"execution_quality_unreadable:{type(exc).__name__}"


def mutation_ok() -> Tuple[bool, str]:
    try:
        a = read_json_dict(ACTION_PATH)
        if not bool(a.get("ok", True)):
            return False, "action_state_ok=false"
        pending = int(a.get("pending_count", 0) or 0)
        rejected = int(a.get("rejected_count", 0) or 0)
        if pending > 0:
            return False, f"pending_count={pending}"
        if rejected > 0:
            return False, f"rejected_count={rejected}"
        return True, "mutation_state=clean"
    except Exception as exc:
        return False, f"mutation_state_unreadable:{type(exc).__name__}"


def canary_ok() -> Tuple[bool, str]:
    try:
        c = read_json_dict(CANARY_PATH)
        factor = float(c.get("canary_factor", 1.0) or 1.0)
        until = c.get("canary_until_ts_utc")
        if factor < 1.0:
            return False, f"canary_factor={factor}"
        if until:
            dt = parse_utc_iso(str(until))
            if dt > datetime.now(timezone.utc):
                return False, f"canary_until={until}"
        return True, "canary=clear"
    except Exception as exc:
        return False, f"canary_state_unreadable:{type(exc).__name__}"


def main() -> int:
    ts = utc_now_iso()
    ts_compact = utc_now_compact()

    # Pull API state
    status = {}
    try:
        status = http_get_json(STATUS_URL)
    except Exception as exc:
        status = {"error": f"status_fetch_failed:{type(exc).__name__}"}

    # Local checks
    ok_stop, stop_reason = stop_ok()
    ok_feed, feed_reason = feed_ok()
    ok_recon, recon_reason = reconciliation_ok()
    ok_truth, truth_reason = lifecycle_truth_ok()
    ok_execq, execq_reason = execution_quality_ok()
    ok_mutation, mutation_reason = mutation_ok()
    ok_canary, canary_reason = canary_ok()

    # Gate interpretation (fail-closed)
    mode = status.get("mode") if isinstance(status.get("mode"), dict) else {}
    chad_mode = str(mode.get("chad_mode") or "")
    live_enabled = bool(mode.get("live_enabled", False))

    operator_mode = "UNKNOWN"
    runtime_files = status.get("runtime_files") if isinstance(status.get("runtime_files"), dict) else {}
    op_meta = runtime_files.get("operator_intent") if isinstance(runtime_files.get("operator_intent"), dict) else {}
    op_path = op_meta.get("path")
    if op_path:
        try:
            op_obj = read_json_dict(Path(op_path))
            operator_mode = str(op_obj.get("operator_mode") or op_obj.get("mode") or "UNKNOWN")
        except Exception:
            operator_mode = "UNKNOWN"

    shadow = status.get("shadow") if isinstance(status.get("shadow"), dict) else {}
    shadow_state = str(shadow.get("state") or "")
    paper_only = bool(shadow.get("paper_only", True))

    checks = {
        "stop": {"ok": ok_stop, "reason": stop_reason},
        "feed": {"ok": ok_feed, "reason": feed_reason},
        "reconciliation": {"ok": ok_recon, "reason": recon_reason},
        "lifecycle_truth": {"ok": ok_truth, "reason": truth_reason},
        "execution_quality": {"ok": ok_execq, "reason": execq_reason},
        "mutation_state": {"ok": ok_mutation, "reason": mutation_reason},
        "canary_state": {"ok": ok_canary, "reason": canary_reason},
        "chad_mode": {"ok": (chad_mode.lower() == "live" and live_enabled), "reason": f"chad_mode={chad_mode} live_enabled={live_enabled}"},
        "operator_intent": {"ok": (operator_mode == "ALLOW_LIVE"), "reason": f"operator_mode={operator_mode}"},
        "scr": {"ok": (not paper_only and shadow_state != "PAUSED"), "reason": f"paper_only={paper_only} state={shadow_state}"},
    }

    ready = all(bool(v.get("ok")) for v in checks.values())

    payload = {
        "ts_utc": ts,
        "schema_version": "live_readiness.v1",
        "ready_for_live": bool(ready),
        "checks": checks,
        "status_snapshot": status,
        "notes": "scaffolding only; does not change any live settings",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / f"LIVE_READINESS_{ts_compact}.json"
    digest = atomic_write_json(report_path, payload)

    pointer = {
        "ts_utc": ts,
        "ttl_seconds": int(POINTER_TTL_SECONDS),
        "schema_version": "live_readiness_state.v1",
        "latest_report_path": str(report_path),
        "latest_report_sha256": f"sha256:{digest}",
        "ready_for_live": bool(ready),
    }
    atomic_write_json(POINTER_PATH, pointer)

    # Publish to Redis state bus (non-blocking, fail-soft)
    try:
        from chad.core.state_bus import get_publisher
        get_publisher().publish_live_readiness(pointer)
    except Exception:
        pass

    print(json.dumps({"ok": True, "ready_for_live": bool(ready), "report_path": str(report_path), "pointer_path": str(POINTER_PATH), "ts_utc": ts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

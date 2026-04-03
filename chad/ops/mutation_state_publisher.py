#!/usr/bin/env python3
"""
CHAD — Mutation Governance Runtime Publishers

Publishes:
- runtime/action_state.json
- runtime/change_canary_state.json

This version bridges real ActionApplier state into action_state.json so
LiveGate can read actual mutation-application history instead of bootstrap
nulls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from chad.ops.config_snapshot import create_config_snapshot, get_current_composite_hash
from chad.utils.runtime_json import write_runtime_state_json

LOG = logging.getLogger("chad.ops.mutation_state_publisher")


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def _repo_root() -> Path:
    root = str(os.environ.get("CHAD_ROOT", "")).strip()
    if root:
        p = Path(root).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parents[2]


def _runtime_dir(repo_root: Path) -> Path:
    rd = str(os.environ.get("CHAD_RUNTIME_DIR", "")).strip()
    if rd:
        return Path(rd).expanduser().resolve()
    return (repo_root / "runtime").resolve()


def _control_dir(repo_root: Path) -> Path:
    cd = str(os.environ.get("CHAD_CONTROL_DIR", "")).strip()
    if cd:
        return Path(cd).expanduser().resolve()
    return (repo_root / "control").resolve()


def _count_json_files(d: Path) -> int:
    try:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    except Exception:
        return 0


def _list_any_dir(d: Path) -> List[str]:
    try:
        if not d.is_dir():
            return []
        return sorted([p.name for p in d.iterdir() if p.is_file()])[:20]
    except Exception:
        return []


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_action_applier_state(runtime_dir: Path) -> Dict[str, Any]:
    path = runtime_dir / "action_applier_state.json"
    obj = _read_json_dict(path)
    if obj is None:
        return {
            "exists": False,
            "path": str(path),
            "last_apply_ts_utc": None,
            "last_action_id": None,
            "last_failure": None,
            "scan_ts_utc": None,
            "queue_scan": {},
            "applied_actions_count": 0,
        }

    applied_actions = obj.get("applied_actions")
    if not isinstance(applied_actions, list):
        applied_actions = []

    queue_scan = obj.get("queue_scan")
    if not isinstance(queue_scan, dict):
        queue_scan = {}

    return {
        "exists": True,
        "path": str(path),
        "last_apply_ts_utc": obj.get("last_apply_ts_utc"),
        "last_action_id": obj.get("last_action_id"),
        "last_failure": obj.get("last_failure"),
        "scan_ts_utc": obj.get("scan_ts_utc"),
        "queue_scan": queue_scan,
        "applied_actions_count": len(applied_actions),
    }


def build_action_state(*, repo_root: Path, runtime_dir: Path, control_dir: Path) -> Dict[str, Any]:
    pending_dir = control_dir / "pending_actions"
    approved_dir = control_dir / "approved_actions"
    rejected_dir = control_dir / "rejected_actions"

    pending = _count_json_files(pending_dir)
    approved = _count_json_files(approved_dir)
    rejected = _count_json_files(rejected_dir)

    applier = _load_action_applier_state(runtime_dir)

    payload: Dict[str, Any] = {
        "schema_version": "action_state.v1",
        "ok": True,
        "pending_count": int(pending),
        "approved_count": int(approved),
        "rejected_count": int(rejected),
        "last_apply_ts_utc": applier.get("last_apply_ts_utc"),
        "last_failure": applier.get("last_failure"),
        "lockout_until_ts_utc": None,
        "paths": {
            "repo_root": str(repo_root),
            "runtime_dir": str(runtime_dir),
            "control_dir": str(control_dir),
            "pending_actions_dir": str(pending_dir),
            "approved_actions_dir": str(approved_dir),
            "rejected_actions_dir": str(rejected_dir),
            "action_applier_state_path": applier.get("path"),
        },
        "debug": {
            "pending_sample": _list_any_dir(pending_dir),
            "approved_sample": _list_any_dir(approved_dir),
            "rejected_sample": _list_any_dir(rejected_dir),
            "action_applier_exists": applier.get("exists"),
            "last_action_id": applier.get("last_action_id"),
            "scan_ts_utc": applier.get("scan_ts_utc"),
            "queue_scan": applier.get("queue_scan"),
            "applied_actions_count": applier.get("applied_actions_count"),
        },
        "notes": (
            "Bridged mutation state publisher. Queue counts come from control/*_actions "
            "and apply history comes from runtime/action_applier_state.json."
        ),
    }
    return payload


_last_known_composite_hash: Optional[str] = None


def build_change_canary_state(*, repo_root: Path, runtime_dir: Path, control_dir: Path) -> Dict[str, Any]:
    global _last_known_composite_hash

    # Compute current config composite hash
    try:
        current_hash = get_current_composite_hash(repo_root=repo_root)
    except Exception:
        LOG.exception("config hash computation failed")
        current_hash = None

    # Detect drift against last known hash
    tamper_detected = False
    if current_hash and _last_known_composite_hash and current_hash != _last_known_composite_hash:
        tamper_detected = True
        LOG.warning(
            "config tamper detected: previous=%s current=%s",
            _last_known_composite_hash[:16],
            current_hash[:16],
        )

    # Take periodic snapshot (best-effort)
    snap_ts = None
    try:
        snap_path = create_config_snapshot(repo_root=repo_root, trigger="periodic")
        snap_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        LOG.debug("periodic config snapshot: %s", snap_path)
    except Exception:
        LOG.exception("periodic config snapshot failed (non-fatal)")

    # Update last known hash
    if current_hash:
        _last_known_composite_hash = current_hash

    # Load last action_id from applier state
    applier = _load_action_applier_state(runtime_dir)
    last_action_id = applier.get("last_action_id")

    payload: Dict[str, Any] = {
        "schema_version": "change_canary_state.v1",
        "canary_factor": 0.5 if tamper_detected else 1.0,
        "canary_until_ts_utc": None,
        "reason": "config_drift_detected" if tamper_detected else "none",
        "tamper_detected": tamper_detected,
        "config_composite_hash": current_hash,
        "last_snapshot_ts": snap_ts,
        "last_action_id": last_action_id,
        "paths": {
            "repo_root": str(repo_root),
            "runtime_dir": str(runtime_dir),
            "control_dir": str(control_dir),
        },
        "notes": (
            "Config tamper detection active. Compares config/*.json composite hash "
            "across cycles. Tamper triggers canary_factor=0.5 sizing reduction."
        ),
    }
    return payload


def publish_once(*, repo_root: Path, runtime_dir: Path, control_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    ttl_action = _env_int("CHAD_ACTION_STATE_TTL_SECONDS", 60)
    ttl_canary = _env_int("CHAD_CANARY_STATE_TTL_SECONDS", 60)

    action_payload = build_action_state(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        control_dir=control_dir,
    )
    canary_payload = build_change_canary_state(
        repo_root=repo_root,
        runtime_dir=runtime_dir,
        control_dir=control_dir,
    )

    action_path = runtime_dir / "action_state.json"
    canary_path = runtime_dir / "change_canary_state.json"

    write_runtime_state_json(action_path, action_payload, ttl_seconds=int(ttl_action), inject_ts=True)
    write_runtime_state_json(canary_path, canary_payload, ttl_seconds=int(ttl_canary), inject_ts=True)

    LOG.info("published action_state=%s ttl=%ss", action_path, ttl_action)
    LOG.info("published change_canary_state=%s ttl=%ss", canary_path, ttl_canary)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Publish action_state + change_canary_state runtime artifacts.")
    ap.add_argument("--once", action="store_true", help="Publish once and exit.")
    ap.add_argument("--loop", action="store_true", help="Loop forever.")
    ap.add_argument("--interval-seconds", type=int, default=_env_int("CHAD_MUTATION_LOOP_INTERVAL_SECONDS", 60))
    ap.add_argument("--log-level", type=str, default=os.environ.get("CHAD_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    repo_root = _repo_root()
    runtime_dir = _runtime_dir(repo_root)
    control_dir = _control_dir(repo_root)

    if args.loop:
        interval = int(max(10, args.interval_seconds))
        LOG.info(
            "starting loop publisher: runtime_dir=%s control_dir=%s interval=%ss",
            runtime_dir,
            control_dir,
            interval,
        )
        while True:
            try:
                publish_once(repo_root=repo_root, runtime_dir=runtime_dir, control_dir=control_dir)
            except Exception as exc:
                LOG.exception("publish failed (will retry): %s", exc)
            time.sleep(interval)
        return 0

    try:
        publish_once(repo_root=repo_root, runtime_dir=runtime_dir, control_dir=control_dir)
        return 0
    except Exception as exc:
        LOG.exception("publish failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

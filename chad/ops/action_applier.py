#!/usr/bin/env python3
"""
CHAD — SSOT v5 ActionApplier

Consumes approved action documents and records deterministic apply state.

Responsibilities
----------------
1. Scan control/approved_actions/*.json
2. Skip or archive expired / already-applied actions
3. Dispatch supported kinds
4. Persist apply state to runtime/action_applier_state.json
5. Never perform broker operations directly
6. Consume successful actions exactly once
7. Keep queue truth aligned by moving completed/expired items out of approved_actions
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from chad.ops.config_snapshot import create_config_snapshot

LOG = logging.getLogger("chad.ops.action_applier")

ROOT = Path("/home/ubuntu/chad_finale")
CONTROL_DIR = ROOT / "control"
APPROVED_DIR = CONTROL_DIR / "approved_actions"
APPLIED_DIR = CONTROL_DIR / "applied_actions"
EXPIRED_DIR = CONTROL_DIR / "expired_actions"
UNSUPPORTED_DIR = CONTROL_DIR / "unsupported_actions"
RUNTIME_DIR = ROOT / "runtime"

STATE_PATH = RUNTIME_DIR / "action_applier_state.json"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def write_json_atomic(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            obj = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {
        "schema_version": "action_applier_state.v2",
        "last_apply_ts_utc": None,
        "last_action_id": None,
        "last_receipt_path": None,
        "last_failure": None,
        "scan_ts_utc": None,
        "queue_scan": {
            "approved_seen": 0,
            "expired_archived": 0,
            "already_applied_archived": 0,
            "unsupported_archived": 0,
            "applied_now": 0,
        },
        "applied_actions": [],
        "applied_action_meta": {},
    }


def save_state(state: Dict[str, Any]) -> None:
    write_json_atomic(STATE_PATH, state)


def is_expired(action: Dict[str, Any]) -> bool:
    exp = action.get("expires_ts_utc")
    if not isinstance(exp, str) or not exp.strip():
        return False
    return exp.strip() < utc_now()


def _move_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    os.replace(src, dst)
    return dst


def archive_applied_action(src: Path) -> Path:
    return _move_file(src, APPLIED_DIR)


def archive_expired_action(src: Path) -> Path:
    return _move_file(src, EXPIRED_DIR)


def archive_unsupported_action(src: Path) -> Path:
    return _move_file(src, UNSUPPORTED_DIR)


def apply_rebalance(action: Dict[str, Any]) -> str:
    action_id = str(action.get("action_id") or "").strip()
    if not action_id:
        raise RuntimeError("missing_action_id")

    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    profile = str(payload.get("profile") or "BALANCED").strip().upper()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CHAD_AUTO_EXECUTE_REBALANCE"] = env.get("CHAD_AUTO_EXECUTE_REBALANCE", "1")
    env["CHAD_PORTFOLIO_PROFILE"] = profile

    cmd = [
        "python3",
        str(ROOT / "ops" / "rebalance_auto_executor_paper.py"),
        "--execute",
        "--approval-id",
        action_id,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        raise RuntimeError(
            f"rebalance_executor_failed:exit_{proc.returncode}:{stderr or stdout or 'no_output'}"
        )

    if not stdout:
        raise RuntimeError("rebalance_executor_empty_stdout")

    try:
        obj = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError(f"rebalance_executor_bad_json:{type(exc).__name__}") from exc

    if not isinstance(obj, dict):
        raise RuntimeError("rebalance_executor_non_dict")

    if obj.get("ok") is not True:
        reason = str(obj.get("reason") or "unknown")
        raise RuntimeError(f"rebalance_executor_blocked:{reason}")

    out_path = str(obj.get("out") or "").strip()
    if not out_path:
        raise RuntimeError("rebalance_executor_missing_out_path")

    out_file = Path(out_path)
    if not out_file.is_file():
        raise RuntimeError(f"rebalance_executor_output_missing:{out_path}")

    return out_path


def main() -> int:
    state = load_state()

    applied: List[str] = list(state.get("applied_actions") or [])
    applied_set = set(applied)

    applied_meta = state.get("applied_action_meta")
    if not isinstance(applied_meta, dict):
        applied_meta = {}

    approved_seen = 0
    expired_archived = 0
    already_applied_archived = 0
    unsupported_archived = 0
    applied_now = 0

    APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    for p in sorted(APPROVED_DIR.glob("*.json")):
        approved_seen += 1

        action = read_json(p)
        if not action:
            continue

        action_id = str(action.get("action_id") or "").strip()
        kind = str(action.get("kind") or "").strip().lower()

        if not action_id:
            continue

        if action_id in applied_set:
            archive_applied_action(p)
            already_applied_archived += 1
            continue

        if is_expired(action):
            archive_expired_action(p)
            expired_archived += 1
            continue

        if kind != "rebalance_execute":
            archive_unsupported_action(p)
            unsupported_archived += 1
            continue

        try:
            # Pre-apply config snapshot for tamper detection
            try:
                snap_path = create_config_snapshot(
                    repo_root=ROOT, trigger=f"pre_apply:{action_id}"
                )
                LOG.info("pre-apply snapshot: %s", snap_path)
            except Exception:
                LOG.exception("config snapshot failed (non-fatal)")
                snap_path = None

            receipt_path = apply_rebalance(action)

            archive_applied_action(p)

            applied.append(action_id)
            applied_set.add(action_id)
            applied_now += 1

            applied_meta[action_id] = {
                "kind": kind,
                "applied_ts_utc": utc_now(),
                "receipt_path": receipt_path,
                "pre_apply_snapshot": str(snap_path) if snap_path else None,
            }

            state["last_apply_ts_utc"] = utc_now()
            state["last_action_id"] = action_id
            state["last_receipt_path"] = receipt_path
            state["last_failure"] = None

        except Exception as exc:
            state["last_failure"] = f"{type(exc).__name__}:{exc}"
            state["scan_ts_utc"] = utc_now()
            state["queue_scan"] = {
                "approved_seen": approved_seen,
                "expired_archived": expired_archived,
                "already_applied_archived": already_applied_archived,
                "unsupported_archived": unsupported_archived,
                "applied_now": applied_now,
            }
            state["applied_actions"] = applied
            state["applied_action_meta"] = applied_meta
            save_state(state)
            return 2

    state["scan_ts_utc"] = utc_now()
    state["queue_scan"] = {
        "approved_seen": approved_seen,
        "expired_archived": expired_archived,
        "already_applied_archived": already_applied_archived,
        "unsupported_archived": unsupported_archived,
        "applied_now": applied_now,
    }
    state["applied_actions"] = applied
    state["applied_action_meta"] = applied_meta

    save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

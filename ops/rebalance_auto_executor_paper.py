#!/usr/bin/env python3
"""
CHAD Phase 12 — Portfolio Rebalance Auto-Executor (PAPER receipts, OFF by default)

File: ops/rebalance_auto_executor_paper.py

Goal
----
Build the Phase 12 autonomy scaffold *safely*:
- Reads the latest rebalance plan artifact (reports/rebalance/REBALANCE_<PROFILE>_<ts>.json)
- Produces PAPER receipts only (no broker calls, no orders)
- Enforces strict gating:
    1) STOP must be false
    2) AUTO_EXECUTE must be enabled via env (default OFF)
    3) LiveGate must allow paper lane (allow_ibkr_paper == True)
    4) CHAD must NOT be live (we never do live here)
- Exactly-once receipts using IdempotencyStore so the same plan hash cannot be "executed" twice.

Artifacts
---------
Reads:
- /home/ubuntu/CHAD FINALE/runtime/stop_state.json
- http://127.0.0.1:9618/live-gate
- /home/ubuntu/CHAD FINALE/reports/rebalance/REBALANCE_<PROFILE>_*.json

Writes (EXECUTE only):
- /home/ubuntu/CHAD FINALE/reports/rebalance_receipts/REBALANCE_RECEIPTS_<PROFILE>_<ts>.json
- SQLite: runtime/exec_state_paper.sqlite3 table rebalance_receipts (idempotency)

Modes
-----
- PREVIEW (default): prints what it would do, writes nothing
- EXECUTE (--execute): writes receipts, but only if gates allow

This is the safe bridge to later real execution lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from chad.execution.idempotency_store import IdempotencyStore, default_paper_db_path


# -----------------------------
# Config
# -----------------------------

PROFILE = os.environ.get("CHAD_PORTFOLIO_PROFILE", "BALANCED").strip().upper()

AUTO_EXECUTE = os.environ.get("CHAD_AUTO_EXECUTE_REBALANCE", "0").strip().lower() in ("1", "true", "yes", "on")

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
STOP_PATH = Path(os.environ.get("CHAD_STOP_STATE_PATH", str(RUNTIME_DIR / "stop_state.json")))

REBALANCE_DIR = Path(os.environ.get("CHAD_REBALANCE_DIR", "/home/ubuntu/CHAD FINALE/reports/rebalance"))
RECEIPTS_DIR = Path(os.environ.get("CHAD_REBALANCE_RECEIPTS_DIR", "/home/ubuntu/CHAD FINALE/reports/rebalance_receipts"))

LIVE_GATE_URL = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")
HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "4.0"))

IDEMP_TABLE = os.environ.get("CHAD_REBALANCE_IDEMP_TABLE", "rebalance_receipts")


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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


def read_json_dict(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_dict:{path}")
    return obj


def sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def hash_obj(obj: Dict[str, Any]) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_hex_bytes(b)


def stop_engaged() -> Tuple[bool, str]:
    try:
        obj = read_json_dict(STOP_PATH)
        return bool(obj.get("stop", False)), str(obj.get("reason") or "")
    except Exception as exc:
        # fail-closed: treat as STOP engaged
        return True, f"stop_state_unreadable:{type(exc).__name__}"


def fetch_live_gate() -> Dict[str, Any]:
    req = Request(LIVE_GATE_URL, headers={"User-Agent": "chad-rebalance-auto/1.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("live_gate_invalid_shape")
    return obj


def select_latest_rebalance_file() -> Path:
    # REBALANCE_<PROFILE>_*.json newest
    pattern = f"REBALANCE_{PROFILE}_*.json"
    files = sorted(REBALANCE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no_rebalance_files:{REBALANCE_DIR}/{pattern}")
    return files[0]


def receipt_id(plan_hash: str, symbol: str) -> str:
    base = f"{plan_hash}|{PROFILE}|{symbol.strip().upper()}"
    return sha256_hex_bytes(base.encode("utf-8"))


# -----------------------------
# Core
# -----------------------------

def run(*, execute: bool) -> int:
    ts = utc_now_iso()

    stop, stop_reason = stop_engaged()
    if stop:
        msg = {"ok": False, "blocked": True, "reason": f"STOP_ENGAGED:{stop_reason}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}
        print(json.dumps(msg, sort_keys=True))
        return 0

    if execute and not AUTO_EXECUTE:
        msg = {"ok": False, "blocked": True, "reason": "AUTO_EXECUTE_DISABLED", "mode": "EXECUTE", "ts_utc": ts}
        print(json.dumps(msg, sort_keys=True))
        return 0

    # LiveGate check (always in execute; optional in preview but useful)
    lg: Optional[Dict[str, Any]] = None
    try:
        lg = fetch_live_gate()
    except Exception as exc:
        if execute:
            msg = {"ok": False, "blocked": True, "reason": f"LIVE_GATE_UNREACHABLE:{type(exc).__name__}", "mode": "EXECUTE", "ts_utc": ts}
            print(json.dumps(msg, sort_keys=True))
            return 0

    if execute and lg is not None:
        allow_paper = bool(lg.get("allow_ibkr_paper", False))
        live_enabled = bool((lg.get("mode") or {}).get("live_enabled", False)) if isinstance(lg.get("mode"), dict) else False
        if not allow_paper:
            msg = {"ok": False, "blocked": True, "reason": "LIVE_GATE_DENY_PAPER", "mode": "EXECUTE", "ts_utc": ts}
            print(json.dumps(msg, sort_keys=True))
            return 0
        if live_enabled:
            msg = {"ok": False, "blocked": True, "reason": "LIVE_MODE_NOT_ALLOWED_IN_PAPER_EXECUTOR", "mode": "EXECUTE", "ts_utc": ts}
            print(json.dumps(msg, sort_keys=True))
            return 0

    # Load latest rebalance plan
    reb_path = select_latest_rebalance_file()
    reb = read_json_dict(reb_path)

    plan_hash = str(reb.get("rebalance_plan_hash") or "")
    if plan_hash.startswith("sha256:"):
        plan_hash = plan_hash.split(":", 1)[1].strip()
    if not plan_hash:
        plan_hash = hash_obj(reb)

    diffs = reb.get("diffs") or []
    if not isinstance(diffs, list):
        diffs = []

    # Build receipts: include only meaningful deltas (>0.0001)
    receipts: List[Dict[str, Any]] = []
    for d in diffs:
        if not isinstance(d, dict):
            continue
        sym = str(d.get("symbol") or "").strip().upper()
        if not sym:
            continue
        try:
            delta = float(d.get("delta_weight", 0.0))
        except Exception:
            continue
        if abs(delta) < 0.0001:
            continue

        rid = receipt_id(plan_hash, sym)
        receipts.append({
            "receipt_id": rid,
            "ts_utc": ts,
            "profile": PROFILE,
            "rebalance_plan_path": str(reb_path),
            "rebalance_plan_hash": f"sha256:{plan_hash}",
            "symbol": sym,
            "delta_weight": float(delta),
            "paper": True,
            "note": "receipt_only_no_broker_calls",
        })

    if not execute:
        report = {
            "ok": True,
            "mode": "PREVIEW",
            "ts_utc": ts,
            "profile": PROFILE,
            "rebalance_plan_path": str(reb_path),
            "rebalance_plan_hash": f"sha256:{plan_hash}",
            "receipts_count": len(receipts),
            "receipts": receipts,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    # EXECUTE: idempotent mark + write report file
    repo_root = Path(__file__).resolve().parents[1]  # .../chad_finale/ops -> parents[1] = repo root
    store = IdempotencyStore(default_paper_db_path(repo_root), table=IDEMP_TABLE)

    for r in receipts:
        payload_hash = sha256_hex_bytes(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        res = store.mark_once(r["receipt_id"], payload_hash, meta={"source": "rebalance_auto_executor_paper", "plan_hash": plan_hash})
        r["idempotency"] = {"inserted": res.inserted, "reason": res.reason}
        r["status"] = "recorded" if res.inserted else "duplicate_skipped"

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS_DIR / f"REBALANCE_RECEIPTS_{PROFILE}_{utc_now_compact()}.json"
    report = {
        "ok": True,
        "mode": "EXECUTE",
        "ts_utc": ts,
        "profile": PROFILE,
        "rebalance_plan_path": str(reb_path),
        "rebalance_plan_hash": f"sha256:{plan_hash}",
        "receipts_count": len(receipts),
        "receipts": receipts,
        "live_gate": lg if isinstance(lg, dict) else None,
    }
    atomic_write_json(out, report)

    print(json.dumps({"ok": True, "mode": "EXECUTE", "out": str(out), "receipts_count": len(receipts), "ts_utc": ts}, sort_keys=True))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 12 rebalance auto-executor (paper receipts; OFF by default).")
    ap.add_argument("--execute", action="store_true", help="Write receipts (requires CHAD_AUTO_EXECUTE_REBALANCE=1 and LiveGate allows paper).")
    args = ap.parse_args(argv)
    return run(execute=bool(args.execute))


if __name__ == "__main__":
    raise SystemExit(main())

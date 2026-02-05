#!/usr/bin/env python3
"""
CHAD Phase 12 — Rebalance Auto-Executor v3 (PAPER receipts, bounded, assisted autonomy)

File: ops/rebalance_auto_executor_paper.py

What this does
--------------
Consumes the newest rebalance plan artifact (reports/rebalance/REBALANCE_<PROFILE>_<ts>.json)
and produces PAPER receipts only (no broker calls, no orders). It is a safety-first autonomy scaffold.

Modes
-----
PREVIEW (default):
  - Always safe
  - Computes decision, receipts, and reasons
  - Writes NOTHING

EXECUTE (--execute):
  - Writes receipts to reports/rebalance_receipts/
  - Writes pointer to runtime/rebalance_autonomy_state.json
  - Uses SQLite idempotency store (exactly-once per receipt_id)
  - Requires ALL gates:

Gates (EXECUTE)
---------------
1) STOP must be false (fail-closed if unreadable)
2) AUTO_EXECUTE env must be enabled (CHAD_AUTO_EXECUTE_REBALANCE=1)
3) LiveGate must allow paper lane (allow_ibkr_paper == true), and live_enabled must be false
4) autonomy_bounds.json must exist + validate schema
5) Event risk severity must NOT be in blocked list
6) Drift thresholds must be met
7) Caps must be met (turnover + per-symbol)
8) Cooldown must be met
9) Approval must be granted:
   - --approval-id <id> REQUIRED in EXECUTE
   - GET /approvals/{id} must return item.status == "approved"
   - item.kind must match "rebalance_execute"
   - item.payload.profile (if present) must match PROFILE

No broker calls. No secrets are written to runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen


from chad.execution.idempotency_store import IdempotencyStore, default_paper_db_path


# =============================================================================
# Configuration
# =============================================================================

PROFILE = os.environ.get("CHAD_PORTFOLIO_PROFILE", "BALANCED").strip().upper()

AUTO_EXECUTE = os.environ.get("CHAD_AUTO_EXECUTE_REBALANCE", "0").strip().lower() in ("1", "true", "yes", "on")

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime")).resolve()
CONFIG_DIR = Path(os.environ.get("CHAD_CONFIG_DIR", "/home/ubuntu/CHAD FINALE/config")).resolve()

STOP_PATH = Path(os.environ.get("CHAD_STOP_STATE_PATH", str(RUNTIME_DIR / "stop_state.json")))
PORTFOLIO_STATE_PATH = Path(os.environ.get("CHAD_PORTFOLIO_STATE_PATH", str(RUNTIME_DIR / "portfolio_state.json")))
EVENT_RISK_PATH = Path(os.environ.get("CHAD_EVENT_RISK_PATH", str(RUNTIME_DIR / "event_risk.json")))
BOUNDS_PATH = Path(os.environ.get("CHAD_AUTONOMY_BOUNDS_PATH", str(CONFIG_DIR / "autonomy_bounds.json")))

REBALANCE_DIR = Path(os.environ.get("CHAD_REBALANCE_DIR", "/home/ubuntu/CHAD FINALE/reports/rebalance")).resolve()
RECEIPTS_DIR = Path(os.environ.get("CHAD_REBALANCE_RECEIPTS_DIR", "/home/ubuntu/CHAD FINALE/reports/rebalance_receipts")).resolve()

AUTONOMY_STATE_PATH = Path(
    os.environ.get("CHAD_REBALANCE_AUTONOMY_STATE_PATH", str(RUNTIME_DIR / "rebalance_autonomy_state.json"))
).resolve()

LIVE_GATE_URL = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")
APPROVALS_BASE_URL = os.environ.get("CHAD_APPROVALS_BASE_URL", "http://127.0.0.1:9618/approvals")

HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "4.0"))

IDEMP_TABLE = os.environ.get("CHAD_REBALANCE_IDEMP_TABLE", "rebalance_receipts")

# delta_weight filter for "meaningful" actions
MIN_ABS_DELTA = float(os.environ.get("CHAD_REBALANCE_MIN_ABS_DELTA", "0.0001"))


# =============================================================================
# Models
# =============================================================================

@dataclass(frozen=True)
class Bounds:
    min_seconds_between_executes: int
    max_total_turnover_per_execute: float
    max_abs_delta_weight_per_symbol: float
    drift_max_gte: float
    drift_turnover_gte: float
    block_event_risk_severities: Tuple[str, ...]


@dataclass(frozen=True)
class Drift:
    max_position_weight_drift: float
    total_turnover_needed: float


@dataclass(frozen=True)
class Decision:
    ok: bool
    blocked: bool
    reason: str
    ts_utc: str
    mode: str
    profile: str
    rebalance_plan_path: str
    rebalance_plan_hash: str
    receipts_count: int
    would_execute: bool
    checks: Dict[str, Any]
    bounds: Dict[str, Any]
    event_risk_severity: str
    cooldown: Dict[str, Any]
    approval: Dict[str, Any]


# =============================================================================
# Utility
# =============================================================================

def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


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


def parse_sha256(s: str) -> str:
    x = str(s or "").strip()
    if x.startswith("sha256:"):
        return x.split(":", 1)[1].strip()
    return x


def _parse_ts_utc(ts: str) -> Optional[int]:
    # expects YYYY-MM-DDTHH:MM:SSZ
    try:
        y = int(ts[0:4]); mo = int(ts[5:7]); d = int(ts[8:10])
        hh = int(ts[11:13]); mm = int(ts[14:16]); ss = int(ts[17:19])
        epoch = int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone
        return epoch
    except Exception:
        return None


# =============================================================================
# Providers (IO / Network)
# =============================================================================

def stop_engaged() -> Tuple[bool, str]:
    try:
        obj = read_json_dict(STOP_PATH)
        return bool(obj.get("stop", False)), str(obj.get("reason") or "")
    except Exception as exc:
        return True, f"stop_state_unreadable:{type(exc).__name__}"


def fetch_live_gate() -> Dict[str, Any]:
    req = Request(LIVE_GATE_URL, headers={"User-Agent": "chad-rebalance-auto/3.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("live_gate_invalid_shape")
    return obj


def fetch_approval_item(approval_id: str) -> Dict[str, Any]:
    aid = approval_id.strip()
    if not aid:
        raise ValueError("empty_approval_id")
    url = f"{APPROVALS_BASE_URL}/{aid}"
    req = Request(url, headers={"User-Agent": "chad-rebalance-auto/approval/1.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or "item" not in obj:
        raise ValueError("approval_invalid_shape")
    item = obj.get("item")
    if not isinstance(item, dict):
        raise ValueError("approval_item_invalid")
    return item


def select_latest_rebalance_file() -> Path:
    pattern = f"REBALANCE_{PROFILE}_*.json"
    files = sorted(REBALANCE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no_rebalance_files:{REBALANCE_DIR}/{pattern}")
    return files[0]


def load_event_risk_severity() -> str:
    try:
        er = read_json_dict(EVENT_RISK_PATH)
        return str(er.get("severity") or "unknown").strip().lower()
    except Exception:
        return "unknown"


def load_drift() -> Drift:
    ps = read_json_dict(PORTFOLIO_STATE_PATH)
    drift = ps.get("drift") if isinstance(ps.get("drift"), dict) else {}
    try:
        max_d = float(drift.get("max_position_weight_drift", 0.0))
    except Exception:
        max_d = 0.0
    try:
        tot_t = float(drift.get("total_turnover_needed", 0.0))
    except Exception:
        tot_t = 0.0
    return Drift(max_position_weight_drift=max_d, total_turnover_needed=tot_t)


def load_bounds() -> Bounds:
    obj = read_json_dict(BOUNDS_PATH)
    if obj.get("schema_version") != "autonomy_bounds.v1":
        raise ValueError("bad_bounds_schema")
    b = obj.get("rebalance_auto_execute")
    if not isinstance(b, dict):
        raise ValueError("bad_bounds_shape")

    cfg = b.get("min_drift_to_execute") if isinstance(b.get("min_drift_to_execute"), dict) else {}
    sev_list = b.get("block_if_event_risk_severity_in") if isinstance(b.get("block_if_event_risk_severity_in"), list) else []
    sev_norm = tuple(str(x).strip().lower() for x in sev_list if str(x).strip())

    return Bounds(
        min_seconds_between_executes=int(b.get("min_seconds_between_executes", 0)),
        max_total_turnover_per_execute=float(b.get("max_total_turnover_per_execute", 0.0)),
        max_abs_delta_weight_per_symbol=float(b.get("max_abs_delta_weight_per_symbol", 0.0)),
        drift_max_gte=float(cfg.get("max_position_weight_drift_gte", 0.0)),
        drift_turnover_gte=float(cfg.get("total_turnover_needed_gte", 0.0)),
        block_event_risk_severities=sev_norm,
    )


def load_autonomy_state() -> Dict[str, Any]:
    if not AUTONOMY_STATE_PATH.is_file():
        return {}
    try:
        return read_json_dict(AUTONOMY_STATE_PATH)
    except Exception:
        return {}


# =============================================================================
# Policy evaluation
# =============================================================================

def drift_ok(bounds: Bounds, drift: Drift) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True
    if drift.max_position_weight_drift < bounds.drift_max_gte:
        ok = False
        reasons.append(f"drift_max_below_threshold:{drift.max_position_weight_drift:.6f}<{bounds.drift_max_gte:.6f}")
    if drift.total_turnover_needed < bounds.drift_turnover_gte:
        ok = False
        reasons.append(f"drift_turnover_below_threshold:{drift.total_turnover_needed:.6f}<{bounds.drift_turnover_gte:.6f}")
    return ok, reasons


def cooldown_ok(bounds: Bounds, now_epoch: int, last_exec_ts_utc: Optional[str]) -> Tuple[bool, str]:
    if bounds.min_seconds_between_executes <= 0:
        return True, "cooldown_disabled"
    if not last_exec_ts_utc:
        return True, "no_prior_execute"
    last_epoch = _parse_ts_utc(last_exec_ts_utc)
    if last_epoch is None:
        return False, "last_exec_ts_unparseable"
    if now_epoch >= last_epoch + bounds.min_seconds_between_executes:
        return True, "cooldown_ok"
    return False, "cooldown_not_met"


def enforce_caps(bounds: Bounds, receipts: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    turnover = 0.0
    for r in receipts:
        turnover += abs(float(r.get("delta_weight", 0.0)))
    if bounds.max_total_turnover_per_execute > 0 and turnover > bounds.max_total_turnover_per_execute:
        reasons.append(f"turnover_cap_exceeded:{turnover:.6f}>{bounds.max_total_turnover_per_execute:.6f}")
    if bounds.max_abs_delta_weight_per_symbol > 0:
        for r in receipts:
            v = abs(float(r.get("delta_weight", 0.0)))
            if v > bounds.max_abs_delta_weight_per_symbol:
                reasons.append(f"per_symbol_cap_exceeded:{r.get('symbol')}:{v:.6f}>{bounds.max_abs_delta_weight_per_symbol:.6f}")
    return (len(reasons) == 0), reasons


def build_receipts(reb_path: Path, reb: Dict[str, Any], plan_hash: str, ts: str) -> List[Dict[str, Any]]:
    diffs = reb.get("diffs") if isinstance(reb.get("diffs"), list) else []
    out: List[Dict[str, Any]] = []
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
        if abs(delta) < MIN_ABS_DELTA:
            continue
        rid = sha256_hex_bytes(f"{plan_hash}|{PROFILE}|{sym}".encode("utf-8"))
        out.append(
            {
                "receipt_id": rid,
                "ts_utc": ts,
                "profile": PROFILE,
                "rebalance_plan_path": str(reb_path),
                "rebalance_plan_hash": f"sha256:{plan_hash}",
                "symbol": sym,
                "delta_weight": float(delta),
                "paper": True,
                "note": "receipt_only_no_broker_calls",
            }
        )
    return out


def approval_ok(approval_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    if not approval_id.strip():
        return False, "APPROVAL_ID_REQUIRED", {}
    try:
        item = fetch_approval_item(approval_id)
    except Exception as exc:
        return False, f"APPROVAL_LOOKUP_FAILED:{type(exc).__name__}", {}

    status = str(item.get("status") or "").strip().lower()
    kind = str(item.get("kind") or "").strip().lower()
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    prof = str(payload.get("profile") or "").strip().upper()

    if kind != "rebalance_execute":
        return False, f"APPROVAL_KIND_MISMATCH:{kind}", item
    if prof and prof != PROFILE:
        return False, f"APPROVAL_PROFILE_MISMATCH:{prof}!={PROFILE}", item
    if status != "approved":
        return False, f"APPROVAL_NOT_GRANTED:{status}", item
    return True, "APPROVED", item


# =============================================================================
# Runner
# =============================================================================

def run(*, execute: bool, approval_id: str) -> int:
    ts = utc_now_iso()
    now_epoch = int(time.time())

    stop, stop_reason = stop_engaged()
    if stop:
        print(json.dumps({"ok": False, "blocked": True, "reason": f"STOP_ENGAGED:{stop_reason}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}, sort_keys=True))
        return 0

    # bounds required even for preview (so preview reflects real policy)
    try:
        bounds = load_bounds()
    except Exception as exc:
        print(json.dumps({"ok": False, "blocked": True, "reason": f"BOUNDS_INVALID:{type(exc).__name__}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}, sort_keys=True))
        return 0

    sev = load_event_risk_severity()
    if sev in bounds.block_event_risk_severities:
        print(json.dumps({"ok": False, "blocked": True, "reason": f"EVENT_RISK_BLOCKED:{sev}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}, sort_keys=True))
        return 0

    drift = load_drift()
    d_ok, d_reasons = drift_ok(bounds, drift)

    state = load_autonomy_state()
    last_exec_ts = str(state.get("last_execute_ts_utc") or "").strip() or None
    cd_ok, cd_reason = cooldown_ok(bounds, now_epoch, last_exec_ts)

    reb_path = select_latest_rebalance_file()
    reb = read_json_dict(reb_path)

    plan_hash = parse_sha256(str(reb.get("rebalance_plan_hash") or ""))
    if not plan_hash:
        plan_hash = sha256_hex_bytes(json.dumps(reb, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    receipts = build_receipts(reb_path, reb, plan_hash, ts)
    c_ok, c_reasons = enforce_caps(bounds, receipts)

    would_execute = bool(AUTO_EXECUTE and d_ok and c_ok and cd_ok)

    decision = Decision(
        ok=True,
        blocked=False,
        reason="OK",
        ts_utc=ts,
        mode="EXECUTE" if execute else "PREVIEW",
        profile=PROFILE,
        rebalance_plan_path=str(reb_path),
        rebalance_plan_hash=f"sha256:{plan_hash}",
        receipts_count=len(receipts),
        would_execute=would_execute,
        checks={
            "drift_ok": d_ok,
            "drift_reasons": d_reasons,
            "caps_ok": c_ok,
            "caps_reasons": c_reasons,
        },
        bounds={
            "min_seconds_between_executes": bounds.min_seconds_between_executes,
            "max_total_turnover_per_execute": bounds.max_total_turnover_per_execute,
            "max_abs_delta_weight_per_symbol": bounds.max_abs_delta_weight_per_symbol,
            "min_drift_to_execute": {
                "max_position_weight_drift_gte": bounds.drift_max_gte,
                "total_turnover_needed_gte": bounds.drift_turnover_gte,
            },
            "block_if_event_risk_severity_in": list(bounds.block_event_risk_severities),
        },
        event_risk_severity=sev,
        cooldown={"ok": cd_ok, "reason": cd_reason, "last_execute_ts_utc": last_exec_ts},
        approval={"required": True, "approval_id": approval_id.strip() or None},
    )

    if not execute:
        out = {
            **decision.__dict__,
            "auto_execute_env": AUTO_EXECUTE,
            "receipts_preview": receipts[:5],
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    # EXECUTE gates
    if not AUTO_EXECUTE:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": "AUTO_EXECUTE_DISABLED"}, sort_keys=True))
        return 0
    if not d_ok:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": "DRIFT_NOT_MET"}, sort_keys=True))
        return 0
    if not c_ok:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": "CAPS_NOT_MET"}, sort_keys=True))
        return 0
    if not cd_ok:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": "COOLDOWN_NOT_MET"}, sort_keys=True))
        return 0

    ok_app, app_reason, app_item = approval_ok(approval_id)
    if not ok_app:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": app_reason}, sort_keys=True))
        return 0

    # LiveGate (paper only)
    try:
        lg = fetch_live_gate()
    except Exception as exc:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": f"LIVE_GATE_UNREACHABLE:{type(exc).__name__}"}, sort_keys=True))
        return 0

    allow_paper = bool(lg.get("allow_ibkr_paper", False))
    live_enabled = bool((lg.get("mode") or {}).get("live_enabled", False)) if isinstance(lg.get("mode"), dict) else False
    if not allow_paper:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": "LIVE_GATE_DENY_PAPER"}, sort_keys=True))
        return 0
    if live_enabled:
        print(json.dumps({**decision.__dict__, "ok": False, "blocked": True, "reason": "LIVE_MODE_NOT_ALLOWED_IN_PAPER_EXECUTOR"}, sort_keys=True))
        return 0

    # EXECUTE: idempotent mark + write report + pointer
    repo_root = Path(__file__).resolve().parents[1]
    store = IdempotencyStore(default_paper_db_path(repo_root), table=IDEMP_TABLE)

    for r in receipts:
        payload_hash = sha256_hex_bytes(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        res = store.mark_once(r["receipt_id"], payload_hash, meta={"source": "rebalance_auto_executor_paper_v3", "plan_hash": plan_hash})
        r["idempotency"] = {"inserted": res.inserted, "reason": res.reason}
        r["status"] = "recorded" if res.inserted else "duplicate_skipped"

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RECEIPTS_DIR / f"REBALANCE_RECEIPTS_{PROFILE}_{utc_now_compact()}.json"
    report = {
        **decision.__dict__,
        "ok": True,
        "blocked": False,
        "reason": "EXECUTED_PAPER_RECEIPTS",
        "auto_execute_env": AUTO_EXECUTE,
        "approval_item": app_item,
        "live_gate": lg,
        "receipts": receipts,
    }
    atomic_write_json(out_path, report)

    atomic_write_json(
        AUTONOMY_STATE_PATH,
        {
            "ts_utc": ts,
            "ttl_seconds": 86400,
            "schema_version": "rebalance_autonomy_state.v2",
            "profile": PROFILE,
            "last_execute_ts_utc": ts,
            "last_rebalance_plan_hash": f"sha256:{plan_hash}",
            "last_receipts_path": str(out_path),
            "approval_id": approval_id.strip(),
            "notes": "assisted_autonomy_paper_only",
        },
    )

    print(json.dumps({"ok": True, "mode": "EXECUTE", "out": str(out_path), "receipts_count": len(receipts), "ts_utc": ts}, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 12 rebalance auto-executor v3 (paper receipts, bounded, approval-gated).")
    ap.add_argument("--execute", action="store_true", help="Write receipts if all gates + bounds + approval pass.")
    ap.add_argument("--approval-id", default="", help="Required for --execute. Must be approved in /approvals/{id}.")
    args = ap.parse_args()
    return run(execute=bool(args.execute), approval_id=str(args.approval_id or ""))


if __name__ == "__main__":
    raise SystemExit(main())

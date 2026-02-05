#!/usr/bin/env python3
"""
CHAD Phase 12 — Portfolio Rebalance Auto-Executor (PAPER receipts, bounded)

File: ops/rebalance_auto_executor_paper.py

Phase 12 Part 2 objective:
- Add bounded automation rules (drift threshold, max turnover, per-symbol cap, cooldown)
- Still PAPER receipts only (no broker calls, no orders)
- Still OFF by default unless CHAD_AUTO_EXECUTE_REBALANCE=1

Gates (EXECUTE):
1) STOP must be false
2) CHAD_AUTO_EXECUTE_REBALANCE=1
3) LiveGate must allow paper lane (allow_ibkr_paper == true)
4) live_enabled must be false (never live here)
5) autonomy_bounds.json must be present + valid
6) Drift thresholds must be met (min_drift_to_execute)
7) Cooldown must be satisfied (min_seconds_between_executes)
8) Turnover + per-symbol delta caps enforced
9) If event_risk severity is in blocked list => block

Modes:
- PREVIEW (default): prints decision + proposed receipts, writes nothing
- EXECUTE (--execute): writes receipts report + idempotency marks (exactly-once)

Artifacts:
- Reads:
  - runtime/stop_state.json
  - runtime/portfolio_state.json
  - runtime/event_risk.json
  - live-gate endpoint
  - latest REBALANCE_<PROFILE>_*.json
  - config/autonomy_bounds.json
- Writes (EXECUTE only):
  - reports/rebalance_receipts/REBALANCE_RECEIPTS_<PROFILE>_<ts>.json
  - sqlite idempotency table: rebalance_receipts
  - runtime/rebalance_autonomy_state.json (pointer + last_execute_ts_utc)

No secrets written. No broker calls.
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
# Config / Paths
# -----------------------------

PROFILE = os.environ.get("CHAD_PORTFOLIO_PROFILE", "BALANCED").strip().upper()
AUTO_EXECUTE = os.environ.get("CHAD_AUTO_EXECUTE_REBALANCE", "0").strip().lower() in ("1", "true", "yes", "on")

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/CHAD FINALE/runtime"))
CONFIG_DIR = Path(os.environ.get("CHAD_CONFIG_DIR", "/home/ubuntu/CHAD FINALE/config"))

STOP_PATH = Path(os.environ.get("CHAD_STOP_STATE_PATH", str(RUNTIME_DIR / "stop_state.json")))
PORTFOLIO_STATE_PATH = Path(os.environ.get("CHAD_PORTFOLIO_STATE_PATH", str(RUNTIME_DIR / "portfolio_state.json")))
EVENT_RISK_PATH = Path(os.environ.get("CHAD_EVENT_RISK_PATH", str(RUNTIME_DIR / "event_risk.json")))
BOUNDS_PATH = Path(os.environ.get("CHAD_AUTONOMY_BOUNDS_PATH", str(CONFIG_DIR / "autonomy_bounds.json")))

REBALANCE_DIR = Path(os.environ.get("CHAD_REBALANCE_DIR", "/home/ubuntu/CHAD FINALE/reports/rebalance"))
RECEIPTS_DIR = Path(os.environ.get("CHAD_REBALANCE_RECEIPTS_DIR", "/home/ubuntu/CHAD FINALE/reports/rebalance_receipts"))

AUTONOMY_STATE_PATH = Path(os.environ.get("CHAD_REBALANCE_AUTONOMY_STATE_PATH", str(RUNTIME_DIR / "rebalance_autonomy_state.json")))

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


def stop_engaged() -> Tuple[bool, str]:
    try:
        obj = read_json_dict(STOP_PATH)
        return bool(obj.get("stop", False)), str(obj.get("reason") or "")
    except Exception as exc:
        return True, f"stop_state_unreadable:{type(exc).__name__}"


def fetch_live_gate() -> Dict[str, Any]:
    req = Request(LIVE_GATE_URL, headers={"User-Agent": "chad-rebalance-auto/2.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("live_gate_invalid_shape")
    return obj


def select_latest_rebalance_file() -> Path:
    pattern = f"REBALANCE_{PROFILE}_*.json"
    files = sorted(REBALANCE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"no_rebalance_files:{REBALANCE_DIR}/{pattern}")
    return files[0]


def parse_sha256(s: str) -> str:
    x = str(s or "").strip()
    if x.startswith("sha256:"):
        return x.split(":", 1)[1].strip()
    return x


def receipt_id(plan_hash: str, symbol: str) -> str:
    base = f"{plan_hash}|{PROFILE}|{symbol.strip().upper()}"
    return sha256_hex_bytes(base.encode("utf-8"))


def _parse_ts_utc(ts: str) -> Optional[int]:
    # expects YYYY-MM-DDTHH:MM:SSZ
    try:
        y = int(ts[0:4]); mo = int(ts[5:7]); d = int(ts[8:10])
        hh = int(ts[11:13]); mm = int(ts[14:16]); ss = int(ts[17:19])
        epoch = int(time.mktime((y, mo, d, hh, mm, ss, 0, 0, 0))) - time.timezone
        return epoch
    except Exception:
        return None


# -----------------------------
# Bounds + state
# -----------------------------

def load_bounds() -> Dict[str, Any]:
    obj = read_json_dict(BOUNDS_PATH)
    if obj.get("schema_version") != "autonomy_bounds.v1":
        raise ValueError("bad_bounds_schema")
    b = obj.get("rebalance_auto_execute")
    if not isinstance(b, dict):
        raise ValueError("bad_bounds_shape")
    return b


def load_autonomy_state() -> Dict[str, Any]:
    if not AUTONOMY_STATE_PATH.is_file():
        return {}
    try:
        return read_json_dict(AUTONOMY_STATE_PATH)
    except Exception:
        return {}


def cooldown_ok(bounds: Dict[str, Any], now_epoch: int, last_exec_ts: Optional[str]) -> Tuple[bool, str]:
    min_s = int(bounds.get("min_seconds_between_executes", 0))
    if min_s <= 0:
        return True, "cooldown_disabled"
    if not last_exec_ts:
        return True, "no_prior_execute"
    last_epoch = _parse_ts_utc(last_exec_ts)
    if last_epoch is None:
        return False, "last_exec_ts_unparseable"
    if now_epoch >= last_epoch + min_s:
        return True, "cooldown_ok"
    return False, "cooldown_not_met"


# -----------------------------
# Decision logic
# -----------------------------

def compute_drift() -> Dict[str, float]:
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
    return {"max_position_weight_drift": max_d, "total_turnover_needed": tot_t}


def event_risk_severity() -> str:
    try:
        er = read_json_dict(EVENT_RISK_PATH)
        return str(er.get("severity") or "unknown").strip().lower()
    except Exception:
        return "unknown"


def enforce_caps(bounds: Dict[str, Any], receipts: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    max_turnover = float(bounds.get("max_total_turnover_per_execute", 0.0))
    max_per_sym = float(bounds.get("max_abs_delta_weight_per_symbol", 0.0))

    turnover = 0.0
    for r in receipts:
        turnover += abs(float(r.get("delta_weight", 0.0)))

    if max_turnover > 0 and turnover > max_turnover:
        reasons.append(f"turnover_cap_exceeded:{turnover:.6f}>{max_turnover:.6f}")

    if max_per_sym > 0:
        for r in receipts:
            if abs(float(r.get("delta_weight", 0.0))) > max_per_sym:
                reasons.append(f"per_symbol_cap_exceeded:{r.get('symbol')}:{abs(float(r.get('delta_weight',0.0))):.6f}>{max_per_sym:.6f}")

    return (len(reasons) == 0), reasons


def drift_meets_threshold(bounds: Dict[str, Any], drift: Dict[str, float]) -> Tuple[bool, List[str]]:
    cfg = bounds.get("min_drift_to_execute") if isinstance(bounds.get("min_drift_to_execute"), dict) else {}
    need_max = float(cfg.get("max_position_weight_drift_gte", 0.0))
    need_tot = float(cfg.get("total_turnover_needed_gte", 0.0))

    ok = True
    reasons: List[str] = []

    if drift["max_position_weight_drift"] < need_max:
        ok = False
        reasons.append(f"drift_max_below_threshold:{drift['max_position_weight_drift']:.6f}<{need_max:.6f}")
    if drift["total_turnover_needed"] < need_tot:
        ok = False
        reasons.append(f"drift_turnover_below_threshold:{drift['total_turnover_needed']:.6f}<{need_tot:.6f}")

    return ok, reasons


# -----------------------------
# Core
# -----------------------------

def run(*, execute: bool) -> int:
    ts = utc_now_iso()
    now_epoch = int(time.time())

    stop, stop_reason = stop_engaged()
    if stop:
        print(json.dumps({"ok": False, "blocked": True, "reason": f"STOP_ENGAGED:{stop_reason}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}, sort_keys=True))
        return 0

    # bounds must exist for BOTH preview + execute (so preview reflects real policy)
    try:
        bounds = load_bounds()
    except Exception as exc:
        print(json.dumps({"ok": False, "blocked": True, "reason": f"BOUNDS_INVALID:{type(exc).__name__}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}, sort_keys=True))
        return 0

    # event risk block list
    sev = event_risk_severity()
    blocked_sev = bounds.get("block_if_event_risk_severity_in") if isinstance(bounds.get("block_if_event_risk_severity_in"), list) else []
    if str(sev).lower() in [str(x).lower() for x in blocked_sev]:
        print(json.dumps({"ok": False, "blocked": True, "reason": f"EVENT_RISK_BLOCKED:{sev}", "mode": "EXECUTE" if execute else "PREVIEW", "ts_utc": ts}, sort_keys=True))
        return 0

    drift = compute_drift()
    drift_ok, drift_reasons = drift_meets_threshold(bounds, drift)

    reb_path = select_latest_rebalance_file()
    reb = read_json_dict(reb_path)

    plan_hash = parse_sha256(str(reb.get("rebalance_plan_hash") or "")) or hash_obj(reb)
    diffs = reb.get("diffs") if isinstance(reb.get("diffs"), list) else []

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
        receipts.append({
            "receipt_id": receipt_id(plan_hash, sym),
            "ts_utc": ts,
            "profile": PROFILE,
            "rebalance_plan_path": str(reb_path),
            "rebalance_plan_hash": f"sha256:{plan_hash}",
            "symbol": sym,
            "delta_weight": float(delta),
            "paper": True,
            "note": "receipt_only_no_broker_calls",
        })

    caps_ok, caps_reasons = enforce_caps(bounds, receipts)

    # cooldown
    state = load_autonomy_state()
    last_exec_ts = str(state.get("last_execute_ts_utc") or "") if isinstance(state, dict) else ""
    cd_ok, cd_reason = cooldown_ok(bounds, now_epoch, last_exec_ts or None)

    decision = {
        "ok": True,
        "mode": "EXECUTE" if execute else "PREVIEW",
        "ts_utc": ts,
        "profile": PROFILE,
        "auto_execute_env": AUTO_EXECUTE,
        "rebalance_plan_path": str(reb_path),
        "rebalance_plan_hash": f"sha256:{plan_hash}",
        "drift": drift,
        "event_risk_severity": sev,
        "cooldown": {"ok": cd_ok, "reason": cd_reason, "last_execute_ts_utc": last_exec_ts or None},
        "bounds": {
            "min_seconds_between_executes": int(bounds.get("min_seconds_between_executes", 0)),
            "max_total_turnover_per_execute": float(bounds.get("max_total_turnover_per_execute", 0.0)),
            "max_abs_delta_weight_per_symbol": float(bounds.get("max_abs_delta_weight_per_symbol", 0.0)),
            "min_drift_to_execute": bounds.get("min_drift_to_execute"),
            "block_if_event_risk_severity_in": bounds.get("block_if_event_risk_severity_in"),
        },
        "checks": {
            "drift_ok": drift_ok,
            "drift_reasons": drift_reasons,
            "caps_ok": caps_ok,
            "caps_reasons": caps_reasons,
        },
        "receipts_count": len(receipts),
    }

    # In preview, always print decision + receipts head
    if not execute:
        decision["would_execute"] = bool(AUTO_EXECUTE and drift_ok and caps_ok and cd_ok)
        decision["receipts_preview"] = receipts[:5]
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0

    # Execute requires env flag ON
    if not AUTO_EXECUTE:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": "AUTO_EXECUTE_DISABLED"}, sort_keys=True))
        return 0

    # Must pass drift/caps/cooldown
    if not drift_ok:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": "DRIFT_NOT_MET"}, sort_keys=True))
        return 0
    if not caps_ok:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": "CAPS_NOT_MET"}, sort_keys=True))
        return 0
    if not cd_ok:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": "COOLDOWN_NOT_MET"}, sort_keys=True))
        return 0

    # LiveGate gate for paper
    try:
        lg = fetch_live_gate()
    except Exception as exc:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": f"LIVE_GATE_UNREACHABLE:{type(exc).__name__}"}, sort_keys=True))
        return 0

    allow_paper = bool(lg.get("allow_ibkr_paper", False))
    live_enabled = bool((lg.get("mode") or {}).get("live_enabled", False)) if isinstance(lg.get("mode"), dict) else False
    if not allow_paper:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": "LIVE_GATE_DENY_PAPER"}, sort_keys=True))
        return 0
    if live_enabled:
        print(json.dumps({**decision, "ok": False, "blocked": True, "reason": "LIVE_MODE_NOT_ALLOWED_IN_PAPER_EXECUTOR"}, sort_keys=True))
        return 0

    # EXECUTE: idempotent mark + write report
    repo_root = Path(__file__).resolve().parents[1]
    store = IdempotencyStore(default_paper_db_path(repo_root), table=IDEMP_TABLE)

    for r in receipts:
        payload_hash = sha256_hex_bytes(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        res = store.mark_once(r["receipt_id"], payload_hash, meta={"source": "rebalance_auto_executor_paper", "plan_hash": plan_hash})
        r["idempotency"] = {"inserted": res.inserted, "reason": res.reason}
        r["status"] = "recorded" if res.inserted else "duplicate_skipped"

    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RECEIPTS_DIR / f"REBALANCE_RECEIPTS_{PROFILE}_{utc_now_compact()}.json"
    report = {**decision, "ok": True, "blocked": False, "live_gate": lg, "receipts": receipts}
    atomic_write_json(out, report)

    # update autonomy state pointer (last execute)
    atomic_write_json(AUTONOMY_STATE_PATH, {
        "ts_utc": ts,
        "ttl_seconds": 86400,
        "schema_version": "rebalance_autonomy_state.v1",
        "profile": PROFILE,
        "last_execute_ts_utc": ts,
        "last_rebalance_plan_hash": f"sha256:{plan_hash}",
        "last_receipts_path": str(out),
        "notes": "paper_only_autonomy_scaffold",
    })

    print(json.dumps({"ok": True, "mode": "EXECUTE", "out": str(out), "receipts_count": len(receipts), "ts_utc": ts}, sort_keys=True))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 12 bounded rebalance auto-executor (paper receipts).")
    ap.add_argument("--execute", action="store_true", help="Write receipts if all gates + bounds pass.")
    args = ap.parse_args(argv)
    return run(execute=bool(args.execute))


if __name__ == "__main__":
    raise SystemExit(main())

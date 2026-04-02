#!/usr/bin/env python3
"""
CHAD — Phase 9.1 Exit-Only Consumer (PAPER receipts; no broker calls)

File: chad/execution/exit_only_consumer_paper.py

Mission
-------
Consume an immutable exit-only plan artifact and emit PAPER receipts with exactly-once semantics.

Hard Guarantees
---------------
- NO broker calls. NO orders. NO portfolio changes.
- PREVIEW mode (default): writes nothing.
- EXECUTE mode (--execute): fail-closed unless *all* gates satisfied; writes receipts idempotently.
- Report emission is idempotent by default (stable filename per plan_hash), preventing file inflation.

Inputs (env-overridable)
------------------------
- CHAD_RUNTIME_DIR (default: /home/ubuntu/chad_finale/runtime)
- CHAD_EXIT_ONLY_PLAN_PATH (default: <runtime>/exit_only_plan.json)
- CHAD_STOP_STATE_PATH (default: <runtime>/stop_state.json)
- CHAD_LIVE_GATE_URL (default: http://127.0.0.1:9618/live-gate)

Outputs (EXECUTE only)
----------------------
- reports: JSON file (atomic) to CHAD_EXIT_ONLY_REPORTS_DIR
- receipts: sqlite via IdempotencyStore(table=CHAD_EXIT_ONLY_IDEMP_TABLE)

Exit Codes
----------
0 = success OR clean blocked (fail-closed) OR preview
2 = internal error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

from chad.execution.idempotency_store import IdempotencyStore, default_paper_db_path


# -----------------------------
# Configuration (env-overridable)
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
DEFAULT_REPORTS_DIR = Path("/home/ubuntu/chad_finale/reports/exit_only")

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
PLAN_PATH = Path(os.environ.get("CHAD_EXIT_ONLY_PLAN_PATH", str(RUNTIME_DIR / "exit_only_plan.json")))
STOP_PATH = Path(os.environ.get("CHAD_STOP_STATE_PATH", str(RUNTIME_DIR / "stop_state.json")))
LIVE_GATE_URL = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")

REPORTS_DIR = Path(os.environ.get("CHAD_EXIT_ONLY_REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
IDEMP_TABLE = os.environ.get("CHAD_EXIT_ONLY_IDEMP_TABLE", "exit_only_receipts")

HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "3.0"))

# Report naming:
# - "stable" (default): EXIT_ONLY_RECEIPTS_<plan_hash>.json (idempotent report artifact)
# - "timestamp": EXIT_ONLY_RECEIPTS_<ts>.json (historical trail; may inflate files)
REPORT_FILENAME_MODE = os.environ.get("CHAD_EXIT_ONLY_REPORT_FILENAME_MODE", "stable").strip().lower()

# Best-effort lock to prevent concurrent writers racing on report files
REPORT_LOCK_TTL_S = int(os.environ.get("CHAD_EXIT_ONLY_REPORT_LOCK_TTL_SECONDS", "30"))

USER_AGENT = "chad-exit-only-consumer/phase9.1"


# -----------------------------
# Types
# -----------------------------

@dataclass(frozen=True)
class LiveGateSnapshot:
    operator_mode: str
    operator_reason: str
    allow_exits_only: bool
    allow_ibkr_paper: bool
    allow_ibkr_live: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class StopSnapshot:
    stop: bool
    reason: str


# -----------------------------
# Deterministic + crash-safe utilities
# -----------------------------

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_now_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _is_finite_float(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_key(s: str) -> str:
    safe = "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip("._-")
    return safe or "stable"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
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


def _read_json_dict(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_dict:{path}")
    return obj


def _stop_state() -> StopSnapshot:
    """
    Fail-closed: if stop_state is unreadable, treat STOP as engaged.
    """
    try:
        obj = _read_json_dict(STOP_PATH)
        return StopSnapshot(stop=bool(obj.get("stop", False)), reason=str(obj.get("reason") or ""))
    except Exception as exc:
        return StopSnapshot(stop=True, reason=f"stop_state_unreadable:{type(exc).__name__}")


def _fetch_live_gate() -> LiveGateSnapshot:
    req = Request(LIVE_GATE_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        raw = resp.read()
    obj = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("live_gate_invalid_shape")
    return LiveGateSnapshot(
        operator_mode=str(obj.get("operator_mode") or ""),
        operator_reason=str(obj.get("operator_reason") or ""),
        allow_exits_only=bool(obj.get("allow_exits_only", False)),
        allow_ibkr_paper=bool(obj.get("allow_ibkr_paper", False)),
        allow_ibkr_live=bool(obj.get("allow_ibkr_live", False)),
        reasons=tuple(obj.get("reasons") or ()),
    )


def _gate_allows_execute(lg: LiveGateSnapshot) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if lg.operator_mode.strip().upper() != "EXIT_ONLY":
        reasons.append(f"operator_mode_not_exit_only:{lg.operator_mode!r}")
    if not lg.allow_exits_only:
        reasons.append("live_gate_allow_exits_only_false")
    if lg.allow_ibkr_paper or lg.allow_ibkr_live:
        reasons.append("live_gate_entry_lanes_not_disabled")
    return (len(reasons) == 0), reasons


def _plan_hash(plan: Dict[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json(plan).encode("utf-8"))


def _receipt_id(plan_hash: str, symbol: str, side: str, qty: float) -> str:
    payload = f"{plan_hash}|{symbol}|{side}|{qty:.10f}".encode("utf-8")
    return _sha256_hex_bytes(payload)


def _payload_hash(payload: Dict[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


def _validate_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    schema = str(plan.get("schema_version") or "")
    if schema != "phase9.1.exit_only_plan.v1":
        raise ValueError(f"unexpected_schema_version:{schema!r}")

    exits = plan.get("exits")
    if not isinstance(exits, list):
        raise ValueError("exits_not_list")

    out: List[Dict[str, Any]] = []
    for e in exits:
        if not isinstance(e, dict):
            continue
        sym = str(e.get("symbol") or "").strip().upper()
        side = str(e.get("side") or "").strip().upper()
        qty_raw = e.get("qty")
        try:
            qty = float(qty_raw or 0.0)
        except Exception:
            raise ValueError(f"exit_bad_qty_type:{sym}:{qty_raw!r}") from None

        if not sym:
            raise ValueError("exit_missing_symbol")
        if side not in ("SELL", "BUY"):
            raise ValueError(f"exit_bad_side:{side!r}")
        if not _is_finite_float(qty) or qty <= 0.0:
            raise ValueError(f"exit_bad_qty:{sym}:{qty}")

        out.append(e)

    if not out:
        raise ValueError("no_exits_in_plan")

    return out


def _acquire_report_lock(lock_path: Path, ttl_s: int) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()

    if lock_path.exists():
        try:
            age = now - lock_path.stat().st_mtime
            if age < float(ttl_s):
                return False
        except Exception:
            return False
        try:
            lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            return False

    try:
        lock_path.write_text(_canonical_json({"pid": os.getpid(), "ts_utc": _utc_now_iso()}), encoding="utf-8")
        return True
    except Exception:
        return False


def _release_report_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def _report_path(*, plan_hash: Optional[str], blocked: bool) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if blocked or REPORT_FILENAME_MODE == "timestamp" or not plan_hash:
        return REPORTS_DIR / f"EXIT_ONLY_RECEIPTS_{_utc_now_compact()}.json"
    return REPORTS_DIR / f"EXIT_ONLY_RECEIPTS_{_safe_key(plan_hash)}.json"


def _write_report(report: Dict[str, Any], *, plan_hash: Optional[str], blocked: bool) -> Path:
    out = _report_path(plan_hash=plan_hash, blocked=blocked)
    _atomic_write_text(out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out


# -----------------------------
# Main execution
# -----------------------------

def run(*, execute: bool) -> int:
    ts = _utc_now_iso()

    stop = _stop_state()
    if stop.stop:
        out: Dict[str, Any] = {
            "ok": False,
            "blocked": True,
            "mode": "EXECUTE" if execute else "PREVIEW",
            "reason": f"STOP_ENGAGED:{stop.reason}",
            "ts_utc": ts,
        }
        if execute:
            report = {
                "schema_version": "phase9.1.exit_only_receipts.v1",
                "ts_utc": ts,
                "mode": "EXECUTE",
                "blocked": True,
                "reason": out["reason"],
                "plan_path": str(PLAN_PATH),
                "receipts": [],
            }
            rp = _write_report(report, plan_hash=None, blocked=True)
            out["report_path"] = str(rp)
        print(_canonical_json(out))
        return 0

    if not PLAN_PATH.is_file():
        print(_canonical_json({
            "ok": False,
            "blocked": True,
            "mode": "EXECUTE" if execute else "PREVIEW",
            "reason": f"missing_exit_only_plan:{str(PLAN_PATH)}",
            "ts_utc": ts,
        }))
        return 0

    plan = _read_json_dict(PLAN_PATH)
    exits = _validate_plan(plan)
    plan_h = _plan_hash(plan)

    live_gate: Optional[LiveGateSnapshot] = None
    if execute:
        try:
            live_gate = _fetch_live_gate()
        except Exception as exc:
            report = {
                "schema_version": "phase9.1.exit_only_receipts.v1",
                "ts_utc": ts,
                "mode": "EXECUTE",
                "blocked": True,
                "reason": f"live_gate_unreachable:{type(exc).__name__}",
                "plan_hash": plan_h,
                "plan_path": str(PLAN_PATH),
                "receipts": [],
            }
            rp = _write_report(report, plan_hash=None, blocked=True)
            print(_canonical_json({
                "ok": False,
                "blocked": True,
                "mode": "EXECUTE",
                "reason": report["reason"],
                "report_path": str(rp),
                "ts_utc": ts,
            }))
            return 0

        ok, gate_reasons = _gate_allows_execute(live_gate)
        if not ok:
            report = {
                "schema_version": "phase9.1.exit_only_receipts.v1",
                "ts_utc": ts,
                "mode": "EXECUTE",
                "blocked": True,
                "reason": " | ".join(gate_reasons),
                "plan_hash": plan_h,
                "plan_path": str(PLAN_PATH),
                "live_gate": {
                    "operator_mode": live_gate.operator_mode,
                    "operator_reason": live_gate.operator_reason,
                    "allow_exits_only": live_gate.allow_exits_only,
                    "allow_ibkr_paper": live_gate.allow_ibkr_paper,
                    "allow_ibkr_live": live_gate.allow_ibkr_live,
                    "reasons": list(live_gate.reasons),
                },
                "receipts": [],
            }
            rp = _write_report(report, plan_hash=None, blocked=True)
            print(_canonical_json({
                "ok": False,
                "blocked": True,
                "mode": "EXECUTE",
                "reason": report["reason"],
                "report_path": str(rp),
                "ts_utc": ts,
            }))
            return 0

    receipts: List[Dict[str, Any]] = []
    store: Optional[IdempotencyStore] = None
    if execute:
        repo_root = Path(__file__).resolve().parents[2]
        store = IdempotencyStore(default_paper_db_path(repo_root), table=IDEMP_TABLE)

    for e in exits:
        sym = str(e.get("symbol") or "").strip().upper()
        side = str(e.get("side") or "").strip().upper()
        qty = float(e.get("qty") or 0.0)

        rid = _receipt_id(plan_h, sym, side, qty)
        payload: Dict[str, Any] = {
            "receipt_id": rid,
            "ts_utc": ts,
            "schema_version": "phase9.1.exit_only_receipt.v1",
            "plan_hash": plan_h,
            "symbol": sym,
            "side": side,
            "qty": qty,
            "currency": str(e.get("currency") or "USD"),
            "asset_class": str(e.get("asset_class") or ""),
            "reason": str(e.get("reason") or ""),
            "paper": True,
            "note": "receipt_only_no_broker_calls",
        }
        ph = _payload_hash(payload)

        if execute and store is not None:
            res = store.mark_once(rid, ph, meta={
                "source": "exit_only_consumer_paper",
                "plan_hash": plan_h,
                "symbol": sym,
                "side": side,
                "qty": qty,
            })
            payload["idempotency"] = {"inserted": bool(res.inserted), "reason": str(res.reason)}
            payload["status"] = "recorded" if res.inserted else "duplicate_skipped"
        else:
            payload["status"] = "preview"

        receipts.append(payload)

    report: Dict[str, Any] = {
        "schema_version": "phase9.1.exit_only_receipts.v1",
        "ts_utc": ts,
        "mode": "EXECUTE" if execute else "PREVIEW",
        "blocked": False,
        "plan_path": str(PLAN_PATH),
        "plan_hash": plan_h,
        "receipts_count": len(receipts),
        "receipts": receipts,
    }
    if live_gate is not None:
        report["live_gate"] = {
            "operator_mode": live_gate.operator_mode,
            "operator_reason": live_gate.operator_reason,
            "allow_exits_only": live_gate.allow_exits_only,
            "allow_ibkr_paper": live_gate.allow_ibkr_paper,
            "allow_ibkr_live": live_gate.allow_ibkr_live,
            "reasons": list(live_gate.reasons),
        }

    if not execute:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    lock_path = REPORTS_DIR / ".exit_only_report.lock"
    if not _acquire_report_lock(lock_path, REPORT_LOCK_TTL_S):
        print(_canonical_json({
            "ok": True,
            "blocked": False,
            "mode": "EXECUTE",
            "receipts_count": len(receipts),
            "ts_utc": ts,
            "note": "report_write_skipped_locked",
        }))
        return 0

    try:
        report_path = _write_report(report, plan_hash=plan_h, blocked=False)
    finally:
        _release_report_lock(lock_path)

    print(_canonical_json({
        "ok": True,
        "mode": "EXECUTE",
        "blocked": False,
        "report_path": str(report_path),
        "receipts_count": len(receipts),
        "ts_utc": ts,
    }))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 9.1 Exit-Only Consumer (paper receipts; no broker calls)."
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Write idempotent receipts + report (requires EXIT_ONLY gates). Default: preview (no writes).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    try:
        return run(execute=bool(args.execute))
    except Exception as exc:
        print(_canonical_json({
            "ok": False,
            "mode": "EXECUTE" if bool(args.execute) else "PREVIEW",
            "blocked": True,
            "reason": f"internal_error:{type(exc).__name__}:{str(exc)}",
            "ts_utc": _utc_now_iso(),
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
CHAD — Phase 9.1 Exit-Only Consumer (PAPER receipts; no broker calls)

File: chad/execution/exit_only_consumer_paper.py

Purpose
-------
Consume runtime/exit_only_plan.json and produce PAPER receipts for each exit intent.
This proves:
- deterministic consumption of the exit-only plan artifact
- gating discipline (STOP + LiveGate + EXIT_ONLY only)
- exactly-once semantics for receipts (no duplicates across retries/reboots)
- auditable artifacts (report JSON + sqlite receipts table)

Hard Guarantees
---------------
- NO broker calls. NO orders. NO portfolio changes.
- PREVIEW mode (default):
    - Always allowed
    - Writes nothing (no sqlite writes, no report file writes)
- EXECUTE mode (--execute):
    - Fail-closed unless all gates are satisfied:
        * STOP is false
        * LiveGate allow_exits_only == True
        * LiveGate operator_mode == EXIT_ONLY
        * LiveGate allow_ibkr_paper == False
        * LiveGate allow_ibkr_live == False
    - Writes receipts idempotently to sqlite (exactly-once) and emits an atomic report JSON.

Runtime Inputs
--------------
- /home/ubuntu/CHAD FINALE/runtime/exit_only_plan.json
- /home/ubuntu/CHAD FINALE/runtime/stop_state.json
- http://127.0.0.1:9618/live-gate  (EXECUTE only; PREVIEW does not require network)

Outputs (EXECUTE only)
----------------------
- reports/exit_only/EXIT_ONLY_RECEIPTS_<ts>.json
- runtime/exec_state_paper.sqlite3 table: exit_only_receipts

Exit Codes
----------
0 = success OR clean blocked (fail-closed) OR preview
2 = internal error (invalid plan shape, IO failure, etc.)
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
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

from chad.execution.idempotency_store import IdempotencyStore, default_paper_db_path

# -----------------------------
# Configuration (env-overridable)
# -----------------------------

DEFAULT_RUNTIME_DIR = Path("/home/ubuntu/CHAD FINALE/runtime")
DEFAULT_REPORTS_DIR = Path("/home/ubuntu/CHAD FINALE/reports/exit_only")

RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
PLAN_PATH = Path(os.environ.get("CHAD_EXIT_ONLY_PLAN_PATH", str(RUNTIME_DIR / "exit_only_plan.json")))
STOP_PATH = Path(os.environ.get("CHAD_STOP_STATE_PATH", str(RUNTIME_DIR / "stop_state.json")))
LIVE_GATE_URL = os.environ.get("CHAD_LIVE_GATE_URL", "http://127.0.0.1:9618/live-gate")

REPORTS_DIR = Path(os.environ.get("CHAD_EXIT_ONLY_REPORTS_DIR", str(DEFAULT_REPORTS_DIR)))
IDEMP_TABLE = os.environ.get("CHAD_EXIT_ONLY_IDEMP_TABLE", "exit_only_receipts")

HTTP_TIMEOUT_S = float(os.environ.get("CHAD_HTTP_TIMEOUT_S", "3.0"))


# -----------------------------
# Types + Small utilities
# -----------------------------

@dataclass(frozen=True)
class LiveGateSnapshot:
    operator_mode: str
    operator_reason: str
    allow_exits_only: bool
    allow_ibkr_paper: bool
    allow_ibkr_live: bool
    reasons: Tuple[str, ...]


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_now_compact() -> str:
    # Safe filename time: YYYYMMDDTHHMMSSZ
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _is_finite_float(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _sha256_hex_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _canonical_json(obj: Any) -> str:
    # Deterministic for hashing
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    # best-effort dir fsync for crash safety
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
        raise ValueError(f"json_not_dict: {path}")
    return obj


def _stop_state() -> Tuple[bool, str]:
    """
    Fail-closed: if stop_state is unreadable, treat STOP as engaged.
    """
    try:
        obj = _read_json_dict(STOP_PATH)
        stop = bool(obj.get("stop", False))
        reason = str(obj.get("reason") or "")
        return stop, reason
    except Exception as exc:
        return True, f"stop_state_unreadable:{type(exc).__name__}"


def _fetch_live_gate() -> LiveGateSnapshot:
    req = Request(LIVE_GATE_URL, headers={"User-Agent": "chad-exit-only-consumer/1.0"})
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
    # In exits-only, entry lanes must be disabled.
    if lg.allow_ibkr_paper or lg.allow_ibkr_live:
        reasons.append("live_gate_entry_lanes_not_disabled")
    return (len(reasons) == 0), reasons


def _validate_plan(plan: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (schema_version, exits_list) or raises ValueError.
    """
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
        qty = float(e.get("qty") or 0.0)

        if not sym:
            raise ValueError("exit_missing_symbol")
        if side not in ("SELL", "BUY"):
            raise ValueError(f"exit_bad_side:{side!r}")
        if not _is_finite_float(qty) or qty <= 0.0:
            raise ValueError(f"exit_bad_qty:{sym}:{qty}")

        out.append(e)

    if not out:
        raise ValueError("no_exits_in_plan")

    return schema, out


def _plan_hash(plan: Dict[str, Any]) -> str:
    # hash canonical JSON of the plan dict (deterministic)
    return _sha256_hex_bytes(_canonical_json(plan).encode("utf-8"))


def _receipt_id(plan_hash: str, symbol: str, side: str, qty: float) -> str:
    """
    Stable receipt id: derived from plan_hash + (symbol, side, qty).
    No wall-clock dependence => safe across reboots/retries.
    """
    base = f"{plan_hash}|{symbol.upper()}|{side.upper()}|{qty:.8f}"
    return _sha256_hex_bytes(base.encode("utf-8"))


def _payload_hash(payload: Dict[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


# -----------------------------
# Main runner
# -----------------------------

def run(*, execute: bool) -> int:
    ts = _utc_now_iso()

    # Stop gate (always checked, even preview)
    stop, stop_reason = _stop_state()
    if stop:
        out = {
            "ok": False,
            "blocked": True,
            "mode": "EXECUTE" if execute else "PREVIEW",
            "reason": f"STOP_ENGAGED:{stop_reason}",
            "ts_utc": ts,
        }
        # In execute mode we still write a report (audit trail). In preview, we don't write.
        if execute:
            report_path = _write_report({
                "schema_version": "phase9.1.exit_only_receipts.v1",
                "ts_utc": ts,
                "mode": "EXECUTE",
                "blocked": True,
                "reason": out["reason"],
                "plan_path": str(PLAN_PATH),
                "receipts": [],
            })
            out["report_path"] = str(report_path)
        print(_canonical_json(out))
        return 0

    # Load & validate plan
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
    schema_version, exits = _validate_plan(plan)
    plan_h = _plan_hash(plan)

    # Gate check only for EXECUTE
    live_gate: Optional[LiveGateSnapshot] = None
    if execute:
        try:
            live_gate = _fetch_live_gate()
        except Exception as exc:
            report_path = _write_report({
                "schema_version": "phase9.1.exit_only_receipts.v1",
                "ts_utc": ts,
                "mode": "EXECUTE",
                "blocked": True,
                "reason": f"live_gate_unreachable:{type(exc).__name__}",
                "plan_hash": plan_h,
                "plan_path": str(PLAN_PATH),
                "receipts": [],
            })
            print(_canonical_json({
                "ok": False,
                "blocked": True,
                "mode": "EXECUTE",
                "reason": f"live_gate_unreachable:{type(exc).__name__}",
                "report_path": str(report_path),
                "ts_utc": ts,
            }))
            return 0

        ok, reasons = _gate_allows_execute(live_gate)
        if not ok:
            report_path = _write_report({
                "schema_version": "phase9.1.exit_only_receipts.v1",
                "ts_utc": ts,
                "mode": "EXECUTE",
                "blocked": True,
                "reason": " | ".join(reasons),
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
            })
            print(_canonical_json({
                "ok": False,
                "blocked": True,
                "mode": "EXECUTE",
                "reason": " | ".join(reasons),
                "report_path": str(report_path),
                "ts_utc": ts,
            }))
            return 0

    # Build receipts
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
            payload["idempotency"] = {"inserted": res.inserted, "reason": res.reason}
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

    if execute:
        report_path = _write_report(report)
        print(_canonical_json({
            "ok": True,
            "mode": "EXECUTE",
            "blocked": False,
            "report_path": str(report_path),
            "receipts_count": len(receipts),
            "ts_utc": ts,
        }))
    else:
        # PREVIEW prints the full report to stdout and writes nothing
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _write_report(report: Dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"EXIT_ONLY_RECEIPTS_{_utc_now_compact()}.json"
    _atomic_write_text(out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Phase 9.1 Exit-Only Consumer (paper receipts; no broker calls)."
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Write idempotent receipts + report (requires EXIT_ONLY gates). Default: preview (no writes).",
    )
    args = ap.parse_args(argv)
    try:
        return run(execute=bool(args.execute))
    except Exception as exc:
        # Fail-safe: never crash silently
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

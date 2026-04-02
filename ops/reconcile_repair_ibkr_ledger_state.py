#!/usr/bin/env python3
"""
ops/reconcile_repair_ibkr_ledger_state.py

PHASE 12 — Ledger State Repair (Broker Snapshot Authority)
Production-grade, deterministic, fail-closed.

What it does
- Reads broker truth: runtime/positions_snapshot.json
- Reads CHAD ledger state: runtime/ibkr_paper_ledger_state.json
- Produces a *surgical* repair so ledger_state.open[*].qty matches broker snapshot qty
  for matching conId+symbol records.
- Writes:
    1) runtime/ibkr_paper_ledger_state.json   (atomic, repaired)
    2) reports/reconciliation/REPAIR_LEDGER_STATE_<ts>.json  (immutable report)

Why this is safe
- It does NOT place trades.
- It does NOT edit broker positions.
- It only repairs CHAD’s internal “belief file” so reconciliation can move to GREEN.
- It leaves an audit trail explaining exactly what changed.

Fail-closed policy
- If inputs are missing/corrupt/unexpected -> no write, non-zero exit.
- If a record is ambiguous (multiple matches) -> no write, non-zero exit.
- If broker snapshot has a position that does not exist in ledger_state.open -> report it
  but do not invent entries (that belongs to a separate “import missing position” workflow).

Exit codes
- 0: repair completed (or no changes needed)
- 2: input error / unsafe to proceed
- 3: write failure
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json_dict(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:
        return None, f"read_failed:{type(exc).__name__}"

    try:
        obj = json.loads(raw)
    except Exception as exc:
        return None, f"json_parse_failed:{type(exc).__name__}"

    if not isinstance(obj, dict):
        return None, "json_not_object"
    return obj, None


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def repo_root() -> Path:
    env = (os.environ.get("CHAD_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "runtime").is_dir() and (p / "chad").is_dir():
            return p

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "runtime").is_dir() and (parent / "chad").is_dir():
            return parent

    for c in (Path("/home/ubuntu/chad_finale"), Path("/home/ubuntu/chad_finale")):
        if (c / "runtime").is_dir() and (c / "chad").is_dir():
            return c

    return Path.cwd().resolve()


@dataclass(frozen=True)
class BrokerPos:
    conId: int
    symbol: str
    qty: float


def load_broker_positions(path: Path) -> Tuple[Dict[Tuple[int, str], BrokerPos], Optional[str]]:
    obj, err = read_json_dict(path)
    if obj is None:
        return {}, err

    positions = obj.get("positions")
    if not isinstance(positions, list):
        return {}, "missing_positions_list"

    out: Dict[Tuple[int, str], BrokerPos] = {}
    for p in positions:
        if not isinstance(p, dict):
            continue
        conId = safe_int(p.get("conId"), 0)
        sym = str(p.get("symbol") or "").strip()
        qty = safe_float(p.get("position"), 0.0)
        if conId <= 0 or not sym:
            continue
        out[(conId, sym)] = BrokerPos(conId=conId, symbol=sym, qty=float(qty))
    return out, None


def load_ledger_open(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    obj, err = read_json_dict(path)
    if obj is None:
        return {}, err
    open_ = obj.get("open")
    if not isinstance(open_, dict):
        return {}, "missing_open_dict"
    # must be dict[str, dict]
    for k, v in open_.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            return {}, "open_dict_bad_shape"
    return open_, None


def main() -> int:
    root = repo_root()
    runtime = root / "runtime"
    reports = root / "reports" / "reconciliation"
    reports.mkdir(parents=True, exist_ok=True)

    broker_path = runtime / "positions_snapshot.json"
    ledger_path = runtime / "ibkr_paper_ledger_state.json"

    ts = utc_now_iso()
    report_path = reports / f"REPAIR_LEDGER_STATE_{ts.replace(':', '').replace('+', '').replace('-', '')}.json"

    broker_map, berr = load_broker_positions(broker_path)
    if berr:
        print(f"FATAL: broker positions load failed: {berr}")
        return 2

    ledger_obj, lerr = read_json_dict(ledger_path)
    if ledger_obj is None:
        print(f"FATAL: ledger state load failed: {lerr}")
        return 2

    open_map, oerr = load_ledger_open(ledger_path)
    if oerr:
        print(f"FATAL: ledger open load failed: {oerr}")
        return 2

    # Build index from ledger open by (conId, symbol) -> key
    idx: Dict[Tuple[int, str], str] = {}
    collisions: List[Dict[str, Any]] = []
    for key, rec in open_map.items():
        conId = safe_int(rec.get("conId"), 0)
        sym = str(rec.get("symbol") or "").strip()
        if conId <= 0 or not sym:
            continue
        k = (conId, sym)
        if k in idx:
            collisions.append({"type": "LEDGER_DUPLICATE_KEY", "conId": conId, "symbol": sym, "keys": [idx[k], key]})
        else:
            idx[k] = key

    if collisions:
        atomic_write_json(
            report_path,
            {
                "ts_utc": ts,
                "status": "ABORTED",
                "reason": "ledger_open_contains_duplicates_for_same_conId_symbol",
                "collisions": collisions,
                "broker_source": str(broker_path),
                "ledger_source": str(ledger_path),
            },
        )
        print("FATAL: ledger open contains duplicates; wrote report:", report_path)
        return 2

    changes: List[Dict[str, Any]] = []
    missing_in_ledger: List[Dict[str, Any]] = []

    # Apply repairs only when ledger has a matching record.
    for k, bp in sorted(broker_map.items(), key=lambda x: (x[0][0], x[0][1])):
        if k not in idx:
            missing_in_ledger.append({"conId": bp.conId, "symbol": bp.symbol, "broker_qty": bp.qty})
            continue
        led_key = idx[k]
        rec = open_map[led_key]
        old_qty = safe_float(rec.get("qty"), 0.0)
        new_qty = float(bp.qty)

        if abs(old_qty - new_qty) > 0.0:
            rec["qty"] = new_qty
            rec["repaired_from_broker_snapshot_utc"] = ts
            rec["repair_note"] = "qty updated to match runtime/positions_snapshot.json (broker truth)"
            changes.append(
                {
                    "conId": bp.conId,
                    "symbol": bp.symbol,
                    "ledger_key": led_key,
                    "old_qty": old_qty,
                    "new_qty": new_qty,
                }
            )

    # Write report (always)
    report_payload: Dict[str, Any] = {
        "ts_utc": ts,
        "status": "OK",
        "broker_source": str(broker_path),
        "ledger_source": str(ledger_path),
        "changes_applied": len(changes),
        "missing_in_ledger": missing_in_ledger,
        "changes": changes,
        "notes": {
            "policy": "repair_only_existing_open_records; do_not_invent_positions",
            "safety": "no broker calls; internal state repair only",
        },
    }
    atomic_write_json(report_path, report_payload)

    # If no changes, do not rewrite ledger file (keeps filesystem quiet).
    if not changes:
        print("OK: no ledger qty changes needed. report:", report_path)
        return 0

    # Update top-level metadata
    ledger_obj["last_repair_utc"] = ts
    ledger_obj["last_repair_report"] = str(report_path)
    # Ensure ledger_obj['open'] points to the modified open_map
    ledger_obj["open"] = open_map

    try:
        atomic_write_json(ledger_path, ledger_obj)
    except Exception as exc:
        print(f"FATAL: failed writing ledger_state: {type(exc).__name__}: {exc}")
        return 3

    print("OK: repaired ledger_state. report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

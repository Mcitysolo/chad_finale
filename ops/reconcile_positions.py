#!/usr/bin/env python3
"""
ops/reconcile_positions.py

PHASE 12 — Reconciliation Producer (Positions Snapshot vs Ledger State)
Production-grade, deterministic, fail-closed.

Purpose
- Compare broker positions (runtime/positions_snapshot.json) to CHAD ledger-derived state
  (runtime/ibkr_paper_ledger_state.json), emit runtime/reconciliation_state.json.

Why Phase 12 cares
- LiveGate must block LIVE when reconciliation is RED or unknown.
- Reconciliation artifacts must be auditable and correctly reference their sources.
- Must be restart-safe, deterministic, and free of side effects (read-only inputs, atomic output).

Key guarantees
- Writes ONLY: runtime/reconciliation_state.json (atomic write)
- Reads ONLY: runtime/positions_snapshot.json, runtime/ibkr_paper_ledger_state.json
- Always includes: ts_utc + ttl_seconds + broker_source + chad_state_source + counts + mismatches + status
- Fail-closed: any error -> status=RED with reason details (never GREEN by accident)
- Stable output ordering for deterministic diffs

Exit codes
- 0: success (GREEN or RED)
- 2: hard failure writing output (rare; still tries best-effort)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Config
# ----------------------------

DEFAULT_TTL_SECONDS = 300  # 5 minutes (matches timer cadence)
DEFAULT_QTY_TOL = 1e-9     # exact match for share quantities; keep strict

RUNTIME_FILES = {
    "broker_positions": "positions_snapshot.json",
    "chad_state": "ibkr_paper_ledger_state.json",
    "out": "reconciliation_state.json",
}


# ----------------------------
# Helpers (deterministic)
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> str:
    def _default(x: Any) -> str:
        return str(x)

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_default,
    )


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


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def repo_root() -> Path:
    """
    Resolve repo root safely:
    - CHAD_ROOT env
    - walk upward from this file
    - known fallbacks
    """
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


# ----------------------------
# Reconciliation logic
# ----------------------------

@dataclass(frozen=True)
class Pos:
    conId: int
    symbol: str
    qty: float


def load_broker_positions(path: Path) -> Tuple[Dict[int, Pos], Dict[str, Any], Optional[str]]:
    obj, err = read_json_dict(path)
    if obj is None:
        return {}, {}, err

    positions = obj.get("positions")
    if not isinstance(positions, list):
        return {}, obj, "missing_positions_list"

    out: Dict[int, Pos] = {}
    for p in positions:
        if not isinstance(p, dict):
            continue
        conId = safe_int(p.get("conId"), 0)
        sym = str(p.get("symbol") or "").strip()
        qty = safe_float(p.get("position"), 0.0)
        if conId <= 0 or not sym:
            continue
        out[conId] = Pos(conId=conId, symbol=sym, qty=float(qty))
    return out, obj, None


def load_chad_state_positions(path: Path) -> Tuple[Dict[int, Pos], Dict[str, Any], Optional[str]]:
    obj, err = read_json_dict(path)
    if obj is None:
        return {}, {}, err

    out: Dict[int, Pos] = {}

    # Shape A: {"positions":[{"conId":..., "symbol":..., "qty":...}, ...]}
    if isinstance(obj.get("positions"), list):
        for p in obj.get("positions"):
            if not isinstance(p, dict):
                continue
            conId = safe_int(p.get("conId"), 0)
            sym = str(p.get("symbol") or "").strip()
            qty = safe_float(p.get("qty", p.get("position", 0.0)), 0.0)
            if conId <= 0 or not sym:
                continue
            out[conId] = Pos(conId=conId, symbol=sym, qty=float(qty))
        return out, obj, None

    # Shape B: {"by_conid":{"123":{"symbol":"AAPL","qty":10}, ...}}
    byc = obj.get("by_conid")
    if isinstance(byc, dict):
        for k, v in byc.items():
            conId = safe_int(k, 0)
            if not isinstance(v, dict):
                continue
            sym = str(v.get("symbol") or "").strip()
            qty = safe_float(v.get("qty", 0.0), 0.0)
            if conId <= 0 or not sym:
                continue
            out[conId] = Pos(conId=conId, symbol=sym, qty=float(qty))
        return out, obj, None

    # Shape C (your real file): {"open": {"ACC::conId::SYM::STK": {conId, symbol, qty, ...}, ...}}
    opened = obj.get("open")
    if isinstance(opened, dict):
        for _, v in opened.items():
            if not isinstance(v, dict):
                continue
            conId = safe_int(v.get("conId"), 0)
            sym = str(v.get("symbol") or "").strip()
            qty = safe_float(v.get("qty", 0.0), 0.0)
            if conId <= 0 or not sym:
                continue
            out[conId] = Pos(conId=conId, symbol=sym, qty=float(qty))
        return out, obj, None

    # Shape D: flat dict keyed by SHA-256 hashes
    # {"hash1": {"conId":..., "symbol":..., "qty":...}, ...}
    if obj and all(isinstance(v, dict) and "conId" in v for v in obj.values()):
        for v in obj.values():
            conId = safe_int(v.get("conId"), 0)
            sym = str(v.get("symbol") or "").strip()
            qty = safe_float(v.get("qty", 0.0), 0.0)
            if conId <= 0 or not sym:
                continue
            out[conId] = Pos(conId=conId, symbol=sym, qty=float(qty))
        return out, obj, None

    return out, obj, "unknown_chad_state_shape"


def reconcile(
    broker: Dict[int, Pos],
    chad: Dict[int, Pos],
    *,
    qty_tol: float,
) -> List[Dict[str, Any]]:
    """
    Returns a deterministic list of mismatch dicts.
    Types:
      - MISSING_IN_BROKER
      - MISSING_IN_CHAD
      - QTY_MISMATCH
      - SYMBOL_MISMATCH
    """
    mismatches: List[Dict[str, Any]] = []

    all_ids = sorted(set(broker.keys()) | set(chad.keys()))
    for conId in all_ids:
        b = broker.get(conId)
        c = chad.get(conId)

        if b is None and c is not None:
            mismatches.append({"type": "MISSING_IN_BROKER", "conId": conId, "symbol": c.symbol, "chad_qty": c.qty})
            continue
        if c is None and b is not None:
            mismatches.append({"type": "MISSING_IN_CHAD", "conId": conId, "symbol": b.symbol, "broker_qty": b.qty})
            continue
        if b is None or c is None:
            continue

        if b.symbol != c.symbol:
            mismatches.append({"type": "SYMBOL_MISMATCH", "conId": conId, "broker_symbol": b.symbol, "chad_symbol": c.symbol})

        if abs(float(b.qty) - float(c.qty)) > float(qty_tol):
            mismatches.append({"type": "QTY_MISMATCH", "conId": conId, "symbol": b.symbol, "broker_qty": b.qty, "chad_qty": c.qty})

    return mismatches


def main() -> int:
    root = repo_root()
    runtime = root / "runtime"
    broker_path = runtime / RUNTIME_FILES["broker_positions"]
    chad_path = runtime / RUNTIME_FILES["chad_state"]
    out_path = runtime / RUNTIME_FILES["out"]

    ts = utc_now_iso()

    broker_map, broker_obj, broker_err = load_broker_positions(broker_path)
    chad_map, chad_obj, chad_err = load_chad_state_positions(chad_path)

    errors: List[str] = []
    if broker_err:
        errors.append(f"broker_positions:{broker_err}")
    if chad_err:
        errors.append(f"chad_state:{chad_err}")

    mismatches: List[Dict[str, Any]] = []
    status = "RED"

    if errors:
        mismatches = [{"type": "INPUT_ERROR", "details": e} for e in errors]
        status = "RED"
    else:
        mismatches = reconcile(broker_map, chad_map, qty_tol=DEFAULT_QTY_TOL)
        status = "GREEN" if len(mismatches) == 0 else "RED"

    payload: Dict[str, Any] = {
        "ts_utc": ts,
        "ttl_seconds": int(DEFAULT_TTL_SECONDS),
        "status": status,
        "broker_source": str(broker_path),
        "chad_state_source": str(chad_path),
        "counts": {
            "broker_positions": int(len(broker_map)),
            "chad_state_positions": int(len(chad_map)),
            "mismatches": int(len(mismatches)),
        },
        "mismatches": mismatches,
        # extra debug (bounded)
        "notes": {
            "producer": "ops/reconcile_positions.py",
            "repo_root": str(root),
        },
    }

    try:
        atomic_write_json(out_path, payload)
    except Exception as exc:
        # last resort: print error and exit 2
        print(f"FATAL: failed to write {out_path}: {type(exc).__name__}: {exc}")
        return 2

    # Always success exit code so timer doesn’t spam; status is in JSON.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

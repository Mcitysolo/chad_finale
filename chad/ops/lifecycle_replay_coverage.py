#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

ABS_QTY_TOL = 1e-6


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _nonzero(x: float) -> bool:
    return abs(float(x)) > ABS_QTY_TOL


def _load_raw(repo_root: Path) -> Dict[str, Any]:
    runtime_dir = repo_root / "runtime"
    return {
        "snapshot": _read_json(runtime_dir / "positions_snapshot.json"),
        "ledger": _read_json(runtime_dir / "ibkr_paper_ledger_state.json"),
        "replay": _read_json(runtime_dir / "lifecycle_replay_state.json"),
    }


def _snapshot_map(snapshot: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in snapshot.get("positions", []):
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        out[sym] = float(row.get("position", 0) or 0)
    return out


def _normalize_ledger_open_records(ledger_state: Any) -> Dict[str, Dict[str, Any]]:
    """Return ledger open records keyed by their original id, regardless of schema.

    Accepts both:
      - Wrapped schema: {"open": {<id>: {"symbol": ..., "qty": ..., ...}, ...}}
      - Flat schema (current writer in chad/portfolio/ibkr_paper_ledger_watcher.py):
          {<id>: {"symbol": ..., "qty": ..., ...}, ...}

    Includes only dict records with symbol present, qty present, and float(qty) != 0.0.
    """
    if not isinstance(ledger_state, dict):
        return {}

    candidate: Any = ledger_state.get("open")
    if not isinstance(candidate, dict):
        candidate = ledger_state

    out: Dict[str, Dict[str, Any]] = {}
    for key, row in candidate.items():
        if not isinstance(row, dict):
            continue
        sym = row.get("symbol")
        if not isinstance(sym, str) or not sym.strip():
            continue
        if "qty" not in row:
            continue
        try:
            qty = float(row.get("qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if qty == 0.0:
            continue
        out[str(key)] = row
    return out


def _ledger_meta_map(ledger: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _normalize_ledger_open_records(ledger).values():
        sym = str(row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        out[sym] = {
            "qty": float(row.get("qty", 0) or 0),
            "strategy": str(row.get("strategy") or "").strip().lower(),
            "tags": [str(x).strip().lower() for x in (row.get("tags") or []) if x is not None],
        }
    return out


def _replay_map(replay: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pos = replay.get("positions") or {}
    if not isinstance(pos, dict):
        return out
    for sym, row in pos.items():
        if not isinstance(row, dict):
            continue
        key = str(sym or "").strip().upper()
        if not key:
            continue
        out[key] = float(row.get("qty", 0) or 0)
    return out


def _detect_scope(raw: Dict[str, Any]) -> Dict[str, Any]:
    replay = raw["replay"]
    ledger_meta = _ledger_meta_map(raw["ledger"])

    manual_count = 0
    ibkr_paper_count = 0
    for row in ledger_meta.values():
        if row["strategy"] == "manual":
            manual_count += 1
        if "ibkr_paper" in row["tags"]:
            ibkr_paper_count += 1

    replay_inputs = replay.get("inputs") or {}
    fills_count = int(replay_inputs.get("fills_count", 0) or 0)
    broker_non_heartbeat_count = int(replay_inputs.get("broker_non_heartbeat_count", 0) or 0)
    replay_notes = str(replay.get("notes") or "")

    replay_source = "paper_exec_like" if fills_count > 0 and broker_non_heartbeat_count == 0 else "unknown"
    manual_ibkr_book = manual_count > 0 and ibkr_paper_count > 0

    return {
        "manual_ibkr_book": manual_ibkr_book,
        "manual_count": manual_count,
        "ibkr_paper_count": ibkr_paper_count,
        "replay_source": replay_source,
        "fills_count": fills_count,
        "broker_non_heartbeat_count": broker_non_heartbeat_count,
        "replay_notes": replay_notes,
        "scope_mismatch": manual_ibkr_book and replay_source == "paper_exec_like",
    }


def classify(raw: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_map = _snapshot_map(raw["snapshot"])
    ledger_meta = _ledger_meta_map(raw["ledger"])
    replay_map = _replay_map(raw["replay"])
    scope = _detect_scope(raw)

    if scope["scope_mismatch"]:
        return {
            "status": "SCOPE_MISMATCH_MANUAL_VS_PAPER_EXEC",
            "summary": {
                "snapshot_nonzero_symbols": len([s for s, q in snapshot_map.items() if _nonzero(q)]),
                "replay_nonzero_symbols": len([s for s, q in replay_map.items() if _nonzero(q)]),
                "matched_symbols": 0,
                "missing_from_replay_count": 0,
                "replay_only_count": 0,
                "qty_mismatch_count": 0,
            },
            "missing_from_replay": [],
            "replay_only": [],
            "qty_mismatches": [],
            "scope": scope,
            "snapshot_symbols": sorted(snapshot_map.keys()),
            "replay_symbols": sorted(replay_map.keys()),
            "ledger_symbols": sorted(ledger_meta.keys()),
            "notes": (
                "Replay coverage comparison skipped because runtime snapshot/ledger is an IBKR manual holdings book "
                "while lifecycle replay is derived from paper_exec fills/fees. These are different authority scopes."
            ),
        }

    all_syms = sorted(set(snapshot_map) | set(replay_map))
    missing_from_replay: List[str] = []
    replay_only: List[str] = []
    qty_mismatches: List[Dict[str, Any]] = []
    matched_symbols = 0

    snapshot_nonzero_symbols = 0
    replay_nonzero_symbols = 0

    for sym in all_syms:
        replay_qty = float(replay_map.get(sym, 0.0))
        snapshot_qty = float(snapshot_map.get(sym, 0.0))

        replay_present = _nonzero(replay_qty)
        snapshot_present = _nonzero(snapshot_qty)

        if snapshot_present:
            snapshot_nonzero_symbols += 1
        if replay_present:
            replay_nonzero_symbols += 1

        if snapshot_present and not replay_present:
            missing_from_replay.append(sym)

        if replay_present and not snapshot_present:
            replay_only.append(sym)

        if replay_present and snapshot_present:
            if abs(replay_qty - snapshot_qty) <= ABS_QTY_TOL:
                matched_symbols += 1
            else:
                qty_mismatches.append(
                    {
                        "symbol": sym,
                        "replay_qty": replay_qty,
                        "snapshot_qty": snapshot_qty,
                        "delta": replay_qty - snapshot_qty,
                    }
                )

    if replay_nonzero_symbols == 0:
        status = "NO_REPLAY_COVERAGE"
    elif missing_from_replay:
        status = "PARTIAL_REPLAY_COVERAGE"
    elif replay_only or qty_mismatches:
        status = "REPLAY_MISMATCH"
    else:
        status = "REPLAY_MATCH_CONFIRMED"

    return {
        "status": status,
        "summary": {
            "snapshot_nonzero_symbols": snapshot_nonzero_symbols,
            "replay_nonzero_symbols": replay_nonzero_symbols,
            "matched_symbols": matched_symbols,
            "missing_from_replay_count": len(missing_from_replay),
            "replay_only_count": len(replay_only),
            "qty_mismatch_count": len(qty_mismatches),
        },
        "missing_from_replay": missing_from_replay,
        "replay_only": replay_only,
        "qty_mismatches": qty_mismatches,
        "scope": scope,
        "snapshot_symbols": sorted(snapshot_map.keys()),
        "replay_symbols": sorted(replay_map.keys()),
        "ledger_symbols": sorted(ledger_meta.keys()),
    }


def main() -> int:
    repo_root = Path("/home/ubuntu/chad_finale")
    raw = _load_raw(repo_root)
    result = classify(raw)
    out = {
        "schema_version": "lifecycle_replay_coverage.v3",
        **result,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

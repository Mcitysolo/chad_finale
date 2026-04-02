#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _load_replay(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    from chad.ops.lifecycle_replay_engine import build_replay_state
    obj = build_replay_state(repo_root)
    return obj.get("positions", {}) if isinstance(obj.get("positions"), dict) else {}


def _load_snapshot(runtime_dir: Path) -> Dict[str, Dict[str, Any]]:
    obj = _read_json(runtime_dir / "positions_snapshot.json")
    positions = obj.get("positions", [])
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(positions, list):
        for p in positions:
            if not isinstance(p, dict):
                continue
            sym = str(p.get("symbol") or "").strip().upper()
            if not sym:
                continue
            out[sym] = {
                "qty": float(p.get("position") or 0.0),
                "avg_cost": float(p.get("avgCost") or 0.0),
            }
    return out


def _load_ledger_state(runtime_dir: Path) -> Dict[str, Dict[str, Any]]:
    obj = _read_json(runtime_dir / "ibkr_paper_ledger_state.json")
    open_map = obj.get("open", {})
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(open_map, dict):
        for _, rec in open_map.items():
            if not isinstance(rec, dict):
                continue
            sym = str(rec.get("symbol") or "").strip().upper()
            if not sym:
                continue
            out[sym] = {
                "qty": float(rec.get("qty") or 0.0),
                "avg_cost": float(rec.get("avg_cost") or 0.0),
            }
    return out


def _sym_union(*maps: Dict[str, Dict[str, Any]]) -> list[str]:
    s = set()
    for m in maps:
        s.update(m.keys())
    return sorted(s)


def _row(sym: str, replay: Dict[str, Dict[str, Any]], snapshot: Dict[str, Dict[str, Any]], ledger: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    r = replay.get(sym, {})
    s = snapshot.get(sym, {})
    l = ledger.get(sym, {})

    replay_qty = float(r.get("qty") or 0.0)
    snapshot_qty = float(s.get("qty") or 0.0)
    ledger_qty = float(l.get("qty") or 0.0)

    replay_avg = float(r.get("avg_cost") or 0.0)
    snapshot_avg = float(s.get("avg_cost") or 0.0)
    ledger_avg = float(l.get("avg_cost") or 0.0)

    return {
        "symbol": sym,
        "replay_qty": replay_qty,
        "snapshot_qty": snapshot_qty,
        "ledger_qty": ledger_qty,
        "replay_vs_snapshot_qty_delta": replay_qty - snapshot_qty,
        "replay_vs_ledger_qty_delta": replay_qty - ledger_qty,
        "replay_avg_cost": replay_avg,
        "snapshot_avg_cost": snapshot_avg,
        "ledger_avg_cost": ledger_avg,
        "present_in_replay": sym in replay,
        "present_in_snapshot": sym in snapshot,
        "present_in_ledger": sym in ledger,
    }


def main() -> int:
    repo_root = Path("/home/ubuntu/chad_finale")
    runtime_dir = repo_root / "runtime"

    replay = _load_replay(repo_root)
    snapshot = _load_snapshot(runtime_dir)
    ledger = _load_ledger_state(runtime_dir)

    symbols = _sym_union(replay, snapshot, ledger)
    rows = [_row(sym, replay, snapshot, ledger) for sym in symbols]

    out = {
        "schema_version": "lifecycle_replay_drift_audit.v1",
        "symbols": rows,
        "counts": {
            "replay_symbols": len(replay),
            "snapshot_symbols": len(snapshot),
            "ledger_symbols": len(ledger),
            "union_symbols": len(symbols),
        },
    }

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

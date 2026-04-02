#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _all_files(dir_path: Path, prefix: str) -> List[Path]:
    return sorted(dir_path.glob(f"{prefix}_*.ndjson"))


def _load_all_rows(dir_path: Path, prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _all_files(dir_path, prefix):
        rows.extend(_read_ndjson(path))
    return rows


def load_evidence(data_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "broker_events": _load_all_rows(data_dir / "broker_events", "BROKER_EVENTS_IBKR"),
        "fills": _load_all_rows(data_dir / "fills", "FILLS"),
        "fees": _load_all_rows(data_dir / "fees", "FEES"),
    }


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("payload")
    return payload if isinstance(payload, dict) else {}


def replay_positions(evidence: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic replay from full fill + fee ledger history.

    broker_events are counted for diagnostics, but not yet used as the
    authority path because current broker event rows are mostly heartbeats.
    """

    positions: Dict[str, Dict[str, Any]] = {}

    for row in evidence["fills"]:
        payload = _payload(row)

        symbol = str(payload.get("symbol") or "").strip().upper()
        side = str(payload.get("side") or "").strip().upper()
        status = str(payload.get("status") or "").strip().upper()
        reject = bool(payload.get("reject", False))

        if not symbol or reject or status not in {"FILLED", "PARTIAL_FILLED", "PARTIALLY_FILLED"}:
            continue

        try:
            qty = float(payload.get("quantity", 0) or 0)
            price = float(payload.get("fill_price", 0) or 0)
        except Exception:
            continue

        if qty == 0:
            continue

        signed_qty = qty if side == "BUY" else -qty if side == "SELL" else 0.0
        if signed_qty == 0:
            continue

        pos = positions.setdefault(
            symbol,
            {
                "qty": 0.0,
                "gross_notional": 0.0,
                "fees": 0.0,
                "fill_count": 0,
            },
        )
        pos["qty"] += signed_qty
        pos["gross_notional"] += signed_qty * price
        pos["fill_count"] += 1

    for row in evidence["fees"]:
        payload = _payload(row)

        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            continue

        try:
            fee = float(payload.get("fee_amount", 0) or 0)
        except Exception:
            continue

        pos = positions.setdefault(
            symbol,
            {
                "qty": 0.0,
                "gross_notional": 0.0,
                "fees": 0.0,
                "fill_count": 0,
            },
        )
        pos["fees"] += fee

    final: Dict[str, Dict[str, Any]] = {}
    for symbol, p in sorted(positions.items()):
        qty = float(p["qty"])
        if qty == 0:
            continue

        gross_notional = float(p["gross_notional"])
        fees = float(p["fees"])

        avg_cost = abs((gross_notional + fees) / qty)

        final[symbol] = {
            "qty": qty,
            "avg_cost": avg_cost,
            "total_fees": fees,
            "fill_count": int(p["fill_count"]),
        }

    return final


def build_replay_state(repo_root: Path) -> Dict[str, Any]:
    data_dir = repo_root / "data"
    evidence = load_evidence(data_dir)

    broker_non_heartbeat = 0
    for row in evidence["broker_events"]:
        event_type = str(row.get("event_type") or "").strip().lower()
        if event_type and event_type != "heartbeat":
            broker_non_heartbeat += 1

    positions = replay_positions(evidence)

    return {
        "schema_version": "lifecycle_replay_state.v3",
        "positions": positions,
        "positions_count": len(positions),
        "inputs": {
            "broker_events_count": len(evidence["broker_events"]),
            "broker_non_heartbeat_count": broker_non_heartbeat,
            "fills_count": len(evidence["fills"]),
            "fees_count": len(evidence["fees"]),
        },
        "notes": (
            "Replay uses full fill + fee ledger history across all daily ndjson files. "
            "Broker events are diagnostic-only until non-heartbeat lifecycle events are present."
        ),
    }


def main() -> None:
    repo_root = Path("/home/ubuntu/chad_finale")
    out = build_replay_state(repo_root)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

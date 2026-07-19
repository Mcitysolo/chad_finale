#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from chad.utils.epoch import is_pre_epoch, load_epoch_state


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


def replay_positions(
    evidence: Dict[str, List[Dict[str, Any]]],
    epoch_cutoff: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Deterministic replay from full fill + fee ledger history.

    broker_events are counted for diagnostics, but not yet used as the
    authority path because current broker event rows are mostly heartbeats.

    W1B-4: when ``epoch_cutoff`` is provided, fills and fees whose realised time
    is strictly before the cutoff are skipped, so the replay reflects the
    CURRENT epoch only — mirroring the one epoch-aware engine
    (chad/analytics/trade_stats_engine.py). Pre-reset residuals (e.g. futures +
    QQQ that netted flat before the Epoch-3 boundary) no longer surface as
    phantom positions. ``epoch_cutoff=None`` preserves byte-for-byte legacy
    behaviour. Records with no usable timestamp are kept (is_pre_epoch -> False)
    — the safe/legacy direction.
    """

    positions: Dict[str, Dict[str, Any]] = {}

    for row in evidence["fills"]:
        if epoch_cutoff is not None and is_pre_epoch(row, epoch_cutoff):
            continue
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
        if epoch_cutoff is not None and is_pre_epoch(row, epoch_cutoff):
            continue
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

    # W1B-4: resolve the current-epoch cutoff from the SAME repo_root the
    # evidence was loaded from, so tests that point repo_root at tmp_path also
    # get their epoch_state.json honoured. Fail-safe: load_epoch_state returns
    # None when runtime/epoch_state.json is absent/corrupt -> epoch_cutoff=None
    # -> no filter -> byte-for-byte legacy behaviour.
    epoch_state = load_epoch_state(runtime_dir=repo_root / "runtime")
    epoch_cutoff = epoch_state.epoch_started_at if epoch_state is not None else None
    epoch_cutoff_raw = epoch_state.epoch_started_at_raw if epoch_state is not None else None

    fills_pre_epoch_skipped = 0
    fees_pre_epoch_skipped = 0
    if epoch_cutoff is not None:
        fills_pre_epoch_skipped = sum(
            1 for r in evidence["fills"] if is_pre_epoch(r, epoch_cutoff)
        )
        fees_pre_epoch_skipped = sum(
            1 for r in evidence["fees"] if is_pre_epoch(r, epoch_cutoff)
        )

    positions = replay_positions(evidence, epoch_cutoff=epoch_cutoff)

    return {
        "schema_version": "lifecycle_replay_state.v3",
        "positions": positions,
        "positions_count": len(positions),
        "epoch_cutoff_utc": epoch_cutoff_raw,
        "inputs": {
            "broker_events_count": len(evidence["broker_events"]),
            "broker_non_heartbeat_count": broker_non_heartbeat,
            "fills_count": len(evidence["fills"]),
            "fees_count": len(evidence["fees"]),
            "fills_pre_epoch_skipped": fills_pre_epoch_skipped,
            "fees_pre_epoch_skipped": fees_pre_epoch_skipped,
        },
        "notes": (
            "Replay uses fill + fee ledger history across all daily ndjson files, "
            "filtered to the current epoch (fills/fees realised before epoch_cutoff_utc "
            "are skipped; a missing/corrupt runtime/epoch_state.json disables the filter "
            "and preserves legacy all-history behaviour). "
            "Broker events are diagnostic-only until non-heartbeat lifecycle events are present."
        ),
    }


def main() -> None:
    repo_root = Path("/home/ubuntu/chad_finale")
    out = build_replay_state(repo_root)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

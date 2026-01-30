#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import Request, urlopen

from chad.execution.exit_only_executor import (
    AssetClass,
    LiveGateDecision,
    Position,
    build_exit_only_plan,
)

LIVEGATE_URL = os.getenv("CHAD_LIVEGATE_URL", "http://127.0.0.1:9618/live-gate").strip()
RUNTIME = Path("/home/ubuntu/CHAD FINALE/runtime")
POS_SNAPSHOT_PATH = Path(os.getenv("CHAD_POSITIONS_SNAPSHOT", str(RUNTIME / "positions_snapshot.json")))


def _http_json(url: str) -> Dict[str, Any]:
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=5) as r:  # nosec B310
        raw = r.read().decode("utf-8")
    return json.loads(raw)


def _to_gate(obj: Dict[str, Any]) -> LiveGateDecision:
    return LiveGateDecision(
        allow_exits_only=bool(obj.get("allow_exits_only", False)),
        allow_ibkr_paper=bool(obj.get("allow_ibkr_paper", False)),
        allow_ibkr_live=bool(obj.get("allow_ibkr_live", False)),
        operator_mode=str(obj.get("operator_mode") or ""),
        reasons=tuple(obj.get("reasons") or ()),
    )


def _asset_class_from_position(rec: Dict[str, Any]) -> AssetClass:
    sec_type = str(rec.get("secType") or "").upper()
    if sec_type == "STK":
        return AssetClass.EQUITY
    return AssetClass.UNKNOWN


def _load_positions(path: Path) -> List[Position]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by = data.get("positions_by_conid") or {}
    out: List[Position] = []
    for _conid, rec in by.items():
        sym = str(rec.get("symbol") or "").strip()
        qty = float(rec.get("qty") or 0.0)
        out.append(
            Position(
                symbol=sym,
                asset_class=_asset_class_from_position(rec),
                qty=qty,
                avg_cost=float(rec.get("avg_cost")) if rec.get("avg_cost") is not None else None,
                currency=str(rec.get("currency") or "USD"),
                venue=str(rec.get("exchange") or "") or None,
            )
        )
    return out


def main() -> int:
    gate_raw = _http_json(LIVEGATE_URL)
    gate = _to_gate(gate_raw)

    positions = _load_positions(POS_SNAPSHOT_PATH)

    plan = build_exit_only_plan(
        live_gate=gate,
        positions=positions,
        lane_id=None,  # future-only; must remain None for now
    )

    print(json.dumps(asdict(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

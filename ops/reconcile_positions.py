#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

RUNTIME = Path("/home/ubuntu/chad_finale/runtime")
POS_PATH = RUNTIME / "positions_snapshot.json"
STATE_PATH = RUNTIME / "ibkr_paper_ledger_state.json"
OUT_PATH = RUNTIME / "reconciliation_state.json"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def main() -> int:
    if not POS_PATH.is_file():
        OUT_PATH.write_text(json.dumps({
            "status": "RED",
            "reason": "missing_positions_snapshot",
            "ts_utc": utc_now_iso(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2

    if not STATE_PATH.is_file():
        OUT_PATH.write_text(json.dumps({
            "status": "RED",
            "reason": "missing_ibkr_paper_ledger_state",
            "ts_utc": utc_now_iso(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 2

    pos = json.loads(POS_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    broker = {int(k): float(v.get("qty") or 0.0) for k, v in (pos.get("positions_by_conid") or {}).items()}
    chad = {int(v.get("conId") or 0): float(v.get("qty") or 0.0)
            for v in (state.get("open") or {}).values()
            if int(v.get("conId") or 0)}

    mismatches = []
    for conid in sorted(set(broker) | set(chad)):
        bq = broker.get(conid)
        cq = chad.get(conid)
        if bq is None:
            mismatches.append({"conId": conid, "type": "MISSING_IN_BROKER", "chad_qty": cq})
        elif cq is None:
            mismatches.append({"conId": conid, "type": "MISSING_IN_CHAD_STATE", "broker_qty": bq})
        else:
            if abs(float(bq) - float(cq)) > 1e-9:
                mismatches.append({"conId": conid, "type": "QTY_MISMATCH", "broker_qty": bq, "chad_qty": cq})

    payload = {
        "broker_source": str(POS_PATH),
        "chad_state_source": str(STATE_PATH),
        "counts": {
            "broker_positions": len(broker),
            "chad_state_positions": len(chad),
            "mismatches": len(mismatches),
        },
        "mismatches": mismatches,
        "status": "GREEN" if len(mismatches) == 0 else "RED",
        "ts_utc": utc_now_iso(),
    }

    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(OUT_PATH)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

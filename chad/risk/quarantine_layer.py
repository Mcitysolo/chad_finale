from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = ROOT / "runtime"

GOV_PATH = RUNTIME / "dynamic_caps_risk_governed.json"
DOM_PATH = RUNTIME / "dominance_allocator.json"
OUT_PATH = RUNTIME / "dynamic_caps_quarantine.json"

# thresholds
MIN_TRADES_ACTIVE = 2
DECAY_NO_TRADE = 0.60   # decay factor if no trades
LOSS_QUARANTINE = -50.0
HARD_QUARANTINE_CAP = 0.08


def iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        n = len(weights) or 1
        return {k: 1.0 / n for k in weights}
    return {k: float(v) / total for k, v in weights.items()}


def main() -> int:
    gov = read_json(GOV_PATH, {})
    dom = read_json(DOM_PATH, {})

    weights = gov.get("risk_governed_weights") or {}
    stats = dom.get("strategy_stats") or {}

    if not isinstance(weights, dict) or not weights:
        raise SystemExit("risk_governed_weights missing")

    if not isinstance(stats, dict):
        stats = {}

    adjusted = {}
    notes = {}

    for strategy, w in weights.items():
        st = stats.get(strategy) or {}
        trades = int(st.get("trades", 0) or 0)
        pnl = float(st.get("total_pnl", 0.0) or 0.0)

        new_w = float(w)
        strategy_notes = []

        # no activity → decay
        if trades < MIN_TRADES_ACTIVE:
            new_w *= DECAY_NO_TRADE
            strategy_notes.append("no_recent_activity_decay")

        # loss quarantine
        if pnl <= LOSS_QUARANTINE:
            new_w = min(new_w, HARD_QUARANTINE_CAP)
            strategy_notes.append("loss_quarantine")

        adjusted[strategy] = new_w
        notes[strategy] = {
            "input_weight": w,
            "adjusted_weight": new_w,
            "trades": trades,
            "total_pnl": pnl,
            "notes": strategy_notes,
        }

    adjusted = normalize(adjusted)

    out = {
        "ts_utc": iso_z(),
        "source_risk_layer": str(GOV_PATH),
        "source_dominance": str(DOM_PATH),
        "quarantine_weights": adjusted,
        "details": notes,
        "note": "Use quarantine_weights as final allocation if available.",
    }

    atomic_write_json(OUT_PATH, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

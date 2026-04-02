from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = ROOT / "runtime"

DOM_PATH = RUNTIME / "dominance_allocator.json"
OVERLAY_PATH = RUNTIME / "dynamic_caps_dominance_overlay.json"
OUT_PATH = RUNTIME / "dynamic_caps_risk_governed.json"

MIN_FLOOR = 0.02
MAX_CEILING = 0.35

LOW_CONFIDENCE_TRADES = 3
MED_CONFIDENCE_TRADES = 8

LOSS_CUT_1 = -25.0
LOSS_CUT_2 = -75.0

HOT_SCORE_SOFTCAP = 0.18
HOT_SCORE_HARDCAP = 0.30


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
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def clamp(weights: Dict[str, float], floors: Dict[str, float], ceilings: Dict[str, float]) -> Dict[str, float]:
    clamped = {}
    for k, v in weights.items():
        lo = floors.get(k, MIN_FLOOR)
        hi = ceilings.get(k, MAX_CEILING)
        clamped[k] = min(hi, max(lo, float(v)))

    total = sum(clamped.values())
    if total <= 0:
        n = len(clamped) or 1
        return {k: 1.0 / n for k in clamped}
    return {k: v / total for k, v in clamped.items()}


def main() -> int:
    dom = read_json(DOM_PATH, {})
    overlay = read_json(OVERLAY_PATH, {})

    strategy_stats = dom.get("strategy_stats") or {}
    if not isinstance(strategy_stats, dict):
        strategy_stats = {}

    weights = overlay.get("overlay_weights") or {}
    if not isinstance(weights, dict) or not weights:
        raise SystemExit("overlay_weights missing from dynamic_caps_dominance_overlay.json")

    weights = {k: float(v) for k, v in weights.items()}
    weights = normalize(weights)

    floors: Dict[str, float] = {}
    ceilings: Dict[str, float] = {}
    adjustments: Dict[str, Dict[str, Any]] = {}

    for strategy, w in weights.items():
        st = strategy_stats.get(strategy) or {}
        trades = int(st.get("trades", 0) or 0)
        total_pnl = float(st.get("total_pnl", 0.0) or 0.0)
        score = float(st.get("score", 0.0) or 0.0)
        win_rate = float(st.get("win_rate", 0.0) or 0.0)

        floor = MIN_FLOOR
        ceiling = MAX_CEILING
        notes = []

        # confidence gating
        if trades < LOW_CONFIDENCE_TRADES:
            ceiling = min(ceiling, 0.18)
            notes.append("low_confidence_cap")
        elif trades < MED_CONFIDENCE_TRADES:
            ceiling = min(ceiling, 0.26)
            notes.append("medium_confidence_cap")

        # drawdown / pnl suppression
        if total_pnl <= LOSS_CUT_1:
            ceiling = min(ceiling, 0.10)
            notes.append("loss_cut_1")
        if total_pnl <= LOSS_CUT_2:
            ceiling = min(ceiling, 0.05)
            notes.append("loss_cut_2")

        # hot streak control
        if score >= HOT_SCORE_SOFTCAP:
            ceiling = min(ceiling, 0.25)
            notes.append("hot_score_softcap")
        if score >= HOT_SCORE_HARDCAP:
            ceiling = min(ceiling, 0.20)
            notes.append("hot_score_hardcap")

        # tiny floor reduction for unproven side engines
        if trades == 0 and strategy in {"alpha_crypto", "alpha_forex", "crypto", "forex"}:
            floor = 0.01
            notes.append("unproven_side_engine_floor")

        floors[strategy] = floor
        ceilings[strategy] = ceiling
        adjustments[strategy] = {
            "input_weight": round(w, 6),
            "floor": round(floor, 6),
            "ceiling": round(ceiling, 6),
            "trades": trades,
            "total_pnl": round(total_pnl, 6),
            "score": round(score, 6),
            "win_rate": round(win_rate, 6),
            "notes": notes,
        }

    governed = clamp(weights, floors, ceilings)

    out = {
        "ts_utc": iso_z(),
        "source_overlay": str(OVERLAY_PATH),
        "source_dominance_allocator": str(DOM_PATH),
        "input_overlay_weights": weights,
        "risk_governed_weights": governed,
        "floors": floors,
        "ceilings": ceilings,
        "adjustments": adjustments,
        "note": "Use risk_governed_weights as final allocation input when available.",
    }

    atomic_write_json(OUT_PATH, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

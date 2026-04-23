from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = ROOT / "runtime"

Q_PATH = RUNTIME / "dynamic_caps_quarantine.json"
CYCLE_PATH = RUNTIME / "full_execution_cycle_last.json"
OUT_PATH = RUNTIME / "dynamic_caps_correlation.json"

# group definitions — simple but practical first pass
GROUPS = {
    "tech_directional": {"alpha", "beta_trend", "gamma"},
    "broad_directional": {"delta", "omega"},
    "side_engines": {"crypto", "forex", "alpha_crypto", "alpha_forex"},
}

GROUP_CAPS = {
    "tech_directional": 0.55,
    "broad_directional": 0.25,
    "side_engines": 0.08,
}


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
    quarantine = read_json(Q_PATH, {})
    cycle = read_json(CYCLE_PATH, {})

    weights = quarantine.get("quarantine_weights") or {}
    if not isinstance(weights, dict) or not weights:
        raise SystemExit("quarantine_weights missing")

    weights = {k: float(v) for k, v in weights.items()}
    notes: Dict[str, Any] = {}

    # apply group caps
    adjusted = dict(weights)
    for group_name, members in GROUPS.items():
        total_group = sum(adjusted.get(m, 0.0) for m in members)
        cap = float(GROUP_CAPS[group_name])

        if total_group > cap and total_group > 0:
            scale = cap / total_group
            for m in members:
                adjusted[m] = adjusted.get(m, 0.0) * scale

            notes[group_name] = {
                "cap": cap,
                "before": total_group,
                "after": sum(adjusted.get(m, 0.0) for m in members),
                "scaled": True,
            }
        else:
            notes[group_name] = {
                "cap": cap,
                "before": total_group,
                "after": total_group,
                "scaled": False,
            }

    adjusted = normalize(adjusted)

    out = {
        "ts_utc": iso_z(),
        "source_quarantine": str(Q_PATH),
        "source_cycle": str(CYCLE_PATH),
        "group_caps": GROUP_CAPS,
        "input_quarantine_weights": weights,
        "correlation_governed_weights": adjusted,
        "group_notes": notes,
        "note": "Use correlation_governed_weights as final published allocation if available.",
    }

    atomic_write_json(OUT_PATH, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

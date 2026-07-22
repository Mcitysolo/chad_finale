from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME = ROOT / "runtime"

DOM_PATH = RUNTIME / "dominance_allocator.json"
CAPS_PATH = RUNTIME / "dynamic_caps.json"
OUT_PATH = RUNTIME / "dynamic_caps_dominance_overlay.json"

FALLBACK = {
    "alpha": 0.10,
    "beta_trend": 0.30,
    "gamma": 0.25,
    "delta": 0.20,
    "omega": 0.10,
    "alpha_crypto": 0.03,
    "alpha_forex": 0.02,
}

MIN_WEIGHT = 0.02
MAX_WEIGHT = 0.60
DOMINANCE_BLEND = 0.65  # how much dominance influences final weights


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
        total = sum(FALLBACK.values())
        return {k: v / total for k, v in FALLBACK.items()}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def clamp(weights: Dict[str, float]) -> Dict[str, float]:
    out = {k: min(MAX_WEIGHT, max(MIN_WEIGHT, float(v))) for k, v in weights.items()}
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def main() -> int:
    dom = read_json(DOM_PATH, {})
    caps = read_json(CAPS_PATH, {})

    dom_weights = dom.get("dominance_weights") or {}
    if not isinstance(dom_weights, dict) or not dom_weights:
        dom_weights = dict(FALLBACK)

    base_weights = caps.get("normalized_weights") or caps.get("raw_weights") or {}
    if not isinstance(base_weights, dict) or not base_weights:
        base_weights = dict(FALLBACK)

    dom_weights = normalize({k: float(v) for k, v in dom_weights.items()})
    base_weights = normalize({k: float(v) for k, v in base_weights.items()})

    keys = sorted(set(FALLBACK) | set(dom_weights) | set(base_weights))

    blended = {}
    for k in keys:
        bw = float(base_weights.get(k, FALLBACK.get(k, 0.0)))
        dw = float(dom_weights.get(k, FALLBACK.get(k, 0.0)))
        blended[k] = ((1.0 - DOMINANCE_BLEND) * bw) + (DOMINANCE_BLEND * dw)

    blended = clamp(blended)

    out = {
        "ts_utc": iso_z(),
        "source_dynamic_caps": str(CAPS_PATH),
        "source_dominance_allocator": str(DOM_PATH),
        "dominance_blend": DOMINANCE_BLEND,
        "base_weights": base_weights,
        "dominance_weights": dom_weights,
        "overlay_weights": blended,
        "note": "Use overlay_weights as strategy allocation input for DynamicRiskAllocator.",
    }

    atomic_write_json(OUT_PATH, out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

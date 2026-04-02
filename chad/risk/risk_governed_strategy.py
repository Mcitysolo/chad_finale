from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
GOVERNED_PATH = ROOT / "runtime" / "dynamic_caps_risk_governed.json"


class RiskGovernedOverlayStrategy:
    name = "risk_governed_overlay"

    def apply(self, *, repo_root: Path, base_weights: Mapping[str, float], log) -> Dict[str, float]:
        try:
            if not GOVERNED_PATH.exists():
                raise RuntimeError("risk governed overlay file missing")

            data = json.loads(GOVERNED_PATH.read_text(encoding="utf-8"))
            governed = data.get("risk_governed_weights")

            if not isinstance(governed, dict) or not governed:
                raise RuntimeError("invalid risk_governed_weights payload")

            adjusted: Dict[str, float] = {}
            for k, bw in base_weights.items():
                adjusted[k] = float(governed.get(k, bw))

            log.info(
                "risk_governed_overlay_applied",
                extra={"weights": adjusted},
            )
            return adjusted

        except Exception as e:
            log.warning(
                "risk_governed_overlay_failed_fallback",
                extra={"error": str(e)},
            )
            return dict(base_weights)

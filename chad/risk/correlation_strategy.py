from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
C_PATH = ROOT / "runtime" / "dynamic_caps_correlation.json"


class CorrelationOverlayStrategy:
    name = "correlation_overlay"

    def apply(self, *, repo_root: Path, base_weights: Mapping[str, float], log) -> Dict[str, float]:
        try:
            if not C_PATH.exists():
                raise RuntimeError("correlation file missing")

            data = json.loads(C_PATH.read_text(encoding="utf-8"))
            corr = data.get("correlation_governed_weights")

            if not isinstance(corr, dict):
                raise RuntimeError("invalid correlation_governed_weights")

            adjusted = {}
            for k, bw in base_weights.items():
                adjusted[k] = float(corr.get(k, bw))

            log.info("correlation_overlay_applied", extra={"weights": adjusted})
            return adjusted

        except Exception as e:
            log.warning("correlation_overlay_failed_fallback", extra={"error": str(e)})
            return dict(base_weights)

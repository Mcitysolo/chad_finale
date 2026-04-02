from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
OVERLAY_PATH = ROOT / "runtime" / "dynamic_caps_dominance_overlay.json"


class DominanceOverlayStrategy:
    name = "dominance_overlay"

    def apply(self, *, repo_root: Path, base_weights: Mapping[str, float], log) -> Dict[str, float]:
        try:
            if not OVERLAY_PATH.exists():
                raise RuntimeError("overlay file missing")

            data = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
            overlay = data.get("overlay_weights")

            if not isinstance(overlay, dict):
                raise RuntimeError("invalid overlay_weights")

            # preserve only keys from base (never resurrect unknown strategies)
            adjusted = {}
            for k, bw in base_weights.items():
                adjusted[k] = float(overlay.get(k, bw))

            log.info(
                "dominance_overlay_applied",
                extra={"weights": adjusted},
            )

            return adjusted

        except Exception as e:
            log.warning(
                "dominance_overlay_failed_fallback",
                extra={"error": str(e)},
            )
            return dict(base_weights)

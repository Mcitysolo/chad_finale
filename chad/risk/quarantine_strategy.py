from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

ROOT = Path("/home/ubuntu/chad_finale")
Q_PATH = ROOT / "runtime" / "dynamic_caps_quarantine.json"


class QuarantineOverlayStrategy:
    name = "quarantine_overlay"

    def apply(self, *, repo_root: Path, base_weights: Mapping[str, float], log) -> Dict[str, float]:
        try:
            if not Q_PATH.exists():
                raise RuntimeError("quarantine file missing")

            data = json.loads(Q_PATH.read_text(encoding="utf-8"))
            q = data.get("quarantine_weights")

            if not isinstance(q, dict):
                raise RuntimeError("invalid quarantine_weights")

            adjusted = {}
            for k, bw in base_weights.items():
                adjusted[k] = float(q.get(k, bw))

            log.info("quarantine_overlay_applied", extra={"weights": adjusted})
            return adjusted

        except Exception as e:
            log.warning("quarantine_overlay_failed_fallback", extra={"error": str(e)})
            return dict(base_weights)

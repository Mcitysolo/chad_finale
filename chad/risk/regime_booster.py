#!/usr/bin/env python3
"""
RegimeBooster — bounded sizing multiplier on favorable conditions.

When CHAD detects a clean regime with strong conviction (high confidence,
low VIX, no event risk, breadth aligned), it applies a small multiplier
to strategy caps so good days produce above-average returns. The
multiplier is bounded — never exceeds 1.5× — so a regime misread can't
blow up the book.

INPUTS:
  - runtime/regime_state.json    (current regime + confidence)
  - runtime/strategy_intelligence.json (regime profile)
  - runtime/event_risk.json      (FOMC/CPI proximity)
  - data/bars/1d/VIX.json        (latest VIX close)
  - config/regime_booster_policy.json

OUTPUT:
  - runtime/regime_booster.json
    {
      "multiplier": 1.0-1.5,
      "active": true/false,
      "reasons": [...],
      "ts_utc": ISO-8601
    }

The multiplier is consumed by the dynamic risk allocator AFTER chassis
enforcement and SCR sizing, capped at policy.max_multiplier.

PHASE 12B per SSOT v8.2 roadmap.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

LOG = logging.getLogger("chad.risk.regime_booster")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
CONFIG_DIR = REPO_ROOT / "config"

REGIME_PATH = RUNTIME_DIR / "regime_state.json"
STRAT_INTEL_PATH = RUNTIME_DIR / "strategy_intelligence.json"
EVENT_RISK_PATH = RUNTIME_DIR / "event_risk.json"
VIX_BARS_PATH = REPO_ROOT / "data" / "bars" / "1d" / "VIX.json"
POLICY_PATH = CONFIG_DIR / "regime_booster_policy.json"
OUT_PATH = RUNTIME_DIR / "regime_booster.json"


DEFAULT_POLICY: Dict[str, Any] = {
    "schema_version": "regime_booster_policy.v1",
    "max_multiplier": 1.50,
    "min_confidence": 0.70,
    "favorable_regimes": ["trending_bull", "trending_bear"],
    "vix_calm_threshold": 18.0,
    "vix_elevated_threshold": 25.0,
    "event_risk_veto_severities": ["high", "extreme"],
    "boost_per_factor": 0.10
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("read_failed path=%s err=%s", path, exc)
        return {}


def _load_policy() -> Dict[str, Any]:
    on_disk = _read_json(POLICY_PATH)
    if not on_disk:
        return dict(DEFAULT_POLICY)
    merged = dict(DEFAULT_POLICY)
    merged.update(on_disk)
    return merged


def _read_vix_close() -> float:
    try:
        d = _read_json(VIX_BARS_PATH)
        bars = d.get("bars", [])
        if not bars:
            return 0.0
        b = bars[-1]
        return float(b.get("close", b.get("c", 0.0)))
    except Exception:
        return 0.0


def compute_booster(
    regime: str,
    confidence: float,
    vix: float,
    event_severity: str,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Pure function: compute the boost multiplier from regime inputs.
    Multiplier starts at 1.0 and accumulates bonuses up to max_multiplier.
    Any single veto condition forces multiplier back to 1.0.
    """
    max_mult = float(policy["max_multiplier"])
    min_conf = float(policy["min_confidence"])
    favorable = list(policy["favorable_regimes"])
    vix_calm = float(policy["vix_calm_threshold"])
    vix_elevated = float(policy["vix_elevated_threshold"])
    veto_severities = [s.lower() for s in policy["event_risk_veto_severities"]]
    boost_per = float(policy["boost_per_factor"])

    multiplier = 1.0
    reasons: List[str] = []
    vetoes: List[str] = []

    # VETOES (any one forces multiplier back to 1.0)
    if event_severity.lower() in veto_severities:
        vetoes.append(f"event_risk_{event_severity}")
    if vix > vix_elevated:
        vetoes.append(f"vix_elevated_{vix:.1f}")
    if confidence < min_conf:
        vetoes.append(f"low_confidence_{confidence:.2f}")
    if regime.lower() not in [r.lower() for r in favorable]:
        vetoes.append(f"unfavorable_regime_{regime}")

    if vetoes:
        return {
            "schema_version": "regime_booster.v1",
            "multiplier": 1.0,
            "active": False,
            "reasons": [f"vetoed: {v}" for v in vetoes],
            "regime": regime,
            "confidence": confidence,
            "vix": vix,
            "event_severity": event_severity,
            "ts_utc": _utc_now_iso(),
        }

    # POSITIVE FACTORS (additive, capped at max)
    if confidence >= 0.75:
        multiplier += boost_per
        reasons.append(f"high_confidence_{confidence:.2f}")
    if vix <= vix_calm:
        multiplier += boost_per
        reasons.append(f"vix_calm_{vix:.1f}")
    if regime.lower() == "trending_bull":
        multiplier += boost_per
        reasons.append("trending_bull_bias")
    if event_severity.lower() in ("low", "none", ""):
        multiplier += boost_per * 0.5
        reasons.append("no_event_risk")

    multiplier = min(multiplier, max_mult)

    return {
        "schema_version": "regime_booster.v1",
        "multiplier": round(multiplier, 3),
        "active": multiplier > 1.0,
        "reasons": reasons,
        "regime": regime,
        "confidence": confidence,
        "vix": vix,
        "event_severity": event_severity,
        "ts_utc": _utc_now_iso(),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    regime_data = _read_json(REGIME_PATH)
    regime = str(regime_data.get("regime", "unknown")).lower()
    confidence = float(regime_data.get("confidence", 0.0))

    vix = _read_vix_close()

    event_data = _read_json(EVENT_RISK_PATH)
    event_severity = str(event_data.get("severity", "none")).lower()

    policy = _load_policy()

    result = compute_booster(regime, confidence, vix, event_severity, policy)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(OUT_PATH)

    LOG.info(
        "regime_booster_published multiplier=%.3f active=%s regime=%s vix=%.1f reasons=%s",
        result["multiplier"], result["active"], regime, vix,
        result.get("reasons", [])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

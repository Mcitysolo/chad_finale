"""
chad/analytics/regime_classifier.py

Phase-8 Session 4 (G1): composite market regime classifier.

Produces one of:  trending_bull | trending_bear | ranging | volatile | unknown

Inputs (all optional — missing inputs degrade gracefully to 'unknown'):

  realized_vol_percentile : float in [0, 1]
      Percentile of recent realized vol against its own rolling history.
      > 0.75 → volatile regime.

  adx                     : float, 0+
      Average Directional Index. ADX > 25 signals a trend.

  trend_slope             : float
      Signed slope of a short SMA vs a longer SMA (or z-scored log-return
      window). Positive → bull bias, negative → bear bias.

  market_breadth          : float in [-1, 1]
      Optional confirmation signal — advance/decline ratio normalized.
      Positive strengthens bull bias, negative strengthens bear bias.

Classification rules (ordered, first match wins):

  1. realized_vol_percentile > 0.75              → volatile
  2. adx >= 25 and trend_slope > 0               → trending_bull
  3. adx >= 25 and trend_slope < 0               → trending_bear
  4. adx  < 25 and realized_vol_percentile <= 0.75 → ranging
  5. otherwise                                   → unknown

If neither vol nor adx inputs are available the classifier returns
'unknown' with inputs_used=[] so the caller can tell this was a
degraded classification rather than a low-confidence one.

Writes runtime/regime_state.json atomically. The schema extends the
existing bootstrap schema (regime_state.v1) so older readers continue
to work without modification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
REGIME_STATE_PATH = ROOT / "runtime" / "regime_state.json"

# Thresholds — tunable via kwargs on classify_regime().
DEFAULT_VOL_PCTILE_THRESHOLD: float = 0.75
DEFAULT_ADX_THRESHOLD: float = 25.0

SCHEMA_VERSION = "regime_state.v1"

VALID_REGIMES = frozenset(
    {"trending_bull", "trending_bear", "ranging", "volatile", "unknown"}
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN guard
        return None
    return f


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    confidence: float
    inputs_used: List[str] = field(default_factory=list)
    details: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_regime(
    realized_vol_percentile: Optional[float] = None,
    adx: Optional[float] = None,
    trend_slope: Optional[float] = None,
    market_breadth: Optional[float] = None,
    vol_pctile_threshold: float = DEFAULT_VOL_PCTILE_THRESHOLD,
    adx_threshold: float = DEFAULT_ADX_THRESHOLD,
) -> RegimeResult:
    """Run the composite classifier.

    Returns a RegimeResult. Never raises on bad input — unusable values
    are treated as missing and the method degrades to 'unknown'.
    """
    vol_p = _safe_float(realized_vol_percentile)
    adx_v = _safe_float(adx)
    slope = _safe_float(trend_slope)
    breadth = _safe_float(market_breadth)

    inputs_used: List[str] = []
    if vol_p is not None:
        inputs_used.append("realized_vol_percentile")
    if adx_v is not None:
        inputs_used.append("adx")
    if slope is not None:
        inputs_used.append("trend_slope")
    if breadth is not None:
        inputs_used.append("market_breadth")

    details: Dict[str, Any] = {
        "realized_vol_percentile": vol_p,
        "adx": adx_v,
        "trend_slope": slope,
        "market_breadth": breadth,
        "vol_pctile_threshold": vol_pctile_threshold,
        "adx_threshold": adx_threshold,
    }

    # No inputs at all → unknown.
    if not inputs_used:
        return RegimeResult("unknown", 0.0, inputs_used, details)

    # Rule 1: volatile regime overrides everything.
    if vol_p is not None and vol_p > vol_pctile_threshold:
        # Confidence scales with how far above threshold we are (capped).
        dist = min(1.0, (vol_p - vol_pctile_threshold) / max(1e-6, 1.0 - vol_pctile_threshold))
        return RegimeResult("volatile", max(0.5, 0.5 + 0.5 * dist), inputs_used, details)

    # Rule 2 & 3: trending regimes require ADX ≥ threshold AND a slope.
    if adx_v is not None and slope is not None and adx_v >= adx_threshold:
        directional_confirm = breadth if breadth is not None else 0.0
        strength = min(1.0, (adx_v - adx_threshold) / adx_threshold)  # 0 at threshold, 1 at 2x threshold
        if slope > 0:
            confidence = min(1.0, 0.5 + 0.5 * strength + 0.05 * max(0.0, directional_confirm))
            return RegimeResult("trending_bull", confidence, inputs_used, details)
        if slope < 0:
            confidence = min(1.0, 0.5 + 0.5 * strength - 0.05 * min(0.0, directional_confirm))
            return RegimeResult("trending_bear", confidence, inputs_used, details)

    # Rule 4: calm + non-trending = ranging.
    if adx_v is not None and adx_v < adx_threshold:
        if vol_p is None or vol_p <= vol_pctile_threshold:
            # Confidence rises as ADX gets smaller (more clearly ranging).
            clarity = min(1.0, (adx_threshold - adx_v) / adx_threshold)
            return RegimeResult("ranging", max(0.5, 0.5 + 0.5 * clarity), inputs_used, details)

    # Rule 5: fallthrough.
    return RegimeResult("unknown", 0.3, inputs_used, details)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_regime_state(path: Path = REGIME_STATE_PATH) -> Dict[str, Any]:
    """Return the current regime_state payload, or a safe default if missing."""
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "ts_utc": "",
            "ttl_seconds": 60,
            "ok": False,
            "regime": "unknown",
            "confidence": 0.0,
            "source": "missing",
            "inputs_used": [],
            "previous_regime": None,
            "notes": "",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": SCHEMA_VERSION,
            "ts_utc": "",
            "ttl_seconds": 60,
            "ok": False,
            "regime": "unknown",
            "confidence": 0.0,
            "source": "unreadable",
            "inputs_used": [],
            "previous_regime": None,
            "notes": "",
        }
    return data if isinstance(data, dict) else {}


def write_regime_state(
    result: RegimeResult,
    source: str = "regime_classifier",
    path: Path = REGIME_STATE_PATH,
    ttl_seconds: int = 60,
) -> Dict[str, Any]:
    """Persist a regime classification result to runtime/regime_state.json.

    previous_regime is captured from whatever the file currently says, so
    G3 (regime-transition reduction) can compare one write to the next
    without holding its own state.
    """
    current = read_regime_state(path)
    previous_regime = current.get("regime") if current.get("regime") in VALID_REGIMES else None

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": _utc_now_iso(),
        "ttl_seconds": int(ttl_seconds),
        "ok": True,
        "regime": result.regime,
        "confidence": float(result.confidence),
        "source": source,
        "inputs_used": list(result.inputs_used),
        "previous_regime": previous_regime,
        "notes": "",
    }
    _write_atomic(path, payload)
    return payload


def classify_and_write(
    realized_vol_percentile: Optional[float] = None,
    adx: Optional[float] = None,
    trend_slope: Optional[float] = None,
    market_breadth: Optional[float] = None,
    source: str = "regime_classifier",
    path: Path = REGIME_STATE_PATH,
) -> Dict[str, Any]:
    """Convenience one-shot: classify, persist, return the payload."""
    result = classify_regime(
        realized_vol_percentile=realized_vol_percentile,
        adx=adx,
        trend_slope=trend_slope,
        market_breadth=market_breadth,
    )
    return write_regime_state(result, source=source, path=path)


__all__ = [
    "REGIME_STATE_PATH",
    "SCHEMA_VERSION",
    "VALID_REGIMES",
    "DEFAULT_VOL_PCTILE_THRESHOLD",
    "DEFAULT_ADX_THRESHOLD",
    "RegimeResult",
    "classify_regime",
    "classify_and_write",
    "read_regime_state",
    "write_regime_state",
]

"""
chad/analytics/signal_confidence.py

Phase-8 Session 3 (S3): canonical confidence computation for CHAD intents.

confidence = signal_strength × regime_quality × liquidity_quality

All three factors are normalized to [0, 1]. The product is clamped to the
same range. Confidence then feeds the sizing multiplier at intent
construction time: final_size = base_size × max(SIZING_FLOOR, confidence).

The SIZING_FLOOR (default 0.1) guarantees that a low-confidence intent is
still sized non-trivially — the multiplier attenuates but never zeroes a
position. Strategies that want a hard reject should emit a suppression
reason at signal generation time, not rely on confidence → 0.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


# Floor applied to the confidence-based sizing multiplier.
# Low confidence reduces size but never eliminates the trade.
SIZING_FLOOR: float = 0.1

# Neutral default — used when confidence cannot be computed.
DEFAULT_CONFIDENCE: float = 0.5


# Regime label → regime_quality multiplier.
# Labels are lowercased before lookup, so callers do not need to normalize.
REGIME_QUALITY: Mapping[str, float] = {
    "trending_bull": 1.0,
    "trending_bear": 1.0,
    "trending": 1.0,
    "risk_on": 1.0,
    "risk_off": 1.0,
    "ranging": 0.7,
    "choppy": 0.5,
    "volatile": 0.5,
    "unknown": 0.5,
    "": 0.5,
    "adverse": 0.2,
}


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def normalize_signal_strength(raw_value: float, method: str = "tanh") -> float:
    """
    Normalize any signal magnitude to [0, 1].

    method='tanh': maps (-inf, inf) → (0, 1) via (tanh(x) + 1) / 2.
                   Good for unbounded values like z-scores.
    method='clip': clips a pre-normalized value to [0, 1].
                   Good for strengths already expressed on a 0-1 scale.
    """
    try:
        v = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    if method == "tanh":
        return _clamp01((math.tanh(v) + 1.0) / 2.0)
    return _clamp01(v)


def regime_quality_from_state(regime_state: Any) -> float:
    """Return the regime_quality multiplier for a regime label.

    Unknown / missing labels map to 0.5 (neutral-penalty), matching the
    audit survey's 'unknown' bucket. Adverse labels map to 0.2 so low-
    quality regimes meaningfully shrink confidence without zeroing it.
    """
    if regime_state is None:
        return REGIME_QUALITY["unknown"]
    key = str(regime_state).strip().lower()
    return float(REGIME_QUALITY.get(key, REGIME_QUALITY["unknown"]))


def compute_confidence(
    signal_strength: float,
    regime_quality: float = 1.0,
    liquidity_quality: float = 1.0,
    tf_multiplier: float = 1.0,
) -> float:
    """Compute confidence = signal_strength * regime_quality * liquidity_quality * tf_multiplier.

    All inputs are clamped to [0, 1] before multiplication. The result is
    clamped to [0, 1] as well. Non-numeric inputs are treated as 0.

    tf_multiplier (Phase-8 Session 5, S2): higher-timeframe confirmation
    attenuator from chad.analytics.timeframe_confirmation. Default 1.0
    keeps pre-Session-5 callers unchanged.
    """
    try:
        s = _clamp01(float(signal_strength))
    except (TypeError, ValueError):
        s = 0.0
    try:
        r = _clamp01(float(regime_quality))
    except (TypeError, ValueError):
        r = 0.0
    try:
        liq = _clamp01(float(liquidity_quality))
    except (TypeError, ValueError):
        liq = 0.0
    try:
        tf = _clamp01(float(tf_multiplier))
    except (TypeError, ValueError):
        tf = 1.0
    return _clamp01(s * r * liq * tf)


def sizing_multiplier(confidence: float, floor: float = SIZING_FLOOR) -> float:
    """Return the sizing multiplier applied to base size.

    multiplier = max(floor, clamp01(confidence))

    A confidence value outside [0, 1] is clamped first; the floor then
    guarantees a minimum size fraction. Callers apply this as:

        final_size = base_size * sizing_multiplier(intent.confidence)
    """
    try:
        c = _clamp01(float(confidence))
    except (TypeError, ValueError):
        c = 0.0
    try:
        f = max(0.0, float(floor))
    except (TypeError, ValueError):
        f = SIZING_FLOOR
    return max(f, c)


def confidence_from_signal(
    signal_strength: Optional[float],
    regime_state: Any = None,
    liquidity_quality: float = 1.0,
    normalize_method: str = "tanh",
) -> float:
    """Convenience: full pipeline from raw signal_strength → confidence.

    If signal_strength is None or non-finite, returns DEFAULT_CONFIDENCE
    (0.5) so missing-metadata intents are not penalized.
    """
    if signal_strength is None:
        return DEFAULT_CONFIDENCE
    try:
        raw = float(signal_strength)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    if math.isnan(raw) or math.isinf(raw):
        return DEFAULT_CONFIDENCE

    s_norm = normalize_signal_strength(raw, method=normalize_method)
    r_q = regime_quality_from_state(regime_state)
    return compute_confidence(s_norm, r_q, liquidity_quality)


__all__ = [
    "SIZING_FLOOR",
    "DEFAULT_CONFIDENCE",
    "REGIME_QUALITY",
    "compute_confidence",
    "confidence_from_signal",
    "normalize_signal_strength",
    "regime_quality_from_state",
    "sizing_multiplier",
]

"""
chad/analytics/threshold_adapter.py

Phase-8 Session 5 (S4): adaptive entry thresholds by market regime.

Utility module — strategies *opt in* by calling adjust(base, regime) or
adjust_rsi(oversold, overbought, regime) instead of using hardcoded
values. Nothing in Session 5 refactors existing strategies; that is
done incrementally as each strategy is touched.

Rationale
---------
A z-score cutoff that is correct in a calm market is too loose in a
volatile one (and too tight in a range). Scaling the cutoff by a
regime-aware multiplier is the simplest durable way to make entry
thresholds self-adaptive without per-strategy bespoke logic.

Regime multipliers (product default):

    trending_bull / trending_bear : 1.0   (no change — trend follows itself)
    ranging                       : 0.8   (easier to enter — range trades
                                           want more entries, not fewer)
    volatile                      : 1.5   (harder to enter — false signals
                                           dominate in chop)
    unknown                       : 1.2   (slightly harder — be conservative
                                           when the classifier can't decide)
    adverse                       : 2.0   (very hard to enter; almost off)

Callers may override the map by passing a custom dict into the helpers.

Safety
------
  * Unknown regime labels fall back to the 'unknown' multiplier (1.2).
  * adjust_rsi clamps to [0, 100] so no strategy ever sees an invalid
    RSI band.
  * All helpers return defaults on non-numeric input rather than raising.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "threshold_adapter_config.json"


# Hardcoded defaults — used when config/threshold_adapter_config.json is
# missing or malformed. Audit-O (2026-04-22) calibrated the file with
# softer values (ranging 0.9, volatile 1.3, unknown 1.1); these fallback
# defaults match the softened values so missing config still fires the
# calibrated behavior.
_DEFAULT_REGIME_MULTIPLIERS: Mapping[str, float] = {
    "trending_bull": 1.0,
    "trending_bear": 1.0,
    "trending": 1.0,
    "risk_on": 1.0,
    "risk_off": 1.0,
    "ranging": 0.9,
    "volatile": 1.3,
    "choppy": 1.3,
    "unknown": 1.1,
    "adverse": 2.0,
}

_DEFAULT_MULTIPLIER_FALLBACK: float = 1.1

# Cached config (loaded once on first use).
_CACHED_MULTIPLIERS: Optional[Mapping[str, float]] = None
_CACHED_DEFAULT: Optional[float] = None


def _coerce_multipliers(raw: Any) -> Mapping[str, float]:
    """Accept a dict from JSON and coerce its values to floats.

    Non-float values fall back to the hardcoded default for their key,
    or to _DEFAULT_MULTIPLIER_FALLBACK when the key itself is unknown.
    """
    if not isinstance(raw, dict):
        return dict(_DEFAULT_REGIME_MULTIPLIERS)
    out: dict = {}
    for k, v in raw.items():
        try:
            out[str(k).lower()] = float(v)
        except (TypeError, ValueError):
            out[str(k).lower()] = float(
                _DEFAULT_REGIME_MULTIPLIERS.get(str(k).lower(), _DEFAULT_MULTIPLIER_FALLBACK)
            )
    return out


def load_regime_multipliers(
    config_path: Path = DEFAULT_CONFIG_PATH,
    force_reload: bool = False,
) -> Mapping[str, float]:
    """Read config/threshold_adapter_config.json and return the map.

    Caches on first call; pass ``force_reload=True`` to re-read the
    file (useful for tests). A missing or malformed file falls back to
    ``_DEFAULT_REGIME_MULTIPLIERS`` — the module never raises.
    """
    global _CACHED_MULTIPLIERS, _CACHED_DEFAULT
    if _CACHED_MULTIPLIERS is not None and not force_reload:
        return _CACHED_MULTIPLIERS
    try:
        if not Path(config_path).is_file():
            _CACHED_MULTIPLIERS = dict(_DEFAULT_REGIME_MULTIPLIERS)
            _CACHED_DEFAULT = _DEFAULT_MULTIPLIER_FALLBACK
            return _CACHED_MULTIPLIERS
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("threshold_adapter: config load failed (%s) — using defaults", exc)
        _CACHED_MULTIPLIERS = dict(_DEFAULT_REGIME_MULTIPLIERS)
        _CACHED_DEFAULT = _DEFAULT_MULTIPLIER_FALLBACK
        return _CACHED_MULTIPLIERS
    if not isinstance(data, dict):
        _CACHED_MULTIPLIERS = dict(_DEFAULT_REGIME_MULTIPLIERS)
        _CACHED_DEFAULT = _DEFAULT_MULTIPLIER_FALLBACK
        return _CACHED_MULTIPLIERS
    _CACHED_MULTIPLIERS = _coerce_multipliers(data.get("regime_multipliers"))
    try:
        _CACHED_DEFAULT = float(data.get("default_multiplier", _DEFAULT_MULTIPLIER_FALLBACK))
    except (TypeError, ValueError):
        _CACHED_DEFAULT = _DEFAULT_MULTIPLIER_FALLBACK
    return _CACHED_MULTIPLIERS


def _get_default_multiplier() -> float:
    """Resolve the default-when-unknown multiplier from cache or fallback."""
    if _CACHED_DEFAULT is not None:
        return _CACHED_DEFAULT
    load_regime_multipliers()
    return _CACHED_DEFAULT if _CACHED_DEFAULT is not None else _DEFAULT_MULTIPLIER_FALLBACK


# Module-level constants kept for backward compatibility. REGIME_MULTIPLIERS
# is now resolved lazily from config; callers that imported it directly get
# the hardcoded default snapshot (matches the config-seeded values).
REGIME_MULTIPLIERS: Mapping[str, float] = _DEFAULT_REGIME_MULTIPLIERS
DEFAULT_MULTIPLIER: float = _DEFAULT_MULTIPLIER_FALLBACK


def _resolve_multiplier(
    regime_state: str,
    multipliers: Optional[Mapping[str, float]] = None,
) -> float:
    # Session-9 calibration: when the caller does not override, pull the
    # map from config/threshold_adapter_config.json (cached after first
    # read). Callers that pass an explicit dict still win.
    m = multipliers if multipliers is not None else load_regime_multipliers()
    key = str(regime_state or "").strip().lower()
    if key in m:
        try:
            return float(m[key])
        except (TypeError, ValueError):
            return _get_default_multiplier()
    # Fallback bucket: prefer a configured 'unknown', else hard default.
    fallback = m.get("unknown")
    try:
        return float(fallback) if fallback is not None else _get_default_multiplier()
    except (TypeError, ValueError):
        return _get_default_multiplier()


def adjust(
    base_threshold: float,
    regime_state: str,
    multipliers: Optional[Mapping[str, float]] = None,
) -> float:
    """Scale a base threshold by the regime-specific multiplier.

    Higher return value = harder to trigger. Strategies call this
    in place of using a hardcoded z-score/indicator cutoff.
    """
    try:
        base = float(base_threshold)
    except (TypeError, ValueError):
        return 0.0
    return base * _resolve_multiplier(regime_state, multipliers)


def adjust_rsi(
    base_oversold: float,
    base_overbought: float,
    regime_state: str,
    multipliers: Optional[Mapping[str, float]] = None,
) -> Tuple[float, float]:
    """Widen or tighten RSI bands by regime.

    Volatile regimes push the bands OUTWARD (lower oversold, higher
    overbought) so random spikes don't trigger entries. Ranging regimes
    pull the bands INWARD so more setups qualify for mean-reversion
    trades. Output is clamped to [0, 100].
    """
    try:
        low = float(base_oversold)
        high = float(base_overbought)
    except (TypeError, ValueError):
        return 0.0, 100.0
    if high < low:
        low, high = high, low

    mult = _resolve_multiplier(regime_state, multipliers)
    center = (low + high) / 2.0
    half_range = (high - low) / 2.0
    widened_half = half_range * mult

    new_low = max(0.0, center - widened_half)
    new_high = min(100.0, center + widened_half)
    if new_low > new_high:
        new_low, new_high = new_high, new_low
    return new_low, new_high


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "REGIME_MULTIPLIERS",
    "DEFAULT_MULTIPLIER",
    "adjust",
    "adjust_rsi",
    "load_regime_multipliers",
]

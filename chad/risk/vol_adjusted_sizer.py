"""
chad/risk/vol_adjusted_sizer.py

Phase-8 Session 7 (R3): per-trade volatility-adjusted position sizing.

Scales base_size inversely with a symbol's 20-day realized daily vol::

    multiplier = target_daily_vol / realized_daily_vol   (clamped)
    adjusted_size = base_size * multiplier

The multiplier is clamped to ``[floor, ceiling]`` so a single quiet
symbol cannot 20× its allocation and a volatility spike cannot drop
size to zero. When bar data is missing or vol is undefined we return
multiplier=1.0 (no adjustment) so the sizer is safe to wire into the
hot path without a data dependency.

Realized vol is computed as the population standard deviation of
close-to-close log-returns over ``lookback_days``. This matches
market_metrics_publisher's convention and keeps the two modules
interchangeable if we need to unify.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from typing import Any, List, Mapping, Optional

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BARS_DIR = ROOT / "data" / "bars" / "1d"
DEFAULT_CONFIG_PATH = ROOT / "config" / "sizing_config.json"

DEFAULT_TARGET_DAILY_VOL: float = 0.01
DEFAULT_LOOKBACK_DAYS: int = 20
DEFAULT_FLOOR_MULTIPLIER: float = 0.1
DEFAULT_CEILING_MULTIPLIER: float = 2.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _load_config_section(
    section: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Mapping[str, Any]:
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sub = data.get(section) if isinstance(data, dict) else None
    return sub if isinstance(sub, dict) else {}


def _load_close_series(symbol: str, bars_dir: Path) -> List[float]:
    if not symbol:
        return []
    path = bars_dir / f"{str(symbol).upper()}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    bars = data.get("bars") if isinstance(data, dict) else None
    if not isinstance(bars, list):
        return []
    closes: List[float] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        c = _safe_float(bar.get("close"), 0.0)
        if c > 0.0:
            closes.append(c)
    return closes


def compute_realized_daily_vol(
    closes: List[float],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> float:
    """Population std of log-returns over the last ``lookback_days``.

    Returns 0.0 when there is not enough data. Callers treat 0.0 as
    "unknown" and skip the adjustment.
    """
    window = max(2, int(lookback_days))
    if len(closes) < window + 1:
        return 0.0
    tail = closes[-(window + 1):]
    rets: List[float] = []
    for i in range(1, len(tail)):
        prev = tail[i - 1]
        cur = tail[i]
        if prev <= 0.0 or cur <= 0.0:
            continue
        rets.append(math.log(cur / prev))
    if len(rets) < 2:
        return 0.0
    return statistics.pstdev(rets)


# ---------------------------------------------------------------------------
# Sizer
# ---------------------------------------------------------------------------


class VolAdjustedSizer:
    """Scale base_size by target_vol / realized_vol within bounded limits."""

    def __init__(
        self,
        target_daily_vol: float = DEFAULT_TARGET_DAILY_VOL,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        floor_multiplier: float = DEFAULT_FLOOR_MULTIPLIER,
        ceiling_multiplier: float = DEFAULT_CEILING_MULTIPLIER,
        bars_dir: Path = DEFAULT_BARS_DIR,
    ) -> None:
        self.target_daily_vol = float(max(1e-8, target_daily_vol))
        self.lookback_days = int(max(2, lookback_days))
        self.floor_multiplier = float(max(0.0, floor_multiplier))
        self.ceiling_multiplier = float(max(self.floor_multiplier, ceiling_multiplier))
        self._bars_dir = Path(bars_dir)

    @classmethod
    def from_config(
        cls,
        config_path: Path = DEFAULT_CONFIG_PATH,
        bars_dir: Path = DEFAULT_BARS_DIR,
    ) -> "VolAdjustedSizer":
        cfg = _load_config_section("vol_adjusted_sizer", config_path)
        return cls(
            target_daily_vol=_safe_float(cfg.get("target_daily_vol"), DEFAULT_TARGET_DAILY_VOL),
            lookback_days=int(cfg.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS),
            floor_multiplier=_safe_float(cfg.get("floor_multiplier"), DEFAULT_FLOOR_MULTIPLIER),
            ceiling_multiplier=_safe_float(cfg.get("ceiling_multiplier"), DEFAULT_CEILING_MULTIPLIER),
            bars_dir=bars_dir,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def _compute_realized_vol(self, symbol: str) -> float:
        closes = _load_close_series(symbol, self._bars_dir)
        return compute_realized_daily_vol(closes, self.lookback_days)

    def get_size_multiplier(self, symbol: str) -> float:
        """Return the bounded vol multiplier for ``symbol``.

        Missing bar data or an undefined realized vol returns 1.0 so the
        caller's base_size survives unchanged.
        """
        realized = self._compute_realized_vol(symbol)
        if realized <= 0.0:
            return 1.0
        raw = self.target_daily_vol / realized
        return max(self.floor_multiplier, min(self.ceiling_multiplier, raw))

    def adjust(self, base_size: int, symbol: str) -> int:
        """Return the vol-adjusted integer size.

        The multiplier is clamped in get_size_multiplier; the integer
        result is floored at 1 so sizing never returns zero — call sites
        treat 0 as a rejection, but "reject" is the gate's job, not the
        sizer's.
        """
        base = max(0, int(base_size))
        if base <= 0:
            return 0
        mult = self.get_size_multiplier(symbol)
        return max(1, int(base * mult))


__all__ = [
    "DEFAULT_BARS_DIR",
    "DEFAULT_CEILING_MULTIPLIER",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_FLOOR_MULTIPLIER",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_TARGET_DAILY_VOL",
    "VolAdjustedSizer",
    "compute_realized_daily_vol",
]

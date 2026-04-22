"""
chad/risk/composite_size_cap.py

Phase-8 Session 7 (R5): composite size cap.

Applies several per-trade caps in sequence and returns the tightest::

    final_size = min(
        vol_adjusted_size,                 # input from R3
        max_per_symbol,                    # absolute per-symbol ceiling
        remaining_sector_exposure,         # room left in sector budget
        avg_daily_volume * max_adv_pct,    # liquidity cap
        account_equity * pos_pct / ref_px, # margin/equity cap
    )

Each cap is optional — missing data simply drops that cap from the
``min`` instead of rejecting the trade. The only hard floor is
``max(1, ...)`` because downstream consumers interpret zero as "no
order" and we want the *gate* to decide rejection, not the sizer.

Configuration
-------------

All tunables live under the ``composite_cap`` key in
``config/sizing_config.json`` and are loaded via ``from_config``.
Callers can also pass a plain dict.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "sizing_config.json"

DEFAULT_MAX_PER_SYMBOL: int = 1000
DEFAULT_MAX_SECTOR_EXPOSURE: int = 5000
DEFAULT_MAX_ADV_PCT: float = 0.01
DEFAULT_MAX_POSITION_PCT: float = 0.10
DEFAULT_AVG_PRICE_ASSUMPTION: float = 50.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Mapping[str, Any]:
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    sub = data.get("composite_cap")
    return sub if isinstance(sub, dict) else {}


# ---------------------------------------------------------------------------
# Composite cap
# ---------------------------------------------------------------------------


class CompositeSizeCap:
    """Apply per-symbol / sector / liquidity / margin caps in sequence."""

    def __init__(
        self,
        max_per_symbol: int = DEFAULT_MAX_PER_SYMBOL,
        max_sector_exposure: int = DEFAULT_MAX_SECTOR_EXPOSURE,
        max_adv_pct: float = DEFAULT_MAX_ADV_PCT,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
        avg_price_assumption: float = DEFAULT_AVG_PRICE_ASSUMPTION,
    ) -> None:
        self.max_per_symbol = max(1, int(max_per_symbol))
        self.max_sector_exposure = max(0, int(max_sector_exposure))
        self.max_adv_pct = max(0.0, float(max_adv_pct))
        self.max_position_pct = max(0.0, float(max_position_pct))
        self.avg_price_assumption = max(0.01, float(avg_price_assumption))

    @classmethod
    def from_config(
        cls,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> "CompositeSizeCap":
        cfg = _load_config(config_path)
        return cls(
            max_per_symbol=_safe_int(cfg.get("max_per_symbol"), DEFAULT_MAX_PER_SYMBOL),
            max_sector_exposure=_safe_int(
                cfg.get("max_sector_exposure"), DEFAULT_MAX_SECTOR_EXPOSURE
            ),
            max_adv_pct=_safe_float(cfg.get("max_adv_pct"), DEFAULT_MAX_ADV_PCT),
            max_position_pct=_safe_float(
                cfg.get("max_position_pct"), DEFAULT_MAX_POSITION_PCT
            ),
            avg_price_assumption=_safe_float(
                cfg.get("avg_price_assumption"), DEFAULT_AVG_PRICE_ASSUMPTION
            ),
        )

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(
        self,
        vol_adjusted_size: int,
        symbol: str = "",
        sector: str = "unknown",
        account_equity: float = 0.0,
        avg_daily_volume: float = 0.0,
        current_sector_exposure: int = 0,
        reference_price: float = 0.0,
    ) -> int:
        """Return the minimum-of-all-applicable-caps integer size.

        Any cap whose inputs are unavailable is skipped (not rejected).
        The result is floored at 1 share; a caller that wants to refuse
        a trade entirely should do so at the gate level.
        """
        base = _safe_int(vol_adjusted_size, 0)
        if base <= 0:
            return 0

        caps: List[int] = [base, self.max_per_symbol]

        sector_budget = self.max_sector_exposure - max(0, _safe_int(current_sector_exposure, 0))
        if sector_budget > 0:
            caps.append(sector_budget)
        elif self.max_sector_exposure > 0:
            # Sector already filled → clamp to 1 share (tests still let the
            # trade through; the gate can reject on notional if desired).
            caps.append(1)

        adv = _safe_float(avg_daily_volume, 0.0)
        if adv > 0.0 and self.max_adv_pct > 0.0:
            caps.append(max(1, int(adv * self.max_adv_pct)))

        equity = _safe_float(account_equity, 0.0)
        if equity > 0.0 and self.max_position_pct > 0.0:
            max_position_value = equity * self.max_position_pct
            px = _safe_float(reference_price, 0.0)
            if px <= 0.0:
                px = self.avg_price_assumption
            if px > 0.0:
                caps.append(max(1, int(max_position_value / px)))

        return max(1, min(caps))


__all__ = [
    "DEFAULT_AVG_PRICE_ASSUMPTION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_MAX_ADV_PCT",
    "DEFAULT_MAX_PER_SYMBOL",
    "DEFAULT_MAX_POSITION_PCT",
    "DEFAULT_MAX_SECTOR_EXPOSURE",
    "CompositeSizeCap",
]

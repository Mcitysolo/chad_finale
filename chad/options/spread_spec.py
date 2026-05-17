#!/usr/bin/env python3
"""
chad/options/spread_spec.py

Typed contract for a two-leg options vertical / combo spread.

Phase D Item 2 Tier 1 (paper-only hardening). This dataclass is additive —
the strategy continues to emit the same legacy string-keyed meta dict, but
also stamps an ``OptionsSpreadSpec`` instance under ``meta["spread_spec"]``.

Downstream consumers (IBKR adapter, paper-fill simulator, preview CLI)
prefer the typed spec when present and fall back to the legacy dict when
absent. This keeps the strategy → adapter contract enforceable at the type
boundary instead of relying on string-key spelling agreements across three
files.

Design constraints
------------------
* Pure data + validators. No IBKR, strategy, or execution imports.
* Frozen + slots so the spec cannot be mutated after construction.
* ``from_legacy_meta`` reconstructs the spec from the existing dict shape
  emitted by ``alpha_options.py`` so the adapter can degrade gracefully if
  a strategy has not yet been migrated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional


_EXPIRY_RE = re.compile(r"^\d{8}$")
_VALID_RIGHTS = frozenset({"C", "P"})


def _norm_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""


def _norm_right(value: Any) -> str:
    return _norm_str(value).upper()


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class OptionsSpreadSpec:
    """Typed specification of a two-leg options spread / combo (BAG)."""

    symbol: str
    expiry: str
    long_strike: float
    short_strike: float
    long_right: str
    short_right: str
    ratio_long: int = 1
    ratio_short: int = 1
    exchange: str = "SMART"
    currency: str = "USD"
    spread_type: str = "CUSTOM"
    max_loss_per_contract: Optional[float] = None
    net_debit_estimate: Optional[float] = None
    spread_id: Optional[str] = None
    dte: Optional[int] = None

    def __post_init__(self) -> None:
        # Normalize string fields in place via object.__setattr__ (frozen).
        sym = _norm_str(self.symbol).upper()
        exp = _norm_str(self.expiry)
        lr = _norm_right(self.long_right)
        sr = _norm_right(self.short_right)
        exch = _norm_str(self.exchange) or "SMART"
        ccy = _norm_str(self.currency).upper() or "USD"
        stype = _norm_str(self.spread_type).upper() or "CUSTOM"

        object.__setattr__(self, "symbol", sym)
        object.__setattr__(self, "expiry", exp)
        object.__setattr__(self, "long_right", lr)
        object.__setattr__(self, "short_right", sr)
        object.__setattr__(self, "exchange", exch)
        object.__setattr__(self, "currency", ccy)
        object.__setattr__(self, "spread_type", stype)

        if not sym:
            raise ValueError("OptionsSpreadSpec.symbol must be non-empty")
        if not _EXPIRY_RE.match(exp):
            raise ValueError(
                f"OptionsSpreadSpec.expiry must be YYYYMMDD (got {self.expiry!r})"
            )
        if not isinstance(self.long_strike, (int, float)) or self.long_strike <= 0:
            raise ValueError(
                f"OptionsSpreadSpec.long_strike must be > 0 (got {self.long_strike!r})"
            )
        if not isinstance(self.short_strike, (int, float)) or self.short_strike <= 0:
            raise ValueError(
                f"OptionsSpreadSpec.short_strike must be > 0 (got {self.short_strike!r})"
            )
        if float(self.long_strike) == float(self.short_strike):
            raise ValueError(
                "OptionsSpreadSpec.long_strike and short_strike must differ"
            )
        if lr not in _VALID_RIGHTS:
            raise ValueError(
                f"OptionsSpreadSpec.long_right must be 'C' or 'P' (got {self.long_right!r})"
            )
        if sr not in _VALID_RIGHTS:
            raise ValueError(
                f"OptionsSpreadSpec.short_right must be 'C' or 'P' (got {self.short_right!r})"
            )
        if not isinstance(self.ratio_long, int) or self.ratio_long < 1:
            raise ValueError(
                f"OptionsSpreadSpec.ratio_long must be int >= 1 (got {self.ratio_long!r})"
            )
        if not isinstance(self.ratio_short, int) or self.ratio_short < 1:
            raise ValueError(
                f"OptionsSpreadSpec.ratio_short must be int >= 1 (got {self.ratio_short!r})"
            )
        if not exch:
            raise ValueError("OptionsSpreadSpec.exchange must be non-empty")
        if not ccy:
            raise ValueError("OptionsSpreadSpec.currency must be non-empty")

    # ---------------------------------------------------------------- helpers

    def to_legacy_meta(self) -> Dict[str, Any]:
        """Project the spec back to the string-keyed dict that ``alpha_options``
        and the IBKR adapter have historically consumed. Keys are kept stable
        so older readers continue to work."""
        meta: Dict[str, Any] = {
            "sec_type": "BAG",
            "required_asset_class": "options",
            "spread_type": self.spread_type,
            "expiry": self.expiry,
            "long_strike": float(self.long_strike),
            "short_strike": float(self.short_strike),
            "long_right": self.long_right,
            "short_right": self.short_right,
            "exchange": self.exchange,
            "currency": self.currency,
            "ratio_long": int(self.ratio_long),
            "ratio_short": int(self.ratio_short),
        }
        if self.max_loss_per_contract is not None:
            meta["max_loss_per_contract"] = float(self.max_loss_per_contract)
        if self.net_debit_estimate is not None:
            meta["net_debit_estimate"] = float(self.net_debit_estimate)
        if self.spread_id is not None:
            meta["spread_id"] = self.spread_id
        if self.dte is not None:
            meta["dte"] = int(self.dte)
        return meta

    @classmethod
    def from_legacy_meta(
        cls, symbol: str, meta: Mapping[str, Any]
    ) -> "OptionsSpreadSpec":
        """Reconstruct an ``OptionsSpreadSpec`` from a legacy dict meta payload.

        Accepts the exact key names emitted by ``alpha_options.py`` today.
        Raises ``ValueError`` (via ``__post_init__``) on any missing or
        malformed field; the caller is expected to translate that to a
        domain-specific exception (e.g. ``ContractResolutionError``)."""
        if not isinstance(meta, Mapping):
            raise ValueError("OptionsSpreadSpec.from_legacy_meta requires a mapping")

        ls = _coerce_float(meta.get("long_strike"))
        ss = _coerce_float(meta.get("short_strike"))
        if ls is None or ss is None:
            raise ValueError(
                "from_legacy_meta requires numeric long_strike and short_strike"
            )

        return cls(
            symbol=_norm_str(symbol) or _norm_str(meta.get("symbol")),
            expiry=_norm_str(meta.get("expiry") or meta.get("lastTradeDateOrContractMonth")),
            long_strike=ls,
            short_strike=ss,
            long_right=_norm_right(meta.get("long_right")),
            short_right=_norm_right(meta.get("short_right")),
            ratio_long=_coerce_int(meta.get("ratio_long"), 1) or 1,
            ratio_short=_coerce_int(meta.get("ratio_short"), 1) or 1,
            exchange=_norm_str(meta.get("exchange")) or "SMART",
            currency=_norm_str(meta.get("currency")) or "USD",
            spread_type=_norm_str(meta.get("spread_type")) or "CUSTOM",
            max_loss_per_contract=_coerce_float(meta.get("max_loss_per_contract")),
            net_debit_estimate=_coerce_float(meta.get("net_debit_estimate")),
            spread_id=(
                _norm_str(meta.get("spread_id")) or None
                if meta.get("spread_id") is not None
                else None
            ),
            dte=_coerce_int(meta.get("dte")),
        )

    def bag_leg_dicts(self) -> List[Dict[str, Any]]:
        """Return the canonical 2-leg dict list (BUY long, SELL short) used by
        the paper-fill evidence writer and downstream FIFO consumers."""
        return [
            {
                "action": "BUY",
                "strike": float(self.long_strike),
                "right": self.long_right,
                "ratio": int(self.ratio_long),
                "expiry": self.expiry,
            },
            {
                "action": "SELL",
                "strike": float(self.short_strike),
                "right": self.short_right,
                "ratio": int(self.ratio_short),
                "expiry": self.expiry,
            },
        ]

    def as_dict(self) -> Dict[str, Any]:
        """Plain dict projection (useful for JSON output and diagnostics)."""
        return asdict(self)


__all__ = ["OptionsSpreadSpec"]

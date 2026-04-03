#!/usr/bin/env python3
"""
chad/strategies/alpha_options_config.py

Production-grade configuration builder for CHAD ALPHA_OPTIONS.

Public API
----------
- build_alpha_options_config() -> StrategyConfig
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from chad.types import StrategyConfig, StrategyName


# ---------------------------------------------------------------------------
# Canonical defaults
# ---------------------------------------------------------------------------

DEFAULT_ALPHA_OPTIONS_UNIVERSE: Tuple[str, ...] = (
    "SPY",  # S&P 500 ETF — most liquid options market
)

DEFAULT_ENABLED: bool = True
DEFAULT_MAX_GROSS_EXPOSURE: float = 0.15
DEFAULT_NOTES: str = "Vertical spread options engine"


# ---------------------------------------------------------------------------
# Internal config model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaOptionsConfigSpec:
    enabled: bool
    target_universe: Tuple[str, ...]
    max_gross_exposure: float
    notes: str

    def to_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            name=StrategyName.ALPHA_OPTIONS,
            enabled=self.enabled,
            target_universe=list(self.target_universe),
            max_gross_exposure=self.max_gross_exposure,
            notes=self.notes,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}.")


def _parse_float(raw: str | None, *, default: float) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid float value: {raw!r}") from exc


def _normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _dedupe_preserve_order(values: Iterable[str]) -> Tuple[str, ...]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        sym = _normalize_symbol(item)
        if not sym:
            continue
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return tuple(out)


def _parse_universe(raw: str | None, *, default: Sequence[str]) -> Tuple[str, ...]:
    if raw is None or not raw.strip():
        return _dedupe_preserve_order(default)
    parts = [p.strip() for p in raw.split(",")]
    return _dedupe_preserve_order(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Options-eligible symbols (highly liquid ETFs with good options markets)
ALLOWED_OPTIONS_SYMBOLS = {"SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "GOOGL"}


def _validate_symbol(sym: str) -> None:
    if sym not in ALLOWED_OPTIONS_SYMBOLS:
        raise ValueError(
            f"Unsupported ALPHA_OPTIONS symbol: {sym!r}. "
            f"Allowed symbols: {sorted(ALLOWED_OPTIONS_SYMBOLS)}"
        )


def _validate_universe(universe: Sequence[str]) -> Tuple[str, ...]:
    normalized = _dedupe_preserve_order(universe)
    if not normalized:
        raise ValueError("ALPHA_OPTIONS target_universe cannot be empty.")
    for sym in normalized:
        _validate_symbol(sym)
    return normalized


def _validate_max_gross_exposure(value: float) -> float:
    if value <= 0.0:
        raise ValueError("max_gross_exposure must be > 0.")
    if value > 1.0:
        raise ValueError("max_gross_exposure must be <= 1.0.")
    return float(value)


def _validate_notes(notes: str) -> str:
    text = str(notes).strip()
    if not text:
        raise ValueError("notes cannot be empty.")
    if len(text) > 500:
        raise ValueError("notes is unexpectedly long (>500 chars).")
    return text


def _build_validated_spec(
    *,
    enabled: bool,
    target_universe: Sequence[str],
    max_gross_exposure: float,
    notes: str,
) -> AlphaOptionsConfigSpec:
    return AlphaOptionsConfigSpec(
        enabled=bool(enabled),
        target_universe=_validate_universe(target_universe),
        max_gross_exposure=_validate_max_gross_exposure(max_gross_exposure),
        notes=_validate_notes(notes),
    )


# ---------------------------------------------------------------------------
# Environment override layer
# ---------------------------------------------------------------------------

def _load_env_overrides() -> AlphaOptionsConfigSpec:
    """
    CHAD_ALPHA_OPTIONS_ENABLED=true|false
    CHAD_ALPHA_OPTIONS_UNIVERSE=SPY
    CHAD_ALPHA_OPTIONS_MAX_GROSS_EXPOSURE=0.15
    """
    enabled = _parse_bool(os.getenv("CHAD_ALPHA_OPTIONS_ENABLED"), default=DEFAULT_ENABLED)
    target_universe = _parse_universe(os.getenv("CHAD_ALPHA_OPTIONS_UNIVERSE"), default=DEFAULT_ALPHA_OPTIONS_UNIVERSE)
    max_gross_exposure = _parse_float(os.getenv("CHAD_ALPHA_OPTIONS_MAX_GROSS_EXPOSURE"), default=DEFAULT_MAX_GROSS_EXPOSURE)
    notes = os.getenv("CHAD_ALPHA_OPTIONS_NOTES", DEFAULT_NOTES)

    return _build_validated_spec(
        enabled=enabled,
        target_universe=target_universe,
        max_gross_exposure=max_gross_exposure,
        notes=notes,
    )


def build_alpha_options_config() -> StrategyConfig:
    spec = _load_env_overrides()
    return spec.to_strategy_config()


__all__ = [
    "DEFAULT_ALPHA_OPTIONS_UNIVERSE",
    "AlphaOptionsConfigSpec",
    "build_alpha_options_config",
]

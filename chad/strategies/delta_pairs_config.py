#!/usr/bin/env python3
"""
chad/strategies/delta_pairs_config.py

Production-grade configuration builder for CHAD DELTA_PAIRS.

Public API
----------
- build_delta_pairs_config() -> StrategyConfig
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from chad.types import StrategyConfig, StrategyName


# ---------------------------------------------------------------------------
# Canonical defaults
# ---------------------------------------------------------------------------

DEFAULT_DELTA_PAIRS_UNIVERSE: Tuple[str, ...] = (
    "SPY",  # S&P 500 ETF
    "QQQ",  # Nasdaq 100 ETF
    "IWM",  # Russell 2000 ETF
)

DEFAULT_ENABLED: bool = True
DEFAULT_MAX_GROSS_EXPOSURE: float = 0.15
DEFAULT_NOTES: str = "Market-neutral pairs trading engine"

# Allowed symbols for validation
ALLOWED_SYMBOLS: frozenset = frozenset({"SPY", "QQQ", "IWM"})


# ---------------------------------------------------------------------------
# Internal config model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaPairsConfigSpec:
    """Internal validated spec before conversion into StrategyConfig."""

    enabled: bool
    target_universe: Tuple[str, ...]
    max_gross_exposure: float
    notes: str

    def to_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            name=StrategyName.DELTA_PAIRS,
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
    raise ValueError(
        f"Invalid boolean value: {raw!r}. "
        "Expected one of: true/false, 1/0, yes/no, on/off."
    )


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

def _validate_symbol(sym: str) -> None:
    if sym not in ALLOWED_SYMBOLS:
        raise ValueError(
            f"Unsupported DELTA_PAIRS symbol: {sym!r}. "
            f"Allowed symbols: {sorted(ALLOWED_SYMBOLS)}"
        )


def _validate_universe(universe: Sequence[str]) -> Tuple[str, ...]:
    normalized = _dedupe_preserve_order(universe)
    if not normalized:
        raise ValueError("DELTA_PAIRS target_universe cannot be empty.")
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
) -> DeltaPairsConfigSpec:
    return DeltaPairsConfigSpec(
        enabled=bool(enabled),
        target_universe=_validate_universe(target_universe),
        max_gross_exposure=_validate_max_gross_exposure(max_gross_exposure),
        notes=_validate_notes(notes),
    )


# ---------------------------------------------------------------------------
# Environment override layer
# ---------------------------------------------------------------------------

def _load_env_overrides() -> DeltaPairsConfigSpec:
    """
    Environment contract
    --------------------
    CHAD_DELTA_PAIRS_ENABLED=true|false
    CHAD_DELTA_PAIRS_UNIVERSE=SPY,QQQ,IWM
    CHAD_DELTA_PAIRS_MAX_GROSS_EXPOSURE=0.15
    CHAD_DELTA_PAIRS_NOTES=Market-neutral pairs trading engine
    """
    enabled = _parse_bool(
        os.getenv("CHAD_DELTA_PAIRS_ENABLED"),
        default=DEFAULT_ENABLED,
    )
    target_universe = _parse_universe(
        os.getenv("CHAD_DELTA_PAIRS_UNIVERSE"),
        default=DEFAULT_DELTA_PAIRS_UNIVERSE,
    )
    max_gross_exposure = _parse_float(
        os.getenv("CHAD_DELTA_PAIRS_MAX_GROSS_EXPOSURE"),
        default=DEFAULT_MAX_GROSS_EXPOSURE,
    )
    notes = os.getenv("CHAD_DELTA_PAIRS_NOTES", DEFAULT_NOTES)

    return _build_validated_spec(
        enabled=enabled,
        target_universe=target_universe,
        max_gross_exposure=max_gross_exposure,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_delta_pairs_config() -> StrategyConfig:
    """
    Build the canonical DELTA_PAIRS StrategyConfig.

    Deterministic, validated, fail-closed.
    """
    spec = _load_env_overrides()
    return spec.to_strategy_config()


__all__ = [
    "DEFAULT_DELTA_PAIRS_UNIVERSE",
    "DEFAULT_ENABLED",
    "DEFAULT_MAX_GROSS_EXPOSURE",
    "DEFAULT_NOTES",
    "ALLOWED_SYMBOLS",
    "DeltaPairsConfigSpec",
    "build_delta_pairs_config",
]

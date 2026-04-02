#!/usr/bin/env python3
"""
chad/strategies/alpha_futures_config.py

Production-grade configuration builder for CHAD Alpha Futures.

Purpose
-------
Provide a single, deterministic, validated source of truth for the
Alpha Futures strategy configuration.

Design goals
------------
- Fully compatible with CHAD's existing StrategyConfig contract.
- Explicit futures trading universe.
- Fail-closed validation.
- Clean environment override support.
- Deterministic behavior across preview, paper, and live flows.
- No side effects at import time.
- Zero hidden mutation.

Expected StrategyConfig fields
------------------------------
This module assumes chad.types.StrategyConfig supports:

    StrategyConfig(
        name=StrategyName.ALPHA_FUTURES,
        enabled=bool,
        target_universe=list[str] | None,
        max_gross_exposure=float | None,
        notes=str,
    )

If your local StrategyConfig expands later, this module can be extended
without changing its public API.

Public API
----------
- build_alpha_futures_config() -> StrategyConfig
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from chad.types import StrategyConfig, StrategyName


# ---------------------------------------------------------------------------
# Canonical defaults
# ---------------------------------------------------------------------------

DEFAULT_ALPHA_FUTURES_UNIVERSE: Tuple[str, ...] = (
    "MES",  # Micro E-mini S&P 500
    "MNQ",  # Micro E-mini Nasdaq-100
    "MCL",  # Micro WTI Crude Oil
    "MGC",  # Micro Gold
)

DEFAULT_ENABLED: bool = True
DEFAULT_MAX_GROSS_EXPOSURE: float = 0.25
DEFAULT_NOTES: str = "Futures momentum engine"


# ---------------------------------------------------------------------------
# Internal config model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaFuturesConfigSpec:
    """
    Internal validated spec before conversion into StrategyConfig.

    Why this exists
    ---------------
    Keeps validation logic isolated from CHAD's external config object.
    That gives us a clean place to enforce correctness before returning a
    StrategyConfig instance to the engine.
    """

    enabled: bool
    target_universe: Tuple[str, ...]
    max_gross_exposure: float
    notes: str

    def to_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            name=StrategyName.ALPHA_FUTURES,
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
    """
    Parse comma-separated futures symbols.

    Example:
        "MES,MNQ,MCL,MGC"
    """
    if raw is None or not raw.strip():
        return _dedupe_preserve_order(default)

    parts = [p.strip() for p in raw.split(",")]
    return _dedupe_preserve_order(parts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_symbol(sym: str) -> None:
    """
    Strict allow-list validation for current Alpha Futures deployment.

    This is intentional.

    We do NOT want arbitrary symbols entering the live engine just because
    someone typoed an env var or tried to sneak a larger contract in.
    """
    allowed = set(DEFAULT_ALPHA_FUTURES_UNIVERSE)
    if sym not in allowed:
        raise ValueError(
            f"Unsupported Alpha Futures symbol: {sym!r}. "
            f"Allowed symbols: {sorted(allowed)}"
        )


def _validate_universe(universe: Sequence[str]) -> Tuple[str, ...]:
    normalized = _dedupe_preserve_order(universe)

    if not normalized:
        raise ValueError("Alpha Futures target_universe cannot be empty.")

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
) -> AlphaFuturesConfigSpec:
    return AlphaFuturesConfigSpec(
        enabled=bool(enabled),
        target_universe=_validate_universe(target_universe),
        max_gross_exposure=_validate_max_gross_exposure(max_gross_exposure),
        notes=_validate_notes(notes),
    )


# ---------------------------------------------------------------------------
# Environment override layer
# ---------------------------------------------------------------------------

def _load_env_overrides() -> AlphaFuturesConfigSpec:
    """
    Environment contract
    --------------------
    CHAD_ALPHA_FUTURES_ENABLED=true|false
    CHAD_ALPHA_FUTURES_UNIVERSE=MES,MNQ,MCL,MGC
    CHAD_ALPHA_FUTURES_MAX_GROSS_EXPOSURE=0.25
    CHAD_ALPHA_FUTURES_NOTES=Futures momentum engine
    """
    enabled = _parse_bool(
        os.getenv("CHAD_ALPHA_FUTURES_ENABLED"),
        default=DEFAULT_ENABLED,
    )

    target_universe = _parse_universe(
        os.getenv("CHAD_ALPHA_FUTURES_UNIVERSE"),
        default=DEFAULT_ALPHA_FUTURES_UNIVERSE,
    )

    max_gross_exposure = _parse_float(
        os.getenv("CHAD_ALPHA_FUTURES_MAX_GROSS_EXPOSURE"),
        default=DEFAULT_MAX_GROSS_EXPOSURE,
    )

    notes = os.getenv("CHAD_ALPHA_FUTURES_NOTES", DEFAULT_NOTES)

    return _build_validated_spec(
        enabled=enabled,
        target_universe=target_universe,
        max_gross_exposure=max_gross_exposure,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_alpha_futures_config() -> StrategyConfig:
    """
    Build the canonical Alpha Futures StrategyConfig.

    This is the only public function the strategy registry should use.
    It is deterministic, validated, and fail-closed.
    """
    spec = _load_env_overrides()
    return spec.to_strategy_config()


__all__ = [
    "DEFAULT_ALPHA_FUTURES_UNIVERSE",
    "DEFAULT_ENABLED",
    "DEFAULT_MAX_GROSS_EXPOSURE",
    "DEFAULT_NOTES",
    "AlphaFuturesConfigSpec",
    "build_alpha_futures_config",
]

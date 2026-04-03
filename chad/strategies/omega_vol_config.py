#!/usr/bin/env python3
"""
chad/strategies/omega_vol_config.py

Production-grade configuration builder for CHAD OMEGA_VOL.

Public API
----------
- build_omega_vol_config() -> StrategyConfig
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from chad.types import StrategyConfig, StrategyName

DEFAULT_OMEGA_VOL_UNIVERSE: Tuple[str, ...] = (
    "SVXY",  # Short VIX short-term futures (0.5x inverse)
    "UVXY",  # Long VIX short-term futures (1.5x)
)

DEFAULT_ENABLED: bool = True
DEFAULT_MAX_GROSS_EXPOSURE: float = 0.06  # Tight — volatile instruments
DEFAULT_NOTES: str = "Volatility regime alpha engine"


@dataclass(frozen=True)
class OmegaVolConfigSpec:
    enabled: bool
    target_universe: Tuple[str, ...]
    max_gross_exposure: float
    notes: str

    def to_strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            name=StrategyName.OMEGA_VOL,
            enabled=self.enabled,
            target_universe=list(self.target_universe),
            max_gross_exposure=self.max_gross_exposure,
            notes=self.notes,
        )


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


ALLOWED_VOL_SYMBOLS = {"SVXY", "UVXY", "VIXY", "VXX", "SVIX", "UVIX", "VIXM"}


def _validate_symbol(sym: str) -> None:
    if sym not in ALLOWED_VOL_SYMBOLS:
        raise ValueError(
            f"Unsupported OMEGA_VOL symbol: {sym!r}. "
            f"Allowed: {sorted(ALLOWED_VOL_SYMBOLS)}"
        )


def _validate_universe(universe: Sequence[str]) -> Tuple[str, ...]:
    normalized = _dedupe_preserve_order(universe)
    if not normalized:
        raise ValueError("OMEGA_VOL target_universe cannot be empty.")
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


def _load_env_overrides() -> OmegaVolConfigSpec:
    """
    CHAD_OMEGA_VOL_ENABLED=true|false
    CHAD_OMEGA_VOL_UNIVERSE=SVXY,UVXY
    CHAD_OMEGA_VOL_MAX_GROSS_EXPOSURE=0.06
    """
    enabled = _parse_bool(os.getenv("CHAD_OMEGA_VOL_ENABLED"), default=DEFAULT_ENABLED)
    universe = _parse_universe(os.getenv("CHAD_OMEGA_VOL_UNIVERSE"), default=DEFAULT_OMEGA_VOL_UNIVERSE)
    exposure = _parse_float(os.getenv("CHAD_OMEGA_VOL_MAX_GROSS_EXPOSURE"), default=DEFAULT_MAX_GROSS_EXPOSURE)
    notes = os.getenv("CHAD_OMEGA_VOL_NOTES", DEFAULT_NOTES)

    universe = _validate_universe(universe)
    exposure = _validate_max_gross_exposure(exposure)
    notes = _validate_notes(notes)

    return OmegaVolConfigSpec(enabled=bool(enabled), target_universe=universe, max_gross_exposure=exposure, notes=notes)


def build_omega_vol_config() -> StrategyConfig:
    spec = _load_env_overrides()
    return spec.to_strategy_config()


__all__ = [
    "DEFAULT_OMEGA_VOL_UNIVERSE",
    "OmegaVolConfigSpec",
    "build_omega_vol_config",
]

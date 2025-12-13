#!/usr/bin/env python3
"""
chad/utils/legend_loader.py

Legend consensus loader for CHAD.

This module provides a production-safe way to read the legend consensus file
produced by the Phase-2 Legend pipeline:

    data/legend_top_stocks.json

It returns a strongly-typed LegendConsensus object and performs basic
validation:
    - file existence
    - JSON structure
    - weights presence and type
    - sum of weights is > 0

This keeps BetaBrain and the decision pipeline decoupled from filesystem and
JSON parsing details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from chad.types import LegendConsensus


class LegendLoaderError(RuntimeError):
    """Raised when legend consensus cannot be loaded or validated."""


@dataclass(frozen=True)
class LegendLoaderConfig:
    """
    Configuration for the legend loader.

    In Phase 3 we only need the path to the JSON file; later phases may add
    alternate sources (e.g. S3, DB, or API).
    """

    root_dir: Path = Path(__file__).resolve().parents[2]
    relative_path: Path = Path("data/legend_top_stocks.json")

    @property
    def json_path(self) -> Path:
        return (self.root_dir / self.relative_path).resolve()


class LegendLoader:
    """
    Loader for LegendConsensus from the Phase-2 output JSON.

    Typical usage:

        loader = LegendLoader()
        legend = loader.load()

    Callers should catch LegendLoaderError if they want to handle missing or
    invalid legend data gracefully.
    """

    def __init__(self, config: LegendLoaderConfig | None = None) -> None:
        self._config = config or LegendLoaderConfig()

    def load(self) -> LegendConsensus:
        """
        Load and validate LegendConsensus from the configured JSON file.

        Raises:
            LegendLoaderError on any structural or validation problem.
        """
        path = self._config.json_path
        if not path.exists():
            raise LegendLoaderError(f"Legend file not found at {path}")

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise LegendLoaderError(f"Failed to parse legend JSON: {exc}") from exc

        if not isinstance(raw, dict):
            raise LegendLoaderError("Legend JSON root must be an object")

        as_of_raw = raw.get("as_of")
        legends = raw.get("legends")
        weights = raw.get("weights")

        if not isinstance(as_of_raw, str):
            raise LegendLoaderError("Legend JSON must contain 'as_of' as ISO string")
        if not isinstance(legends, list) or not legends:
            raise LegendLoaderError("Legend JSON must contain non-empty 'legends' list")
        if not isinstance(weights, dict) or not weights:
            raise LegendLoaderError("Legend JSON must contain non-empty 'weights' object")

        try:
            as_of = datetime.fromisoformat(as_of_raw)
            if as_of.tzinfo is None:
                # Treat naive timestamps as UTC for safety.
                as_of = as_of.replace(tzinfo=timezone.utc)
        except Exception as exc:  # noqa: BLE001
            raise LegendLoaderError(f"Invalid 'as_of' timestamp: {exc}") from exc

        numeric_weights: dict[str, float] = {}
        total = 0.0
        for sym, w in weights.items():
            try:
                fv = float(w)
            except Exception as exc:  # noqa: BLE001
                raise LegendLoaderError(f"Non-numeric weight for {sym}: {exc}") from exc
            if fv <= 0.0:
                # For Phase 3 we ignore non-positive weights completely.
                continue
            numeric_weights[str(sym)] = fv
            total += fv

        if total <= 0.0:
            raise LegendLoaderError("Sum of legend weights must be > 0")

        # We do *not* normalize here; the Legend pipeline already emits
        # normalized weights. Keeping them as-is preserves upstream intent.
        return LegendConsensus(as_of=as_of, weights=numeric_weights)


def load_legend() -> LegendConsensus:
    """
    Convenience function to load legend consensus using default config.
    """
    return LegendLoader().load()

#!/usr/bin/env python3
"""
chad/portfolio/capital_allocator.py

Deterministic capital allocator for CHAD.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Mapping
import json


@dataclass(frozen=True)
class AllocationResult:
    weights: Dict[str, float]
    scores: Dict[str, float]
    total_score: float


def _clean_score(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return max(v, 0.0)


def allocate_capital(strategy_scores: Mapping[str, float]) -> AllocationResult:
    cleaned: Dict[str, float] = {
        str(k): _clean_score(v)
        for k, v in strategy_scores.items()
    }

    total = sum(cleaned.values())

    if total <= 0.0:
        return AllocationResult(
            weights={k: 0.0 for k in cleaned},
            scores=cleaned,
            total_score=0.0,
        )

    weights = {k: (v / total) for k, v in cleaned.items()}

    return AllocationResult(
        weights=weights,
        scores=cleaned,
        total_score=total,
    )


def load_allocation_weights(path: str = "/home/ubuntu/chad_finale/runtime/capital_allocator.json") -> Dict[str, float]:
    p = Path(path)
    if not p.is_file():
        return {}

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        weights = raw.get("weights", {})
        if isinstance(weights, dict):
            return {str(k): float(v) for k, v in weights.items()}
        return {}
    except Exception:
        return {}


def save_allocation_result(
    result: AllocationResult,
    path: str = "/home/ubuntu/chad_finale/runtime/capital_allocator.json",
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")


def get_strategy_weight(
    strategy: str,
    path: str = "/home/ubuntu/chad_finale/runtime/capital_allocator.json",
    default: float = 1.0,
) -> float:
    weights = load_allocation_weights(path)
    try:
        value = float(weights.get(strategy, default))
    except Exception:
        return default
    return max(0.0, value)


if __name__ == "__main__":
    demo = {
        "alpha_futures": 25.0,
        "beta": 10.0,
        "gamma": -5.0,
        "alpha_crypto": 5.0,
    }
    result = allocate_capital(demo)
    save_allocation_result(result)
    print(asdict(result))

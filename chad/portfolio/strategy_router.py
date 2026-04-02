#!/usr/bin/env python3
"""
chad/portfolio/strategy_router.py

Simple deterministic strategy router for CHAD.

Purpose
-------
Choose which strategy lane should be preferred right now based on:
- available signals
- allocator weights
- fallback order

This is NOT execution.
This is routing preference logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class RouteDecision:
    selected_strategy: str | None
    selected_symbols: List[str]
    reason: str
    available_counts: Dict[str, int]
    weights: Dict[str, float]
    rejected_strategies: Dict[str, str] = None  # type: ignore[assignment]


PREFERRED_ORDER: Sequence[str] = (
    "alpha_futures",
    "alpha",
    "alpha_crypto",
    "gamma",
    "beta",
)


def choose_strategy_route(
    *,
    available_signals: Mapping[str, Sequence[object]],
    weights: Mapping[str, float],
) -> RouteDecision:
    """
    Pick the best currently-available strategy.

    Rules:
    1. A strategy must have at least 1 signal to be considered.
    2. Among available strategies, higher allocator weight wins.
    3. Tie-break by preferred order.
    """

    counts = {
        name: len(list(signals or []))
        for name, signals in available_signals.items()
    }

    candidates = [
        name for name, count in counts.items()
        if count > 0
    ]

    # Build rejection reasons for strategies with no signals
    rejected: Dict[str, str] = {}
    for name, count in counts.items():
        if count == 0:
            rejected[name] = "no_signal"

    if not candidates:
        return RouteDecision(
            selected_strategy=None,
            selected_symbols=[],
            reason="no_available_signals",
            available_counts=counts,
            weights={str(k): float(v) for k, v in weights.items()},
            rejected_strategies=rejected,
        )

    preferred_rank = {name: idx for idx, name in enumerate(PREFERRED_ORDER)}

    def sort_key(name: str) -> tuple[float, int]:
        weight = float(weights.get(name, 0.0))
        rank = preferred_rank.get(name, 999)
        return (-weight, rank)

    chosen = sorted(candidates, key=sort_key)[0]
    chosen_weight = float(weights.get(chosen, 0.0))
    chosen_rank = preferred_rank.get(chosen, 999)

    # Build rejection reasons for losing candidates
    for name in candidates:
        if name == chosen:
            continue
        cand_weight = float(weights.get(name, 0.0))
        if cand_weight < chosen_weight:
            rejected[name] = "lower_weight_than_selected"
        else:
            # Equal weight but lost on preference tie-break
            rejected[name] = "lower_priority_than_selected"

    chosen_signals = list(available_signals.get(chosen, []) or [])
    chosen_symbols = [
        getattr(sig, "symbol", None)
        for sig in chosen_signals
        if getattr(sig, "symbol", None)
    ]

    return RouteDecision(
        selected_strategy=chosen,
        selected_symbols=chosen_symbols,
        reason="selected_best_available_strategy",
        available_counts=counts,
        weights={str(k): float(v) for k, v in weights.items()},
        rejected_strategies=rejected,
    )


if __name__ == "__main__":
    class DummySignal:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

    decision = choose_strategy_route(
        available_signals={
            "alpha_futures": [DummySignal("MES")],
            "alpha": [DummySignal("AAPL"), DummySignal("MSFT")],
            "alpha_crypto": [],
        },
        weights={
            "alpha_futures": 1.0,
            "alpha": 0.5,
            "alpha_crypto": 0.2,
        },
    )
    print(decision)

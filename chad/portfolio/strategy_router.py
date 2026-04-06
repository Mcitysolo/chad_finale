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

Failure mode (SSOT v6.4)
-------------------------
The router is **fail-closed**: on any internal exception or invariant
violation it returns selected_strategy=None with a diagnostic reason
code.  It never guesses, never silently picks an arbitrary strategy,
and never raises past its public API boundary.  Callers already handle
None correctly (log warning, return empty signal list).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

logger = logging.getLogger(__name__)


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
    "alpha_forex",
    "alpha_options",
    "gamma",
    "gamma_futures",
    "gamma_reversion",
    "beta",
    "omega",
    "omega_macro",
    "omega_vol",
    "delta",
    "delta_pairs",
)


def _fail_closed(
    weights: Mapping[str, float] | None,
    reason: str,
) -> RouteDecision:
    """Return a safe fail-closed decision with no selected strategy."""
    safe_weights: Dict[str, float] = {}
    if weights is not None:
        try:
            safe_weights = {str(k): float(v) for k, v in weights.items()}
        except Exception:
            pass
    return RouteDecision(
        selected_strategy=None,
        selected_symbols=[],
        reason=reason,
        available_counts={},
        weights=safe_weights,
        rejected_strategies={},
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

    Fail-closed: returns selected_strategy=None on any exception or
    invariant violation.  Never raises past this boundary.
    """
    try:
        return _choose_strategy_route_inner(
            available_signals=available_signals,
            weights=weights,
        )
    except Exception as exc:
        logger.error("Router exception (fail-closed): %s", exc, exc_info=True)
        return _fail_closed(weights, "router_error")


def _choose_strategy_route_inner(
    *,
    available_signals: Mapping[str, Sequence[object]],
    weights: Mapping[str, float],
) -> RouteDecision:
    """Core routing logic — may raise; caller wraps in fail-closed."""

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

    # Invariant check: selected strategy must have >0 signals
    if counts.get(chosen, 0) <= 0:
        logger.error(
            "Router invariant violation: selected %s has 0 signals (fail-closed)",
            chosen,
        )
        return _fail_closed(weights, "router_invariant_violation")

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


@dataclass(frozen=True)
class AllStrategiesDecision:
    """
    Always-active routing result (SSOT v6.4).

    Unlike RouteDecision (single-winner), this returns ALL strategies
    that produced signals, ordered by weight+preference.  The primary
    strategy is the highest-ranked, but every active strategy's signals
    are available for execution.
    """
    primary_strategy: str | None
    active_strategies: Dict[str, List[str]]   # strategy -> [symbols]
    ordered_strategies: List[str]             # weight+preference ranked
    reason: str
    available_counts: Dict[str, int]
    weights: Dict[str, float]
    rejected_strategies: Dict[str, str]       # only strategies with NO signals


def evaluate_all_strategies(
    *,
    available_signals: Mapping[str, Sequence[object]],
    weights: Mapping[str, float],
) -> AllStrategiesDecision:
    """
    Evaluate ALL enabled strategies and return every one that has signals.

    Fail-closed: returns empty active_strategies on exception.
    """
    try:
        return _evaluate_all_strategies_inner(
            available_signals=available_signals,
            weights=weights,
        )
    except Exception as exc:
        logger.error("All-strategies evaluation exception (fail-closed): %s", exc, exc_info=True)
        safe_weights: Dict[str, float] = {}
        if weights is not None:
            try:
                safe_weights = {str(k): float(v) for k, v in weights.items()}
            except Exception:
                pass
        return AllStrategiesDecision(
            primary_strategy=None,
            active_strategies={},
            ordered_strategies=[],
            reason="router_error",
            available_counts={},
            weights=safe_weights,
            rejected_strategies={},
        )


def _evaluate_all_strategies_inner(
    *,
    available_signals: Mapping[str, Sequence[object]],
    weights: Mapping[str, float],
) -> AllStrategiesDecision:
    """Core all-strategies logic — may raise; caller wraps in fail-closed."""

    counts = {
        name: len(list(signals or []))
        for name, signals in available_signals.items()
    }

    rejected: Dict[str, str] = {}
    for name, count in counts.items():
        if count == 0:
            rejected[name] = "no_signal"

    candidates = [name for name, count in counts.items() if count > 0]

    if not candidates:
        return AllStrategiesDecision(
            primary_strategy=None,
            active_strategies={},
            ordered_strategies=[],
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

    ordered = sorted(candidates, key=sort_key)
    primary = ordered[0]

    active: Dict[str, List[str]] = {}
    for name in ordered:
        sigs = list(available_signals.get(name, []) or [])
        symbols = [
            getattr(sig, "symbol", None)
            for sig in sigs
            if getattr(sig, "symbol", None)
        ]
        active[name] = symbols

    return AllStrategiesDecision(
        primary_strategy=primary,
        active_strategies=active,
        ordered_strategies=ordered,
        reason="all_active_strategies_evaluated",
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

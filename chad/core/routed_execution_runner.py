#!/usr/bin/env python3
"""
chad/core/routed_execution_runner.py

Route-aware execution builder for CHAD.

Purpose
-------
Build strategy signals, apply allocator weights, choose the best available
strategy lane, then return only that lane for downstream execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

from chad.utils.context_builder import ContextBuilder
from chad.portfolio.capital_allocator import load_allocation_weights
from chad.portfolio.strategy_router import choose_strategy_route


@dataclass(frozen=True)
class RoutedBuildResult:
    selected_strategy: str | None
    selected_symbols: List[str]
    available_counts: Dict[str, int]
    weights: Dict[str, float]
    selected_signals: Dict[str, list]


def _safe_alpha_futures(ctx):
    try:
        from chad.strategies.alpha_futures import alpha_futures_handler
        return alpha_futures_handler(ctx) or []
    except Exception:
        return []


def _safe_alpha(ctx):
    try:
        from chad.strategies import alpha_handler
        return alpha_handler(ctx) or []
    except Exception:
        return []


def _safe_alpha_crypto(ctx):
    try:
        from chad.strategies import alpha_crypto_handler
        return alpha_crypto_handler(ctx) or []
    except Exception:
        return []


def _safe_gamma(ctx):
    try:
        from chad.strategies import gamma_handler
        return gamma_handler(ctx) or []
    except Exception:
        return []


def _safe_beta(ctx):
    try:
        from chad.strategies import beta_handler
        return beta_handler(ctx) or []
    except Exception:
        return []


def build_routed_signals(logger: logging.Logger | None = None) -> RoutedBuildResult:
    logger = logger or logging.getLogger("chad.routed_execution")

    result = ContextBuilder().build()
    ctx = result.context

    available_signals = {
        "alpha_futures": _safe_alpha_futures(ctx),
        "alpha": _safe_alpha(ctx),
        "alpha_crypto": _safe_alpha_crypto(ctx),
        "gamma": _safe_gamma(ctx),
        "beta": _safe_beta(ctx),
    }

    weights = load_allocation_weights()

    decision = choose_strategy_route(
        available_signals=available_signals,
        weights=weights,
    )

    selected = {}
    if decision.selected_strategy:
        selected[decision.selected_strategy] = list(
            available_signals.get(decision.selected_strategy, []) or []
        )

    logger.info(
        "Route selected strategy=%s symbols=%s counts=%s weights=%s",
        decision.selected_strategy,
        decision.selected_symbols,
        decision.available_counts,
        decision.weights,
    )

    return RoutedBuildResult(
        selected_strategy=decision.selected_strategy,
        selected_symbols=decision.selected_symbols,
        available_counts=decision.available_counts,
        weights=decision.weights,
        selected_signals=selected,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = build_routed_signals()
    print(out)

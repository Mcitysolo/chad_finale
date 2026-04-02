#!/usr/bin/env python3
"""
chad/core/live_execution_router.py

Final routing layer:
Chooses best available strategy and returns execution-ready signals.

Supports two modes (SSOT v6.4):
- Single-winner: build_live_signals() — returns only the top strategy's signals
- Always-active:  build_all_live_signals() — returns signals from ALL active strategies

Mode is controlled by CHAD_ALWAYS_ACTIVE_ROUTING env var (default: off).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

from chad.utils.context_builder import ContextBuilder
from chad.portfolio.capital_allocator import load_allocation_weights
from chad.portfolio.strategy_router import (
    choose_strategy_route,
    evaluate_all_strategies,
    AllStrategiesDecision,
    RouteDecision,
)


def _safe_import(name: str, fn: str):
    try:
        module = __import__(name, fromlist=[fn])
        return getattr(module, fn)
    except Exception:
        return None


def _run_handler(handler, ctx):
    try:
        if handler is None:
            return []
        return handler(ctx) or []
    except Exception:
        return []


def _build_available_signals(logger: logging.Logger) -> Tuple[Dict[str, List], Dict[str, float]]:
    """Shared signal-building logic for both routing modes."""
    ctx = ContextBuilder().build().context

    handlers = {
        "alpha_futures": _safe_import("chad.strategies.alpha_futures", "alpha_futures_handler"),
        "alpha": _safe_import("chad.strategies", "alpha_handler"),
        "alpha_crypto": _safe_import("chad.strategies", "alpha_crypto_handler"),
        "gamma": _safe_import("chad.strategies", "gamma_handler"),
        "beta": _safe_import("chad.strategies", "beta_handler"),
    }

    available_signals: Dict[str, List] = {
        name: _run_handler(handler, ctx)
        for name, handler in handlers.items()
    }

    weights = load_allocation_weights()
    return available_signals, weights


def build_live_signals(logger: logging.Logger | None = None):
    """Single-winner routing: returns only the top strategy's signals."""
    logger = logger or logging.getLogger("chad.live_router")

    available_signals, weights = _build_available_signals(logger)

    decision = choose_strategy_route(
        available_signals=available_signals,
        weights=weights,
    )

    selected = decision.selected_strategy

    if selected is None:
        logger.warning("No strategies available. System idle.")
        return []

    signals = available_signals.get(selected, [])

    logger.info(
        "LIVE ROUTE → strategy=%s symbols=%s",
        selected,
        decision.selected_symbols,
    )

    return signals


@dataclass
class AllSignalsResult:
    """Result from always-active routing."""
    all_signals: List                         # flat list of all signals from all active strategies
    signals_by_strategy: Dict[str, List]      # strategy -> signals
    decision: AllStrategiesDecision           # full routing decision for tracing


def build_all_live_signals(logger: logging.Logger | None = None) -> AllSignalsResult:
    """
    Always-active routing: return signals from ALL strategies that
    produced at least one signal, plus the AllStrategiesDecision for tracing.
    """
    logger = logger or logging.getLogger("chad.live_router")

    available_signals, weights = _build_available_signals(logger)

    decision = evaluate_all_strategies(
        available_signals=available_signals,
        weights=weights,
    )

    if not decision.active_strategies:
        logger.warning("No strategies available (always-active). System idle.")
        return AllSignalsResult(
            all_signals=[],
            signals_by_strategy={},
            decision=decision,
        )

    # Collect signals from all active strategies in priority order
    all_signals: List = []
    signals_by_strategy: Dict[str, List] = {}

    for strat_name in decision.ordered_strategies:
        strat_signals = list(available_signals.get(strat_name, []) or [])
        if strat_signals:
            signals_by_strategy[strat_name] = strat_signals
            all_signals.extend(strat_signals)

    logger.info(
        "ALWAYS-ACTIVE ROUTE → active=%s primary=%s total_signals=%d",
        decision.ordered_strategies,
        decision.primary_strategy,
        len(all_signals),
    )

    return AllSignalsResult(
        all_signals=all_signals,
        signals_by_strategy=signals_by_strategy,
        decision=decision,
    )


def is_always_active_routing() -> bool:
    """Check if always-active routing is enabled via env var."""
    return os.getenv("CHAD_ALWAYS_ACTIVE_ROUTING", "0").strip() == "1"

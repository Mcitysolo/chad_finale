#!/usr/bin/env python3
"""
chad/core/live_execution_router.py

Final routing layer:
Chooses best available strategy and returns execution-ready signals.

This is what makes CHAD always active.
"""

from __future__ import annotations

from typing import Dict, List
import logging

from chad.utils.context_builder import ContextBuilder
from chad.portfolio.capital_allocator import load_allocation_weights
from chad.portfolio.strategy_router import choose_strategy_route


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


def build_live_signals(logger: logging.Logger | None = None):
    logger = logger or logging.getLogger("chad.live_router")

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

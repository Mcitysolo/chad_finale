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
from chad.core.context_positions import build_cycle_context, filter_overlay_owned_exits
from chad.portfolio.capital_allocator import load_allocation_weights
from chad.portfolio.strategy_router import (
    choose_strategy_route,
    evaluate_all_strategies,
    AllStrategiesDecision,
    RouteDecision,
)


def _run_handler(name: str, handler, ctx, logger: logging.Logger) -> List:
    """Run a strategy handler, logging any exception at WARNING level."""
    try:
        if handler is None:
            return []
        return handler(ctx) or []
    except Exception as exc:
        logger.warning("Strategy handler %s raised %s: %s", name, type(exc).__name__, exc)
        return []


def _build_available_signals(logger: logging.Logger) -> Tuple[Dict[str, List], Dict[str, float]]:
    """
    Shared signal-building logic for both routing modes.

    Loads all registered strategies from the canonical registry and
    evaluates each handler against the current market context.

    W2B: the market context is built via ``build_cycle_context`` so strategies
    can see the CHAD-attributed positions when ``CHAD_CTX_POSITIONS`` is on.
    OFF (default) is byte-identical to the legacy ``ContextBuilder().build()``.
    A ``None`` result means positions are UNKNOWN in ON mode (D3) — idle this
    cycle (no signals), never run strategies on a false-empty book.
    """
    result, _view, _mode = build_cycle_context(logger=logger)
    if result is None:
        return {}, load_allocation_weights()
    ctx = result.context

    from chad.strategies import iter_strategy_registrations
    regs = list(iter_strategy_registrations())

    available_signals: Dict[str, List] = {}
    for reg in regs:
        name = reg.name.value  # StrategyName enum → string
        available_signals[name] = _run_handler(name, reg.handler, ctx, logger)

    # W2B-5: D4 double-exit guardrail. In ON mode the injected book makes the
    # strategies' native qty>0 exits reachable — unclamped equity/ETF SELLs that
    # would race the ACTIVE exit overlay (the sole equity/ETF exit authority).
    # Drop exactly those here so ON delivers BUY-suppression + beta-weighting with
    # ZERO new SELL surface. INERT unless ON (filter returns the input unchanged),
    # so OFF stays byte-identical. Futures/crypto/options SELLs are untouched.
    _exits_filtered = 0
    if _mode == "on":
        for name in list(available_signals):
            kept, dropped = filter_overlay_owned_exits(available_signals[name], mode=_mode)
            if dropped:
                available_signals[name] = kept
                _exits_filtered += len(dropped)
        if _exits_filtered:
            logger.warning(
                "CTX_POSITIONS_EXIT_FILTERED site=router dropped=%d "
                "(overlay is sole equity/ETF exit authority — D4)",
                _exits_filtered,
            )

    # W2B-3 + W2BS (Q4): shadow-compare + heartbeat. The HEARTBEAT is now written
    # EVERY cycle — including OFF — so the health monitor (R25) can distinguish
    # "installed but off" (fresh off-heartbeat) from "the instrument is dead"
    # (stale/missing). This is observer-only: OFF stays byte-identical in its
    # TRADING path — record_cycle in off mode computes NO counterfactual, writes
    # NO evidence ndjson, and the D4 guardrail already returned the signals
    # unchanged; it only stamps a liveness file that no trading decision reads.
    # The shadow DIFF (+ evidence) still happens only in shadow; the guardrail
    # only acts in on. Best-effort: an observability failure never breaks a cycle.
    try:
        from chad.core.ctx_positions_shadow import record_cycle
        record_cycle(
            mode=_mode, regs=regs, run_handler=_run_handler,
            ctx_off=ctx, available_off=available_signals, view=_view, logger=logger,
            exits_filtered=_exits_filtered,
        )
    except Exception as exc:  # pragma: no cover - non-fatal observability
        logger.warning("ctx_positions shadow/heartbeat failed (non-fatal): %s", exc)

    weights = load_allocation_weights()
    return available_signals, weights


@dataclass
class SingleWinnerResult:
    """Result from single-winner routing, preserving the RouteDecision for tracing."""
    signals: List
    decision: RouteDecision


def build_live_signals(logger: logging.Logger | None = None) -> SingleWinnerResult:
    """Single-winner routing: returns the top strategy's signals + RouteDecision."""
    logger = logger or logging.getLogger("chad.live_router")

    available_signals, weights = _build_available_signals(logger)

    decision = choose_strategy_route(
        available_signals=available_signals,
        weights=weights,
    )

    selected = decision.selected_strategy

    if selected is None:
        logger.warning("No strategies available. System idle.")
        return SingleWinnerResult(signals=[], decision=decision)

    signals = available_signals.get(selected, [])

    logger.info(
        "LIVE ROUTE → strategy=%s symbols=%s",
        selected,
        decision.selected_symbols,
    )

    return SingleWinnerResult(signals=signals, decision=decision)


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

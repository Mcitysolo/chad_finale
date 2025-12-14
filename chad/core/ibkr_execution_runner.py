"""
IBKR Execution Runner (Phase 4b + 5 + 8-prep – WHAT-IF Only, SCR + LiveGate)

This module runs a **single, safe execution pass** that:

    Polygon NDJSON  ->  ContextBuilder
                      -> StrategyEngine (Alpha/Beta, etc.)
                      -> DecisionPipeline (Policy + Router)
                      -> ExecutionPlan (PlannedOrder objects)
                      -> IBKR StrategyTradeIntent
                      -> Shadow Confidence Router (SCR + Shadow Router)
                      -> LiveGate (ExecConfig + CHAD_MODE + SCR)
                      -> IBKRExecutor.execute_with_risk(..., live=False)

Key properties (current Phase-7 build)
--------------------------------------
* Uses the SAME pipeline as `full_execution_cycle` to build intents.
* Applies Shadow Confidence Router and Shadow Router to gate per-intent flow.
* Evaluates global LiveGateDecision (ExecutionConfig + CHAD_MODE + SCR).
* Connects to IB Gateway via `IBKRExecutor` in WHAT-IF mode only (live=False).
* EVEN IF CHAD_MODE=LIVE and CHAD_EXECUTION_MODE=ibkr_live:
    - ExecutionConfig is hard-locked to DRY_RUN for IBKR.
    - LiveGateDecision.allow_ibkr_live will be False.
    - This runner STILL calls execute_with_risk(..., live=False).

In other words:
    This runner is a **full risk & gating rehearsal** for Phase 8,
    but it does NOT place LIVE orders in this build.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, List, Tuple

from chad.engine import StrategyEngine
from chad.execution.execution_pipeline import (
    build_execution_plan,
    build_ibkr_intents_from_plan,
)
from chad.execution.ibkr_executor import (
    IBKRExecutor,
    StrategyTradeIntent as IBKRStrategyTradeIntent,
    _build_executor_from_env,
)
from chad.strategies import register_core_strategies
from chad.utils.context_builder import ContextBuilder
from chad.utils.pipeline import DecisionPipeline, PipelineConfig
from chad.utils.signal_router import SignalRouter

from chad.analytics.trade_stats_engine import load_and_compute
from chad.analytics.shadow_confidence_router import evaluate_confidence, ShadowState
from chad.analytics.shadow_router import route_orders, summarize_routing
from chad.core.live_gate import evaluate_live_gate


LOGGER_NAME = "chad.ibkr_execution_runner"


def _get_logger() -> logging.Logger:
    """
    Obtain a logger configured consistently with other CHAD components.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def _build_intents_from_pipeline(
    logger: logging.Logger,
) -> Tuple[Any, Any, List[IBKRStrategyTradeIntent]]:
    """
    Run the same logical pipeline as `full_execution_cycle` and return
    (context, execution_plan, ibkr_intents).

    This does NOT touch any broker; it only computes what CHAD *would* do.
    """
    logger.info("Building context and IBKR intents via decision pipeline...")

    builder = ContextBuilder()
    result = builder.build()

    ctx = result.context
    prices = result.prices
    current_symbol_notional = result.current_symbol_notional
    current_total_notional = result.current_total_notional

    logger.info(
        "Context built: now=%s symbols=%s total_notional=%s",
        getattr(ctx, "now", None),
        sorted(prices.keys()),
        current_total_notional,
    )

    engine = StrategyEngine()
    register_core_strategies(engine)

    router = SignalRouter()
    config = PipelineConfig(use_policy=True)
    pipeline = DecisionPipeline(engine=engine, router=router, config=config)

    pipeline_result = pipeline.run(
        ctx=ctx,
        prices=prices,
        current_symbol_notional=current_symbol_notional,
        current_total_notional=current_total_notional,
    )

    logger.info(
        "PipelineResult: raw=%s evaluated=%s routed=%s",
        getattr(pipeline_result, "raw_signals_count", None),
        getattr(pipeline_result, "evaluated_signals_count", None),
        getattr(pipeline_result, "routed_signals_count", None),
    )

    routed_signals = getattr(pipeline_result, "routed_signals", None) or []
    if not routed_signals:
        logger.info("No routed signals produced – nothing to send to IBKR.")
        return ctx, None, []

    execution_plan = build_execution_plan(
        routed_signals=routed_signals,
        prices=prices,
    )

    logger.info(
        "ExecutionPlan: orders=%s total_notional=%s",
        getattr(execution_plan, "orders_count", None),
        getattr(execution_plan, "total_notional", None),
    )

    ibkr_intents = build_ibkr_intents_from_plan(execution_plan)
    logger.info("IBKR intents built: count=%d", len(ibkr_intents))

    for idx, intent in enumerate(ibkr_intents, start=1):
        logger.info("Intent[%d]: %s", idx, repr(intent))

    return ctx, execution_plan, ibkr_intents


def _compute_shadow_state(logger: logging.Logger) -> ShadowState:
    """
    Compute ShadowState (SCR) from recent trade history.
    """
    logger.info("Loading recent trade stats for SCR evaluation...")
    stats = load_and_compute(
        max_trades=200,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    shadow_state = evaluate_confidence(stats)

    logger.info(
        "ShadowState: state=%s sizing_factor=%.3f paper_only=%s",
        shadow_state.state,
        shadow_state.sizing_factor,
        shadow_state.paper_only,
    )
    for reason in shadow_state.reasons:
        logger.info("SCR reason: %s", reason)

    return shadow_state


def run_ibkr_execution(dry_run: bool = True) -> int:
    """
    Run a single IBKR execution pass using WHAT-IF orders, gated by SCR + LiveGate.

    Steps:
        1. Compute ShadowState (SCR) from recent trade stats.
        2. Build context + execution plan + IBKR intents via CHAD pipeline.
        3. Route intents via Shadow Router to decide per-intent eligibility.
        4. Compute global LiveGateDecision (ExecutionConfig + CHAD_MODE + SCR).
        5. If no intents allowed (or LiveGate denies), exit successfully (no-op).
        6. Connect to IB Gateway using IBKRExecutor.
        7. For each allowed intent, call `execute_with_risk(intent, live=False)`
           (WHAT-IF only in this Phase-7/8-prep build).
        8. Log all results; fail cleanly if anything goes wrong.

    NOTE:
        The `dry_run` flag is currently enforced to True; this function will
        never place LIVE orders in this Phase-4b/5/7 build. Even if the
        LiveGate says LIVE is theoretically allowed in the future, this runner
        still uses live=False.
    """
    logger = _get_logger()
    logger.info(
        "=== CHAD IBKR Execution Runner (WHAT-IF / DRY-RUN ONLY, SCR + LiveGate) ==="
    )

    if not dry_run:
        # Hard safety gate – we do NOT allow live trading through this runner yet.
        logger.warning(
            "dry_run=False requested, but LIVE trading is not enabled via this runner. "
            "Proceeding in WHAT-IF (live=False) mode."
        )
        dry_run = True

    # 1) Shadow Confidence (SCR).
    try:
        shadow_state = _compute_shadow_state(logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to compute ShadowState for SCR: %s", exc)
        return 1

    # 2) Build context + plan + intents.
    try:
        _, execution_plan, ibkr_intents = _build_intents_from_pipeline(logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build IBKR intents from pipeline: %s", exc)
        return 2

    if not ibkr_intents:
        logger.info("No IBKR intents – nothing to execute against IBKR.")
        logger.info("=== IBKR Execution Runner completed (no-op: no intents) ===")
        return 0

    # 3) Apply Shadow Router gating per intent.
    routed_orders = route_orders(shadow_state, ibkr_intents)
    routing_summary = summarize_routing(routed_orders, shadow_state)

    logger.info(
        "Shadow Router summary: total=%d live_allowed=%d live_blocked=%d "
        "state=%s sizing_factor=%.3f paper_only=%s",
        routing_summary.total,
        routing_summary.live_allowed,
        routing_summary.live_blocked,
        routing_summary.state,
        routing_summary.sizing_factor,
        routing_summary.paper_only,
    )
    for reason in routing_summary.reasons:
        logger.info("Shadow Router reason: %s", reason)

    if routing_summary.live_allowed == 0:
        logger.info(
            "No intents allowed for live execution by Shadow Router – "
            "skipping IBKR calls. Paper execution/simulation should still occur elsewhere."
        )
        logger.info("=== IBKR Execution Runner completed (ShadowRouter no-op) ===")
        return 0

    # 4) Compute global LiveGateDecision – unifies ExecutionConfig, CHAD_MODE, SCR.
    try:
        live_decision = evaluate_live_gate()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to evaluate LiveGate decision: %s", exc)
        return 3

    logger.info(
        "LiveGateDecision: allow_ibkr_live=%s allow_ibkr_paper=%s",
        live_decision.allow_ibkr_live,
        live_decision.allow_ibkr_paper,
    )
    for reason in live_decision.reasons:
        logger.info("LiveGate reason: %s", reason)

    # In this build, we only care if IBKR is allowed for PAPER/Dry-Run.
    if not live_decision.allow_ibkr_paper:
        logger.info(
            "LiveGate denies IBKR even for paper/dry-run – skipping execution entirely."
        )
        logger.info("=== IBKR Execution Runner completed (LiveGate no-op) ===")
        return 0

    # 5) Connect to IBKR via IBKRExecutor.
    try:
        executor: IBKRExecutor = _build_executor_from_env()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create IBKRExecutor from environment: %s", exc)
        return 4

    logger.info(
        "IBKRExecutor successfully created – starting WHAT-IF execution for "
        "%d routed intents...", routing_summary.live_allowed
    )

    successes = 0
    failures = 0

    for idx, routed in enumerate(routed_orders, start=1):
        if not routed.allowed_live:
            # Skip live execution; order may still be logged/paper-traded elsewhere.
            logger.info(
                "Skipping live execution for intent[%d]: blocked by SCR/Shadow Router. "
                "Reason: %s",
                idx,
                routed.reason,
            )
            continue

        intent = routed.original_intent
        logger.info(
            "Executing intent[%d] in WHAT-IF mode "
            "(SCR state=%s, sizing_factor=%.3f, CHAD live_decision.allow_ibkr_live=%s): %s",
            idx,
            shadow_state.state,
            shadow_state.sizing_factor,
            live_decision.allow_ibkr_live,
            repr(intent),
        )

        try:
            # NOTE:
            #   * We currently do NOT pass live=True under ANY circumstances.
            #   * Even if LiveGate later allows LIVE in Phase 8, this runner
            #     must be explicitly updated to use live_decision.allow_ibkr_live.
            result = executor.execute_with_risk(intent=intent, live=False)
            logger.info("Result for intent[%d]: %s", idx, repr(result))
            successes += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.exception("Error executing intent[%d] against IBKR: %s", idx, exc)

    logger.info(
        "IBKR execution summary: successes=%d failures=%d",
        successes,
        failures,
    )

    if failures > 0:
        logger.warning(
            "At least one IBKR WHAT-IF execution failed – inspect logs before proceeding."
        )
        return 5

    logger.info(
        "=== IBKR Execution Runner completed successfully "
        "(WHAT-IF, SCR + LiveGate-gated) ==="
    )
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    """
    Parse CLI arguments.

    For now, only a `--dry-run` flag is provided and defaults to True.
    LIVE mode is intentionally not exposed via this runner in Phase-4b/5/7.
    """
    parser = argparse.ArgumentParser(
        prog="chad.core.ibkr_execution_runner",
        description=(
            "Run a single IBKR WHAT-IF execution pass using CHAD's "
            "decision pipeline, Shadow Confidence Router, and LiveGate."
        ),
    )

    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Run in WHAT-IF (non-live) mode against IBKR (default: True).",
    )


    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """
    CLI entrypoint:

        python -m chad.core.ibkr_execution_runner
    """
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)
    exit_code = run_ibkr_execution(dry_run=args.dry_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

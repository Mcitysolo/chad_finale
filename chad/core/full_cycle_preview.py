from __future__ import annotations

"""
chad/core/full_cycle_preview.py

Single-cycle, read-only CHAD preview CLI.

This module runs ONE full logical decision cycle, end-to-end, without talking
to any broker:

    Polygon NDJSON  ->  ContextBuilder
                      -> StrategyEngine (Alpha/Beta, etc.)
                      -> DecisionPipeline (Policy + Router)
                      -> ExecutionPlan (PlannedOrder objects)
                      -> IBKR StrategyTradeIntent preview

It is designed for:
    * sanity checks (is CHAD “thinking” correctly given today’s data?)
    * debugging strategy + policy behaviour without touching IBKR/Kraken
    * operator visibility (what would CHAD try to trade right now?)

NO network I/O to brokers is performed: this is purely data-plane.
"""

import argparse
import sys
from typing import NoReturn

from chad.engine import StrategyEngine
from chad.execution.execution_pipeline import (
    ExecutionPlan,
    PlannedOrder,
    build_execution_plan,
    build_ibkr_intents_from_plan,
)
from chad.strategies import register_core_strategies
from chad.utils.context_builder import ContextBuilder
from chad.utils.pipeline import DecisionPipeline, PipelineConfig
from chad.utils.signal_router import SignalRouter


def _run_single_cycle() -> int:
    """
    Run a single, read-only CHAD decision cycle and print a summary.

    Returns:
        Exit code (0 for success, non-zero for recoverable errors).
    """
    builder = ContextBuilder()
    try:
        result = builder.build()
    except Exception as exc:  # noqa: BLE001
        print(f"[full_cycle_preview] ERROR building context: {exc}", file=sys.stderr)
        return 1

    ctx = result.context
    prices = result.prices
    current_symbol_notional = result.current_symbol_notional
    current_total_notional = result.current_total_notional

    # Strategy engine + strategies
    engine = StrategyEngine()
    register_core_strategies(engine)

    # Router + pipeline (policy-enabled by default)
    router = SignalRouter()
    config = PipelineConfig(use_policy=True)
    pipeline = DecisionPipeline(engine=engine, router=router, config=config)

    try:
        pipeline_result = pipeline.run(
            ctx=ctx,
            prices=prices,
            current_symbol_notional=current_symbol_notional,
            current_total_notional=current_total_notional,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[full_cycle_preview] ERROR in DecisionPipeline.run: {exc}", file=sys.stderr)
        return 2

    # Build execution plan from routed signals
    plan: ExecutionPlan = build_execution_plan(
        routed_signals=pipeline_result.routed_signals,
        prices=prices,
    )

    # Build IBKR intents (equity-like only) — still NO network I/O
    ibkr_intents = build_ibkr_intents_from_plan(plan)

    # ------------------------------------------------------------------ #
    # Human-readable summary                                            #
    # ------------------------------------------------------------------ #

    print("=== CHAD Full Cycle Preview (Read-Only) ===")
    print(f"now: {ctx.now.isoformat()}")
    print(f"tick_symbols: {sorted(ctx.ticks.keys())}")
    print(f"legend_num_symbols: {len(ctx.legend.weights)}")

    print("\n--- PipelineResult counts ---")
    print(f"raw_signals:       {len(pipeline_result.raw_signals)}")
    print(f"evaluated_signals: {len(pipeline_result.evaluated_signals)}")
    print(f"routed_signals:    {len(pipeline_result.routed_signals)}")

    print("\n--- ExecutionPlan ---")
    print(f"orders_count:   {len(plan.orders)}")
    print(f"total_notional: {plan.total_notional:.2f}")

    for order in plan.orders:
        _print_planned_order(order)

    print("\n--- IBKR StrategyTradeIntents (preview only) ---")
    print(f"intents_count: {len(ibkr_intents)}")
    for intent in ibkr_intents:
        print(
            "  intent:",
            f"strategy={intent.strategy}",
            f"symbol={intent.symbol}",
            f"side={intent.side}",
            f"sec_type={intent.sec_type}",
            f"exchange={intent.exchange}",
            f"currency={intent.currency}",
            f"order_type={intent.order_type}",
            f"quantity={intent.quantity}",
            f"notional_estimate={intent.notional_estimate:.2f}",
        )

    print("\n[full_cycle_preview] NOTE: No broker calls were made. This is a logical preview only.")
    return 0


def _print_planned_order(order: PlannedOrder) -> None:
    """
    Pretty-print a single PlannedOrder for humans.
    """
    contributors = [s.name for s in order.contributing_strategies]
    print(
        "  order:",
        f"strategy={order.primary_strategy.name}",
        f"symbol={order.symbol}",
        f"side={order.side.name}",
        f"size={order.size}",
        f"price={order.price:.2f}",
        f"notional={order.notional:.2f}",
        f"asset_class={order.asset_class.value}",
        f"contributors={contributors}",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single CHAD decision cycle end-to-end (data → context → "
            "strategies → policy → execution plan) and print a read-only preview. "
            "No broker calls are made."
        )
    )
    # Reserved for future flags (e.g. selecting subset of strategies).
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> NoReturn:
    _parse_args(argv)  # currently unused, kept for future extensions
    code = _run_single_cycle()
    raise SystemExit(code)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

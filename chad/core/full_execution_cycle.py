from __future__ import annotations

"""
chad/core/full_execution_cycle.py

Single-cycle, execution-grade CHAD CLI (DRY_RUN-only in Phase 7).

This module runs ONE full logical decision cycle, end-to-end, using the same
pipeline as full_cycle_preview, but structured for production use:

    Polygon NDJSON  ->  ContextBuilder
                      -> StrategyEngine (Alpha/Beta, etc.)
                      -> DecisionPipeline (Policy + Router)
                      -> ExecutionPlan (PlannedOrder objects)
                      -> IBKR StrategyTradeIntent preview (no network I/O)

In this Phase 7 build, full_execution_cycle is **strictly DRY_RUN/preview**:
it does NOT call any broker executors. ExecutionConfig and LiveGate enforce
that all IBKR/Kraken usage is DRY_RUN. This CLI is intended for:

    * operational verification (is CHAD's logic consistent with risk caps?)
    * debugging strategy + policy behaviour with real market data
    * generating a machine-readable summary for dashboards/monitoring

Later Phase 8+ can extend this CLI to call executors in WHAT-IF or PAPER
mode, guarded by ExecutionConfig + LiveGate.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, NoReturn

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


RUNTIME_DIR = Path(__file__).resolve().parents[2] / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
LAST_SUMMARY_PATH = RUNTIME_DIR / "full_execution_cycle_last.json"


@dataclass
class FullCycleSummary:
    """
    Structured summary of a single full execution cycle.

    This is designed to be machine-friendly (JSON serializable) so it can be
    scraped by dashboards or downstream monitoring tools.
    """

    now_iso: str
    tick_symbols: List[str]
    legend_num_symbols: int

    raw_signals: int
    evaluated_signals: int
    routed_signals: int

    orders_count: int
    total_notional: float

    ibkr_intents_count: int


def _run_single_cycle() -> FullCycleSummary:
    """
    Run a single CHAD decision cycle and return a structured summary.

    This is intentionally side-effect-light: no broker calls are made, and we
    only read from data + write a JSON summary to runtime/.
    """
    builder = ContextBuilder()
    result = builder.build()

    ctx = result.context
    prices = result.prices
    current_symbol_notional = result.current_symbol_notional
    current_total_notional = result.current_total_notional

    # Strategy engine + strategies
    engine = StrategyEngine()
    register_core_strategies(engine)

    # Router + pipeline (policy-enabled)
    router = SignalRouter()
    config = PipelineConfig(use_policy=True)
    pipeline = DecisionPipeline(engine=engine, router=router, config=config)

    pipeline_result = pipeline.run(
        ctx=ctx,
        prices=prices,
        current_symbol_notional=current_symbol_notional,
        current_total_notional=current_total_notional,
    )

    # Build execution plan from routed signals
    plan: ExecutionPlan = build_execution_plan(
        routed_signals=pipeline_result.routed_signals,
        prices=prices,
    )

    # Build IBKR intents (equity-like only) — still NO network I/O
    ibkr_intents = build_ibkr_intents_from_plan(plan)

    summary = FullCycleSummary(
        now_iso=ctx.now.isoformat(),
        tick_symbols=sorted(ctx.ticks.keys()),
        legend_num_symbols=len(ctx.legend.weights),
        raw_signals=len(pipeline_result.raw_signals),
        evaluated_signals=len(pipeline_result.evaluated_signals),
        routed_signals=len(pipeline_result.routed_signals),
        orders_count=len(plan.orders),
        total_notional=float(plan.total_notional),
        ibkr_intents_count=len(ibkr_intents),
    )

    _write_summary_json(summary=summary, plan=plan, ibkr_intents=ibkr_intents)
    _print_human_summary(summary=summary, plan=plan, ibkr_intents=ibkr_intents)

    return summary


def _write_summary_json(
    *,
    summary: FullCycleSummary,
    plan: ExecutionPlan,
    ibkr_intents: List[Any],
) -> None:
    """
    Persist the summary + orders + intents to runtime/full_execution_cycle_last.json.
    """
    payload: Dict[str, Any] = {
        "summary": asdict(summary),
        "orders": [
            {
                "primary_strategy": o.primary_strategy.name,
                "symbol": o.symbol,
                "side": o.side.name,
                "size": o.size,
                "price": float(o.price),
                "notional": float(o.notional),
                "asset_class": o.asset_class.value,
                "contributors": [s.name for s in o.contributing_strategies],
            }
            for o in plan.orders
        ],
        "ibkr_intents": [
            {
                "strategy": str(i.strategy),
                "symbol": i.symbol,
                "side": str(i.side),
                "sec_type": str(i.sec_type),
                "exchange": i.exchange,
                "currency": i.currency,
                "order_type": str(i.order_type),
                "quantity": float(i.quantity),
                "notional_estimate": float(i.notional_estimate),
            }
            for i in ibkr_intents
        ],
    }

    tmp = LAST_SUMMARY_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(LAST_SUMMARY_PATH)


def _print_human_summary(
    *,
    summary: FullCycleSummary,
    plan: ExecutionPlan,
    ibkr_intents: List[Any],
) -> None:
    """
    Pretty-print a human-readable summary to stdout.
    """
    print("=== CHAD Full Execution Cycle (Phase 7 DRY_RUN) ===")
    print(f"now: {summary.now_iso}")
    print(f"tick_symbols: {summary.tick_symbols}")
    print(f"legend_num_symbols: {summary.legend_num_symbols}")

    print("\n--- PipelineResult counts ---")
    print(f"raw_signals:       {summary.raw_signals}")
    print(f"evaluated_signals: {summary.evaluated_signals}")
    print(f"routed_signals:    {summary.routed_signals}")

    print("\n--- ExecutionPlan ---")
    print(f"orders_count:   {summary.orders_count}")
    print(f"total_notional: {summary.total_notional:.2f}")

    for order in plan.orders:
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

    print("\n--- IBKR StrategyTradeIntents (preview only, no broker I/O) ---")
    print(f"intents_count: {summary.ibkr_intents_count}")
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

    print(
        "\n[full_execution_cycle] NOTE: No broker calls were made. "
        "This is a DRY_RUN logical execution cycle only."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single CHAD decision cycle end-to-end and emit both a "
            "machine-readable JSON summary and a human-readable preview. "
            "No broker calls are made in this Phase 7 build."
        )
    )
    # Reserved for future options (e.g. strategy subsets, dry-run toggles).
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> NoReturn:
    _parse_args(argv)  # kept for future flags, even if unused now
    _run_single_cycle()
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])

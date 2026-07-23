from __future__ import annotations

"""
chad/core/full_execution_cycle.py

CHAD Full Execution Cycle — Production-Grade Plan Generator (DRY_RUN)

Purpose
-------
Generate the *authoritative* runtime plan artifact used by paper_shadow_runner:

  runtime/full_execution_cycle_last.json

This script runs ONE full decision cycle end-to-end:
  ContextBuilder -> StrategyEngine -> DecisionPipeline -> ExecutionPlan -> IBKR intents (preview)

Hard guarantees
---------------
- DRY_RUN only: no broker execution calls are made.
- Deterministic for a given input snapshot (market data/context).
- Fail-safe: never crashes due to missing optional context (e.g., legend=None).
- Atomic write: runtime/full_execution_cycle_last.json is written via tmp+replace.
- Machine-friendly output: includes counts + orders + intents + diagnostics.
- Zero third-party dependencies.

Design patterns
---------------
- Dependency Injection (DI) for all major components.
- Single-responsibility helpers for: building pipeline, generating plan, persisting artifact.
- Strict schema normalization for downstream stability.

Why we changed it
-----------------
Your previous version crashed when ctx.legend was None:
  AttributeError: 'NoneType' object has no attribute 'weights'
This prevented plan generation, which blocked paper_shadow_runner with "no_plan".

This rewrite is bulletproof against:
- missing legend
- missing ticks
- missing prices
- missing optional portfolio fields

It also adds:
- richer diagnostic fields
- safer serialization
- stronger invariants (types, normalization)
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from chad.core.context_positions import build_cycle_context, filter_overlay_owned_exits
from chad.engine import StrategyEngine
from chad.execution.execution_pipeline import ExecutionPlan, build_execution_plan, build_ibkr_intents_from_plan
from chad.strategies import register_core_strategies
from chad.utils.context_builder import ContextBuilder
from chad.utils.pipeline import DecisionPipeline, PipelineConfig
from chad.utils.signal_router import SignalRouter


# ----------------------------
# Runtime paths
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
LAST_SUMMARY_PATH = RUNTIME_DIR / "full_execution_cycle_last.json"


# ----------------------------
# Small utilities
# ----------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if (v == v) else default  # NaN check
    except Exception:
        return default


def _safe_keys(mapping: Any) -> List[str]:
    try:
        if isinstance(mapping, Mapping):
            return sorted([str(k) for k in mapping.keys()])
    except Exception:
        pass
    return []


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


# ----------------------------
# DI contracts
# ----------------------------

class ContextProvider(Protocol):
    def build(self) -> Any: ...


class EngineProvider(Protocol):
    def create(self) -> StrategyEngine: ...


class PipelineProvider(Protocol):
    def create(self) -> DecisionPipeline: ...


# ----------------------------
# Output schema
# ----------------------------

@dataclass(frozen=True)
class FullCycleSummary:
    now_iso: str
    tick_symbols: List[str]
    legend_num_symbols: int

    raw_signals: int
    evaluated_signals: int
    routed_signals: int

    orders_count: int
    total_notional: float
    ibkr_intents_count: int

    # Diagnostics / provenance
    repo_root: str
    runtime_dir: str
    plan_path: str
    execution_mode: str


# ----------------------------
# Core builder
# ----------------------------

def _build_cycle_components() -> Tuple[ContextBuilder, StrategyEngine, DecisionPipeline, SignalRouter]:
    """
    Create all components required for a single cycle.
    Kept centralized to preserve invariants and simplify future expansion.
    """
    builder = ContextBuilder()

    engine = StrategyEngine()
    register_core_strategies(engine)

    router = SignalRouter()
    pipeline = DecisionPipeline(
        engine=engine,
        config=PipelineConfig(use_policy=True),
    )
    return builder, engine, pipeline, router


def _legend_count(ctx: Any) -> int:
    """
    Defensive legend symbol count.
    """
    try:
        legend = getattr(ctx, "legend", None)
        if legend is None:
            return 0
        weights = getattr(legend, "weights", None)
        if isinstance(weights, Mapping):
            return len(weights)
        return 0
    except Exception:
        return 0


def _run_single_cycle() -> FullCycleSummary:
    """
    Run one cycle and persist runtime/full_execution_cycle_last.json.
    """
    builder, engine, pipeline, router = _build_cycle_components()

    # W2B: position-aware context (CHAD_CTX_POSITIONS). OFF is byte-identical to
    # the legacy builder.build(). A None result means positions are UNKNOWN in ON
    # mode (D3) — persist an idle summary rather than plan on a false-empty book.
    result, _ctx_view, _ctx_mode = build_cycle_context(builder=builder)
    if result is None:
        return _persist_idle_cycle(reason="ctx_positions_unknown")
    ctx = result.context
    prices = result.prices or {}
    current_symbol_notional = result.current_symbol_notional or {}
    current_total_notional = result.current_total_notional or 0.0

    # Run pipeline
    pipeline_result = pipeline.run(
        ctx=ctx,
        prices=prices,
        current_symbol_notional=current_symbol_notional,
        current_total_notional=current_total_notional,
    )

    # W2B-5: D4 double-exit guardrail. In ON mode drop strategy equity/ETF exit
    # SELLs so the ACTIVE exit overlay stays the sole equity/ETF exit authority;
    # the persisted plan artifact then reflects exactly what would execute. INERT
    # unless ON (returns routed_signals unchanged) -> OFF byte-identical.
    _routed = list(getattr(pipeline_result, "routed_signals", []) or [])
    if _ctx_mode == "on":
        _routed, _dropped_exits = filter_overlay_owned_exits(_routed, mode=_ctx_mode)
        if _dropped_exits:
            # W4B-1 (J16): this site previously dropped SILENTLY (assigned,
            # never used) — parity log with the other two sites, plus advice
            # recording. Observer-only: never breaks the preview cycle.
            import logging as _logging
            _log = _logging.getLogger(__name__)
            _log.warning(
                "CTX_POSITIONS_EXIT_FILTERED site=full_cycle dropped=%d "
                "(overlay is sole equity/ETF exit authority — D4)",
                len(_dropped_exits),
            )
            try:
                from chad.core.exit_advice import record_dropped_urges
                record_dropped_urges(
                    _dropped_exits, site="full_cycle", view=_ctx_view, logger=_log,
                )
            except Exception as exc:  # pragma: no cover - non-fatal observability
                _log.warning("exit_advice recorder failed (non-fatal): %s", exc)

    # Build execution plan
    plan: ExecutionPlan = build_execution_plan(
        routed_signals=_routed,
        prices=prices,
    )

    # Build IBKR intents (preview only)
    ibkr_intents = build_ibkr_intents_from_plan(plan)

    now_dt: datetime = ctx.now if isinstance(getattr(ctx, "now", None), datetime) else _utc_now()

    from chad.execution.execution_config import get_execution_mode as _get_exec_mode
    summary = FullCycleSummary(
        now_iso=now_dt.isoformat(),
        tick_symbols=_safe_keys(getattr(ctx, "ticks", {})),
        legend_num_symbols=_legend_count(ctx),
        raw_signals=len(pipeline_result.raw_signals),
        evaluated_signals=len(pipeline_result.evaluated_signals),
        routed_signals=len(pipeline_result.routed_signals),
        orders_count=len(plan.orders),
        total_notional=float(getattr(plan, "total_notional", 0.0) or 0.0),
        ibkr_intents_count=len(ibkr_intents),
        repo_root=str(REPO_ROOT),
        runtime_dir=str(RUNTIME_DIR),
        plan_path=str(LAST_SUMMARY_PATH),
        execution_mode=_get_exec_mode().value,
    )

    _persist_plan_artifact(summary=summary, plan=plan, ibkr_intents=ibkr_intents)
    _print_human_summary(summary=summary, plan=plan, ibkr_intents=ibkr_intents)
    return summary


def _persist_idle_cycle(*, reason: str) -> FullCycleSummary:
    """W2B (D3): persist an honest idle summary when positions are UNKNOWN in ON
    mode. Zero counts, no orders/intents, and an ``idle_reason`` marker — never a
    plan built on a false-empty book."""
    from chad.execution.execution_config import get_execution_mode as _get_exec_mode

    now_dt = _utc_now()
    summary = FullCycleSummary(
        now_iso=now_dt.isoformat(),
        tick_symbols=[],
        legend_num_symbols=0,
        raw_signals=0,
        evaluated_signals=0,
        routed_signals=0,
        orders_count=0,
        total_notional=0.0,
        ibkr_intents_count=0,
        repo_root=str(REPO_ROOT),
        runtime_dir=str(RUNTIME_DIR),
        plan_path=str(LAST_SUMMARY_PATH),
        execution_mode=_get_exec_mode().value,
    )
    payload: Dict[str, Any] = {
        "counts": {
            "raw_signals": 0,
            "evaluated_signals": 0,
            "routed_signals": 0,
            "orders_count": 0,
            "ibkr_intents_count": 0,
            "total_notional": 0.0,
        },
        "summary": asdict(summary),
        "orders": [],
        "ibkr_intents": [],
        "idle_reason": reason,
    }
    _atomic_write_json(LAST_SUMMARY_PATH, payload)
    print(f"=== CHAD Full Execution Cycle: IDLE ({reason}) ===")
    return summary


def _persist_plan_artifact(*, summary: FullCycleSummary, plan: ExecutionPlan, ibkr_intents: List[Any]) -> None:
    """
    Persist runtime/full_execution_cycle_last.json as the authoritative downstream plan input.
    """
    orders_out: List[Dict[str, Any]] = []
    for o in plan.orders:
        # Defensive: not all fields are guaranteed across versions
        primary = getattr(o, "primary_strategy", None)
        side = getattr(o, "side", None)
        asset_class = getattr(o, "asset_class", None)
        contrib = getattr(o, "contributing_strategies", None)

        order_metadata = getattr(o, "metadata", None) or {}
        # Pass through PlannedOrder.metadata so downstream consumers
        # (paper_trade_executor, dashboards) see signal_meta — the BAG/
        # combo leg structure preserved from the survivor TradeSignal.
        order_dict = {
            "primary_strategy": getattr(primary, "name", str(primary)),
            "symbol": str(getattr(o, "symbol", "")),
            "side": getattr(side, "name", str(side)),
            "size": _safe_float(getattr(o, "size", 0.0)),
            "price": _safe_float(getattr(o, "price", 0.0)),
            "notional": _safe_float(getattr(o, "notional", 0.0)),
            "asset_class": getattr(asset_class, "value", str(asset_class)),
            "contributors": [getattr(s, "name", str(s)) for s in (contrib or [])],
        }
        if isinstance(order_metadata, dict) and order_metadata:
            order_dict["metadata"] = dict(order_metadata)
        orders_out.append(order_dict)

    intents_out: List[Dict[str, Any]] = []
    for i in ibkr_intents:
        intents_out.append(
            {
                "strategy": str(getattr(i, "strategy", "")),
                "symbol": str(getattr(i, "symbol", "")),
                "side": str(getattr(i, "side", "")),
                "sec_type": str(getattr(i, "sec_type", "")),
                "exchange": str(getattr(i, "exchange", "")),
                "currency": str(getattr(i, "currency", "")),
                "order_type": str(getattr(i, "order_type", "")),
                "quantity": _safe_float(getattr(i, "quantity", 0.0)),
                "notional_estimate": _safe_float(getattr(i, "notional_estimate", 0.0)),
            }
        )

    payload: Dict[str, Any] = {
        "counts": {
            "raw_signals": int(summary.raw_signals),
            "evaluated_signals": int(summary.evaluated_signals),
            "routed_signals": int(summary.routed_signals),
            "orders_count": int(summary.orders_count),
            "ibkr_intents_count": int(summary.ibkr_intents_count),
            "total_notional": float(summary.total_notional),
        },
        "summary": asdict(summary),
        "orders": orders_out,
        "ibkr_intents": intents_out,
    }

    _atomic_write_json(LAST_SUMMARY_PATH, payload)


def _print_human_summary(*, summary: FullCycleSummary, plan: ExecutionPlan, ibkr_intents: List[Any]) -> None:
    print("=== CHAD Full Execution Cycle (DRY_RUN Plan Generator) ===")
    print(f"now: {summary.now_iso}")
    print(f"tick_symbols_count: {len(summary.tick_symbols)}")
    print(f"legend_num_symbols: {summary.legend_num_symbols}")
    print("")
    print("--- Pipeline counts ---")
    print(f"raw_signals:       {summary.raw_signals}")
    print(f"evaluated_signals: {summary.evaluated_signals}")
    print(f"routed_signals:    {summary.routed_signals}")
    print("")
    print("--- Plan ---")
    print(f"orders_count:      {summary.orders_count}")
    print(f"total_notional:    {summary.total_notional:.2f}")
    print(f"ibkr_intents:      {summary.ibkr_intents_count}")
    print("")
    print(f"[artifact] wrote: {summary.plan_path}")


# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one CHAD decision cycle end-to-end and persist "
            "runtime/full_execution_cycle_last.json (DRY_RUN only)."
        )
    )
    # Future flags reserved intentionally (keeps stable interface)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _parse_args(argv)
    _run_single_cycle()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

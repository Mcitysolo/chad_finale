#!/usr/bin/env python3
"""
chad/core/ibkr_execution_runner.py

CHAD IBKR Execution Runner
==========================

Production-grade single-pass execution runner for IBKR-backed intents.

Design goals
------------
- Fail-closed by default.
- Never place live orders unless LiveGate explicitly allows them.
- Preserve paper / what-if behavior when live is not allowed.
- Produce deterministic, audit-grade JSON summaries.
- Remain compatible with surrounding CHAD modules using guarded adapters.
- Avoid hidden hard-locks inside the runner itself.

Core behavior
-------------
1) Build trading intents from the CHAD pipeline.
2) Compute SCR / shadow state.
3) Route intents through Shadow Router.
4) Evaluate LiveGate.
5) If paper lane is denied -> clean no-op.
6) If live lane is denied but paper lane is allowed -> execute in WHAT-IF mode.
7) If live lane is explicitly allowed -> execute with live=True.
8) Emit structured summary for auditing.

Important
---------
This runner does NOT force live trading on its own.
It only follows the final system authority:

    LiveGateDecision.allow_ibkr_live
    LiveGateDecision.allow_ibkr_paper

So in your current environment, where service config still sets:

    CHAD_EXECUTION_MODE=dry_run

this runner should continue to execute in safe WHAT-IF mode.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

from chad.analytics.shadow_confidence_router import evaluate_confidence
from chad.analytics.trade_stats_engine import load_and_compute
from chad.core.live_gate import evaluate_live_gate
from chad.engine import StrategyEngine
from chad.execution.execution_pipeline import build_execution_plan, build_ibkr_intents_from_plan
from chad.strategies import register_core_strategies
from chad.utils.context_builder import ContextBuilder
from chad.utils.pipeline import DecisionPipeline, PipelineConfig
from chad.utils.signal_router import SignalRouter

try:
    from chad.analytics.shadow_router import route_orders, summarize_routing
except Exception:  # noqa: BLE001
    route_orders = None
    summarize_routing = None


LOGGER_NAME = "chad.ibkr_execution_runner"


class ExitCode(IntEnum):
    OK = 0
    SHADOW_ERROR = 1
    PIPELINE_ERROR = 2
    LIVEGATE_ERROR = 3
    EXECUTOR_ERROR = 4
    PARTIAL_FAILURE = 5
    INVALID_ARGUMENT = 6


class SupportsExecuteWithRisk(Protocol):
    def execute_with_risk(self, intent: Any, live: bool = False) -> Any:
        ...


@dataclass(frozen=True)
class RunnerConfig:
    log_level: str = "INFO"
    max_trades_for_shadow: int = 500
    days_back_for_shadow: int = 30
    include_paper_stats: bool = True
    include_live_stats: bool = True
    json_output: bool = True
    fail_on_partial: bool = False
    dry_run_requested: bool = True

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunnerConfig":
        return cls(
            log_level=str(args.log_level).upper(),
            max_trades_for_shadow=int(args.max_trades_for_shadow),
            days_back_for_shadow=int(args.days_back_for_shadow),
            include_paper_stats=bool(args.include_paper_stats),
            include_live_stats=bool(args.include_live_stats),
            json_output=bool(args.json_output),
            fail_on_partial=bool(args.fail_on_partial),
            dry_run_requested=bool(args.dry_run),
        )


@dataclass(frozen=True)
class HostContext:
    hostname: str
    pid: int
    cwd: str
    repo_root: str
    ts_utc: str


@dataclass(frozen=True)
class ShadowSnapshot:
    state: str
    sizing_factor: float
    paper_only: bool
    reasons: list[str]
    stats: dict[str, Any]


@dataclass(frozen=True)
class LiveGateSnapshot:
    allow_exits_only: bool
    allow_ibkr_live: bool
    allow_ibkr_paper: bool
    operator_mode: str
    operator_reason: str
    reasons: list[str]
    execution: dict[str, Any]
    mode: dict[str, Any]
    shadow: dict[str, Any]


@dataclass(frozen=True)
class ExecutionRecord:
    index: int
    symbol: Optional[str]
    intent_repr: str
    routed: bool
    route_reason: str
    live_flag: bool
    ok: bool
    response_quality: str = "unknown"
    submitted_quantity: Optional[float] = None
    original_quantity: Optional[float] = None
    normalized_from_quantity: Optional[float] = None
    risk_allowed: Optional[bool] = None
    risk_reason: Optional[str] = None
    broker_status: Optional[str] = None
    what_if_has_margin: Optional[bool] = None
    what_if_has_commission: Optional[bool] = None
    result_repr: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class RunnerSummary:
    ok: bool
    exit_code: int
    host: HostContext
    config: RunnerConfig
    shadow: ShadowSnapshot
    live_gate: LiveGateSnapshot
    intents_total: int
    intents_routed: int
    executed_success: int
    executed_failed: int
    records: list[ExecutionRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, IntEnum):
        return int(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _repo_root() -> Path:
    env = (os.environ.get("CHAD_ROOT") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "chad").is_dir():
            return p
    return Path(__file__).resolve().parents[2]


def _host_context() -> HostContext:
    root = _repo_root()
    return HostContext(
        hostname=socket.gethostname(),
        pid=os.getpid(),
        cwd=str(Path.cwd()),
        repo_root=str(root),
        ts_utc=_utc_now_iso(),
    )


def _get_logger(level: str) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def _safe_repr(value: Any, max_len: int = 2000) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001
        text = f"<repr_failed:{type(exc).__name__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _compute_shadow_snapshot(cfg: RunnerConfig) -> ShadowSnapshot:
    stats_raw = load_and_compute(
        max_trades=cfg.max_trades_for_shadow,
        days_back=cfg.days_back_for_shadow,
        include_paper=cfg.include_paper_stats,
        include_live=cfg.include_live_stats,
    )
    state = evaluate_confidence(stats_raw)
    return ShadowSnapshot(
        state=str(getattr(state, "state", "UNKNOWN")),
        sizing_factor=float(getattr(state, "sizing_factor", 0.0)),
        paper_only=bool(getattr(state, "paper_only", True)),
        reasons=[str(x) for x in list(getattr(state, "reasons", []) or [])][:50],
        stats=dict(stats_raw),
    )


def _normalize_live_gate() -> LiveGateSnapshot:
    raw = evaluate_live_gate().to_dict()
    return LiveGateSnapshot(
        allow_exits_only=bool(raw.get("allow_exits_only", False)),
        allow_ibkr_live=bool(raw.get("allow_ibkr_live", False)),
        allow_ibkr_paper=bool(raw.get("allow_ibkr_paper", False)),
        operator_mode=str(raw.get("operator_mode", "")),
        operator_reason=str(raw.get("operator_reason", "")),
        reasons=[str(x) for x in list(raw.get("reasons", []) or [])][:100],
        execution=dict(raw.get("execution") or {}),
        mode=dict(raw.get("mode") or {}),
        shadow=dict(raw.get("shadow") or {}),
    )


def _call_with_fallbacks(fn: Callable[..., Any], *ordered_variants: Mapping[str, Any]) -> Any:
    last_exc: Optional[Exception] = None
    for kwargs in ordered_variants:
        try:
            return fn(**dict(kwargs))
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no_call_variants_provided")


def _build_context() -> Any:
    builder = ContextBuilder()
    candidates = ("build_context", "build", "create_context", "__call__")
    for name in candidates:
        method = getattr(builder, name, None)
        if callable(method):
            return method()
    raise RuntimeError("ContextBuilder has no supported build method")


def _build_engine() -> StrategyEngine:
    engine = StrategyEngine()
    try:
        registrations = register_core_strategies()
    except TypeError:
        registrations = None

    if registrations is not None:
        add_many = getattr(engine, "register_many", None)
        if callable(add_many):
            add_many(registrations)
            return engine

        add_one = getattr(engine, "register", None)
        if callable(add_one) and isinstance(registrations, Iterable):
            for item in registrations:
                add_one(item)
            return engine

    # Fallback: some CHAD builds let register_core_strategies mutate engine directly.
    try:
        register_core_strategies(engine)
    except TypeError:
        pass

    return engine


def _build_pipeline(engine: StrategyEngine) -> DecisionPipeline:
    """
    Build the DecisionPipeline using the repo-native wiring proven in:
      - chad/core/full_execution_cycle.py
      - chad/core/full_cycle_preview.py
    """
    from chad.policy import (
        PolicyEngine,
        build_default_strategy_limits,
        build_default_global_limits,
    )

    router = SignalRouter()

    policy = PolicyEngine(
        strategy_limits=build_default_strategy_limits(),
        global_limits=build_default_global_limits(),
    )

    config = PipelineConfig(use_policy=True)

    return DecisionPipeline(
        engine=engine,
        policy=policy,
        router=router,
        config=config,
    )


def _build_plan_and_intents(logger: logging.Logger) -> tuple[Any, Any, list[Any]]:
    """
    Build context -> pipeline_result -> execution plan -> IBKR intents
    using the canonical CHAD flow already validated elsewhere in this repo.
    """
    result = ContextBuilder().build()
    ctx = result.context
    prices = result.prices or {}
    current_symbol_notional = result.current_symbol_notional or {}
    current_total_notional = result.current_total_notional or 0.0

    engine = _build_engine()
    pipeline = _build_pipeline(engine)

    pipeline_result = pipeline.run(
        ctx=ctx,
        prices=prices,
        current_symbol_notional=current_symbol_notional,
        current_total_notional=current_total_notional,
    )

    plan = build_execution_plan(
        routed_signals=pipeline_result.routed_signals,
        prices=prices,
    )

    intents = build_ibkr_intents_from_plan(plan)
    intents_list = list(intents or [])

    _meta = getattr(pipeline_result, "meta", {}) or {}
    _passed = int(_meta.get("passed_signals", 0) or 0)
    logger.info(
        "Pipeline built raw=%d policy_records=%d passed=%d routed=%d plan=%d intents=%d",
        len(getattr(pipeline_result, "raw_signals", []) or []),
        len(getattr(pipeline_result, "evaluated_signals", []) or []),
        _passed,
        len(getattr(pipeline_result, "routed_signals", []) or []),
        len(getattr(plan, "orders", []) or []),
        len(intents_list),
    )

    return ctx, plan, intents_list


def _route_intents(logger: logging.Logger, intents: Sequence[Any], shadow: ShadowSnapshot) -> tuple[list[Any], dict[str, Any]]:
    if not intents:
        return [], {"live_allowed": 0, "total": 0, "mode": "empty"}

    if route_orders is None:
        logger.warning("shadow_router unavailable; using pass-through routing.")
        return list(intents), {"live_allowed": len(intents), "total": len(intents), "mode": "passthrough"}

    routed = _call_with_fallbacks(
        route_orders,
        {"intents": list(intents), "shadow_state": shadow},
        {"orders": list(intents), "shadow_state": shadow},
        {"intents": list(intents), "shadow_state": asdict(shadow)},
        {"orders": list(intents), "shadow_state": asdict(shadow)},
    )

    routed_list = list(routed or [])

    if summarize_routing is not None:
        try:
            summary = summarize_routing(routed_list, shadow)
            if hasattr(summary, "__dict__"):
                return routed_list, dict(summary.__dict__)
            if isinstance(summary, dict):
                return routed_list, summary
        except TypeError:
            try:
                summary = summarize_routing(routed_list, asdict(shadow))
                if hasattr(summary, "__dict__"):
                    return routed_list, dict(summary.__dict__)
                if isinstance(summary, dict):
                    return routed_list, summary
            except Exception as exc:  # noqa: BLE001
                logger.warning("summarize_routing failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summarize_routing failed: %s", exc)

    live_allowed = 0
    for item in routed_list:
        if _route_item_allowed(item):
            live_allowed += 1
    return routed_list, {"live_allowed": live_allowed, "total": len(routed_list), "mode": "shadow_router"}


def _route_item_allowed(item: Any) -> bool:
    for attr in ("allowed_for_live", "allow_live", "approved", "ok"):
        value = getattr(item, attr, None)
        if isinstance(value, bool):
            return value
    return True


def _route_item_reason(item: Any) -> str:
    for attr in ("reason", "route_reason", "block_reason", "decision_reason"):
        value = getattr(item, attr, None)
        if value is not None:
            return str(value)
    return "ok"


def _route_item_intent(item: Any) -> Any:
    for attr in ("original_intent", "intent", "order", "payload"):
        if hasattr(item, attr):
            return getattr(item, attr)
    return item



def _intent_quantity(intent: Any) -> Optional[float]:
    for attr in ("quantity", "qty", "size", "shares", "units", "volume"):
        try:
            value = getattr(intent, attr, None)
            if value is not None:
                return float(value)
        except Exception:
            pass
    if isinstance(intent, Mapping):
        for key in ("quantity", "qty", "size", "shares", "units", "volume"):
            try:
                value = intent.get(key)
                if value is not None:
                    return float(value)
            except Exception:
                pass
    return None


def _safe_nested_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default



def _extract_risk_gate_receipt(result: Any) -> dict[str, Any]:
    """
    Pull receipt fields from executor-native RiskGateResult when result is:
        (RiskGateResult(...), IBKRTradeResponse(...))
    """
    out = {
        "risk_allowed": None,
        "risk_reason": None,
        "receipt_original_quantity": None,
        "receipt_submitted_quantity": None,
        "receipt_normalized": None,
        "receipt_normalization_reason": None,
    }

    if not isinstance(result, tuple) or len(result) < 1:
        return out

    risk = result[0]
    if risk is None:
        return out

    out["risk_allowed"] = getattr(risk, "allowed", None)
    out["risk_reason"] = getattr(risk, "reason", None)
    out["receipt_original_quantity"] = getattr(risk, "original_quantity", None)
    out["receipt_submitted_quantity"] = getattr(risk, "submitted_quantity", None)
    out["receipt_normalized"] = getattr(risk, "normalized", None)
    out["receipt_normalization_reason"] = getattr(risk, "normalization_reason", None)
    return out


def _classify_execution_result(result: Any) -> dict[str, Any]:
    out = {
        "ok": False,
        "response_quality": "malformed_response",
        "submitted_quantity": None,
        "original_quantity": None,
        "normalized_from_quantity": None,
        "risk_allowed": None,
        "risk_reason": None,
        "broker_status": None,
        "what_if_has_margin": None,
        "what_if_has_commission": None,
    }

    if not isinstance(result, tuple) or len(result) != 2:
        return out

    risk_result, broker_resp = result

    risk_allowed = bool(_safe_nested_get(risk_result, "allowed", False))
    risk_reason = _safe_nested_get(risk_result, "reason", None)
    adjusted_notional = _safe_nested_get(risk_result, "adjusted_notional", None)

    out["risk_allowed"] = risk_allowed
    out["risk_reason"] = str(risk_reason) if risk_reason is not None else None

    if not risk_allowed:
        out["ok"] = True
        out["response_quality"] = "risk_blocked"
        return out

    if broker_resp is None:
        out["response_quality"] = "empty_response"
        return out

    broker_status = _safe_nested_get(broker_resp, "status", None)
    raw = _safe_nested_get(broker_resp, "raw", {}) or {}
    what_if = raw.get("what_if_order") if isinstance(raw, Mapping) else None
    what_if = what_if or {}

    out["broker_status"] = str(broker_status) if broker_status is not None else None

    has_commission = isinstance(what_if, Mapping) and ("commission" in what_if)
    has_margin = isinstance(what_if, Mapping) and (
        "initMarginAfter" in what_if
        or "maintMarginAfter" in what_if
        or "initMarginBefore" in what_if
        or "maintMarginBefore" in what_if
    )

    out["what_if_has_commission"] = bool(has_commission)
    out["what_if_has_margin"] = bool(has_margin)

    if str(broker_status).lower() == "what-if":
        if has_commission or has_margin:
            out["ok"] = True
            out["response_quality"] = "what_if_strong"
        elif isinstance(what_if, Mapping) and len(what_if) > 0:
            out["ok"] = True
            out["response_quality"] = "what_if_weak"
        else:
            out["ok"] = False
            out["response_quality"] = "empty_response"
        return out

    if broker_status:
        out["ok"] = True
        out["response_quality"] = "live_submitted"
        return out

    return out


def _intent_symbol(intent: Any) -> Optional[str]:
    for attr in ("symbol", "ticker", "instrument", "asset"):
        value = getattr(intent, attr, None)
        if value:
            return str(value)
    if isinstance(intent, Mapping):
        for key in ("symbol", "ticker", "instrument", "asset"):
            value = intent.get(key)
            if value:
                return str(value)
    return None


def _resolve_executor_factory() -> SupportsExecuteWithRisk:
    """
    Resolve the real IBKR executor using repo-native wiring.

    Proven by audit:
      - module: chad.execution.ibkr_executor
      - module-level factory: _build_executor_from_env(caps_path=None)

    We prefer the module-level builder because this repo's IBKRExecutor is
    dependency-injected and may not support zero-arg construction.
    """
    candidates: list[str] = [
        "chad.execution.ibkr_executor",
        "chad.brokers.ibkr_executor",
        "chad.execution.ibkr",
        "chad.ibkr.executor",
    ]

    errors: list[str] = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}:import:{type(exc).__name__}:{exc}")
            continue

        # 1) Preferred: module-level builder used by this repo
        build_fn = getattr(module, "_build_executor_from_env", None)
        if callable(build_fn):
            try:
                obj = build_fn(caps_path=None)
                if hasattr(obj, "execute_with_risk"):
                    return obj
                errors.append(f"{module_name}:_build_executor_from_env:no_execute_with_risk")
            except TypeError:
                try:
                    obj = build_fn()
                    if hasattr(obj, "execute_with_risk"):
                        return obj
                    errors.append(f"{module_name}:_build_executor_from_env:no_execute_with_risk")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{module_name}:_build_executor_from_env:{type(exc).__name__}:{exc}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{module_name}:_build_executor_from_env:{type(exc).__name__}:{exc}")

        # 2) Fallback: class-level factories if present
        cls = getattr(module, "IBKRExecutor", None)
        if cls is None:
            errors.append(f"{module_name}:missing:IBKRExecutor")
            continue

        for factory_name in ("from_env", "build_from_env", "create_from_env"):
            factory = getattr(cls, factory_name, None)
            if callable(factory):
                try:
                    obj = factory()
                    if hasattr(obj, "execute_with_risk"):
                        return obj
                    errors.append(f"{module_name}:{factory_name}:no_execute_with_risk")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{module_name}:{factory_name}:{type(exc).__name__}:{exc}")

        # 3) Final fallback: plain constructor
        try:
            obj = cls()
            if hasattr(obj, "execute_with_risk"):
                return obj
            errors.append(f"{module_name}:ctor:no_execute_with_risk")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}:ctor:{type(exc).__name__}:{exc}")

    detail = " | ".join(errors[:12]) if errors else "no_matching_factory"
    raise RuntimeError(f"unable_to_resolve_ibkr_executor:{detail}")


def run_ibkr_execution(cfg: RunnerConfig) -> RunnerSummary:
    logger = _get_logger(cfg.log_level)
    host = _host_context()
    notes: list[str] = []

    logger.info("=== CHAD IBKR Execution Runner starting ===")
    logger.info("Host=%s pid=%s repo_root=%s", host.hostname, host.pid, host.repo_root)

    if not cfg.dry_run_requested:
        logger.warning(
            "dry_run=False requested. Final live/paper behavior will still be determined "
            "by LiveGate and execution config."
        )
        notes.append("dry_run_requested_false_but_final_behavior_deferred_to_livegate")

    try:
        shadow = _compute_shadow_snapshot(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to compute shadow snapshot: %s", exc)
        return RunnerSummary(
            ok=False,
            exit_code=int(ExitCode.SHADOW_ERROR),
            host=host,
            config=cfg,
            shadow=ShadowSnapshot("ERROR", 0.0, True, [f"shadow_error:{type(exc).__name__}"], {}),
            live_gate=LiveGateSnapshot(False, False, False, "UNKNOWN", "shadow_failed", [], {}, {}, {}),
            intents_total=0,
            intents_routed=0,
            executed_success=0,
            executed_failed=0,
            notes=notes + [f"shadow_compute_failed:{type(exc).__name__}:{exc}"],
        )

    try:
        _, _, intents = _build_plan_and_intents(logger)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build intents from pipeline: %s", exc)
        return RunnerSummary(
            ok=False,
            exit_code=int(ExitCode.PIPELINE_ERROR),
            host=host,
            config=cfg,
            shadow=shadow,
            live_gate=LiveGateSnapshot(False, False, False, "UNKNOWN", "pipeline_failed", [], {}, {}, {}),
            intents_total=0,
            intents_routed=0,
            executed_success=0,
            executed_failed=0,
            notes=notes + [f"pipeline_build_failed:{type(exc).__name__}:{exc}"],
        )

    routed, routing_summary = _route_intents(logger, intents, shadow)
    routed_allowed = [item for item in routed if _route_item_allowed(item)]

    try:
        live_gate = _normalize_live_gate()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to evaluate LiveGate: %s", exc)
        return RunnerSummary(
            ok=False,
            exit_code=int(ExitCode.LIVEGATE_ERROR),
            host=host,
            config=cfg,
            shadow=shadow,
            live_gate=LiveGateSnapshot(False, False, False, "ERROR", "live_gate_failed", [str(exc)], {}, {}, {}),
            intents_total=len(intents),
            intents_routed=len(routed_allowed),
            executed_success=0,
            executed_failed=0,
            notes=notes + [f"live_gate_failed:{type(exc).__name__}:{exc}"],
        )

    logger.info(
        "LiveGateDecision allow_ibkr_live=%s allow_ibkr_paper=%s operator_mode=%s",
        live_gate.allow_ibkr_live,
        live_gate.allow_ibkr_paper,
        live_gate.operator_mode,
    )
    for reason in live_gate.reasons:
        logger.info("LiveGate reason: %s", reason)

    if not live_gate.allow_ibkr_paper:
        logger.info("Paper lane denied by LiveGate. Clean no-op.")
        notes.append("live_gate_denied_paper_lane")
        return RunnerSummary(
            ok=True,
            exit_code=int(ExitCode.OK),
            host=host,
            config=cfg,
            shadow=shadow,
            live_gate=live_gate,
            intents_total=len(intents),
            intents_routed=len(routed_allowed),
            executed_success=0,
            executed_failed=0,
            notes=notes,
        )

    live_flag = bool(live_gate.allow_ibkr_live)
    if live_flag:
        notes.append("live_flag_resolved_true")
        logger.warning("LiveGate explicitly разрешил live execution. Runner will pass live=True.")
    else:
        notes.append("live_flag_resolved_false")
        logger.info("LiveGate denied live lane. Runner will execute WHAT-IF / paper path only.")

    try:
        executor = _resolve_executor_factory()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to resolve IBKR executor: %s", exc)
        return RunnerSummary(
            ok=False,
            exit_code=int(ExitCode.EXECUTOR_ERROR),
            host=host,
            config=cfg,
            shadow=shadow,
            live_gate=live_gate,
            intents_total=len(intents),
            intents_routed=len(routed_allowed),
            executed_success=0,
            executed_failed=0,
            notes=notes + [f"executor_resolution_failed:{type(exc).__name__}:{exc}"],
        )

    records: list[ExecutionRecord] = []
    successes = 0
    failures = 0

    for idx, routed_item in enumerate(routed, start=1):
        allowed = _route_item_allowed(routed_item)
        reason = _route_item_reason(routed_item)
        intent = _route_item_intent(routed_item)
        symbol = _intent_symbol(intent)
        intent_repr = _safe_repr(intent)

        if not allowed:
            logger.info("Skipping intent[%d] blocked by Shadow Router. reason=%s", idx, reason)
            records.append(
                ExecutionRecord(
                    index=idx,
                    symbol=symbol,
                    intent_repr=intent_repr,
                    routed=False,
                    route_reason=reason,
                    live_flag=live_flag,
                    ok=True,
                    result_repr=None,
                    error=None,
                )
            )
            continue

        logger.info(
            "Executing intent[%d] live_flag=%s symbol=%s route_reason=%s",
            idx,
            live_flag,
            symbol,
            reason,
        )

        try:
            orig_qty = _intent_quantity(intent)
            result = executor.execute_with_risk(intent=intent, live=live_flag)
            result_repr = _safe_repr(result)
            classification = _classify_execution_result(result)
            receipt = _extract_risk_gate_receipt(result)

            submitted_qty = receipt.get("receipt_submitted_quantity")
            if submitted_qty is None:
                submitted_qty = classification.get("submitted_quantity")

            normalized_from_qty = None
            receipt_orig_qty = receipt.get("receipt_original_quantity")
            if receipt_orig_qty is not None and submitted_qty is not None:
                try:
                    if abs(float(receipt_orig_qty) - float(submitted_qty)) > 1e-12:
                        normalized_from_qty = float(receipt_orig_qty)
                except Exception:
                    pass
            elif orig_qty is not None and submitted_qty is not None:
                try:
                    if abs(float(orig_qty) - float(submitted_qty)) > 1e-12:
                        normalized_from_qty = float(orig_qty)
                except Exception:
                    pass
            if orig_qty is not None and classification.get("risk_allowed") is True:
                try:
                    risk_result, _broker_resp = result
                    adjusted_notional = getattr(risk_result, "adjusted_notional", None)
                    requested_notional = getattr(intent, "notional_estimate", None)
                    if (
                        orig_qty is not None
                        and requested_notional is not None
                        and adjusted_notional is not None
                        and float(requested_notional) > 0
                    ):
                        ratio = float(adjusted_notional) / float(requested_notional)
                        estimated_qty = float(orig_qty) * ratio
                        if getattr(intent, "sec_type", "").upper() == "STK":
                            submitted_qty = float(max(1, int(estimated_qty)))
                        else:
                            submitted_qty = float(estimated_qty)
                        if abs(float(submitted_qty) - float(orig_qty)) > 1e-9:
                            normalized_from_qty = float(orig_qty)
                except Exception:
                    pass

            logger.info(
                "Intent[%d] completed. symbol=%s quality=%s submitted_qty=%s orig_qty=%s result=%s",
                idx,
                symbol,
                classification.get("response_quality"),
                submitted_qty,
                orig_qty,
                result_repr,
            )
            logger.info(
                "EXECUTION_RESULT",
                extra={
                    "symbol": symbol,
                    "sec_type": getattr(intent, "sec_type", ""),
                    "exchange": getattr(intent, "exchange", ""),
                    "side": getattr(intent, "side", ""),
                    "quantity": submitted_qty,
                    "status": str(classification.get("broker_status") or classification.get("response_quality") or "unknown"),
                    "classification": str(classification.get("response_quality") or "unknown"),
                    "error": None,
                    "strategy": getattr(intent, "strategy", ""),
                    "ts_utc": _utc_now_iso(),
                },
            )

            records.append(
                ExecutionRecord(
                    index=idx,
                    symbol=symbol,
                    intent_repr=intent_repr,
                    routed=True,
                    route_reason=reason,
                    live_flag=live_flag,
                    ok=bool(classification.get("ok")),
                    response_quality=str(classification.get("response_quality") or "unknown"),
                    submitted_quantity=submitted_qty,
                    original_quantity=orig_qty,
                    normalized_from_quantity=normalized_from_qty,
                    risk_allowed=receipt.get("risk_allowed", classification.get("risk_allowed")),
                    risk_reason=receipt.get("risk_reason", classification.get("risk_reason")),
                    broker_status=classification.get("broker_status"),
                    what_if_has_margin=classification.get("what_if_has_margin"),
                    what_if_has_commission=classification.get("what_if_has_commission"),
                    result_repr=result_repr,
                    error=None,
                )
            )

            if classification.get("response_quality") in ("what_if_strong", "live_submitted"):
                successes += 1
            elif classification.get("response_quality") in ("risk_blocked", "what_if_weak"):
                notes.append(
                    f"intent_{idx}_{symbol or 'UNKNOWN'}_nonfatal_quality={classification.get('response_quality')}"
                )
            else:
                failures += 1
                notes.append(
                    f"intent_{idx}_{symbol or 'UNKNOWN'}_response_quality={classification.get('response_quality')}"
                )

        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.exception("Intent[%d] failed. symbol=%s error=%s", idx, symbol, exc)
            logger.info(
                "EXECUTION_RESULT",
                extra={
                    "symbol": symbol,
                    "sec_type": getattr(intent, "sec_type", ""),
                    "exchange": getattr(intent, "exchange", ""),
                    "side": getattr(intent, "side", ""),
                    "quantity": _intent_quantity(intent),
                    "status": "error",
                    "classification": "FAILED",
                    "error": f"{type(exc).__name__}:{exc}",
                    "strategy": getattr(intent, "strategy", ""),
                    "ts_utc": _utc_now_iso(),
                },
            )
            records.append(
                ExecutionRecord(
                    index=idx,
                    symbol=symbol,
                    intent_repr=intent_repr,
                    routed=True,
                    route_reason=reason,
                    live_flag=live_flag,
                    ok=False,
                    result_repr=None,
                    error=f"{type(exc).__name__}:{exc}",
                )
            )

    exit_code = ExitCode.OK
    ok = True

    if failures > 0 and cfg.fail_on_partial:
        ok = False
        exit_code = ExitCode.PARTIAL_FAILURE
        notes.append("partial_failures_detected")
    elif failures > 0:
        notes.append("partial_failures_detected_but_nonfatal")

    notes.append(f"routing_summary={json.dumps(routing_summary, default=_json_default, sort_keys=True)}")

    return RunnerSummary(
        ok=ok,
        exit_code=int(exit_code),
        host=host,
        config=cfg,
        shadow=shadow,
        live_gate=live_gate,
        intents_total=len(intents),
        intents_routed=len(routed_allowed),
        executed_success=successes,
        executed_failed=failures,
        records=records,
        notes=notes,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CHAD IBKR execution runner. Final live/paper behavior is determined by LiveGate."
    )
    p.add_argument("--log-level", default=os.environ.get("CHAD_LOG_LEVEL", "INFO"))
    p.add_argument("--max-trades-for-shadow", type=int, default=500)
    p.add_argument("--days-back-for-shadow", type=int, default=30)
    p.add_argument("--include-paper-stats", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-live-stats", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--json-output", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fail-on-partial", action="store_true", help="Return non-zero if any routed execution fails.")
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Operator request only. Final execution lane still comes from LiveGate.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = RunnerConfig.from_args(args)
    summary = run_ibkr_execution(cfg)

    if cfg.json_output:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True, default=_json_default))
    else:
        print(f"ok={summary.ok}")
        print(f"exit_code={summary.exit_code}")
        print(f"intents_total={summary.intents_total}")
        print(f"intents_routed={summary.intents_routed}")
        print(f"executed_success={summary.executed_success}")
        print(f"executed_failed={summary.executed_failed}")
        print(f"live_flag={summary.live_gate.allow_ibkr_live}")

    return int(summary.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

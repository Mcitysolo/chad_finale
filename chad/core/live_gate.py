"""
chad.core.live_gate

World-class LiveGate evaluator with:
- deny-by-default gate chain
- DecisionTrace emission on every evaluation
- runtime freshness enforcement for critical state (IBKR status)

Key guarantees:
- Never executes trades.
- Always returns a safe decision.
- Deterministic gate order + reasons.
- Writes audit-first DecisionTrace (append-only NDJSON).
- Enforces runtime TTL freshness per CSB: stale/missing runtime => fail closed.

"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from chad.core.decision_trace import (
    GateResult,
    LiveGateDecisionTrace,
    build_decision_trace_record,
    default_writer,
    new_cycle_id,
    new_trace_id,
)
from chad.utils.runtime_json import read_runtime_state_json

# ----------------------------
# Helpers
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() != "" else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_dir() -> Path:
    return _repo_root() / "runtime"


# ----------------------------
# Runtime state readers
# ----------------------------

@dataclass(frozen=True)
class StopState:
    stop: bool
    reason: str


def _load_stop_state() -> StopState:
    p = _runtime_dir() / "stop_state.json"
    obj = _read_json(p)
    stop = bool(obj.get("stop", False))
    reason = str(obj.get("reason", obj.get("stop_reason", "")))[:240]
    return StopState(stop=stop, reason=reason)


@dataclass(frozen=True)
class OperatorIntent:
    mode: str
    reason: str


def _load_operator_intent() -> OperatorIntent:
    p = _runtime_dir() / "operator_intent.json"
    obj = _read_json(p)
    mode = str(obj.get("operator_mode", obj.get("mode", "EXIT_ONLY"))).upper()
    reason = str(obj.get("operator_reason", obj.get("reason", "")))[:240]
    if mode not in ("EXIT_ONLY", "ALLOW", "DENY_ALL"):
        mode = "EXIT_ONLY"
    return OperatorIntent(mode=mode, reason=reason)


@dataclass(frozen=True)
class ExecutionConfig:
    exec_mode: str  # "dry_run" | "paper" | "live"
    ibkr_enabled: bool
    ibkr_dry_run: bool
    kraken_enabled: bool


def _load_execution_config_best_effort() -> ExecutionConfig:
    exec_mode = _env_str("CHAD_EXEC_MODE", "").lower()
    chad_mode = _env_str("CHAD_MODE", "DRY_RUN").upper()

    if exec_mode in ("dry_run", "paper", "live"):
        normalized = exec_mode
    else:
        normalized = "live" if chad_mode == "LIVE" else ("paper" if chad_mode == "PAPER" else "dry_run")

    ibkr_enabled = _env_bool("CHAD_IBKR_ENABLED", True)
    kraken_enabled = _env_bool("CHAD_KRAKEN_ENABLED", True)
    ibkr_dry_run = _env_bool("CHAD_IBKR_DRY_RUN", True)

    return ExecutionConfig(
        exec_mode=normalized,
        ibkr_enabled=ibkr_enabled,
        ibkr_dry_run=ibkr_dry_run,
        kraken_enabled=kraken_enabled,
    )


@dataclass(frozen=True)
class ShadowState:
    state: str
    sizing_factor: float
    paper_only: bool
    reasons: List[str]


def _load_shadow_state_best_effort() -> ShadowState:
    try:
        from chad.analytics.shadow_state_snapshot import load_shadow_snapshot  # type: ignore
        snap = load_shadow_snapshot()
        state = str(snap.get("state", "UNKNOWN"))
        sizing = float(snap.get("sizing_factor", 0.1))
        paper_only = bool(snap.get("paper_only", True))
        rs = snap.get("reasons", [])
        reasons = [str(x) for x in rs] if isinstance(rs, list) else []
        return ShadowState(state=state, sizing_factor=sizing, paper_only=paper_only, reasons=reasons)
    except Exception:
        obj = _read_json(_runtime_dir() / "shadow_state.json")
        state = str(obj.get("state", "WARMUP"))
        sizing = float(obj.get("sizing_factor", 0.1)) if isinstance(obj.get("sizing_factor", 0.1), (int, float)) else 0.1
        paper_only = bool(obj.get("paper_only", True))
        rs = obj.get("reasons", [])
        reasons = [str(x) for x in rs] if isinstance(rs, list) else []
        return ShadowState(state=state, sizing_factor=float(sizing), paper_only=paper_only, reasons=reasons[:20])


@dataclass(frozen=True)
class IBKRStatus:
    ok: bool
    fresh_ok: bool
    reason: str
    age_seconds: float
    ttl_seconds: float
    error: Optional[str]
    server_time_iso: Optional[str]


def _load_ibkr_status() -> IBKRStatus:
    path = _runtime_dir() / "ibkr_status.json"
    obj, fr = read_runtime_state_json(path)
    if obj is None:
        return IBKRStatus(
            ok=False,
            fresh_ok=False,
            reason=f"runtime_ibkr_status_{fr.reason}",
            age_seconds=fr.age_seconds,
            ttl_seconds=fr.ttl_seconds,
            error=None,
            server_time_iso=None,
        )
    ok = bool(obj.get("ok", False))
    return IBKRStatus(
        ok=ok,
        fresh_ok=bool(fr.ok),
        reason=fr.reason,
        age_seconds=float(fr.age_seconds),
        ttl_seconds=float(fr.ttl_seconds),
        error=str(obj.get("error")) if obj.get("error") is not None else None,
        server_time_iso=str(obj.get("server_time_iso")) if obj.get("server_time_iso") is not None else None,
    )


# ----------------------------
# LiveGate domain objects
# ----------------------------

@dataclass(frozen=True)
class LiveGateContext:
    execution: ExecutionConfig
    stop_state: StopState
    operator_intent: OperatorIntent
    shadow_state: ShadowState
    ibkr_status: IBKRStatus
    chad_mode: str


@dataclass(frozen=True)
class LiveGateDecision:
    context: LiveGateContext
    mode: str  # "DENY_ALL" | "EXIT_ONLY" | "ALLOW_LIVE"
    reasons: List[str]
    allow_exits_only: bool
    allow_ibkr_live: bool
    allow_ibkr_paper: bool


def _build_default_context() -> LiveGateContext:
    execution = _load_execution_config_best_effort()
    stop_state = _load_stop_state()
    operator_intent = _load_operator_intent()
    shadow_state = _load_shadow_state_best_effort()
    ibkr_status = _load_ibkr_status()
    chad_mode = _env_str("CHAD_MODE", "DRY_RUN").upper()
    return LiveGateContext(
        execution=execution,
        stop_state=stop_state,
        operator_intent=operator_intent,
        shadow_state=shadow_state,
        ibkr_status=ibkr_status,
        chad_mode=chad_mode,
    )


def evaluate_live_gate() -> LiveGateDecision:
    """
    Evaluate LiveGate with deterministic gate order.
    Always writes a DecisionTrace record (best-effort).
    """
    ctx = _build_default_context()

    gates: List[GateResult] = []
    reasons: List[str] = []

    # Gate 0: ExecutionConfig (adapter-level posture)
    gates.append(
        GateResult(
            name="ExecutionConfig",
            passed=True,
            reason=f"exec_mode={ctx.execution.exec_mode} ibkr_enabled={ctx.execution.ibkr_enabled} ibkr_dry_run={ctx.execution.ibkr_dry_run}",
            details={},
        )
    )

    # Gate 1: STOP (authoritative DENY_ALL)
    if ctx.stop_state.stop:
        gates.append(GateResult(name="STOP", passed=False, reason="enabled", details={"reason": ctx.stop_state.reason}))
        reasons.append(f"STOP is ENABLED (DENY_ALL). reason={ctx.stop_state.reason!r}.")
        decision = LiveGateDecision(ctx, "DENY_ALL", reasons, False, False, False)
        _emit_decision_trace(ctx, gates, decision)
        return decision
    gates.append(GateResult(name="STOP", passed=True, reason="not_engaged", details={}))

    # Gate 2: IBKR status freshness + ok (fail closed)
    if ctx.execution.ibkr_enabled:
        passed = bool(ctx.ibkr_status.fresh_ok and ctx.ibkr_status.ok)
        detail = {
            "fresh_ok": ctx.ibkr_status.fresh_ok,
            "ok": ctx.ibkr_status.ok,
            "fresh_reason": ctx.ibkr_status.reason,
            "age_seconds": ctx.ibkr_status.age_seconds,
            "ttl_seconds": ctx.ibkr_status.ttl_seconds,
            "error": ctx.ibkr_status.error,
            "server_time_iso": ctx.ibkr_status.server_time_iso,
        }
        if not passed:
            gates.append(GateResult(name="IBKR_STATUS", passed=False, reason="stale_or_not_ok", details=detail))
            reasons.append("IBKR status is missing/stale or ok=false. Fail-closed for any IBKR execution path.")
            decision = LiveGateDecision(ctx, "DENY_ALL", reasons, False, False, False)
            _emit_decision_trace(ctx, gates, decision)
            return decision
        gates.append(GateResult(name="IBKR_STATUS", passed=True, reason="fresh_ok_true_ok_true", details=detail))
    else:
        gates.append(GateResult(name="IBKR_STATUS", passed=True, reason="ibkr_disabled", details={}))

    # Gate 3: Global CHAD_MODE (must be LIVE for live trading)
    if ctx.chad_mode != "LIVE":
        gates.append(GateResult(name="CHAD_MODE", passed=False, reason=f"mode={ctx.chad_mode}", details={}))
        reasons.append("CHAD_MODE is not LIVE (or live_enabled=False). Global mode does not permit live trading.")
    else:
        gates.append(GateResult(name="CHAD_MODE", passed=True, reason="LIVE", details={}))

    # Gate 4: Operator intent
    if ctx.operator_intent.mode == "DENY_ALL":
        gates.append(GateResult(name="OperatorIntent", passed=False, reason="DENY_ALL", details={"reason": ctx.operator_intent.reason}))
        reasons.append(f"OperatorIntent=DENY_ALL. reason={ctx.operator_intent.reason!r}.")
        decision = LiveGateDecision(ctx, "DENY_ALL", reasons, False, False, False)
        _emit_decision_trace(ctx, gates, decision)
        return decision

    operator_exit_only = (ctx.operator_intent.mode == "EXIT_ONLY")
    gates.append(GateResult(name="OperatorIntent", passed=True, reason=ctx.operator_intent.mode, details={"reason": ctx.operator_intent.reason}))
    if operator_exit_only:
        reasons.append(f"OperatorIntent=EXIT_ONLY. reason={ctx.operator_intent.reason!r}. No new entries; exits-only permitted.")

    # Gate 5: SCR paper_only
    if ctx.shadow_state.paper_only:
        gates.append(
            GateResult(
                name="SCR",
                passed=False,
                reason=f"paper_only=True state={ctx.shadow_state.state}",
                details={"sizing_factor": ctx.shadow_state.sizing_factor, "reasons": ctx.shadow_state.reasons[:10]},
            )
        )
        reasons.append("ShadowState.paper_only is True. SCR currently requires paper-only operation; live trading is blocked.")
        decision = LiveGateDecision(ctx, "EXIT_ONLY", reasons, True, False, False)
        _emit_decision_trace(ctx, gates, decision)
        return decision

    gates.append(
        GateResult(
            name="SCR",
            passed=True,
            reason=f"paper_only=False state={ctx.shadow_state.state}",
            details={"sizing_factor": ctx.shadow_state.sizing_factor},
        )
    )

    # Final allow decisions
    allow_ibkr_live = bool(ctx.execution.ibkr_enabled and (not ctx.execution.ibkr_dry_run) and ctx.chad_mode == "LIVE" and (not operator_exit_only))
    allow_ibkr_paper = bool(ctx.execution.ibkr_enabled and (not allow_ibkr_live) and (not operator_exit_only) and _env_bool("CHAD_ALLOW_IBKR_PAPER", False))

    if allow_ibkr_live:
        mode = "ALLOW_LIVE"
        reasons.append("All gates satisfied for live trading (subject to per-order checks).")
    elif operator_exit_only:
        mode = "EXIT_ONLY"
        reasons.append("Operator intent is EXIT_ONLY; entries blocked.")
    else:
        mode = "DENY_ALL"
        if ctx.execution.ibkr_dry_run:
            reasons.append("ExecutionConfig is hard-locked to DRY_RUN for IBKR (ibkr_dry_run=True). LIVE execution is disabled at adapter level.")

    decision = LiveGateDecision(ctx, mode, reasons, (mode == "EXIT_ONLY"), allow_ibkr_live, allow_ibkr_paper)
    _emit_decision_trace(ctx, gates, decision)
    return decision


def _emit_decision_trace(ctx: LiveGateContext, gates: List[GateResult], decision: LiveGateDecision) -> None:
    """
    Emit one DecisionTrace record. Never throws.
    """
    try:
        tid = new_trace_id()
        cid = new_cycle_id()

        inputs: Dict[str, Any] = {
            "operator_intent_ref": "runtime/operator_intent.json",
            "stop_state_ref": "runtime/stop_state.json",
            "shadow_state_ref": "runtime/shadow_state.json",
            "ibkr_status_ref": "runtime/ibkr_status.json",
            "execution": {
                "exec_mode": ctx.execution.exec_mode,
                "ibkr_enabled": ctx.execution.ibkr_enabled,
                "ibkr_dry_run": ctx.execution.ibkr_dry_run,
                "kraken_enabled": ctx.execution.kraken_enabled,
                "chad_mode": ctx.chad_mode,
            },
        }

        lg = LiveGateDecisionTrace(
            mode=decision.mode if decision.mode in ("DENY_ALL", "EXIT_ONLY", "ALLOW_LIVE") else "DENY_ALL",
            reasons=list(decision.reasons)[:50],
            allow_exits_only=bool(decision.allow_exits_only),
            allow_ibkr_live=bool(decision.allow_ibkr_live),
            allow_ibkr_paper=bool(decision.allow_ibkr_paper),
        )

        rec = build_decision_trace_record(
            trace_id=tid,
            cycle_id=cid,
            inputs=inputs,
            gates=list(gates),
            livegate=lg,
            signal_intents=[],
            execution_plans=[],
            artifacts={"decision_trace_dir": "data/traces/"},
        )

        default_writer().append(rec)
    except Exception:
        return

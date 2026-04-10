#!/usr/bin/env python3
"""
CHAD API Gateway — SSOT v4.2 Operator Surface (Production)

Read-only operator visibility endpoints:
- /             (root: must mention /health and /risk-state per tests)
- /health
- /status
- /live-gate
- /risk-state
- /shadow
- /orders       (disabled in Phase 7: must return 403)

SSOT alignment
--------------
- Observational only: no broker orders, no config mutations. :contentReference[oaicite:1]{index=1}
- Fail-closed for LIVE; safe paper "what-if" lane remains available when IBKR adapter is enabled. 
- Stable response shapes (tests + operator UX).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chad.analytics.shadow_confidence_router import ShadowState, evaluate_confidence
from chad.analytics.trade_stats_engine import load_and_compute
from backend.operator_intent_store import OperatorIntentStore, OperatorMode as StoreOperatorMode
from backend.approval_surface import router as approvals_router
from backend.ai_surface import router as ai_router
from backend.portfolio_surface import router as portfolio_router
from chad.core.live_gate import evaluate_live_gate as evaluate_live_gate_core

LOGGER = logging.getLogger("chad.api_gateway")


# =============================================================================
# Paths + runtime helpers
# =============================================================================


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_root() -> Path:
    env = os.environ.get("CHAD_REPO_ROOT", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parents[1]


REPO_DIR = _repo_root()
RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", str(REPO_DIR / "runtime"))).expanduser().resolve()

FEED_STATE_PATH = RUNTIME_DIR / "feed_state.json"
POSITIONS_PATH = RUNTIME_DIR / "positions_snapshot.json"
RECONCILIATION_STATE_PATH = RUNTIME_DIR / "reconciliation_state.json"
DYNAMIC_CAPS_PATH = RUNTIME_DIR / "dynamic_caps.json"
OPERATOR_INTENT_PATH = RUNTIME_DIR / "operator_intent.json"
PORTFOLIO_SNAPSHOT_PATH = RUNTIME_DIR / "portfolio_snapshot.json"
SCR_STATE_PATH = RUNTIME_DIR / "scr_state.json"
TIER_STATE_PATH = RUNTIME_DIR / "tier_state.json"
STOP_STATE_PATH = RUNTIME_DIR / "stop_state.json"


def _read_runtime_json(path: Path) -> dict:
    try:
        if not path.is_file():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


# =============================================================================
# Snapshots + LiveGate
# =============================================================================


class ChadMode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


class OperatorMode(str, Enum):
    DENY_ALL = "DENY_ALL"
    EXIT_ONLY = "EXIT_ONLY"
    ALLOW_LIVE = "ALLOW_LIVE"


@dataclass(frozen=True)
class OperatorIntent:
    mode: OperatorMode
    reason: str
    updated_at_utc: str


def _get_chad_mode() -> ChadMode:
    raw = str(os.environ.get("CHAD_EXECUTION_MODE", "dry_run")).strip().lower()
    if raw == "live":
        return ChadMode.LIVE
    if raw == "paper":
        return ChadMode.PAPER
    return ChadMode.DRY_RUN


def _is_live_mode_enabled() -> bool:
    return _get_chad_mode() == ChadMode.LIVE


def _load_operator_intent() -> OperatorIntent:
    """
    Load OperatorIntent with strict TTL freshness enforcement (FAIL-CLOSED).

    Preserves production behavior:
      operator_intent_stale_or_missing:expired age_s=... ttl_s=...
    """
    store = OperatorIntentStore(path=OPERATOR_INTENT_PATH)
    st = store.load_fail_closed()

    mode_raw = str(st.mode or "").strip().upper()
    if mode_raw == StoreOperatorMode.DENY_ALL:
        mode = OperatorMode.DENY_ALL
    elif mode_raw == StoreOperatorMode.EXIT_ONLY:
        mode = OperatorMode.EXIT_ONLY
    elif mode_raw == StoreOperatorMode.ALLOW_LIVE:
        mode = OperatorMode.ALLOW_LIVE
    else:
        mode = OperatorMode.DENY_ALL  # fail-closed

    updated = str(st.ts_utc or _utc_now_iso())
    reason = str(st.reason or "operator_intent_missing").strip()
    return OperatorIntent(mode=mode, reason=reason, updated_at_utc=updated)

    # Phase 7 safe default: do not block paper lane; live remains blocked elsewhere unless truly enabled.
    return OperatorIntent(OperatorMode.ALLOW_LIVE, reason, updated)


def _load_stop_state() -> tuple[bool, str]:
    obj = _read_runtime_json(STOP_STATE_PATH)
    stop = bool(obj.get("stop", False))
    reason = str(obj.get("reason") or "unknown").strip()
    return stop, reason


class ExecutionConfigSnapshot(BaseModel):
    exec_mode: str
    ibkr_enabled: bool
    ibkr_dry_run: bool
    kraken_enabled: bool
    raw_mode_enum: str


class ModeSnapshot(BaseModel):
    chad_mode: str
    live_enabled: bool


class ShadowStats(BaseModel):
    total_trades: int
    live_trades: int
    paper_trades: int
    effective_trades: int
    excluded_manual: int
    excluded_untrusted: int
    excluded_nonfinite: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_like: float


class ShadowSnapshot(BaseModel):
    state: str
    sizing_factor: float
    paper_only: bool
    reasons: List[str]
    stats: ShadowStats


class LiveGateSnapshot(BaseModel):
    execution: ExecutionConfigSnapshot
    mode: ModeSnapshot
    shadow: ShadowSnapshot

    operator_mode: OperatorMode
    operator_reason: str

    allow_exits_only: bool
    allow_ibkr_live: bool
    allow_ibkr_paper: bool

    reasons: List[str]


class HealthResponse(BaseModel):
    healthy: bool
    message: str
    service: str
    details: Dict[str, Any] = Field(default_factory=dict)


class RiskStateResponse(BaseModel):
    mode: ModeSnapshot
    dynamic_caps: Optional[dict] = None
    shadow: ShadowSnapshot


class ShadowOnlyResponse(BaseModel):
    shadow: ShadowSnapshot


def _build_execution_snapshot() -> ExecutionConfigSnapshot:
    mode = _get_chad_mode()

    # Phase 7 baseline: IBKR adapter exists.
    ibkr_enabled = True

    kraken_enabled = _env_bool("CHAD_KRAKEN_ENABLED", False)

    # DRY_RUN/PAPER => never live
    ibkr_dry_run = mode != ChadMode.LIVE

    return ExecutionConfigSnapshot(
        exec_mode=str(mode.value),
        ibkr_enabled=bool(ibkr_enabled),
        ibkr_dry_run=bool(ibkr_dry_run),
        kraken_enabled=bool(kraken_enabled),
        raw_mode_enum=str(mode.value),
    )


def _build_mode_snapshot() -> ModeSnapshot:
    mode = _get_chad_mode()
    return ModeSnapshot(chad_mode=str(mode.value), live_enabled=bool(_is_live_mode_enabled()))


def _build_shadow_snapshot() -> ShadowSnapshot:
    stats_raw = load_and_compute(max_trades=5000, days_back=60, include_paper=True, include_live=True)
    shadow_state: ShadowState = evaluate_confidence(stats_raw)

    stats_model = ShadowStats(
        total_trades=int(stats_raw.get("total_trades", 0)),
        live_trades=int(stats_raw.get("live_trades", 0)),
        paper_trades=int(stats_raw.get("paper_trades", 0)),
        effective_trades=int(stats_raw.get("effective_trades", 0)),
        excluded_manual=int(stats_raw.get("excluded_manual", 0)),
        excluded_untrusted=int(stats_raw.get("excluded_untrusted", 0)),
        excluded_nonfinite=int(stats_raw.get("excluded_nonfinite", 0)),
        win_rate=float(stats_raw.get("win_rate", 0.0)),
        total_pnl=float(stats_raw.get("total_pnl", 0.0)),
        max_drawdown=float(stats_raw.get("max_drawdown", 0.0)),
        sharpe_like=float(stats_raw.get("sharpe_like", 0.0)),
    )

    return ShadowSnapshot(
        state=str(shadow_state.state),
        sizing_factor=float(shadow_state.sizing_factor),
        paper_only=bool(shadow_state.paper_only),
        reasons=list(shadow_state.reasons),
        stats=stats_model,
    )


def _operator_mode_from_core(raw: str) -> OperatorMode:
    s = str(raw or "").strip().upper()
    try:
        return OperatorMode(s)
    except Exception:
        pass

    if s == "ALLOW":
        try:
            return OperatorMode.ALLOW_LIVE
        except Exception:
            pass

    try:
        return OperatorMode.DENY_ALL
    except Exception:
        # Final fallback for enum implementations that behave differently.
        return list(OperatorMode)[0]


def _evaluate_live_gate() -> LiveGateSnapshot:
    """
    Canonical API adapter:
    - Delegates ALL gate logic to chad.core.live_gate.evaluate_live_gate()
    - Converts the canonical decision into the legacy API response model
    - Eliminates drift between CLI/core execution path and HTTP /live-gate
    """
    core = evaluate_live_gate_core().to_dict()

    exec_raw = core.get("execution") if isinstance(core.get("execution"), dict) else {}
    mode_raw = core.get("mode") if isinstance(core.get("mode"), dict) else {}
    shadow_raw = core.get("shadow") if isinstance(core.get("shadow"), dict) else {}
    stats_raw = shadow_raw.get("stats") if isinstance(shadow_raw.get("stats"), dict) else {}

    execution_snapshot = ExecutionConfigSnapshot(
        exec_mode=str(exec_raw.get("exec_mode") or "dry_run"),
        ibkr_enabled=bool(exec_raw.get("ibkr_enabled", True)),
        ibkr_dry_run=bool(exec_raw.get("ibkr_dry_run", True)),
        kraken_enabled=bool(exec_raw.get("kraken_enabled", False)),
        raw_mode_enum=str(exec_raw.get("raw_mode_enum") or exec_raw.get("exec_mode") or "dry_run"),
    )

    mode_snapshot = ModeSnapshot(
        chad_mode=str(mode_raw.get("chad_mode") or execution_snapshot.exec_mode),
        live_enabled=bool(mode_raw.get("live_enabled", False)),
    )

    shadow_snapshot = ShadowSnapshot(
        state=str(shadow_raw.get("state") or "PAUSED"),
        sizing_factor=float(shadow_raw.get("sizing_factor", 0.0)),
        paper_only=bool(shadow_raw.get("paper_only", True)),
        reasons=[str(x) for x in (shadow_raw.get("reasons") if isinstance(shadow_raw.get("reasons"), list) else [])],
        stats=ShadowStats(
            total_trades=int(stats_raw.get("total_trades", 0)),
            live_trades=int(stats_raw.get("live_trades", 0)),
            paper_trades=int(stats_raw.get("paper_trades", 0)),
            effective_trades=int(stats_raw.get("effective_trades", 0)),
            excluded_manual=int(stats_raw.get("excluded_manual", 0)),
            excluded_untrusted=int(stats_raw.get("excluded_untrusted", 0)),
            excluded_nonfinite=int(stats_raw.get("excluded_nonfinite", 0)),
            win_rate=float(stats_raw.get("win_rate", 0.0)),
            total_pnl=float(stats_raw.get("total_pnl", 0.0)),
            max_drawdown=float(stats_raw.get("max_drawdown", 0.0)),
            sharpe_like=float(stats_raw.get("sharpe_like", 0.0)),
        ),
    )

    return LiveGateSnapshot(
        execution=execution_snapshot,
        mode=mode_snapshot,
        shadow=shadow_snapshot,
        operator_mode=_operator_mode_from_core(str(core.get("operator_mode") or "DENY_ALL")),
        operator_reason=str(core.get("operator_reason") or ""),
        allow_exits_only=bool(core.get("allow_exits_only", False)),
        allow_ibkr_live=bool(core.get("allow_ibkr_live", False)),
        allow_ibkr_paper=bool(core.get("allow_ibkr_paper", False)),
        reasons=[str(x) for x in (core.get("reasons") if isinstance(core.get("reasons"), list) else [])],
    )


# =============================================================================
# FastAPI app + endpoints
# =============================================================================


app = FastAPI(
    title="CHAD API Gateway",
    version="0.2.0-phase7-10",
    description="Read-only, risk-aware API gateway for CHAD (SSOT v4.2).",
)

# Phase 6: Read-only portfolio endpoints (SSOT v4.2)
app.include_router(portfolio_router)
app.include_router(approvals_router)
app.include_router(ai_router)


@app.get("/", tags=["system"])
def root() -> dict:
    # Must include /health and /risk-state in message string (test contract).
    return {
        "service": "CHAD API Gateway",
        "message": "OK — see /health, /risk-state, /shadow, /status, /live-gate",
        "ts_utc": _utc_now_iso(),
    }


@app.api_route("/orders", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], tags=["disabled"])
def orders_disabled() -> None:
    # Must return 403 and detail must include "disabled in Phase 7" (test contract).
    raise HTTPException(status_code=403, detail="orders endpoint disabled in Phase 7")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    exec_snapshot = _build_execution_snapshot()
    mode_snapshot = _build_mode_snapshot()
    shadow_snapshot = _build_shadow_snapshot()

    return HealthResponse(
        healthy=True,
        message="CHAD API Gateway is up. Trading is DRY_RUN-only in Phase 7 unless explicitly enabled.",
        service="CHAD API Gateway",
        details={
            "execution": exec_snapshot.model_dump(),
            "mode": mode_snapshot.model_dump(),
            "shadow": shadow_snapshot.model_dump(),
        },
    )


@app.get("/live-gate", response_model=LiveGateSnapshot, tags=["risk"])
def live_gate() -> LiveGateSnapshot:
    return _evaluate_live_gate()


@app.get("/risk-state", response_model=RiskStateResponse, tags=["risk"])
def risk_state() -> RiskStateResponse:
    mode_snapshot = _build_mode_snapshot()
    shadow_snapshot = _build_shadow_snapshot()
    dynamic_caps = _read_runtime_json(DYNAMIC_CAPS_PATH) if DYNAMIC_CAPS_PATH.is_file() else None
    return RiskStateResponse(mode=mode_snapshot, dynamic_caps=dynamic_caps, shadow=shadow_snapshot)


@app.get("/shadow", response_model=ShadowOnlyResponse, tags=["risk"])
def shadow() -> ShadowOnlyResponse:
    return ShadowOnlyResponse(shadow=_build_shadow_snapshot())


@app.get("/status", tags=["system"])
def status() -> dict:
    exec_snapshot = _build_execution_snapshot()
    mode_snapshot = _build_mode_snapshot()
    live_gate_snapshot = _evaluate_live_gate()
    shadow_snapshot = _build_shadow_snapshot()

    runtime_files = {
        "feed_state": {"path": str(FEED_STATE_PATH), "exists": FEED_STATE_PATH.is_file()},
        "positions_snapshot": {"path": str(POSITIONS_PATH), "exists": POSITIONS_PATH.is_file()},
        "reconciliation_state": {"path": str(RECONCILIATION_STATE_PATH), "exists": RECONCILIATION_STATE_PATH.is_file()},
        "dynamic_caps": {"path": str(DYNAMIC_CAPS_PATH), "exists": DYNAMIC_CAPS_PATH.is_file()},
        "operator_intent": {"path": str(OPERATOR_INTENT_PATH), "exists": OPERATOR_INTENT_PATH.is_file()},
        "portfolio_snapshot": {"path": str(PORTFOLIO_SNAPSHOT_PATH), "exists": PORTFOLIO_SNAPSHOT_PATH.is_file()},
        "scr_state": {"path": str(SCR_STATE_PATH), "exists": SCR_STATE_PATH.is_file()},
        "tier_state": {"path": str(TIER_STATE_PATH), "exists": TIER_STATE_PATH.is_file()},
    }

    return {
        "service": "CHAD API Gateway",
        "ts_utc": _utc_now_iso(),
        "execution": exec_snapshot.model_dump(),
        "mode": mode_snapshot.model_dump(),
        "live_gate": live_gate_snapshot.model_dump(),
        "shadow": shadow_snapshot.model_dump(),
        "runtime_files": runtime_files,
    }


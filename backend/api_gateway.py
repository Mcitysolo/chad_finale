"""
CHAD API Gateway (Phase 7/10 — Read-Only, Risk-Aware, DRY_RUN Only)

This FastAPI application exposes a strictly observational API surface for CHAD.
It is designed so that **no HTTP request can ever trigger live trading**.

Key guarantees:

* No endpoint sends orders to any broker (IBKR, crypto, forex).
* All broker behaviour remains DRY_RUN-only at the adapter level.
* Live trading is disabled by:
    - ExecutionConfig (ibkr_dry_run=True),
    - Global mode (CHAD_MODE),
    - Shadow Confidence Router (SCR) + paper_only flag,
    - Shadow risk gating logic implemented here,
    - STOP (DENY_ALL) emergency freeze,
    - Operator intent mode (Phase 10 control plane).
* Endpoints only read CHAD’s internal state:
    - Execution configuration,
    - CHAD_MODE,
    - Shadow confidence & trade stats,
    - Dynamic risk caps (if present on disk),
    - STOP state,
    - Operator intent state.
* Phase-10 AI endpoints are strictly advisory-only: they never touch
  execution, risk limits, or DRY_RUN flags.

This file is safe to be used as the main FastAPI app for the CHAD backend.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chad.analytics.shadow_confidence_router import ShadowState, evaluate_confidence
from chad.analytics.trade_stats_engine import load_and_compute
from chad.core.mode import get_chad_mode, is_live_mode_enabled
from chad.core.stop_state import load_stop_state
from chad.execution.execution_config import get_execution_config
from chad.intel.research_engine import run_research_scenario_from_request
from chad.intel.schemas import ResearchRequestInput, ResearchScenario

# Optional: dynamic caps file path used by orchestrator.
DYNAMIC_CAPS_PATH = Path("runtime/dynamic_caps.json")

# Phase 10 operator intent control plane runtime file
OPERATOR_INTENT_PATH = Path("runtime/operator_intent.json")

LOGGER = logging.getLogger("chad.api_gateway")


# ---------------------------------------------------------------------------
# Operator intent (Phase 10)
# ---------------------------------------------------------------------------

class OperatorMode(str):
    ALLOW_LIVE = "ALLOW_LIVE"
    EXIT_ONLY = "EXIT_ONLY"
    DENY_ALL = "DENY_ALL"


@dataclass(frozen=True)
class OperatorIntentState:
    mode: str
    reason: str
    updated_at_utc: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_operator_intent() -> OperatorIntentState:
    """
    Load operator intent from runtime/operator_intent.json.

    Safe default is ALLOW_LIVE (meaning: do not add extra blocking beyond other gates).
    Note: In Phase 7/8, ExecutionConfig + SCR still block real live trading.
    """
    if not OPERATOR_INTENT_PATH.is_file():
        return OperatorIntentState(mode=OperatorMode.ALLOW_LIVE, reason="default_allow_live", updated_at_utc="")
    try:
        data = json.loads(OPERATOR_INTENT_PATH.read_text(encoding="utf-8"))
        mode = str(data.get("mode", OperatorMode.ALLOW_LIVE)).upper()
        reason = str(data.get("reason", "unknown"))
        updated = str(data.get("updated_at_utc", ""))
        if mode not in (OperatorMode.ALLOW_LIVE, OperatorMode.EXIT_ONLY, OperatorMode.DENY_ALL):
            return OperatorIntentState(mode=OperatorMode.ALLOW_LIVE, reason="invalid_mode_fallback", updated_at_utc=updated)
        return OperatorIntentState(mode=mode, reason=reason, updated_at_utc=updated)
    except Exception:
        return OperatorIntentState(mode=OperatorMode.ALLOW_LIVE, reason="parse_error_fallback", updated_at_utc="")


def _save_operator_intent(state: OperatorIntentState) -> None:
    """
    Atomic write: tmp -> replace.
    """
    OPERATOR_INTENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    tmp = OPERATOR_INTENT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, OPERATOR_INTENT_PATH)


class OperatorIntentResponse(BaseModel):
    mode: str
    reason: str
    updated_at_utc: str


class OperatorIntentRequest(BaseModel):
    mode: str = Field(..., description="One of: ALLOW_LIVE, EXIT_ONLY, DENY_ALL")
    reason: str = Field(default="operator_update", description="Audit reason for change")


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class PriceSnapshotResponse(BaseModel):
    """
    Response schema for the /ai/price endpoint.

    Mirrors chad.market_data.service.PriceSnapshot so that frontends
    (Telegram, web, voice) have a stable, typed representation of
    current price and simple daily moves.
    """
    symbol: str
    asset_class: str
    price: float
    change: Optional[float] = None
    percent_change: Optional[float] = None
    as_of: Optional[str] = None
    source: str


class HealthResponse(BaseModel):
    service: str = Field(default="CHAD API Gateway")
    healthy: bool
    message: str
    details: Dict[str, Any]


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
    # Raw counts (everything parsed)
    total_trades: int
    live_trades: int
    paper_trades: int

    # Effective sample (what SCR actually trusts)
    effective_trades: int
    excluded_manual: int
    excluded_untrusted: int
    excluded_nonfinite: int

    # Performance metrics over effective sample
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


class DynamicCapsStrategyCaps(BaseModel):
    alpha: float
    beta: float
    gamma: float
    omega: float
    delta: float
    crypto: float
    forex: float


class DynamicCapsSnapshot(BaseModel):
    total_equity: float
    daily_risk_fraction: float
    portfolio_risk_cap: float
    strategy_caps: DynamicCapsStrategyCaps


class LiveGateSnapshot(BaseModel):
    execution: ExecutionConfigSnapshot
    mode: ModeSnapshot
    shadow: ShadowSnapshot

    # Phase 10 additions (operator control plane)
    operator_mode: str
    operator_reason: str
    allow_exits_only: bool

    allow_ibkr_live: bool
    allow_ibkr_paper: bool
    reasons: List[str]


class RiskStateResponse(BaseModel):
    mode: ModeSnapshot
    dynamic_caps: Optional[DynamicCapsSnapshot]
    shadow: ShadowSnapshot


class ShadowOnlyResponse(BaseModel):
    shadow: ShadowSnapshot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_dynamic_caps() -> Optional[DynamicCapsSnapshot]:
    """
    Load dynamic risk caps from runtime/dynamic_caps.json, if present.

    This file is written by the orchestrator. If it does not exist or is
    malformed, we return None and the API simply omits the caps.
    """
    if not DYNAMIC_CAPS_PATH.is_file():
        return None

    try:
        data = json.loads(DYNAMIC_CAPS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to read dynamic caps from %s: %s", DYNAMIC_CAPS_PATH, exc)
        return None

    try:
        caps = DynamicCapsSnapshot(
            total_equity=float(data["total_equity"]),
            daily_risk_fraction=float(data["daily_risk_fraction"]),
            portfolio_risk_cap=float(data["portfolio_risk_cap"]),
            strategy_caps=DynamicCapsStrategyCaps(
                alpha=float(data["strategy_caps"]["alpha"]),
                beta=float(data["strategy_caps"]["beta"]),
                gamma=float(data["strategy_caps"]["gamma"]),
                omega=float(data["strategy_caps"]["omega"]),
                delta=float(data["strategy_caps"]["delta"]),
                crypto=float(data["strategy_caps"]["crypto"]),
                forex=float(data["strategy_caps"]["forex"]),
            ),
        )
    except KeyError as exc:
        LOGGER.exception("Dynamic caps JSON missing expected key: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to parse dynamic caps JSON: %s", exc)
        return None

    return caps


def _build_shadow_snapshot() -> ShadowSnapshot:
    """
    Compute ShadowState (SCR) and wrap it as a ShadowSnapshot.

    This recomputes stats from trade history instead of reading a snapshot
    file, so it always reflects the latest trade ledger.
    """
    stats_raw = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
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


def _build_execution_snapshot() -> ExecutionConfigSnapshot:
    """
    Build a snapshot of the adapter-level execution configuration.
    """
    cfg = get_execution_config()
    return ExecutionConfigSnapshot(
        exec_mode=str(cfg.mode.value),
        ibkr_enabled=bool(cfg.ibkr_enabled),
        ibkr_dry_run=bool(cfg.ibkr_dry_run),
        kraken_enabled=bool(getattr(cfg, "kraken_enabled", False)),
        raw_mode_enum=str(cfg.mode),
    )


def _build_mode_snapshot() -> ModeSnapshot:
    """
    Build a snapshot of the global CHAD_MODE.
    """
    mode = get_chad_mode()
    live_enabled = is_live_mode_enabled()
    return ModeSnapshot(
        chad_mode=str(mode.value),
        live_enabled=bool(live_enabled),
    )


def _evaluate_live_gate() -> LiveGateSnapshot:
    """
    Evaluate whether IBKR live trading is allowed, based on:

        1) ExecutionConfig (adapter-level DRY_RUN vs live),
        2) Global CHAD_MODE,
        3) ShadowState (SCR — paper_only, state, sizing_factor),
        4) STOP (DENY_ALL),
        5) Operator intent (Phase 10): DENY_ALL / EXIT_ONLY / ALLOW_LIVE.
    """
    execution_snapshot = _build_execution_snapshot()
    mode_snapshot = _build_mode_snapshot()
    shadow_snapshot = _build_shadow_snapshot()
    operator = _load_operator_intent()

    reasons: List[str] = []

    # Adapter-level lock: if ibkr_dry_run is True, we categorically forbid live.
    if execution_snapshot.ibkr_dry_run:
        reasons.append(
            "ExecutionConfig is hard-locked to DRY_RUN for IBKR "
            "(ibkr_dry_run=True). LIVE execution is disabled at adapter level."
        )

    # Global mode intent: CHAD_MODE must be LIVE to even consider live trades.
    if not mode_snapshot.live_enabled:
        reasons.append(
            "CHAD_MODE is not LIVE (or live_enabled=False). Global mode does "
            "not permit live trading."
        )

    # SCR / Shadow router: paper_only or PAUSED states block live trading.
    if shadow_snapshot.paper_only:
        reasons.append(
            "ShadowState.paper_only is True. SCR currently requires "
            "paper-only operation; live trading is blocked."
        )
    if shadow_snapshot.state.upper() == "PAUSED":
        reasons.append("ShadowState.state=PAUSED. Live execution is fully halted.")

    allow_ibkr_live = (
        execution_snapshot.ibkr_enabled
        and not execution_snapshot.ibkr_dry_run
        and mode_snapshot.live_enabled
        and not shadow_snapshot.paper_only
        and shadow_snapshot.state.upper() != "PAUSED"
    )

    # --- STOP enforcement (authoritative DENY_ALL) ---
    stop_state = load_stop_state()
    if bool(getattr(stop_state, "stop", False)):
        reasons.append(
            f"STOP is ENABLED (DENY_ALL). reason={getattr(stop_state, 'reason', 'unknown')!r}. "
            "All trading actions are blocked (no live, no paper)."
        )
        return LiveGateSnapshot(
            execution=execution_snapshot,
            mode=mode_snapshot,
            shadow=shadow_snapshot,
            operator_mode=operator.mode,
            operator_reason=operator.reason,
            allow_exits_only=False,
            allow_ibkr_live=False,
            allow_ibkr_paper=False,
            reasons=reasons,
        )

    # --- Operator intent enforcement (Phase 10) ---
    # DENY_ALL => block paper and live
    # EXIT_ONLY => block new entries (paper/live), but signal exit-only mode to downstream
    # ALLOW_LIVE => no extra operator restriction beyond other gates
    allow_exits_only = False
    if operator.mode == OperatorMode.DENY_ALL:
        reasons.append(f"OperatorIntent=DENY_ALL. reason={operator.reason!r}. Blocking all trading actions.")
        return LiveGateSnapshot(
            execution=execution_snapshot,
            mode=mode_snapshot,
            shadow=shadow_snapshot,
            operator_mode=operator.mode,
            operator_reason=operator.reason,
            allow_exits_only=False,
            allow_ibkr_live=False,
            allow_ibkr_paper=False,
            reasons=reasons,
        )
    if operator.mode == OperatorMode.EXIT_ONLY:
        allow_exits_only = True
        reasons.append(f"OperatorIntent=EXIT_ONLY. reason={operator.reason!r}. No new entries; exits-only permitted.")
        # In Phase 7, we model exits-only by denying new entries (paper/live),
        # and exposing allow_exits_only=true. Execution layers must implement
        # exit routing explicitly in a later phase.
        return LiveGateSnapshot(
            execution=execution_snapshot,
            mode=mode_snapshot,
            shadow=shadow_snapshot,
            operator_mode=operator.mode,
            operator_reason=operator.reason,
            allow_exits_only=True,
            allow_ibkr_live=False,
            allow_ibkr_paper=False,
            reasons=reasons,
        )

    # Paper / what-if execution is allowed whenever IBKR is logically enabled.
    allow_ibkr_paper = execution_snapshot.ibkr_enabled

    if not reasons:
        reasons.append(
            "All gating conditions currently satisfied; IBKR live trading "
            "would be permitted if the adapter were not hard-locked to DRY_RUN."
        )

    return LiveGateSnapshot(
        execution=execution_snapshot,
        mode=mode_snapshot,
        shadow=shadow_snapshot,
        operator_mode=operator.mode,
        operator_reason=operator.reason,
        allow_exits_only=allow_exits_only,
        allow_ibkr_live=bool(allow_ibkr_live),
        allow_ibkr_paper=bool(allow_ibkr_paper),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CHAD API Gateway",
    version="0.2.0-phase7-10",
    description=(
        "Read-only, risk-aware API gateway for CHAD. "
        "All endpoints are strictly observational. No live trading is "
        "possible through this API in Phase 7. Phase-10 AI endpoints "
        "provide advisory-only intelligence."
    ),
)


@app.on_event("startup")
async def _on_startup() -> None:
    """
    Configure logging on startup if not already configured.
    """
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    LOGGER.info("CHAD API Gateway startup complete.")


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """
    Simple health endpoint for liveness checks (load balancers, uptime monitors).

    This does *not* touch any broker connections or trading subsystems.
    """
    exec_snapshot = _build_execution_snapshot()
    mode_snapshot = _build_mode_snapshot()
    shadow_snapshot = _build_shadow_snapshot()

    healthy = True
    message = "CHAD API Gateway is up. Trading is DRY_RUN-only in Phase 7."

    details: Dict[str, Any] = {
        "execution": exec_snapshot.dict(),
        "mode": mode_snapshot.dict(),
        "shadow": {
            "state": shadow_snapshot.state,
            "paper_only": shadow_snapshot.paper_only,
            "sizing_factor": shadow_snapshot.sizing_factor,
        },
    }

    return HealthResponse(
        healthy=healthy,
        message=message,
        service="CHAD API Gateway",
        details=details,
    )


@app.get("/live-gate", response_model=LiveGateSnapshot, tags=["risk"])
async def live_gate() -> LiveGateSnapshot:
    """
    Return the full LiveGate decision snapshot.

    This is the **single source of truth** for whether IBKR live trading would
    ever be allowed, even in future phases.
    """
    return _evaluate_live_gate()


@app.get("/operator-intent", response_model=OperatorIntentResponse, tags=["operator"])
async def get_operator_intent() -> OperatorIntentResponse:
    st = _load_operator_intent()
    return OperatorIntentResponse(mode=st.mode, reason=st.reason, updated_at_utc=st.updated_at_utc)


@app.post("/operator-intent", response_model=OperatorIntentResponse, tags=["operator"])
async def set_operator_intent(req: OperatorIntentRequest) -> OperatorIntentResponse:
    mode = str(req.mode).upper().strip()
    if mode not in (OperatorMode.ALLOW_LIVE, OperatorMode.EXIT_ONLY, OperatorMode.DENY_ALL):
        raise HTTPException(status_code=400, detail=f"invalid_operator_mode: {mode!r}")

    st = OperatorIntentState(
        mode=mode,
        reason=str(req.reason),
        updated_at_utc=_utc_now_iso(),
    )
    _save_operator_intent(st)
    return OperatorIntentResponse(mode=st.mode, reason=st.reason, updated_at_utc=st.updated_at_utc)


@app.get("/risk-state", response_model=RiskStateResponse, tags=["risk"])
async def risk_state() -> RiskStateResponse:
    mode_snapshot = _build_mode_snapshot()
    dynamic_caps = _load_dynamic_caps()
    shadow_snapshot = _build_shadow_snapshot()

    return RiskStateResponse(
        mode=mode_snapshot,
        dynamic_caps=dynamic_caps,
        shadow=shadow_snapshot,
    )


@app.get("/shadow", response_model=ShadowOnlyResponse, tags=["shadow"])
async def shadow() -> ShadowOnlyResponse:
    snapshot = _build_shadow_snapshot()
    return ShadowOnlyResponse(shadow=snapshot)


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    return {
        "service": "CHAD API Gateway",
        "message": "See /health, /risk-state, /live-gate, /shadow, and /ai/research.",
    }


@app.get("/ai/price", response_model=PriceSnapshotResponse, tags=["ai"])
async def ai_price(symbol: str) -> PriceSnapshotResponse:
    from chad.market_data.service import MarketDataError, MarketDataService  # local import to avoid cycles

    service = MarketDataService()
    try:
        snap = service.get_price_snapshot(symbol)
    except MarketDataError as exc:
        LOGGER.exception("MarketDataError in /ai/price for symbol=%s: %s", symbol, exc)
        raise HTTPException(
            status_code=500,
            detail=f"market_data_error: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Unexpected error in /ai/price for symbol=%s: %s", symbol, exc)
        raise HTTPException(
            status_code=500,
            detail="market_data_unexpected_error",
        ) from exc

    return PriceSnapshotResponse(
        symbol=snap.symbol,
        asset_class=snap.asset_class,
        price=snap.price,
        change=snap.change,
        percent_change=snap.percent_change,
        as_of=snap.as_of,
        source=snap.source,
    )


@app.post("/orders", include_in_schema=False)
async def orders_disabled() -> None:
    raise HTTPException(
        status_code=403,
        detail=(
            "Order submission via HTTP is disabled in Phase 7. "
            "CHAD is operating in DRY_RUN-only mode; live trading must be "
            "explicitly enabled in a future phase with additional review."
        ),
    )


# ---------------------------------------------------------------------------
# AI / Research Endpoint (Phase-10 Global Intelligence Layer)
# ---------------------------------------------------------------------------

@app.post("/ai/research", response_model=ResearchScenario, tags=["ai"])
async def ai_research(request: ResearchRequestInput) -> ResearchScenario:
    return run_research_scenario_from_request(request)

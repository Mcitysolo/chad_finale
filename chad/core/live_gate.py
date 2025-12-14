"""
Live Gate Controller (Phase 8 – Unified Live/DRY_RUN Decision Layer)

This module centralizes the logic for deciding whether CHAD is allowed
to *even attempt* LIVE IBKR execution.

It combines:

    * ExecutionConfig (adapter-level):
        - CHAD_EXECUTION_MODE
        - ibkr_enabled
        - ibkr_dry_run

    * CHAD_MODE (global):
        - DRY_RUN / LIVE

    * Operator Live Intent (live_mode.json):
        - live (True/False)
        - reason (why the operator set this)

    * Shadow Confidence State (SCR):
        - state (WARMUP / CONFIDENT / CAUTIOUS / PAUSED)
        - paper_only
        - sizing_factor
        - reasons

IMPORTANT (Phase 7/early Phase 8):
    - ExecutionConfig is currently hard-locked to DRY_RUN for IBKR, so
      allow_ibkr_live will always be False until ExecutionConfig is
      relaxed in a future step.
    - This controller is **read-only**: it does NOT execute trades or
      talk to brokers. It only answers: "Would live be allowed if the
      adapter were not hard-locked?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from chad.analytics.trade_stats_engine import load_and_compute
from chad.analytics.shadow_confidence_router import evaluate_confidence, ShadowState
from chad.core.mode import get_chad_mode, CHADMode
from chad.core.live_mode import load_live_mode, LiveModeState
from chad.execution.execution_config import get_execution_config


@dataclass(frozen=True)
class LiveGateContext:
    """
    Immutable snapshot of the inputs used to make a live/DRY_RUN decision.
    """

    chad_mode: CHADMode
    exec_mode: str
    ibkr_enabled: bool
    ibkr_dry_run: bool
    live_intent: bool
    live_reason: str
    shadow_state: ShadowState


@dataclass(frozen=True)
class LiveGateDecision:
    """
    Unified live/DRY_RUN decision for IBKR.

    Fields:
        allow_ibkr_live:
            True iff ALL of the following are satisfied:
                - ibkr_enabled
                - ibkr_dry_run is False
                - chad_mode == LIVE
                - operator live intent (live_mode.live) is True
                - shadow_state.paper_only is False
                - shadow_state.state in {"CONFIDENT", "CAUTIOUS"}

        allow_ibkr_paper:
            True iff ibkr_enabled (paper / what-if is always allowed).

        reasons:
            List of human-readable messages explaining why live is
            allowed or blocked.

        context:
            The LiveGateContext used for this decision.
    """

    allow_ibkr_live: bool
    allow_ibkr_paper: bool
    reasons: List[str] = field(default_factory=list)
    context: LiveGateContext = field(
        default_factory=lambda: _build_default_context()
    )


def _build_default_context() -> LiveGateContext:
    """
    Build a conservative default context for situations where a
    LiveGateDecision is constructed without an explicit context.

    This should not be used in normal operation; evaluate_live_gate()
    always populates a full context.
    """
    stats = load_and_compute(
        max_trades=50,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    shadow_state = evaluate_confidence(stats)
    live_state = load_live_mode()
    return LiveGateContext(
        chad_mode=CHADMode.DRY_RUN,
        exec_mode="unknown",
        ibkr_enabled=False,
        ibkr_dry_run=True,
        live_intent=live_state.live,
        live_reason=live_state.reason,
        shadow_state=shadow_state,
    )


def evaluate_live_gate() -> LiveGateDecision:
    """
    Evaluate whether IBKR LIVE execution is even theoretically allowed.

    Returns:
        LiveGateDecision with:
            - allow_ibkr_live
            - allow_ibkr_paper
            - reasons
            - context
    """
    reasons: List[str] = []

    # --- ExecutionConfig (adapter-level) ---
    exec_cfg = get_execution_config()
    exec_mode_str = str(getattr(exec_cfg, "mode", "unknown"))
    ibkr_enabled = bool(getattr(exec_cfg, "ibkr_enabled", False))
    ibkr_dry_run = bool(getattr(exec_cfg, "ibkr_dry_run", True))

    # --- CHAD_MODE (global) ---
    chad_mode = get_chad_mode()

    # --- Operator live intent (live_mode.json) ---
    live_state: LiveModeState = load_live_mode()
    live_intent = bool(live_state.live)
    live_reason = live_state.reason

    # --- SCR (Shadow Confidence) ---
    stats = load_and_compute(
        max_trades=200,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    shadow_state = evaluate_confidence(stats)

    # Build context snapshot
    ctx = LiveGateContext(
        chad_mode=chad_mode,
        exec_mode=exec_mode_str,
        ibkr_enabled=ibkr_enabled,
        ibkr_dry_run=ibkr_dry_run,
        live_intent=live_intent,
        live_reason=live_reason,
        shadow_state=shadow_state,
    )

    # --- Base paper allowance ---
    allow_ibkr_paper = ibkr_enabled

    # --- Compute allow_ibkr_live with layered conditions ---
    # Start pessimistic; only lift to True if we pass all gates.
    allow_ibkr_live = False

    # 1) IBKR must be enabled at all.
    if not ibkr_enabled:
        reasons.append("IBKR disabled by ExecutionConfig (ibkr_enabled=False).")
        return LiveGateDecision(
            allow_ibkr_live=False,
            allow_ibkr_paper=allow_ibkr_paper,
            reasons=reasons,
            context=ctx,
        )

    # 2) Adapter-level mode must not be hard-locked to DRY_RUN.
    if ibkr_dry_run:
        reasons.append(
            "ExecutionConfig is hard-locked to DRY_RUN for IBKR "
            "(ibkr_dry_run=True). LIVE execution is disabled at adapter level."
        )
        return LiveGateDecision(
            allow_ibkr_live=False,
            allow_ibkr_paper=allow_ibkr_paper,
            reasons=reasons,
            context=ctx,
        )

    # 3) Global CHAD_MODE must be LIVE.
    if chad_mode != CHADMode.LIVE:
        reasons.append(
            f"CHAD_MODE={chad_mode.value} – LIVE execution requires CHAD_MODE=LIVE."
        )
        return LiveGateDecision(
            allow_ibkr_live=False,
            allow_ibkr_paper=allow_ibkr_paper,
            reasons=reasons,
            context=ctx,
        )

    # 4) Operator live intent must be True.
    if not live_intent:
        reasons.append(
            f"Operator live intent is False (live_mode.live=False, "
            f"reason={live_reason!r}). LIVE execution requires "
            "operator approval via live_mode.json."
        )
        return LiveGateDecision(
            allow_ibkr_live=False,
            allow_ibkr_paper=allow_ibkr_paper,
            reasons=reasons,
            context=ctx,
        )

    # 5) SCR must not be in paper-only mode, and must be in CONFIDENT/CAUTIOUS band.
    if shadow_state.paper_only:
        reasons.append(
            f"SCR is paper_only=True in state={shadow_state.state}. "
            "LIVE execution blocked until SCR lifts paper-only mode."
        )
        return LiveGateDecision(
            allow_ibkr_live=False,
            allow_ibkr_paper=allow_ibkr_paper,
            reasons=reasons,
            context=ctx,
        )

    allowed_states = {"CONFIDENT", "CAUTIOUS"}
    if shadow_state.state not in allowed_states:
        reasons.append(
            f"SCR state={shadow_state.state} not in allowed LIVE band {allowed_states}."
        )
        return LiveGateDecision(
            allow_ibkr_live=False,
            allow_ibkr_paper=allow_ibkr_paper,
            reasons=reasons,
            context=ctx,
        )

    # If we reach here, all gates passed.
    allow_ibkr_live = True
    reasons.append(
        "All live gates passed: IBKR enabled, IBKR not hard-locked to DRY_RUN, "
        "CHAD_MODE=LIVE, operator live intent=True, SCR in allowed band and "
        "not paper-only."
    )

    return LiveGateDecision(
        allow_ibkr_live=allow_ibkr_live,
        allow_ibkr_paper=allow_ibkr_paper,
        reasons=reasons,
        context=ctx,
    )

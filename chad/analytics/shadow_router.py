"""
Shadow Router (Phase 5 – SCR → Execution Gating & Sizing)

This module applies the Shadow Confidence Router (SCR) state to a list of
execution intents (e.g., IBKR StrategyTradeIntent objects).

It does NOT talk to brokers directly. It only decides:

    * Is live execution allowed for this batch, given current confidence?
    * If allowed, what sizing factor should be applied to each order?
    * If not allowed, why was the order blocked?

Typical usage (inside an executor or orchestrator):

    from chad.analytics.trade_stats_engine import load_and_compute
    from chad.analytics.shadow_confidence_router import evaluate_confidence
    from chad.analytics.shadow_router import route_orders, summarize_routing

    stats = load_and_compute(max_trades=200, days_back=30)
    shadow_state = evaluate_confidence(stats)

    # intents = list of StrategyTradeIntent or similar objects
    routed = route_orders(shadow_state, intents)
    summary = summarize_routing(routed)

    # Executor then decides:
    #   - send only routed orders with allowed_live=True
    #   - use adjusted_quantity for sizing
    #   - log summary + reasons for coach / reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from chad.analytics.shadow_confidence_router import ShadowState, ConfidenceState


@dataclass(frozen=True)
class RoutedOrder:
    """
    Result of routing a single intent through SCR.

    Fields:
        original_intent:
            The original broker-agnostic intent object (e.g. StrategyTradeIntent).

        allowed_live:
            Whether this order is permitted to be sent as a live order to a broker.
            (Paper execution is always allowed, regardless of this flag.)

        adjusted_quantity:
            Quantity after applying SCR sizing_factor. The executor is responsible
            for rounding and enforcing minimum lot sizes. If <= 0, the order is
            effectively blocked for live execution.

        reason:
            Human-readable explanation of why this order is allowed or blocked.
    """

    original_intent: Any
    allowed_live: bool
    adjusted_quantity: float
    reason: str = ""


@dataclass(frozen=True)
class RoutingSummary:
    """
    Aggregated view of a routing decision for a batch of intents.

    Fields:
        total:
            Total number of intents processed.

        live_allowed:
            Number of intents allowed for live execution (allowed_live=True).

        live_blocked:
            Number of intents blocked from live execution.

        state:
            SCR confidence state that was applied ("WARMUP", "CONFIDENT",
            "CAUTIOUS", "PAUSED").

        sizing_factor:
            Sizing factor applied to all intents (from ShadowState).

        paper_only:
            Whether SCR state indicated paper-only mode.

        reasons:
            A deduplicated list of routing-level reasons (merged from all orders).
    """

    total: int
    live_allowed: int
    live_blocked: int
    state: ConfidenceState
    sizing_factor: float
    paper_only: bool
    reasons: List[str] = field(default_factory=list)


def _get_quantity(intent: Any) -> float:
    """
    Extract an order quantity from an intent object or dict.

    We intentionally keep this flexible to avoid tight coupling to any specific
    Intent implementation.

    Priority:
        1) attribute `quantity`
        2) key "quantity" if intent is a dict
        3) fallback: 0.0
    """
    try:
        if hasattr(intent, "quantity"):
            value = getattr(intent, "quantity")
            return float(value)
        if isinstance(intent, dict) and "quantity" in intent:
            return float(intent["quantity"])
    except Exception:
        return 0.0
    return 0.0


def _describe_state(shadow_state: ShadowState) -> str:
    """
    Short human description of the SCR state.
    """
    return (
        f"SCR state={shadow_state.state} "
        f"sizing_factor={shadow_state.sizing_factor:.3f} "
        f"paper_only={shadow_state.paper_only}"
    )


def route_orders(shadow_state: ShadowState, intents: List[Any]) -> List[RoutedOrder]:
    """
    Apply SCR state to a batch of intents and produce RoutedOrder objects.

    Rules:
        * If shadow_state.paper_only is True:
            - allowed_live=False for all orders
            - adjusted_quantity = 0.0
            - reason indicates paper-only mode.

        * If SCR state is "PAUSED":
            - same as paper_only=True (pauses all live trading).

        * For "WARMUP", "CONFIDENT", "CAUTIOUS":
            - adjusted_quantity = original_quantity * shadow_state.sizing_factor
            - If adjusted_quantity <= 0, allowed_live=False (blocked by size).
            - If paper_only=False and adjusted_quantity > 0, allowed_live=True.

    Executors are expected to:
        * Send live orders only when allowed_live=True and quantity > 0.
        * Always log / possibly simulate paper execution regardless of live status.
    """
    routed: List[RoutedOrder] = []

    if not intents:
        return routed

    # If paper_only or PAUSED, we block all live execution.
    hard_block = shadow_state.paper_only or (shadow_state.state == "PAUSED")
    base_reason = _describe_state(shadow_state)

    if hard_block:
        reason = (
            f"{base_reason} → live trading blocked; run paper-only. "
            "Orders may still be simulated/paper-executed."
        )
        for intent in intents:
            routed.append(
                RoutedOrder(
                    original_intent=intent,
                    allowed_live=False,
                    adjusted_quantity=0.0,
                    reason=reason,
                )
            )
        return routed

    # Otherwise we are in WARMUP / CAUTIOUS / CONFIDENT but potentially allowing live.
    sizing_factor = float(shadow_state.sizing_factor)

    for intent in intents:
        orig_qty = _get_quantity(intent)
        adj_qty = orig_qty * sizing_factor

        if adj_qty <= 0.0:
            # Blocked because sizing wiped out the quantity (or original was invalid).
            reason = (
                f"{base_reason} → adjusted_quantity={adj_qty:.6f} "
                "≤ 0, live order blocked."
            )
            routed.append(
                RoutedOrder(
                    original_intent=intent,
                    allowed_live=False,
                    adjusted_quantity=0.0,
                    reason=reason,
                )
            )
        else:
            # Live is allowed, subject to downstream executor limits.
            reason = (
                f"{base_reason} → live allowed with adjusted_quantity="
                f"{adj_qty:.6f} (orig={orig_qty:.6f})."
            )
            routed.append(
                RoutedOrder(
                    original_intent=intent,
                    allowed_live=True,
                    adjusted_quantity=adj_qty,
                    reason=reason,
                )
            )

    return routed


def summarize_routing(
    routed_orders: List[RoutedOrder],
    shadow_state: ShadowState,
) -> RoutingSummary:
    """
    Build an aggregated summary of the routing decision.

    This is intended for:
        * logging
        * coach / reports
        * quick sanity checks in executors
    """
    total = len(routed_orders)
    live_allowed = sum(1 for r in routed_orders if r.allowed_live)
    live_blocked = total - live_allowed

    # Collect unique reasons at the routing level.
    reason_set = {r.reason for r in routed_orders if r.reason}
    reasons = sorted(reason_set)

    return RoutingSummary(
        total=total,
        live_allowed=live_allowed,
        live_blocked=live_blocked,
        state=shadow_state.state,
        sizing_factor=shadow_state.sizing_factor,
        paper_only=shadow_state.paper_only,
        reasons=reasons,
    )

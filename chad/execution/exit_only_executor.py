#!/usr/bin/env python3
"""
chad/execution/exit_only_executor.py

Phase 9.1 — Exit-Only Execution (Paper) — IMPLEMENTATION PART 1 (Interface + Guardrails)

This module defines the *only* allowed interface for Phase 9.1 exit-only execution.
It DOES NOT place broker orders in Part 1. It only enforces invariants and produces
a normalized execution plan that downstream executors can implement later.

Non-negotiable guarantees:
- No new entries (ever) in Phase 9.1
- No live trading
- No lane logic (lane_id is accepted but MUST NOT affect behavior yet)
- All actions must be gated by LiveGate allow_exits_only == True
- Exactly-once semantics will be implemented in Part 3 (idempotency keys), but the
  structures are defined here.

Design intent: keep execution interfaces clean so future `lane_id` can be passed
through without refactor, while preserving a single global execution path today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, List, Optional, Sequence


class ExitOnlyError(RuntimeError):
    """Raised when exit-only invariants are violated."""


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    FOREX = "forex"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"
    UNKNOWN = "unknown"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class LiveGateDecision:
    """
    Minimal LiveGate surface required for Phase 9.1.

    IMPORTANT:
    - We intentionally do not bind to backend schema objects here.
    - Caller must map HTTP /live-gate response into this model.
    """

    allow_exits_only: bool
    allow_ibkr_paper: bool
    allow_ibkr_live: bool
    operator_mode: str
    reasons: Sequence[str]


@dataclass(frozen=True)
class Position:
    """
    A normalized open position snapshot.

    qty semantics:
      - For equities/ETFs: shares (can be fractional if broker supports)
      - For forex: base units
      - For crypto: coin units
    """

    symbol: str
    asset_class: AssetClass
    qty: float
    avg_cost: Optional[float] = None
    currency: str = "USD"
    venue: Optional[str] = None
    # Future-only: lane identifier to thread through without behavior change.
    lane_id: Optional[str] = None


@dataclass(frozen=True)
class ExitIntent:
    """
    Exit intent for a single position.

    This is an execution-layer intent (post-strategy):
    - It can only *reduce to flat*.
    - No partial exits in Phase 9.1.
    """

    symbol: str
    asset_class: AssetClass
    side: Side
    qty: float
    currency: str = "USD"
    lane_id: Optional[str] = None  # future-only passthrough
    reason: str = "exit_only"


@dataclass(frozen=True)
class ExitPlan:
    """
    Final plan produced by the exit-only executor.
    """
    ts_utc: str
    lane_id: Optional[str]
    exits: Sequence[ExitIntent]
    notes: Sequence[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_finite(x: float) -> bool:
    # Avoid importing math for a single check: NaN != NaN
    return (x == x) and (x not in (float("inf"), float("-inf")))


def _infer_exit_side(qty: float) -> Side:
    """
    If qty > 0, we must SELL to go flat.
    If qty < 0, we must BUY to cover to zero.
    """
    if qty > 0:
        return Side.SELL
    return Side.BUY


def _validate_positions(positions: Iterable[Position]) -> List[Position]:
    out: List[Position] = []
    for p in positions:
        if not p.symbol or not isinstance(p.symbol, str):
            raise ExitOnlyError("invalid_position: missing symbol")
        if not _is_finite(float(p.qty)) or float(p.qty) == 0.0:
            # Ignore flat or invalid quantities safely
            continue
        if p.asset_class in (AssetClass.OPTIONS, AssetClass.FUTURES):
            # Phase 9.1 is intentionally conservative: no derivatives exits yet.
            # This is NOT a permanent limitation; later phases expand.
            continue
        out.append(p)
    return out


def _require_exit_only_gate(gate: LiveGateDecision) -> None:
    """
    Enforce the single condition that unlocks Phase 9.1:
    allow_exits_only must be true.

    We also require that live is NOT allowed.
    """
    if not bool(gate.allow_exits_only):
        raise ExitOnlyError("livegate_denied: allow_exits_only is false")

    # Defensive: even if allow_exits_only is true, we must never allow live in Phase 9.1.
    if bool(gate.allow_ibkr_live):
        raise ExitOnlyError("livegate_invalid: allow_ibkr_live is true (Phase 9.1 forbids live)")

    # Paper allowance may be false in exit-only mode depending on enforcement.
    # Phase 9.1 implementation will route exits via dedicated exit path,
    # so allow_ibkr_paper is NOT used as an entry permission here.


def build_exit_only_plan(
    *,
    live_gate: LiveGateDecision,
    positions: Sequence[Position],
    lane_id: Optional[str] = None,
) -> ExitPlan:
    """
    Build an exit-only execution plan.

    Guarantees:
    - Never produces an entry (only flatting exits).
    - Never produces live actions.
    - Lane_id is accepted but must not affect the plan yet.
    """
    _require_exit_only_gate(live_gate)

    # Enforce "single global execution path" for now.
    # lane_id is accepted as metadata only.
    safe_positions = _validate_positions(positions)

    exits: List[ExitIntent] = []
    notes: List[str] = []

    if not safe_positions:
        notes.append("no_action: no eligible positions (non-zero, non-derivative)")
        return ExitPlan(ts_utc=_utc_now_iso(), lane_id=lane_id, exits=tuple(), notes=tuple(notes))

    for p in safe_positions:
        side = _infer_exit_side(float(p.qty))
        qty_abs = abs(float(p.qty))
        if qty_abs <= 0.0 or not _is_finite(qty_abs):
            continue
        exits.append(
            ExitIntent(
                symbol=p.symbol,
                asset_class=p.asset_class,
                side=side,
                qty=qty_abs,
                currency=p.currency,
                lane_id=lane_id,  # passthrough only
                reason="exit_only_flatten",
            )
        )

    if not exits:
        notes.append("no_action: all eligible positions filtered out")
        return ExitPlan(ts_utc=_utc_now_iso(), lane_id=lane_id, exits=tuple(), notes=tuple(notes))

    notes.append(f"exit_count={len(exits)}")
    notes.append("phase9.1: exits-only plan (no broker calls in Part 1)")

    return ExitPlan(ts_utc=_utc_now_iso(), lane_id=lane_id, exits=tuple(exits), notes=tuple(notes))

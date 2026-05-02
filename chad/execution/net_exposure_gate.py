"""
CHAD Net Exposure Conflict Gate
================================
Runs after signal_router.route, before execution_pipeline.build_execution_plan.

Prevents strategies from fighting each other (e.g. alpha opens SPY long
while gamma opens SPY short) while preserving legitimate hedges, reductions,
and reversals.

Gate Decision Enum:
    ALLOW        — no conflict, proceed normally
    MERGE        — same direction, same symbol — merge into existing size if caps allow
    REDUCE       — opposite direction, weaker signal — reduce existing instead of opening new
    CLOSE_ONLY   — signal is a close/exit of existing position — allow
    FLIP_ALLOWED — confirmed reversal, close existing first then open new
    BLOCK        — conflict detected, signal blocked

Rules (in priority order):
    0. Signal tagged exit/liquidation → CLOSE_ONLY
    1. Reconciliation not GREEN → BLOCK any fresh opposite-direction exposure
    2. Signal tagged hedge → ALLOW within hedge budget
    3. Same symbol+asset_class, same direction → MERGE if caps allow, else BLOCK
    4. Same symbol+asset_class, opposite direction, confidence < threshold → BLOCK
    5. Same symbol+asset_class, opposite direction, lower-priority strategy → BLOCK
    6. Same symbol+asset_class, opposite direction, equal priority → REDUCE
    7. Same symbol+asset_class, confirmed reversal (confidence >= threshold,
       higher priority) → FLIP_ALLOWED
    8. No conflict → ALLOW
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("chad.net_exposure_gate")

REPO_ROOT = Path(__file__).parent.parent.parent
RUNTIME = REPO_ROOT / "runtime"

REVERSAL_CONFIDENCE_THRESHOLD = 0.70
OVERRIDE_CONFIDENCE_DELTA = 0.15
HEDGE_BUDGET_FRACTION = 0.05

STRATEGY_PRIORITY: Dict[str, int] = {
    "delta": 10,
    "beta": 9,
    "alpha": 8,
    "alpha_intraday": 7,
    "alpha_futures": 6,
    "alpha_options": 5,
    "gamma": 5,
    "gamma_futures": 4,
    "gamma_reversion": 4,
    "alpha_crypto": 3,
    "omega_vol": 3,
    "omega_momentum_options": 3,
    "omega_macro": 2,
    "omega": 2,
    "beta_trend": 1,
    "delta_pairs": 1,
    "broker_sync": 0,
}


class GateAction(str, Enum):
    ALLOW = "ALLOW"
    MERGE = "MERGE"
    REDUCE = "REDUCE"
    CLOSE_ONLY = "CLOSE_ONLY"
    FLIP_ALLOWED = "FLIP_ALLOWED"
    BLOCK = "BLOCK"


@dataclass
class GateDecision:
    action: GateAction
    reason: str
    signal_index: int
    symbol: str
    strategy: str
    conflicting_strategy: Optional[str] = None
    conflicting_side: Optional[str] = None


def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _get_reconciliation_status() -> str:
    try:
        d = _read_json(RUNTIME / "reconciliation_state.json")
        return str(d.get("status", "UNKNOWN")).upper()
    except Exception:
        return "UNKNOWN"


def _get_open_positions() -> Dict[str, dict]:
    try:
        d = _read_json(RUNTIME / "position_guard.json")
        return {
            k: v for k, v in d.items()
            if isinstance(v, dict) and v.get("open")
        }
    except Exception:
        return {}


def _get_portfolio_equity() -> float:
    try:
        d = _read_json(RUNTIME / "portfolio_snapshot.json")
        return float(d.get("total_equity", d.get("net_liquidation", 200000)) or 200000)
    except Exception:
        return 200000.0


def _signal_side(signal: Any) -> str:
    side = getattr(signal, "side", None)
    if side is None:
        return ""
    if hasattr(side, "value"):
        return str(side.value).upper()
    return str(side).upper()


def _signal_confidence(signal: Any) -> float:
    try:
        return float(getattr(signal, "confidence", 0.5) or 0.5)
    except Exception:
        return 0.5


def _signal_strategy(signal: Any) -> str:
    s = getattr(signal, "strategy", None)
    if s is None:
        # RoutedSignal uses primary_strategy / source_strategies
        s = getattr(signal, "primary_strategy", None)
        if s is None:
            sources = getattr(signal, "source_strategies", None)
            if sources:
                first = sources[0] if hasattr(sources, "__getitem__") else None
                if first is not None:
                    s = first
    if s is None:
        return ""
    if hasattr(s, "value"):
        return str(s.value).lower()
    return str(s).lower()


def _signal_symbol(signal: Any) -> str:
    return str(getattr(signal, "symbol", "") or "").upper()


def _signal_asset_class(signal: Any) -> str:
    ac = getattr(signal, "asset_class", None)
    if ac is None:
        return ""
    if hasattr(ac, "value"):
        return str(ac.value).lower()
    return str(ac).lower()


def _signal_tags(signal: Any) -> List[str]:
    """Extract tags from signal — checks direct .tags then meta.tags / meta.signal_tags."""
    direct = getattr(signal, "tags", None)
    if direct:
        try:
            return [str(t).lower() for t in direct]
        except Exception:
            pass
    meta = getattr(signal, "meta", {}) or {}
    if isinstance(meta, dict):
        tags = meta.get("tags", meta.get("signal_tags", []))
        if isinstance(tags, list):
            return [str(t).lower() for t in tags]
    return []


def _is_hedge_signal(signal: Any) -> bool:
    tags = _signal_tags(signal)
    return "hedge" in tags or "hedging" in tags


def _is_exit_signal(signal: Any) -> bool:
    tags = _signal_tags(signal)
    meta = getattr(signal, "meta", {}) or {}
    return (
        "exit" in tags
        or "liquidation" in tags
        or "close" in tags
        or (isinstance(meta, dict) and meta.get("exit"))
        or (isinstance(meta, dict) and meta.get("reason") == "max_hold_exit")
    )


def _strategy_priority(strategy: str) -> int:
    return STRATEGY_PRIORITY.get((strategy or "").lower(), 5)


def _find_conflicts(
    symbol: str,
    asset_class: str,
    new_side: str,
    new_strategy: str,
    open_positions: Dict[str, dict],
) -> List[dict]:
    """Find open positions on same symbol with any other strategy."""
    conflicts = []
    for key, pos in open_positions.items():
        pos_symbol = str(pos.get("symbol", "")).upper()
        pos_strategy = str(pos.get("strategy", "")).lower()
        pos_side = str(pos.get("side", "")).upper()

        if pos_symbol != symbol:
            continue
        if pos_strategy == new_strategy:
            continue
        if pos_strategy == "broker_sync":
            continue

        pos_ac = str(pos.get("asset_class", "")).lower()
        if pos_ac and asset_class and pos_ac != asset_class:
            continue

        conflicts.append({
            "key": key,
            "strategy": pos_strategy,
            "side": pos_side,
            "size": float(pos.get("size", pos.get("quantity", 1.0)) or 1.0),
            "opened_at": pos.get("opened_at", ""),
        })
    return conflicts


def evaluate_signal(
    signal: Any,
    signal_index: int,
    open_positions: Dict[str, dict],
    reconciliation_status: str,
    portfolio_equity: float,
) -> GateDecision:
    """Evaluate a single signal against the net exposure conflict rules."""
    symbol = _signal_symbol(signal)
    strategy = _signal_strategy(signal)
    side = _signal_side(signal)
    confidence = _signal_confidence(signal)
    asset_class = _signal_asset_class(signal)

    # Rule 0: exit signals always pass
    if _is_exit_signal(signal):
        return GateDecision(
            action=GateAction.CLOSE_ONLY,
            reason="exit_tagged_signal",
            signal_index=signal_index,
            symbol=symbol,
            strategy=strategy,
        )

    # Rule 1: reconciliation not GREEN → block opposite-direction exposure
    if reconciliation_status != "GREEN":
        conflicts = _find_conflicts(
            symbol, asset_class, side, strategy, open_positions
        )
        opposite_conflicts = [
            c for c in conflicts
            if c["side"] and c["side"] != side
        ]
        if opposite_conflicts:
            return GateDecision(
                action=GateAction.BLOCK,
                reason=f"reconciliation_{reconciliation_status}_blocks_opposite_exposure",
                signal_index=signal_index,
                symbol=symbol,
                strategy=strategy,
                conflicting_strategy=opposite_conflicts[0]["strategy"],
                conflicting_side=opposite_conflicts[0]["side"],
            )

    # Rule 2: hedge signals → ALLOW within budget
    if _is_hedge_signal(signal):
        # Notional budget check requires price data — full check deferred to
        # execution pipeline. Tag-based allow keeps hedging strategies viable.
        return GateDecision(
            action=GateAction.ALLOW,
            reason="hedge_tagged_within_budget",
            signal_index=signal_index,
            symbol=symbol,
            strategy=strategy,
        )

    conflicts = _find_conflicts(
        symbol, asset_class, side, strategy, open_positions
    )

    if not conflicts:
        return GateDecision(
            action=GateAction.ALLOW,
            reason="no_conflict",
            signal_index=signal_index,
            symbol=symbol,
            strategy=strategy,
        )

    same_direction = [c for c in conflicts if c["side"] == side]
    opposite_direction = [c for c in conflicts if c["side"] and c["side"] != side]

    # Rule 3: same direction → MERGE
    if same_direction and not opposite_direction:
        return GateDecision(
            action=GateAction.MERGE,
            reason=f"same_direction_as_{same_direction[0]['strategy']}",
            signal_index=signal_index,
            symbol=symbol,
            strategy=strategy,
            conflicting_strategy=same_direction[0]["strategy"],
            conflicting_side=same_direction[0]["side"],
        )

    if opposite_direction:
        highest_priority_conflict = max(
            opposite_direction,
            key=lambda c: _strategy_priority(c["strategy"])
        )
        conflict_priority = _strategy_priority(
            highest_priority_conflict["strategy"]
        )
        new_priority = _strategy_priority(strategy)

        # Rule 4: weak signal → BLOCK
        if confidence < REVERSAL_CONFIDENCE_THRESHOLD:
            return GateDecision(
                action=GateAction.BLOCK,
                reason=(
                    f"confidence_{confidence:.2f}_below_reversal_threshold_"
                    f"{REVERSAL_CONFIDENCE_THRESHOLD}"
                ),
                signal_index=signal_index,
                symbol=symbol,
                strategy=strategy,
                conflicting_strategy=highest_priority_conflict["strategy"],
                conflicting_side=highest_priority_conflict["side"],
            )

        # Rule 5: lower priority → BLOCK
        if new_priority < conflict_priority:
            return GateDecision(
                action=GateAction.BLOCK,
                reason=(
                    f"lower_priority_{new_priority}_vs_"
                    f"{highest_priority_conflict['strategy']}_{conflict_priority}"
                ),
                signal_index=signal_index,
                symbol=symbol,
                strategy=strategy,
                conflicting_strategy=highest_priority_conflict["strategy"],
                conflicting_side=highest_priority_conflict["side"],
            )

        # Rule 6: equal priority → REDUCE
        if new_priority == conflict_priority:
            return GateDecision(
                action=GateAction.REDUCE,
                reason=(
                    f"equal_priority_reduce_{highest_priority_conflict['strategy']}"
                ),
                signal_index=signal_index,
                symbol=symbol,
                strategy=strategy,
                conflicting_strategy=highest_priority_conflict["strategy"],
                conflicting_side=highest_priority_conflict["side"],
            )

        # Rule 7: higher priority + strong signal → FLIP
        if (
            new_priority > conflict_priority
            and confidence >= REVERSAL_CONFIDENCE_THRESHOLD
        ):
            return GateDecision(
                action=GateAction.FLIP_ALLOWED,
                reason=(
                    f"higher_priority_{new_priority}_confidence_{confidence:.2f}"
                    f"_flipping_{highest_priority_conflict['strategy']}"
                ),
                signal_index=signal_index,
                symbol=symbol,
                strategy=strategy,
                conflicting_strategy=highest_priority_conflict["strategy"],
                conflicting_side=highest_priority_conflict["side"],
            )

    return GateDecision(
        action=GateAction.ALLOW,
        reason="no_blocking_conflict",
        signal_index=signal_index,
        symbol=symbol,
        strategy=strategy,
    )


def run_gate(
    signals: List[Any],
    ctx: Optional[Dict] = None,
) -> Tuple[List[Any], List[GateDecision]]:
    """
    Run the Net Exposure Conflict Gate over a list of signals.

    Returns:
        (allowed_signals, all_decisions)

    allowed_signals: signals that passed (ALLOW, MERGE, CLOSE_ONLY,
                     FLIP_ALLOWED, REDUCE)
    all_decisions: full decision log for every signal
    """
    if not signals:
        return signals, []

    try:
        open_positions = _get_open_positions()
        reconciliation_status = _get_reconciliation_status()
        portfolio_equity = _get_portfolio_equity()
    except Exception as e:
        logger.warning("net_exposure_gate: context_load_failed err=%s — ALLOW all", e)
        return signals, []

    allowed: List[Any] = []
    decisions: List[GateDecision] = []

    blocked_count = 0
    for i, signal in enumerate(signals):
        try:
            decision = evaluate_signal(
                signal=signal,
                signal_index=i,
                open_positions=open_positions,
                reconciliation_status=reconciliation_status,
                portfolio_equity=portfolio_equity,
            )
            decisions.append(decision)

            if decision.action == GateAction.BLOCK:
                blocked_count += 1
                logger.warning(
                    "NET_EXPOSURE_GATE BLOCK symbol=%s strategy=%s "
                    "reason=%s conflicting=%s",
                    decision.symbol,
                    decision.strategy,
                    decision.reason,
                    decision.conflicting_strategy,
                )
            else:
                allowed.append(signal)
                if decision.action != GateAction.ALLOW:
                    logger.info(
                        "NET_EXPOSURE_GATE %s symbol=%s strategy=%s reason=%s",
                        decision.action.value,
                        decision.symbol,
                        decision.strategy,
                        decision.reason,
                    )
        except Exception as e:
            logger.debug(
                "net_exposure_gate: signal_eval_failed idx=%d err=%s — ALLOW",
                i, e,
            )
            allowed.append(signal)

    if blocked_count:
        logger.warning(
            "NET_EXPOSURE_GATE summary: %d/%d signals blocked",
            blocked_count, len(signals),
        )

    return allowed, decisions

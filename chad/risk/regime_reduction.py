"""
chad/risk/regime_reduction.py

Phase-8 Session 4 (G3): regime-change-triggered position reduction.

When the regime classifier transitions to a riskier label, open positions
are reduced by a configurable fraction via the existing close-intent
pipeline (chad.core.position_reconciler.apply_close_intents). No new
order submission path is introduced.

Reduction rules (first match):

    any regime → "adverse"   : reduce 50% of open quantity per symbol
    any regime → "volatile"  : reduce 30% of open quantity per symbol
    any regime → "unknown"   : WARNING only — no automatic reduction
    all other transitions    : no reduction

The 'unknown' carve-out matters: a missed-input cycle briefly classifies
as unknown, and auto-reducing on every blip would churn the book. The
operator gets a warning so transient unknowns are visible in logs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

LOG = logging.getLogger(__name__)


# Reduction percentages per adverse transition target.
DEFAULT_ADVERSE_REDUCTION_PCT: float = 0.5
DEFAULT_VOLATILE_REDUCTION_PCT: float = 0.3

# Regime labels that trigger a reduction.
_REDUCTION_TARGETS = ("adverse", "volatile")
_WARN_ONLY_TARGETS = ("unknown",)


@dataclass(frozen=True)
class ReductionDecision:
    should_reduce: bool
    pct: float = 0.0
    reason: str = ""
    warn_only: bool = False


def should_reduce_on_transition(
    from_regime: Optional[str],
    to_regime: str,
    adverse_pct: float = DEFAULT_ADVERSE_REDUCTION_PCT,
    volatile_pct: float = DEFAULT_VOLATILE_REDUCTION_PCT,
) -> ReductionDecision:
    """Decide whether a regime transition triggers a position reduction.

    from_regime == to_regime → no reduction (no actual transition).
    from_regime == None      → no reduction (first-ever classification).
    """
    if not to_regime:
        return ReductionDecision(False, 0.0, "missing_to_regime")
    from_r = (from_regime or "").lower()
    to_r = str(to_regime).lower()

    if not from_r:
        return ReductionDecision(False, 0.0, "no_previous_regime")
    if from_r == to_r:
        return ReductionDecision(False, 0.0, "no_transition")

    if to_r in _WARN_ONLY_TARGETS:
        return ReductionDecision(
            False,
            0.0,
            f"warn_only_transition:{from_r}->{to_r}",
            warn_only=True,
        )

    if to_r == "adverse":
        return ReductionDecision(
            True,
            float(adverse_pct),
            f"adverse_transition:{from_r}->{to_r}",
        )
    if to_r == "volatile":
        return ReductionDecision(
            True,
            float(volatile_pct),
            f"volatile_transition:{from_r}->{to_r}",
        )

    return ReductionDecision(False, 0.0, f"neutral_transition:{from_r}->{to_r}")


def _position_open_side(position: Mapping[str, Any]) -> Optional[str]:
    for key in ("side", "open_side", "last_side"):
        v = position.get(key)
        if v:
            return str(v).upper()
    return None


def generate_partial_close_intents(
    open_positions: Mapping[str, Mapping[str, Any]],
    reduction_pct: float,
    reason: str,
) -> List[Dict[str, Any]]:
    """Build close-intent dicts compatible with position_reconciler.apply_close_intents.

    One dict per open position. Quantities are rounded down and any
    position that would reduce to less than 1 unit is dropped — the
    close path cannot submit a fractional equity order.
    """
    pct = max(0.0, min(1.0, float(reduction_pct)))
    if pct <= 0.0:
        return []

    intents: List[Dict[str, Any]] = []
    for position_key, position in (open_positions or {}).items():
        if not isinstance(position, Mapping):
            continue
        try:
            current_qty = float(position.get("quantity", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if current_qty <= 0.0:
            continue

        open_side = _position_open_side(position)
        if open_side not in ("BUY", "SELL"):
            continue

        # Partial reduction quantity. Round down to whole units so we
        # don't ship fractional equity closes through the existing path.
        reduce_qty = int(current_qty * pct)
        if reduce_qty < 1:
            continue

        close_side = "SELL" if open_side == "BUY" else "BUY"
        strategy = position.get("strategy")
        if not strategy and isinstance(position_key, str) and "|" in position_key:
            strategy = position_key.split("|", 1)[0]

        intents.append({
            "symbol": position.get("symbol") or (
                position_key.split("|", 1)[1] if isinstance(position_key, str) and "|" in position_key else ""
            ),
            "action": "CLOSE",
            "open_side": open_side,
            "close_side": close_side,
            "quantity": float(reduce_qty),
            "reason": reason,
            "position_key": position_key,
            "strategy": strategy or "regime_reduction",
        })
    return intents


def handle_regime_transition(
    from_regime: Optional[str],
    to_regime: str,
    open_positions: Mapping[str, Mapping[str, Any]],
    adverse_pct: float = DEFAULT_ADVERSE_REDUCTION_PCT,
    volatile_pct: float = DEFAULT_VOLATILE_REDUCTION_PCT,
) -> Dict[str, Any]:
    """End-to-end G3 orchestration — returns a summary dict.

    The caller is responsible for actually applying returned close
    intents via apply_close_intents; this function does no I/O so it
    stays easy to test without a broker adapter.
    """
    decision = should_reduce_on_transition(
        from_regime, to_regime, adverse_pct=adverse_pct, volatile_pct=volatile_pct,
    )

    close_intents: List[Dict[str, Any]] = []
    if decision.should_reduce:
        close_intents = generate_partial_close_intents(
            open_positions,
            reduction_pct=decision.pct,
            reason=f"regime_change_{(from_regime or '').lower()}_to_{to_regime.lower()}",
        )
        if close_intents:
            LOG.warning(
                "REGIME_REDUCTION from=%s to=%s pct=%.2f symbols=%s",
                from_regime, to_regime, decision.pct,
                [i["symbol"] for i in close_intents],
            )
        else:
            LOG.info(
                "REGIME_REDUCTION_NOOP from=%s to=%s pct=%.2f open_positions=%d (no qualifying positions)",
                from_regime, to_regime, decision.pct, len(open_positions or {}),
            )
    elif decision.warn_only:
        LOG.warning(
            "REGIME_UNKNOWN_TRANSITION from=%s to=%s — no reduction applied (warn-only rule)",
            from_regime, to_regime,
        )

    return {
        "decision": {
            "should_reduce": decision.should_reduce,
            "pct": decision.pct,
            "reason": decision.reason,
            "warn_only": decision.warn_only,
        },
        "close_intents": close_intents,
    }


__all__ = [
    "DEFAULT_ADVERSE_REDUCTION_PCT",
    "DEFAULT_VOLATILE_REDUCTION_PCT",
    "ReductionDecision",
    "should_reduce_on_transition",
    "generate_partial_close_intents",
    "handle_regime_transition",
]

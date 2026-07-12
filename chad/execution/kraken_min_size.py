"""
Lunch-money sizing: Kraken per-pair min-order-size handling (U3 / CRYPTO-TRUST).

The mission: small accounts prove edge honestly. When the strategy's computed
size falls below the pair minimum we do NOT silently starve the order (the old
execution_pipeline.py:1546-1548 behaviour was a silent `return None`). Instead:

  * computed >= min                      -> PASS   (unchanged, no marker)
  * below min AND min affordable         -> BUMP   (round up to min, CRYPTO_MIN_SIZE_BUMP)
  * below min AND NOT affordable         -> SKIP   (loud, CRYPTO_BELOW_MIN_SKIP)

"Affordable" = the minimum's notional fits within BOTH the available account
balance AND the per-strategy risk cap. Pure + stdlib-only so it is trivially
testable at the $185 x SCR-0.1 matrix without any live feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Markers (grep-able; carried onto the intent + evidence tags).
MARKER_MIN_SIZE_BUMP = "CRYPTO_MIN_SIZE_BUMP"
MARKER_BELOW_MIN_SKIP = "CRYPTO_BELOW_MIN_SKIP"

ACTION_PASS = "PASS"
ACTION_BUMP = "BUMP"
ACTION_SKIP = "SKIP"


@dataclass(frozen=True)
class MinSizeDecision:
    action: str                 # PASS | BUMP | SKIP
    final_volume: float         # 0.0 on SKIP
    marker: Optional[str]       # None on PASS
    reason: str
    min_volume: float
    min_notional: float
    computed_volume: float

    @property
    def is_skip(self) -> bool:
        return self.action == ACTION_SKIP

    @property
    def is_bump(self) -> bool:
        return self.action == ACTION_BUMP


def _pos(x: object) -> float:
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):
        return 0.0
    return f


def decide_min_size(
    *,
    pair: str,
    computed_volume: float,
    price: float,
    min_volume: float,
    available_notional: Optional[float],
    risk_cap_notional: Optional[float],
) -> MinSizeDecision:
    """
    Decide PASS / BUMP / SKIP for a below-min crypto order.

    available_notional / risk_cap_notional are the affordability constraints for
    the minimum order's notional (min_volume * price). Either may be None to
    mean "unconstrained" (skip that check); if BOTH are None the minimum is
    always affordable (bump). A non-finite / non-positive constraint is treated
    as 0 (fail-closed -> not affordable), never as unconstrained.
    """
    cv = _pos(computed_volume)
    px = _pos(price)
    mv = _pos(min_volume)
    min_notional = mv * px

    # Degenerate min or price: nothing sensible to bump to -> PASS through the
    # computed size (upstream min-vol reject already handled the < min case in
    # the legacy path; here a zero min means "no minimum configured").
    if mv <= 0.0:
        return MinSizeDecision(
            action=ACTION_PASS, final_volume=cv, marker=None,
            reason="no_min_configured", min_volume=mv,
            min_notional=min_notional, computed_volume=cv,
        )

    if cv >= mv:
        return MinSizeDecision(
            action=ACTION_PASS, final_volume=cv, marker=None,
            reason="at_or_above_min", min_volume=mv,
            min_notional=min_notional, computed_volume=cv,
        )

    # Below the minimum. Can the account afford the minimum within risk caps?
    # None => that constraint is unconstrained; a present-but-non-positive
    # constraint is fail-closed (0) and blocks affordability.
    def _cap(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        v = _pos(x)
        return v  # non-finite/non-positive -> 0.0 (blocks)

    avail = _cap(available_notional)
    cap = _cap(risk_cap_notional)

    affordable = True
    if avail is not None and min_notional > avail:
        affordable = False
    if cap is not None and min_notional > cap:
        affordable = False

    if affordable:
        return MinSizeDecision(
            action=ACTION_BUMP, final_volume=mv, marker=MARKER_MIN_SIZE_BUMP,
            reason="below_min_affordable_bumped", min_volume=mv,
            min_notional=min_notional, computed_volume=cv,
        )

    return MinSizeDecision(
        action=ACTION_SKIP, final_volume=0.0, marker=MARKER_BELOW_MIN_SKIP,
        reason="below_min_unaffordable_skipped", min_volume=mv,
        min_notional=min_notional, computed_volume=cv,
    )

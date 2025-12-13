from __future__ import annotations

"""
OmegaBrain – crash-hedge / volatility-aware strategy.

Design goals in this Phase:
- Safe to register and enable: emits NO orders by default.
- Clean integration with existing StrategyEngine and StrategyConfig.
- Easy extension later for drawdown / volatility based hedging.

Assumptions (aligned with existing CHAD types):
- StrategyName has an OMEGA member.
- StrategyConfig matches the structure used by alpha.py / beta.py / gamma.py.
"""

from dataclasses import dataclass
from typing import List

from chad.types import StrategyConfig, StrategyName
# NOTE:
# If you have a concrete StrategySignal type, you can later switch
# the handler return type to List[StrategySignal] and update imports.


@dataclass
class OmegaParams:
    """
    Tunable parameters for OmegaBrain.

    This class is intentionally minimal for now. You can safely extend it later
    with additional controls, for example:

        crash_drawdown_threshold: float  # e.g. 0.08 for -8% equity drawdown
        vix_spike_threshold: float      # e.g. +25% VIX spike vs baseline
        hedge_notional_fraction: float  # max fraction of portfolio to hedge

    In this initial version, Omega is effectively a dormant watchdog that can
    be wired into the StrategyEngine without impacting live behaviour.
    """

    enabled: bool = True


def build_omega_config() -> StrategyConfig:
    """
    Build the StrategyConfig for OmegaBrain.

    The target universe is kept small and conservative (SPY, QQQ) so that when
    you do implement hedging logic, it naturally focuses on broad market risk.

    Returns
    -------
    StrategyConfig
        Configuration object for registering Omega with the StrategyEngine.
    """
    return StrategyConfig(
        name=StrategyName.OMEGA,
        enabled=True,
        target_universe=["SPY", "QQQ"],
    )


def omega_handler(ctx, params: OmegaParams) -> List:
    """
    Omega strategy handler (currently no-op, production-safe).

    Behaviour (Phase-7 / DRY_RUN baseline):
    - If `params.enabled` is False: returns an empty list immediately.
    - If enabled: still returns an empty list (no hedge signals).
    - No external I/O, no broker calls, no state mutation.

    Later extension points (when you are ready to activate hedging):

        1. Observe portfolio drawdown and realized/unrealized PnL in `ctx`.
        2. Observe volatility regimes (e.g., VIX, realized vol of SPY/QQQ).
        3. When both conditions justify a hedge, emit hedge signals:
           - e.g., BUY inverse ETF, BUY put options, REDUCE net exposure, etc.
        4. Size hedges using your DynamicRiskAllocator + SCR signals so that
           Omega never violates global risk caps.

    Parameters
    ----------
    ctx
        Strategy context object produced by your ContextBuilder. This is kept
        untyped here to remain compatible with your existing infrastructure.
    params : OmegaParams
        Configuration flags and tunables for Omega.

    Returns
    -------
    List
        Currently always an empty list (no strategy signals).
    """
    if not params.enabled:
        return []

    # PRODUCTION-SAFE DEFAULT:
    # Do not emit any signals until you have fully specified and tested
    # your hedging logic. This guarantees that adding Omega to the registry
    # cannot accidentally produce trades.
    return []

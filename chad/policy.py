#!/usr/bin/env python3
"""
chad/policy.py

Phase-3 Policy Layer for CHAD.

This module encapsulates *deterministic, inspectable policy decisions* that sit
between raw strategy output (TradeSignal) and downstream execution/risk
components (which are implemented in later phases).

Goals for this phase:
- Provide a central place to express per-strategy and global risk limits.
- Make it trivial to audit *why* a given signal was accepted, resized, or
  rejected.
- Keep the logic fully deterministic, side-effect free, and testable.

This does *not* talk to brokers, mutate portfolios, or track PnL; it only
evaluates proposed signals against static limits and simple exposure snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chad.types import (
    AssetClass,
    BrainRegistry,
    RoutedSignal,
    StrategyName,
    TradeSignal,
)


# ---------------------------------------------------------------------------
# Per-strategy risk and policy configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyRiskLimits:
    """
    Per-strategy risk limits and toggles.

    These are conservative defaults for Phase 3; more advanced sizing (Kelly,
    VaR, etc.) and dynamic adaptation come in later phases.
    """

    enabled: bool = True

    # Maximum absolute notional allowed per symbol for this strategy.
    # If None, no per-symbol cap is enforced here (global policy may still cap).
    max_symbol_notional: Optional[float] = None

    # Maximum aggregate notional for this strategy across all symbols.
    max_total_notional: Optional[float] = None

    # Maximum per-trade notional (before any execution slicing).
    max_trade_notional: Optional[float] = None

    # Whether the strategy is allowed to propose short exposure (< 0 quantity).
    allow_short: bool = False


@dataclass(frozen=True)
class GlobalRiskLimits:
    """
    Global risk limits that apply across all strategies.

    These are deliberately simple in Phase 3; they guard against obviously
    excessive exposure even if strategy-level limits are permissive.
    """

    # Maximum absolute notional across *all* open positions (pre-trade).
    max_total_exposure: Optional[float] = None

    # Maximum absolute notional in any single symbol (across all strategies).
    max_symbol_exposure: Optional[float] = None

    # If True, all trading is disabled regardless of other limits.
    kill_switch_enabled: bool = False


# ---------------------------------------------------------------------------
# Policy decisions / outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDecision:
    """
    Result of evaluating a single TradeSignal against policy.

    - accepted == False  → signal must not be executed.
    - accepted == True   → signal is allowed; `adjusted_size` is the size
                           that may be sent downstream (often equal to the
                           original size, but may be trimmed).
    """

    accepted: bool
    reason: str
    adjusted_size: float

    @property
    def rejected(self) -> bool:
        return not self.accepted


@dataclass(frozen=True)
class EvaluatedSignal:
    """
    Bundles a TradeSignal with its policy decision.

    Used for logging, auditing, and passing both the original intent and the
    final allowed sizing to downstream components.
    """

    signal: TradeSignal
    decision: PolicyDecision


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------


@dataclass
class PolicyEngine:
    """
    PolicyEngine evaluates TradeSignals against static policy limits.

    Inputs:
      - strategy_limits: per-strategy risk knobs.
      - global_limits: global exposure caps.
      - current_symbol_notional: snapshot of current exposure by symbol.
      - current_total_notional: snapshot of current total exposure.
      - prices: mapping symbol → latest trade price (for notional calc).

    Outputs:
      - A list of EvaluatedSignal instances indicating what is allowed and why.

    This engine is intentionally *pure* at this phase:
    - It does not mutate exposures.
    - It does not remember past decisions.
    - It does not talk to brokers.
    """

    strategy_limits: Dict[StrategyName, StrategyRiskLimits] = field(
        default_factory=dict
    )
    global_limits: GlobalRiskLimits = field(
        default_factory=lambda: GlobalRiskLimits()
    )

    # -------------------------- Public API ---------------------------------

    def evaluate_signals(
        self,
        signals: Iterable[TradeSignal],
        current_symbol_notional: Mapping[str, float],
        current_total_notional: float,
        prices: Mapping[str, float],
    ) -> List[EvaluatedSignal]:
        """
        Evaluate a batch of TradeSignals against policy.

        `current_symbol_notional` and `current_total_notional` should reflect
        *pre-trade* exposures. This function treats the batch as a set of
        atomic proposals and checks each one in isolation against limits.
        Later phases may add cumulative batch-effects if desired.
        """
        evaluated: List[EvaluatedSignal] = []

        for sig in signals:
            decision = self._evaluate_single(
                signal=sig,
                current_symbol_notional=current_symbol_notional,
                current_total_notional=current_total_notional,
                prices=prices,
            )
            evaluated.append(EvaluatedSignal(signal=sig, decision=decision))

        return evaluated

    # ---------------------- Internal implementation ------------------------

    def _evaluate_single(
        self,
        signal: TradeSignal,
        current_symbol_notional: Mapping[str, float],
        current_total_notional: float,
        prices: Mapping[str, float],
    ) -> PolicyDecision:
        """
        Evaluate a single TradeSignal against all applicable limits.
        """
        # 1) Global kill-switch check
        if self.global_limits.kill_switch_enabled:
            return PolicyDecision(
                accepted=False,
                reason="global_kill_switch_enabled",
                adjusted_size=0.0,
            )

        # 2) Strategy enabled check
        strat_limits = self.strategy_limits.get(signal.strategy)
        if strat_limits is None:
            # Default: if we have no explicit limits, treat as disabled.
            return PolicyDecision(
                accepted=False,
                reason=f"strategy_limits_missing:{signal.strategy.value}",
                adjusted_size=0.0,
            )

        if not strat_limits.enabled:
            return PolicyDecision(
                accepted=False,
                reason=f"strategy_disabled:{signal.strategy.value}",
                adjusted_size=0.0,
            )

        # 3) Shorts allowed?
        if not strat_limits.allow_short and signal.side == signal.side.SELL:
            # We treat SELL as reducing or closing long; shorting is enforced
            # at the execution layer via exposure, so we only block if this
            # would *create* net negative exposure. That requires position
            # awareness, which is not part of this Phase-3 policy engine.
            # Here we allow SELL but assume execution/risk will handle
            # no-new-short constraints.
            pass  # nothing to do at this layer

        # 4) Price and notional sanity
        price = prices.get(signal.symbol)
        if price is None or price <= 0.0:
            return PolicyDecision(
                accepted=False,
                reason="missing_or_invalid_price",
                adjusted_size=0.0,
            )

        if signal.size <= 0.0:
            return PolicyDecision(
                accepted=False,
                reason="non_positive_size",
                adjusted_size=0.0,
            )

        intended_notional = float(price) * float(signal.size)

        # 5) Per-strategy trade notional cap
        if strat_limits.max_trade_notional is not None:
            if intended_notional > strat_limits.max_trade_notional:
                # Resize the trade down to the maximum allowed notional
                max_size = strat_limits.max_trade_notional / price
                if max_size <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="max_trade_notional_zero_or_negative",
                        adjusted_size=0.0,
                    )
                return PolicyDecision(
                    accepted=True,
                    reason="resized_to_max_trade_notional",
                    adjusted_size=max_size,
                )

        # 6) Per-symbol notional cap (strategy-level)
        symbol_notional = abs(current_symbol_notional.get(signal.symbol, 0.0))
        if strat_limits.max_symbol_notional is not None:
            if symbol_notional + intended_notional > strat_limits.max_symbol_notional:
                # If even a 1-unit trade would breach, reject outright.
                remaining = strat_limits.max_symbol_notional - symbol_notional
                if remaining <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="max_symbol_notional_reached",
                        adjusted_size=0.0,
                    )
                max_size = remaining / price
                if max_size <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="max_symbol_notional_zero_or_negative",
                        adjusted_size=0.0,
                    )
                return PolicyDecision(
                    accepted=True,
                    reason="resized_to_symbol_notional_cap",
                    adjusted_size=max_size,
                )

        # 7) Per-strategy total notional cap
        if strat_limits.max_total_notional is not None:
            # For Phase 3 we approximate strategy total notional by the global
            # total notional; a more precise per-strategy decomposition is left
            # to later phases where position labelling is implemented.
            if current_total_notional + intended_notional > strat_limits.max_total_notional:
                remaining = strat_limits.max_total_notional - current_total_notional
                if remaining <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="max_strategy_notional_reached",
                        adjusted_size=0.0,
                    )
                max_size = remaining / price
                if max_size <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="max_strategy_notional_zero_or_negative",
                        adjusted_size=0.0,
                    )
                return PolicyDecision(
                    accepted=True,
                    reason="resized_to_strategy_notional_cap",
                    adjusted_size=max_size,
                )

        # 8) Global symbol exposure cap
        if self.global_limits.max_symbol_exposure is not None:
            symbol_total = abs(current_symbol_notional.get(signal.symbol, 0.0))
            if symbol_total + intended_notional > self.global_limits.max_symbol_exposure:
                remaining = self.global_limits.max_symbol_exposure - symbol_total
                if remaining <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="global_symbol_exposure_cap_reached",
                        adjusted_size=0.0,
                    )
                max_size = remaining / price
                if max_size <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="global_symbol_exposure_cap_zero_or_negative",
                        adjusted_size=0.0,
                    )
                return PolicyDecision(
                    accepted=True,
                    reason="resized_to_global_symbol_exposure_cap",
                    adjusted_size=max_size,
                )

        # 9) Global total exposure cap
        if self.global_limits.max_total_exposure is not None:
            if current_total_notional + intended_notional > self.global_limits.max_total_exposure:
                remaining = self.global_limits.max_total_exposure - current_total_notional
                if remaining <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="global_total_exposure_cap_reached",
                        adjusted_size=0.0,
                    )
                max_size = remaining / price
                if max_size <= 0.0:
                    return PolicyDecision(
                        accepted=False,
                        reason="global_total_exposure_cap_zero_or_negative",
                        adjusted_size=0.0,
                    )
                return PolicyDecision(
                    accepted=True,
                    reason="resized_to_global_total_exposure_cap",
                    adjusted_size=max_size,
                )

        # If we reach here, no limit was breached.
        return PolicyDecision(
            accepted=True,
            reason="accepted",
            adjusted_size=signal.size,
        )


# ---------------------------------------------------------------------------
# Default policy set for Phase 3
# ---------------------------------------------------------------------------


def build_default_strategy_limits() -> Dict[StrategyName, StrategyRiskLimits]:
    """
    Construct a conservative default set of per-strategy limits for Phase 3.

    These can be replaced or extended later via config files or a database,
    but this function provides a single, auditable baseline for the current
    instance.
    """
    limits: Dict[StrategyName, StrategyRiskLimits] = {}

    # Alpha: intraday tactical, small per-trade, no explicit total cap here
    limits[StrategyName.ALPHA] = StrategyRiskLimits(
        enabled=True,
        max_symbol_notional=50_000.0,
        max_total_notional=150_000.0,
        max_trade_notional=10_000.0,
        allow_short=False,
    )

    # Beta: slow legend allocator, smaller trade and symbol caps
    limits[StrategyName.BETA] = StrategyRiskLimits(
        enabled=True,
        max_symbol_notional=40_000.0,
        max_total_notional=200_000.0,
        max_trade_notional=7_500.0,
        allow_short=False,
    )

    # Gamma / Omega / AlphaCrypto / AlphaForex / Delta are not yet wired on
    # this instance; keep them disabled at the policy layer until their
    # full strategies are restored.
    limits[StrategyName.GAMMA] = StrategyRiskLimits(
        enabled=False,
    )
    limits[StrategyName.OMEGA] = StrategyRiskLimits(
        enabled=False,
    )
    limits[StrategyName.ALPHA_CRYPTO] = StrategyRiskLimits(
        enabled=False,
    )
    limits[StrategyName.ALPHA_FOREX] = StrategyRiskLimits(
        enabled=False,
    )
    limits[StrategyName.DELTA] = StrategyRiskLimits(
        enabled=False,
    )

    return limits


def build_default_global_limits() -> GlobalRiskLimits:
    """
    Construct a conservative default global risk limit set for Phase 3.

    These values are deliberately modest; as CHAD's full risk engine is
    restored (drawdown tracking, Kelly sizing, etc.), we can tighten or
    relax these based on real performance.
    """
    return GlobalRiskLimits(
        max_total_exposure=500_000.0,
        max_symbol_exposure=150_000.0,
        kill_switch_enabled=False,
    )

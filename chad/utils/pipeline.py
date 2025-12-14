#!/usr/bin/env python3
"""
chad/utils/pipeline.py

Decision pipeline for CHAD Phase 3.

This module wires together:
    - StrategyEngine      (brains: Alpha, Beta, etc.)
    - PolicyEngine        (risk/policy limits)
    - SignalRouter        (merging and consolidation)

Into a single, deterministic call that:
    1) builds/receives a MarketContext,
    2) runs all registered strategies,
    3) evaluates their TradeSignals against policy,
    4) keeps only accepted signals (with adjusted sizes),
    5) routes the accepted signals into RoutedSignal objects.

It does NOT:
    - Talk to brokers
    - Mutate positions
    - Track PnL
    - Persist state

Those responsibilities are handled in later phases by the execution, accounting,
and storage layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping

from chad.engine import StrategyEngine
from chad.policy import (
    EvaluatedSignal,
    PolicyDecision,
    PolicyEngine,
    build_default_global_limits,
    build_default_strategy_limits,
)
from chad.types import MarketContext, TradeSignal
from chad.utils.event_bus import get_event_bus
from chad.utils.signal_router import RouterConfig, RoutedSignal, SignalRouter


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuration for the decision pipeline.

    Phase 3 keeps this intentionally lean; later phases can add knobs for
    toggling specific strategy sets, alternative policy profiles, etc.
    """

    use_policy: bool = True
    router_config: RouterConfig = field(default_factory=RouterConfig)


@dataclass(frozen=True)
class PipelineResult:
    """
    Full result of a pipeline cycle.

    Contains:
      - raw_signals: all TradeSignals produced by strategies before policy.
      - evaluated_signals: per-signal policy decisions.
      - routed_signals: merged signals that passed policy and routing.
    """

    context: MarketContext
    raw_signals: List[TradeSignal]
    evaluated_signals: List[EvaluatedSignal]
    routed_signals: List[RoutedSignal]


class DecisionPipeline:
    """
    High-level decision pipeline for CHAD.

    Typical usage:

        engine = StrategyEngine()
        # ... register strategies ...
        policy = PolicyEngine(
            strategy_limits=build_default_strategy_limits(),
            global_limits=build_default_global_limits(),
        )
        router = SignalRouter()

        pipeline = DecisionPipeline(engine, policy, router)

        result = pipeline.run(
            ctx=context,
            prices=latest_prices,
            current_symbol_notional=exposure_by_symbol,
            current_total_notional=total_exposure,
        )

    This class is deliberately stateless; all stateful concerns (portfolio,
    exposures, logs, persistence) are handled by the caller.
    """

    def __init__(
        self,
        engine: StrategyEngine,
        policy: PolicyEngine | None = None,
        router: SignalRouter | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.engine = engine
        self.policy = policy or PolicyEngine(
            strategy_limits=build_default_strategy_limits(),
            global_limits=build_default_global_limits(),
        )
        # If a router is provided, prefer it; otherwise build from config
        cfg = config or PipelineConfig()
        self.router = router or SignalRouter(config=cfg.router_config)
        self.config = cfg
        self._bus = get_event_bus()

    # ------------------------------------------------------------------ #
    # Core entrypoint
    # ------------------------------------------------------------------ #

    def run(
        self,
        ctx: MarketContext,
        prices: Mapping[str, float],
        current_symbol_notional: Mapping[str, float],
        current_total_notional: float,
    ) -> PipelineResult:
        """
        Run one full decision pipeline cycle.

        Args:
            ctx:
                MarketContext (ticks, legend, portfolio) for this cycle.
            prices:
                Mapping symbol → latest price, used for policy notional checks.
            current_symbol_notional:
                Mapping symbol → current absolute notional exposure.
            current_total_notional:
                Current absolute total notional exposure.

        Returns:
            PipelineResult with raw, evaluated, and routed signals.
        """
        # 1) Run strategies through the StrategyEngine
        raw_signals = self.engine.run_cycle(ctx)

        self._bus.emit(
            "pipeline.raw_signals",
            {
                "count": len(raw_signals),
                "timestamp": ctx.now.isoformat(),
            },
        )

        # 2) Apply policy (if enabled)
        if self.config.use_policy:
            evaluated = self.policy.evaluate_signals(
                signals=raw_signals,
                current_symbol_notional=current_symbol_notional,
                current_total_notional=current_total_notional,
                prices=prices,
            )
        else:
            # Passthrough: mark all as accepted with no resizing.
            evaluated = [
                EvaluatedSignal(
                    signal=s,
                    decision=PolicyDecision(
                        accepted=True,
                        reason="policy_disabled",
                        adjusted_size=s.size,
                    ),
                )
                for s in raw_signals
            ]

        accepted_signals: List[TradeSignal] = []
        for ev in evaluated:
            if ev.decision.accepted and ev.decision.adjusted_size > 0.0:
                sig = ev.signal
                if ev.decision.adjusted_size != sig.size:
                    sig = TradeSignal(
                        strategy=sig.strategy,
                        symbol=sig.symbol,
                        side=sig.side,
                        size=ev.decision.adjusted_size,
                        confidence=sig.confidence,
                        asset_class=sig.asset_class,
                        time_in_force=sig.time_in_force,
                        origin=sig.origin,
                        created_at=sig.created_at,
                        tags=sig.tags,
                        meta=sig.meta,
                    )
                accepted_signals.append(sig)

        self._bus.emit(
            "pipeline.post_policy",
            {
                "raw_count": len(raw_signals),
                "accepted_count": len(accepted_signals),
                "timestamp": ctx.now.isoformat(),
            },
        )

        # 3) Route accepted signals
        routed = self.router.route(accepted_signals)

        self._bus.emit(
            "pipeline.routed",
            {
                "routed_count": len(routed),
                "timestamp": ctx.now.isoformat(),
            },
        )

        return PipelineResult(
            context=ctx,
            raw_signals=raw_signals,
            evaluated_signals=evaluated,
            routed_signals=routed,
        )

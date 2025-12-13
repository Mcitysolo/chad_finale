#!/usr/bin/env python3
"""
chad/engine.py

Phase-3 Strategy Engine for CHAD.

Responsibilities:
- Hold a registry of strategies (brains) as callables.
- For each run cycle:
    * Build a MarketContext (done by caller).
    * Call each enabled strategy.
    * Collect TradeSignals.
    * Update BrainRegistry health.
    * Emit events via the EventBus.

Execution / risk / broker wiring is handled in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from chad.types import (
    BrainRegistry,
    BrainStatus,
    MarketContext,
    StrategyConfig,
    StrategyName,
    TradeSignal,
)
from chad.utils.event_bus import Event, get_event_bus


# A strategy handler takes a MarketContext and returns a list of TradeSignals.
StrategyHandler = Callable[[MarketContext], Sequence[TradeSignal]]


@dataclass
class RegisteredStrategy:
    """Internal wrapper for a strategy brain."""

    config: StrategyConfig
    handler: StrategyHandler


@dataclass
class StrategyEngine:
    """
    Core strategy engine for CHAD.

    This engine does not know about brokers, orders, or positions updates.
    It only:
      - Runs strategies
      - Produces TradeSignals
      - Tracks per-brain health/status
    """

    strategies: Dict[StrategyName, RegisteredStrategy] = field(default_factory=dict)
    brain_registry: BrainRegistry = field(default_factory=BrainRegistry)

    def register(self, config: StrategyConfig, handler: StrategyHandler) -> None:
        """
        Register a strategy with its config and handler.
        """
        if config.name in self.strategies:
            raise ValueError(f"Strategy {config.name.value} already registered")

        self.strategies[config.name] = RegisteredStrategy(config=config, handler=handler)
        self.brain_registry.ensure(config.name)

    def list_strategies(self) -> List[StrategyConfig]:
        """
        Return configs for all registered strategies.
        """
        return [rs.config for rs in self.strategies.values()]

    # ------------------------------------------------------------------ #
    # Run cycle
    # ------------------------------------------------------------------ #

    def run_cycle(self, ctx: MarketContext) -> List[TradeSignal]:
        """
        Run one full strategy cycle:

        - Iterate all registered strategies.
        - Skip disabled ones.
        - Call each handler with the MarketContext.
        - Collect signals.
        - Update BrainRegistry status.
        - Emit events to EventBus.

        Returns an aggregated list of TradeSignals.
        """
        event_bus = get_event_bus()
        all_signals: List[TradeSignal] = []

        for name, registered in self.strategies.items():
            config = registered.config
            brain_status: BrainStatus = self.brain_registry.ensure(name)

            if not config.enabled:
                brain_status.record_error("disabled_by_config")
                event_bus.emit(
                    "strategy.skipped",
                    {
                        "strategy": name.value,
                        "reason": "disabled",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                continue

            try:
                signals = list(registered.handler(ctx))
                all_signals.extend(signals)
                brain_status.heartbeat(signal_count=len(signals))

                event_bus.emit(
                    "strategy.ran",
                    {
                        "strategy": name.value,
                        "signal_count": len(signals),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                brain_status.record_error(str(exc))
                event_bus.emit(
                    "strategy.error",
                    {
                        "strategy": name.value,
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

        # Aggregate-level event
        event_bus.emit(
            "engine.cycle_completed",
            {
                "total_signals": len(all_signals),
                "strategies": [n.value for n in self.strategies.keys()],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return all_signals

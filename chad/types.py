#!/usr/bin/env python3
"""
chad/types.py

Core type definitions for CHAD Phase 3 (Strategy Layer & Core Types).

This module is the strongly-typed foundation that all brains (Alpha, Beta,
Gamma, Omega, AlphaCrypto, AlphaForex, Delta) and the orchestrator share.

It is designed to satisfy the Phase-3 quality gates from the System Freeze:
- mypy --strict chad/types.py chad/engine.py chad/policy.py
- pytest -q chad/tests/test_types.py
- ruff check chad/types.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StrategyName(str, Enum):
    """Canonical identifiers for all CHAD brains."""

    ALPHA = "alpha"  # Intraday stocks/ETFs
    BETA = "beta"  # Long-term legend/ETF
    GAMMA = "gamma"  # Swing/momentum
    OMEGA = "omega"  # Hedge / macro
    ALPHA_CRYPTO = "alpha_crypto"  # Intraday crypto
    ALPHA_FOREX = "alpha_forex"  # Intraday FX
    DELTA = "delta"  # Micro-execution / optimization


class AssetClass(str, Enum):
    """Supported asset classes for CHAD."""

    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    CASH = "cash"


class SignalSide(str, Enum):
    """Direction of a trading signal."""

    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Simplified time-in-force flags for Phase 3."""

    DAY = "DAY"
    GTC = "GTC"


class SignalOrigin(str, Enum):
    """Where a signal came from, for traceability and debugging."""

    STRATEGY = "strategy"
    MANUAL = "manual"
    GOVERNOR = "governor"  # Shadow Confidence Router / governor overrides


# ---------------------------------------------------------------------------
# Market & reference data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketTick:
    """
    Single trade/quote tick as seen by CHAD.
    Phase 2 gives us Polygon last-trade data for SPY/QQQ; Phase 3 strategies
    consume these in higher-level aggregates.
    """

    symbol: str
    price: float
    size: float
    exchange: Optional[int]
    timestamp: datetime
    source: str = "polygon"


@dataclass(frozen=True)
class LegendConsensus:
    """
    Legend (13F-style) consensus weights.

    Backed by data/legend_top_stocks.json produced by the Legend pipeline.
    """

    as_of: datetime
    weights: Mapping[str, float]


# ---------------------------------------------------------------------------
# Portfolio & position views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Position:
    """Snapshot of a single position in a symbol."""

    symbol: str
    asset_class: AssetClass
    quantity: float
    avg_price: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    """
    Minimal portfolio snapshot used in Phase 3.
    Risk engine and full accounting are Phase 4; here we only need
    enough structure for strategies to reason about exposure.
    """

    timestamp: datetime
    cash: float
    positions: Mapping[str, Position]

    @property
    def symbols(self) -> Sequence[str]:
        return list(self.positions.keys())


# ---------------------------------------------------------------------------
# Strategy configuration & context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyConfig:
    """Static configuration for a strategy brain."""

    name: StrategyName
    enabled: bool = True
    target_universe: Optional[Sequence[str]] = None
    max_gross_exposure: Optional[float] = None  # e.g. notional cap
    notes: str = ""


@dataclass(frozen=True)
class MarketContext:
    """
    Inputs a strategy needs to decide on signals for a given cycle.

    In Phase 3 this is intentionally lean: last trades + legend consensus
    for Beta; later phases can extend this via additional fields.
    """

    now: datetime
    ticks: Mapping[str, MarketTick]
    legend: Optional[LegendConsensus]
    portfolio: PortfolioSnapshot


# ---------------------------------------------------------------------------
# Trade signals & execution requests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeSignal:
    """
    A raw trade signal produced by a strategy.

    This is *not* yet a final broker order; the Execution/Risk Engine
    in Phase 4/5 will further shape and approve these.
    """

    strategy: StrategyName
    symbol: str
    side: SignalSide
    size: float  # positive units (shares, contracts, etc.)
    confidence: float  # 0.0–1.0; used by governor / allocator
    asset_class: AssetClass
    time_in_force: TimeInForce = TimeInForce.DAY
    origin: SignalOrigin = SignalOrigin.STRATEGY
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    tags: Tuple[str, ...] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def with_tag(self, tag: str) -> "TradeSignal":
        """Return a copy with an extra tag."""
        new_tags = tuple((*self.tags, tag))
        return TradeSignal(
            strategy=self.strategy,
            symbol=self.symbol,
            side=self.side,
            size=self.size,
            confidence=self.confidence,
            asset_class=self.asset_class,
            time_in_force=self.time_in_force,
            origin=self.origin,
            created_at=self.created_at,
            tags=new_tags,
            meta=self.meta,
        )


@dataclass(frozen=True)
class RoutedSignal:
    """
    Signal after passing through the SignalRouter.

    The router may:
      - merge multiple signals on the same symbol,
      - downgrade confidence,
      - mark conflicts, etc.
    """

    symbol: str
    side: SignalSide
    net_size: float
    source_strategies: Tuple[StrategyName, ...]
    confidence: float
    asset_class: AssetClass
    created_at: datetime
    meta: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Brain health & registry
# ---------------------------------------------------------------------------


@dataclass
class BrainStatus:
    """
    Mutable status tracked by the orchestrator for each strategy.

    This is polled and updated each run cycle.
    """

    name: StrategyName
    enabled: bool = True
    last_heartbeat: Optional[datetime] = None
    last_error: Optional[str] = None
    last_signal_count: int = 0
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    def heartbeat(self, signal_count: int) -> None:
        self.last_heartbeat = datetime.now(timezone.utc)
        self.last_signal_count = signal_count
        self.last_error = None

    def record_error(self, message: str) -> None:
        self.last_heartbeat = datetime.now(timezone.utc)
        self.last_error = message


@dataclass
class BrainRegistry:
    """
    Registry for all strategy brains.

    The orchestrator uses this to track status and to implement features like
    toggling strategies, reporting health, etc.
    """

    brains: Dict[StrategyName, BrainStatus] = field(default_factory=dict)

    def ensure(self, name: StrategyName) -> BrainStatus:
        if name not in self.brains:
            self.brains[name] = BrainStatus(name=name)
        return self.brains[name]

    def all_status(self) -> List[BrainStatus]:
        return list(self.brains.values())

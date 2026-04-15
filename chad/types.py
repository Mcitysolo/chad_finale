#!/usr/bin/env python3
"""
chad/types.py

CHAD Core Types — Single Source of Truth (SSOT)

This module defines CHAD’s foundational domain types used across:
- ContextBuilder / MarketContext (input plane)
- Strategy signals and configurations
- Portfolio snapshots
- Risk policy references
- Execution planning interfaces

Design Guarantees
-----------------
- Backward compatible defaults: adding new fields is non-breaking.
- Strong typing with conservative, stable dataclasses.
- JSON-serializable patterns where appropriate.
- No external dependencies.
- No runtime side effects.

Key Upgrade (Phase 4 enablement)
--------------------------------
Adds optional `bars` to MarketContext:
- Strategies like Alpha/Gamma/Delta/Omega can now access OHLCV bars when provided.
- Defaults to None so existing callers remain valid.

Bars format expectation (best-effort)
-------------------------------------
MarketContext.bars is an Optional mapping:
  { "AAPL": [ {open,high,low,close,volume,ts_utc}, ... ], ... }

Strategies should remain defensive if bars are missing or malformed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math


# -----------------------------
# Enums
# -----------------------------

class StrategyName(str, Enum):
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    OMEGA = "omega"
    DELTA = "delta"
    ALPHA_CRYPTO = "alpha_crypto"
    ALPHA_INTRADAY = "alpha_intraday"
    ALPHA_FOREX = "alpha_forex"
    ALPHA_FUTURES = "alpha_futures"
    GAMMA_FUTURES = "gamma_futures"
    OMEGA_MACRO = "omega_macro"
    GAMMA_REVERSION = "gamma_reversion"
    ALPHA_OPTIONS = "alpha_options"
    OMEGA_VOL = "omega_vol"
    DELTA_PAIRS = "delta_pairs"

class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    FOREX = "forex"
    CASH = "cash"
    FUTURES = "futures"
    OPTIONS = "options"

class SignalSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# -----------------------------
# Strategy config + signals
# -----------------------------

@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """
    Strategy registration configuration for StrategyEngine.

    target_universe may be None for strategies that derive symbols dynamically.
    """
    name: StrategyName
    enabled: bool = True
    target_universe: Optional[Sequence[str]] = None
    max_gross_exposure: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TradeSignal:
    """
    Strategy output intent. Still subject to policy + risk + execution gates.

    Notes:
    - size is "units" (strategy-level), not notional dollars.
    - confidence is a 0..1 hint, not a guarantee.
    - meta is optional diagnostic payload.

    Backward-compatibility contract:
    - created_at defaults to datetime.utcnow so older callers remain valid
    - meta defaults to {} so older callers do not break
    """
    strategy: StrategyName
    symbol: str
    side: SignalSide
    size: float
    confidence: float
    asset_class: AssetClass
    created_at: datetime = field(default_factory=datetime.utcnow)
    meta: Dict[str, Any] = field(default_factory=dict)
# -----------------------------
# Routed output (Phase 3 contract)
# -----------------------------

@dataclass(frozen=True, slots=True)
class RoutedSignal:
    """
    Phase 3 routed signal (SSOT contract).

    This type is the canonical output of SignalRouter and the canonical input to:
      - policy
      - daily_throttle
      - execution planning / adapters

    Contract compatibility (locked by tests):
      - net_size (float) is the routed quantity (units, not dollars)
      - source_strategies is a tuple[StrategyName, ...]
      - created_at may be None (tests use None in some cases)
    """
    symbol: str
    side: SignalSide
    net_size: float
    source_strategies: Tuple[StrategyName, ...]
    confidence: float
    asset_class: AssetClass
    created_at: Optional[datetime] = None
    meta: Optional[Dict[str, Any]] = None

    # Best-effort market price for notional estimation (may be 0.0 if unknown)
    price: float = 0.0

    # Primary strategy: highest-size contributor, alphabetical tie-break
    primary_strategy: Optional[str] = None

    # Optional upstream idempotency + tags
    idempotency_key: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        def _ff(x: Any, d: float = 0.0) -> float:
            try:
                v = float(x)
                return v if math.isfinite(v) else float(d)
            except Exception:
                return float(d)

        sym = (self.symbol or "").strip().upper()
        object.__setattr__(self, "symbol", sym)

        object.__setattr__(self, "net_size", _ff(self.net_size, 0.0))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, _ff(self.confidence, 0.0))))
        object.__setattr__(self, "price", max(0.0, _ff(self.price, 0.0)))

        object.__setattr__(self, "source_strategies", tuple(self.source_strategies or ()))
        object.__setattr__(self, "tags", tuple(self.tags or ()))

    @property
    def size(self) -> float:
        """
        Backward-compatible alias: some newer code may refer to .size.
        The SSOT contract uses net_size.
        """
        return float(self.net_size)

        object.__setattr__(self, "size", _ff(self.size, 0.0))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, _ff(self.confidence, 0.0))))
        object.__setattr__(self, "price", max(0.0, _ff(self.price, 0.0)))

        # Normalize symbol
        sym = (self.symbol or "").strip().upper()
        object.__setattr__(self, "symbol", sym)

        # Ensure tuple type for strategies/tags
        object.__setattr__(self, "source_strategies", tuple(self.source_strategies or ()))
        object.__setattr__(self, "tags", tuple(self.tags or ()))


# -----------------------------
# Market + portfolio types
# -----------------------------

@dataclass(frozen=True, slots=True)
class MarketTick:
    symbol: str
    price: float
    size: float
    exchange: Optional[int]
    timestamp: datetime
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class LegendConsensus:
    as_of: datetime
    weights: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    asset_class: AssetClass
    quantity: float
    avg_price: float


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    positions: Mapping[str, Position]
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class MarketContext:
    """
    Canonical context passed to strategies.

    Optional context surfaces:
      - bars: optional OHLCV series per symbol
      - spread_bps: optional per-symbol spread in basis points
      - dollar_volume / volume_usd / liquidity_usd: optional per-symbol liquidity views
      - volatility: optional per-symbol realized/estimated volatility

    Backward compatibility:
      all optional fields default to None so existing callers remain valid.
    """
    now: datetime
    ticks: Mapping[str, MarketTick]
    legend: Optional[LegendConsensus]
    portfolio: PortfolioSnapshot
    bars: Optional[Mapping[str, list]] = None
    bars_1m: Mapping[str, Any] = field(default_factory=dict)
    spread_bps: Optional[Mapping[str, float]] = None
    dollar_volume: Optional[Mapping[str, float]] = None
    volume_usd: Optional[Mapping[str, float]] = None
    liquidity_usd: Optional[Mapping[str, float]] = None
    volatility: Optional[Mapping[str, float]] = None


# -----------------------------
# Optional: policy/risk helper enums used elsewhere
# -----------------------------

class RiskLane(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    EXIT_ONLY = "exit_only"
    DENY_ALL = "deny_all"


class GovernorName(str, Enum):
    GOVERNOR = "governor"
# -----------------------------
# Brain health registry (Phase 3 engine contract)
# -----------------------------

@dataclass(slots=True)
class BrainStatus:
    """
    Lightweight per-brain health tracker.

    Used by StrategyEngine:
      - record_error(reason)
      - heartbeat(signal_count=...)
    """
    name: StrategyName
    ok: bool = True
    last_heartbeat_utc: Optional[datetime] = None
    last_error_utc: Optional[datetime] = None
    last_error_reason: str = ""
    total_cycles: int = 0
    total_signals: int = 0
    total_errors: int = 0

    def heartbeat(self, *, signal_count: int = 0) -> None:
        self.ok = True
        self.last_heartbeat_utc = datetime.utcnow()
        self.total_cycles += 1
        if isinstance(signal_count, int) and signal_count > 0:
            self.total_signals += int(signal_count)

    def record_error(self, reason: str) -> None:
        self.ok = False
        self.total_errors += 1
        self.last_error_utc = datetime.utcnow()
        self.last_error_reason = str(reason or "unknown_error")


@dataclass(slots=True)
class BrainRegistry:
    """
    Registry mapping StrategyName -> BrainStatus.

    StrategyEngine expects:
      - ensure(name) -> BrainStatus
    """
    brains: Dict[StrategyName, BrainStatus] = field(default_factory=dict)

    def ensure(self, name: StrategyName) -> BrainStatus:
        if name not in self.brains:
            self.brains[name] = BrainStatus(name=name)
        return self.brains[name]

    def snapshot(self) -> Dict[str, Any]:
        # Safe JSON-ish snapshot for status surfaces (optional).
        out: Dict[str, Any] = {}
        for k, v in self.brains.items():
            out[str(k.value)] = {
                "ok": bool(v.ok),
                "last_heartbeat_utc": v.last_heartbeat_utc.isoformat() if v.last_heartbeat_utc else None,
                "last_error_utc": v.last_error_utc.isoformat() if v.last_error_utc else None,
                "last_error_reason": v.last_error_reason,
                "total_cycles": int(v.total_cycles),
                "total_signals": int(v.total_signals),
                "total_errors": int(v.total_errors),
            }
        return out

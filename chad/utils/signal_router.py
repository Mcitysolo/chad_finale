#!/usr/bin/env python3
"""
chad/utils/signal_router.py

SignalRouter: Phase-3 consolidation of raw TradeSignal objects into
RoutedSignal objects.

Responsibilities (Phase 3):
- Group signals by (symbol, side, asset_class).
- Merge sizes.
- Aggregate confidence (size-weighted average).
- Track which strategies contributed to the routed signal.
- Optionally apply simple filters (e.g. net_size == 0 → drop).

Later phases (4/5/6) will add:
- Shadow Confidence Router (SCR) overlays.
- Governor policies for live-fraction control.
- Conflict resolution between brains.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

from chad.types import (
    AssetClass,
    RoutedSignal,
    SignalSide,
    StrategyName,
    TradeSignal,
)


@dataclass
class RouterConfig:
    """
    Configuration for the SignalRouter.

    For Phase 3 we keep this intentionally small. Later phases may extend it.
    """

    drop_zero_net: bool = True


class SignalRouter:
    """
    Merge raw TradeSignals into RoutedSignals.

    Example:
        alpha:  BUY  +100 AAPL (conf=0.8)
        gamma: SELL  -30  AAPL (conf=0.6)  [notional in opposite dir]

    These are grouped by (symbol, side) separately:
        BUY bucket:  +100 from alpha
        SELL bucket: 30  from gamma

    So you end up with two RoutedSignals. Netting between sides is handled
    at the risk/execution layer, not here, to keep routing semantics clear.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def route(self, signals: Iterable[TradeSignal]) -> List[RoutedSignal]:
        """
        Merge a sequence of TradeSignals into a list of RoutedSignals.
        """
        buckets: Dict[
            Tuple[str, SignalSide, AssetClass],
            Dict[str, object],
        ] = {}

        for sig in signals:
            key = (sig.symbol, sig.side, sig.asset_class)
            if key not in buckets:
                buckets[key] = {
                    "size": 0.0,
                    "weighted_conf": 0.0,
                    "strategies": set(),  # type: ignore[var-annotated]
                    "created_at": sig.created_at,
                }

            bucket = buckets[key]
            size = float(sig.size)

            bucket["size"] = float(bucket["size"]) + size
            bucket["weighted_conf"] = float(bucket["weighted_conf"]) + size * float(
                sig.confidence
            )
            bucket["strategies"].add(sig.strategy)  # type: ignore[union-attr]

            # Track the most recent timestamp for the bucket
            if sig.created_at > bucket["created_at"]:  # type: ignore[operator]
                bucket["created_at"] = sig.created_at  # type: ignore[assignment]

        routed: List[RoutedSignal] = []

        for (symbol, side, asset_class), data in buckets.items():
            net_size: float = float(data["size"])  # type: ignore[assignment]

            if self.config.drop_zero_net and abs(net_size) <= 0.0:
                # Nothing to do for this bucket
                continue

            total_weight = float(data["size"])  # type: ignore[arg-type]
            if total_weight <= 0:
                # Defensive: skip buckets that somehow ended up non-positive
                continue

            weighted_conf: float = float(data["weighted_conf"])  # type: ignore[assignment]
            confidence = max(0.0, min(1.0, weighted_conf / total_weight))

            source_strategies = tuple(
                sorted(
                    data["strategies"],  # type: ignore[arg-type]
                    key=lambda s: s.value,
                )
            )

            created_at = data["created_at"]  # type: ignore[assignment]

            routed.append(
                RoutedSignal(
                    symbol=symbol,
                    side=side,
                    net_size=net_size,
                    source_strategies=source_strategies,
                    confidence=confidence,
                    asset_class=asset_class,
                    created_at=created_at,
                    meta={},
                )
            )

        # Stable ordering: symbol then side
        routed.sort(key=lambda rs: (rs.symbol, rs.side.value))
        return routed

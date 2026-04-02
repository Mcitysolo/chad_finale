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

        Merge contract (SSOT v6.4):
        - Same-symbol, same-side signals are merged additively (sizes sum).
        - source_strategies lists all contributing strategies (sorted by name).
        - primary_strategy is the contributor with the largest total size;
          ties broken alphabetically by strategy name.
        - Confidence is the size-weighted average across contributors,
          clamped to [0.0, 1.0].
        - Opposite-side signals on the same symbol remain as separate
          RoutedSignals — netting is deferred to risk/execution.
        - Buckets with net_size == 0 are dropped (configurable via drop_zero_net).

        Cross-strategy duplicate handling:
        When two or more strategies produce signals for the same symbol+side,
        this is NOT treated as a duplicate or conflict — it is a MERGE.
        No signal is ever lost: every input TradeSignal contributes to exactly
        one output RoutedSignal bucket keyed by (symbol, side, asset_class).

        Example:
          alpha:  BUY AAPL size=100 confidence=0.8
          gamma:  BUY AAPL size=50  confidence=0.6
          →  RoutedSignal BUY AAPL net_size=150
             confidence=(100*0.8 + 50*0.6)/150 = 0.733
             source_strategies=(alpha, gamma)
             primary_strategy=alpha  (largest size contributor)
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
                    "size_by_strategy": {},  # type: ignore[var-annotated]
                    "created_at": sig.created_at,
                }

            bucket = buckets[key]
            size = float(sig.size)

            bucket["size"] = float(bucket["size"]) + size
            bucket["weighted_conf"] = float(bucket["weighted_conf"]) + size * float(
                sig.confidence
            )
            bucket["strategies"].add(sig.strategy)  # type: ignore[union-attr]

            # Track per-strategy size for primary_strategy resolution
            strat_key = sig.strategy
            sbs = bucket["size_by_strategy"]  # type: ignore[assignment]
            sbs[strat_key] = float(sbs.get(strat_key, 0.0)) + size  # type: ignore[union-attr]

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

            # Resolve primary_strategy: largest size contributor, alphabetical tie-break
            sbs = data["size_by_strategy"]  # type: ignore[assignment]
            primary = sorted(
                sbs.items(),  # type: ignore[union-attr]
                key=lambda kv: (-float(kv[1]), kv[0].value),
            )[0][0]

            created_at = data["created_at"]  # type: ignore[assignment]

            routed.append(
                RoutedSignal(
                    symbol=symbol,
                    side=side,
                    net_size=net_size,
                    source_strategies=source_strategies,
                    primary_strategy=primary.value if hasattr(primary, "value") else str(primary),
                    confidence=confidence,
                    asset_class=asset_class,
                    created_at=created_at,
                    meta={},
                )
            )

        # Stable ordering: symbol then side
        routed.sort(key=lambda rs: (rs.symbol, rs.side.value))
        return routed
# -----------------------------
# Backwards-compatible functional entrypoint
# -----------------------------
def route_signals(signals: Iterable[TradeSignal], config: RouterConfig | None = None) -> List[RoutedSignal]:
    """
    Compatibility wrapper expected by chad.utils.pipeline.

    Phase-3 router lives as SignalRouter.route(); pipeline imports route_signals().
    """
    return SignalRouter(config=config).route(signals)


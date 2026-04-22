"""
chad/risk/stop_bus_triggers.py

Phase-8 Session 3 (R2): expanded STOP-bus triggers.

Existing CHAD STOP machinery is equity-threshold driven (profit_lock.py)
and daily-notional throttled (daily_throttle.py). This module adds four
orthogonal halt triggers that the audit survey flagged as missing:

  1. daily_loss_breach    — realized PnL below a configured daily floor.
  2. reject_rate_spike    — order rejects dominate recent attempts.
  3. data_staleness       — Session 2 A4 gate is rejecting most intents
                            (data feed almost certainly down).
  4. broker_latency_spike — broker submission latency above threshold.

Each trigger is a pure function: inputs in, bool out. The aggregator
evaluate_all_stop_triggers() accepts a snapshot dict and returns a list
of active trigger names with reasons — never raises. Callers (the live
loop or an ops publisher) decide how to act on the result: publish to
runtime/stop_bus.json, set a Redis flag, or log and continue.

No global state. No I/O. Composable by design so the existing
profit_lock and daily_throttle modules remain authoritative for the
halt decision while these triggers contribute additional signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_REJECT_RATE_THRESHOLD: float = 0.30
DEFAULT_REJECT_RATE_MIN_SAMPLES: int = 10

DEFAULT_DATA_STALENESS_THRESHOLD: float = 0.80
DEFAULT_DATA_STALENESS_MIN_SAMPLES: int = 5

DEFAULT_BROKER_LATENCY_THRESHOLD_MS: float = 2000.0


# ---------------------------------------------------------------------------
# Trigger result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StopTriggerResult:
    name: str
    active: bool
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual triggers
# ---------------------------------------------------------------------------


def check_daily_loss_limit(
    realized_pnl: float,
    daily_loss_limit: float,
) -> StopTriggerResult:
    """Return active=True when realized PnL breaches the daily floor.

    daily_loss_limit is taken as a magnitude; callers may pass it as a
    positive number (e.g., 5000 means 'halt if we lose more than $5k').
    """
    try:
        pnl = float(realized_pnl)
    except (TypeError, ValueError):
        return StopTriggerResult("daily_loss", False, "invalid_pnl")
    try:
        floor = -abs(float(daily_loss_limit))
    except (TypeError, ValueError):
        return StopTriggerResult("daily_loss", False, "invalid_limit")

    active = pnl < floor
    reason = (
        f"realized_pnl={pnl:.2f} < floor={floor:.2f}" if active else "ok"
    )
    return StopTriggerResult(
        "daily_loss",
        active,
        reason,
        {"realized_pnl": pnl, "daily_loss_limit": -floor},
    )


def check_reject_rate(
    reject_count: int,
    total_count: int,
    threshold: float = DEFAULT_REJECT_RATE_THRESHOLD,
    min_samples: int = DEFAULT_REJECT_RATE_MIN_SAMPLES,
) -> StopTriggerResult:
    """Return active=True when reject rate in the recent window exceeds threshold."""
    try:
        rej = int(reject_count)
        total = int(total_count)
    except (TypeError, ValueError):
        return StopTriggerResult("reject_rate", False, "invalid_counts")

    if total < int(min_samples):
        return StopTriggerResult(
            "reject_rate",
            False,
            f"insufficient_samples:{total}<{min_samples}",
            {"reject_count": rej, "total_count": total},
        )

    rate = rej / total if total > 0 else 0.0
    active = rate > float(threshold)
    reason = (
        f"reject_rate={rate:.3f}>threshold={threshold:.3f}"
        if active
        else "ok"
    )
    return StopTriggerResult(
        "reject_rate",
        active,
        reason,
        {"reject_count": rej, "total_count": total, "rate": rate},
    )


def check_data_staleness(
    gate_reject_count: int,
    total_intent_count: int,
    threshold: float = DEFAULT_DATA_STALENESS_THRESHOLD,
    min_samples: int = DEFAULT_DATA_STALENESS_MIN_SAMPLES,
) -> StopTriggerResult:
    """Return active=True when A4 gate rejects dominate recent intents.

    If the data_freshness gate is knocking out the vast majority of
    intents the upstream bar feed is almost certainly down.
    """
    try:
        rej = int(gate_reject_count)
        total = int(total_intent_count)
    except (TypeError, ValueError):
        return StopTriggerResult("data_staleness", False, "invalid_counts")

    if total < int(min_samples):
        return StopTriggerResult(
            "data_staleness",
            False,
            f"insufficient_samples:{total}<{min_samples}",
            {"gate_reject_count": rej, "total_intent_count": total},
        )

    rate = rej / total if total > 0 else 0.0
    active = rate > float(threshold)
    reason = (
        f"stale_reject_rate={rate:.3f}>threshold={threshold:.3f}"
        if active
        else "ok"
    )
    return StopTriggerResult(
        "data_staleness",
        active,
        reason,
        {"gate_reject_count": rej, "total_intent_count": total, "rate": rate},
    )


def check_broker_latency(
    avg_latency_ms: float,
    threshold_ms: float = DEFAULT_BROKER_LATENCY_THRESHOLD_MS,
) -> StopTriggerResult:
    """Return active=True when avg broker submission latency exceeds threshold."""
    try:
        lat = float(avg_latency_ms)
    except (TypeError, ValueError):
        return StopTriggerResult("broker_latency", False, "invalid_latency")
    try:
        thr = float(threshold_ms)
    except (TypeError, ValueError):
        return StopTriggerResult("broker_latency", False, "invalid_threshold")

    active = lat > thr
    reason = f"avg_latency_ms={lat:.1f}>threshold={thr:.1f}" if active else "ok"
    return StopTriggerResult(
        "broker_latency",
        active,
        reason,
        {"avg_latency_ms": lat, "threshold_ms": thr},
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def evaluate_all_stop_triggers(
    snapshot: Optional[Mapping[str, Any]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all 4 triggers against a snapshot and return a result summary.

    snapshot keys (all optional — missing keys → that trigger is skipped):
      - realized_pnl:        float
      - daily_loss_limit:    float  (magnitude; halt if pnl < -|limit|)
      - reject_count:        int    (orders rejected in last window)
      - total_order_count:   int    (orders attempted in same window)
      - gate_reject_count:   int    (A4 gate rejections in last window)
      - total_intent_count:  int    (intents evaluated in same window)
      - avg_latency_ms:      float

    config keys override trigger thresholds (see module-level DEFAULTs).

    Returns:
      {
        "any_active": bool,
        "active_triggers": [StopTriggerResult, ...],
        "all_results": [StopTriggerResult, ...],
        "reasons": ["<name>:<reason>", ...],
      }

    This function never raises — unexpected snapshot values yield
    inactive triggers with an 'invalid_*' reason so the caller can see
    the mis-wire without the STOP bus crashing the live loop.
    """
    snap = dict(snapshot or {})
    cfg = dict(config or {})

    all_results: List[StopTriggerResult] = []

    if "realized_pnl" in snap and "daily_loss_limit" in snap:
        all_results.append(
            check_daily_loss_limit(
                snap["realized_pnl"], snap["daily_loss_limit"]
            )
        )
    if "reject_count" in snap and "total_order_count" in snap:
        all_results.append(
            check_reject_rate(
                snap["reject_count"],
                snap["total_order_count"],
                threshold=float(
                    cfg.get("reject_rate_threshold", DEFAULT_REJECT_RATE_THRESHOLD)
                ),
                min_samples=int(
                    cfg.get("reject_rate_min_samples", DEFAULT_REJECT_RATE_MIN_SAMPLES)
                ),
            )
        )
    if "gate_reject_count" in snap and "total_intent_count" in snap:
        all_results.append(
            check_data_staleness(
                snap["gate_reject_count"],
                snap["total_intent_count"],
                threshold=float(
                    cfg.get("data_staleness_threshold", DEFAULT_DATA_STALENESS_THRESHOLD)
                ),
                min_samples=int(
                    cfg.get("data_staleness_min_samples", DEFAULT_DATA_STALENESS_MIN_SAMPLES)
                ),
            )
        )
    if "avg_latency_ms" in snap:
        all_results.append(
            check_broker_latency(
                snap["avg_latency_ms"],
                threshold_ms=float(
                    cfg.get("broker_latency_threshold_ms", DEFAULT_BROKER_LATENCY_THRESHOLD_MS)
                ),
            )
        )

    active = [r for r in all_results if r.active]
    return {
        "any_active": bool(active),
        "active_triggers": active,
        "all_results": all_results,
        "reasons": [f"{r.name}:{r.reason}" for r in active],
    }


__all__ = [
    "DEFAULT_REJECT_RATE_THRESHOLD",
    "DEFAULT_REJECT_RATE_MIN_SAMPLES",
    "DEFAULT_DATA_STALENESS_THRESHOLD",
    "DEFAULT_DATA_STALENESS_MIN_SAMPLES",
    "DEFAULT_BROKER_LATENCY_THRESHOLD_MS",
    "StopTriggerResult",
    "check_daily_loss_limit",
    "check_reject_rate",
    "check_data_staleness",
    "check_broker_latency",
    "evaluate_all_stop_triggers",
]

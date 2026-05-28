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
from datetime import datetime, timezone
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

# --- Fix A: symmetric hysteresis on the broker_latency TRIP side ----------
# The auto-clear (release) side in chad/risk/stop_bus_state.py already demands
# a SUSTAINED clean streak before it lets the bus go:
#     DEFAULT_AUTO_CLEAR_CONSECUTIVE_CLEAN_REQUIRED = 5   (count gate)
#     DEFAULT_AUTO_CLEAR_MIN_CLEAN_SECONDS          = 240 (wall-clock gate)
# The trip side historically fired on a SINGLE cycle above threshold, so a
# one-cycle latency spike could halt trading for ~one auto-clear window. These
# constants give the trip side the same count + wall-clock discipline. The
# count gate is kept symmetric with the clear side (5 == 5); the wall-clock
# floor is intentionally shorter (60s vs the clear side's 240s) so a genuine
# sustained breach still halts promptly.
DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED: int = 5
DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS: float = 60.0
DEFAULT_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED: bool = True

# Env-var threading note (Fix A Phase D): these knobs are consumed via the
# ``config`` mapping passed to evaluate_all_stop_triggers (cfg keys:
# broker_latency_trip_consecutive_required, broker_latency_trip_min_breach_seconds,
# broker_latency_trip_hysteresis_enabled). Today chad/core/live_loop.py calls
# stop_bus_state.evaluate_and_persist WITHOUT a config dict, so the module
# defaults above apply. To make these operator-tunable, assemble a config dict
# where the snapshot is built (live_loop._build_stop_bus_snapshot already reads
# CHAD_* env vars) and read CHAD_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED (int),
# CHAD_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS (float),
# CHAD_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED (bool: 1/true/yes). No new config
# layer is introduced in this commit.


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


def _check_time_gate(
    last_above_threshold_at: Optional[str],
    min_seconds: float,
) -> bool:
    """Return True iff ``last_above_threshold_at`` is at least ``min_seconds``
    of wall-clock in the past.

    Mirrors the elapsed-time discipline in
    stop_bus_state._maybe_auto_clear_on_clean_streak (which compares
    ``now - streak_start`` against ``auto_clear_min_clean_seconds``). Fails
    closed: an empty, missing, or unparseable timestamp returns False so a
    transient spike or a malformed value can never satisfy the trip's time
    gate. Accepts ISO-8601 with a trailing 'Z' or an explicit offset; a naive
    timestamp is assumed UTC.
    """
    if not last_above_threshold_at:
        return False
    raw = str(last_above_threshold_at).strip()
    if not raw:
        return False
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        elapsed = (datetime.now(timezone.utc) - parsed).total_seconds()
    except (TypeError, ValueError):
        return False
    return elapsed >= float(min_seconds)


def check_broker_latency(
    avg_latency_ms: float,
    threshold_ms: float = DEFAULT_BROKER_LATENCY_THRESHOLD_MS,
    *,
    consecutive_cycles_above: Optional[int] = None,
    last_above_threshold_at: Optional[str] = None,
    trip_consecutive_required: int = DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED,
    trip_min_breach_seconds: float = DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS,
    hysteresis_enabled: bool = DEFAULT_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED,
) -> StopTriggerResult:
    """Return active=True when avg broker submission latency breaches threshold.

    Legacy single-cycle behaviour (active = latency > threshold) is preserved
    whenever hysteresis is disabled OR the caller does not supply the
    ``consecutive_cycles_above`` counter — this keeps un-migrated callers and
    existing tests intact.

    When hysteresis is enabled AND the counter is supplied, the trigger trips
    ONLY on a SUSTAINED breach: both a count gate (``consecutive_cycles_above
    >= trip_consecutive_required``) and a wall-clock gate
    (``now - last_above_threshold_at >= trip_min_breach_seconds``) must pass.
    This is the symmetric counterpart of the auto-clear hysteresis on the
    release side (stop_bus_state._maybe_auto_clear_on_clean_streak).
    """
    try:
        lat = float(avg_latency_ms)
    except (TypeError, ValueError):
        return StopTriggerResult("broker_latency", False, "invalid_latency")
    try:
        thr = float(threshold_ms)
    except (TypeError, ValueError):
        return StopTriggerResult("broker_latency", False, "invalid_threshold")
    if thr <= 0:
        return StopTriggerResult("broker_latency", False, "invalid_threshold")

    above_now = lat > thr

    # Backward-compat path: legacy single-cycle behaviour when hysteresis is
    # disabled OR the caller did not pass the consecutive-cycle counter.
    if not hysteresis_enabled or consecutive_cycles_above is None:
        active = above_now
        reason = f"avg_latency_ms={lat:.1f}>threshold={thr:.1f}" if above_now else "ok"
        return StopTriggerResult(
            "broker_latency",
            active,
            reason,
            {"avg_latency_ms": lat, "threshold_ms": thr},
        )

    # Hysteresis path: trip ONLY when BOTH the count gate and the wall-clock
    # time gate pass. Either gate failing keeps the trigger inactive.
    try:
        counter = int(consecutive_cycles_above)
    except (TypeError, ValueError):
        counter = 0
    try:
        required = int(trip_consecutive_required)
    except (TypeError, ValueError):
        required = DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED
    try:
        min_seconds = float(trip_min_breach_seconds)
    except (TypeError, ValueError):
        min_seconds = DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS

    count_gate = above_now and counter >= required
    time_gate = above_now and _check_time_gate(last_above_threshold_at, min_seconds)
    active = count_gate and time_gate

    if active:
        reason = (
            f"sustained_latency:avg_latency_ms={lat:.1f}>threshold={thr:.1f}"
            f" consecutive={counter}/{required}"
        )
    elif not above_now:
        reason = "ok"
    else:
        reason = (
            f"transient:avg_latency_ms={lat:.1f}>threshold={thr:.1f}"
            f" consecutive={counter}/{required}_below_trip_streak"
        )
    return StopTriggerResult(
        "broker_latency",
        active,
        reason,
        {
            "avg_latency_ms": lat,
            "threshold_ms": thr,
            "consecutive_cycles_above": counter,
            "trip_consecutive_required": required,
        },
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
                consecutive_cycles_above=snap.get("consecutive_cycles_above_stop_threshold"),
                last_above_threshold_at=snap.get("last_above_threshold_at"),
                trip_consecutive_required=int(
                    cfg.get(
                        "broker_latency_trip_consecutive_required",
                        DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED,
                    )
                ),
                trip_min_breach_seconds=float(
                    cfg.get(
                        "broker_latency_trip_min_breach_seconds",
                        DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS,
                    )
                ),
                hysteresis_enabled=bool(
                    cfg.get(
                        "broker_latency_trip_hysteresis_enabled",
                        DEFAULT_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED,
                    )
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
    "DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED",
    "DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS",
    "DEFAULT_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED",
    "StopTriggerResult",
    "check_daily_loss_limit",
    "check_reject_rate",
    "check_data_staleness",
    "check_broker_latency",
    "evaluate_all_stop_triggers",
]

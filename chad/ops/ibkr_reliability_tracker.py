"""IBKR reliability tracker (STOP-BUS-RECOVERY-1).

Observability-only helper that augments runtime/ibkr_status.json with:
  - consecutive_cycles_above_stop_threshold: int
  - last_above_threshold_at: ISO-8601 UTC or None  (MOST RECENT above-threshold
    cycle — re-stamped every above cycle; kept for backward compatibility)
  - breach_streak_started_at: ISO-8601 UTC or ""  (WHEN THE CURRENT BREACH STREAK
    BEGAN — stamped once at counter 0->1, preserved across the streak, cleared on
    recovery; the correct anchor for stop_bus's broker_latency time gate. Mirrors
    stop_bus_state auto-clear's last_clean_metric_ts. Backward-compatible additive
    field — schema_version intentionally not bumped.)
  - max_latency_observed_in_window: float (ms)
  - current_recovery_state: "healthy" / "degrading" / "above_threshold" / "recovering"
  - last_gateway_churn_at: ISO-8601 UTC or None  (detected when client_id changes
    between consecutive cycles or when 'gateway_churn=true' is asserted upstream)

This module does NOT change any recovery behaviour. Its goal is to surface
the sustained-latency pattern that operators currently have to infer from
journal scanning, and to provide an alerting hook (``should_alert()``) for
the existing chad-feed-watchdog / chad-health-monitor pipelines.

Behaviour-change for recovery logic is deliberately out of scope here
(documented in ops/pending_actions/IBKR_AUTO_RECOVERY_design_2026-05-27.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STOP_THRESHOLD_MS = 2000.0  # mirrors DEFAULT_BROKER_LATENCY_THRESHOLD_MS in stop_bus_triggers
DEFAULT_WINDOW_SECONDS = 1800       # 30 min sliding window for max_latency
DEFAULT_ALERT_AT_CONSECUTIVE = 5    # raise alert after 5 consecutive cycles above threshold

RECOVERY_HEALTHY = "healthy"
RECOVERY_DEGRADING = "degrading"        # one cycle above threshold
RECOVERY_ABOVE = "above_threshold"      # two or more consecutive cycles above threshold
RECOVERY_RECOVERING = "recovering"      # currently below but counter still positive (window resets to 0 on next OK cycle)


@dataclass
class ReliabilityFields:
    consecutive_cycles_above_stop_threshold: int
    last_above_threshold_at: str | None
    max_latency_observed_in_window: float
    current_recovery_state: str
    last_gateway_churn_at: str | None
    # Fix A activation: anchor for the stop_bus broker_latency time gate.
    # NOTE the deliberate asymmetry with last_above_threshold_at:
    #   - last_above_threshold_at   = the MOST RECENT above-threshold cycle
    #                                 (re-stamped every above cycle; kept for
    #                                 backward-compatible downstream consumers).
    #   - breach_streak_started_at  = WHEN THE CURRENT BREACH STREAK BEGAN
    #                                 (stamped once when counter goes 0->1,
    #                                 preserved across the streak, cleared on
    #                                 recovery). This mirrors the auto-clear
    #                                 last_clean_metric_ts pattern and is the
    #                                 correct anchor for "breach age >= N s".
    breach_streak_started_at: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "consecutive_cycles_above_stop_threshold": int(
                self.consecutive_cycles_above_stop_threshold
            ),
            "last_above_threshold_at": self.last_above_threshold_at,
            "max_latency_observed_in_window": float(self.max_latency_observed_in_window),
            "current_recovery_state": self.current_recovery_state,
            "last_gateway_churn_at": self.last_gateway_churn_at,
            "breach_streak_started_at": self.breach_streak_started_at,
        }


def _parse_ts(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_prev_status(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def compute_reliability_fields(
    prev_payload: dict[str, Any],
    current_payload: dict[str, Any],
    *,
    stop_threshold_ms: float = DEFAULT_STOP_THRESHOLD_MS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    now: datetime | None = None,
) -> ReliabilityFields:
    """Pure function — no I/O.

    Reads previous and current status payloads, emits the new observability
    fields without mutating either dict.
    """
    now = now or datetime.now(timezone.utc)
    current_lat = _safe_float(current_payload.get("latency_ms"))
    above = current_lat is not None and current_lat > stop_threshold_ms

    prev_counter = int(prev_payload.get("consecutive_cycles_above_stop_threshold") or 0)
    prev_last_above = prev_payload.get("last_above_threshold_at")
    prev_breach_streak = prev_payload.get("breach_streak_started_at") or ""
    prev_max_in_window = _safe_float(prev_payload.get("max_latency_observed_in_window")) or 0.0
    prev_max_ts = _parse_ts(prev_payload.get("ts_utc"))
    prev_client_id = prev_payload.get("client_id")
    current_client_id = current_payload.get("client_id")
    prev_churn_at = prev_payload.get("last_gateway_churn_at")
    explicit_churn = bool(current_payload.get("gateway_churn"))

    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if above:
        counter = prev_counter + 1
        last_above_at = now_iso
        # breach_streak_started_at: stamp ONCE when the breach streak begins
        # (counter 0->1), then preserve it through subsequent above cycles.
        # This mirrors stop_bus_state._maybe_auto_clear_on_clean_streak's
        # last_clean_metric_ts (stamped once at streak start, preserved).
        if prev_counter == 0:
            breach_streak_started_at = now_iso
        else:
            breach_streak_started_at = prev_breach_streak or now_iso
    else:
        counter = 0
        last_above_at = prev_last_above
        breach_streak_started_at = ""  # streak ended — clear the anchor

    # max_latency_observed_in_window: keep the larger of (prev within window) and current.
    if prev_max_ts is not None and (now - prev_max_ts).total_seconds() <= window_seconds:
        max_in_window = prev_max_in_window
    else:
        # Prior max is outside the window; reset.
        max_in_window = 0.0
    if current_lat is not None and current_lat > max_in_window:
        max_in_window = current_lat

    # gateway churn: client_id flip is the only locally-observable signal.
    churn_now = explicit_churn or (
        prev_client_id is not None
        and current_client_id is not None
        and prev_client_id != current_client_id
    )
    if churn_now:
        last_churn = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        last_churn = prev_churn_at

    if above and counter >= 2:
        state = RECOVERY_ABOVE
    elif above and counter == 1:
        state = RECOVERY_DEGRADING
    elif (not above) and prev_counter > 0:
        state = RECOVERY_RECOVERING
    else:
        state = RECOVERY_HEALTHY

    return ReliabilityFields(
        consecutive_cycles_above_stop_threshold=counter,
        last_above_threshold_at=last_above_at,
        max_latency_observed_in_window=max_in_window,
        current_recovery_state=state,
        last_gateway_churn_at=last_churn,
        breach_streak_started_at=breach_streak_started_at,
    )


def merge_reliability_into_payload(
    payload: dict[str, Any],
    prev_status_path: Path,
    *,
    stop_threshold_ms: float = DEFAULT_STOP_THRESHOLD_MS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read the previous ibkr_status.json from disk, compute the new fields,
    return ``payload`` augmented with them. Does NOT write to disk — the caller
    is responsible for persistence.
    """
    prev = _read_prev_status(prev_status_path)
    fields = compute_reliability_fields(
        prev,
        payload,
        stop_threshold_ms=stop_threshold_ms,
        window_seconds=window_seconds,
        now=now,
    )
    out = dict(payload)
    out.update(fields.as_payload())
    return out


def should_alert(
    payload: dict[str, Any],
    *,
    alert_at: int = DEFAULT_ALERT_AT_CONSECUTIVE,
) -> tuple[bool, dict[str, Any]]:
    """Return (alert_now, alert_payload).

    Fires exactly once when the counter crosses ``alert_at`` (i.e. transitions
    from below to equal-or-above). Callers should compare with prior state to
    avoid duplicate alerts; this function reports the current crossing.
    """
    counter = int(payload.get("consecutive_cycles_above_stop_threshold") or 0)
    if counter < alert_at:
        return False, {}
    alert = {
        "rule": "ibkr_reliability.sustained_latency_above_threshold",
        "consecutive_cycles_above_stop_threshold": counter,
        "stop_threshold_ms": DEFAULT_STOP_THRESHOLD_MS,
        "alert_at_consecutive": alert_at,
        "max_latency_observed_in_window": payload.get("max_latency_observed_in_window"),
        "current_recovery_state": payload.get("current_recovery_state"),
        "last_above_threshold_at": payload.get("last_above_threshold_at"),
        "last_gateway_churn_at": payload.get("last_gateway_churn_at"),
        "ts_utc": payload.get("ts_utc"),
    }
    return True, alert

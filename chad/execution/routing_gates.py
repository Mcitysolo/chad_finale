"""
routing_gates.py — Pre-execution intent validation gates.

All gates return (passed: bool, reason: str).
A failed gate produces a GATE_REJECT log entry and stops
the intent from reaching the OMS.

Gates (applied in order by run_all_gates):
  1. data_freshness_gate (A4) — bar data not stale
  2. stale_intent_gate (E2) — intent not expired
  3. too_late_to_chase_gate (E5) — price not moved away
  4. net_ev_gate (R7) — expected edge positive after costs
  5. event_risk_gate (S5) — inside a macro-catalyst suppression window

All gates are backward compatible: if optional fields on older intent
objects are missing, the gate degrades to a pass (with a reason string
that flags the degradation) rather than crashing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from chad.execution.intent_schema import DEFAULT_TTL_SECONDS

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reason codes (short, machine-readable)
# ---------------------------------------------------------------------------

REASON_OK = "ok"
REASON_BAR_STALE = "bar_stale"
REASON_BAR_MISSING = "bar_timestamp_missing"
REASON_INTENT_EXPIRED = "intent_expired"
REASON_INTENT_CREATED_AT_MISSING = "created_at_missing"
REASON_PRICE_MOVED = "price_moved_beyond_tolerance"
REASON_CURRENT_PRICE_MISSING = "current_price_missing"
REASON_INTENT_PRICE_MISSING = "intent_price_missing"
REASON_NEGATIVE_EV = "net_ev_below_min_edge"
REASON_MISSING_EXPECTED_PNL = "expected_pnl_missing"
REASON_EVENT_RISK_REJECT = "event_risk_reject"
REASON_EVENT_RISK_REDUCE = "event_risk_reduce_50pct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string to a timezone-aware UTC datetime, or None."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_intent_price(intent: Any) -> Optional[float]:
    """
    Best-effort intent price extraction.

    Order of preference:
      1. explicit limit_price (IBKR) / price (Kraken) when non-null and positive
      2. notional_estimate / quantity  (IBKR)
      3. notional_estimate / volume    (Kraken)
    """
    for attr in ("limit_price", "price", "price_at_creation"):
        p = getattr(intent, attr, None)
        try:
            if p is not None and float(p) > 0:
                return float(p)
        except (TypeError, ValueError):
            continue

    notional = getattr(intent, "notional_estimate", None)
    try:
        notional_f = float(notional) if notional is not None else 0.0
    except (TypeError, ValueError):
        notional_f = 0.0
    if notional_f <= 0:
        return None

    for attr in ("quantity", "volume"):
        q = getattr(intent, attr, None)
        try:
            q_f = float(q) if q is not None else 0.0
        except (TypeError, ValueError):
            q_f = 0.0
        if q_f > 0:
            return notional_f / q_f

    return None


def _intent_label(intent: Any) -> Tuple[str, str]:
    """Return (symbol-or-pair, strategy) — used for GATE_REJECT logs."""
    symbol = getattr(intent, "symbol", None) or getattr(intent, "pair", None) or "?"
    strategy = getattr(intent, "strategy", "?") or "?"
    return str(symbol), str(strategy)


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def data_freshness_gate(
    intent: Any,
    bar_timestamp: Optional[datetime] = None,
    max_bar_age_seconds: int = 300,
) -> Tuple[bool, str]:
    """A4 — reject if bar feeding the signal is older than max_bar_age_seconds.

    Graceful degradation: if no bar_timestamp is supplied, the gate passes
    (older intent sources that do not emit bar_timestamp continue to work).
    """
    if bar_timestamp is None:
        return True, REASON_BAR_MISSING
    bar_dt = _parse_iso_utc(bar_timestamp)
    if bar_dt is None:
        return True, REASON_BAR_MISSING
    age = (_utcnow() - bar_dt).total_seconds()
    if age > float(max_bar_age_seconds):
        return False, f"{REASON_BAR_STALE}:age={age:.1f}s>max={max_bar_age_seconds}s"
    return True, REASON_OK


def stale_intent_gate(
    intent: Any,
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """E2 — reject if (utcnow - created_at) > ttl_seconds.

    Uses created_at and ttl_seconds fields added in Session 1.
    Graceful degradation: missing/unparseable created_at passes.
    """
    created_at_raw = getattr(intent, "created_at", None)
    created_dt = _parse_iso_utc(created_at_raw)
    if created_dt is None:
        return True, REASON_INTENT_CREATED_AT_MISSING

    try:
        ttl = int(getattr(intent, "ttl_seconds", DEFAULT_TTL_SECONDS) or DEFAULT_TTL_SECONDS)
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_SECONDS

    now_dt = now if now is not None else _utcnow()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)

    age = (now_dt - created_dt).total_seconds()
    if age >= ttl:
        return False, f"{REASON_INTENT_EXPIRED}:age={age:.1f}s>=ttl={ttl}s"
    return True, REASON_OK


def too_late_to_chase_gate(
    intent: Any,
    current_price: Optional[float] = None,
    price_tolerance_pct: float = 0.005,
    degraded_ttl_seconds: int = 60,
) -> Tuple[bool, str]:
    """E5 — reject if price has moved beyond price_tolerance_pct since creation.

    If current_price is available: gate compares |current - intent_price| /
    intent_price against price_tolerance_pct.

    If current_price is NOT available (or intent has no derivable price):
    gate degrades to a tight time-based check — reject if intent age exceeds
    degraded_ttl_seconds (default 60s). Rationale: with no drift signal, the
    safest proxy for 'too late to chase' is 'too much time has passed'.
    """
    intent_price = _extract_intent_price(intent)

    if current_price is not None and intent_price is not None:
        try:
            cp = float(current_price)
        except (TypeError, ValueError):
            cp = None
        if cp is not None and intent_price > 0:
            tol_abs = abs(intent_price) * float(price_tolerance_pct)
            drift = abs(cp - intent_price)
            if drift > tol_abs:
                return (
                    False,
                    f"{REASON_PRICE_MOVED}:drift={drift:.6f}>tol={tol_abs:.6f}",
                )
            return True, REASON_OK

    # Degraded path: no usable price comparison available.
    created_dt = _parse_iso_utc(getattr(intent, "created_at", None))
    if created_dt is None:
        # No time basis either — pass with flagged reason.
        return True, REASON_CURRENT_PRICE_MISSING

    age = (_utcnow() - created_dt).total_seconds()
    if age >= degraded_ttl_seconds:
        return (
            False,
            f"{REASON_PRICE_MOVED}:degraded_time_age={age:.1f}s>=ttl={degraded_ttl_seconds}s",
        )
    return True, REASON_CURRENT_PRICE_MISSING


def net_ev_gate(
    intent: Any,
    estimated_commission: float = 1.0,
    estimated_spread: float = 0.0,
    min_edge: float = 0.0,
) -> Tuple[bool, str]:
    """R7 — reject if expected_pnl - commission - spread < min_edge.

    Graceful degradation: if expected_pnl is missing or zero (older intents
    that do not populate it), the gate passes with a flagged reason —
    the session-1 default is 0.0 and many legitimate intents ship with that.
    """
    expected_pnl = getattr(intent, "expected_pnl", None)
    if expected_pnl is None:
        return True, REASON_MISSING_EXPECTED_PNL
    try:
        ev = float(expected_pnl)
    except (TypeError, ValueError):
        return True, REASON_MISSING_EXPECTED_PNL

    # If expected_pnl is exactly zero (defaulted), skip EV rejection — strategy
    # has not opted in to EV-based gating. Only actively-populated positive OR
    # negative expected_pnl should be scored.
    if ev == 0.0:
        return True, REASON_MISSING_EXPECTED_PNL

    try:
        commission = float(estimated_commission)
    except (TypeError, ValueError):
        commission = 0.0
    try:
        spread = float(estimated_spread)
    except (TypeError, ValueError):
        spread = 0.0
    try:
        floor = float(min_edge)
    except (TypeError, ValueError):
        floor = 0.0

    net_edge = ev - commission - spread
    if net_edge < floor:
        return (
            False,
            f"{REASON_NEGATIVE_EV}:net={net_edge:.6f}<min={floor:.6f}"
            f" (ev={ev:.6f} comm={commission:.6f} spread={spread:.6f})",
        )
    return True, REASON_OK


def event_risk_gate(
    intent: Any,
    calendar: Any = None,
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """S5 — suppress entries near scheduled macro catalysts.

    ``calendar`` is an optional EventCalendar instance. When None, or
    when the intent has no symbol, the gate passes (backward compatible
    with pre-Session-7 callers that don't pass a calendar).

    Behavior inside an event window:

      * intent.order_urgency == "high"  → REJECT (return False)
      * intent.order_urgency == "normal" (or missing) → reduce intent
        size by 50% in place and PASS (return True)

    The 50% reduction is applied to both ``quantity`` (IBKR) and
    ``volume`` (Kraken) if present. Quantity floors at 1 unit so the
    intent still survives to the broker. Notional is scaled alongside.
    """
    if calendar is None:
        return True, REASON_OK
    symbol = getattr(intent, "symbol", None) or getattr(intent, "pair", None) or ""
    urgency = getattr(intent, "order_urgency", "normal") or "normal"

    try:
        verdict = calendar.get_suppression(symbol=symbol, urgency=urgency, now=now)
    except Exception as exc:  # noqa: BLE001
        # Calendar errors must not halt the pipeline — fail open.
        LOG.warning("event_risk_gate calendar error: %s", exc)
        return True, REASON_OK

    if not verdict.get("suppress"):
        return True, REASON_OK

    action = verdict.get("action")
    if action == "reject":
        return False, f"{REASON_EVENT_RISK_REJECT}:{verdict.get('reason', '')}"

    # reduce_50pct — halve quantity in place; quantity must floor at 1.
    for attr in ("quantity", "volume"):
        raw = getattr(intent, attr, None)
        if raw is None:
            continue
        try:
            current = float(raw)
        except (TypeError, ValueError):
            continue
        if current <= 0:
            continue
        new_val = current * 0.5
        if isinstance(raw, int):
            new_val = max(1, int(new_val))
        else:
            new_val = max(1e-8, new_val)
        try:
            object.__setattr__(intent, attr, new_val)
        except (AttributeError, TypeError):
            # Frozen dataclass — leave in place; we still pass with a
            # flagged reason so the operator sees that reduction was
            # requested but not applied.
            return True, f"{REASON_EVENT_RISK_REDUCE}:frozen"
    notional = getattr(intent, "notional_estimate", None)
    if notional is not None:
        try:
            scaled = float(notional) * 0.5
            object.__setattr__(intent, "notional_estimate", scaled)
        except (TypeError, AttributeError):
            pass

    return True, f"{REASON_EVENT_RISK_REDUCE}:{verdict.get('reason', '')}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


_GATE_ORDER = (
    "data_freshness",
    "stale_intent",
    "too_late_to_chase",
    "net_ev",
    "event_risk",
)


def run_all_gates(
    intent: Any,
    bar_timestamp: Optional[datetime] = None,
    current_price: Optional[float] = None,
    config: Optional[dict] = None,
    event_calendar: Any = None,
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Run all 5 gates in order. Return (True, "ok") if all pass, else
    (False, "<gate_name>:<reason>") on the first failure.

    On failure, emits a GATE_REJECT structured log entry with intent
    symbol and strategy for operational observability.

    Session 7 addition: Gate 5 is the event-risk suppression gate.
    Pass ``event_calendar=EventCalendar()`` to activate it; omitting
    the argument preserves pre-Session-7 4-gate behavior.
    """
    cfg = dict(config or {})
    max_bar_age_seconds = int(cfg.get("max_bar_age_seconds", 300))
    price_tolerance_pct = float(cfg.get("price_tolerance_pct", 0.005))
    degraded_ttl_seconds = int(cfg.get("degraded_ttl_seconds", 60))
    estimated_commission = float(cfg.get("estimated_commission", 1.0))
    estimated_spread = float(cfg.get("estimated_spread", 0.0))
    min_edge = float(cfg.get("min_edge", 0.0))

    gate_fns = (
        ("data_freshness", lambda: data_freshness_gate(intent, bar_timestamp, max_bar_age_seconds)),
        ("stale_intent", lambda: stale_intent_gate(intent)),
        (
            "too_late_to_chase",
            lambda: too_late_to_chase_gate(
                intent, current_price, price_tolerance_pct, degraded_ttl_seconds
            ),
        ),
        (
            "net_ev",
            lambda: net_ev_gate(intent, estimated_commission, estimated_spread, min_edge),
        ),
        (
            "event_risk",
            lambda: event_risk_gate(intent, event_calendar, now),
        ),
    )

    for gate_name, gate_call in gate_fns:
        try:
            passed, reason = gate_call()
        except Exception as exc:
            LOG.warning(
                "GATE_ERROR intent_symbol=%s strategy=%s gate=%s error=%s",
                *_intent_label(intent), gate_name, exc,
            )
            # Fail open on unexpected errors — don't break the pipeline.
            continue

        if not passed:
            symbol, strategy = _intent_label(intent)
            intent_id = getattr(intent, "idempotency_key", None) or getattr(
                intent, "trace_id", None
            ) or ""
            LOG.warning(
                "GATE_REJECT intent_id=%s gate=%s reason=%s symbol=%s strategy=%s",
                intent_id,
                gate_name,
                reason,
                symbol,
                strategy,
            )
            return False, f"{gate_name}:{reason}"

    return True, REASON_OK


__all__ = [
    "data_freshness_gate",
    "stale_intent_gate",
    "too_late_to_chase_gate",
    "net_ev_gate",
    "event_risk_gate",
    "run_all_gates",
    "REASON_OK",
    "REASON_BAR_STALE",
    "REASON_BAR_MISSING",
    "REASON_INTENT_EXPIRED",
    "REASON_INTENT_CREATED_AT_MISSING",
    "REASON_PRICE_MOVED",
    "REASON_CURRENT_PRICE_MISSING",
    "REASON_INTENT_PRICE_MISSING",
    "REASON_NEGATIVE_EV",
    "REASON_MISSING_EXPECTED_PNL",
    "REASON_EVENT_RISK_REJECT",
    "REASON_EVENT_RISK_REDUCE",
]

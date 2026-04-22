"""
chad/execution/intent_schema.py

Canonical intent schema shared by IBKR and Kraken StrategyTradeIntent dataclasses.

The Phase-8 survey (audit_m_build_survey_20260421) identified the intent
object as the foundational schema that unblocks 11 downstream items
(stale expiry, slippage tracking, too-late-to-chase, net EV gate,
confidence-based gating, per-setup expectancy, etc).

The six fields added here are OPTIONAL on both dataclasses so every
existing construction site continues to work without modification.
Strategies that have the relevant data populate the fields; the
remaining sites use the documented defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


DEFAULT_TTL_SECONDS = 300

INTENT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "confidence": {
        "type": float,
        "default": 0.5,
        "range": (0.0, 1.0),
        "description": "Composite signal quality: signal_strength * regime_quality * liquidity_quality",
    },
    "entry_reason": {
        "type": str,
        "default": "",
        "description": "Human-readable description of why the strategy wants this trade",
    },
    "regime_state": {
        "type": str,
        "default": "unknown",
        "description": "Regime label at intent creation (RISK_ON, RISK_OFF, CHOPPY, TRENDING, unknown)",
    },
    "expected_pnl": {
        "type": float,
        "default": 0.0,
        "description": "Gross edge estimate in quote currency; used by net-EV gate (R7)",
    },
    "created_at": {
        "type": str,
        "default": "",
        "description": "ISO8601 UTC timestamp of intent creation; used by stale-intent expiry (E2)",
    },
    "ttl_seconds": {
        "type": int,
        "default": DEFAULT_TTL_SECONDS,
        "description": "Intent validity window; submit layer drops intents older than created_at + ttl",
    },
    "signal_family": {
        "type": str,
        "default": "unknown",
        "description": "Signal family tag for S1 voting: momentum|mean_reversion|volatility|trend|options|macro|sentiment|unknown",
    },
    "order_urgency": {
        "type": str,
        "default": "normal",
        "description": "Passive vs aggressive hint for E4 order-type selector: 'normal' → passive LMT at mid; 'high' → marketable LMT through market",
    },
    "bar_timestamp": {
        "type": str,
        "default": "",
        "description": "ISO8601 UTC (or YYYY-MM-DD) timestamp of the bar that generated this signal. A4 data_freshness_gate validates its age. Empty string → gate degrades to None-passthrough.",
    },
}


def utc_now_iso() -> str:
    """Return current UTC time as ISO8601 string with microsecond precision."""
    return datetime.now(timezone.utc).isoformat()


def validate_intent(intent: Any) -> Tuple[bool, List[str]]:
    """
    Check that `intent` carries the six canonical fields with acceptable types.

    Returns (ok, errors). Uses getattr with defaults so plain dicts, dataclasses,
    and slotted classes all pass equivalently. Callers that need hard enforcement
    should raise on the returned errors; most callers should log and continue.
    """
    errors: List[str] = []
    for name, spec in INTENT_SCHEMA.items():
        value = getattr(intent, name, None)
        if value is None and isinstance(intent, dict):
            value = intent.get(name)
        expected = spec["type"]
        if value is None:
            errors.append(f"{name}: missing")
            continue
        if expected is float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{name}: expected float, got {type(value).__name__}")
                continue
        elif expected is int:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{name}: expected int, got {type(value).__name__}")
                continue
        elif expected is str:
            if not isinstance(value, str):
                errors.append(f"{name}: expected str, got {type(value).__name__}")
                continue
        if "range" in spec:
            lo, hi = spec["range"]
            if isinstance(value, (int, float)) and not (lo <= float(value) <= hi):
                errors.append(f"{name}: {value} outside range [{lo}, {hi}]")
    return (len(errors) == 0, errors)


def intent_is_fresh(intent: Any, now_iso: str = "") -> bool:
    """
    Return True if the intent's created_at + ttl_seconds has not elapsed.

    Falls back to True (treat as fresh) when created_at is missing or
    unparseable — downstream sessions are expected to tighten this once
    every creation site populates created_at via utc_now_iso().
    """
    created_at = getattr(intent, "created_at", "") or ""
    if not created_at:
        return True
    ttl = int(getattr(intent, "ttl_seconds", DEFAULT_TTL_SECONDS) or DEFAULT_TTL_SECONDS)
    try:
        created_dt = datetime.fromisoformat(created_at)
    except ValueError:
        return True
    now_dt = datetime.fromisoformat(now_iso) if now_iso else datetime.now(timezone.utc)
    if created_dt.tzinfo is None:
        created_dt = created_dt.replace(tzinfo=timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    return (now_dt - created_dt).total_seconds() <= ttl

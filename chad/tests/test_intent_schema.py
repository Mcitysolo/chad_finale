"""Tests for the canonical intent schema extension (Phase-8 Session 1)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from chad.execution.ibkr_executor import StrategyTradeIntent as IBKRIntent
from chad.execution.kraken_executor import StrategyTradeIntent as KrakenIntent
from chad.execution.intent_schema import (
    DEFAULT_TTL_SECONDS,
    INTENT_SCHEMA,
    intent_is_fresh,
    utc_now_iso,
    validate_intent,
)


CANONICAL_FIELDS = (
    "confidence",
    "entry_reason",
    "regime_state",
    "expected_pnl",
    "created_at",
    "ttl_seconds",
    # Phase-8 Session 5 (S1): signal_family added to the canonical schema.
    "signal_family",
)


def _ibkr_intent(**overrides):
    base = dict(
        strategy="alpha",
        symbol="SPY",
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="MKT",
        quantity=10.0,
        notional_estimate=5000.0,
    )
    base.update(overrides)
    return IBKRIntent(**base)


def _kraken_intent(**overrides):
    base = dict(
        strategy="alpha_crypto",
        pair="XBT/USD",
        side="buy",
        ordertype="market",
        volume=0.1,
        notional_estimate=5000.0,
    )
    base.update(overrides)
    return KrakenIntent(**base)


def test_intent_schema_has_required_fields():
    ibkr = _ibkr_intent()
    kraken = _kraken_intent()
    for name in CANONICAL_FIELDS:
        assert hasattr(ibkr, name), f"IBKR intent missing {name}"
        assert hasattr(kraken, name), f"Kraken intent missing {name}"


def test_intent_schema_field_types():
    ibkr = _ibkr_intent()
    assert isinstance(ibkr.confidence, float)
    assert isinstance(ibkr.entry_reason, str)
    assert isinstance(ibkr.regime_state, str)
    assert isinstance(ibkr.expected_pnl, float)
    assert isinstance(ibkr.created_at, str)
    assert isinstance(ibkr.ttl_seconds, int)


def test_intent_created_at_is_valid_iso8601():
    intent = _ibkr_intent()
    dt = datetime.fromisoformat(intent.created_at)
    # Must be within 5s of now and carry a tzinfo
    delta = datetime.now(timezone.utc) - dt
    assert delta.total_seconds() < 5
    assert dt.tzinfo is not None


def test_intent_ttl_default_is_300():
    assert DEFAULT_TTL_SECONDS == 300
    assert _ibkr_intent().ttl_seconds == 300
    assert _kraken_intent().ttl_seconds == 300


def test_intent_defaults_when_unspecified():
    intent = _ibkr_intent()
    assert intent.confidence == 0.5
    assert intent.entry_reason == ""
    assert intent.regime_state == "unknown"
    assert intent.expected_pnl == 0.0


def test_validate_intent_accepts_canonical_intent():
    ok, errors = validate_intent(_ibkr_intent())
    assert ok, errors


def test_validate_intent_rejects_confidence_out_of_range():
    intent = _ibkr_intent(confidence=1.5)
    ok, errors = validate_intent(intent)
    assert not ok
    assert any("confidence" in e for e in errors)


def test_intent_schema_constant_shape():
    # INTENT_SCHEMA must describe exactly the six canonical fields
    assert set(INTENT_SCHEMA.keys()) == set(CANONICAL_FIELDS)


def test_intent_is_fresh_returns_true_inside_ttl():
    intent = _ibkr_intent()
    assert intent_is_fresh(intent)


def test_intent_is_fresh_returns_false_past_ttl():
    past = (datetime.now(timezone.utc) - timedelta(seconds=1000)).isoformat()
    intent = _ibkr_intent(created_at=past, ttl_seconds=60)
    assert not intent_is_fresh(intent)


def test_intent_is_fresh_treats_empty_created_at_as_fresh():
    intent = _ibkr_intent(created_at="")
    assert intent_is_fresh(intent)


def test_intent_created_at_stable_within_call():
    # Two back-to-back constructions should produce monotonic created_at
    a = _ibkr_intent().created_at
    time.sleep(0.001)
    b = _ibkr_intent().created_at
    assert a <= b


def test_utc_now_iso_carries_tz_suffix():
    s = utc_now_iso()
    # ISO8601 UTC should carry +00:00 or Z
    assert s.endswith("+00:00") or s.endswith("Z")

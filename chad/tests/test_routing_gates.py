"""Tests for Phase-8 Session 2 routing gates (A4/E2/E5/R7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chad.execution.ibkr_executor import StrategyTradeIntent as IBKRIntent
from chad.execution.intent_schema import DEFAULT_TTL_SECONDS, utc_now_iso
from chad.execution.routing_gates import (
    data_freshness_gate,
    net_ev_gate,
    run_all_gates,
    stale_intent_gate,
    too_late_to_chase_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _intent(
    *,
    created_at: str | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    expected_pnl: float = 0.0,
    quantity: float = 10.0,
    notional_estimate: float = 1000.0,
    limit_price: float | None = None,
    symbol: str = "SPY",
    strategy: str = "alpha",
) -> IBKRIntent:
    return IBKRIntent(
        strategy=strategy,
        symbol=symbol,
        sec_type="STK",
        exchange="SMART",
        currency="USD",
        side="BUY",
        order_type="MKT",
        quantity=quantity,
        notional_estimate=notional_estimate,
        limit_price=limit_price,
        confidence=0.7,
        entry_reason="test",
        regime_state="RISK_ON",
        expected_pnl=expected_pnl,
        created_at=created_at if created_at is not None else utc_now_iso(),
        ttl_seconds=ttl_seconds,
    )


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# A4 — data_freshness_gate
# ---------------------------------------------------------------------------


def test_fresh_bar_passes():
    intent = _intent()
    fresh_bar = datetime.now(timezone.utc) - timedelta(seconds=30)
    passed, reason = data_freshness_gate(intent, fresh_bar, max_bar_age_seconds=300)
    assert passed is True
    assert reason == "ok"


def test_stale_bar_rejects():
    intent = _intent()
    stale_bar = datetime.now(timezone.utc) - timedelta(seconds=600)
    passed, reason = data_freshness_gate(intent, stale_bar, max_bar_age_seconds=300)
    assert passed is False
    assert "bar_stale" in reason


def test_none_bar_timestamp_passes():
    """Graceful degradation: no bar_timestamp supplied -> gate passes."""
    intent = _intent()
    passed, reason = data_freshness_gate(intent, None, max_bar_age_seconds=300)
    assert passed is True
    assert "bar_timestamp_missing" in reason


# ---------------------------------------------------------------------------
# E2 — stale_intent_gate
# ---------------------------------------------------------------------------


def test_fresh_intent_passes():
    intent = _intent(ttl_seconds=300)
    passed, reason = stale_intent_gate(intent)
    assert passed is True
    assert reason == "ok"


def test_expired_intent_rejects():
    old_ts = datetime.now(timezone.utc) - timedelta(seconds=1000)
    intent = _intent(created_at=_iso(old_ts), ttl_seconds=300)
    passed, reason = stale_intent_gate(intent)
    assert passed is False
    assert "intent_expired" in reason


def test_intent_at_exactly_ttl_rejects():
    """age == ttl_seconds -> reject (boundary: >= ttl)."""
    created = datetime.now(timezone.utc) - timedelta(seconds=60)
    intent = _intent(created_at=_iso(created), ttl_seconds=60)
    now = datetime.now(timezone.utc)
    passed, reason = stale_intent_gate(intent, now=now)
    assert passed is False
    assert "intent_expired" in reason


# ---------------------------------------------------------------------------
# E5 — too_late_to_chase_gate
# ---------------------------------------------------------------------------


def test_price_within_tolerance_passes():
    intent = _intent(limit_price=100.0)
    # Drift of 0.3% on a 100.0 price = 0.3 — below 0.5% tolerance.
    passed, reason = too_late_to_chase_gate(intent, current_price=100.3, price_tolerance_pct=0.005)
    assert passed is True
    assert reason == "ok"


def test_price_beyond_tolerance_rejects():
    intent = _intent(limit_price=100.0)
    # Drift of 1% on a 100.0 price = 1.0 — above 0.5% tolerance.
    passed, reason = too_late_to_chase_gate(intent, current_price=101.0, price_tolerance_pct=0.005)
    assert passed is False
    assert "price_moved_beyond_tolerance" in reason


def test_no_current_price_degrades_gracefully():
    """Without current_price, gate degrades to time-based check and still passes for fresh intents."""
    intent = _intent(limit_price=100.0)
    passed, reason = too_late_to_chase_gate(intent, current_price=None, price_tolerance_pct=0.005)
    assert passed is True
    # Degraded reason flag must surface.
    assert "current_price_missing" in reason


def test_no_current_price_degraded_time_rejects_old_intent():
    """Degraded path: very old intent with no current_price -> reject."""
    old_ts = datetime.now(timezone.utc) - timedelta(seconds=300)
    intent = _intent(limit_price=100.0, created_at=_iso(old_ts), ttl_seconds=DEFAULT_TTL_SECONDS)
    passed, reason = too_late_to_chase_gate(
        intent, current_price=None, price_tolerance_pct=0.005, degraded_ttl_seconds=60
    )
    assert passed is False
    assert "price_moved_beyond_tolerance" in reason
    assert "degraded_time_age" in reason


# ---------------------------------------------------------------------------
# R7 — net_ev_gate
# ---------------------------------------------------------------------------


def test_positive_edge_passes():
    intent = _intent(expected_pnl=10.0)
    passed, reason = net_ev_gate(intent, estimated_commission=1.0, estimated_spread=0.5, min_edge=0.0)
    assert passed is True
    assert reason == "ok"


def test_negative_edge_after_costs_rejects():
    # expected_pnl=1.0, commission=2.0 -> net=-1.0 < min_edge=0.0 -> reject
    intent = _intent(expected_pnl=1.0)
    passed, reason = net_ev_gate(intent, estimated_commission=2.0, estimated_spread=0.5, min_edge=0.0)
    assert passed is False
    assert "net_ev_below_min_edge" in reason


def test_zero_edge_with_zero_min_passes():
    """expected_pnl=0.0 (unset default) -> gate skips; treated as not opted-in."""
    intent = _intent(expected_pnl=0.0)
    passed, reason = net_ev_gate(intent, estimated_commission=0.0, estimated_spread=0.0, min_edge=0.0)
    assert passed is True
    assert "expected_pnl_missing" in reason


# ---------------------------------------------------------------------------
# run_all_gates
# ---------------------------------------------------------------------------


def test_first_failing_gate_stops_chain():
    """Order: data_freshness (A4) runs first. If it rejects, net_ev is never reached."""
    stale_bar = datetime.now(timezone.utc) - timedelta(seconds=900)
    # Use an intent with a negative EV too — but A4 should short-circuit.
    intent = _intent(expected_pnl=-100.0)
    passed, reason = run_all_gates(
        intent=intent,
        bar_timestamp=stale_bar,
        current_price=None,
        config={"max_bar_age_seconds": 300},
    )
    assert passed is False
    # Reason prefix identifies the gate that fired.
    assert reason.startswith("data_freshness:")


def test_all_gates_pass_returns_ok():
    intent = _intent(limit_price=100.0, expected_pnl=5.0, ttl_seconds=300)
    fresh_bar = datetime.now(timezone.utc) - timedelta(seconds=30)
    passed, reason = run_all_gates(
        intent=intent,
        bar_timestamp=fresh_bar,
        current_price=100.1,
        config={
            "max_bar_age_seconds": 300,
            "price_tolerance_pct": 0.005,
            "estimated_commission": 1.0,
            "estimated_spread": 0.5,
            "min_edge": 0.0,
        },
    )
    assert passed is True
    assert reason == "ok"


def test_run_all_gates_with_empty_config_uses_defaults():
    """Zero-config invocation must still apply gates with sensible defaults."""
    intent = _intent(limit_price=100.0, expected_pnl=5.0)
    passed, reason = run_all_gates(intent=intent)
    assert passed is True
    assert reason == "ok"


# ---------------------------------------------------------------------------
# Phase-8 Session 8: full A4/E5 threading — tests with real values
# ---------------------------------------------------------------------------


def test_e5_with_real_price_within_tolerance_passes():
    """E5 accepts the intent when current_price is within 0.5% of intent_price."""
    intent = _intent(limit_price=100.0)
    passed, reason = too_late_to_chase_gate(
        intent, current_price=100.40, price_tolerance_pct=0.005
    )
    assert passed is True
    assert reason == "ok"


def test_e5_with_real_price_beyond_tolerance_rejects():
    """E5 rejects the intent when current_price drifts beyond tolerance."""
    intent = _intent(limit_price=100.0)
    passed, reason = too_late_to_chase_gate(
        intent, current_price=101.0, price_tolerance_pct=0.005
    )
    assert passed is False
    assert "price_moved" in reason


def test_a4_with_real_bar_timestamp_stale_rejects():
    """A4 rejects when the bar predates max_bar_age_seconds."""
    intent = _intent()
    # Bar 10 minutes old; tolerance 5 minutes → reject.
    stale_bar_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    passed, reason = data_freshness_gate(
        intent, bar_timestamp=stale_bar_ts, max_bar_age_seconds=300
    )
    assert passed is False
    assert "bar_stale" in reason


def test_a4_with_real_bar_timestamp_fresh_passes():
    """A4 accepts when the bar is within the freshness window."""
    intent = _intent()
    fresh_bar_ts = datetime.now(timezone.utc) - timedelta(seconds=30)
    passed, reason = data_freshness_gate(
        intent, bar_timestamp=fresh_bar_ts, max_bar_age_seconds=300
    )
    assert passed is True
    assert reason == "ok"


def test_a4_with_daily_bar_in_48h_window_passes():
    """Real-world: 1d bars sampled yesterday still pass at 48h tolerance."""
    intent = _intent()
    yesterday = datetime.now(timezone.utc) - timedelta(hours=20)
    passed, reason = data_freshness_gate(
        intent, bar_timestamp=yesterday, max_bar_age_seconds=172800
    )
    assert passed is True


def test_run_all_gates_threading_end_to_end():
    """run_all_gates receives real bar_ts + current_price and passes cleanly."""
    intent = _intent(limit_price=100.0, expected_pnl=10.0)
    fresh_bar_ts = datetime.now(timezone.utc) - timedelta(seconds=5)
    passed, reason = run_all_gates(
        intent=intent,
        bar_timestamp=fresh_bar_ts,
        current_price=100.10,
        config={
            "max_bar_age_seconds": 300,
            "price_tolerance_pct": 0.005,
            "estimated_commission": 1.0,
            "min_edge": 0.0,
        },
    )
    assert passed is True
    assert reason == "ok"


# ---------------------------------------------------------------------------
# Phase-8 Session 8: E4 Kraken order-type selector — basic coverage
# ---------------------------------------------------------------------------


def test_e4_kraken_passive_order_params():
    """Kraken builder emits ordertype='limit' with passive pricing on normal urgency."""
    from chad.execution.execution_pipeline import _build_kraken_intent_from_routed_signal
    from chad.types import AssetClass, SignalSide
    from dataclasses import dataclass

    @dataclass
    class _RS:
        symbol: str = "BTC-USD"
        side: object = SignalSide.BUY
        net_size: float = 0.01
        asset_class: object = AssetClass.CRYPTO
        notional: float = 800.0
        order_urgency: str = "normal"
        signal_strength: float = 0.0
        regime_state: str = "unknown"
        confidence: float = 0.5
        expected_pnl: float = 0.0
        reason: str = ""
        source_strategies: tuple = ()

    intent = _build_kraken_intent_from_routed_signal(_RS(), current_price=80000.0)
    assert intent is not None
    assert intent.ordertype == "limit"
    # Passive: price at reference (no 10bps offset).
    assert intent.price == pytest.approx(80000.0, rel=1e-9)


def test_e4_kraken_aggressive_order_params():
    """Kraken builder emits aggressive limit (through market) on high urgency."""
    from chad.execution.execution_pipeline import _build_kraken_intent_from_routed_signal
    from chad.types import AssetClass, SignalSide
    from dataclasses import dataclass

    @dataclass
    class _RS:
        symbol: str = "BTC-USD"
        side: object = SignalSide.BUY
        net_size: float = 0.01
        asset_class: object = AssetClass.CRYPTO
        notional: float = 800.0
        order_urgency: str = "high"
        signal_strength: float = 0.0
        regime_state: str = "unknown"
        confidence: float = 0.5
        expected_pnl: float = 0.0
        reason: str = ""
        source_strategies: tuple = ()

    intent = _build_kraken_intent_from_routed_signal(_RS(), current_price=80000.0)
    assert intent is not None
    assert intent.ordertype == "limit"
    # Aggressive BUY: 10bps above reference → 80080.
    assert intent.price > 80000.0
    assert intent.price == pytest.approx(80080.0, rel=1e-3)

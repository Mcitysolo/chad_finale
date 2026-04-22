"""Tests for Phase-8 Session 6 (E4) order_type_selector."""

from __future__ import annotations

import pytest

from chad.execution.order_type_selector import (
    AGGRESSIVE_PRICE_OFFSET_PCT,
    DEFAULT_SPREAD_PCT,
    DEFAULT_SPREAD_THRESHOLD_PCT,
    ORDER_TYPE_LIMIT,
    compute_aggressive_limit_price,
    estimate_spread_pct,
    select_order_type,
)


def test_normal_urgency_tight_spread_passive():
    res = select_order_type(urgency="normal", estimated_spread_pct=0.0005)
    assert res["order_type"] == ORDER_TYPE_LIMIT
    assert res["aggressive"] is False
    assert res["price_offset_pct"] == 0.0
    assert "passive" in res["reason"]


def test_high_urgency_aggressive():
    res = select_order_type(urgency="high", estimated_spread_pct=0.0005)
    assert res["order_type"] == ORDER_TYPE_LIMIT
    assert res["aggressive"] is True
    assert res["price_offset_pct"] == AGGRESSIVE_PRICE_OFFSET_PCT
    assert res["reason"] == "urgency_high"


def test_wide_spread_aggressive_regardless_urgency():
    res = select_order_type(urgency="normal", estimated_spread_pct=0.01)
    assert res["aggressive"] is True
    assert res["reason"] == "wide_spread"
    assert res["order_type"] == ORDER_TYPE_LIMIT


def test_never_returns_market_order():
    for urgency in ("normal", "high", "weird", "", None):
        for spread in (0.0, 0.0005, 0.01, 0.1, -1.0, 1e9):
            res = select_order_type(urgency=urgency, estimated_spread_pct=spread)
            assert res["order_type"] == "LMT"
            assert res["order_type"] != "MKT"


def test_unknown_urgency_treated_as_normal():
    res = select_order_type(urgency="catastrophic", estimated_spread_pct=0.0001)
    assert res["aggressive"] is False


def test_estimate_spread_uses_bar_range():
    bar = {"high": 110.0, "low": 100.0}
    # (10 / 105) * 0.1 ≈ 0.0095
    got = estimate_spread_pct("AAPL", bar_data=bar)
    assert 0.009 < got < 0.01


def test_estimate_spread_defaults_on_missing_bar():
    assert estimate_spread_pct("AAPL", bar_data=None) == DEFAULT_SPREAD_PCT


def test_estimate_spread_defaults_on_bad_bar():
    assert estimate_spread_pct("X", bar_data={}) == DEFAULT_SPREAD_PCT
    assert estimate_spread_pct("X", bar_data={"high": -1, "low": -2}) == DEFAULT_SPREAD_PCT


def test_compute_aggressive_limit_price_buy_pushes_up():
    px = compute_aggressive_limit_price("BUY", 100.0, price_offset_pct=0.001)
    assert px == pytest.approx(100.1, rel=1e-9)


def test_compute_aggressive_limit_price_sell_pushes_down():
    px = compute_aggressive_limit_price("SELL", 100.0, price_offset_pct=0.001)
    assert px == pytest.approx(99.9, rel=1e-9)


def test_compute_aggressive_limit_price_passthrough_on_zero_reference():
    assert compute_aggressive_limit_price("BUY", 0.0) == 0.0


def test_threshold_boundary_is_passive():
    # Equal to threshold → not wide, so passive when urgency is normal.
    res = select_order_type(
        urgency="normal",
        estimated_spread_pct=DEFAULT_SPREAD_THRESHOLD_PCT,
    )
    assert res["aggressive"] is False

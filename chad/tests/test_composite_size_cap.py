"""Tests for Phase-8 Session 7 (R5) CompositeSizeCap."""

from __future__ import annotations

import pytest

from chad.risk.composite_size_cap import CompositeSizeCap


def test_per_symbol_cap_applied():
    cap = CompositeSizeCap(max_per_symbol=100, max_sector_exposure=0)
    # vol_adjusted wants 500 shares but per-symbol cap is 100.
    assert cap.apply(vol_adjusted_size=500, symbol="AAPL") == 100


def test_sector_cap_applied():
    cap = CompositeSizeCap(
        max_per_symbol=10_000,
        max_sector_exposure=500,
    )
    # 400 shares already in sector — only 100 room left.
    result = cap.apply(
        vol_adjusted_size=300,
        symbol="AAPL",
        current_sector_exposure=400,
    )
    assert result == 100


def test_liquidity_cap_applied():
    cap = CompositeSizeCap(max_per_symbol=1_000_000, max_adv_pct=0.01)
    # ADV=500_000, 1% = 5000 shares.
    result = cap.apply(
        vol_adjusted_size=10_000,
        symbol="X",
        avg_daily_volume=500_000,
    )
    assert result == 5_000


def test_margin_cap_applied():
    cap = CompositeSizeCap(
        max_per_symbol=1_000_000,
        max_position_pct=0.10,
        avg_price_assumption=100.0,
    )
    # equity=$100k, 10% = $10k, at $100/share → 100 shares.
    result = cap.apply(
        vol_adjusted_size=10_000,
        symbol="X",
        account_equity=100_000.0,
    )
    assert result == 100


def test_margin_cap_uses_reference_price_when_provided():
    cap = CompositeSizeCap(
        max_per_symbol=1_000_000,
        max_position_pct=0.10,
        avg_price_assumption=100.0,
    )
    # $10k budget at real $200 price → 50 shares (not 100).
    result = cap.apply(
        vol_adjusted_size=10_000,
        symbol="X",
        account_equity=100_000.0,
        reference_price=200.0,
    )
    assert result == 50


def test_minimum_1_share_always():
    cap = CompositeSizeCap(max_per_symbol=1, max_sector_exposure=1)
    # Every cap wants 1 share.
    assert cap.apply(vol_adjusted_size=1_000_000, symbol="X") == 1


def test_zero_base_size_returns_zero():
    cap = CompositeSizeCap()
    assert cap.apply(vol_adjusted_size=0, symbol="X") == 0


def test_missing_liquidity_data_skipped():
    cap = CompositeSizeCap(max_per_symbol=500)
    # No adv, no equity → only per-symbol and (default) sector caps.
    assert cap.apply(vol_adjusted_size=1000, symbol="X") == 500


def test_from_config_reads_defaults(tmp_path):
    # Missing config path → defaults.
    cap = CompositeSizeCap.from_config(config_path=tmp_path / "nope.json")
    assert cap.max_per_symbol == 1000
    assert cap.max_sector_exposure == 5000


def test_sector_full_clamps_to_one():
    cap = CompositeSizeCap(max_per_symbol=10_000, max_sector_exposure=500)
    # Sector already at budget.
    assert cap.apply(
        vol_adjusted_size=100,
        symbol="X",
        current_sector_exposure=500,
    ) == 1

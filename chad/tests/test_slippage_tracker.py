"""Tests for Phase-8 Session 3 SlippageTracker (E3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.slippage_tracker import SlippageTracker


@pytest.fixture()
def tracker(tmp_path: Path) -> SlippageTracker:
    return SlippageTracker(ledger_dir=tmp_path / "slippage", rolling_window=50)


def test_record_fill_computes_buy_slippage_correctly(tracker: SlippageTracker):
    rec = tracker.record_fill(
        symbol="SPY",
        strategy="alpha",
        side="BUY",
        expected_price=400.00,
        fill_price=400.50,
        quantity=10,
        intent_id="i1",
    )
    # BUY: fill_price - expected_price = 0.50 (adverse = positive)
    assert rec["slippage_per_share"] == pytest.approx(0.50)
    assert rec["total_slippage"] == pytest.approx(5.0)
    assert rec["raw_price_diff"] == pytest.approx(0.50)
    assert rec["symbol"] == "SPY"


def test_record_fill_computes_sell_slippage_correctly(tracker: SlippageTracker):
    rec = tracker.record_fill(
        symbol="SPY",
        strategy="alpha",
        side="SELL",
        expected_price=400.00,
        fill_price=399.50,  # filled for less than expected (adverse for sell)
        quantity=10,
        intent_id="i2",
    )
    # SELL: expected_price - fill_price = 0.50 (adverse = positive)
    assert rec["slippage_per_share"] == pytest.approx(0.50)
    assert rec["total_slippage"] == pytest.approx(5.0)
    # Raw diff preserves sign (fill - expected).
    assert rec["raw_price_diff"] == pytest.approx(-0.50)


def test_rolling_stats_mean_correct_after_multiple_fills(tracker: SlippageTracker):
    # Three BUY fills with slippage 0.10, 0.30, 0.50 -> mean 0.30
    for fp in (400.10, 400.30, 400.50):
        tracker.record_fill(
            symbol="SPY",
            strategy="alpha",
            side="BUY",
            expected_price=400.00,
            fill_price=fp,
            quantity=10,
        )
    stats = tracker.get_rolling_stats(symbol="SPY", strategy="alpha")
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.30)
    assert stats["min"] == pytest.approx(0.10)
    assert stats["max"] == pytest.approx(0.50)


def test_slippage_ledger_file_written(tracker: SlippageTracker, tmp_path: Path):
    tracker.record_fill(
        symbol="MES",
        strategy="alpha_futures",
        side="BUY",
        expected_price=5000.0,
        fill_price=5000.25,
        quantity=1,
    )
    rows = tracker.read_ledger()
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "MES"
    assert row["strategy"] == "alpha_futures"
    assert row["slippage_per_share"] == pytest.approx(0.25)


def test_filter_by_symbol(tracker: SlippageTracker):
    tracker.record_fill(symbol="SPY", strategy="alpha", side="BUY",
                        expected_price=400.0, fill_price=400.10, quantity=1)
    tracker.record_fill(symbol="QQQ", strategy="alpha", side="BUY",
                        expected_price=300.0, fill_price=300.60, quantity=1)
    spy = tracker.get_rolling_stats(symbol="SPY")
    qqq = tracker.get_rolling_stats(symbol="QQQ")
    assert spy["n"] == 1
    assert qqq["n"] == 1
    assert spy["mean"] == pytest.approx(0.10)
    assert qqq["mean"] == pytest.approx(0.60)


def test_filter_by_strategy(tracker: SlippageTracker):
    tracker.record_fill(symbol="SPY", strategy="alpha", side="BUY",
                        expected_price=400.0, fill_price=400.20, quantity=1)
    tracker.record_fill(symbol="SPY", strategy="gamma", side="BUY",
                        expected_price=400.0, fill_price=400.80, quantity=1)
    alpha = tracker.get_rolling_stats(strategy="alpha")
    gamma = tracker.get_rolling_stats(strategy="gamma")
    assert alpha["n"] == 1
    assert gamma["n"] == 1
    assert alpha["mean"] == pytest.approx(0.20)
    assert gamma["mean"] == pytest.approx(0.80)


def test_missing_expected_price_does_not_crash_or_pollute_stats(
    tracker: SlippageTracker,
):
    """expected_price=0.0 -> record is written with slippage=None; stats unchanged."""
    rec = tracker.record_fill(
        symbol="SPY", strategy="alpha", side="BUY",
        expected_price=0.0, fill_price=400.5, quantity=1,
    )
    assert rec["slippage_per_share"] is None
    assert rec["total_slippage"] is None
    stats = tracker.get_rolling_stats(symbol="SPY", strategy="alpha")
    assert stats["n"] == 0


def test_rolling_window_bounded(tmp_path: Path):
    """The rolling window caps the number of samples retained."""
    t = SlippageTracker(ledger_dir=tmp_path / "s", rolling_window=3)
    for fp in (400.10, 400.20, 400.30, 400.40):
        t.record_fill(symbol="SPY", strategy="alpha", side="BUY",
                      expected_price=400.0, fill_price=fp, quantity=1)
    stats = t.get_rolling_stats(symbol="SPY", strategy="alpha")
    assert stats["n"] == 3
    # The first 0.10 entry should have been evicted; mean of 0.20, 0.30, 0.40 = 0.30
    assert stats["mean"] == pytest.approx(0.30)

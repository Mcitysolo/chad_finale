"""Tests for Phase-8 Session 7 (R6) CorrelationMonitor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.risk.correlation_monitor import CorrelationMonitor, _pearson_correlation


def _write_bars(bars_dir: Path, symbol: str, closes):
    bars_dir.mkdir(parents=True, exist_ok=True)
    from datetime import date, timedelta
    start = date(2025, 1, 1)
    bars = [
        {
            "ts_utc": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1_000_000,
        }
        for i, c in enumerate(closes)
    ]
    (bars_dir / f"{symbol}.json").write_text(json.dumps({"bars": bars}), encoding="utf-8")


def test_single_position_no_reduction(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    _write_bars(bars_dir, "A", [100.0 + i for i in range(40)])
    mon = CorrelationMonitor(threshold=0.7, bars_dir=bars_dir)
    # Empty open book.
    assert mon.get_size_multiplier([], "A") == 1.0
    # Single existing symbol equal to new → combined has 1 symbol.
    assert mon.get_size_multiplier(["A"], "A") == 1.0


def test_high_correlation_reduces_size(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # Two near-identical series → avg pairwise corr ≈ 1.0.
    ascending = [100.0 + i for i in range(40)]
    _write_bars(bars_dir, "A", ascending)
    _write_bars(bars_dir, "B", [c + 0.01 * i for i, c in enumerate(ascending)])
    _write_bars(bars_dir, "C", [c + 0.02 * i for i, c in enumerate(ascending)])

    mon = CorrelationMonitor(threshold=0.7, bars_dir=bars_dir, floor_multiplier=0.1)
    mult = mon.get_size_multiplier(["A", "B"], "C")
    # threshold/avg_corr ≤ 1.0 since avg_corr > threshold.
    assert mult < 1.0
    assert mult >= 0.1


def test_below_threshold_no_reduction(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # One ascending series, one descending — correlation ~ -1, |r| ≈ 1.
    # To keep below threshold, use two series where log-returns are near-zero
    # correlated: ascending vs random walk-ish series.
    import math
    _write_bars(bars_dir, "A", [100.0 * math.exp(0.001 * i) for i in range(40)])
    # Alternating small ups/downs (near-zero mean return), uncorrelated to A.
    series_b = []
    p = 100.0
    for i in range(40):
        p *= 1.003 if (i % 3 == 0) else 0.998 if (i % 3 == 1) else 1.0005
        series_b.append(p)
    _write_bars(bars_dir, "B", series_b)

    mon = CorrelationMonitor(threshold=0.99, bars_dir=bars_dir)
    mult = mon.get_size_multiplier(["A"], "B")
    # High threshold → always 1.0 regardless of actual corr.
    assert mult == 1.0


def test_missing_bars_returns_1x(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()  # empty
    mon = CorrelationMonitor(threshold=0.7, bars_dir=bars_dir)
    assert mon.get_size_multiplier(["A", "B"], "C") == 1.0


def test_floor_multiplier_not_exceeded(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # Perfectly identical series → avg corr = 1.0, threshold/corr = 0.1.
    closes = [100.0 + i for i in range(40)]
    _write_bars(bars_dir, "A", closes)
    _write_bars(bars_dir, "B", closes)
    _write_bars(bars_dir, "C", closes)

    mon = CorrelationMonitor(threshold=0.1, bars_dir=bars_dir, floor_multiplier=0.2)
    mult = mon.get_size_multiplier(["A", "B"], "C")
    # threshold/avg_corr = 0.1/1.0 = 0.1; clamped up to floor 0.2.
    assert mult == pytest.approx(0.2)


def test_pearson_correlation_constant_series_returns_none():
    assert _pearson_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None
    assert _pearson_correlation([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]) is None


def test_pearson_correlation_perfect_positive():
    r = _pearson_correlation([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert r == pytest.approx(1.0, abs=1e-9)


def test_deduped_open_symbols(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    closes = [100.0 + i for i in range(40)]
    _write_bars(bars_dir, "A", closes)
    mon = CorrelationMonitor(threshold=0.7, bars_dir=bars_dir)
    # Duplicate 'A' + 'A' still counts as 1 symbol → no reduction.
    assert mon.get_size_multiplier(["A", "A"], "A") == 1.0

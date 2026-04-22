"""Tests for Phase-8 Session 7 (R3) VolAdjustedSizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.risk.vol_adjusted_sizer import (
    DEFAULT_CEILING_MULTIPLIER,
    DEFAULT_FLOOR_MULTIPLIER,
    DEFAULT_TARGET_DAILY_VOL,
    VolAdjustedSizer,
    compute_realized_daily_vol,
)


def _write_bars(bars_dir: Path, symbol: str, closes):
    bars_dir.mkdir(parents=True, exist_ok=True)
    from datetime import date, timedelta
    start = date(2025, 1, 1)
    bars = [
        {
            "ts_utc": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
            "open": c,
            "high": c * 1.002,
            "low": c * 0.998,
            "close": c,
            "volume": 1_000_000,
        }
        for i, c in enumerate(closes)
    ]
    (bars_dir / f"{symbol}.json").write_text(json.dumps({"bars": bars}), encoding="utf-8")


def test_high_vol_reduces_size(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # 5% daily swings for 30 bars: realized_vol ≈ 0.05, target=0.01 → mult≈0.2.
    closes = []
    price = 100.0
    for i in range(30):
        price *= (1.05 if i % 2 == 0 else 0.95)
        closes.append(price)
    _write_bars(bars_dir, "HIVOL", closes)

    sizer = VolAdjustedSizer(target_daily_vol=0.01, bars_dir=bars_dir)
    mult = sizer.get_size_multiplier("HIVOL")
    assert mult < 1.0
    assert sizer.adjust(1000, "HIVOL") < 1000


def test_low_vol_increases_size_but_capped(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # 0.1% daily swings: realized_vol very small → raw multiplier huge.
    closes = []
    price = 100.0
    for i in range(30):
        price *= 1.001 if i % 2 == 0 else 0.999
        closes.append(price)
    _write_bars(bars_dir, "LOVOL", closes)

    sizer = VolAdjustedSizer(target_daily_vol=0.01, ceiling_multiplier=2.0, bars_dir=bars_dir)
    mult = sizer.get_size_multiplier("LOVOL")
    assert mult == pytest.approx(2.0, abs=1e-9)


def test_no_data_returns_1x(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    sizer = VolAdjustedSizer(bars_dir=bars_dir)
    assert sizer.get_size_multiplier("NOTAFILE") == 1.0
    assert sizer.adjust(500, "NOTAFILE") == 500


def test_floor_applied(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # Extremely noisy: realized vol » target.
    closes = []
    price = 100.0
    for i in range(30):
        price *= 1.2 if i % 2 == 0 else 0.8
        closes.append(price)
    _write_bars(bars_dir, "NOISY", closes)

    sizer = VolAdjustedSizer(
        target_daily_vol=0.01,
        floor_multiplier=0.1,
        ceiling_multiplier=2.0,
        bars_dir=bars_dir,
    )
    mult = sizer.get_size_multiplier("NOISY")
    assert mult == pytest.approx(0.1, abs=1e-9)


def test_ceiling_applied(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    # Flat series → tiny vol → ceiling engaged.
    closes = [100.0 + 0.001 * (i % 3) for i in range(30)]
    _write_bars(bars_dir, "FLAT", closes)
    sizer = VolAdjustedSizer(
        target_daily_vol=0.05,
        floor_multiplier=0.1,
        ceiling_multiplier=2.0,
        bars_dir=bars_dir,
    )
    mult = sizer.get_size_multiplier("FLAT")
    assert mult == pytest.approx(2.0, abs=1e-9)


def test_adjust_never_returns_zero_for_positive_base():
    sizer = VolAdjustedSizer()
    # Even with a worst-case 0.1× multiplier, adjust(1, ...) ≥ 1.
    assert sizer.adjust(1, "NOPE_NO_DATA") == 1


def test_compute_realized_daily_vol_insufficient_data_returns_zero():
    assert compute_realized_daily_vol([100.0, 101.0], lookback_days=20) == 0.0


def test_compute_realized_daily_vol_matches_pstdev():
    import math
    import statistics
    closes = [100.0 * (1.0 + 0.001 * ((-1) ** i)) for i in range(30)]
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, 22)]
    expected = statistics.pstdev(rets[-20:])
    got = compute_realized_daily_vol(closes, lookback_days=20)
    assert got == pytest.approx(expected, rel=1e-6)

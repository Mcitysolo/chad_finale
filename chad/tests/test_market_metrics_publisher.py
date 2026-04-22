"""Tests for Phase-8 Session 6 (G1 feed) market_metrics_publisher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.market_metrics_publisher import (
    MARKET_METRICS_PATH,
    SCHEMA_VERSION,
    MarketMetricsPublisher,
    compute_adx_proxy,
    compute_realized_vol_percentile,
    compute_trend_slope,
)


def _write_bars(bars_dir: Path, symbol: str, bars):
    bars_dir.mkdir(parents=True, exist_ok=True)
    (bars_dir / f"{symbol}.json").write_text(
        json.dumps({"bars": bars}), encoding="utf-8"
    )


def _ascending_closes(n, start=100.0, step=1.0):
    bars = []
    from datetime import date, timedelta
    base = date(2025, 1, 1)
    for i in range(n):
        c = start + i * step
        bars.append(
            {
                "ts_utc": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": c - step / 2,
                "high": c + step,
                "low": c - step,
                "close": c,
                "volume": 1_000_000,
            }
        )
    return bars


def test_vol_percentile_computation_in_range():
    closes = [100.0 + i for i in range(60)]
    pct = compute_realized_vol_percentile(closes, window=20, lookback=252)
    assert 0.0 <= pct <= 1.0


def test_vol_percentile_defaults_on_insufficient_data():
    assert compute_realized_vol_percentile([100.0, 101.0], window=20) == 0.5


def test_adx_proxy_computation_nonneg():
    bars = _ascending_closes(40)
    adx = compute_adx_proxy(bars, period=14)
    assert adx >= 0.0


def test_adx_proxy_zero_on_short_input():
    bars = _ascending_closes(3)
    assert compute_adx_proxy(bars, period=14) == 0.0


def test_trend_slope_positive_trending():
    closes = [100.0 + i for i in range(40)]
    slope = compute_trend_slope(closes, period=20)
    assert slope > 0.0


def test_trend_slope_negative_downtrending():
    closes = [200.0 - i for i in range(40)]
    slope = compute_trend_slope(closes, period=20)
    assert slope < 0.0


def test_trend_slope_zero_on_short_input():
    assert compute_trend_slope([100.0, 101.0], period=20) == 0.0


def test_insufficient_bars_returns_defaults(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()  # empty
    out = tmp_path / "market_metrics.json"
    publisher = MarketMetricsPublisher(bars_dir=bars_dir, output_path=out)
    payload = publisher.compute_and_publish()
    assert payload["symbols_covered"] == 0
    assert payload["realized_vol_percentile"] == 0.5
    assert payload["adx"] == 0.0
    assert payload["trend_slope"] == 0.0
    assert payload["notes"] == "no_bars_found"
    assert out.is_file()


def test_output_written_to_json(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    _write_bars(bars_dir, "AAPL", _ascending_closes(60))
    _write_bars(bars_dir, "MSFT", _ascending_closes(60, start=200.0, step=2.0))
    out = tmp_path / "market_metrics.json"
    publisher = MarketMetricsPublisher(bars_dir=bars_dir, output_path=out)
    payload = publisher.compute_and_publish()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["symbols_covered"] == 2
    data = json.loads(out.read_text())
    assert data["symbols_covered"] == 2
    assert "AAPL" in data["per_symbol"]
    assert "MSFT" in data["per_symbol"]
    # Aggregates are within valid ranges.
    assert 0.0 <= data["realized_vol_percentile"] <= 1.0
    assert -1.0 <= data["market_breadth"] <= 1.0


def test_breadth_positive_when_most_above_sma(tmp_path: Path):
    bars_dir = tmp_path / "bars"
    _write_bars(bars_dir, "A", _ascending_closes(40))
    _write_bars(bars_dir, "B", _ascending_closes(40, start=300.0, step=3.0))
    _write_bars(bars_dir, "C", _ascending_closes(40, start=50.0, step=0.5))
    out = tmp_path / "market_metrics.json"
    publisher = MarketMetricsPublisher(bars_dir=bars_dir, output_path=out)
    payload = publisher.compute_and_publish()
    # All three are above their SMA (ascending series) → breadth = +1.
    assert payload["market_breadth"] == pytest.approx(1.0, abs=1e-6)


def test_feeds_regime_classifier_into_real_label(tmp_path: Path):
    """End-to-end: publisher output feeds the classifier to a real label."""
    from chad.analytics.regime_classifier import classify_regime

    bars_dir = tmp_path / "bars"
    # 5 trending symbols → classifier should produce trending_bull when ADX > 25.
    for i, sym in enumerate(["A", "B", "C", "D", "E"]):
        _write_bars(bars_dir, sym, _ascending_closes(60, start=100.0 + i * 10))
    out = tmp_path / "market_metrics.json"
    publisher = MarketMetricsPublisher(bars_dir=bars_dir, output_path=out)
    payload = publisher.compute_and_publish()
    result = classify_regime(
        realized_vol_percentile=payload["realized_vol_percentile"],
        adx=payload["adx"],
        trend_slope=payload["trend_slope"],
        market_breadth=payload["market_breadth"],
    )
    # Not "unknown" — the classifier now has real inputs.
    assert result.regime != "unknown"

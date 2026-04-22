"""Tests for Phase-8 Session 6 (F2) signal_decay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.signal_decay import (
    DEFAULT_HORIZONS_DAYS,
    SignalDecayAnalyzer,
    SignalDecayRecorder,
)


def _write_bars(bars_dir: Path, symbol: str, closes):
    """Write synthetic daily bars that start on 2025-01-01 and advance by one day."""
    bars_dir.mkdir(parents=True, exist_ok=True)
    payload_bars = []
    from datetime import date, timedelta
    start = date(2025, 1, 1)
    for i, c in enumerate(closes):
        payload_bars.append(
            {
                "ts_utc": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "open": c,
                "high": c * 1.005,
                "low": c * 0.995,
                "close": c,
                "volume": 1_000_000,
            }
        )
    (bars_dir / f"{symbol}.json").write_text(
        json.dumps({"bars": payload_bars}), encoding="utf-8"
    )


def test_record_entry_creates_pending(tmp_path: Path):
    ledger = tmp_path / "signal_decay"
    bars = tmp_path / "bars"
    rec = SignalDecayRecorder(ledger_dir=ledger, bars_dir=bars)
    record = rec.record_entry(
        strategy="alpha",
        symbol="AAPL",
        side="BUY",
        entry_price=100.0,
        entry_time="2025-01-05",
        intent_id="i1",
    )
    assert record["measured"] is False
    assert record["entry_price"] == 100.0
    # File exists and contains one line.
    path = ledger / "alpha_decay.ndjson"
    assert path.is_file()
    assert len(path.read_text().splitlines()) == 1


def test_compute_decay_with_sufficient_bars(tmp_path: Path):
    ledger = tmp_path / "signal_decay"
    bars_dir = tmp_path / "bars"
    # 40 ascending closes so every horizon has data.
    _write_bars(bars_dir, "AAPL", [100.0 + i for i in range(40)])

    rec = SignalDecayRecorder(ledger_dir=ledger, bars_dir=bars_dir)
    rec.record_entry(
        strategy="alpha",
        symbol="AAPL",
        side="BUY",
        entry_price=104.0,       # entry @ index 4 (2025-01-05)
        entry_time="2025-01-05",
        intent_id="i1",
    )
    measured = rec.compute_decay_for_pending()
    assert len(measured) == 1
    m = measured[0]
    # At T+1 close is 105, alpha ≈ (105-104)/104 ~ 0.0096
    assert m["alpha"]["1"] == pytest.approx(1.0 / 104.0, rel=1e-6)
    # At T+30 close is 134, alpha = 30/104
    assert m["alpha"]["30"] == pytest.approx(30.0 / 104.0, rel=1e-6)
    assert m["measured"] is True


def test_decay_analyzer_mean_alpha(tmp_path: Path):
    ledger = tmp_path / "signal_decay"
    bars_dir = tmp_path / "bars"
    _write_bars(bars_dir, "SPY", [200.0 + i for i in range(40)])

    rec = SignalDecayRecorder(ledger_dir=ledger, bars_dir=bars_dir)
    # Two BUYs with different entry points.
    rec.record_entry("alpha", "SPY", "BUY", 200.0, "2025-01-01", "i1")
    rec.record_entry("alpha", "SPY", "BUY", 203.0, "2025-01-04", "i2")
    rec.compute_decay_for_pending()

    analyzer = SignalDecayAnalyzer(ledger_dir=ledger)
    stats = analyzer.get_decay_stats("alpha")
    assert stats["sample_count"] == 2
    # Mean alpha should be positive (all-ascending series, BUY trades).
    for h in DEFAULT_HORIZONS_DAYS:
        key = f"mean_alpha_t{h}"
        assert stats[key] is not None
        assert stats[key] > 0.0


def test_missing_bars_skips_gracefully(tmp_path: Path):
    ledger = tmp_path / "signal_decay"
    bars_dir = tmp_path / "bars"  # empty
    rec = SignalDecayRecorder(ledger_dir=ledger, bars_dir=bars_dir)
    rec.record_entry("alpha", "NOBARS", "BUY", 100.0, "2025-01-05", "i1")
    measured = rec.compute_decay_for_pending()
    assert measured == []
    # Record is still pending.
    path = ledger / "alpha_decay.ndjson"
    line = json.loads(path.read_text().splitlines()[0])
    assert line["measured"] is False


def test_sell_side_alpha_is_positive_when_price_falls(tmp_path: Path):
    ledger = tmp_path / "signal_decay"
    bars_dir = tmp_path / "bars"
    # Descending closes so a SELL position is winning.
    _write_bars(bars_dir, "QQQ", [400.0 - i for i in range(40)])
    rec = SignalDecayRecorder(ledger_dir=ledger, bars_dir=bars_dir)
    rec.record_entry("beta", "QQQ", "SELL", 400.0, "2025-01-01", "i1")
    measured = rec.compute_decay_for_pending()
    assert measured
    m = measured[0]
    for h in DEFAULT_HORIZONS_DAYS:
        assert m["alpha"][str(h)] > 0.0


def test_partial_horizon_keeps_pending(tmp_path: Path):
    # Only 10 bars — T+15 and T+30 are not yet computable.
    ledger = tmp_path / "signal_decay"
    bars_dir = tmp_path / "bars"
    _write_bars(bars_dir, "AAPL", [100.0 + i for i in range(10)])
    rec = SignalDecayRecorder(ledger_dir=ledger, bars_dir=bars_dir)
    rec.record_entry("alpha", "AAPL", "BUY", 100.0, "2025-01-01", "i1")
    measured = rec.compute_decay_for_pending()
    # Partial measurement — record stays pending.
    assert measured == []
    path = ledger / "alpha_decay.ndjson"
    line = json.loads(path.read_text().splitlines()[0])
    assert line["measured"] is False
    # But T+1 and T+5 alphas are populated.
    assert line["alpha"]["1"] is not None
    assert line["alpha"]["5"] is not None

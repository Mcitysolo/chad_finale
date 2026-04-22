"""Tests for Phase-8 Session 5 timeframe_confirmation (S2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.signal_confidence import compute_confidence
from chad.analytics.timeframe_confirmation import (
    TF_AGREE_MULT,
    TF_DISAGREE_MULT,
    TF_MIN_MULT,
    TF_NEUTRAL_MULT,
    get_higher_tf_bias,
    timeframe_confidence_multiplier,
)


# ---------------------------------------------------------------------------
# multiplier pure-function behavior
# ---------------------------------------------------------------------------


def test_bullish_buy_returns_1_0():
    assert timeframe_confidence_multiplier("BUY", "bullish") == pytest.approx(TF_AGREE_MULT)


def test_bearish_sell_returns_1_0():
    assert timeframe_confidence_multiplier("SELL", "bearish") == pytest.approx(TF_AGREE_MULT)


def test_bearish_buy_returns_0_6():
    assert timeframe_confidence_multiplier("BUY", "bearish") == pytest.approx(TF_DISAGREE_MULT)


def test_bullish_sell_returns_0_6():
    assert timeframe_confidence_multiplier("SELL", "bullish") == pytest.approx(TF_DISAGREE_MULT)


def test_neutral_returns_0_85():
    assert timeframe_confidence_multiplier("BUY", "neutral") == pytest.approx(TF_NEUTRAL_MULT)
    assert timeframe_confidence_multiplier("SELL", "neutral") == pytest.approx(TF_NEUTRAL_MULT)


def test_lowercase_side_accepted():
    # Kraken uses lowercase sides.
    assert timeframe_confidence_multiplier("buy", "bullish") == pytest.approx(TF_AGREE_MULT)


def test_invalid_bias_degrades_to_neutral():
    assert timeframe_confidence_multiplier("BUY", "garbage") == pytest.approx(TF_NEUTRAL_MULT)


def test_multiplier_never_below_min():
    # Even in the worst case the result is >= TF_MIN_MULT.
    v = timeframe_confidence_multiplier("BUY", "bearish")
    assert v >= TF_MIN_MULT


# ---------------------------------------------------------------------------
# bias extraction from bar files
# ---------------------------------------------------------------------------


def _write_bars(path: Path, closes: list[float]) -> None:
    bars = [{"close": c, "ts_utc": f"2026-01-{i + 1:02d}"} for i, c in enumerate(closes)]
    path.write_text(json.dumps({"bars": bars}), encoding="utf-8")


def test_get_higher_tf_bias_bullish(tmp_path: Path):
    bars_dir = tmp_path
    # Rising closes — last close is above the SMA.
    _write_bars(bars_dir / "SPY.json", [100.0] * 10 + [110.0] * 5 + [140.0])
    assert get_higher_tf_bias("SPY", bars_dir=bars_dir) == "bullish"


def test_get_higher_tf_bias_bearish(tmp_path: Path):
    bars_dir = tmp_path
    _write_bars(bars_dir / "SPY.json", [140.0] * 10 + [130.0] * 5 + [100.0])
    assert get_higher_tf_bias("SPY", bars_dir=bars_dir) == "bearish"


def test_missing_bars_returns_neutral(tmp_path: Path):
    assert get_higher_tf_bias("NOPE", bars_dir=tmp_path) == "neutral"


def test_malformed_bars_file_returns_neutral(tmp_path: Path):
    (tmp_path / "SPY.json").write_text("{not valid json", encoding="utf-8")
    assert get_higher_tf_bias("SPY", bars_dir=tmp_path) == "neutral"


# ---------------------------------------------------------------------------
# compute_confidence integration
# ---------------------------------------------------------------------------


def test_full_confidence_with_tf_agree():
    c = compute_confidence(0.8, 1.0, 1.0, tf_multiplier=1.0)
    assert c == pytest.approx(0.8)


def test_confidence_attenuated_on_disagree():
    agree = compute_confidence(0.8, 1.0, 1.0, tf_multiplier=1.0)
    disagree = compute_confidence(0.8, 1.0, 1.0, tf_multiplier=0.6)
    assert disagree < agree
    assert disagree == pytest.approx(0.48)

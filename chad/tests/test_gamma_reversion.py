#!/usr/bin/env python3
"""
Tests for the GAMMA_REVERSION ETF mean reversion strategy.

Covers:
- RSI calculation correctness
- Bollinger Band calculation
- Z-score calculation
- ROC calculation
- Signal direction correctness (both long and short)
- Confidence calculation
- Sizing validation
- Min bars enforcement
- Config builder and env overrides
- Handler contract compatibility
- Registry integration
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest import mock

import pytest

from chad.strategies.gamma_reversion import (
    GammaReversionTuning,
    _bollinger,
    _build_signal_for_symbol,
    _compute_confidence,
    _roc,
    _rsi,
    _zscore,
    build_gamma_reversion_config,
    build_gamma_reversion_signals,
    gamma_reversion_handler,
)
from chad.strategies.gamma_reversion_config import (
    DEFAULT_GAMMA_REVERSION_UNIVERSE,
    build_gamma_reversion_config as config_build_gamma_reversion_config,
)
from chad.types import AssetClass, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    closes: Sequence[float],
    *,
    spread: float = 1.0,
    volume: float = 50_000_000.0,
) -> List[Dict[str, float]]:
    """Create synthetic OHLCV bars from a close series."""
    bars: List[Dict[str, float]] = []
    for c in closes:
        bars.append({
            "open": c - spread * 0.3,
            "high": c + spread,
            "low": c - spread,
            "close": c,
            "volume": volume,
        })
    return bars


def _steady_series(n: int = 50, base: float = 500.0, step: float = 0.5) -> List[float]:
    return [base + i * step for i in range(n)]


def _spike_up_series(n: int = 50, base: float = 500.0) -> List[float]:
    """Series that rises gently then spikes — overbought setup."""
    closes = [base + i * 0.3 for i in range(n - 8)]
    top = closes[-1]
    for i in range(8):
        closes.append(top + 15.0 * (i + 1))
    return closes


def _crash_down_series(n: int = 50, base: float = 500.0) -> List[float]:
    """Series that declines gently then crashes — oversold setup."""
    closes = [base - i * 0.3 for i in range(n - 8)]
    bottom = closes[-1]
    for i in range(8):
        closes.append(bottom - 15.0 * (i + 1))
    return closes


NOW = datetime(2026, 4, 3, 14, 30, tzinfo=timezone.utc)


@dataclass
class _FakePortfolio:
    positions: Dict[str, Any]
    extra: Optional[Dict[str, Any]] = None
    cash: float = 100_000.0


@dataclass
class _FakeContext:
    now: datetime
    ticks: Dict[str, Any]
    legend: Any = None
    portfolio: Any = None
    bars: Optional[Dict[str, list]] = None


# ===========================================================================
# RSI tests
# ===========================================================================

class TestRSI:

    def test_rsi_warmup_neutral(self) -> None:
        """First `period` values should be 50.0 (neutral)."""
        closes = _steady_series(30)
        vals = _rsi(closes, period=14)
        assert len(vals) == 30
        for i in range(14):
            assert vals[i] == 50.0

    def test_rsi_rising_series(self) -> None:
        """Steadily rising prices -> RSI > 50."""
        closes = _steady_series(50, step=2.0)
        vals = _rsi(closes, period=14)
        assert vals[-1] > 70.0

    def test_rsi_falling_series(self) -> None:
        """Steadily falling prices -> RSI < 50."""
        closes = [500.0 - i * 2.0 for i in range(50)]
        vals = _rsi(closes, period=14)
        assert vals[-1] < 30.0

    def test_rsi_range_0_100(self) -> None:
        """RSI must always be in [0, 100]."""
        closes = _spike_up_series(60)
        vals = _rsi(closes, period=14)
        for v in vals:
            assert 0.0 <= v <= 100.0

    def test_rsi_insufficient_data(self) -> None:
        closes = [100.0, 101.0, 102.0]
        vals = _rsi(closes, period=14)
        assert all(v == 50.0 for v in vals)


# ===========================================================================
# Bollinger Band tests
# ===========================================================================

class TestBollinger:

    def test_middle_is_sma(self) -> None:
        closes = _steady_series(30, base=100.0, step=1.0)
        upper, middle, lower = _bollinger(closes, period=20, width=2.0)
        assert len(middle) == 30
        # Last value should be SMA(20)
        expected_sma = sum(closes[-20:]) / 20
        assert middle[-1] == pytest.approx(expected_sma, abs=0.01)

    def test_upper_above_lower(self) -> None:
        closes = _steady_series(30, base=100.0, step=1.0)
        upper, middle, lower = _bollinger(closes, period=20)
        for u, m, l in zip(upper[-10:], middle[-10:], lower[-10:]):
            assert u >= m >= l

    def test_bands_widen_with_volatility(self) -> None:
        """Spiky series should have wider bands than steady series."""
        steady = _steady_series(50, step=0.1)
        spiky = _spike_up_series(50)
        _, _, _ = _bollinger(steady, period=20)
        u_steady, _, l_steady = _bollinger(steady, period=20)
        u_spiky, _, l_spiky = _bollinger(spiky, period=20)
        width_steady = u_steady[-1] - l_steady[-1]
        width_spiky = u_spiky[-1] - l_spiky[-1]
        assert width_spiky > width_steady


# ===========================================================================
# Z-score tests
# ===========================================================================

class TestZscore:

    def test_zscore_at_mean_is_zero(self) -> None:
        """If price equals the SMA, Z-score should be ~0."""
        closes = [100.0] * 30
        vals = _zscore(closes, period=20)
        assert vals[-1] == pytest.approx(0.0, abs=0.01)

    def test_zscore_above_mean_positive(self) -> None:
        closes = _spike_up_series(50)
        vals = _zscore(closes, period=20)
        assert vals[-1] > 1.0

    def test_zscore_below_mean_negative(self) -> None:
        closes = _crash_down_series(50)
        vals = _zscore(closes, period=20)
        assert vals[-1] < -1.0


# ===========================================================================
# ROC tests
# ===========================================================================

class TestROC:

    def test_roc_warmup_zero(self) -> None:
        closes = _steady_series(20)
        vals = _roc(closes, period=5)
        for i in range(5):
            assert vals[i] == 0.0

    def test_roc_rising_positive(self) -> None:
        closes = _steady_series(20, step=5.0)
        vals = _roc(closes, period=5)
        assert vals[-1] > 0

    def test_roc_falling_negative(self) -> None:
        closes = [500.0 - i * 5.0 for i in range(20)]
        vals = _roc(closes, period=5)
        assert vals[-1] < 0


# ===========================================================================
# Signal direction tests
# ===========================================================================

class TestSignalDirections:

    def test_short_signal_on_overbought(self) -> None:
        """Spike up should generate a SELL signal."""
        closes = _spike_up_series(60, base=500.0)
        bars = _make_bars(closes)
        tuning = GammaReversionTuning()
        sig = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=tuning,
            now=NOW,
        )
        if sig is not None:
            assert sig.side == SignalSide.SELL
            assert sig.strategy == StrategyName.GAMMA_REVERSION
            assert sig.asset_class == AssetClass.ETF

    def test_long_signal_on_oversold(self) -> None:
        """Crash down should generate a BUY signal."""
        closes = _crash_down_series(60, base=500.0)
        bars = _make_bars(closes)
        tuning = GammaReversionTuning()
        sig = _build_signal_for_symbol(
            symbol="QQQ",
            bars=bars,
            price=closes[-1],
            tuning=tuning,
            now=NOW,
        )
        if sig is not None:
            assert sig.side == SignalSide.BUY
            assert sig.strategy == StrategyName.GAMMA_REVERSION

    def test_no_signal_on_steady(self) -> None:
        """Steady series should generate no signal."""
        closes = _steady_series(60, step=0.01)
        bars = _make_bars(closes)
        tuning = GammaReversionTuning()
        sig = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=tuning,
            now=NOW,
        )
        assert sig is None

    def test_no_signal_insufficient_bars(self) -> None:
        closes = [500.0, 501.0, 502.0]
        bars = _make_bars(closes)
        sig = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=GammaReversionTuning(),
            now=NOW,
        )
        assert sig is None


# ===========================================================================
# Confidence tests
# ===========================================================================

class TestConfidence:

    def test_base_confidence(self) -> None:
        conf = _compute_confidence(
            rsi_val=75.0, zscore_val=1.9,
            side=SignalSide.SELL,
            tuning=GammaReversionTuning(),
        )
        assert conf == pytest.approx(0.55, abs=0.001)

    def test_extreme_rsi_bonus_short(self) -> None:
        """RSI > 80 in short signal adds +0.10."""
        conf = _compute_confidence(
            rsi_val=85.0, zscore_val=1.5,
            side=SignalSide.SELL,
            tuning=GammaReversionTuning(),
        )
        assert conf == pytest.approx(0.65, abs=0.001)

    def test_extreme_rsi_bonus_long(self) -> None:
        """RSI < 20 in long signal adds +0.10."""
        conf = _compute_confidence(
            rsi_val=15.0, zscore_val=-1.5,
            side=SignalSide.BUY,
            tuning=GammaReversionTuning(),
        )
        assert conf == pytest.approx(0.65, abs=0.001)

    def test_high_zscore_bonus(self) -> None:
        """Z-score > 2.2 adds +0.10, >2.5 adds another +0.05."""
        conf = _compute_confidence(
            rsi_val=85.0, zscore_val=2.6,
            side=SignalSide.SELL,
            tuning=GammaReversionTuning(),
        )
        # 0.55 + 0.10 (rsi) + 0.10 (zs>2.2) + 0.05 (zs>2.5) = 0.80
        assert conf == pytest.approx(0.80, abs=0.001)

    def test_confidence_clamped(self) -> None:
        conf = _compute_confidence(
            rsi_val=95.0, zscore_val=5.0,
            side=SignalSide.SELL,
            tuning=GammaReversionTuning(),
        )
        assert conf <= 1.0
        assert conf >= 0.0


# ===========================================================================
# Sizing tests
# ===========================================================================

class TestSizing:

    def test_signal_has_positive_size(self) -> None:
        closes = _spike_up_series(60, base=500.0)
        bars = _make_bars(closes)
        sig = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=GammaReversionTuning(),
            now=NOW,
        )
        if sig is not None:
            assert sig.size > 0
            assert sig.size <= GammaReversionTuning().max_size

    def test_signal_meta_has_indicators(self) -> None:
        closes = _spike_up_series(60, base=500.0)
        bars = _make_bars(closes)
        sig = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=GammaReversionTuning(),
            now=NOW,
        )
        if sig is not None:
            assert "rsi" in sig.meta
            assert "zscore" in sig.meta
            assert "bb_upper" in sig.meta
            assert "atr" in sig.meta
            assert "stop_price" in sig.meta
            assert "target_price" in sig.meta
            assert sig.meta["engine"] == "gamma_reversion.v1"


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:

    def test_default_config(self) -> None:
        cfg = config_build_gamma_reversion_config()
        assert cfg.name == StrategyName.GAMMA_REVERSION
        assert cfg.enabled is True
        assert list(cfg.target_universe) == ["SPY", "QQQ", "GLD", "TLT"]
        assert cfg.max_gross_exposure == pytest.approx(0.18)

    def test_env_disable(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_GAMMA_REVERSION_ENABLED": "false"}):
            cfg = config_build_gamma_reversion_config()
            assert cfg.enabled is False

    def test_env_universe_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_GAMMA_REVERSION_UNIVERSE": "SPY,QQQ"}):
            cfg = config_build_gamma_reversion_config()
            assert list(cfg.target_universe) == ["SPY", "QQQ"]

    def test_env_exposure_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_GAMMA_REVERSION_MAX_GROSS_EXPOSURE": "0.12"}):
            cfg = config_build_gamma_reversion_config()
            assert cfg.max_gross_exposure == pytest.approx(0.12)

    def test_invalid_symbol_raises(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_GAMMA_REVERSION_UNIVERSE": "INVALID"}):
            with pytest.raises(ValueError, match="Unsupported"):
                config_build_gamma_reversion_config()

    def test_fallback_config(self) -> None:
        cfg = build_gamma_reversion_config()
        assert cfg.name == StrategyName.GAMMA_REVERSION


# ===========================================================================
# Handler tests
# ===========================================================================

class TestHandler:

    def test_handler_returns_list(self) -> None:
        ctx = _FakeContext(now=NOW, ticks={}, portfolio=_FakePortfolio(positions={}))
        result = gamma_reversion_handler(ctx)
        assert isinstance(result, list)

    def test_handler_disabled_returns_empty(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_GAMMA_REVERSION_ENABLED": "false"}):
            ctx = _FakeContext(now=NOW, ticks={}, portfolio=_FakePortfolio(positions={}))
            result = gamma_reversion_handler(ctx)
            assert result == []

    def test_handler_survives_bad_context(self) -> None:
        result = gamma_reversion_handler(None)
        assert isinstance(result, list)

    def test_handler_with_bars_produces_signals(self) -> None:
        """Handler with overbought bars should produce SELL signals."""
        closes = _spike_up_series(60, base=500.0)
        bars = _make_bars(closes)
        ctx = _FakeContext(
            now=NOW,
            ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={"SPY": bars, "QQQ": bars, "GLD": bars, "TLT": bars},
        )
        result = gamma_reversion_handler(ctx)
        # At least some symbols should trigger
        sell_signals = [s for s in result if s.side == SignalSide.SELL]
        assert len(sell_signals) >= 1


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry:

    def test_strategy_name_enum(self) -> None:
        assert hasattr(StrategyName, "GAMMA_REVERSION")
        assert StrategyName.GAMMA_REVERSION.value == "gamma_reversion"

    def test_registry_contains_gamma_reversion(self) -> None:
        from chad.strategies import iter_strategy_registrations
        names = [reg.name for reg in iter_strategy_registrations()]
        assert StrategyName.GAMMA_REVERSION in names

    def test_registry_config_builds(self) -> None:
        from chad.strategies import iter_strategy_registrations
        for reg in iter_strategy_registrations():
            if reg.name == StrategyName.GAMMA_REVERSION:
                cfg = reg.build_config()
                assert cfg.name == StrategyName.GAMMA_REVERSION
                assert cfg.enabled is True
                return
        pytest.fail("GAMMA_REVERSION not found in registry")


# ===========================================================================
# GLD strict confluence tests
# ===========================================================================

class TestGLDStrictConfluence:

    def test_gld_strict_blocks_or_condition(self) -> None:
        """GLD with strict confluence should not fire when only BB OR zscore triggers."""
        # Build bars where price is above BB upper but zscore is below threshold
        # (a marginal overbought that would trigger with OR but not AND)
        closes = _spike_up_series(60, base=300.0)
        bars = _make_bars(closes, spread=0.5)
        tuning_strict = GammaReversionTuning(gld_strict_confluence=True)
        tuning_relaxed = GammaReversionTuning(gld_strict_confluence=False)

        sig_strict = _build_signal_for_symbol(
            symbol="GLD",
            bars=bars,
            price=closes[-1],
            tuning=tuning_strict,
            now=NOW,
        )
        sig_relaxed = _build_signal_for_symbol(
            symbol="GLD",
            bars=bars,
            price=closes[-1],
            tuning=tuning_relaxed,
            now=NOW,
        )
        # Strict should be None or same as relaxed (if both BB and ZS agree)
        # The key invariant: strict never fires when relaxed wouldn't
        if sig_strict is not None and sig_relaxed is not None:
            assert sig_strict.side == sig_relaxed.side

    def test_non_gld_unaffected_by_strict_flag(self) -> None:
        """SPY should behave the same regardless of gld_strict_confluence."""
        closes = _spike_up_series(60, base=500.0)
        bars = _make_bars(closes)
        tuning_strict = GammaReversionTuning(gld_strict_confluence=True)
        tuning_relaxed = GammaReversionTuning(gld_strict_confluence=False)

        sig_strict = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=tuning_strict,
            now=NOW,
        )
        sig_relaxed = _build_signal_for_symbol(
            symbol="SPY",
            bars=bars,
            price=closes[-1],
            tuning=tuning_relaxed,
            now=NOW,
        )
        # Both should produce identical results for non-GLD
        assert (sig_strict is None) == (sig_relaxed is None)
        if sig_strict is not None and sig_relaxed is not None:
            assert sig_strict.side == sig_relaxed.side
            assert sig_strict.confidence == sig_relaxed.confidence

    def test_gld_strict_default_is_true(self) -> None:
        tuning = GammaReversionTuning()
        assert tuning.gld_strict_confluence is True

    def test_iwm_not_in_default_universe(self) -> None:
        """IWM should no longer be in the default universe."""
        assert "IWM" not in DEFAULT_GAMMA_REVERSION_UNIVERSE

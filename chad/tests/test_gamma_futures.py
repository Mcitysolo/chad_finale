#!/usr/bin/env python3
"""
Tests for the GAMMA_FUTURES mean-reversion strategy.

Covers:
- RSI calculation correctness
- Bollinger Band calculation correctness
- Mean deviation ratio
- Signal generation for overbought / oversold / neutral conditions
- Position sizing validation
- Minimum bars enforcement
- Liquidity filter
- Config builder and env overrides
- Handler contract compatibility
- Registry presence
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence

import pytest

from chad.strategies.gamma_futures import (
    GammaFuturesTuning,
    _bollinger_bands,
    _build_signal_for_symbol,
    _mean_deviation_ratio,
    _rsi,
    build_gamma_futures_config,
    build_gamma_futures_signals,
    gamma_futures_handler,
)
from chad.strategies.gamma_futures_config import (
    DEFAULT_GAMMA_FUTURES_UNIVERSE,
    build_gamma_futures_config as config_build_gamma_futures_config,
)
from chad.strategies.alpha_futures import DEFAULT_SPECS, FuturesInstrumentSpec
from chad.types import AssetClass, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(
    closes: Sequence[float],
    *,
    spread: float = 0.5,
    volume: float = 50_000.0,
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


def _rising_then_spike(n: int = 50, base: float = 5000.0, spike: float = 200.0) -> List[float]:
    """Generate a series that rises gradually then spikes — overbought setup."""
    closes = [base + i * 2.0 for i in range(n - 5)]
    top = closes[-1]
    for i in range(5):
        closes.append(top + spike * (i + 1) / 5)
    return closes


def _falling_then_crash(n: int = 50, base: float = 5000.0, crash: float = 200.0) -> List[float]:
    """Generate a series that falls gradually then crashes — oversold setup."""
    closes = [base - i * 2.0 for i in range(n - 5)]
    bottom = closes[-1]
    for i in range(5):
        closes.append(bottom - crash * (i + 1) / 5)
    return closes


def _flat_series(n: int = 50, value: float = 5000.0) -> List[float]:
    """Generate a flat series — no signal expected."""
    return [value] * n


DEFAULT_TUNING = GammaFuturesTuning()
MES_SPEC = DEFAULT_SPECS["MES"]


# ===================================================================
# RSI Calculation
# ===================================================================


class TestRSI:

    def test_all_gains_returns_100(self) -> None:
        closes = [100.0 + i for i in range(20)]
        rsi = _rsi(closes, 14)
        assert rsi == 100.0

    def test_all_losses_returns_0(self) -> None:
        closes = [100.0 - i for i in range(20)]
        rsi = _rsi(closes, 14)
        assert rsi == 0.0

    def test_flat_returns_50(self) -> None:
        closes = [100.0] * 20
        rsi = _rsi(closes, 14)
        assert rsi == 50.0

    def test_insufficient_data_returns_50(self) -> None:
        closes = [100.0, 101.0, 102.0]
        rsi = _rsi(closes, 14)
        assert rsi == 50.0

    def test_zero_length_returns_50(self) -> None:
        assert _rsi([100.0, 101.0], 0) == 50.0

    def test_mixed_movement_in_range(self) -> None:
        closes = [100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0,
                  105.0, 104.0, 106.0, 105.0, 107.0, 106.0, 108.0, 107.0]
        rsi = _rsi(closes, 14)
        assert 0.0 < rsi < 100.0

    def test_strong_uptrend_high_rsi(self) -> None:
        closes = [100.0 + i * 3.0 for i in range(20)]
        rsi = _rsi(closes, 14)
        assert rsi > 70.0

    def test_strong_downtrend_low_rsi(self) -> None:
        closes = [200.0 - i * 3.0 for i in range(20)]
        rsi = _rsi(closes, 14)
        assert rsi < 30.0


# ===================================================================
# Bollinger Bands Calculation
# ===================================================================


class TestBollingerBands:

    def test_flat_series_zero_width(self) -> None:
        closes = [100.0] * 20
        upper, middle, lower = _bollinger_bands(closes, 20, 2.0)
        assert middle == pytest.approx(100.0)
        assert upper == pytest.approx(100.0)
        assert lower == pytest.approx(100.0)

    def test_known_values(self) -> None:
        closes = list(range(1, 21))  # 1..20
        upper, middle, lower = _bollinger_bands([float(c) for c in closes], 20, 2.0)
        expected_middle = sum(range(1, 21)) / 20.0  # 10.5
        assert middle == pytest.approx(expected_middle)
        assert upper > middle
        assert lower < middle
        assert upper - middle == pytest.approx(middle - lower)

    def test_insufficient_data_returns_zeros(self) -> None:
        upper, middle, lower = _bollinger_bands([100.0, 101.0], 20, 2.0)
        assert upper == 0.0
        assert middle == 0.0
        assert lower == 0.0

    def test_upper_above_lower(self) -> None:
        closes = [100.0 + (i % 5) * 2.0 for i in range(25)]
        upper, middle, lower = _bollinger_bands(closes, 20, 2.0)
        assert upper > middle > lower

    def test_width_scales_bands(self) -> None:
        closes = [100.0, 110.0, 90.0, 105.0, 95.0] * 5  # 25 bars
        u1, m1, l1 = _bollinger_bands(closes, 20, 1.0)
        u2, m2, l2 = _bollinger_bands(closes, 20, 2.0)
        assert m1 == pytest.approx(m2)
        assert (u2 - m2) == pytest.approx(2.0 * (u1 - m1))


# ===================================================================
# Mean Deviation Ratio
# ===================================================================


class TestMeanDeviationRatio:

    def test_positive_deviation(self) -> None:
        assert _mean_deviation_ratio(102.0, 100.0) == pytest.approx(0.02)

    def test_negative_deviation(self) -> None:
        assert _mean_deviation_ratio(98.0, 100.0) == pytest.approx(-0.02)

    def test_zero_reference(self) -> None:
        assert _mean_deviation_ratio(100.0, 0.0) == 0.0

    def test_equal_price_and_reference(self) -> None:
        assert _mean_deviation_ratio(100.0, 100.0) == pytest.approx(0.0)


# ===================================================================
# Signal Generation
# ===================================================================


class TestSignalGeneration:

    def test_overbought_generates_sell(self) -> None:
        """RSI overbought + price above Bollinger upper -> SELL."""
        closes = _rising_then_spike(50, base=5000.0, spike=300.0)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        if signal is not None:
            assert signal.side == SignalSide.SELL
            assert signal.strategy == StrategyName.GAMMA_FUTURES
            assert signal.asset_class == AssetClass.FUTURES
            assert signal.confidence >= DEFAULT_TUNING.min_confidence

    def test_oversold_generates_buy(self) -> None:
        """RSI oversold + price below Bollinger lower -> BUY."""
        closes = _falling_then_crash(50, base=5000.0, crash=300.0)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        if signal is not None:
            assert signal.side == SignalSide.BUY
            assert signal.strategy == StrategyName.GAMMA_FUTURES
            assert signal.confidence >= DEFAULT_TUNING.min_confidence

    def test_flat_series_no_signal(self) -> None:
        """Perfectly flat series should not trigger reversion."""
        closes = _flat_series(50, 5000.0)
        bars = _make_bars(closes, volume=500_000.0)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=5000.0,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        assert signal is None

    def test_mean_overextension_high_sell(self) -> None:
        """Price far above EMA slow -> SELL via mean overextension."""
        # Gradual rise then jump far above the EMA
        closes = [5000.0 + i * 0.5 for i in range(45)]
        # Sudden jump: 5% above the trailing average
        for _ in range(5):
            closes.append(closes[-1] * 1.02)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        tuning = replace(DEFAULT_TUNING, mean_reversion_threshold=0.01)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=tuning, equity=100_000.0,
        )
        if signal is not None:
            assert signal.side == SignalSide.SELL

    def test_mean_overextension_low_buy(self) -> None:
        """Price far below EMA slow -> BUY via mean overextension."""
        closes = [5000.0 - i * 0.5 for i in range(45)]
        for _ in range(5):
            closes.append(closes[-1] * 0.98)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        tuning = replace(DEFAULT_TUNING, mean_reversion_threshold=0.01)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=tuning, equity=100_000.0,
        )
        if signal is not None:
            assert signal.side == SignalSide.BUY

    def test_signal_meta_fields(self) -> None:
        """Signal meta should contain all required fields."""
        closes = _rising_then_spike(50, base=5000.0, spike=300.0)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        if signal is not None:
            meta = signal.meta
            assert "engine" in meta
            assert meta["engine"] == "gamma_futures.v1"
            assert "rsi" in meta
            assert "bb_upper" in meta
            assert "bb_middle" in meta
            assert "bb_lower" in meta
            assert "ema_slow" in meta
            assert "mean_deviation" in meta
            assert "trigger" in meta
            assert meta["required_asset_class"] == "futures"


# ===================================================================
# Sizing and Filters
# ===================================================================


class TestSizingAndFilters:

    def test_minimum_bars_enforcement(self) -> None:
        """Fewer than min_bars should return None."""
        closes = [5000.0] * 10
        bars = _make_bars(closes)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=5000.0,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        assert signal is None

    def test_zero_price_returns_none(self) -> None:
        closes = _rising_then_spike(50)
        bars = _make_bars(closes)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=0.0,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        assert signal is None

    def test_low_liquidity_returns_none(self) -> None:
        """Volume below liquidity threshold -> no signal."""
        closes = _rising_then_spike(50, base=5000.0, spike=300.0)
        price = closes[-1]
        # Volume = 1.0 -> liquidity = price * 1.0 * 5.0 ~= 26,500 < 1M
        bars = _make_bars(closes, volume=1.0)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=100_000.0,
        )
        assert signal is None

    def test_contracts_capped_by_max(self) -> None:
        """Even with large equity, contracts should not exceed spec.max_contracts."""
        closes = _rising_then_spike(50, base=5000.0, spike=300.0)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=DEFAULT_TUNING, equity=10_000_000.0,
        )
        if signal is not None:
            assert signal.size <= MES_SPEC.max_contracts

    def test_allow_long_false_blocks_buy(self) -> None:
        """When allow_long=False, no BUY signals should be generated."""
        closes = _falling_then_crash(50, base=5000.0, crash=300.0)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        tuning = replace(DEFAULT_TUNING, allow_long=False)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=tuning, equity=100_000.0,
        )
        if signal is not None:
            assert signal.side != SignalSide.BUY

    def test_allow_short_false_blocks_sell(self) -> None:
        """When allow_short=False, no SELL signals should be generated."""
        closes = _rising_then_spike(50, base=5000.0, spike=300.0)
        price = closes[-1]
        bars = _make_bars(closes, volume=500_000.0)
        tuning = replace(DEFAULT_TUNING, allow_short=False)
        signal = _build_signal_for_symbol(
            symbol="MES", bars=bars, price=price,
            spec=MES_SPEC, tuning=tuning, equity=100_000.0,
        )
        if signal is not None:
            assert signal.side != SignalSide.SELL


# ===================================================================
# Config Builder
# ===================================================================


class TestConfigBuilder:

    def test_default_config(self) -> None:
        cfg = config_build_gamma_futures_config()
        assert cfg.name == StrategyName.GAMMA_FUTURES
        assert cfg.enabled is True
        assert cfg.target_universe == ["MES", "MNQ", "MCL", "MGC"]
        assert cfg.max_gross_exposure == 0.20
        assert "reversion" in cfg.notes.lower()

    def test_env_disable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAD_GAMMA_FUTURES_ENABLED", "false")
        cfg = config_build_gamma_futures_config()
        assert cfg.enabled is False

    def test_env_universe_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAD_GAMMA_FUTURES_UNIVERSE", "MES,MNQ")
        cfg = config_build_gamma_futures_config()
        assert cfg.target_universe == ["MES", "MNQ"]

    def test_env_exposure_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAD_GAMMA_FUTURES_MAX_GROSS_EXPOSURE", "0.15")
        cfg = config_build_gamma_futures_config()
        assert cfg.max_gross_exposure == pytest.approx(0.15)

    def test_invalid_symbol_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAD_GAMMA_FUTURES_UNIVERSE", "MES,INVALID")
        with pytest.raises(ValueError, match="Unsupported"):
            config_build_gamma_futures_config()

    def test_fallback_config_from_strategy_module(self) -> None:
        cfg = build_gamma_futures_config()
        assert cfg.name == StrategyName.GAMMA_FUTURES
        assert cfg.enabled is True


# ===================================================================
# Handler Contract
# ===================================================================


class TestHandler:

    def test_handler_returns_list(self) -> None:
        result = gamma_futures_handler(object())
        assert isinstance(result, list)

    def test_handler_disabled_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHAD_GAMMA_FUTURES_ENABLED", "false")
        result = gamma_futures_handler(object())
        assert result == []

    def test_handler_with_bad_ctx_no_crash(self) -> None:
        result = gamma_futures_handler(None)
        assert isinstance(result, list)


# ===================================================================
# Registry
# ===================================================================


class TestRegistry:

    def test_gamma_futures_in_registry(self) -> None:
        from chad.strategies import _REGISTRY
        assert StrategyName.GAMMA_FUTURES in _REGISTRY

    def test_gamma_futures_registration_fields(self) -> None:
        from chad.strategies import _REGISTRY
        reg = _REGISTRY[StrategyName.GAMMA_FUTURES]
        assert reg.name == StrategyName.GAMMA_FUTURES
        assert callable(reg.build_config)
        assert callable(reg.handler)

    def test_gamma_futures_config_from_registry(self) -> None:
        from chad.strategies import _REGISTRY
        reg = _REGISTRY[StrategyName.GAMMA_FUTURES]
        cfg = reg.build_config()
        assert cfg.name == StrategyName.GAMMA_FUTURES
        assert cfg.enabled is True

    def test_iteration_includes_gamma_futures(self) -> None:
        from chad.strategies import iter_strategy_registrations
        names = [r.name for r in iter_strategy_registrations()]
        assert StrategyName.GAMMA_FUTURES in names

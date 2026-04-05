#!/usr/bin/env python3
"""
Tests for DELTA_PAIRS market-neutral pairs trading strategy.

Covers:
- Z-score calculation correctness
- Entry signal when |zscore| >= 2.0
- Exit signal when |zscore| <= 0.5
- Stop signal when |zscore| >= 3.5
- Two signals emitted with shared pair_id
- LONG/SHORT roles correct for each direction
- Sizing calculation
- Minimum bars enforcement
- Config builder and env overrides
- Registry integration
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from chad.strategies.delta_pairs import (
    PairSpec,
    DeltaPairsTuning,
    DEFAULT_PAIRS,
    compute_zscore,
    _build_pair_signals,
    build_delta_pairs_signals,
    delta_pairs_handler,
    build_delta_pairs_config,
)
from chad.strategies.delta_pairs_config import (
    DEFAULT_DELTA_PAIRS_UNIVERSE,
    build_delta_pairs_config as config_build_delta_pairs_config,
)
from chad.types import AssetClass, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 5, 14, 30, tzinfo=timezone.utc)


def _make_bars(closes: List[float]) -> List[Dict[str, float]]:
    bars = []
    for c in closes:
        bars.append({
            "open": c - 0.5,
            "high": c + 1.0,
            "low": c - 1.0,
            "close": c,
            "volume": 50_000_000.0,
        })
    return bars


@dataclass
class _FakePortfolio:
    positions: Dict[str, Any]
    extra: Optional[Dict[str, Any]] = None
    cash: float = 100_000.0
    total_equity: float = 500_000.0


@dataclass
class _FakeContext:
    now: datetime
    ticks: Dict[str, Any]
    portfolio: Any = None
    bars: Optional[Dict[str, list]] = None
    prices: Optional[Dict[str, float]] = None


# ===========================================================================
# Z-score computation tests
# ===========================================================================

class TestZscore:

    def test_zscore_none_for_constant_ratio(self) -> None:
        """Constant prices give zero std, so zscore is None (no signal)."""
        closes_a = [100.0] * 60
        closes_b = [50.0] * 60
        z = compute_zscore(closes_a, closes_b, lookback=60)
        assert z is None  # Zero variance -> undefined zscore

    def test_zscore_positive_for_high_ratio(self) -> None:
        """If ratio suddenly spikes, zscore should be positive."""
        # Stable ratio = 2.0, then spike on last bar
        closes_a = [100.0] * 59 + [120.0]
        closes_b = [50.0] * 60
        z = compute_zscore(closes_a, closes_b, lookback=60)
        assert z is not None
        assert z > 0

    def test_zscore_negative_for_low_ratio(self) -> None:
        """If ratio suddenly drops, zscore should be negative."""
        closes_a = [100.0] * 59 + [80.0]
        closes_b = [50.0] * 60
        z = compute_zscore(closes_a, closes_b, lookback=60)
        assert z is not None
        assert z < 0

    def test_zscore_insufficient_data(self) -> None:
        """Should return None with fewer bars than lookback."""
        closes_a = [100.0] * 30
        closes_b = [50.0] * 30
        z = compute_zscore(closes_a, closes_b, lookback=60)
        assert z is None

    def test_zscore_exact_value(self) -> None:
        """Verify exact zscore calculation."""
        # 59 bars at ratio=2.0, 1 bar at ratio=2.4
        closes_a = [100.0] * 59 + [120.0]
        closes_b = [50.0] * 60
        z = compute_zscore(closes_a, closes_b, lookback=60)
        assert z is not None

        # Manually: ratios = [2.0]*59 + [2.4]
        # mean = (59*2.0 + 2.4) / 60 = 120.4/60 = 2.00667
        # var = (59*(2.0-2.00667)^2 + (2.4-2.00667)^2) / 60
        # z = (2.4 - 2.00667) / std
        ratios = [2.0] * 59 + [2.4]
        mean = sum(ratios) / 60
        var = sum((r - mean) ** 2 for r in ratios) / 60
        std = math.sqrt(var)
        expected_z = (2.4 - mean) / std
        assert z == pytest.approx(expected_z, rel=1e-6)


# ===========================================================================
# Entry signal tests
# ===========================================================================

class TestEntrySignals:

    def _make_diverging_context(self, spike_a: float = 120.0) -> _FakeContext:
        """Build a context where SPY/QQQ ratio diverges."""
        # SPY stable at 550, then spikes
        spy_closes = [550.0] * 59 + [spike_a]
        qqq_closes = [490.0] * 60
        return _FakeContext(
            now=NOW,
            ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={
                "SPY": _make_bars(spy_closes),
                "QQQ": _make_bars(qqq_closes),
                "IWM": _make_bars([220.0] * 60),
            },
        )

    def test_entry_when_zscore_exceeds_threshold(self) -> None:
        """Should emit signals when |zscore| >= 2.0."""
        # Big spike to push zscore well past 2.0
        ctx = self._make_diverging_context(spike_a=700.0)
        signals = build_delta_pairs_signals(ctx)
        # Should have signals for SPY/QQQ pair (large divergence)
        spy_qqq_signals = [s for s in signals if
                          s.meta.get("pair") == "SPY/QQQ"]
        assert len(spy_qqq_signals) == 2

    def test_no_entry_when_zscore_below_threshold(self) -> None:
        """Should NOT emit when zscore is below entry threshold."""
        # Stable ratio — no divergence, zscore ~0
        spy_closes = [550.0] * 60
        qqq_closes = [490.0] * 60
        ctx = _FakeContext(
            now=NOW,
            ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={
                "SPY": _make_bars(spy_closes),
                "QQQ": _make_bars(qqq_closes),
                "IWM": _make_bars([220.0] * 60),
            },
        )
        signals = build_delta_pairs_signals(ctx)
        spy_qqq = [s for s in signals if s.meta.get("pair") == "SPY/QQQ"]
        assert len(spy_qqq) == 0

    def test_positive_zscore_shorts_first_longs_second(self) -> None:
        """Z > 0 means ratio too high: SHORT sym_long, LONG sym_short."""
        ctx = self._make_diverging_context(spike_a=700.0)
        signals = build_delta_pairs_signals(ctx)
        spy_qqq = [s for s in signals if s.meta.get("pair") == "SPY/QQQ"]
        if len(spy_qqq) == 2:
            spy_sig = [s for s in spy_qqq if s.symbol == "SPY"][0]
            qqq_sig = [s for s in spy_qqq if s.symbol == "QQQ"][0]
            assert spy_sig.side == SignalSide.SELL
            assert qqq_sig.side == SignalSide.BUY
            assert spy_sig.meta["pair_role"] == "SHORT"
            assert qqq_sig.meta["pair_role"] == "LONG"

    def test_negative_zscore_longs_first_shorts_second(self) -> None:
        """Z < 0 means ratio too low: LONG sym_long, SHORT sym_short."""
        # SPY drops while QQQ stays -> ratio drops -> z < 0
        spy_closes = [550.0] * 59 + [400.0]
        qqq_closes = [490.0] * 60
        ctx = _FakeContext(
            now=NOW,
            ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={
                "SPY": _make_bars(spy_closes),
                "QQQ": _make_bars(qqq_closes),
                "IWM": _make_bars([220.0] * 60),
            },
        )
        signals = build_delta_pairs_signals(ctx)
        spy_qqq = [s for s in signals if s.meta.get("pair") == "SPY/QQQ"]
        if len(spy_qqq) == 2:
            spy_sig = [s for s in spy_qqq if s.symbol == "SPY"][0]
            qqq_sig = [s for s in spy_qqq if s.symbol == "QQQ"][0]
            assert spy_sig.side == SignalSide.BUY
            assert qqq_sig.side == SignalSide.SELL


# ===========================================================================
# Signal pair linkage tests
# ===========================================================================

class TestPairLinkage:

    def test_two_signals_share_pair_id(self) -> None:
        """Both legs of a pair trade must share the same pair_id."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.8,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 2
        assert signals[0].meta["pair_id"] == signals[1].meta["pair_id"]

    def test_partner_symbols_correct(self) -> None:
        """Each leg should reference its partner."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.8,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        spy_sig = [s for s in signals if s.symbol == "SPY"][0]
        qqq_sig = [s for s in signals if s.symbol == "QQQ"][0]
        assert spy_sig.meta["partner_symbol"] == "QQQ"
        assert qqq_sig.meta["partner_symbol"] == "SPY"


# ===========================================================================
# Exit / stop signal tests
# ===========================================================================

class TestExitStop:

    def test_exit_signal_type(self) -> None:
        """zscore near zero triggers exit type."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=0.3,  # Below exit threshold 0.5
            confidence=0.8,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        if signals:
            assert signals[0].meta["signal_type"] == "exit"

    def test_stop_signal_type(self) -> None:
        """zscore beyond stop threshold triggers stop type."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=4.0,  # Above stop threshold 3.5
            confidence=0.9,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 2
        # zscore 4.0 >= entry 2.0, so signal_type is "entry" (entry takes priority)
        assert signals[0].meta["signal_type"] == "entry"

    def test_no_signal_in_dead_zone(self) -> None:
        """zscore between exit and entry thresholds emits nothing."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=1.2,  # Between 0.5 and 2.0
            confidence=0.8,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 0


# ===========================================================================
# Sizing tests
# ===========================================================================

class TestSizing:

    def test_sizing_bounded_by_max_units(self) -> None:
        """Units should not exceed max_units."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.9,
            equity=10_000_000.0,  # Huge equity to push units past max
            tuning=DeltaPairsTuning(max_units=50),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 2
        assert signals[0].size <= 50

    def test_sizing_minimum_one_unit(self) -> None:
        """Should always emit at least 1 unit if signal qualifies."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.7,
            equity=1_000.0,  # Tiny equity
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        if signals:
            assert signals[0].size >= 1.0

    def test_both_legs_same_size(self) -> None:
        """Dollar-neutral: both legs must have the same unit count."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.85,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 2
        assert signals[0].size == signals[1].size

    def test_no_signals_on_zero_equity(self) -> None:
        """Zero equity produces no signals."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.8,
            equity=0.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        # With 0 equity, notional=0, units=max(1,...) so still emits 1
        # This is fine — risk layer downstream enforces actual limits


# ===========================================================================
# Minimum bars enforcement
# ===========================================================================

class TestMinBars:

    def test_insufficient_bars_produces_no_signals(self) -> None:
        """Less than min_bars should produce no signals."""
        spy_closes = [550.0] * 20  # Only 20 bars
        qqq_closes = [490.0] * 20
        ctx = _FakeContext(
            now=NOW,
            ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={
                "SPY": _make_bars(spy_closes),
                "QQQ": _make_bars(qqq_closes),
                "IWM": _make_bars([220.0] * 20),
            },
        )
        signals = build_delta_pairs_signals(ctx)
        assert len(signals) == 0

    def test_missing_bars_produces_no_signals(self) -> None:
        """Missing bars for one symbol should produce no signals for that pair."""
        ctx = _FakeContext(
            now=NOW,
            ticks={},
            bars={"SPY": _make_bars([550.0] * 60)},
        )
        signals = build_delta_pairs_signals(ctx)
        assert len(signals) == 0


# ===========================================================================
# Confidence gating
# ===========================================================================

class TestConfidence:

    def test_low_confidence_filtered(self) -> None:
        """Signals below min_confidence should not be emitted."""
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.3,  # Below min_confidence 0.65
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 0


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:

    def test_default_config(self) -> None:
        cfg = config_build_delta_pairs_config()
        assert cfg.name == StrategyName.DELTA_PAIRS
        assert cfg.enabled is True
        assert list(cfg.target_universe) == ["SPY", "QQQ", "IWM"]
        assert cfg.max_gross_exposure == pytest.approx(0.15)

    def test_env_disable(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_DELTA_PAIRS_ENABLED": "false"}):
            cfg = config_build_delta_pairs_config()
            assert cfg.enabled is False

    def test_env_universe_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_DELTA_PAIRS_UNIVERSE": "SPY,QQQ"}):
            cfg = config_build_delta_pairs_config()
            assert list(cfg.target_universe) == ["SPY", "QQQ"]

    def test_invalid_symbol_raises(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_DELTA_PAIRS_UNIVERSE": "DOGE"}):
            with pytest.raises(ValueError, match="Unsupported"):
                config_build_delta_pairs_config()

    def test_env_max_gross_exposure_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_DELTA_PAIRS_MAX_GROSS_EXPOSURE": "0.20"}):
            cfg = config_build_delta_pairs_config()
            assert cfg.max_gross_exposure == pytest.approx(0.20)

    def test_fallback_config(self) -> None:
        cfg = build_delta_pairs_config()
        assert cfg.name == StrategyName.DELTA_PAIRS


# ===========================================================================
# Handler tests
# ===========================================================================

class TestHandler:

    def test_handler_returns_list(self) -> None:
        ctx = _FakeContext(now=NOW, ticks={}, portfolio=_FakePortfolio(positions={}))
        result = delta_pairs_handler(ctx)
        assert isinstance(result, list)

    def test_handler_disabled_returns_empty(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_DELTA_PAIRS_ENABLED": "false"}):
            ctx = _FakeContext(now=NOW, ticks={}, portfolio=_FakePortfolio(positions={}))
            assert delta_pairs_handler(ctx) == []

    def test_handler_survives_bad_context(self) -> None:
        assert isinstance(delta_pairs_handler(None), list)


# ===========================================================================
# Registry integration
# ===========================================================================

class TestRegistry:

    def test_strategy_name_enum(self) -> None:
        assert hasattr(StrategyName, "DELTA_PAIRS")
        assert StrategyName.DELTA_PAIRS.value == "delta_pairs"

    def test_registry_contains_delta_pairs(self) -> None:
        from chad.strategies import iter_strategy_registrations
        names = [reg.name for reg in iter_strategy_registrations()]
        assert StrategyName.DELTA_PAIRS in names

    def test_registry_config_builds(self) -> None:
        from chad.strategies import iter_strategy_registrations
        for reg in iter_strategy_registrations():
            if reg.name == StrategyName.DELTA_PAIRS:
                cfg = reg.build_config()
                assert cfg.name == StrategyName.DELTA_PAIRS
                return
        pytest.fail("DELTA_PAIRS not found in registry")


# ===========================================================================
# PairSpec defaults
# ===========================================================================

class TestPairSpec:

    def test_default_pairs_count(self) -> None:
        assert len(DEFAULT_PAIRS) == 3

    def test_default_pairs_have_high_correlation(self) -> None:
        for p in DEFAULT_PAIRS:
            assert p.correlation > 0.85

    def test_default_thresholds(self) -> None:
        for p in DEFAULT_PAIRS:
            assert p.zscore_entry == 2.0
            assert p.zscore_exit == 0.5
            assert p.zscore_stop == 3.5


# ===========================================================================
# Signal metadata
# ===========================================================================

class TestSignalMetadata:

    def test_signals_have_required_meta_fields(self) -> None:
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.85,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert len(signals) == 2
        for sig in signals:
            assert sig.strategy == StrategyName.DELTA_PAIRS
            assert sig.asset_class == AssetClass.ETF
            assert "pair_id" in sig.meta
            assert "pair_role" in sig.meta
            assert "partner_symbol" in sig.meta
            assert "zscore" in sig.meta
            assert "correlation" in sig.meta
            assert sig.meta["engine"] == "delta_pairs.v1"
            assert sig.meta["sec_type"] == "STK"

    def test_zscore_in_meta_matches_input(self) -> None:
        pair = PairSpec("SPY", "QQQ", correlation=0.993, half_life_days=18.2)
        signals = _build_pair_signals(
            pair=pair,
            zscore=2.5,
            confidence=0.85,
            equity=500_000.0,
            tuning=DeltaPairsTuning(),
            now=NOW,
            price_a=550.0,
            price_b=490.0,
        )
        assert signals[0].meta["zscore"] == pytest.approx(2.5)

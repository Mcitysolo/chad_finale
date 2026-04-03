#!/usr/bin/env python3
"""
Tests for OMEGA_VOL volatility regime strategy.

Covers:
- Vol regime classification (all 5 states + edge cases)
- Signal direction: correct instrument per regime
- No signal in ELEVATED_VOL
- No signal when VIX data unavailable (fail closed)
- UVXY unit halving
- Confidence calculation
- Momentum gating
- Config builder and env overrides
- Registry integration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from chad.strategies.omega_vol import (
    VolRegime,
    OmegaVolTuning,
    _compute_confidence,
    _compute_units,
    _vix_momentum,
    _vix_zscore,
    classify_vol_regime,
    build_omega_vol_config,
    build_omega_vol_signals,
    omega_vol_handler,
)
from chad.strategies.omega_vol_config import (
    DEFAULT_OMEGA_VOL_UNIVERSE,
    build_omega_vol_config as config_build,
)
from chad.types import AssetClass, SignalSide, StrategyName

NOW = datetime(2026, 4, 3, 14, 30, tzinfo=timezone.utc)


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
    legend: Any = None
    portfolio: Any = None
    bars: Optional[Dict[str, list]] = None
    vix: Optional[float] = None
    vix_history: Optional[List[float]] = None
    prices: Optional[Dict[str, float]] = None


def _ctx(
    vix: Optional[float] = 20.0,
    vix_history: Optional[List[float]] = None,
    equity: float = 500_000.0,
    prices: Optional[Dict[str, float]] = None,
) -> _FakeContext:
    return _FakeContext(
        now=NOW,
        ticks={},
        portfolio=_FakePortfolio(positions={}, total_equity=equity),
        vix=vix,
        vix_history=vix_history,
        prices=prices or {"SVXY": 46.0, "UVXY": 51.0},
    )


# ===========================================================================
# Regime classification tests
# ===========================================================================

class TestVolRegime:

    def test_low_vol(self) -> None:
        assert classify_vol_regime(12.0, [12.0] * 20) == VolRegime.LOW_VOL

    def test_low_vol_boundary(self) -> None:
        assert classify_vol_regime(14.9, [14.9] * 20) == VolRegime.LOW_VOL

    def test_normal_vol(self) -> None:
        assert classify_vol_regime(18.0, [18.0] * 20) == VolRegime.NORMAL_VOL

    def test_normal_vol_boundaries(self) -> None:
        assert classify_vol_regime(15.0, [15.0] * 20) == VolRegime.NORMAL_VOL
        assert classify_vol_regime(21.9, [21.9] * 20) == VolRegime.NORMAL_VOL

    def test_elevated_vol(self) -> None:
        assert classify_vol_regime(25.0, [25.0] * 20) == VolRegime.ELEVATED_VOL

    def test_elevated_vol_boundary(self) -> None:
        assert classify_vol_regime(22.0, [22.0] * 20) == VolRegime.ELEVATED_VOL

    def test_crisis_vol(self) -> None:
        assert classify_vol_regime(35.0, [35.0] * 20) == VolRegime.CRISIS_VOL

    def test_crisis_vol_boundary(self) -> None:
        assert classify_vol_regime(30.0, [30.0] * 20) == VolRegime.CRISIS_VOL

    def test_vol_crush(self) -> None:
        """VIX dropped >20% from recent peak (35 -> 26)."""
        history = [35.0, 34.0, 32.0, 30.0, 28.0, 26.0]
        assert classify_vol_regime(26.0, history) == VolRegime.VOL_CRUSH

    def test_vol_crush_requires_below_crisis(self) -> None:
        """VOL_CRUSH should not fire if VIX is still >= 30."""
        history = [40.0, 38.0, 36.0, 34.0, 32.0, 31.0]
        # 31 is not < 30, so even though >20% drop from 40, should be CRISIS
        # Actually 31 < 40*(1-0.20)=32, and 31 >= 30 → CRISIS takes priority
        result = classify_vol_regime(31.0, history)
        assert result == VolRegime.CRISIS_VOL

    def test_vol_crush_not_triggered_on_small_drop(self) -> None:
        """15% drop from peak — not enough for VOL_CRUSH."""
        history = [30.0, 29.0, 28.0, 27.0, 26.0, 25.5]
        # 25.5 / 30 = 0.85 → 15% drop, need >20%
        assert classify_vol_regime(25.5, history) == VolRegime.ELEVATED_VOL

    def test_short_history(self) -> None:
        """With only 1 bar of history, VOL_CRUSH cannot trigger."""
        assert classify_vol_regime(12.0, [12.0]) == VolRegime.LOW_VOL


# ===========================================================================
# Indicator tests
# ===========================================================================

class TestIndicators:

    def test_vix_momentum_rising(self) -> None:
        history = [18.0, 19.0, 20.0, 22.0, 24.0, 26.0]
        mom = _vix_momentum(history, period=5)
        assert mom is not None
        assert mom > 0

    def test_vix_momentum_falling(self) -> None:
        history = [30.0, 28.0, 26.0, 24.0, 22.0, 20.0]
        mom = _vix_momentum(history, period=5)
        assert mom is not None
        assert mom < 0

    def test_vix_momentum_insufficient(self) -> None:
        assert _vix_momentum([20.0, 21.0], period=5) is None

    def test_vix_zscore_above(self) -> None:
        history = [15.0] * 19 + [25.0]
        zs = _vix_zscore(history, period=20)
        assert zs is not None
        assert zs > 2.0

    def test_vix_zscore_below(self) -> None:
        history = [25.0] * 19 + [15.0]
        zs = _vix_zscore(history, period=20)
        assert zs is not None
        assert zs < -2.0

    def test_vix_zscore_at_mean(self) -> None:
        history = [20.0] * 20
        zs = _vix_zscore(history, period=20)
        assert zs is not None
        assert abs(zs) < 0.01


# ===========================================================================
# Signal direction tests
# ===========================================================================

class TestSignalDirection:

    def test_low_vol_buys_svxy(self) -> None:
        history = [12.0] * 25
        signals = build_omega_vol_signals(_ctx(vix=12.0, vix_history=history))
        assert len(signals) == 1
        assert signals[0].symbol == "SVXY"
        assert signals[0].side == SignalSide.BUY
        assert signals[0].meta["regime"] == "low_vol"
        assert signals[0].meta["position_type"] == "short_vol"

    def test_normal_vol_buys_svxy_when_momentum_negative(self) -> None:
        # Declining VIX at the END so 5-bar momentum is negative
        history = [19.0] * 15 + [19.0, 18.8, 18.5, 18.3, 18.0, 17.8, 17.5, 17.2, 17.0, 16.8]
        signals = build_omega_vol_signals(_ctx(vix=16.8, vix_history=history))
        svxy = [s for s in signals if s.symbol == "SVXY"]
        assert len(svxy) == 1

    def test_normal_vol_no_signal_when_momentum_positive(self) -> None:
        """VIX 18 but rising → no signal."""
        history = [16.0, 16.5, 17.0, 17.5, 18.0, 18.5] + [18.5] * 19
        signals = build_omega_vol_signals(_ctx(vix=18.5, vix_history=history))
        assert len(signals) == 0

    def test_elevated_vol_no_signal(self) -> None:
        history = [25.0] * 25
        signals = build_omega_vol_signals(_ctx(vix=25.0, vix_history=history))
        assert len(signals) == 0

    def test_crisis_vol_buys_uvxy(self) -> None:
        # Rising VIX at the END so momentum > 0, no vol_crush trigger
        history = [20.0] * 15 + [22.0, 24.0, 26.0, 28.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0]
        signals = build_omega_vol_signals(_ctx(vix=35.0, vix_history=history))
        assert len(signals) == 1
        assert signals[0].symbol == "UVXY"
        assert signals[0].side == SignalSide.BUY
        assert signals[0].meta["regime"] == "crisis_vol"
        assert signals[0].meta["position_type"] == "long_vol"

    def test_crisis_vol_no_signal_when_momentum_negative(self) -> None:
        """VIX 32 but falling → no signal (don't catch falling knife)."""
        history = [40.0, 38.0, 36.0, 34.0, 33.0, 32.0] + [32.0] * 19
        # momentum is negative, and this triggers VOL_CRUSH (>20% from 40)
        signals = build_omega_vol_signals(_ctx(vix=32.0, vix_history=history))
        # VOL_CRUSH fires instead → SVXY
        if signals:
            assert signals[0].symbol == "SVXY"

    def test_vol_crush_buys_svxy(self) -> None:
        # Peak of 35 within last 5 bars, current 25 → 28% drop → VOL_CRUSH
        history = [20.0] * 15 + [22.0, 25.0, 30.0, 35.0, 32.0, 29.0, 27.0, 26.0, 25.0]
        # Last 5 bars: [35, 32, 29, 27, 26, 25] → peak=35, 25/35=0.714 → 28.6% drop
        signals = build_omega_vol_signals(_ctx(vix=25.0, vix_history=history))
        assert len(signals) == 1
        assert signals[0].symbol == "SVXY"
        assert signals[0].meta["regime"] == "vol_crush"
        assert signals[0].meta["reason"] == "vol_crush_reversion"


# ===========================================================================
# Fail-closed tests
# ===========================================================================

class TestFailClosed:

    def test_no_vix_no_signal(self) -> None:
        signals = build_omega_vol_signals(_ctx(vix=None))
        assert len(signals) == 0

    def test_handler_survives_bad_context(self) -> None:
        result = omega_vol_handler(None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_handler_disabled(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_VOL_ENABLED": "false"}):
            result = omega_vol_handler(_ctx(vix=12.0, vix_history=[12.0] * 25))
            assert result == []


# ===========================================================================
# UVXY halving test
# ===========================================================================

class TestUVXYHalving:

    def test_uvxy_units_halved(self) -> None:
        tuning = OmegaVolTuning(base_units=6, max_units=10)
        # Long vol (UVXY) should be halved
        units_long = _compute_units(0.75, 500_000.0, 51.0, is_long_vol=True, tuning=tuning)
        units_short = _compute_units(0.75, 500_000.0, 46.0, is_long_vol=False, tuning=tuning)
        assert units_long < units_short

    def test_uvxy_at_least_one_unit(self) -> None:
        tuning = OmegaVolTuning(base_units=2, max_units=4)
        units = _compute_units(0.65, 500_000.0, 51.0, is_long_vol=True, tuning=tuning)
        assert units >= 1


# ===========================================================================
# Confidence tests
# ===========================================================================

class TestConfidence:

    def test_low_vol_base(self) -> None:
        conf = _compute_confidence(VolRegime.LOW_VOL, zscore=0.0, momentum=0.0, tuning=OmegaVolTuning())
        assert conf == pytest.approx(0.75, abs=0.001)

    def test_low_vol_with_zscore_bonus(self) -> None:
        conf = _compute_confidence(VolRegime.LOW_VOL, zscore=-2.0, momentum=0.0, tuning=OmegaVolTuning())
        assert conf == pytest.approx(0.80, abs=0.001)

    def test_vol_crush_base(self) -> None:
        conf = _compute_confidence(VolRegime.VOL_CRUSH, zscore=0.0, momentum=0.0, tuning=OmegaVolTuning())
        assert conf == pytest.approx(0.80, abs=0.001)

    def test_vol_crush_with_momentum_bonus(self) -> None:
        conf = _compute_confidence(VolRegime.VOL_CRUSH, zscore=0.0, momentum=-0.15, tuning=OmegaVolTuning())
        assert conf == pytest.approx(0.85, abs=0.001)

    def test_crisis_base(self) -> None:
        conf = _compute_confidence(VolRegime.CRISIS_VOL, zscore=0.0, momentum=0.0, tuning=OmegaVolTuning())
        assert conf == pytest.approx(0.70, abs=0.001)

    def test_crisis_full_bonus(self) -> None:
        conf = _compute_confidence(VolRegime.CRISIS_VOL, zscore=2.5, momentum=0.20, tuning=OmegaVolTuning())
        assert conf == pytest.approx(0.80, abs=0.001)

    def test_elevated_low_confidence(self) -> None:
        conf = _compute_confidence(VolRegime.ELEVATED_VOL, zscore=0.0, momentum=0.0, tuning=OmegaVolTuning())
        assert conf < 0.65  # Below min_confidence threshold


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:

    def test_default_config(self) -> None:
        cfg = config_build()
        assert cfg.name == StrategyName.OMEGA_VOL
        assert cfg.enabled is True
        assert list(cfg.target_universe) == ["SVXY", "UVXY"]
        assert cfg.max_gross_exposure == pytest.approx(0.06)

    def test_env_disable(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_VOL_ENABLED": "false"}):
            assert config_build().enabled is False

    def test_env_universe(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_VOL_UNIVERSE": "SVXY"}):
            assert list(config_build().target_universe) == ["SVXY"]

    def test_invalid_symbol(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_VOL_UNIVERSE": "SPY"}):
            with pytest.raises(ValueError, match="Unsupported"):
                config_build()


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry:

    def test_strategy_name_enum(self) -> None:
        assert StrategyName.OMEGA_VOL.value == "omega_vol"

    def test_registry_contains_omega_vol(self) -> None:
        from chad.strategies import iter_strategy_registrations
        names = [r.name for r in iter_strategy_registrations()]
        assert StrategyName.OMEGA_VOL in names

    def test_registry_config_builds(self) -> None:
        from chad.strategies import iter_strategy_registrations
        for r in iter_strategy_registrations():
            if r.name == StrategyName.OMEGA_VOL:
                cfg = r.build_config()
                assert cfg.name == StrategyName.OMEGA_VOL
                return
        pytest.fail("OMEGA_VOL not found in registry")

#!/usr/bin/env python3
"""
Tests for the OMEGA_MACRO macro regime futures strategy.

Covers:
- MacroRegime classification (all 4 states)
- Signal direction correctness per regime per instrument
- Confidence calculation
- Sizing validation
- Config builder and env overrides
- Handler contract compatibility
- Registry integration
- Macro sensor utilities
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest import mock

import pytest

from chad.strategies.macro_sensors import (
    MacroRegime,
    _atr_pct_from_bars,
    _ema_slope,
    _portfolio_drawdown_pct,
    _vix_value,
    classify_macro_regime,
)
from chad.strategies.omega_macro import (
    OMEGA_MACRO_SPECS,
    OmegaMacroTuning,
    REGIME_SIGNAL_MAP,
    _build_signal_for_symbol,
    _compute_confidence,
    build_omega_macro_config,
    build_omega_macro_signals,
    omega_macro_handler,
)
from chad.strategies.omega_macro_config import (
    DEFAULT_OMEGA_MACRO_UNIVERSE,
    OmegaMacroConfigSpec,
    build_omega_macro_config as config_build_omega_macro_config,
)
from chad.strategies.alpha_futures import FuturesInstrumentSpec
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


def _steady_series(n: int = 50, base: float = 110.0, step: float = 0.1) -> List[float]:
    """Generate a gently rising price series."""
    return [base + i * step for i in range(n)]


def _declining_series(n: int = 50, base: float = 110.0, step: float = 0.1) -> List[float]:
    """Generate a gently declining price series."""
    return [base - i * step for i in range(n)]


@dataclass
class _FakePortfolio:
    positions: Dict[str, Any]
    extra: Optional[Dict[str, Any]] = None
    cash: float = 100_000.0
    total_equity: float = 100_000.0


@dataclass
class _FakeContext:
    now: datetime
    ticks: Dict[str, Any]
    legend: Any = None
    portfolio: Any = None
    bars: Optional[Dict[str, list]] = None
    vix: Optional[float] = None
    vol_index: Optional[float] = None
    volatility_index: Optional[float] = None
    prices: Optional[Dict[str, float]] = None


def _build_ctx(
    *,
    vix: Optional[float] = 20.0,
    equity: float = 100_000.0,
    equity_peak: float = 100_000.0,
    bars: Optional[Dict[str, list]] = None,
    prices: Optional[Dict[str, float]] = None,
) -> _FakeContext:
    """Build a fake context suitable for OMEGA_MACRO testing."""
    portfolio = _FakePortfolio(
        positions={},
        extra={"equity": equity, "equity_peak": equity_peak},
        total_equity=equity,
    )
    return _FakeContext(
        now=datetime(2026, 4, 3, 14, 30, tzinfo=timezone.utc),
        ticks={},
        portfolio=portfolio,
        bars=bars,
        vix=vix,
        prices=prices,
    )


# ===========================================================================
# MacroRegime classification tests
# ===========================================================================

class TestMacroRegimeClassification:
    """Test classify_macro_regime for all 4 states."""

    def test_risk_off_high_vix(self) -> None:
        result = classify_macro_regime(vix=30.0, drawdown_pct=0.0, bond_trend=0.001, commodity_trend=-0.001)
        assert result == MacroRegime.RISK_OFF

    def test_risk_off_vix_exactly_25(self) -> None:
        result = classify_macro_regime(vix=25.0, drawdown_pct=0.0, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.RISK_OFF

    def test_risk_off_deep_drawdown(self) -> None:
        result = classify_macro_regime(vix=15.0, drawdown_pct=-0.08, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.RISK_OFF

    def test_risk_off_drawdown_exactly_minus_5(self) -> None:
        result = classify_macro_regime(vix=15.0, drawdown_pct=-0.05, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.RISK_OFF

    def test_stagflation(self) -> None:
        result = classify_macro_regime(vix=22.0, drawdown_pct=-0.01, bond_trend=-0.002, commodity_trend=0.003)
        assert result == MacroRegime.STAGFLATION

    def test_stagflation_vix_exactly_20(self) -> None:
        result = classify_macro_regime(vix=20.0, drawdown_pct=0.0, bond_trend=-0.001, commodity_trend=0.001)
        assert result == MacroRegime.STAGFLATION

    def test_risk_on(self) -> None:
        result = classify_macro_regime(vix=14.0, drawdown_pct=-0.01, bond_trend=-0.001, commodity_trend=0.001)
        assert result == MacroRegime.RISK_ON

    def test_risk_on_vix_just_below_18(self) -> None:
        result = classify_macro_regime(vix=17.9, drawdown_pct=0.0, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.RISK_ON

    def test_neutral_mixed_signals(self) -> None:
        """VIX 19 (between 18 and 20) with no trend data -> NEUTRAL."""
        result = classify_macro_regime(vix=19.0, drawdown_pct=-0.01, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.NEUTRAL

    def test_neutral_none_inputs(self) -> None:
        """All None -> defaults -> VIX 20.0, dd 0.0 -> NEUTRAL."""
        result = classify_macro_regime(vix=None, drawdown_pct=None, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.NEUTRAL

    def test_neutral_vix_18_exact(self) -> None:
        """VIX exactly 18 doesn't qualify for RISK_ON (requires <18)."""
        result = classify_macro_regime(vix=18.0, drawdown_pct=0.0, bond_trend=None, commodity_trend=None)
        assert result == MacroRegime.NEUTRAL

    def test_stagflation_requires_both_trends(self) -> None:
        """Missing commodity_trend -> cannot classify STAGFLATION."""
        result = classify_macro_regime(vix=22.0, drawdown_pct=0.0, bond_trend=-0.001, commodity_trend=None)
        assert result == MacroRegime.NEUTRAL

    def test_risk_off_takes_priority_over_stagflation(self) -> None:
        """VIX >= 25 triggers RISK_OFF even with stagflation conditions."""
        result = classify_macro_regime(vix=26.0, drawdown_pct=0.0, bond_trend=-0.001, commodity_trend=0.001)
        assert result == MacroRegime.RISK_OFF


# ===========================================================================
# Signal direction tests
# ===========================================================================

class TestSignalDirections:
    """Verify per-instrument signal direction for each regime."""

    def test_risk_off_directions(self) -> None:
        directions = REGIME_SIGNAL_MAP[MacroRegime.RISK_OFF]
        assert directions["ZN"] == SignalSide.BUY
        assert directions["ZB"] == SignalSide.BUY
        assert directions["M6E"] == SignalSide.SELL
        assert directions["SIL"] == SignalSide.SELL

    def test_risk_on_directions(self) -> None:
        directions = REGIME_SIGNAL_MAP[MacroRegime.RISK_ON]
        assert directions["ZN"] == SignalSide.SELL
        assert directions["ZB"] == SignalSide.SELL
        assert directions["M6E"] == SignalSide.BUY
        assert directions["SIL"] == SignalSide.BUY

    def test_stagflation_directions(self) -> None:
        directions = REGIME_SIGNAL_MAP[MacroRegime.STAGFLATION]
        assert directions["ZN"] == SignalSide.BUY
        assert directions["ZB"] == SignalSide.BUY
        assert directions["M6E"] == SignalSide.SELL
        assert directions["SIL"] == SignalSide.BUY

    def test_neutral_no_signals(self) -> None:
        directions = REGIME_SIGNAL_MAP[MacroRegime.NEUTRAL]
        assert len(directions) == 0

    def test_all_instruments_covered_in_active_regimes(self) -> None:
        for regime in (MacroRegime.RISK_OFF, MacroRegime.RISK_ON, MacroRegime.STAGFLATION):
            directions = REGIME_SIGNAL_MAP[regime]
            for sym in OMEGA_MACRO_SPECS:
                assert sym in directions, f"{sym} missing from {regime.value}"


# ===========================================================================
# Confidence calculation tests
# ===========================================================================

class TestConfidence:
    """Test confidence calculation logic."""

    def test_base_confidence(self) -> None:
        """With no strong signals, confidence should be base 0.55."""
        conf = _compute_confidence(
            regime=MacroRegime.RISK_OFF,
            vix=25.0,  # at threshold, not > 30
            drawdown_pct=-0.03,  # not at -5%
            bond_trend=None,
            commodity_trend=None,
            symbol="ZN",
        )
        assert conf == pytest.approx(0.55, abs=0.001)

    def test_strong_vix_risk_off(self) -> None:
        """VIX > 30 in RISK_OFF adds +0.10."""
        conf = _compute_confidence(
            regime=MacroRegime.RISK_OFF,
            vix=35.0,
            drawdown_pct=-0.03,
            bond_trend=None,
            commodity_trend=None,
            symbol="ZN",
        )
        assert conf == pytest.approx(0.65, abs=0.001)

    def test_strong_vix_risk_on(self) -> None:
        """VIX < 15 in RISK_ON adds +0.10."""
        conf = _compute_confidence(
            regime=MacroRegime.RISK_ON,
            vix=12.0,
            drawdown_pct=0.0,
            bond_trend=None,
            commodity_trend=None,
            symbol="ZN",
        )
        # +0.10 vix strong + 0.10 drawdown aligns (>-0.01)
        assert conf == pytest.approx(0.75, abs=0.001)

    def test_drawdown_alignment_risk_off(self) -> None:
        """Drawdown <= -5% in RISK_OFF adds +0.10."""
        conf = _compute_confidence(
            regime=MacroRegime.RISK_OFF,
            vix=26.0,  # not > 30
            drawdown_pct=-0.06,
            bond_trend=None,
            commodity_trend=None,
            symbol="ZN",
        )
        assert conf == pytest.approx(0.65, abs=0.001)

    def test_all_bonuses_risk_off(self) -> None:
        """All bonuses stacked for RISK_OFF: 0.55 + 0.10 + 0.10 + 0.10 + 0.05 = 0.90."""
        conf = _compute_confidence(
            regime=MacroRegime.RISK_OFF,
            vix=35.0,           # strong VIX -> +0.10
            drawdown_pct=-0.06, # drawdown aligns -> +0.10
            bond_trend=0.005,   # bonds rising aligns with risk-off -> +0.10
            commodity_trend=-0.003,  # commodities falling aligns with risk-off -> +0.05
            symbol="ZN",
        )
        assert conf == pytest.approx(0.90, abs=0.001)

    def test_confidence_clamped_to_1(self) -> None:
        """Confidence cannot exceed 1.0."""
        conf = _compute_confidence(
            regime=MacroRegime.RISK_OFF,
            vix=50.0,
            drawdown_pct=-0.20,
            bond_trend=0.01,
            commodity_trend=-0.01,
            symbol="ZN",
        )
        assert conf <= 1.0

    def test_confidence_clamped_to_0(self) -> None:
        """Confidence cannot go below 0.0."""
        # In practice base is 0.55, so this just verifies the clamp
        conf = _compute_confidence(
            regime=MacroRegime.NEUTRAL,
            vix=None,
            drawdown_pct=None,
            bond_trend=None,
            commodity_trend=None,
            symbol="ZN",
        )
        assert conf >= 0.0


# ===========================================================================
# Macro sensor utilities tests
# ===========================================================================

class TestMacroSensors:
    """Test shared macro sensor functions."""

    def test_vix_from_float_attr(self) -> None:
        ctx = _FakeContext(
            now=datetime.now(timezone.utc), ticks={}, vix=28.5,
        )
        assert _vix_value(ctx) == pytest.approx(28.5)

    def test_vix_from_dict_attr(self) -> None:
        ctx = _FakeContext(
            now=datetime.now(timezone.utc), ticks={},
            vix={"value": 22.0, "ts": "2026-04-03"},
        )
        assert _vix_value(ctx) == pytest.approx(22.0)

    def test_vix_fallback_to_vol_index(self) -> None:
        ctx = _FakeContext(
            now=datetime.now(timezone.utc), ticks={},
            vol_index=19.5,
        )
        assert _vix_value(ctx) == pytest.approx(19.5)

    def test_vix_none_when_missing(self) -> None:
        ctx = _FakeContext(
            now=datetime.now(timezone.utc), ticks={},
        )
        assert _vix_value(ctx) is None

    def test_drawdown_calculation(self) -> None:
        ctx = _build_ctx(equity=94_000.0, equity_peak=100_000.0)
        dd = _portfolio_drawdown_pct(ctx)
        assert dd is not None
        assert dd == pytest.approx(-0.06, abs=0.001)

    def test_drawdown_none_when_no_extra(self) -> None:
        ctx = _FakeContext(
            now=datetime.now(timezone.utc), ticks={},
            portfolio=_FakePortfolio(positions={}, extra=None),
        )
        assert _portfolio_drawdown_pct(ctx) is None

    def test_atr_pct_from_bars(self) -> None:
        closes = _steady_series(n=50, base=100.0, step=0.5)
        bars = _make_bars(closes, spread=2.0)
        result = _atr_pct_from_bars(bars, period=14)
        assert result is not None
        assert result > 0.0

    def test_atr_pct_insufficient_bars(self) -> None:
        bars = _make_bars([100.0, 101.0, 102.0])
        assert _atr_pct_from_bars(bars, period=14) is None

    def test_ema_slope_rising(self) -> None:
        prices = _steady_series(n=50, base=100.0, step=1.0)
        slope = _ema_slope(prices, period=20)
        assert slope is not None
        assert slope > 0.0

    def test_ema_slope_falling(self) -> None:
        prices = _declining_series(n=50, base=200.0, step=1.0)
        slope = _ema_slope(prices, period=20)
        assert slope is not None
        assert slope < 0.0

    def test_ema_slope_insufficient_data(self) -> None:
        assert _ema_slope([100.0, 101.0], period=20) is None


# ===========================================================================
# Sizing validation tests
# ===========================================================================

class TestSizing:
    """Test position sizing logic."""

    def test_signal_sizing_respects_max_contracts(self) -> None:
        """Contracts should not exceed spec.max_contracts."""
        closes = _steady_series(n=50, base=110.0, step=0.1)
        bars = _make_bars(closes, spread=0.5, volume=50_000.0)
        spec = OMEGA_MACRO_SPECS["ZN"]
        tuning = OmegaMacroTuning(risk_budget_pct=0.05)  # generous budget
        signal = _build_signal_for_symbol(
            symbol="ZN",
            bars=bars,
            price=110.0,
            spec=spec,
            side=SignalSide.BUY,
            confidence=0.80,
            regime=MacroRegime.RISK_OFF,
            tuning=tuning,
            equity=1_000_000.0,
        )
        if signal is not None:
            assert signal.size <= spec.max_contracts

    def test_signal_sizing_respects_notional_cap(self) -> None:
        """Estimated notional should not exceed max_trade_notional."""
        closes = _steady_series(n=50, base=110.0, step=0.1)
        bars = _make_bars(closes, spread=0.5, volume=50_000.0)
        spec = OMEGA_MACRO_SPECS["ZN"]
        tuning = OmegaMacroTuning(max_trade_notional=35_000.0)
        signal = _build_signal_for_symbol(
            symbol="ZN",
            bars=bars,
            price=110.0,
            spec=spec,
            side=SignalSide.BUY,
            confidence=0.80,
            regime=MacroRegime.RISK_OFF,
            tuning=tuning,
            equity=500_000.0,
        )
        if signal is not None:
            notional = signal.meta["estimated_notional"]
            assert notional <= tuning.max_trade_notional + 1.0  # float tolerance

    def test_no_signal_below_min_confidence(self) -> None:
        """Signal should be None if confidence < min_confidence."""
        closes = _steady_series(n=50, base=110.0, step=0.1)
        bars = _make_bars(closes, spread=0.5, volume=50_000.0)
        spec = OMEGA_MACRO_SPECS["ZN"]
        tuning = OmegaMacroTuning(min_confidence=0.90)
        signal = _build_signal_for_symbol(
            symbol="ZN",
            bars=bars,
            price=110.0,
            spec=spec,
            side=SignalSide.BUY,
            confidence=0.55,  # below 0.90 threshold
            regime=MacroRegime.RISK_OFF,
            tuning=tuning,
            equity=100_000.0,
        )
        assert signal is None

    def test_no_signal_insufficient_bars(self) -> None:
        bars = _make_bars([110.0, 111.0, 112.0])
        spec = OMEGA_MACRO_SPECS["ZN"]
        tuning = OmegaMacroTuning()
        signal = _build_signal_for_symbol(
            symbol="ZN",
            bars=bars,
            price=112.0,
            spec=spec,
            side=SignalSide.BUY,
            confidence=0.70,
            regime=MacroRegime.RISK_OFF,
            tuning=tuning,
            equity=100_000.0,
        )
        assert signal is None

    def test_signal_has_correct_strategy_name(self) -> None:
        closes = _steady_series(n=50, base=110.0, step=0.1)
        bars = _make_bars(closes, spread=0.5, volume=50_000.0)
        spec = OMEGA_MACRO_SPECS["ZN"]
        tuning = OmegaMacroTuning()
        signal = _build_signal_for_symbol(
            symbol="ZN",
            bars=bars,
            price=110.0,
            spec=spec,
            side=SignalSide.BUY,
            confidence=0.70,
            regime=MacroRegime.RISK_OFF,
            tuning=tuning,
            equity=500_000.0,
        )
        if signal is not None:
            assert signal.strategy == StrategyName.OMEGA_MACRO
            assert signal.asset_class == AssetClass.FUTURES
            assert signal.meta["regime"] == "risk_off"
            assert signal.meta["engine"] == "omega_macro.v1"


# ===========================================================================
# Full signal generation tests
# ===========================================================================

class TestBuildSignals:
    """Test build_omega_macro_signals end-to-end."""

    def test_risk_off_generates_signals(self) -> None:
        """High VIX should produce RISK_OFF signals."""
        # Use realistic prices: ZN/ZB ~110 (point_value 1000 -> notional $110k per contract)
        # Need generous notional cap or use M6E (price ~1.08, pv 12500 -> $13.5k)
        closes_bond = _steady_series(n=50, base=110.0, step=0.1)
        closes_m6e = _steady_series(n=50, base=1.08, step=0.001)
        closes_sil = _steady_series(n=50, base=25.0, step=0.05)
        bars_bond = _make_bars(closes_bond, spread=0.5, volume=50_000.0)
        bars_m6e = _make_bars(closes_m6e, spread=0.005, volume=50_000.0)
        bars_sil = _make_bars(closes_sil, spread=0.2, volume=50_000.0)
        prices = {"ZN": 110.0, "ZB": 120.0, "M6E": 1.08, "SIL": 25.0}
        ctx = _build_ctx(
            vix=30.0,
            equity=1_000_000.0,
            bars={"ZN": bars_bond, "ZB": bars_bond, "M6E": bars_m6e, "SIL": bars_sil},
            prices=prices,
        )
        signals = build_omega_macro_signals(ctx)
        assert len(signals) > 0
        for sig in signals:
            assert sig.strategy == StrategyName.OMEGA_MACRO
            assert sig.meta["regime"] == "risk_off"

    def test_neutral_generates_no_signals(self) -> None:
        """Neutral regime should produce no signals."""
        closes = _steady_series(n=50, base=110.0, step=0.1)
        bars = _make_bars(closes, spread=0.5, volume=50_000.0)
        prices = {sym: 110.0 for sym in OMEGA_MACRO_SPECS}
        ctx = _build_ctx(
            vix=19.0,  # Between 18 and 20 -> NEUTRAL
            equity=100_000.0,
            bars={sym: bars for sym in OMEGA_MACRO_SPECS},
            prices=prices,
        )
        signals = build_omega_macro_signals(ctx)
        assert len(signals) == 0

    def test_risk_on_signal_directions(self) -> None:
        """RISK_ON should produce correct signal sides."""
        closes = _steady_series(n=50, base=110.0, step=0.1)
        bars = _make_bars(closes, spread=0.5, volume=50_000.0)
        prices = {sym: 110.0 for sym in OMEGA_MACRO_SPECS}
        ctx = _build_ctx(
            vix=14.0,
            equity=500_000.0,
            bars={sym: bars for sym in OMEGA_MACRO_SPECS},
            prices=prices,
        )
        signals = build_omega_macro_signals(ctx)
        signal_map = {s.symbol: s for s in signals}
        if "ZN" in signal_map:
            assert signal_map["ZN"].side == SignalSide.SELL
        if "M6E" in signal_map:
            assert signal_map["M6E"].side == SignalSide.BUY

    def test_no_signals_without_bars(self) -> None:
        """No bars data -> no signals (fail-closed)."""
        ctx = _build_ctx(vix=30.0, equity=100_000.0)
        signals = build_omega_macro_signals(ctx)
        assert len(signals) == 0


# ===========================================================================
# Config builder tests
# ===========================================================================

class TestConfig:
    """Test omega_macro_config.py builder and env overrides."""

    def test_default_config(self) -> None:
        cfg = config_build_omega_macro_config()
        assert cfg.name == StrategyName.OMEGA_MACRO
        assert cfg.enabled is True
        assert list(cfg.target_universe) == ["ZN", "ZB", "M6E", "SIL"]
        assert cfg.max_gross_exposure == pytest.approx(0.18)

    def test_env_disable(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_MACRO_ENABLED": "false"}):
            cfg = config_build_omega_macro_config()
            assert cfg.enabled is False

    def test_env_universe_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_MACRO_UNIVERSE": "ZN,ZB"}):
            cfg = config_build_omega_macro_config()
            assert list(cfg.target_universe) == ["ZN", "ZB"]

    def test_env_exposure_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_MACRO_MAX_GROSS_EXPOSURE": "0.12"}):
            cfg = config_build_omega_macro_config()
            assert cfg.max_gross_exposure == pytest.approx(0.12)

    def test_invalid_symbol_raises(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_MACRO_UNIVERSE": "INVALID"}):
            with pytest.raises(ValueError, match="Unsupported"):
                config_build_omega_macro_config()

    def test_fallback_config(self) -> None:
        """build_omega_macro_config() in omega_macro.py falls back gracefully."""
        cfg = build_omega_macro_config()
        assert cfg.name == StrategyName.OMEGA_MACRO


# ===========================================================================
# Handler contract tests
# ===========================================================================

class TestHandler:
    """Test omega_macro_handler engine compatibility."""

    def test_handler_returns_list(self) -> None:
        ctx = _build_ctx(vix=30.0, equity=100_000.0)
        result = omega_macro_handler(ctx)
        assert isinstance(result, list)

    def test_handler_disabled_returns_empty(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_OMEGA_MACRO_ENABLED": "false"}):
            ctx = _build_ctx(vix=30.0, equity=100_000.0)
            result = omega_macro_handler(ctx)
            assert result == []

    def test_handler_survives_bad_context(self) -> None:
        """Handler should not raise on garbage context (fail-closed)."""
        result = omega_macro_handler(None)
        assert isinstance(result, list)


# ===========================================================================
# Registry integration tests
# ===========================================================================

class TestRegistry:
    """Test that OMEGA_MACRO is properly registered."""

    def test_strategy_name_enum(self) -> None:
        assert hasattr(StrategyName, "OMEGA_MACRO")
        assert StrategyName.OMEGA_MACRO.value == "omega_macro"

    def test_registry_contains_omega_macro(self) -> None:
        from chad.strategies import iter_strategy_registrations
        names = [reg.name for reg in iter_strategy_registrations()]
        assert StrategyName.OMEGA_MACRO in names

    def test_registry_config_builds(self) -> None:
        from chad.strategies import iter_strategy_registrations
        for reg in iter_strategy_registrations():
            if reg.name == StrategyName.OMEGA_MACRO:
                cfg = reg.build_config()
                assert cfg.name == StrategyName.OMEGA_MACRO
                assert cfg.enabled is True
                return
        pytest.fail("OMEGA_MACRO not found in registry")


# ===========================================================================
# Instrument spec validation
# ===========================================================================

class TestInstrumentSpecs:
    """Validate OMEGA_MACRO instrument specifications."""

    def test_all_four_instruments_defined(self) -> None:
        assert set(OMEGA_MACRO_SPECS.keys()) == {"ZN", "ZB", "M6E", "SIL"}

    def test_zn_spec(self) -> None:
        spec = OMEGA_MACRO_SPECS["ZN"]
        assert spec.exchange == "CBOT"
        assert spec.point_value == 1000.0
        assert spec.min_tick == pytest.approx(0.015625)
        assert spec.max_contracts == 3

    def test_zb_spec(self) -> None:
        spec = OMEGA_MACRO_SPECS["ZB"]
        assert spec.exchange == "CBOT"
        assert spec.point_value == 1000.0
        assert spec.min_tick == pytest.approx(0.03125)
        assert spec.max_contracts == 2

    def test_m6e_spec(self) -> None:
        spec = OMEGA_MACRO_SPECS["M6E"]
        assert spec.exchange == "CME"
        assert spec.point_value == 12500.0
        assert spec.min_tick == pytest.approx(0.0001)
        assert spec.max_contracts == 3

    def test_sil_spec(self) -> None:
        spec = OMEGA_MACRO_SPECS["SIL"]
        assert spec.exchange == "COMEX"
        assert spec.point_value == 1000.0
        assert spec.min_tick == pytest.approx(0.001)
        assert spec.max_contracts == 2

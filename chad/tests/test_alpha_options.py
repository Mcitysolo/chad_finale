#!/usr/bin/env python3
"""
Tests for ALPHA_OPTIONS vertical spread strategy.

Covers:
- OptionsChain construction and TTL
- Strike selection (bull call, bear put, edge cases)
- DTE calculation
- SpreadSpec generation
- Signal generation (two legs per spread, linked by spread_id)
- Confidence gating
- Sizing validation
- Config and env overrides
- Handler contract
- Registry integration
- OPT lane in types
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

import pytest

from chad.options.chain_provider import OptionsChain
from chad.options.strike_selector import (
    SpreadSpec,
    dte_from_expiry,
    select_vertical_spread,
)
from chad.strategies.alpha_options import (
    AlphaOptionsTuning,
    _build_spread_signals,
    _extract_directional_from_bars,
    build_alpha_options_config,
    build_alpha_options_signals,
    alpha_options_handler,
)
from chad.strategies.alpha_options_config import (
    DEFAULT_ALPHA_OPTIONS_UNIVERSE,
    build_alpha_options_config as config_build_alpha_options_config,
)
from chad.types import AssetClass, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 3, 14, 30, tzinfo=timezone.utc)

# Future expiry ~30 days out
EXPIRY_30D = (NOW + timedelta(days=30)).strftime("%Y%m%d")
EXPIRY_21D = (NOW + timedelta(days=21)).strftime("%Y%m%d")
EXPIRY_45D = (NOW + timedelta(days=45)).strftime("%Y%m%d")
EXPIRY_5D = (NOW + timedelta(days=5)).strftime("%Y%m%d")
EXPIRY_60D = (NOW + timedelta(days=60)).strftime("%Y%m%d")


def _make_chain(
    symbol: str = "SPY",
    price: float = 650.0,
    expirations: Optional[List[str]] = None,
    strikes: Optional[List[float]] = None,
) -> OptionsChain:
    """Create a synthetic options chain for testing."""
    if expirations is None:
        expirations = [EXPIRY_5D, EXPIRY_21D, EXPIRY_30D, EXPIRY_45D, EXPIRY_60D]
    if strikes is None:
        strikes = [float(s) for s in range(int(price - 50), int(price + 51))]
    return OptionsChain(
        symbol=symbol,
        exchange="SMART",
        expirations=expirations,
        strikes=strikes,
        ts_utc=NOW.isoformat().replace("+00:00", "Z"),
        ttl_seconds=3600,
    )


def _make_bars(closes: Sequence[float]) -> List[Dict[str, float]]:
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
class _FakeSignal:
    strategy: str
    symbol: str
    side: str
    confidence: float


@dataclass
class _FakeContext:
    now: datetime
    ticks: Dict[str, Any]
    legend: Any = None
    portfolio: Any = None
    bars: Optional[Dict[str, list]] = None
    strategy_signals: Optional[List[Any]] = None
    prices: Optional[Dict[str, float]] = None


# ===========================================================================
# OptionsChain tests
# ===========================================================================

class TestOptionsChain:

    def test_chain_construction(self) -> None:
        chain = _make_chain()
        assert chain.symbol == "SPY"
        assert len(chain.expirations) == 5
        assert len(chain.strikes) > 50

    def test_chain_not_expired(self) -> None:
        from datetime import datetime, timezone
        fresh_ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        chain = OptionsChain(
            symbol="SPY", exchange="SMART",
            expirations=[EXPIRY_30D],
            strikes=[650.0],
            ts_utc=fresh_ts,
            ttl_seconds=3600,
        )
        assert not chain.is_expired()

    def test_chain_expired(self) -> None:
        chain = OptionsChain(
            symbol="SPY",
            exchange="SMART",
            expirations=[EXPIRY_30D],
            strikes=[650.0],
            ts_utc="2020-01-01T00:00:00Z",
            ttl_seconds=1,
        )
        assert chain.is_expired()

    def test_chain_roundtrip(self) -> None:
        chain = _make_chain()
        d = chain.to_dict()
        restored = OptionsChain.from_dict(d)
        assert restored.symbol == chain.symbol
        assert restored.expirations == chain.expirations
        assert len(restored.strikes) == len(chain.strikes)


# ===========================================================================
# DTE tests
# ===========================================================================

class TestDTE:

    def test_dte_future_expiry(self) -> None:
        dte = dte_from_expiry(EXPIRY_30D)
        assert 28 <= dte <= 32

    def test_dte_past_expiry(self) -> None:
        dte = dte_from_expiry("20200101")
        assert dte < 0

    def test_dte_invalid(self) -> None:
        assert dte_from_expiry("not_a_date") == -1


# ===========================================================================
# Strike selector tests
# ===========================================================================

class TestStrikeSelector:

    def test_bull_call_spread(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(
            chain, current_price=650.0, direction="bullish",
        )
        assert spread is not None
        assert spread.spread_type == "BULL_CALL"
        assert spread.right == "C"
        assert spread.long_strike <= spread.short_strike
        assert spread.max_loss_per_contract > 0
        assert 21 <= spread.dte <= 45

    def test_bear_put_spread(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(
            chain, current_price=650.0, direction="bearish",
        )
        assert spread is not None
        assert spread.spread_type == "BEAR_PUT"
        assert spread.right == "P"
        assert spread.long_strike >= spread.short_strike
        assert spread.max_loss_per_contract > 0

    def test_invalid_direction(self) -> None:
        chain = _make_chain()
        assert select_vertical_spread(chain, 650.0, "sideways") is None

    def test_no_suitable_expiry(self) -> None:
        chain = OptionsChain(
            symbol="SPY", exchange="SMART",
            expirations=[EXPIRY_5D],  # Too close, below dte_min=21
            strikes=[650.0, 660.0, 670.0],
            ts_utc=NOW.isoformat(), ttl_seconds=3600,
        )
        assert select_vertical_spread(chain, 650.0, "bullish") is None

    def test_no_strikes(self) -> None:
        chain = OptionsChain(
            symbol="SPY", exchange="SMART",
            expirations=[EXPIRY_30D],
            strikes=[],
            ts_utc=NOW.isoformat(), ttl_seconds=3600,
        )
        assert select_vertical_spread(chain, 650.0, "bullish") is None

    def test_zero_price(self) -> None:
        chain = _make_chain()
        assert select_vertical_spread(chain, 0.0, "bullish") is None

    def test_spread_width(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(
            chain, 650.0, "bullish", spread_width_pct=0.05,
        )
        assert spread is not None
        width = abs(spread.short_strike - spread.long_strike)
        # ~5% of 650 = 32.5, nearest strikes should be close
        assert 25 <= width <= 40
        assert spread.max_loss_per_contract == width * 100


# ===========================================================================
# Signal generation tests
# ===========================================================================

class TestSignalGeneration:

    def test_spread_signals_come_in_pairs(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        assert spread is not None

        signals = _build_spread_signals(
            spread=spread,
            confidence=0.75,
            equity=500_000.0,
            tuning=AlphaOptionsTuning(),
            now=NOW,
            source_info={"source_strategy": "alpha"},
        )
        assert len(signals) == 2

    def test_legs_share_spread_id(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        signals = _build_spread_signals(
            spread=spread,
            confidence=0.75,
            equity=500_000.0,
            tuning=AlphaOptionsTuning(),
            now=NOW,
            source_info={},
        )
        assert signals[0].meta["spread_id"] == signals[1].meta["spread_id"]

    def test_long_leg_is_buy(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        long_leg = [s for s in signals if s.meta["leg_role"] == "LONG"]
        short_leg = [s for s in signals if s.meta["leg_role"] == "SHORT"]
        assert len(long_leg) == 1
        assert len(short_leg) == 1
        assert long_leg[0].side == SignalSide.BUY
        assert short_leg[0].side == SignalSide.SELL

    def test_signals_have_options_metadata(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        for sig in signals:
            assert sig.strategy == StrategyName.ALPHA_OPTIONS
            assert sig.asset_class == AssetClass.OPTIONS
            assert "expiry" in sig.meta
            assert "strike" in sig.meta
            assert "right" in sig.meta
            assert sig.meta["engine"] == "alpha_options.v1"

    def test_sizing_respects_risk_budget(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish")
        signals = _build_spread_signals(
            spread=spread, confidence=0.75,
            equity=100_000.0,
            tuning=AlphaOptionsTuning(max_risk_per_trade_pct=0.005),
            now=NOW, source_info={},
        )
        if signals:
            contracts = signals[0].meta["contracts"]
            max_loss = spread.max_loss_per_contract * contracts
            # Risk should not exceed 0.5% of equity
            assert max_loss <= 100_000.0 * 0.005 + spread.max_loss_per_contract

    def test_no_signals_on_zero_equity(self) -> None:
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish")
        signals = _build_spread_signals(
            spread=spread, confidence=0.75,
            equity=0.0,
            tuning=AlphaOptionsTuning(),
            now=NOW, source_info={},
        )
        assert len(signals) == 0


# ===========================================================================
# Directional signal extraction tests
# ===========================================================================

class TestDirectionalExtraction:

    def test_bullish_from_bars(self) -> None:
        # Rising series: price > ema12 > ema26
        closes = [600.0 + i * 2.0 for i in range(50)]
        bars = _make_bars(closes)
        ctx = _FakeContext(
            now=NOW, ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={"SPY": bars},
        )
        result = _extract_directional_from_bars(ctx, "SPY", AlphaOptionsTuning())
        assert result is not None
        assert result["direction"] == "bullish"

    def test_bearish_from_bars(self) -> None:
        closes = [700.0 - i * 2.0 for i in range(50)]
        bars = _make_bars(closes)
        ctx = _FakeContext(
            now=NOW, ticks={},
            portfolio=_FakePortfolio(positions={}),
            bars={"SPY": bars},
        )
        result = _extract_directional_from_bars(ctx, "SPY", AlphaOptionsTuning())
        assert result is not None
        assert result["direction"] == "bearish"

    def test_no_signal_from_flat_bars(self) -> None:
        closes = [650.0] * 50
        bars = _make_bars(closes)
        ctx = _FakeContext(
            now=NOW, ticks={},
            bars={"SPY": bars},
        )
        result = _extract_directional_from_bars(ctx, "SPY", AlphaOptionsTuning())
        assert result is None

    def test_insufficient_bars(self) -> None:
        bars = _make_bars([650.0, 651.0])
        ctx = _FakeContext(now=NOW, ticks={}, bars={"SPY": bars})
        result = _extract_directional_from_bars(ctx, "SPY", AlphaOptionsTuning())
        assert result is None


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:

    def test_default_config(self) -> None:
        cfg = config_build_alpha_options_config()
        assert cfg.name == StrategyName.ALPHA_OPTIONS
        assert cfg.enabled is True
        assert list(cfg.target_universe) == ["SPY"]
        assert cfg.max_gross_exposure == pytest.approx(0.15)

    def test_env_disable(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_ALPHA_OPTIONS_ENABLED": "false"}):
            cfg = config_build_alpha_options_config()
            assert cfg.enabled is False

    def test_env_universe_override(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_ALPHA_OPTIONS_UNIVERSE": "SPY,QQQ"}):
            cfg = config_build_alpha_options_config()
            assert list(cfg.target_universe) == ["SPY", "QQQ"]

    def test_invalid_symbol_raises(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_ALPHA_OPTIONS_UNIVERSE": "DOGE"}):
            with pytest.raises(ValueError, match="Unsupported"):
                config_build_alpha_options_config()

    def test_fallback_config(self) -> None:
        cfg = build_alpha_options_config()
        assert cfg.name == StrategyName.ALPHA_OPTIONS


# ===========================================================================
# Handler tests
# ===========================================================================

class TestHandler:

    def test_handler_returns_list(self) -> None:
        ctx = _FakeContext(now=NOW, ticks={}, portfolio=_FakePortfolio(positions={}))
        result = alpha_options_handler(ctx)
        assert isinstance(result, list)

    def test_handler_disabled_returns_empty(self) -> None:
        with mock.patch.dict(os.environ, {"CHAD_ALPHA_OPTIONS_ENABLED": "false"}):
            ctx = _FakeContext(now=NOW, ticks={}, portfolio=_FakePortfolio(positions={}))
            assert alpha_options_handler(ctx) == []

    def test_handler_survives_bad_context(self) -> None:
        assert isinstance(alpha_options_handler(None), list)


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistry:

    def test_strategy_name_enum(self) -> None:
        assert hasattr(StrategyName, "ALPHA_OPTIONS")
        assert StrategyName.ALPHA_OPTIONS.value == "alpha_options"

    def test_asset_class_options(self) -> None:
        assert hasattr(AssetClass, "OPTIONS")
        assert AssetClass.OPTIONS.value == "options"

    def test_registry_contains_alpha_options(self) -> None:
        from chad.strategies import iter_strategy_registrations
        names = [reg.name for reg in iter_strategy_registrations()]
        assert StrategyName.ALPHA_OPTIONS in names

    def test_registry_config_builds(self) -> None:
        from chad.strategies import iter_strategy_registrations
        for reg in iter_strategy_registrations():
            if reg.name == StrategyName.ALPHA_OPTIONS:
                cfg = reg.build_config()
                assert cfg.name == StrategyName.ALPHA_OPTIONS
                return
        pytest.fail("ALPHA_OPTIONS not found in registry")

#!/usr/bin/env python3
"""
Tests for ALPHA_OPTIONS vertical spread strategy.

Covers:
- OptionsChain construction and TTL
- Strike selection (bull call, bear put, edge cases)
- DTE calculation
- SpreadSpec generation
- BAG combo signal generation (one signal per spread)
- Confidence gating
- Sizing validation
- Config and env overrides
- Handler contract
- Registry integration
- Options routing in ibkr_adapter
- _resolve_combo with mocked IB session
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
# BAG signal generation tests
# ===========================================================================

class TestSignalGeneration:

    def test_spread_emits_one_bag_signal(self) -> None:
        """Phase 6b: one BAG signal per spread instead of two leg signals."""
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
        assert len(signals) == 1

    def test_bag_signal_side_is_buy(self) -> None:
        """BAG signal is always BUY — direction encoded in legs."""
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        assert signals[0].side == SignalSide.BUY

    def test_bag_meta_fields_present(self) -> None:
        """All required BAG meta fields must be present."""
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        meta = signals[0].meta
        required_fields = [
            "spread_type", "expiry", "long_strike", "short_strike",
            "long_right", "short_right", "net_debit_estimate",
            "max_loss_per_contract", "spread_id", "sec_type",
        ]
        for f in required_fields:
            assert f in meta, f"Missing meta field: {f}"
        assert meta["sec_type"] == "BAG"
        assert meta["spread_type"] in ("BULL_CALL", "BEAR_PUT")

    def test_bag_limit_price_is_net_debit(self) -> None:
        """limit_price (net_debit_estimate) is set in meta for BAG signal."""
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish", spread_width_pct=0.01)
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        meta = signals[0].meta
        assert meta["net_debit_estimate"] == spread.net_debit_estimate
        assert meta["net_debit_estimate"] > 0

    def test_bag_bull_call_rights(self) -> None:
        """Bull call spread should have C/C for long/short rights."""
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bullish")
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        if signals:
            assert signals[0].meta["long_right"] == "C"
            assert signals[0].meta["short_right"] == "C"

    def test_bag_bear_put_rights(self) -> None:
        """Bear put spread should have P/P for long/short rights."""
        chain = _make_chain(price=650.0)
        spread = select_vertical_spread(chain, 650.0, "bearish")
        signals = _build_spread_signals(
            spread=spread, confidence=0.75, equity=500_000.0,
            tuning=AlphaOptionsTuning(), now=NOW, source_info={},
        )
        if signals:
            assert signals[0].meta["long_right"] == "P"
            assert signals[0].meta["short_right"] == "P"

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


# ===========================================================================
# Options routing gap fix test
# ===========================================================================

class TestOptionsRouting:

    def test_options_asset_class_routes_in_adapter(self) -> None:
        """Verify that asset_class='options' is handled in _intent_from_routed_signal."""
        from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig

        config = IbkrConfig(dry_run=True, validate_contracts_in_dry_run=False)
        adapter = IbkrAdapter(config=config)

        @dataclass
        class _FakeRouted:
            symbol: str = "SPY"
            side: str = "BUY"
            net_size: float = 1.0
            asset_class: str = "options"
            source_strategies: tuple = ("alpha_options",)
            created_at: Any = None
            meta: dict = None

            def __post_init__(self):
                if self.meta is None:
                    self.meta = {
                        "sec_type": "BAG",
                        "expiry": "20260516",
                        "long_strike": 655.0,
                        "short_strike": 660.0,
                        "long_right": "C",
                        "short_right": "C",
                        "net_debit_estimate": 2.50,
                    }

        routed = _FakeRouted()
        intent = adapter._intent_from_routed_signal(routed)
        assert intent.sec_type == "BAG"
        assert intent.asset_class == "options"
        assert intent.exchange == "SMART"
        assert intent.currency == "USD"

    def test_options_routing_sets_limit_price_from_net_debit(self) -> None:
        """Verify net_debit_estimate flows through to meta.limit_price."""
        from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig

        config = IbkrConfig(dry_run=True, validate_contracts_in_dry_run=False)
        adapter = IbkrAdapter(config=config)

        @dataclass
        class _FakeRouted:
            symbol: str = "SPY"
            side: str = "BUY"
            net_size: float = 1.0
            asset_class: str = "options"
            source_strategies: tuple = ("alpha_options",)
            created_at: Any = None
            meta: dict = None

            def __post_init__(self):
                if self.meta is None:
                    self.meta = {
                        "sec_type": "BAG",
                        "net_debit_estimate": 3.25,
                    }

        routed = _FakeRouted()
        intent = adapter._intent_from_routed_signal(routed)
        assert intent.meta["limit_price"] == 3.25


# ===========================================================================
# _resolve_combo tests with mocked IB
# ===========================================================================

class TestResolveCombo:

    def test_resolve_combo_builds_bag_contract(self) -> None:
        """_resolve_combo should produce a BAG contract with two combo legs."""
        from chad.execution.ibkr_adapter import IbkrConfig, _ContractResolver, NormalizedIntent
        from datetime import datetime, timezone

        config = IbkrConfig(dry_run=True)
        resolver = _ContractResolver(config, now_fn=lambda: datetime.now(timezone.utc))

        intent = NormalizedIntent(
            strategy="alpha_options",
            symbol="SPY",
            sec_type="BAG",
            exchange="SMART",
            currency="USD",
            side="BUY",
            order_type="LMT",
            quantity=2.0,
            notional_estimate=0.0,
            asset_class="options",
            source_strategies=("alpha_options",),
            created_at=datetime.now(timezone.utc),
            meta={
                "expiry": "20260516",
                "long_strike": 655.0,
                "short_strike": 660.0,
                "long_right": "C",
                "short_right": "C",
                "spread_type": "BULL_CALL",
            },
        )

        # Mock IB session
        mock_ib = mock.MagicMock()
        # qualifyContracts returns the same objects with conId set
        def fake_qualify(*contracts):
            for i, c in enumerate(contracts):
                c.conId = 100 + i
            return list(contracts)
        mock_ib.qualifyContracts.side_effect = fake_qualify

        resolved = resolver.resolve(mock_ib, intent)
        contract = resolved.contract

        assert contract.secType == "BAG"
        assert contract.symbol == "SPY"
        assert len(contract.comboLegs) == 2

        long_leg = contract.comboLegs[0]
        short_leg = contract.comboLegs[1]
        assert long_leg.action == "BUY"
        assert short_leg.action == "SELL"
        assert long_leg.conId == 100
        assert short_leg.conId == 101
        assert long_leg.ratio == 1
        assert short_leg.ratio == 1

    def test_resolve_combo_without_ib_session(self) -> None:
        """_resolve_combo should work with ib=None (dry run, conIds=0)."""
        from chad.execution.ibkr_adapter import IbkrConfig, _ContractResolver, NormalizedIntent
        from datetime import datetime, timezone

        config = IbkrConfig(dry_run=True)
        resolver = _ContractResolver(config, now_fn=lambda: datetime.now(timezone.utc))

        intent = NormalizedIntent(
            strategy="alpha_options",
            symbol="SPY",
            sec_type="BAG",
            exchange="SMART",
            currency="USD",
            side="BUY",
            order_type="LMT",
            quantity=1.0,
            notional_estimate=0.0,
            asset_class="options",
            source_strategies=("alpha_options",),
            created_at=datetime.now(timezone.utc),
            meta={
                "expiry": "20260516",
                "long_strike": 655.0,
                "short_strike": 660.0,
                "long_right": "C",
                "short_right": "C",
                "spread_type": "BULL_CALL",
            },
        )

        resolved = resolver.resolve(None, intent)
        assert resolved.summary["sec_type"] == "BAG"
        assert resolved.summary["resolution"] == "unqualified"
        assert resolved.summary["long_conId"] == 0
        assert resolved.summary["short_conId"] == 0

    def test_resolve_combo_missing_expiry_raises(self) -> None:
        """Missing expiry should raise ContractResolutionError."""
        from chad.execution.ibkr_adapter import IbkrConfig, _ContractResolver, NormalizedIntent, ContractResolutionError
        from datetime import datetime, timezone

        config = IbkrConfig(dry_run=True)
        resolver = _ContractResolver(config, now_fn=lambda: datetime.now(timezone.utc))

        intent = NormalizedIntent(
            strategy="alpha_options",
            symbol="SPY",
            sec_type="BAG",
            exchange="SMART",
            currency="USD",
            side="BUY",
            order_type="LMT",
            quantity=1.0,
            notional_estimate=0.0,
            asset_class="options",
            source_strategies=("alpha_options",),
            created_at=datetime.now(timezone.utc),
            meta={
                "long_strike": 655.0,
                "short_strike": 660.0,
                "long_right": "C",
                "short_right": "C",
            },
        )

        with pytest.raises(ContractResolutionError, match="expiry"):
            resolver.resolve(None, intent)

    def test_bag_in_whole_unit_sec_types(self) -> None:
        """BAG must be in default_whole_unit_sec_types for integer quantity enforcement."""
        from chad.execution.ibkr_adapter import IbkrConfig
        config = IbkrConfig()
        assert "BAG" in config.default_whole_unit_sec_types

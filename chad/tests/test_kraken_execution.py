"""
chad/tests/test_kraken_execution.py

Tests for the Kraken live execution wiring:

  - split_signals_by_asset_class()  asset-class router
  - _build_kraken_intent_from_routed_signal() intent builder
  - validate_only execution path through KrakenExecutor
  - per-strategy notional cap enforcement
  - paper_kraken mode gate (CHAD_KRAKEN_MODE)
"""

from __future__ import annotations

import os
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from chad.execution.execution_pipeline import (
    _build_kraken_intent_from_routed_signal,
    build_kraken_intents_from_routed_signals,
    normalize_kraken_pair,
    split_signals_by_asset_class,
)
from chad.types import AssetClass, RoutedSignal, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_signal(
    symbol: str,
    side: SignalSide,
    net_size: float,
    asset_class: AssetClass,
    strategy: StrategyName = StrategyName.ALPHA_CRYPTO,
) -> RoutedSignal:
    return RoutedSignal(
        symbol=symbol,
        side=side,
        net_size=net_size,
        source_strategies=(strategy,),
        confidence=0.8,
        asset_class=asset_class,
        created_at=None,
        meta=None,
        price=0.0,
    )


# ---------------------------------------------------------------------------
# 1) Asset-class split
# ---------------------------------------------------------------------------


def test_split_routes_crypto_to_kraken_bucket() -> None:
    sigs = [
        _mk_signal("AAPL", SignalSide.BUY, 10.0, AssetClass.EQUITY, StrategyName.ALPHA),
        _mk_signal("BTC-USD", SignalSide.BUY, 0.01, AssetClass.CRYPTO),
        _mk_signal("MES", SignalSide.SELL, 1.0, AssetClass.FUTURES, StrategyName.ALPHA_FUTURES),
        _mk_signal("ETH-USD", SignalSide.SELL, 0.5, AssetClass.CRYPTO),
    ]
    ibkr_sigs, kraken_sigs = split_signals_by_asset_class(sigs)
    assert {s.symbol for s in ibkr_sigs} == {"AAPL", "MES"}
    assert {s.symbol for s in kraken_sigs} == {"BTC-USD", "ETH-USD"}


def test_split_handles_empty() -> None:
    ibkr, kraken = split_signals_by_asset_class([])
    assert ibkr == []
    assert kraken == []


# ---------------------------------------------------------------------------
# 2) Symbol normalization
# ---------------------------------------------------------------------------


def test_kraken_pair_normalization() -> None:
    assert normalize_kraken_pair("BTC-USD") == "XBT/USD"
    assert normalize_kraken_pair("ETH-USD") == "ETH/USD"
    assert normalize_kraken_pair("SOL-USD") == "SOL/USD"
    assert normalize_kraken_pair("DOGE-USD") is None
    assert normalize_kraken_pair("") is None


# ---------------------------------------------------------------------------
# 3) Intent builder
# ---------------------------------------------------------------------------


def test_intent_builder_btc_basic() -> None:
    sig = _mk_signal("BTC-USD", SignalSide.BUY, 0.01, AssetClass.CRYPTO)
    intent = _build_kraken_intent_from_routed_signal(sig, current_price=80000.0)
    assert intent is not None
    assert intent.pair == "XBT/USD"
    assert intent.side == "buy"
    # Phase-8 Session 8 (E4 Kraken): ordertype is now "limit" when bar data
    # is available (BTC-USD has seeded 1d bars in the repo). Falls back to
    # "market" only when the bar file is absent.
    assert intent.ordertype == "limit"
    assert intent.price is not None
    assert intent.price > 0
    assert intent.strategy == "alpha_crypto"
    # volume = notional/price; notional defaults to net_size*price = 0.01*80000 = 800
    # so volume back-computes to 0.01 (within float tolerance)
    assert intent.volume == pytest.approx(0.01, rel=1e-9)
    assert intent.notional_estimate == pytest.approx(800.0, rel=1e-9)


def test_intent_builder_rejects_below_min_volume() -> None:
    # 0.000001 BTC at $80k = $0.08 notional -> volume way below 0.0001 BTC min
    sig = _mk_signal("BTC-USD", SignalSide.BUY, 0.000001, AssetClass.CRYPTO)
    intent = _build_kraken_intent_from_routed_signal(sig, current_price=80000.0)
    assert intent is None


def test_intent_builder_unsupported_symbol() -> None:
    sig = _mk_signal("DOGE-USD", SignalSide.BUY, 100.0, AssetClass.CRYPTO)
    intent = _build_kraken_intent_from_routed_signal(sig, current_price=0.10)
    assert intent is None


def test_intent_builder_bad_price() -> None:
    sig = _mk_signal("BTC-USD", SignalSide.BUY, 0.01, AssetClass.CRYPTO)
    assert _build_kraken_intent_from_routed_signal(sig, current_price=0.0) is None
    assert _build_kraken_intent_from_routed_signal(sig, current_price=-1.0) is None


def test_intent_builder_cap_clamps_notional() -> None:
    # Raw notional: 1.0 BTC * 80k = 80000. Cap = 500.
    sig = _mk_signal("BTC-USD", SignalSide.BUY, 1.0, AssetClass.CRYPTO)
    intent = _build_kraken_intent_from_routed_signal(
        sig, current_price=80000.0, dynamic_cap_for_crypto=500.0
    )
    assert intent is not None
    assert intent.notional_estimate == pytest.approx(500.0, rel=1e-9)
    assert intent.volume == pytest.approx(500.0 / 80000.0, rel=1e-9)


def test_build_kraken_intents_skips_non_crypto_and_missing_prices() -> None:
    sigs = [
        _mk_signal("AAPL", SignalSide.BUY, 10.0, AssetClass.EQUITY, StrategyName.ALPHA),
        _mk_signal("BTC-USD", SignalSide.BUY, 0.01, AssetClass.CRYPTO),
        _mk_signal("ETH-USD", SignalSide.BUY, 0.1, AssetClass.CRYPTO),
    ]
    intents = build_kraken_intents_from_routed_signals(
        sigs, prices={"BTC-USD": 80000.0}, dynamic_cap_for_crypto=10000.0
    )
    # ETH skipped (no price), AAPL skipped (not CRYPTO)
    assert len(intents) == 1
    assert intents[0].pair == "XBT/USD"


# ---------------------------------------------------------------------------
# 4) Validate-only execution path
# ---------------------------------------------------------------------------


def test_executor_validate_only_does_not_call_live_router() -> None:
    """
    KrakenExecutor.execute_with_risk(intent, live=False) must route through
    the router with validate_only=True and never invoke a live submission.
    """
    from chad.execution.kraken_executor import KrakenExecutor, StrategyTradeIntent

    fake_router = MagicMock()
    # router.execute returns a TradeResponse-like object
    fake_resp = MagicMock()
    fake_resp.txids = []
    fake_resp.raw = {"validate": True}
    fake_router.execute.return_value = fake_resp

    executor = KrakenExecutor(router=fake_router)

    intent = StrategyTradeIntent(
        strategy="alpha_crypto",
        pair="XBT/USD",
        side="buy",
        ordertype="market",
        volume=0.001,
        notional_estimate=80.0,
    )
    risk_result, resp = executor.execute_with_risk(intent=intent, live=False)
    assert risk_result.allowed is True
    assert resp is fake_resp
    # Inspect the TradeRequest passed to the router
    assert fake_router.execute.called
    req = fake_router.execute.call_args.args[0]
    assert getattr(req, "validate_only") is True


# ---------------------------------------------------------------------------
# 5) Cap enforcement at executor level
# ---------------------------------------------------------------------------


def test_executor_blocks_intent_over_cap(tmp_path) -> None:
    """
    KrakenExecutor must block an intent whose notional exceeds the
    runtime cap and not call the router at all.
    """
    from chad.execution.kraken_executor import KrakenExecutor, StrategyTradeIntent
    import json as _json

    caps_path = tmp_path / "dynamic_caps.json"
    caps_path.write_text(_json.dumps({"strategy_caps": {"crypto": 100.0}}))

    fake_router = MagicMock()
    executor = KrakenExecutor(router=fake_router, caps_path=caps_path)

    intent = StrategyTradeIntent(
        strategy="alpha_crypto",
        pair="XBT/USD",
        side="buy",
        ordertype="market",
        volume=0.01,
        notional_estimate=10_000.0,  # well above $100 cap
    )
    risk_result, resp = executor.execute_with_risk(intent=intent, live=False)
    assert risk_result.allowed is False
    assert resp is None
    fake_router.execute.assert_not_called()


# ---------------------------------------------------------------------------
# 6) paper_kraken mode gate
# ---------------------------------------------------------------------------


def test_paper_kraken_mode_resolution(monkeypatch) -> None:
    from chad.core.kraken_execution import resolve_kraken_mode

    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    monkeypatch.delenv("CHAD_EXECUTION_MODE", raising=False)
    assert resolve_kraken_mode() == "off"

    monkeypatch.setenv("CHAD_KRAKEN_MODE", "paper_kraken")
    assert resolve_kraken_mode() == "paper_kraken"

    monkeypatch.setenv("CHAD_KRAKEN_MODE", "live")
    assert resolve_kraken_mode() == "live"

    monkeypatch.setenv("CHAD_KRAKEN_MODE", "garbage")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "live")
    assert resolve_kraken_mode() == "live"

    monkeypatch.setenv("CHAD_EXECUTION_MODE", "dry_run")
    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    assert resolve_kraken_mode() == "off"


def test_execute_kraken_intents_gate_denied(monkeypatch, caplog) -> None:
    """When kraken_enabled gate is False, execution is a no-op."""
    import logging as _logging
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: False)
    fake_intent = MagicMock()
    fake_intent.pair = "XBT/USD"
    logger = _logging.getLogger("test.kraken")
    with caplog.at_level(_logging.INFO, logger="test.kraken"):
        ke.execute_kraken_intents(logger, [fake_intent])
    assert any("KRAKEN_GATE_DENIED" in r.getMessage() for r in caplog.records)


def test_execute_kraken_intents_mode_off(monkeypatch, caplog) -> None:
    """When mode resolves to 'off', execution is a no-op even if gate is open."""
    import logging as _logging
    from chad.core import kraken_execution as ke

    monkeypatch.setattr(ke, "is_kraken_gate_enabled", lambda: True)
    monkeypatch.setattr(ke, "resolve_kraken_mode", lambda: "off")
    fake_intent = MagicMock()
    fake_intent.pair = "XBT/USD"
    logger = _logging.getLogger("test.kraken2")
    with caplog.at_level(_logging.INFO, logger="test.kraken2"):
        ke.execute_kraken_intents(logger, [fake_intent])
    assert any("KRAKEN_MODE_OFF" in r.getMessage() for r in caplog.records)

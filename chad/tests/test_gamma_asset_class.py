"""Regression tests for chad.strategies.gamma — AssetClass enum usage.

Background
----------
Live-loop logs reported:
    Strategy handler gamma raised AttributeError:
    type object 'AssetClass' has no attribute 'STOCK'

The canonical enum (chad.types.AssetClass) defines EQUITY/ETF/CRYPTO/FOREX/
CASH/FUTURES/OPTIONS — there is no STOCK member. Gamma's _asset_class()
fallback returned AssetClass.STOCK, which crashed the handler whenever a
non-ETF symbol was processed. This module locks in the EQUITY fallback and
proves gamma_handler does not raise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import pytest

from chad.strategies import gamma as gamma_mod
from chad.strategies.gamma import (
    GammaParams,
    _asset_class,
    _propose_for_symbol,
    build_gamma_config,
    gamma_handler,
)
from chad.types import (
    AssetClass,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyName,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(n: int = 200, start: float = 100.0, step: float = 0.25) -> List[Dict[str, float]]:
    """Deterministic upward-drifting OHLC bars to satisfy gamma's history gate."""
    out: List[Dict[str, float]] = []
    px = start
    for _ in range(n):
        o = px
        c = px + step
        h = max(o, c) + 0.10
        l = min(o, c) - 0.10
        out.append({"open": o, "high": h, "low": l, "close": c, "volume": 1_000_000})
        px = c
    return out


def _make_ctx(
    *,
    symbol: str,
    price: float = 150.0,
    bars: List[Dict[str, float]] | None = None,
    cash: float = 1_000_000.0,
    positions: Dict[str, Position] | None = None,
) -> MarketContext:
    now = datetime.now(timezone.utc)
    tick = MarketTick(
        symbol=symbol,
        price=price,
        size=100.0,
        exchange=None,
        timestamp=now,
        source="test",
    )
    portfolio = PortfolioSnapshot(
        timestamp=now,
        cash=cash,
        positions=positions or {},
    )
    return MarketContext(
        now=now,
        ticks={symbol: tick},
        legend=None,
        portfolio=portfolio,
        bars={symbol: bars or _make_bars()},
    )


@pytest.fixture(autouse=True)
def _reset_gamma_state():
    gamma_mod._STATE.pos.clear()
    yield
    gamma_mod._STATE.pos.clear()


# ---------------------------------------------------------------------------
# Asset-class enum regression
# ---------------------------------------------------------------------------

def test_asset_class_enum_has_no_STOCK_member() -> None:
    """Documents the canonical surface: STOCK is *not* a valid AssetClass."""
    assert not hasattr(AssetClass, "STOCK")


def test_gamma_asset_class_helper_etf_symbols_return_etf() -> None:
    for sym in ("SPY", "QQQ", "IWM", "TLT", "GLD"):
        assert _asset_class(sym) is AssetClass.ETF


def test_gamma_asset_class_helper_non_etf_falls_back_to_equity() -> None:
    for sym in ("AAPL", "MSFT", "NVDA", "TSLA"):
        ac = _asset_class(sym)
        assert ac is AssetClass.EQUITY
        # The bug: previous code returned AssetClass.STOCK which doesn't exist.
        assert ac.value in {a.value for a in AssetClass}


# ---------------------------------------------------------------------------
# Handler does not raise + emitted signals carry valid AssetClass
# ---------------------------------------------------------------------------

def test_gamma_handler_does_not_raise_for_equity_symbol(monkeypatch) -> None:
    """gamma_handler must not raise AttributeError on a non-ETF symbol."""
    monkeypatch.setattr(gamma_mod, "get_trade_universe", lambda: ["AAPL"])
    ctx = _make_ctx(symbol="AAPL", price=150.0)
    # Should not raise — even if no signal is emitted, handler must complete.
    out = list(gamma_handler(ctx))
    assert isinstance(out, list)


def test_gamma_handler_does_not_raise_for_etf_symbol(monkeypatch) -> None:
    monkeypatch.setattr(gamma_mod, "get_trade_universe", lambda: ["SPY"])
    ctx = _make_ctx(symbol="SPY", price=450.0)
    out = list(gamma_handler(ctx))
    assert isinstance(out, list)


def test_gamma_emitted_signals_use_valid_asset_class(monkeypatch) -> None:
    """Any TradeSignal gamma emits must carry a real AssetClass member."""
    valid = {a for a in AssetClass}
    # Use a config that should fire a trend-momentum entry: drifting bars + no position.
    bars = _make_bars(n=200, start=100.0, step=0.30)
    last_close = bars[-1]["close"]
    # Set price meaningfully above fast EMA to trigger momentum entry.
    ctx = _make_ctx(symbol="AAPL", price=last_close + 5.0, bars=bars)
    monkeypatch.setattr(gamma_mod, "get_trade_universe", lambda: ["AAPL"])
    signals = list(gamma_handler(ctx))
    # Whether a signal fires depends on regime classification — but if any
    # do fire, every signal must carry a valid AssetClass.
    for sig in signals:
        assert sig.asset_class in valid
        assert sig.strategy == StrategyName.GAMMA


def test_gamma_exit_signal_no_history_path_uses_equity_for_non_etf(monkeypatch) -> None:
    """The 'no_history_exit_only' branch hits _asset_class — was the original crash site."""
    now = datetime.now(timezone.utc)
    pos = Position(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        quantity=10.0,
        avg_price=140.0,
    )
    tick = MarketTick(
        symbol="AAPL", price=150.0, size=100.0, exchange=None, timestamp=now, source="test",
    )
    ctx = MarketContext(
        now=now,
        ticks={"AAPL": tick},
        legend=None,
        portfolio=PortfolioSnapshot(timestamp=now, cash=1_000_000.0, positions={"AAPL": pos}),
        bars={"AAPL": []},  # empty bars triggers the no-history branch
    )
    out = list(_propose_for_symbol("AAPL", ctx, GammaParams()))
    assert len(out) == 1
    sig = out[0]
    assert sig.side == SignalSide.SELL
    assert sig.asset_class is AssetClass.EQUITY
    assert sig.meta.get("reason") == "no_history_exit_only"


# ---------------------------------------------------------------------------
# Config sanity — gamma is still a registered strategy with build_gamma_config
# ---------------------------------------------------------------------------

def test_build_gamma_config_returns_strategy_config() -> None:
    cfg = build_gamma_config()
    assert cfg.name == StrategyName.GAMMA
    assert cfg.enabled is True

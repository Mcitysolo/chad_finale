"""W2B-5 — the D4 double-exit guardrail: the ACTIVE exit overlay stays the SOLE
equity/ETF exit authority.

Injecting positions (mode ``on``) makes strategies position-aware. That is safe
for BUY-suppression (defects b, c) but ALSO makes their ``qty>0``-gated native
exits reachable — unclamped equity/ETF SELLs that would race the overlay's
reduce-only closes (oversell / short-flip, INCIDENT-0713). ``filter_overlay_owned_exits``
drops exactly those, and ONLY in ON mode, so:

  * OFF / shadow  -> byte-identical (the guardrail is inert);
  * ON            -> equity/ETF strategy SELLs removed; BUYs and non-equity/ETF
                     SELLs (futures/crypto/options) untouched.

Hermetic: pure-filter matrix + the REAL gamma native exit + a router-chokepoint
integration. No live tree read.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from datetime import datetime, timezone
from types import SimpleNamespace

from chad.core import context_positions as cp
from chad.strategies import gamma as gamma_mod
from chad.strategies.gamma import GammaParams, gamma_handler
from chad.types import (
    AssetClass,
    MarketContext,
    MarketTick,
    PortfolioSnapshot,
    Position,
    SignalSide,
    StrategyName,
)

_LOG = logging.getLogger("test")

# A lightweight signal that carries the two fields the filter reads plus an
# explicit asset_class (as real TradeSignals do).
_Sig = namedtuple("Sig", "strategy symbol side asset_class")


def _sig(strategy, symbol, side, asset_class):
    return _Sig(strategy, symbol, side, asset_class)


# --------------------------------------------------------------------------- #
# 1) pure filter — inert unless ON
# --------------------------------------------------------------------------- #

def test_filter_inert_when_off_and_shadow():
    sells = [_sig("gamma", "AAPL", SignalSide.SELL, AssetClass.EQUITY),
             _sig("gamma", "SPY", SignalSide.SELL, AssetClass.ETF)]
    for mode in ("off", "shadow"):
        kept, dropped = cp.filter_overlay_owned_exits(sells, mode=mode)
        assert kept == sells and dropped == []          # unchanged -> byte-identical


def test_filter_on_drops_equity_and_etf_sells_only():
    sigs = [
        _sig("gamma", "AAPL", SignalSide.SELL, AssetClass.EQUITY),   # drop
        _sig("omega", "SH", SignalSide.SELL, AssetClass.ETF),        # drop
        _sig("gamma", "MSFT", SignalSide.BUY, AssetClass.EQUITY),    # keep (BUY)
        _sig("beta", "SPY", SignalSide.BUY, AssetClass.ETF),         # keep (BUY)
        _sig("alpha_futures", "MES", SignalSide.SELL, AssetClass.FUTURES),  # keep (futures)
        _sig("alpha_crypto", "BTC-USD", SignalSide.SELL, AssetClass.CRYPTO),  # keep (crypto)
    ]
    kept, dropped = cp.filter_overlay_owned_exits(sigs, mode="on")
    assert {(s.symbol) for s in dropped} == {"AAPL", "SH"}
    assert {(s.symbol) for s in kept} == {"MSFT", "SPY", "MES", "BTC-USD"}
    # order preserved among kept
    assert [s.symbol for s in kept] == ["MSFT", "SPY", "MES", "BTC-USD"]


def test_is_overlay_owned_exit_truth_table():
    assert cp.is_overlay_owned_exit(_sig("g", "AAPL", SignalSide.SELL, AssetClass.EQUITY)) is True
    assert cp.is_overlay_owned_exit(_sig("g", "SPY", SignalSide.SELL, AssetClass.ETF)) is True
    assert cp.is_overlay_owned_exit(_sig("g", "AAPL", SignalSide.BUY, AssetClass.EQUITY)) is False
    assert cp.is_overlay_owned_exit(_sig("g", "MES", SignalSide.SELL, AssetClass.FUTURES)) is False
    assert cp.is_overlay_owned_exit(_sig("g", "BTC", SignalSide.SELL, AssetClass.CRYPTO)) is False


# --------------------------------------------------------------------------- #
# 2) classification safety — the signal's OWN asset_class wins; symbol fallback
# --------------------------------------------------------------------------- #

def test_prefers_signal_own_asset_class_over_symbol():
    # A futures SELL whose *symbol* the equity classifier would read as EQUITY
    # (ZQ is not in the overlay's futures allow-list) must be KEPT because the
    # signal declares FUTURES. This is the mis-drop the design guards against.
    s = _sig("alpha_futures", "ZQ", SignalSide.SELL, AssetClass.FUTURES)
    assert cp.is_overlay_owned_exit(s) is False
    kept, dropped = cp.filter_overlay_owned_exits([s], mode="on")
    assert dropped == [] and kept == [s]


def test_symbol_classifier_fallback_when_no_asset_class():
    # No asset_class on the signal -> fall back to the D6 symbol classifier.
    _Bare = namedtuple("Bare", "symbol side")
    spy = _Bare("SPY", SignalSide.SELL)   # ETF by symbol -> drop
    mes = _Bare("MES", SignalSide.SELL)   # futures by symbol -> keep
    aapl = _Bare("AAPL", SignalSide.SELL)  # equity by symbol -> drop
    kept, dropped = cp.filter_overlay_owned_exits([spy, mes, aapl], mode="on")
    assert {s.symbol for s in dropped} == {"SPY", "AAPL"}
    assert [s.symbol for s in kept] == ["MES"]


# --------------------------------------------------------------------------- #
# 3) the REAL gamma native exit — the executed danger — is dropped in ON only
# --------------------------------------------------------------------------- #

def _downtrend_bars(n=200, start=500.0, step=-1.0):
    out = []
    px = start
    for _ in range(n):
        o = px
        c = px + step
        out.append({"open": o, "high": max(o, c) + 0.1, "low": min(o, c) - 0.1,
                    "close": c, "volume": 5_000_000})
        px = c
    return out


def _gamma_native_exit_signals():
    """Small held equity leg (AAPL 14, < the 50 cap) in a downtrend -> gamma's
    trend_break_exit fires an equity SELL (the native exit that becomes reachable
    once the book is injected)."""
    bars = _downtrend_bars()
    now = datetime.now(timezone.utc)
    ctx = MarketContext(
        now=now,
        ticks={"AAPL": MarketTick(symbol="AAPL", price=bars[-1]["close"], size=100.0,
                                  exchange=None, timestamp=now, source="test")},
        legend=None,
        portfolio=PortfolioSnapshot(timestamp=now, cash=1_000_000.0,
                                    positions={"AAPL": Position("AAPL", AssetClass.EQUITY, 14.0, 600.0)}),
        bars={"AAPL": bars},
    )
    gamma_mod.get_trade_universe = lambda: ["AAPL"]  # module attr; restored below
    gamma_mod._STATE.pos.clear()
    try:
        return list(gamma_handler(ctx, GammaParams(max_position_units=50.0)))
    finally:
        gamma_mod._STATE.pos.clear()


def test_real_gamma_native_exit_present_then_dropped_on(monkeypatch):
    monkeypatch.setattr(gamma_mod, "get_trade_universe", lambda: ["AAPL"])
    sigs = _gamma_native_exit_signals()
    # sanity: the danger exists — gamma DID emit an equity SELL exit
    assert any(s.side == SignalSide.SELL and s.asset_class == AssetClass.EQUITY for s in sigs)

    off_kept, off_dropped = cp.filter_overlay_owned_exits(sigs, mode="off")
    assert off_kept == sigs and off_dropped == []            # OFF inert: exit survives (today)

    on_kept, on_dropped = cp.filter_overlay_owned_exits(sigs, mode="on")
    assert all(s.side != SignalSide.SELL for s in on_kept)   # ON: no equity/ETF SELL survives
    assert sigs and len(on_dropped) >= 1                     # the native exit was neutralized


# --------------------------------------------------------------------------- #
# 4) router chokepoint integration — ON drops the strategy equity SELL and
#    heartbeats the count; BUY-suppression (beta) is untouched
# --------------------------------------------------------------------------- #

class _FakeBuilder:
    def build(self, *, current_positions=None, current_cash=0.0):
        return SimpleNamespace(
            context=SimpleNamespace(portfolio=SimpleNamespace(positions=dict(current_positions or {}))),
            prices={}, current_symbol_notional={}, current_total_notional=0.0)


def _fake_registry():
    def gamma_handler_(ctx):   # a native equity exit SELL + a fresh equity BUY
        return [_sig("gamma", "AAPL", SignalSide.SELL, AssetClass.EQUITY),
                _sig("gamma", "NVDA", SignalSide.BUY, AssetClass.EQUITY)]

    def beta_handler_(ctx):    # BUY-only strategy (no SELL leg) — must pass through intact
        return [_sig("beta", "MSFT", SignalSide.BUY, AssetClass.EQUITY)]

    return [SimpleNamespace(name=SimpleNamespace(value="gamma"), handler=gamma_handler_),
            SimpleNamespace(name=SimpleNamespace(value="beta"), handler=beta_handler_)]


def test_router_on_drops_strategy_equity_sell(monkeypatch):
    import chad.strategies as strategies
    import chad.core.live_execution_router as router
    import chad.core.ctx_positions_shadow as shadow

    _UNH = Position("UNH", AssetClass.EQUITY, 228.0, 424.98)
    monkeypatch.setattr("chad.utils.context_builder.ContextBuilder", _FakeBuilder)
    monkeypatch.setattr(cp, "load_context_positions",
                        lambda *a, **k: cp.PositionsView(cp.STATUS_KNOWN, {"UNH": _UNH}, "fresh", {"n_injected": 1}))
    monkeypatch.setattr(strategies, "iter_strategy_registrations", _fake_registry)
    monkeypatch.setattr(router, "load_allocation_weights", lambda: {})

    captured = {}
    monkeypatch.setattr(shadow, "record_cycle", lambda **k: captured.update(k))

    monkeypatch.setenv("CHAD_CTX_POSITIONS", "on")
    available, _weights = router._build_available_signals(_LOG)

    keys = {(n, s.symbol, s.side) for n, sigs in available.items() for s in sigs}
    assert ("gamma", "AAPL", SignalSide.SELL) not in keys    # native equity exit dropped
    assert ("gamma", "NVDA", SignalSide.BUY) in keys         # entries untouched
    assert ("beta", "MSFT", SignalSide.BUY) in keys          # BUY-suppression strategy intact
    assert captured.get("exits_filtered") == 1               # heartbeat carries the count


def test_router_off_keeps_the_sell_byte_identical(monkeypatch):
    import chad.strategies as strategies
    import chad.core.live_execution_router as router
    import chad.core.ctx_positions_shadow as shadow

    monkeypatch.setattr("chad.utils.context_builder.ContextBuilder", _FakeBuilder)
    monkeypatch.setattr(cp, "load_context_positions",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader consulted in OFF")))
    monkeypatch.setattr(strategies, "iter_strategy_registrations", _fake_registry)
    monkeypatch.setattr(router, "load_allocation_weights", lambda: {})
    monkeypatch.setattr(shadow, "record_cycle", lambda **k: None)

    monkeypatch.setenv("CHAD_CTX_POSITIONS", "off")
    available, _ = router._build_available_signals(_LOG)
    keys = {(n, s.symbol, s.side) for n, sigs in available.items() for s in sigs}
    assert ("gamma", "AAPL", SignalSide.SELL) in keys        # OFF: the SELL survives (guardrail inert)

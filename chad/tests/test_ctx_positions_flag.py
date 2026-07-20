"""W2B-2 — CHAD_CTX_POSITIONS tri-state flag + OFF-inert wiring.

Hermetic: no live tree read. ContextBuilder / loader / registry are all faked.
Proves (D7): OFF is byte-identical (same emitted-signal set as legacy), ON
characterizes the diff (the gamma re-buy key disappears), UNKNOWN idles.
"""
from __future__ import annotations

import logging
from collections import namedtuple
from types import SimpleNamespace

import pytest

from chad.types import AssetClass, Position
from chad.core import context_positions as cp

_LOG = logging.getLogger("test")
_Sig = namedtuple("Sig", "strategy symbol side")


# --------------------------------------------------------------------------- #
# 1) parser
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("val,expected", [
    (None, "off"), ("", "off"), ("0", "off"), ("off", "off"), ("OFF", "off"),
    ("shadow", "shadow"), ("  shadow ", "shadow"),
    ("on", "on"), ("ON", "on"),
    ("garbage", "off"), ("1", "off"), ("true", "off"),
])
def test_resolve_mode_parser(val, expected):
    env = {} if val is None else {"CHAD_CTX_POSITIONS": val}
    assert cp.resolve_ctx_positions_mode(env) == expected


# --------------------------------------------------------------------------- #
# 2) build_cycle_context helper — OFF is inert; ON injects; UNKNOWN idles
# --------------------------------------------------------------------------- #

class _FakeBuilder:
    def __init__(self):
        self.calls = []

    def build(self, *, current_positions=None, current_cash=0.0):
        self.calls.append(dict(current_positions or {}) if current_positions is not None else None)
        return SimpleNamespace(
            context=SimpleNamespace(portfolio=SimpleNamespace(positions=dict(current_positions or {}))),
            prices={}, current_symbol_notional={}, current_total_notional=0.0,
        )


_UNH = Position(symbol="UNH", asset_class=AssetClass.EQUITY, quantity=228.0, avg_price=424.98)


def test_off_is_inert_loader_not_consulted(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(cp, "load_context_positions", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader called in OFF")))
    fake = _FakeBuilder()
    result, view, mode = cp.build_cycle_context(builder=fake, env={"CHAD_CTX_POSITIONS": "off"})
    assert mode == "off"
    assert view is None
    assert fake.calls == [None]                # build() called with NO current_positions
    assert result.context.portfolio.positions == {}


def test_on_known_injects_positions(monkeypatch):
    monkeypatch.setattr(cp, "load_context_positions",
                        lambda *a, **k: cp.PositionsView(cp.STATUS_KNOWN, {"UNH": _UNH}, "fresh"))
    fake = _FakeBuilder()
    result, view, mode = cp.build_cycle_context(builder=fake, env={"CHAD_CTX_POSITIONS": "on"})
    assert mode == "on"
    assert view.known
    assert fake.calls == [{"UNH": _UNH}]        # injected
    assert result.context.portfolio.positions["UNH"].quantity == 228.0


def test_on_unknown_idles(monkeypatch):
    monkeypatch.setattr(cp, "load_context_positions",
                        lambda *a, **k: cp.PositionsView(cp.STATUS_UNKNOWN, {}, "snapshot_stale"))
    fake = _FakeBuilder()
    result, view, mode = cp.build_cycle_context(builder=fake, env={"CHAD_CTX_POSITIONS": "on"}, logger=_LOG)
    assert mode == "on"
    assert result is None                        # caller must idle (D3)
    assert view.status == cp.STATUS_UNKNOWN
    assert fake.calls == []                       # build() never called on UNKNOWN


def test_shadow_acts_on_empty_book_but_exposes_view(monkeypatch):
    monkeypatch.setattr(cp, "load_context_positions",
                        lambda *a, **k: cp.PositionsView(cp.STATUS_KNOWN, {"UNH": _UNH}, "fresh"))
    fake = _FakeBuilder()
    result, view, mode = cp.build_cycle_context(builder=fake, env={"CHAD_CTX_POSITIONS": "shadow"})
    assert mode == "shadow"
    assert fake.calls == [None]                   # acts on the EMPTY book (no behaviour change)
    assert result.context.portfolio.positions == {}
    assert view.known and "UNH" in view.positions  # view exposed for the shadow recorder


# --------------------------------------------------------------------------- #
# 3) router-level D7 proof — OFF set-equality, ON characterized diff, idle
# --------------------------------------------------------------------------- #

def _fake_registry():
    def gamma_handler(ctx):
        pos = ctx.portfolio.positions.get("UNH")
        if pos is not None and float(getattr(pos, "quantity", 0.0)) > 0.0:
            return []                             # already held -> no re-buy (the fix)
        return [_Sig("gamma", "UNH", "BUY")]      # empty book -> the churn re-buy

    def alpha_handler(ctx):                        # position-blind constant
        return [_Sig("alpha", "SPY", "BUY")]

    return [
        SimpleNamespace(name=SimpleNamespace(value="gamma"), handler=gamma_handler),
        SimpleNamespace(name=SimpleNamespace(value="alpha"), handler=alpha_handler),
    ]


def _keyset(available):
    return {(n, s.symbol, s.side) for n, sigs in available.items() for s in sigs}


def _wire(monkeypatch, *, view):
    import chad.strategies as strategies
    import chad.core.live_execution_router as router
    monkeypatch.setattr("chad.utils.context_builder.ContextBuilder", _FakeBuilder)
    monkeypatch.setattr(cp, "load_context_positions", lambda *a, **k: view)
    monkeypatch.setattr(strategies, "iter_strategy_registrations", _fake_registry)
    monkeypatch.setattr(router, "load_allocation_weights", lambda: {})
    return router


def test_router_off_equals_unset_and_baseline(monkeypatch):
    router = _wire(monkeypatch, view=cp.PositionsView(cp.STATUS_KNOWN, {"UNH": _UNH}, "fresh"))

    monkeypatch.delenv("CHAD_CTX_POSITIONS", raising=False)
    unset, _ = router._build_available_signals(_LOG)

    monkeypatch.setenv("CHAD_CTX_POSITIONS", "off")
    off, _ = router._build_available_signals(_LOG)

    assert _keyset(off) == _keyset(unset)         # OFF byte-identical to legacy
    assert ("gamma", "UNH", "BUY") in _keyset(off)  # the churn re-buy present when OFF


def test_router_on_removes_the_gamma_rebuy(monkeypatch):
    router = _wire(monkeypatch, view=cp.PositionsView(cp.STATUS_KNOWN, {"UNH": _UNH}, "fresh"))

    monkeypatch.delenv("CHAD_CTX_POSITIONS", raising=False)
    off, _ = router._build_available_signals(_LOG)

    monkeypatch.setenv("CHAD_CTX_POSITIONS", "on")
    on, _ = router._build_available_signals(_LOG)

    # D7 characterized diff: ON removes exactly the gamma re-buy; nothing added.
    assert _keyset(off) - _keyset(on) == {("gamma", "UNH", "BUY")}
    assert _keyset(on) - _keyset(off) == set()
    assert ("alpha", "SPY", "BUY") in _keyset(on)  # position-blind strategy unchanged


def test_router_on_unknown_idles(monkeypatch):
    router = _wire(monkeypatch, view=cp.PositionsView(cp.STATUS_UNKNOWN, {}, "snapshot_stale"))
    monkeypatch.setenv("CHAD_CTX_POSITIONS", "on")
    available, weights = router._build_available_signals(_LOG)
    assert available == {}                         # idle: no signals on a false-empty book

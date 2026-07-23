"""W4B-1 (J16): exit-advice recorder — dropped D4 urges become typed records.

Pins:
  - off mode: no rows, no evidence file; state heartbeat only when a path is
    explicitly injected (G3C-HF: default paths are skipped under pytest).
  - record mode: one exit_advice.v1 row per dropped urge with the full
    forensic payload (strategy, symbol, side, size, confidence, reason,
    excluded, position_open).
  - excluded symbols are recorded WITH excluded=true — never silently dropped
    (observability), and never consumable later (W4B-3 pins that side).
  - RoutedSignal-shaped objects (source_strategies, net_size) are handled.
  - the recorder never raises (bad objects degrade, writes are best-effort).
  - router site integration: dropped urges reach the recorder; OFF stays
    byte-identical (blast-radius list-equality unchanged).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from chad.core import exit_advice as ea
from chad.types import AssetClass, Position, SignalSide, StrategyName, TradeSignal

_LOG = logging.getLogger("test_w4b_exit_advice")
_NOW = datetime(2026, 7, 23, 14, 0, 0, tzinfo=timezone.utc)


def _sig(strategy=StrategyName.GAMMA, symbol="PSQ", side=SignalSide.SELL,
         size=5.0, confidence=0.7, asset_class=AssetClass.ETF, meta=None):
    return TradeSignal(
        strategy=strategy, symbol=symbol, side=side, size=size,
        confidence=confidence, asset_class=asset_class,
        meta=dict(meta or {"reason": "trend_break_exit"}),
    )


class _View:
    """Minimal PositionsView-shaped object."""
    def __init__(self, positions):
        self.known = True
        self.status = "KNOWN"
        self.positions = positions
        self.evidence = {"n_injected": len(positions)}


# --------------------------------------------------------------------------- #
# mode resolution
# --------------------------------------------------------------------------- #

def test_mode_default_off(monkeypatch):
    monkeypatch.delenv(ea.MODE_ENV, raising=False)
    assert ea.resolve_exit_advice_mode() == "off"


@pytest.mark.parametrize("raw,expect", [
    ("off", "off"), ("record", "record"), ("consume", "consume"),
    ("RECORD", "record"), (" garbage ", "off"), ("", "off"), ("on", "off"),
])
def test_mode_resolution(monkeypatch, raw, expect):
    monkeypatch.setenv(ea.MODE_ENV, raw)
    assert ea.resolve_exit_advice_mode() == expect


# --------------------------------------------------------------------------- #
# off mode
# --------------------------------------------------------------------------- #

def test_off_mode_records_nothing_but_heartbeats(tmp_path, monkeypatch):
    monkeypatch.setenv(ea.MODE_ENV, "off")
    state = tmp_path / "exit_advice_state.json"
    evidence = tmp_path / "evidence"
    n = ea.record_dropped_urges(
        [_sig()], site="router", evidence_dir=evidence, state_path=state,
        logger=_LOG, now=_NOW,
    )
    assert n == 0
    assert not evidence.exists()                      # no rows in off
    payload = json.loads(state.read_text())
    assert payload["schema_version"] == "exit_advice_state.v1"
    assert payload["mode"] == "off"                    # off-liveness stamped
    assert payload["ttl_seconds"] == ea.STATE_TTL_SECONDS
    assert payload["today"]["recorded"] == 0


def test_pytest_default_paths_are_skipped(monkeypatch):
    """G3C-HF: with no explicit paths under pytest, nothing touches disk."""
    monkeypatch.setenv(ea.MODE_ENV, "record")
    n = ea.record_dropped_urges([_sig()], site="router", logger=_LOG, now=_NOW)
    assert n == 1  # rows are still counted (logic ran); writes were skipped


# --------------------------------------------------------------------------- #
# record mode — row content
# --------------------------------------------------------------------------- #

def test_record_mode_row_content(tmp_path, monkeypatch):
    monkeypatch.setenv(ea.MODE_ENV, "record")
    state = tmp_path / "state.json"
    evidence = tmp_path / "evidence"
    view = _View({"PSQ": Position("PSQ", AssetClass.ETF, 5.0, 33.10)})

    n = ea.record_dropped_urges(
        [_sig()], site="router", view=view,
        excluded_symbols=frozenset({"MSFT"}),
        evidence_dir=evidence, state_path=state, logger=_LOG, now=_NOW,
    )
    assert n == 1
    fname = evidence / "exit_advice_20260723.ndjson"
    rows = [json.loads(l) for l in fname.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == "exit_advice.v1"
    assert row["ts_utc"] == "2026-07-23T14:00:00Z"
    assert row["site"] == "router"
    assert row["strategy"] == "gamma"
    assert row["symbol"] == "PSQ"
    assert row["side"] == "SELL"
    assert row["size"] == 5.0
    assert row["confidence"] == 0.7
    assert row["asset_class"] == "etf"
    assert row["reason"] == "trend_break_exit"
    assert row["excluded"] is False
    assert row["position_open"] is True

    st = json.loads(state.read_text())
    assert st["mode"] == "record"
    assert st["last_site"] == "router"
    assert st["last_count"] == 1
    assert st["today"]["recorded"] == 1
    assert st["today"]["by_key"] == {"gamma|PSQ|SELL": 1}


def test_excluded_symbol_recorded_with_flag(tmp_path, monkeypatch):
    """gamma|MSFT (operator-excluded) must be RECORDED, flagged excluded=true —
    observability keeps it; consumption (W4B-3+) must never touch it."""
    monkeypatch.setenv(ea.MODE_ENV, "record")
    state = tmp_path / "state.json"
    evidence = tmp_path / "evidence"
    n = ea.record_dropped_urges(
        [_sig(symbol="MSFT", asset_class=AssetClass.EQUITY)],
        site="router", excluded_symbols=frozenset({"MSFT", "AAPL"}),
        evidence_dir=evidence, state_path=state, logger=_LOG, now=_NOW,
    )
    assert n == 1
    row = json.loads((evidence / "exit_advice_20260723.ndjson").read_text())
    assert row["symbol"] == "MSFT"
    assert row["excluded"] is True
    st = json.loads(state.read_text())
    assert st["today"]["excluded"] == 1


def test_position_open_states(tmp_path, monkeypatch):
    monkeypatch.setenv(ea.MODE_ENV, "record")
    evidence = tmp_path / "evidence"
    view = _View({"PSQ": Position("PSQ", AssetClass.ETF, 5.0, 33.10)})
    ea.record_dropped_urges(
        [_sig(symbol="PSQ"), _sig(symbol="TLT", asset_class=AssetClass.ETF)],
        site="router", view=view, excluded_symbols=frozenset(),
        evidence_dir=evidence, state_path=tmp_path / "s.json",
        logger=_LOG, now=_NOW,
    )
    rows = [json.loads(l) for l in
            (evidence / "exit_advice_20260723.ndjson").read_text().splitlines()]
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["PSQ"]["position_open"] is True      # in the injected book
    assert by_sym["TLT"]["position_open"] is False     # known book, not held
    # no view at all -> unknowable, never guessed
    ea.record_dropped_urges(
        [_sig(symbol="VWO", asset_class=AssetClass.ETF)],
        site="pipeline", view=None, excluded_symbols=frozenset(),
        evidence_dir=evidence, state_path=tmp_path / "s.json",
        logger=_LOG, now=_NOW,
    )
    rows = [json.loads(l) for l in
            (evidence / "exit_advice_20260723.ndjson").read_text().splitlines()]
    assert rows[-1]["position_open"] is None


def test_routed_signal_shape(tmp_path, monkeypatch):
    """Pipeline-site objects: source_strategies + net_size, no .strategy."""
    monkeypatch.setenv(ea.MODE_ENV, "record")
    evidence = tmp_path / "evidence"
    routed = SimpleNamespace(
        symbol="PSQ", side=SignalSide.SELL, net_size=3.0,
        source_strategies=(StrategyName.GAMMA, StrategyName.ALPHA),
        confidence=0.6, asset_class=AssetClass.ETF, meta={},
    )
    ea.record_dropped_urges(
        [routed], site="pipeline", excluded_symbols=frozenset(),
        evidence_dir=evidence, state_path=tmp_path / "s.json",
        logger=_LOG, now=_NOW,
    )
    row = json.loads((evidence / "exit_advice_20260723.ndjson").read_text())
    assert row["strategy"] == "gamma"                  # first source
    assert row["source_strategies"] == ["gamma", "alpha"]
    assert row["size"] == 3.0
    assert row["reason"] is None


# --------------------------------------------------------------------------- #
# resilience
# --------------------------------------------------------------------------- #

def test_never_raises_on_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv(ea.MODE_ENV, "record")
    n = ea.record_dropped_urges(
        [object(), None, _sig()], site="router",
        excluded_symbols=frozenset(),
        evidence_dir=tmp_path / "e", state_path=tmp_path / "s.json",
        logger=_LOG, now=_NOW,
    )
    # garbage objects degrade to rows with None fields, never an exception
    assert n == 3


def test_state_accumulates_and_rolls_over(tmp_path, monkeypatch):
    monkeypatch.setenv(ea.MODE_ENV, "record")
    state = tmp_path / "s.json"
    kw = dict(site="router", excluded_symbols=frozenset(),
              evidence_dir=tmp_path / "e", state_path=state, logger=_LOG)
    ea.record_dropped_urges([_sig()], now=_NOW, **kw)
    ea.record_dropped_urges([_sig()], now=_NOW, **kw)
    st = json.loads(state.read_text())
    assert st["today"]["recorded"] == 2
    assert st["today"]["by_key"] == {"gamma|PSQ|SELL": 2}
    # next UTC day resets the rollup
    next_day = datetime(2026, 7, 24, 0, 5, 0, tzinfo=timezone.utc)
    ea.record_dropped_urges([_sig()], now=next_day, **kw)
    st = json.loads(state.read_text())
    assert st["today"]["date"] == "2026-07-24"
    assert st["today"]["recorded"] == 1


# --------------------------------------------------------------------------- #
# router-site integration (the live seam)
# --------------------------------------------------------------------------- #

def _fake_registry():
    def gamma_handler(ctx):
        return [
            _sig(symbol="AAPL", asset_class=AssetClass.EQUITY,
                 meta={"reason": "trend_break_exit"}),
            _sig(symbol="NVDA", side=SignalSide.BUY,
                 asset_class=AssetClass.EQUITY, meta={}),
        ]
    reg = SimpleNamespace(name=SimpleNamespace(value="gamma"), handler=gamma_handler)
    return [reg]


class _FakeBuilder:
    """Shape contract from test_ctx_positions_blast_radius._FakeBuilder."""

    def build(self, *, current_positions=None, current_cash=0.0):
        return SimpleNamespace(
            context=SimpleNamespace(
                portfolio=SimpleNamespace(positions=dict(current_positions or {}))),
            prices={}, current_symbol_notional={}, current_total_notional=0.0)


def test_router_on_routes_dropped_urges_to_recorder(monkeypatch):
    import chad.strategies as strategies
    import chad.core.live_execution_router as router
    import chad.core.ctx_positions_shadow as shadow
    import chad.core.context_positions as cp
    import chad.core.exit_advice as ea_mod

    _AAPL = Position("AAPL", AssetClass.EQUITY, 14.0, 190.0)
    monkeypatch.setattr("chad.utils.context_builder.ContextBuilder", _FakeBuilder)
    monkeypatch.setattr(
        cp, "load_context_positions",
        lambda *a, **k: cp.PositionsView(cp.STATUS_KNOWN, {"AAPL": _AAPL},
                                         "fresh", {"n_injected": 1}),
    )
    monkeypatch.setattr(strategies, "iter_strategy_registrations", _fake_registry)
    monkeypatch.setattr(router, "load_allocation_weights", lambda: {})
    monkeypatch.setattr(shadow, "record_cycle", lambda **k: None)

    captured = {}

    def _capture(dropped, **kw):
        captured["dropped"] = list(dropped)
        captured.update(kw)
        return len(dropped)

    monkeypatch.setattr(ea_mod, "record_dropped_urges", _capture)
    monkeypatch.setenv("CHAD_CTX_POSITIONS", "on")

    available, _ = router._build_available_signals(_LOG)

    keys = {(n, s.symbol, s.side) for n, sigs in available.items() for s in sigs}
    assert ("gamma", "AAPL", SignalSide.SELL) not in keys   # still dropped (D4 intact)
    assert ("gamma", "NVDA", SignalSide.BUY) in keys        # entries untouched
    # the dropped urge reached the recorder with site + view
    assert [s.symbol for s in captured["dropped"]] == ["AAPL"]
    assert captured["site"] == "router"
    assert getattr(captured["view"], "known", False) is True


def test_router_off_recorder_gets_empty_and_filter_inert(monkeypatch):
    import chad.strategies as strategies
    import chad.core.live_execution_router as router
    import chad.core.ctx_positions_shadow as shadow
    import chad.core.context_positions as cp
    import chad.core.exit_advice as ea_mod

    monkeypatch.setattr("chad.utils.context_builder.ContextBuilder", _FakeBuilder)
    monkeypatch.setattr(
        cp, "load_context_positions",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader consulted in OFF")),
    )
    monkeypatch.setattr(strategies, "iter_strategy_registrations", _fake_registry)
    monkeypatch.setattr(router, "load_allocation_weights", lambda: {})
    monkeypatch.setattr(shadow, "record_cycle", lambda **k: None)

    captured = {}
    monkeypatch.setattr(
        ea_mod, "record_dropped_urges",
        lambda dropped, **kw: captured.update(n=len(list(dropped))) or 0,
    )
    monkeypatch.setenv("CHAD_CTX_POSITIONS", "off")

    available, _ = router._build_available_signals(_LOG)
    keys = {(n, s.symbol, s.side) for n, sigs in available.items() for s in sigs}
    assert ("gamma", "AAPL", SignalSide.SELL) in keys       # OFF byte-identical
    assert captured["n"] == 0                               # heartbeat-only call

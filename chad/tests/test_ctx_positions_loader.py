"""W2B-1 — context_positions loader: no-over-count proof (D7) + netting/fail-closed.

Hermetic: every fixture is written to ``tmp_path``; no live-tree / runtime read.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from chad.types import AssetClass, Position
from chad.core import context_positions as cp
from chad.core import position_guard as pg

_NOW = datetime(2026, 7, 20, 23, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# fixture writers
# --------------------------------------------------------------------------- #

def _write_snapshot(tmp_path, rows, *, ts=None, ttl=300):
    ts = ts or _NOW
    doc = {
        "source": "ibkr_portfolio_collector_v2",
        "ts_utc": ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "ttl_seconds": ttl,
        "positions": [
            {"symbol": s, "position": q, "avgCost": a, "secType": "STK", "currency": "USD"}
            for (s, q, a) in rows
        ],
    }
    p = tmp_path / "positions_snapshot.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _guard_strategy_leg(strategy, sym, qty, side="BUY"):
    return f"{strategy}|{sym}", {
        "open": True, "strategy": strategy, "symbol": sym,
        "side": side, "quantity": abs(qty), "last_state": "MAINTAINED",
        "source": "paper_ledger_rebuild",
    }


def _guard_broker_leg(sym, qty, side="BUY"):
    return f"broker_sync|{sym}", {
        "open": True, "strategy": "broker_sync", "symbol": sym,
        "side": side, "quantity": abs(qty), "source": "broker_truth_rebuild",
    }


def _write_guard(tmp_path, entries):
    doc = {"_version": 1, "_written_by": "test"}
    for k, v in entries:
        doc[k] = v
    p = tmp_path / "position_guard.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _load(tmp_path, snap, guard):
    return cp.load_context_positions(
        snapshot_path=snap, guard_path=guard, now=_NOW,
    )


# --------------------------------------------------------------------------- #
# D7 — the no-over-count proof (the SVXY dual-book)
# --------------------------------------------------------------------------- #

def test_dual_booked_symbol_is_never_summed(tmp_path):
    """gamma|SVXY 156 + broker_sync|SVXY 156 + snapshot 156  ->  ONE 156, not 312."""
    snap = _write_snapshot(tmp_path, [("SVXY", 156.0, 58.18)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "SVXY", 156.0),
        _guard_broker_leg("SVXY", 156.0),
    ])
    view = _load(tmp_path, snap, guard)

    assert view.known
    assert set(view.positions) == {"SVXY"}
    assert view.positions["SVXY"].quantity == 156.0          # NOT 312
    assert view.evidence["n_injected"] == 1


def test_aggregation_legs_are_separate_never_summed(tmp_path):
    """Mirror the W2A D7 idiom: the two guard aggregates are DISTINCT sets."""
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "SVXY", 156.0),
        _guard_broker_leg("SVXY", 156.0),
    ])
    state = json.loads(guard.read_text())
    assert pg._agg_guard_strategy(state) == {"SVXY": 156.0}
    assert pg._agg_guard_broker_mirror(state) == {"SVXY": 156.0}


# --------------------------------------------------------------------------- #
# D2 — CHAD-attributed only (clamp to broker; operator inventory invisible)
# --------------------------------------------------------------------------- #

def test_mixed_symbol_injects_only_chad_portion(tmp_path):
    """BAC: broker 202 blends operator + CHAD; only CHAD's 26 is injected."""
    snap = _write_snapshot(tmp_path, [("BAC", 202.0, 40.0)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "BAC", 26.0),
        _guard_broker_leg("BAC", 202.0),
    ])
    view = _load(tmp_path, snap, guard)
    assert view.known
    assert view.positions["BAC"].quantity == 26.0


def test_operator_only_symbol_is_invisible(tmp_path):
    """LLY: broker holds 182, no strategy leg -> not injected at all."""
    snap = _write_snapshot(tmp_path, [("LLY", 182.0, 1189.0)])
    guard = _write_guard(tmp_path, [_guard_broker_leg("LLY", 182.0)])
    view = _load(tmp_path, snap, guard)
    assert view.known
    assert "LLY" not in view.positions
    assert view.positions == {}


def test_over_attribution_is_clamped_down_to_broker(tmp_path):
    """AAPL: CHAD claims 14 but broker holds only 7 -> injected 7 (min)."""
    snap = _write_snapshot(tmp_path, [("AAPL", 7.0, 200.0)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "AAPL", 14.0),
        _guard_broker_leg("AAPL", 7.0),
    ])
    view = _load(tmp_path, snap, guard)
    assert view.positions["AAPL"].quantity == 7.0


def test_short_position_sign_is_preserved(tmp_path):
    """A SELL strategy leg + a negative snapshot -> signed negative injection."""
    snap = _write_snapshot(tmp_path, [("SH", -10.0, 12.0)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("omega", "SH", 10.0, side="SELL"),
        _guard_broker_leg("SH", 10.0, side="SELL"),
    ])
    view = _load(tmp_path, snap, guard)
    assert view.positions["SH"].quantity == -10.0


def test_strategy_leg_broker_absent_not_injected(tmp_path):
    """CHAD claims a symbol the broker does not hold -> nothing injected."""
    snap = _write_snapshot(tmp_path, [("UNH", 228.0, 424.98)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "UNH", 228.0),
        _guard_broker_leg("UNH", 228.0),
        _guard_strategy_leg("gamma", "ZZZ", 5.0),   # broker has no ZZZ
    ])
    view = _load(tmp_path, snap, guard)
    assert set(view.positions) == {"UNH"}
    assert "ZZZ" not in view.positions


# --------------------------------------------------------------------------- #
# D6 — asset_class classifier (overlay allow-list, mapped to the enum)
# --------------------------------------------------------------------------- #

def test_asset_class_classifier(tmp_path):
    snap = _write_snapshot(tmp_path, [("SPY", 30.0, 500.0), ("UNH", 228.0, 424.98), ("SVXY", 156.0, 58.0)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "SPY", 30.0), _guard_broker_leg("SPY", 30.0),
        _guard_strategy_leg("gamma", "UNH", 228.0), _guard_broker_leg("UNH", 228.0),
        _guard_strategy_leg("gamma", "SVXY", 156.0), _guard_broker_leg("SVXY", 156.0),
    ])
    view = _load(tmp_path, snap, guard)
    assert view.positions["SPY"].asset_class is AssetClass.ETF
    assert view.positions["SVXY"].asset_class is AssetClass.ETF
    assert view.positions["UNH"].asset_class is AssetClass.EQUITY
    assert view.positions["UNH"].avg_price == 424.98


# --------------------------------------------------------------------------- #
# D3 — fail-closed to UNKNOWN (never empty-as-truth)
# --------------------------------------------------------------------------- #

def test_missing_snapshot_is_unknown(tmp_path):
    guard = _write_guard(tmp_path, [_guard_strategy_leg("gamma", "UNH", 228.0)])
    view = cp.load_context_positions(
        snapshot_path=tmp_path / "does_not_exist.json", guard_path=guard, now=_NOW,
    )
    assert view.status == cp.STATUS_UNKNOWN
    assert view.positions == {}
    assert view.reason == "snapshot_missing"


def test_stale_snapshot_is_unknown(tmp_path):
    snap = _write_snapshot(tmp_path, [("UNH", 228.0, 424.98)], ts=_NOW - timedelta(seconds=901), ttl=300)
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "UNH", 228.0), _guard_broker_leg("UNH", 228.0),
    ])
    view = _load(tmp_path, snap, guard)
    assert view.status == cp.STATUS_UNKNOWN
    assert view.reason == "snapshot_stale"
    assert view.positions == {}


def test_malformed_snapshot_is_unknown_not_raise(tmp_path):
    p = tmp_path / "positions_snapshot.json"
    p.write_text("{not json", encoding="utf-8")
    guard = _write_guard(tmp_path, [])
    view = cp.load_context_positions(snapshot_path=p, guard_path=guard, now=_NOW)
    assert view.status == cp.STATUS_UNKNOWN
    assert view.positions == {}


def test_snapshot_mirror_conflict_is_unknown(tmp_path):
    """Independent snapshot says 228, the guard mirror says 143 -> do not guess."""
    snap = _write_snapshot(tmp_path, [("UNH", 228.0, 424.98)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "UNH", 228.0),
        _guard_broker_leg("UNH", 143.0),
    ])
    view = _load(tmp_path, snap, guard)
    assert view.status == cp.STATUS_UNKNOWN
    assert view.reason == "snapshot_mirror_conflict"
    assert view.evidence["symbol"] == "UNH"


def test_return_type_is_positions_of_chad_types(tmp_path):
    snap = _write_snapshot(tmp_path, [("UNH", 228.0, 424.98)])
    guard = _write_guard(tmp_path, [
        _guard_strategy_leg("gamma", "UNH", 228.0), _guard_broker_leg("UNH", 228.0),
    ])
    view = _load(tmp_path, snap, guard)
    assert isinstance(view.positions["UNH"], Position)

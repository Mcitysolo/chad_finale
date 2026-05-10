"""Tests for lifecycle_replay_coverage ledger schema normalization.

Validates that _ledger_meta_map (and the underlying _normalize_ledger_open_records
helper) accepts both the wrapped {"open": {...}} schema and the current flat
fingerprint-keyed schema produced by chad/portfolio/ibkr_paper_ledger_watcher.py.
"""

from __future__ import annotations

from chad.ops.lifecycle_replay_coverage import (
    _ledger_meta_map,
    _normalize_ledger_open_records,
)


def _record(symbol: str, qty: float, *, strategy: str = "manual", tags=None) -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_cost": 100.0,
        "conId": 1,
        "currency": "USD",
        "secType": "STK",
        "strategy": strategy,
        "tags": list(tags) if tags is not None else ["ibkr_paper", "manual"],
    }


def test_normalize_handles_flat_schema():
    flat = {
        "h1": _record("SPY", 5.0),
        "h2": _record("QQQ", -3.0),
    }
    out = _normalize_ledger_open_records(flat)
    assert len(out) == 2
    syms = sorted(r["symbol"] for r in out.values())
    assert syms == ["QQQ", "SPY"]


def test_normalize_handles_wrapped_schema():
    wrapped = {
        "open": {
            "h1": _record("SPY", 5.0),
            "h2": _record("QQQ", -3.0),
        }
    }
    out = _normalize_ledger_open_records(wrapped)
    assert len(out) == 2
    syms = sorted(r["symbol"] for r in out.values())
    assert syms == ["QQQ", "SPY"]


def test_normalize_excludes_zero_qty_rows():
    flat = {
        "h1": _record("SPY", 5.0),
        "h2": _record("ZOMBIE", 0.0),
    }
    out = _normalize_ledger_open_records(flat)
    assert len(out) == 1
    assert next(iter(out.values()))["symbol"] == "SPY"


def test_normalize_excludes_rows_missing_symbol_or_qty():
    flat = {
        "ok": _record("SPY", 5.0),
        "no_symbol": {"qty": 1.0, "avg_cost": 1.0},
        "no_qty": {"symbol": "AAPL", "avg_cost": 1.0},
        "not_dict": "garbage",
    }
    out = _normalize_ledger_open_records(flat)
    assert list(out.keys()) == ["ok"]


def test_normalize_returns_empty_for_non_dict_input():
    assert _normalize_ledger_open_records(None) == {}
    assert _normalize_ledger_open_records([]) == {}
    assert _normalize_ledger_open_records("nope") == {}


def test_ledger_meta_map_flat_schema():
    flat = {
        "h1": _record("spy", 5.0, strategy="manual", tags=["ibkr_paper", "manual"]),
        "h2": _record("qqq", -3.0, strategy="momo", tags=["ibkr_paper"]),
    }
    meta = _ledger_meta_map(flat)
    assert set(meta.keys()) == {"SPY", "QQQ"}
    assert meta["SPY"]["qty"] == 5.0
    assert meta["SPY"]["strategy"] == "manual"
    assert "ibkr_paper" in meta["SPY"]["tags"]
    assert meta["QQQ"]["qty"] == -3.0


def test_ledger_meta_map_wrapped_schema():
    wrapped = {
        "open": {
            "h1": _record("SPY", 5.0),
            "h2": _record("QQQ", -3.0),
        }
    }
    meta = _ledger_meta_map(wrapped)
    assert set(meta.keys()) == {"SPY", "QQQ"}


def test_ledger_meta_map_drops_zero_qty():
    flat = {
        "h1": _record("SPY", 5.0),
        "h2": _record("ZOMBIE", 0.0),
    }
    meta = _ledger_meta_map(flat)
    assert set(meta.keys()) == {"SPY"}

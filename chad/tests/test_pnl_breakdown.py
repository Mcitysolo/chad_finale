"""Tests for the per-trade P&L breakdown layer.

Covers:
  * trade_closer writes a nested ``pnl_breakdown`` on closed trades
  * top-level net_pnl/pnl/gross_pnl remain backwards compatible
  * the normalizer handles legacy records (no nested breakdown)
  * unavailable costs are flagged explicitly, not silently zeroed
  * profit_lock's NdjsonPnlProvider still extracts net_pnl from the new shape
  * ``pnl_untrusted`` records are surfaced via the breakdown's status
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

from chad.analytics.pnl_breakdown import (
    PNL_BREAKDOWN_SCHEMA,
    build_pnl_breakdown,
    normalize_to_breakdown,
)
from chad.execution.trade_closer import TradeCloser
from chad.risk.profit_lock import NdjsonPnlProvider


# ---------------------------------------------------------------------------
# Helpers (mirror chad/tests/test_trade_closer.py)
# ---------------------------------------------------------------------------

def _fill(
    fid: str,
    side: str,
    qty: float,
    px: float,
    *,
    strategy: str = "alpha_test",
    symbol: str = "MES",
    ts: str = "2026-04-08T00:00:00+00:00",
    seq: int = 1,
) -> dict:
    return {
        "payload": {
            "schema_version": "paper_exec_fill.v4",
            "fill_id": fid,
            "strategy": strategy,
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "fill_price": px,
            "fill_time_utc": ts,
            "entry_time_utc": ts,
            "is_live": False,
            "reject": False,
            "status": "dry_run",
        },
        "sequence_id": seq,
        "timestamp_utc": ts,
        "prev_hash": "GENESIS",
        "record_hash": fid,
    }


def _write_fills(path: pathlib.Path, fills: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for f in fills:
            fh.write(json.dumps(f) + "\n")


def _make_closer(tmp_path: pathlib.Path) -> TradeCloser:
    return TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=tmp_path / "state.json",
        routing_path=tmp_path / "profit_routing.json",
    )


def _close_one_trade(tmp_path: pathlib.Path) -> dict:
    """Run TradeCloser on a single open/close pair and return the payload."""
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 1, 5000.0, symbol="MES", seq=1),
        _fill("b", "SELL", 1, 5010.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260501.ndjson", fills)
    closer.run_once("20260501")

    out = tmp_path / "trades" / "trade_history_20260501.ndjson"
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    return json.loads(lines[0])["payload"]


# ---------------------------------------------------------------------------
# 1. Closed trade payload carries a nested pnl_breakdown
# ---------------------------------------------------------------------------

def test_closed_trade_writes_pnl_breakdown(tmp_path):
    payload = _close_one_trade(tmp_path)

    assert "pnl_breakdown" in payload, "trade payload missing pnl_breakdown"
    breakdown = payload["pnl_breakdown"]
    assert breakdown["schema_version"] == PNL_BREAKDOWN_SCHEMA

    # Gross = (5010 - 5000) * 1 * MES_multiplier(5) = 50.0
    assert breakdown["gross_price_pnl"] == pytest.approx(50.0)
    assert breakdown["entry_price"] == pytest.approx(5000.0)
    assert breakdown["exit_price"] == pytest.approx(5010.0)
    assert breakdown["quantity"] == pytest.approx(1.0)
    assert breakdown["contract_multiplier"] == pytest.approx(5.0)
    assert breakdown["currency"] == "USD"
    assert breakdown["source"] == "paper_exec"


# ---------------------------------------------------------------------------
# 2. Top-level net_pnl / pnl / gross_pnl unchanged for legacy readers
# ---------------------------------------------------------------------------

def test_pnl_breakdown_preserves_net_pnl_compatibility(tmp_path):
    payload = _close_one_trade(tmp_path)

    # All legacy top-level fields must be present and numerically consistent
    # with the gross-only path: when costs are unavailable, net == gross.
    assert payload["pnl"] == pytest.approx(50.0)
    assert payload["gross_pnl"] == pytest.approx(50.0)
    assert payload["net_pnl"] == pytest.approx(50.0)
    # Top-level commission / slippage stay numeric (legacy readers expect floats);
    # the *truth* about whether they were known lives in pnl_breakdown.
    assert payload["commission"] == 0.0
    assert payload["slippage"] == 0.0


# ---------------------------------------------------------------------------
# 3. Legacy records normalize to a breakdown
# ---------------------------------------------------------------------------

def test_old_trade_record_normalizes_to_pnl_breakdown():
    legacy = {
        "schema_version": "closed_trade.v1",
        "strategy": "alpha",
        "symbol": "SPY",
        "side": "BUY",
        "pnl": -6200.0,
        "gross_pnl": -6200.0,
        "commission": 0.0,
        "slippage": 0.0,
        "fees": None,
        "net_pnl": -6200.0,
        "entry_price": 720.0,
        "exit_price": 100.0,
        "quantity": 10.0,
        "contract_multiplier": 1.0,
        "broker": "paper_exec",
    }
    breakdown = normalize_to_breakdown(legacy)

    assert breakdown["schema_version"] == PNL_BREAKDOWN_SCHEMA
    assert breakdown["gross_price_pnl"] == pytest.approx(-6200.0)
    assert breakdown["net_pnl"] == pytest.approx(-6200.0)
    assert breakdown["entry_price"] == pytest.approx(720.0)
    assert breakdown["exit_price"] == pytest.approx(100.0)
    assert breakdown["quantity"] == pytest.approx(10.0)
    # Legacy records cannot prove their stored 0.0 cost is actually known.
    assert breakdown["cost_basis_status"] == "legacy"
    assert "normalized_from_legacy_record" in breakdown["notes"]


# ---------------------------------------------------------------------------
# 4. Missing costs are explicit (None + status), not silent zero
# ---------------------------------------------------------------------------

def test_missing_costs_are_marked_unavailable_not_silent_zero():
    breakdown = build_pnl_breakdown(
        gross_price_pnl=125.0,
        entry_price=100.0,
        exit_price=125.0,
        quantity=1.0,
    )
    assert breakdown["cost_basis_status"] == "unavailable"
    assert breakdown["commission"] is None
    assert breakdown["slippage"] is None
    assert breakdown["fees"] is None
    # net_pnl == gross_price_pnl when no costs are known
    assert breakdown["net_pnl"] == pytest.approx(125.0)
    # Notes spell out which fields were unavailable
    notes = set(breakdown["notes"])
    assert "commission_unavailable" in notes
    assert "slippage_unavailable" in notes
    assert "fees_unavailable" in notes


def test_partial_costs_are_marked_estimated_and_net_uses_known_only():
    breakdown = build_pnl_breakdown(
        gross_price_pnl=100.0,
        commission=2.0,        # known
        slippage=None,         # unknown — must NOT silently become 0
        fees=None,             # unknown
        source="paper_exec",
    )
    assert breakdown["cost_basis_status"] == "estimated"
    assert breakdown["commission"] == pytest.approx(2.0)
    assert breakdown["slippage"] is None
    assert breakdown["fees"] is None
    # net = gross - known costs only
    assert breakdown["net_pnl"] == pytest.approx(98.0)


# ---------------------------------------------------------------------------
# 5. Profit lock still reads net_pnl after the breakdown is added
# ---------------------------------------------------------------------------

def test_profit_lock_still_reads_net_pnl_after_breakdown(tmp_path):
    """A trade record with the new pnl_breakdown nested object must still
    feed NdjsonPnlProvider.get_realized_pnl() through net_pnl unchanged."""
    repo_root = tmp_path
    trades_dir = repo_root / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    today_payload = {
        "schema_version": "closed_trade.v1",
        "strategy": "alpha_test",
        "symbol": "MES",
        "side": "BUY",
        "pnl": 50.0,
        "gross_pnl": 50.0,
        "commission": 0.0,
        "slippage": 0.0,
        "fees": None,
        "net_pnl": 50.0,
        "entry_price": 5000.0,
        "exit_price": 5010.0,
        "quantity": 1.0,
        "contract_multiplier": 5.0,
        "fill_ids": ["a", "b"],
        "broker": "paper_exec",
        "account_id": "PAPER_EXEC",
        "is_live": False,
        "tags": ["paper", "closed", "alpha_test"],
        "pnl_breakdown": build_pnl_breakdown(
            gross_price_pnl=50.0,
            entry_price=5000.0,
            exit_price=5010.0,
            quantity=1.0,
            contract_multiplier=5.0,
            source="paper_exec",
            cost_basis_status="unavailable",
        ),
    }
    record = {
        "payload": today_payload,
        "prev_hash": "GENESIS",
        "sequence_id": 1,
        "timestamp_utc": "2026-05-04T12:00:00Z",
        "record_hash": "deadbeef",
    }
    import datetime as _dt
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    provider = NdjsonPnlProvider()
    pnl, count, sources = asyncio.run(provider.get_realized_pnl(repo_root, days=0))
    assert count == 1
    assert pnl == pytest.approx(50.0)
    assert sources, "profit_lock should report a source file"


# ---------------------------------------------------------------------------
# 6. Untrusted records never claim trustworthy net_pnl
# ---------------------------------------------------------------------------

def test_untrusted_trade_breakdown_not_counted_if_pnl_untrusted():
    payload = {
        "schema_version": "closed_trade.v1",
        "strategy": "kraken_alpha",
        "symbol": "BTC-USD",
        "side": "BUY",
        "pnl": 0.0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "entry_price": 50000.0,
        "exit_price": 50000.0,
        "quantity": 0.1,
        "contract_multiplier": 1.0,
        "broker": "kraken",
        "extra": {"pnl_untrusted": True, "pnl_untrusted_reason": "kraken_pending"},
        "tags": ["paper", "filled", "pnl_untrusted"],
    }
    breakdown = normalize_to_breakdown(payload)
    assert breakdown["cost_basis_status"] == "untrusted"
    assert "pnl_untrusted" in breakdown["notes"]


def test_normalizer_passes_through_existing_breakdown():
    payload = {
        "schema_version": "closed_trade.v1",
        "pnl_breakdown": {
            "schema_version": PNL_BREAKDOWN_SCHEMA,
            "gross_price_pnl": 12.5,
            "commission": 0.5,
            "fees": 0.1,
            "slippage": 0.4,
            "net_pnl": 11.5,
            "entry_price": 100.0,
            "exit_price": 112.5,
            "quantity": 1.0,
            "contract_multiplier": 1.0,
            "currency": "USD",
            "source": "ibkr",
            "cost_basis_status": "real",
            "notes": [],
        },
    }
    out = normalize_to_breakdown(payload)
    assert out["cost_basis_status"] == "real"
    assert out["source"] == "ibkr"
    assert out["commission"] == pytest.approx(0.5)
    assert out["fees"] == pytest.approx(0.1)
    assert out["slippage"] == pytest.approx(0.4)
    assert out["net_pnl"] == pytest.approx(11.5)

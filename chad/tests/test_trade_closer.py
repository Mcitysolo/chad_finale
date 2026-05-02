"""Tests for chad.execution.trade_closer FIFO matcher."""

from __future__ import annotations

import json
import pathlib

import pytest

from chad.execution.trade_closer import (
    CONTRACT_MULTIPLIERS,
    DEFAULT_MULTIPLIER,
    TradeCloser,
    get_multiplier,
)


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Multiplier table
# ---------------------------------------------------------------------------

def test_multiplier_known_symbols():
    assert get_multiplier("MES") == 5.0
    assert get_multiplier("MNQ") == 2.0
    assert get_multiplier("MCL") == 100.0
    assert get_multiplier("MGC") == 10.0
    assert get_multiplier("SPY") == 1.0
    assert get_multiplier("ZN") == 1000.0


def test_multiplier_unknown_defaults_to_one():
    assert get_multiplier("FOOBAR") == DEFAULT_MULTIPLIER == 1.0
    assert get_multiplier("") == DEFAULT_MULTIPLIER


def test_multiplier_case_insensitive():
    assert get_multiplier("mes") == 5.0


# ---------------------------------------------------------------------------
# Round-trip PnL
# ---------------------------------------------------------------------------

def test_long_round_trip_mes_positive_pnl(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 1, 5000.0, symbol="MES", seq=1),
        _fill("b", "SELL", 1, 5010.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    assert len(closed) == 1
    ct = closed[0]
    assert ct.side == "BUY"
    assert ct.pnl == pytest.approx((5010 - 5000) * 1 * 5.0)
    assert ct.contract_multiplier == 5.0


def test_short_round_trip_mes_correct_direction(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "SELL", 1, 5010.0, symbol="MES", seq=1),
        _fill("b", "BUY", 1, 5000.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    assert len(closed) == 1
    ct = closed[0]
    assert ct.side == "SELL"
    # Short profits when price falls: (5000-5010) * 1 * 5 * -1 = +50
    assert ct.pnl == pytest.approx(50.0)


def test_mcl_multiplier(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 1, 70.0, symbol="MCL", seq=1),
        _fill("b", "SELL", 1, 71.0, symbol="MCL", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    assert len(closed) == 1
    assert closed[0].pnl == pytest.approx(100.0)


def test_unknown_symbol_default_multiplier(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 10, 5.0, symbol="WEIRD", seq=1),
        _fill("b", "SELL", 10, 6.0, symbol="WEIRD", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    assert len(closed) == 1
    # (6-5) * 10 * 1.0 = 10
    assert closed[0].pnl == pytest.approx(10.0)
    assert closed[0].contract_multiplier == 1.0


# ---------------------------------------------------------------------------
# Partial closes
# ---------------------------------------------------------------------------

def test_partial_close_remaining_stays_open(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 5, 5000.0, symbol="MES", seq=1),
        _fill("b", "SELL", 2, 5010.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    assert len(closed) == 1
    assert closed[0].quantity == 2
    assert closed[0].pnl == pytest.approx(2 * 10 * 5.0)
    # Three contracts left in the queue
    key = ("alpha_test", "MES")
    assert sum(lot["quantity"] for lot in closer.queues[key]) == 3


def test_partial_close_consumes_multiple_lots_fifo(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 2, 5000.0, symbol="MES", seq=1),
        _fill("b", "BUY", 2, 5005.0, symbol="MES", seq=2),
        _fill("c", "SELL", 3, 5020.0, symbol="MES", seq=3),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    # Closes 2 from lot a + 1 from lot b => two ClosedTrade records
    assert len(closed) == 2
    assert closed[0].entry_price == 5000.0
    assert closed[0].quantity == 2
    assert closed[0].pnl == pytest.approx(2 * 20 * 5.0)
    assert closed[1].entry_price == 5005.0
    assert closed[1].quantity == 1
    assert closed[1].pnl == pytest.approx(1 * 15 * 5.0)
    # 1 contract left from lot b
    key = ("alpha_test", "MES")
    assert sum(lot["quantity"] for lot in closer.queues[key]) == 1


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def test_state_persists_across_restart(tmp_path):
    fills_path = tmp_path / "fills" / "FILLS_20260408.ndjson"

    # Round 1: open the position
    closer = _make_closer(tmp_path)
    _write_fills(fills_path, [_fill("a", "BUY", 1, 5000.0, symbol="MES", seq=1)])
    closer.load_state()
    closed1 = closer.process_fills("20260408")
    closer.save_state()
    assert closed1 == []
    assert closer.state_path.exists()

    # Round 2: new closer instance, append closing fill
    closer2 = _make_closer(tmp_path)
    closer2.load_state()
    # State must show the open lot
    key = ("alpha_test", "MES")
    assert key in closer2.queues
    assert sum(lot["quantity"] for lot in closer2.queues[key]) == 1
    # Already-processed id must NOT be re-matched
    _write_fills(
        fills_path,
        [
            _fill("a", "BUY", 1, 5000.0, symbol="MES", seq=1),
            _fill("b", "SELL", 1, 5010.0, symbol="MES", seq=2),
        ],
    )
    closed2 = closer2.process_fills("20260408")
    assert len(closed2) == 1
    assert closed2[0].pnl == pytest.approx(50.0)


def test_already_processed_fills_not_double_counted(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 1, 5000.0, symbol="MES", seq=1),
        _fill("b", "SELL", 1, 5010.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed_first = closer.process_fills("20260408")
    assert len(closed_first) == 1
    # Process again — nothing should match because both fill_ids are tracked
    closed_second = closer.process_fills("20260408")
    assert closed_second == []


# ---------------------------------------------------------------------------
# Independence across strategies
# ---------------------------------------------------------------------------

def test_multiple_strategies_independent_queues(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 1, 5000.0, strategy="alpha_a", symbol="MES", seq=1),
        _fill("b", "BUY", 1, 5000.0, strategy="alpha_b", symbol="MES", seq=2),
        _fill("c", "SELL", 1, 5020.0, strategy="alpha_a", symbol="MES", seq=3),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")
    assert len(closed) == 1
    assert closed[0].strategy == "alpha_a"
    assert closed[0].pnl == pytest.approx(100.0)
    # alpha_b still open
    key_b = ("alpha_b", "MES")
    assert sum(lot["quantity"] for lot in closer.queues[key_b]) == 1


# ---------------------------------------------------------------------------
# write_trade_history + get_summary
# ---------------------------------------------------------------------------

def test_run_once_writes_trade_history(tmp_path):
    closer = _make_closer(tmp_path)
    fills = [
        _fill("a", "BUY", 1, 5000.0, symbol="MES", seq=1),
        _fill("b", "SELL", 1, 5010.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)

    result = closer.run_once("20260408")
    assert result["closed_count"] == 1
    assert result["written"] == 1
    out = tmp_path / "trades" / "trade_history_20260408.ndjson"
    assert out.exists()

    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["payload"]["schema_version"] == "closed_trade.v1"
    assert rec["payload"]["pnl"] == pytest.approx(50.0)
    # Per-trade PnL breakdown — schema must expose the four explicit fields.
    # Values for commission / slippage start at 0.0 (populated by future
    # fill enrichment), so we only assert keys are present here.
    payload = rec["payload"]
    for k in ("gross_pnl", "commission", "slippage", "net_pnl"):
        assert k in payload, f"missing payload key: {k}"
    assert "record_hash" in rec
    assert "sequence_id" in rec

    summary = closer.get_summary("20260408")
    assert summary["total_closed"] == 1
    assert summary["total_pnl"] == pytest.approx(50.0)
    assert "alpha_test" in summary["by_strategy"]
    assert summary["by_strategy"]["alpha_test"]["count"] == 1


# ---------------------------------------------------------------------------
# ISSUE-74: fill_id synthesis for broker_sync anchor lots
# ---------------------------------------------------------------------------

def test_synthesize_fill_id_for_broker_sync_lot(tmp_path):
    """
    Step 13.5 broker_sync rebuild writes queue lots from a position snapshot
    that has no exec_id, so the lot lacks fill_id. load_state() must
    synthesize a deterministic syn_<sha256[:24]> id on load so downstream
    matching at _match_fill does not raise KeyError.
    """
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "processed_fill_ids": [],
                "queues": [
                    {
                        "strategy": "broker_sync",
                        "symbol": "GLD",
                        "lots": [
                            {
                                "side": "SELL",
                                "quantity": 1108.0,
                                "fill_price": 311.50,
                                "lot_ts_utc": "2026-04-20T20:27:54.400505Z",
                                "source": "step_13_5_post_paper_fills",
                                "tags": ["reconciled_step_13_5"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    closer = TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=state_path,
    )
    closer.load_state()

    key = ("broker_sync", "GLD")
    assert key in closer.queues
    lot = closer.queues[key][0]
    assert "fill_id" in lot
    assert lot["fill_id"].startswith("syn_")
    assert len(lot["fill_id"]) == len("syn_") + 24

    # Determinism: a fresh TradeCloser loading the same state yields the
    # same synthetic id.
    closer2 = TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=state_path,
    )
    closer2.load_state()
    assert closer2.queues[key][0]["fill_id"] == lot["fill_id"]


def test_preserve_existing_fill_id_when_present(tmp_path):
    """
    When a lot already has a fill_id, load_state() must not overwrite it.
    """
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "processed_fill_ids": [],
                "queues": [
                    {
                        "strategy": "alpha_test",
                        "symbol": "MES",
                        "lots": [
                            {
                                "fill_id": "existing_123",
                                "side": "BUY",
                                "quantity": 1.0,
                                "fill_price": 5000.0,
                                "ts_utc": "2026-04-20T20:00:00Z",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    closer = TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=state_path,
    )
    closer.load_state()

    key = ("alpha_test", "MES")
    assert closer.queues[key][0]["fill_id"] == "existing_123"

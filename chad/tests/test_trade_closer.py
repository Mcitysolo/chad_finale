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


def test_trade_closer_skips_pnl_untrusted_fill_ids(tmp_path):
    """Fills tagged pnl_untrusted (via extra, payload, or tags) must never
    enter FIFO matching — even when a counter-side trusted fill is present.
    Trade closer must not produce a closed_trade record from them."""
    closer = _make_closer(tmp_path)

    untrusted_via_extra = _fill("u_extra", "SELL", 10, 100.0, symbol="SPY", seq=1)
    untrusted_via_extra["payload"]["extra"] = {
        "pnl_untrusted": True,
        "pnl_untrusted_reason": "fill_price=100.0 deviates from price_cache",
    }

    untrusted_via_tags = _fill("u_tags", "SELL", 10, 100.0, symbol="SPY", seq=2)
    untrusted_via_tags["payload"]["tags"] = [
        "paper", "filled", "delta", "pnl_untrusted",
    ]

    untrusted_via_payload = _fill("u_payload", "SELL", 10, 100.0, symbol="SPY", seq=3)
    untrusted_via_payload["payload"]["pnl_untrusted"] = True

    # A real trusted closing fill on the other side — must not match anything
    # because the queues are still empty (untrusted fills were skipped).
    trusted_close = _fill("t_close", "BUY", 10, 720.0, symbol="SPY", seq=4)

    _write_fills(
        tmp_path / "fills" / "FILLS_20260408.ndjson",
        [untrusted_via_extra, untrusted_via_tags, untrusted_via_payload, trusted_close],
    )
    closer.load_state()
    closed = closer.process_fills("20260408")

    # No FIFO match should be produced. The trusted_close becomes a fresh
    # opening lot (no opposite-side trusted lot to match against).
    assert closed == [], (
        "untrusted fills must be skipped before FIFO matching; "
        f"unexpectedly produced: {closed}"
    )
    # None of the untrusted fill_ids should be tracked as processed.
    assert "u_extra" not in closer.processed_fill_ids
    assert "u_tags" not in closer.processed_fill_ids
    assert "u_payload" not in closer.processed_fill_ids
    # The trusted close is processed (opens a SELL-side lot from BUY 720,
    # held until a counter-side trusted fill arrives).
    assert "t_close" in closer.processed_fill_ids


def test_trade_closer_dedupes_duplicate_fill_ids(tmp_path):
    """The same fill_id appearing multiple times in a FILLS_*.ndjson must
    produce only ONE FIFO event. The IBKR harvester occasionally emits
    duplicate records for the same execID; trade_closer must dedup via
    its processed_fill_ids set."""
    closer = _make_closer(tmp_path)

    # Opening BUY at $100, then 3 identical SELL records all sharing the
    # same fill_id. Only the first SELL should match against the BUY; the
    # other two duplicates must be dropped.
    fills = [
        _fill("open", "BUY", 1, 5000.0, symbol="MES", seq=1),
        _fill("dup", "SELL", 1, 5010.0, symbol="MES", seq=2),
        _fill("dup", "SELL", 1, 5010.0, symbol="MES", seq=3),
        _fill("dup", "SELL", 1, 5010.0, symbol="MES", seq=4),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills)
    closer.load_state()
    closed = closer.process_fills("20260408")

    assert len(closed) == 1, (
        f"expected exactly one closed trade after dedup, got {len(closed)}"
    )
    ct = closed[0]
    assert ct.fill_ids == ["open", "dup"]
    # Run again — duplicates should still produce nothing because both
    # fill_ids are in processed_fill_ids.
    closed_again = closer.process_fills("20260408")
    assert closed_again == []


# ---------------------------------------------------------------------------
# Placeholder / fallback / duplicate-fill protection (2026-05-08 incident)
# ---------------------------------------------------------------------------

def test_trade_closer_skips_placeholder_100_fill(tmp_path):
    """Reproduces the 2026-05-08 SPY delta incident: opening fills landed at
    fill_price=100.0 with extra.expected_price=100.0 because price_cache had
    no SPY entry and the executor fell back to the strategy's default expected
    price. trade_closer must refuse such fills BEFORE they enter FIFO matching
    so the real closing fill at $731.44 cannot generate a phantom -$6314 SELL
    close. No closed_trade may be produced."""
    closer = _make_closer(tmp_path)

    # Opening SELL with the placeholder fingerprint: fill_price == expected
    # == 100.0. No other untrusted markers — exact shape from FILLS_20260503.
    placeholder_open = _fill("p_open", "SELL", 10, 100.0, symbol="SPY", seq=1)
    placeholder_open["payload"]["status"] = "filled"
    placeholder_open["payload"]["source"] = "paper_trade_executor"
    placeholder_open["payload"]["tags"] = ["paper", "filled", "delta", "equity"]
    placeholder_open["payload"]["extra"] = {
        "source_strategies": ["delta"],
        "slippage_bps": 0.0,
        "latency_ms": 0.0,
        "expected_price": 100.0,
    }

    # Real closing fill at the actual SPY price. With the placeholder open
    # rejected, this must NOT generate a closed trade — it just opens a new
    # BUY-side lot.
    real_close = _fill("p_close", "BUY", 10, 731.44, symbol="SPY", seq=2)
    real_close["payload"]["status"] = "paper_fill"
    real_close["payload"]["source"] = "paper_trade_executor"
    real_close["payload"]["tags"] = ["paper", "filled", "delta", "etf"]
    real_close["payload"]["extra"] = {
        "source_strategies": ["delta"],
        "slippage_bps": 0.0,
        "latency_ms": 0.0,
        "expected_price": 733.83,
    }

    _write_fills(
        tmp_path / "fills" / "FILLS_20260508.ndjson",
        [placeholder_open, real_close],
    )
    closer.load_state()
    closed = closer.process_fills("20260508")

    assert closed == [], (
        "placeholder fill at $100 must be skipped; got phantom closes: "
        f"{[(c.side, c.entry_price, c.exit_price, c.pnl) for c in closed]}"
    )
    assert "p_open" not in closer.processed_fill_ids
    # The real fill is processed and held as a fresh open lot.
    assert "p_close" in closer.processed_fill_ids
    key = ("alpha_test", "SPY")
    assert sum(lot["quantity"] for lot in closer.queues[key]) == 10


def test_trade_closer_skips_pnl_untrusted_fill(tmp_path):
    """Variants where pnl_untrusted is set at extra, payload, or tags level
    must all be rejected before FIFO matching."""
    closer = _make_closer(tmp_path)

    via_extra = _fill("u1", "SELL", 5, 100.0, symbol="SPY", seq=1)
    via_extra["payload"]["extra"] = {
        "pnl_untrusted": True,
        "pnl_untrusted_reason": "fill_price=100.0 deviates 86% from price_cache=720.0",
    }

    via_payload = _fill("u2", "SELL", 5, 100.0, symbol="SPY", seq=2)
    via_payload["payload"]["pnl_untrusted"] = True

    via_tag = _fill("u3", "SELL", 5, 100.0, symbol="SPY", seq=3)
    via_tag["payload"]["tags"] = ["paper", "filled", "pnl_untrusted"]

    # Real closing fill — must not match because no untrusted lot is queued.
    real_close = _fill("real_close", "BUY", 5, 720.0, symbol="SPY", seq=4)

    _write_fills(
        tmp_path / "fills" / "FILLS_20260508.ndjson",
        [via_extra, via_payload, via_tag, real_close],
    )
    closer.load_state()
    closed = closer.process_fills("20260508")

    assert closed == []
    for fid in ("u1", "u2", "u3"):
        assert fid not in closer.processed_fill_ids
    assert "real_close" in closer.processed_fill_ids


def test_trade_closer_does_not_reuse_consumed_fill_id(tmp_path):
    """A fill_id that already appears in an existing trade_history closed_trade
    record must be skipped from new FIFO matching. Defends against state-file
    loss / corruption where the FIFO queue might be rebuilt and a previously
    consumed fill_id could otherwise re-match."""
    closer = _make_closer(tmp_path)

    # Round 1: real BUY/SELL pair → one closed trade.
    fills_r1 = [
        _fill("open_r1", "BUY", 1, 5000.0, symbol="MES", seq=1),
        _fill("close_r1", "SELL", 1, 5010.0, symbol="MES", seq=2),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260408.ndjson", fills_r1)
    closer.load_state()
    closed_r1 = closer.process_fills("20260408")
    assert len(closed_r1) == 1
    closer.write_trade_history(closed_r1, "20260408")

    # Simulate state-file loss between runs by deleting it. The trade_history
    # file remains as the durable record of what was already consumed.
    closer.save_state()
    if closer.state_path.exists():
        closer.state_path.unlink()

    # Round 2: a fresh closer instance with empty in-memory state. The same
    # fill_ids reappear in a same-day re-replay (e.g., harvester re-emit).
    # Without the trade_history seed, the FIFO matcher would rebuild a phantom
    # close for this pair. With the seed, both fill_ids are already in
    # processed_fill_ids and produce nothing.
    closer2 = _make_closer(tmp_path)
    closer2.load_state()
    assert "open_r1" in closer2.processed_fill_ids
    assert "close_r1" in closer2.processed_fill_ids

    closed_r2 = closer2.process_fills("20260408")
    assert closed_r2 == [], (
        "fill_ids already consumed in a prior closed_trade must not generate "
        f"another close; got: {closed_r2}"
    )


def test_trade_closer_real_fill_still_closes_normally(tmp_path):
    """Sanity: the new placeholder/status/source guards must NOT regress real,
    well-formed fills. A normal BUY-then-SELL pair on SPY at realistic prices
    with status=paper_fill still produces exactly one closed trade."""
    closer = _make_closer(tmp_path)

    open_buy = _fill("real_open", "BUY", 10, 720.50, symbol="SPY", seq=1)
    open_buy["payload"]["status"] = "paper_fill"
    open_buy["payload"]["source"] = "paper_trade_executor"
    open_buy["payload"]["tags"] = ["paper", "filled", "delta", "etf"]
    open_buy["payload"]["extra"] = {
        "source_strategies": ["delta"],
        "expected_price": 721.00,
    }

    close_sell = _fill("real_close", "SELL", 10, 731.44, symbol="SPY", seq=2)
    close_sell["payload"]["status"] = "paper_fill"
    close_sell["payload"]["source"] = "paper_trade_executor"
    close_sell["payload"]["tags"] = ["paper", "filled", "delta", "etf"]
    close_sell["payload"]["extra"] = {
        "source_strategies": ["delta"],
        "expected_price": 733.83,
    }

    _write_fills(
        tmp_path / "fills" / "FILLS_20260508.ndjson",
        [open_buy, close_sell],
    )
    closer.load_state()
    closed = closer.process_fills("20260508")

    assert len(closed) == 1
    ct = closed[0]
    assert ct.side == "BUY"
    assert ct.entry_price == pytest.approx(720.50)
    assert ct.exit_price == pytest.approx(731.44)
    assert ct.quantity == 10
    # SPY multiplier == 1.0 → pnl = (731.44 - 720.50) * 10 = 109.4
    assert ct.pnl == pytest.approx((731.44 - 720.50) * 10)


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

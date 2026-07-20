"""PFF1 (2026-07-20) — trade_closer must not double-book an order that is
written from BOTH fill sources.

Root cause reproduced here: the paper_trade_executor emits one aggregate SIM
fill per order (account PAPER_EXEC), and the ibkr_paper_fill_harvester
ADDITIVELY mirrors the SAME order as the real broker's slice fills (tag
ibkr_harvest). Feeding both into FIFO turned one 273-share exit-overlay SELL
into ~506 booked shares: the executor aggregate closed the Epoch-3 seed lot
(the untrusted +625.17 close) and the harvester's 273-in-slices opened phantom
SHORT lots that round-tripped the re-buys into 6 fabricated "trusted" closes,
netting the gamma queue to zero against a real +228 broker long.

The fix drops harvester fills for any symbol the executor has already booked
(a redundant broker mirror) while preserving harvester fills for symbols with
no executor fill (orphans — e.g. broker-side positions CHAD never executed).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from chad.execution.trade_closer import TradeCloser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _executor_fill(fid, side, qty, px, *, symbol="UNH", strategy="gamma",
                   ts, seq):
    """A paper_trade_executor SIM aggregate fill (one record per order)."""
    return {
        "payload": {
            "schema_version": "paper_exec_fill.v4",
            "account_id": "PAPER_EXEC",
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
            "status": "paper_fill",
            "source": "paper_trade_executor",
            "order_type": "SIM",
            "tags": ["paper", "filled", strategy, "equity"],
            "broker": "ibkr_paper",
        },
        "sequence_id": seq,
        "timestamp_utc": ts,
        "prev_hash": "GENESIS",
        "record_hash": fid,
    }


def _harvester_fill(fid, side, qty, px, *, symbol="UNH", strategy="gamma",
                    ts, seq, exec_id="x"):
    """An ibkr_paper_fill_harvester broker slice — the redundant mirror."""
    return {
        "payload": {
            "schema_version": "paper_exec_fill.v4",
            "account_id": "DUR119533",
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
            "status": "paper_fill",
            "source": "ibkr_paper_fill_harvester",
            "order_type": "LMT",
            "tags": ["paper", "filled", "ibkr_harvest", strategy, "equity"],
            "extra": {"ibkr_exec_id": exec_id, "source": "ibkr_paper_fill_harvester"},
            "broker": "ibkr_paper",
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


def _seed_untrusted_long(state_path: pathlib.Path, qty: float, px: float,
                         *, symbol="UNH", strategy="gamma"):
    """Persist an Epoch-3 seed lot (untrusted, fabricated cost basis) into the
    trade_closer FIFO state, exactly as the reconciler adopts a broker position.
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "processed_fill_ids": [],
        "queues": [{
            "strategy": strategy, "symbol": symbol,
            "lots": [{
                "fill_id": f"RECON_ADOPT_{symbol}_seed",
                "side": "BUY", "quantity": qty, "fill_price": px,
                "ts_utc": "2026-07-15T17:36:18Z", "multiplier": 1.0,
                "meta": {"pnl_untrusted": True, "scoring_excluded": True,
                         "provenance": "UNATTRIBUTED_EPOCH3_ACCUMULATION",
                         "seeded_from": "broker_truth", "reconciled": True},
            }],
        }],
    }))


def _make_closer(tmp_path: pathlib.Path) -> TradeCloser:
    return TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=tmp_path / "state.json",
        routing_path=tmp_path / "profit_routing.json",
    )


def _summarize(trades_dir: pathlib.Path, date_str: str):
    out = trades_dir / f"trade_history_{date_str}.ndjson"
    trusted_qty = untrusted_qty = 0.0
    trusted_n = untrusted_n = 0
    if out.exists():
        for ln in out.read_text().splitlines():
            if not ln.strip():
                continue
            p = json.loads(ln)["payload"]
            tags = p.get("tags", [])
            untrusted = (
                "pnl_untrusted" in tags
                or bool((p.get("extra") or {}).get("pnl_untrusted"))
                or bool((p.get("meta") or {}).get("pnl_untrusted"))
            )
            if untrusted:
                untrusted_n += 1
                untrusted_qty += p["quantity"]
            else:
                trusted_n += 1
                trusted_qty += p["quantity"]
    return trusted_n, trusted_qty, untrusted_n, untrusted_qty


def _gamma_unh_qty(state_path: pathlib.Path) -> float:
    st = json.loads(state_path.read_text())
    net = 0.0
    for e in st.get("queues", []):
        if (e.get("strategy"), e.get("symbol")) == ("gamma", "UNH"):
            for lot in e.get("lots", []):
                sign = 1.0 if lot["side"] == "BUY" else -1.0
                net += sign * lot["quantity"]
    return net


# ---------------------------------------------------------------------------
# The regression: one 273-share SELL must not book ~506 shares of closes
# ---------------------------------------------------------------------------

def test_harvester_mirror_does_not_double_book(tmp_path):
    """Faithful replay of the 2026-07-20 UNH churn: executor aggregate + the
    harvester's mirror slices for the same order + the re-buys. Only the seed
    close (untrusted) may be booked; the harvester mirror must not fabricate
    trusted round-trips, and the gamma queue must reflect the real +228 long.
    """
    closer = _make_closer(tmp_path)
    _seed_untrusted_long(tmp_path / "state.json", 273.0, 420.71)

    fills = [
        # executor aggregate SELL 273 (closes the seed) — written first
        _executor_fill("exec_sell273", "SELL", 273.0, 423.0,
                       ts="2026-07-20T13:50:55.4Z", seq=1),
        # executor re-buys (the real re-open of a +228 long)
        _executor_fill("exec_buy5", "BUY", 5.0, 422.45,
                       ts="2026-07-20T13:51:10.6Z", seq=2),
        _executor_fill("exec_buy223", "BUY", 223.0, 425.57,
                       ts="2026-07-20T13:55:45.4Z", seq=3),
        # harvester broker slices of the SAME 273 SELL (redundant mirror)
        _harvester_fill("h_sell42", "SELL", 42.0, 424.86,
                        ts="2026-07-20T13:50:56Z", seq=4, exec_id="a"),
        _harvester_fill("h_sell40", "SELL", 40.0, 424.87,
                        ts="2026-07-20T13:51:02Z", seq=5, exec_id="b"),
        _harvester_fill("h_sell80", "SELL", 80.0, 424.87,
                        ts="2026-07-20T13:51:02Z", seq=6, exec_id="c"),
        _harvester_fill("h_sell71", "SELL", 71.0, 424.90,
                        ts="2026-07-20T13:51:06Z", seq=7, exec_id="d"),
        _harvester_fill("h_buy5", "BUY", 5.0, 425.57,
                        ts="2026-07-20T13:51:11Z", seq=8, exec_id="e"),
        # harvester re-buy mirror, re-attributed to broker_sync by the harvester
        _harvester_fill("h_bsy80", "BUY", 80.0, 424.96, strategy="broker_sync",
                        ts="2026-07-20T13:55:46Z", seq=9, exec_id="f"),
        _harvester_fill("h_bsy143", "BUY", 143.0, 424.95, strategy="broker_sync",
                        ts="2026-07-20T13:55:46Z", seq=10, exec_id="g"),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260720.ndjson", fills)

    closer.load_state()
    closed = closer.process_fills("20260720")
    closer.write_trade_history(closed, "20260720")
    closer.save_state()

    trusted_n, trusted_qty, untrusted_n, untrusted_qty = _summarize(
        tmp_path / "trades", "20260720"
    )

    # Exactly one close: the untrusted seed-lot close of 273 shares.
    assert untrusted_n == 1
    assert untrusted_qty == pytest.approx(273.0)
    # No fabricated trusted round-trips from the mirror slices.
    assert trusted_n == 0
    assert trusted_qty == pytest.approx(0.0)
    # Total booked closes == 273 (the real sell), NOT ~506 (273 + 233 mirror).
    assert untrusted_qty + trusted_qty == pytest.approx(273.0)
    # The gamma queue reflects the real re-opened +228 long (5 + 223), so it
    # matches broker truth and does not net to zero.
    assert _gamma_unh_qty(tmp_path / "state.json") == pytest.approx(228.0)
    # And no phantom broker_sync|UNH lot was created from the re-buy mirror.
    st = json.loads((tmp_path / "state.json").read_text())
    assert not any(
        e.get("strategy") == "broker_sync" and e.get("symbol") == "UNH"
        for e in st.get("queues", [])
    )


def test_harvester_orphan_symbol_still_feeds_fifo(tmp_path):
    """A harvester fill for a symbol with NO executor fill is a genuine orphan
    (e.g. a broker-side position) and must keep feeding FIFO — the mirror dedup
    is strictly scoped to executor-booked symbols. This preserves the Bug B
    Fix B behaviour of matching harvester-only opens/closes.
    """
    closer = _make_closer(tmp_path)

    fills = [
        # An unrelated executor order so executor_symbols is non-empty but
        # does NOT include MSFT.
        _executor_fill("exec_spy_buy", "BUY", 1.0, 500.0, symbol="SPY",
                       strategy="delta", ts="2026-07-20T14:00:00Z", seq=1),
        _executor_fill("exec_spy_sell", "SELL", 1.0, 501.0, symbol="SPY",
                       strategy="delta", ts="2026-07-20T14:05:00Z", seq=2),
        # harvester-only MSFT round-trip (orphan) — must still be matched.
        _harvester_fill("h_msft_buy", "BUY", 7.0, 410.0, symbol="MSFT",
                        strategy="alpha", ts="2026-07-20T14:01:00Z", seq=3,
                        exec_id="m1"),
        _harvester_fill("h_msft_sell", "SELL", 7.0, 412.0, symbol="MSFT",
                        strategy="alpha", ts="2026-07-20T14:06:00Z", seq=4,
                        exec_id="m2"),
    ]
    _write_fills(tmp_path / "fills" / "FILLS_20260720.ndjson", fills)

    closer.load_state()
    closed = closer.process_fills("20260720")

    symbols_closed = sorted(ct.symbol for ct in closed)
    # Both the executor SPY round-trip AND the orphan harvester MSFT round-trip
    # produce a closed trade.
    assert symbols_closed == ["MSFT", "SPY"]
    msft = next(ct for ct in closed if ct.symbol == "MSFT")
    assert msft.quantity == pytest.approx(7.0)
    assert msft.pnl == pytest.approx((412.0 - 410.0) * 7.0)

"""INCIDENT-0723 regression — drill dry_run exhaust must never move the books.

Replays the 2026-07-23 corruption sequence with row shapes lifted from the
real ledger (audits/INCIDENT_20260723_DRILL_EXHAUST_FALSE_FLAT.md):

  1. real paper_fill BUYs open lots (gamma|PSQ 5 @26.2, gamma|IWM 200);
  2. a flatten DRILL writes status=dry_run SELL rows for those legs
     (the adapter's dry_run short-circuit — nothing traded at the broker);
  3. the trade closer scans the ledger;
  4. the guard rebuild mirrors the closer queues.

The incident outcome (before the fix): the dry_run SELLs netted the lots,
minted fake closed round-trips, and the rebuild false-flatted the guard —
which then re-entered real positions at the next open. Every assertion here
pins the opposite: rehearsal rows are inert at every stage.

Root cause fixed: chad/execution/trade_closer.py _TRUSTED_FILL_STATUSES no
longer blesses "dry_run".
"""

from __future__ import annotations

import json
import logging
import pathlib

from chad.execution import trade_closer as tc_mod
from chad.execution.trade_closer import TradeCloser


DATE = "20260723"
TS_OPEN = "2026-07-21T13:34:42+00:00"
TS_DRILL = "2026-07-23T10:26:04+00:00"


def _row(fid, strategy, symbol, side, qty, px, status, ts, seq):
    """FILLS_*.ndjson row in the exact incident shape (paper_exec_fill.v4)."""
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
            "status": status,
            "source": "paper_trade_executor",
            "order_type": "SIM",
        },
        "sequence_id": seq,
        "timestamp_utc": ts,
        "prev_hash": "GENESIS",
        "record_hash": f"rh_{fid}",
    }


def _write_ndjson(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _make_closer(tmp_path: pathlib.Path) -> TradeCloser:
    return TradeCloser(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
        state_path=tmp_path / "closer_state.json",
        routing_path=tmp_path / "profit_routing.json",
    )


def _open_rows():
    return [
        _row("open_psq", "gamma", "PSQ", "BUY", 5.0, 26.2, "paper_fill", TS_OPEN, 1),
        _row("open_iwm", "gamma", "IWM", "BUY", 200.0, 295.9, "paper_fill", TS_OPEN, 2),
    ]


def _drill_rows():
    """The drill exhaust: dry_run SELLs for every open leg + the phantom
    single-char strategy row (the real incident's "g" SELL that minted a
    short lot from nothing)."""
    return [
        _row("drill_psq", "gamma", "PSQ", "SELL", 5.0, 26.2, "dry_run", TS_DRILL, 3),
        _row("drill_iwm", "gamma", "IWM", "SELL", 200.0, 293.6, "dry_run", TS_DRILL, 4),
        _row("drill_g", "g", "PSQ", "SELL", 5.0, 26.2, "dry_run", TS_DRILL, 5),
    ]


# ---------------------------------------------------------------------------
# Stage gates
# ---------------------------------------------------------------------------

def test_trusted_set_excludes_dry_run():
    """The root-cause line: dry_run must not be a trusted money status."""
    assert "dry_run" not in tc_mod._TRUSTED_FILL_STATUSES
    assert tc_mod._TRUSTED_FILL_STATUSES == frozenset({"filled", "paper_fill"})


def test_extract_fill_drops_dry_run_row():
    """Ingest gate: a dry_run row never enters FIFO matching."""
    assert tc_mod._extract_fill(_drill_rows()[0]) is None


def test_extract_fill_keeps_paper_fill_row():
    """Control: the same shape with a real status IS ingested."""
    extracted = tc_mod._extract_fill(_open_rows()[0])
    assert extracted is not None
    assert extracted["fill_id"] == "open_psq"


# ---------------------------------------------------------------------------
# Full replay: drill -> closer -> rebuild -> guard must NOT flat
# ---------------------------------------------------------------------------

def test_replay_drill_exhaust_does_not_net_lots_or_mint_trades(tmp_path):
    closer = _make_closer(tmp_path)
    _write_ndjson(tmp_path / "fills" / f"FILLS_{DATE}.ndjson",
                  _open_rows() + _drill_rows())

    closed = closer.process_fills(DATE)
    closer.save_state()

    # No fake round-trips: the dry_run SELLs must close nothing.
    assert closed == []
    trades_file = tmp_path / "trades" / f"trade_history_{DATE}.ndjson"
    assert not trades_file.exists() or trades_file.read_text().strip() == ""

    # Open lots survive intact; no phantom short from the "g" row.
    state = json.loads((tmp_path / "closer_state.json").read_text())
    queues = {(q["strategy"], q["symbol"]): q["lots"] for q in state["queues"]}
    assert ("g", "PSQ") not in queues, "phantom short lot minted from dry_run row"
    psq = queues[("gamma", "PSQ")]
    iwm = queues[("gamma", "IWM")]
    assert sum(l["quantity"] for l in psq) == 5.0
    assert sum(l["quantity"] for l in iwm) == 200.0
    assert all(l["side"] == "BUY" for l in psq + iwm)


def test_replay_guard_rebuild_keeps_positions_open(tmp_path, monkeypatch):
    """End-to-end: after the drill exhaust is scanned, the guard rebuild must
    keep every real leg OPEN — the false-flat is the incident's signature."""
    from chad.core import live_loop, position_guard

    closer = _make_closer(tmp_path)
    _write_ndjson(tmp_path / "fills" / f"FILLS_{DATE}.ndjson",
                  _open_rows() + _drill_rows())
    closer.process_fills(DATE)
    closer.save_state()

    guard_path = tmp_path / "position_guard.json"
    guard_path.write_text(json.dumps({
        "gamma|PSQ": {"open": True, "strategy": "gamma", "symbol": "PSQ",
                      "side": "BUY", "quantity": 5.0,
                      "opened_at": TS_OPEN, "last_state": "OPEN"},
        "gamma|IWM": {"open": True, "strategy": "gamma", "symbol": "IWM",
                      "side": "BUY", "quantity": 200.0,
                      "opened_at": TS_OPEN, "last_state": "OPEN"},
    }))
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)
    monkeypatch.setattr(live_loop, "_TRADE_CLOSER_STATE_PATH",
                        tmp_path / "closer_state.json")

    live_loop._rebuild_guard_from_paper_ledger(logging.getLogger("test"))

    guard = json.loads(guard_path.read_text())
    assert guard["gamma|PSQ"]["open"] is True
    assert guard["gamma|PSQ"]["quantity"] == 5.0
    assert guard["gamma|IWM"]["open"] is True
    assert guard["gamma|IWM"]["quantity"] == 200.0
    assert "g|PSQ" not in guard, "phantom short leg reached the guard"

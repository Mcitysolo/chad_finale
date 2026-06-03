"""Bug B Fix B — harvester phantom-close poisoning removed.

Covers the two halves of the fix plus the end-to-end flow:

(a) SEEDER (consumer defense): TradeCloser.seed_processed_from_trade_history
    must IGNORE harvester phantom round-trips (closed_trade.v1 with the
    ``ibkr_harvest`` tag and/or a single fill_id) while still seeding
    legitimate FIFO closes ([open_id, close_id], no tag).

(b) HARVESTER (source removal): harvest() writes FILLS_*.ndjson but emits
    NO trade_history_*.ndjson / closed_trade.v1 records.

(c) E2E (hermetic): a new open fill — even with a pre-existing phantom
    record for the same fill_id — is NOT pre-marked processed, IS enqueued
    by process_fills, and the (strategy, symbol) FIFO queue is non-empty.

Discriminator validated against production data 2026-06-03: 2,432
closed_trade.v1 records — (ibkr_harvest, len=1): 655 phantoms;
(no tag, len=2): 1,777 legit; zero crossover in either direction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chad.execution.trade_closer import TradeCloser


DATE = "20260603"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_closer(tmp_path: Path) -> TradeCloser:
    fills_dir = tmp_path / "fills"
    trades_dir = tmp_path / "trades"
    fills_dir.mkdir(exist_ok=True)
    trades_dir.mkdir(exist_ok=True)
    return TradeCloser(
        fills_dir=fills_dir,
        trades_dir=trades_dir,
        state_path=tmp_path / "trade_closer_state.json",
        position_guard_path=tmp_path / "position_guard.json",  # absent → meta {}
    )


def _write_ndjson(path: Path, payloads: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps({"payload": p}) + "\n")


def _phantom(fill_id: str, symbol: str = "M6E", strategy: str = "omega_macro") -> dict:
    """Harvester phantom: pnl=0 round trip, single fill_id, ibkr_harvest tag."""
    ts = "2026-06-03T12:00:00Z"
    return {
        "schema_version": "closed_trade.v1",
        "strategy": strategy,
        "symbol": symbol,
        "side": "BUY",
        "pnl": 0.0,
        "entry_time_utc": ts,
        "exit_time_utc": ts,
        "fill_price": 1.16,
        "entry_price": 1.16,
        "exit_price": 1.16,
        "quantity": 2.0,
        "contract_multiplier": 12500.0,
        "notional": 29000.0,
        "fill_ids": [fill_id],
        "broker": "ibkr_paper",
        "account_id": "DUTEST",
        "is_live": False,
        "tags": ["paper", "filled", "ibkr_harvest", strategy],
    }


def _legit_close(open_id: str, close_id: str) -> dict:
    """Legitimate FIFO close: two fill_ids, no ibkr_harvest tag."""
    return {
        "schema_version": "closed_trade.v1",
        "strategy": "alpha",
        "symbol": "SPY",
        "side": "BUY",
        "pnl": 12.5,
        "entry_time_utc": "2026-06-03T10:00:00Z",
        "exit_time_utc": "2026-06-03T11:00:00Z",
        "entry_price": 100.0,
        "exit_price": 100.5,
        "quantity": 25.0,
        "contract_multiplier": 1.0,
        "fill_ids": [open_id, close_id],
        "broker": "ibkr_paper",
        "account_id": "DUTEST",
        "is_live": False,
        "tags": ["paper", "filled"],
    }


def _open_fill(fill_id: str, symbol: str = "M6E", strategy: str = "omega_macro") -> dict:
    """A trusted paper open fill as written to FILLS_*.ndjson."""
    ts = "2026-06-03T12:00:00Z"
    return {
        "schema_version": "paper_exec_fill.v4",
        "account_id": "DUTEST",
        "asset_class": "futures",
        "broker": "ibkr_paper",
        "fill_id": fill_id,
        "fill_price": 1.16,
        "fill_time_utc": ts,
        "entry_time_utc": ts,
        "exit_time_utc": ts,
        "quantity": 2.0,
        "reject": False,
        "side": "BUY",
        "status": "paper_fill",
        "strategy": strategy,
        "symbol": symbol,
        "extra": {"source_strategies": [strategy], "sec_type": "FUT"},
    }


# ---------------------------------------------------------------------------
# (a) seeder consumer defense
# ---------------------------------------------------------------------------

def test_seeder_ignores_phantom_keeps_legit(tmp_path) -> None:
    closer = _make_closer(tmp_path)
    _write_ndjson(
        closer.trades_dir / f"trade_history_{DATE}.ndjson",
        [
            _phantom("PHANTOM_OPEN_1"),               # tag + single id → ignored
            _legit_close("LEGIT_OPEN_1", "LEGIT_CLOSE_1"),  # 2 ids, no tag → seeded
        ],
    )
    added = closer.seed_processed_from_trade_history()
    assert "PHANTOM_OPEN_1" not in closer.processed_fill_ids, (
        "phantom fill_id must NOT be seeded — seeding it suppresses the open"
    )
    assert "LEGIT_OPEN_1" in closer.processed_fill_ids
    assert "LEGIT_CLOSE_1" in closer.processed_fill_ids
    assert added == 2


def test_seeder_ignores_tagless_single_id_record(tmp_path) -> None:
    """The len(fill_ids) < 2 leg of the discriminator binds on its own."""
    closer = _make_closer(tmp_path)
    rec = _phantom("SINGLE_ID_NO_TAG")
    rec["tags"] = ["paper", "filled"]  # tag removed — only the length leg fires
    _write_ndjson(closer.trades_dir / f"trade_history_{DATE}.ndjson", [rec])
    closer.seed_processed_from_trade_history()
    assert "SINGLE_ID_NO_TAG" not in closer.processed_fill_ids


def test_seeder_ignores_tagged_two_id_record(tmp_path) -> None:
    """The ibkr_harvest tag leg binds even with two fill_ids."""
    closer = _make_closer(tmp_path)
    rec = _legit_close("TAGGED_OPEN", "TAGGED_CLOSE")
    rec["tags"] = ["paper", "filled", "ibkr_harvest"]
    _write_ndjson(closer.trades_dir / f"trade_history_{DATE}.ndjson", [rec])
    closer.seed_processed_from_trade_history()
    assert "TAGGED_OPEN" not in closer.processed_fill_ids
    assert "TAGGED_CLOSE" not in closer.processed_fill_ids


# ---------------------------------------------------------------------------
# (b) harvester no longer emits trade history
# ---------------------------------------------------------------------------

class _FakeExecution:
    def __init__(self) -> None:
        self.execId = "EXEC-TEST-1"
        self.side = "BOT"
        self.shares = 2.0
        self.price = 1.16
        self.acctNumber = "DUTEST"
        self.time = datetime(2026, 6, 3, 12, 0, 0, tzinfo=timezone.utc)


class _FakeContract:
    def __init__(self) -> None:
        self.symbol = "M6E"
        self.secType = "FUT"
        self.multiplier = "12500"


class _FakeFill:
    def __init__(self) -> None:
        self.contract = _FakeContract()
        self.execution = _FakeExecution()


class _FakeIB:
    def fills(self):
        return [_FakeFill()]

    def positions(self):
        return []

    def disconnect(self):
        pass


def test_harvester_writes_fills_but_no_trade_history(tmp_path, monkeypatch) -> None:
    from chad.portfolio import ibkr_paper_fill_harvester as mod

    fills_dir = tmp_path / "fills"
    trades_dir = tmp_path / "trades"
    locks_dir = tmp_path / "locks"
    for d in (fills_dir, trades_dir, locks_dir):
        d.mkdir()

    monkeypatch.setattr(mod, "FILLS_DIR", fills_dir)
    monkeypatch.setattr(mod, "TRADES_DIR", trades_dir)
    monkeypatch.setattr(mod, "LOCKS_DIR", locks_dir)
    monkeypatch.setattr(mod, "HARVESTED_IDS_PATH", tmp_path / "harvested_fill_ids.json")
    monkeypatch.setattr(mod, "POSITION_GUARD_PATH", tmp_path / "position_guard.json")
    monkeypatch.setattr(mod, "connect_ibkr", lambda: _FakeIB())

    result = mod.harvest(dry_run=False)

    # FILLS written, dedupe state updated.
    fills_files = list(fills_dir.glob("FILLS_*.ndjson"))
    assert len(fills_files) == 1, "harvester must still write the fills ledger"
    line = fills_files[0].read_text(encoding="utf-8").strip().splitlines()[0]
    payload = json.loads(line)["payload"]
    assert payload["symbol"] == "M6E"
    assert payload["status"] == "paper_fill"
    assert result["fills_wrote"] == 1
    assert (tmp_path / "harvested_fill_ids.json").is_file(), "FILLS dedupe must persist"

    # NO trade history of any kind.
    assert list(trades_dir.glob("trade_history_*.ndjson")) == [], (
        "harvester must no longer emit closed_trade.v1 trade-history records"
    )
    assert result["trades_wrote"] == 0
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "Queued trade history" not in src, "trade-history emission code must be gone"


def test_harvester_dedupe_skips_already_harvested(tmp_path, monkeypatch) -> None:
    """FILLS dedupe is intact: a second harvest of the same exec_id writes nothing."""
    from chad.portfolio import ibkr_paper_fill_harvester as mod

    fills_dir = tmp_path / "fills"
    locks_dir = tmp_path / "locks"
    fills_dir.mkdir()
    locks_dir.mkdir()

    monkeypatch.setattr(mod, "FILLS_DIR", fills_dir)
    monkeypatch.setattr(mod, "TRADES_DIR", tmp_path / "trades")
    monkeypatch.setattr(mod, "LOCKS_DIR", locks_dir)
    monkeypatch.setattr(mod, "HARVESTED_IDS_PATH", tmp_path / "harvested_fill_ids.json")
    monkeypatch.setattr(mod, "POSITION_GUARD_PATH", tmp_path / "position_guard.json")
    monkeypatch.setattr(mod, "connect_ibkr", lambda: _FakeIB())

    first = mod.harvest(dry_run=False)
    second = mod.harvest(dry_run=False)
    assert first["fills_wrote"] == 1
    assert second["fills_wrote"] == 0
    assert second["fills_skipped"] == 1
    lines = list(fills_dir.glob("FILLS_*.ndjson"))[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "duplicate exec_id must not produce a second fill record"


# ---------------------------------------------------------------------------
# (c) end-to-end: open fill enqueues despite a pre-existing phantom
# ---------------------------------------------------------------------------

def test_e2e_open_fill_enqueues_into_fifo(tmp_path) -> None:
    closer = _make_closer(tmp_path)
    fill_id = "OPEN_FILL_E2E_1"

    # A phantom for this very fill_id already on disk (historical pollution
    # shape) — the consumer defense must neutralize it.
    _write_ndjson(closer.trades_dir / f"trade_history_{DATE}.ndjson", [_phantom(fill_id)])
    _write_ndjson(closer.fills_dir / f"FILLS_{DATE}.ndjson", [_open_fill(fill_id)])

    closer.load_state()  # runs seed_processed_from_trade_history internally
    assert fill_id not in closer.processed_fill_ids, (
        "seeder must not pre-mark the open as consumed"
    )

    closed = closer.process_fills(DATE)
    assert closed == [], "an opening fill must not emit closed trades"
    assert fill_id in closer.processed_fill_ids, "fill must be processed exactly once"

    queue = closer.queues.get(("omega_macro", "M6E"))
    assert queue, "(omega_macro, M6E) FIFO queue must hold the opening lot"
    lot = queue[0]
    assert lot["fill_id"] == fill_id
    assert lot["side"] == "BUY"
    assert float(lot["quantity"]) == 2.0

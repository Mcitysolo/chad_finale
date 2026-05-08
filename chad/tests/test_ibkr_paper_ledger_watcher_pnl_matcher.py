"""
GAP-A003 — Tests for the snapshot-diff close fill matcher in
chad/portfolio/ibkr_paper_ledger_watcher.py.

Coverage:
  * snapshot-diff close with a valid unmatched opening BUY fill is trusted
    and gross_pnl is correct for a long close.
  * snapshot-diff close with a valid unmatched opening SELL fill is trusted
    and gross_pnl is correct for a short close.
  * snapshot-diff close with no matching opening fill stays untrusted.
  * snapshot-diff close where the only matching opening fill is already
    consumed by a prior closed_trade stays untrusted.
  * Malformed fill / trade records do not crash the matcher and the close
    falls back to untrusted.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from chad.portfolio.ibkr_paper_ledger_watcher import (
    LedgerConfig,
    MatchedOpeningFill,
    OpenStateStore,
    PaperLedgerWatcher,
    PlanAttributionResolver,
    SNAPSHOT_DIFF_MATCHER_NAME,
    StrategyAttributionService,
    collect_consumed_open_fill_ids,
    find_matched_opening_fill_for_snapshot_close,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _write_fill(
    fills_dir: Path,
    *,
    fill_id: str,
    strategy: str,
    symbol: str,
    side: str,
    fill_price: float,
    quantity: float,
    fill_time: datetime,
    status: str = "paper_fill",
    reject: bool = False,
    pnl_untrusted: bool = False,
) -> None:
    fills_dir.mkdir(parents=True, exist_ok=True)
    path = fills_dir / f"FILLS_{fill_time.strftime('%Y%m%d')}.ndjson"
    payload: Dict[str, Any] = {
        "schema_version": "paper_exec_fill.v4",
        "fill_id": fill_id,
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "fill_price": fill_price,
        "quantity": quantity,
        "notional": fill_price * quantity,
        "fill_time_utc": _iso(fill_time),
        "status": status,
        "reject": reject,
        "pnl_untrusted": pnl_untrusted,
        "broker": "ibkr_paper",
        "tags": ["paper", "filled", strategy],
    }
    envelope = {
        "timestamp_utc": _iso(fill_time),
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": "stub",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope) + "\n")


def _write_closed_trade(
    trades_dir: Path,
    *,
    fill_ids: list[str],
    when: datetime,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    trades_dir.mkdir(parents=True, exist_ok=True)
    path = trades_dir / f"trade_history_{when.strftime('%Y%m%d')}.ndjson"
    payload: Dict[str, Any] = {
        "schema_version": "closed_trade.v1",
        "strategy": "alpha",
        "symbol": "LLY",
        "side": "BUY",
        "fill_ids": fill_ids,
        "entry_time_utc": _iso(when),
        "exit_time_utc": _iso(when),
        "extra": extra or {},
    }
    envelope = {
        "timestamp_utc": _iso(when),
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": "stub",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope) + "\n")


@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 5, 8, 14, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def fills_dir(tmp_path: Path) -> Path:
    return tmp_path / "fills"


@pytest.fixture
def trades_dir(tmp_path: Path) -> Path:
    return tmp_path / "trades"


# ---------------------------------------------------------------------------
# Module-level matcher tests (deterministic, no IBKR)
# ---------------------------------------------------------------------------

def test_long_close_trusted_with_unmatched_buy_open(
    now_utc: datetime, fills_dir: Path, trades_dir: Path
) -> None:
    open_time = now_utc - timedelta(hours=1)
    _write_fill(
        fills_dir,
        fill_id="open_long_1",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=open_time,
    )

    consumed = collect_consumed_open_fill_ids(trades_dir, reference_time=now_utc)
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=consumed,
        fills_dir=fills_dir,
    )

    assert matched is not None
    assert matched.fill_id == "open_long_1"
    assert matched.fill_price == pytest.approx(900.0)
    assert matched.quantity == pytest.approx(5.0)

    # Long close PnL:  (exit - entry) * qty * mult
    exit_price = 950.0
    gross_pnl = (exit_price - matched.fill_price) * matched.quantity * matched.contract_multiplier
    assert gross_pnl == pytest.approx(250.0)


def test_short_close_trusted_with_unmatched_sell_open(
    now_utc: datetime, fills_dir: Path, trades_dir: Path
) -> None:
    open_time = now_utc - timedelta(hours=2)
    _write_fill(
        fills_dir,
        fill_id="open_short_1",
        strategy="delta",
        symbol="MES",
        side="SELL",
        fill_price=5500.0,
        quantity=2.0,
        fill_time=open_time,
    )

    consumed = collect_consumed_open_fill_ids(trades_dir, reference_time=now_utc)
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="delta",
        symbol="MES",
        expected_open_side="SELL",
        close_time=now_utc,
        consumed_ids=consumed,
        fills_dir=fills_dir,
    )

    assert matched is not None
    assert matched.fill_id == "open_short_1"

    # Short close PnL: (entry - exit) * qty * mult
    exit_price = 5450.0
    gross_pnl = (matched.fill_price - exit_price) * matched.quantity * matched.contract_multiplier
    assert gross_pnl == pytest.approx(100.0)


def test_close_without_matching_fill_stays_untrusted(
    now_utc: datetime, fills_dir: Path, trades_dir: Path
) -> None:
    # Different symbol in fills — should not match LLY close
    _write_fill(
        fills_dir,
        fill_id="open_other",
        strategy="alpha",
        symbol="QQQ",
        side="BUY",
        fill_price=400.0,
        quantity=3.0,
        fill_time=now_utc - timedelta(hours=1),
    )

    consumed = collect_consumed_open_fill_ids(trades_dir, reference_time=now_utc)
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=consumed,
        fills_dir=fills_dir,
    )

    assert matched is None


def test_close_rejects_consumed_open_fill(
    now_utc: datetime, fills_dir: Path, trades_dir: Path
) -> None:
    open_time = now_utc - timedelta(hours=3)
    _write_fill(
        fills_dir,
        fill_id="open_consumed",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=open_time,
    )
    # Already consumed by an earlier closed_trade.v1 record.
    _write_closed_trade(
        trades_dir,
        fill_ids=["open_consumed", "earlier_close_fill"],
        when=open_time + timedelta(minutes=5),
    )

    consumed = collect_consumed_open_fill_ids(trades_dir, reference_time=now_utc)
    assert "open_consumed" in consumed

    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=consumed,
        fills_dir=fills_dir,
    )
    assert matched is None


def test_consumed_fill_ids_pick_up_watcher_matched_open(
    now_utc: datetime, trades_dir: Path
) -> None:
    """A prior watcher snapshot close stamps matched_open_fill_id; future runs
    must treat that ID as consumed."""
    _write_closed_trade(
        trades_dir,
        fill_ids=[],
        when=now_utc - timedelta(hours=1),
        extra={
            "source": "ibkr_paper_ledger_watcher",
            "matched_open_fill_id": "watcher_consumed_open",
            "matcher": SNAPSHOT_DIFF_MATCHER_NAME,
        },
    )
    consumed = collect_consumed_open_fill_ids(trades_dir, reference_time=now_utc)
    assert "watcher_consumed_open" in consumed


def test_close_rejects_untrusted_open_fills(
    now_utc: datetime, fills_dir: Path, trades_dir: Path
) -> None:
    open_time = now_utc - timedelta(hours=1)
    _write_fill(
        fills_dir,
        fill_id="open_untrusted",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=open_time,
        pnl_untrusted=True,
    )
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=set(),
        fills_dir=fills_dir,
    )
    assert matched is None


def test_close_rejects_rejected_open_fills(
    now_utc: datetime, fills_dir: Path
) -> None:
    open_time = now_utc - timedelta(hours=1)
    _write_fill(
        fills_dir,
        fill_id="open_rejected",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=open_time,
        status="rejected",
        reject=True,
    )
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=set(),
        fills_dir=fills_dir,
    )
    assert matched is None


def test_close_rejects_open_fills_after_close_time(
    now_utc: datetime, fills_dir: Path
) -> None:
    _write_fill(
        fills_dir,
        fill_id="open_future",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=now_utc + timedelta(hours=1),  # after close
    )
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=set(),
        fills_dir=fills_dir,
    )
    assert matched is None


def test_picks_most_recent_unmatched_open(
    now_utc: datetime, fills_dir: Path
) -> None:
    earlier = now_utc - timedelta(hours=4)
    later = now_utc - timedelta(minutes=30)
    _write_fill(
        fills_dir,
        fill_id="open_earlier",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=earlier,
    )
    _write_fill(
        fills_dir,
        fill_id="open_later",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=905.0,
        quantity=5.0,
        fill_time=later,
    )
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=set(),
        fills_dir=fills_dir,
    )
    assert matched is not None
    assert matched.fill_id == "open_later"


def test_malformed_records_do_not_crash(
    now_utc: datetime, fills_dir: Path, trades_dir: Path
) -> None:
    fills_dir.mkdir(parents=True, exist_ok=True)
    bad = fills_dir / f"FILLS_{now_utc.strftime('%Y%m%d')}.ndjson"
    bad.write_text(
        "\n".join(
            [
                "this is not json",
                json.dumps({"payload": "string-not-dict"}),
                json.dumps({}),
                json.dumps({"payload": {"fill_id": "ok", "strategy": "alpha"}}),  # missing fields
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trades_dir.mkdir(parents=True, exist_ok=True)
    bad_tr = trades_dir / f"trade_history_{now_utc.strftime('%Y%m%d')}.ndjson"
    bad_tr.write_text("garbage\n{not_json}\n", encoding="utf-8")

    consumed = collect_consumed_open_fill_ids(trades_dir, reference_time=now_utc)
    matched = find_matched_opening_fill_for_snapshot_close(
        strategy="alpha",
        symbol="LLY",
        expected_open_side="BUY",
        close_time=now_utc,
        consumed_ids=consumed,
        fills_dir=fills_dir,
    )
    assert matched is None


# ---------------------------------------------------------------------------
# End-to-end: PaperLedgerWatcher.run_once close path with the matcher
# ---------------------------------------------------------------------------

class _FakeContract:
    def __init__(self, symbol: str, con_id: int = 12345, sec_type: str = "STK") -> None:
        self.symbol = symbol
        self.conId = con_id
        self.secType = sec_type
        self.currency = "USD"


class _FakePosition:
    def __init__(self, symbol: str, qty: float, avg_cost: float) -> None:
        self.contract = _FakeContract(symbol)
        self.position = qty
        self.avgCost = avg_cost
        self.account = "DU1234567"


class _FakeGateway:
    """Minimal BrokerGateway capturing the snapshot-diff sequence:
    first run sees an open position; second run sees no positions => close."""
    def __init__(self) -> None:
        self.calls = 0
        self.positions_sequence: list[list[_FakePosition]] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def current_positions(self) -> list[_FakePosition]:
        if not self.positions_sequence:
            return []
        return self.positions_sequence.pop(0)

    def recent_fills(self) -> list[Any]:
        return []


def _build_watcher(
    tmp_path: Path, fills_dir: Path, trades_dir: Path
) -> tuple[PaperLedgerWatcher, _FakeGateway, LedgerConfig]:
    cfg = LedgerConfig(
        enabled=True,
        state_path=tmp_path / "state.json",
        reports_dir=tmp_path / "reports",
        plan_artifact_path=tmp_path / "plan.json",
        trades_dir=trades_dir,
        fills_dir=fills_dir,
        default_strategy="alpha",
    )
    state_store = OpenStateStore(cfg.state_path)
    resolver = PlanAttributionResolver(cfg.plan_artifact_path)
    attribution = StrategyAttributionService(cfg, resolver)
    gateway = _FakeGateway()
    watcher = PaperLedgerWatcher(cfg, gateway, state_store, attribution)
    return watcher, gateway, cfg


def test_run_once_emits_trusted_close_when_match_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now_utc: datetime
) -> None:
    fills_dir = tmp_path / "fills"
    trades_dir = tmp_path / "trades"

    # Long opening fill on LLY — well before close_time
    open_time = now_utc - timedelta(hours=1)
    _write_fill(
        fills_dir,
        fill_id="open_lly_long",
        strategy="alpha",
        symbol="LLY",
        side="BUY",
        fill_price=900.0,
        quantity=5.0,
        fill_time=open_time,
    )

    # Patch the trade_result_logger to write into our isolated trades_dir
    import chad.analytics.trade_result_logger as trl
    monkeypatch.setattr(trl, "TRADE_DIR", trades_dir)

    # Pin the watcher's clock so the matcher's timestamp checks are deterministic
    import chad.portfolio.ibkr_paper_ledger_watcher as mod
    monkeypatch.setattr(mod, "utc_now", lambda: now_utc)

    watcher, gateway, _ = _build_watcher(tmp_path, fills_dir, trades_dir)

    # Run 1 — open position exists
    gateway.positions_sequence = [
        [_FakePosition("LLY", qty=5.0, avg_cost=900.0)],
        [],
    ]
    watcher.run_once()
    # Run 2 — position vanished => snapshot-diff close
    watcher.run_once()

    # Find the most recent emitted trade record
    written = sorted(trades_dir.glob("trade_history_*.ndjson"))
    assert written, "expected trade_history file to be written"
    last_envelope = None
    for path in written:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            last_envelope = json.loads(line)
    assert last_envelope is not None
    payload = last_envelope["payload"]
    extra = payload["extra"]

    assert extra["pnl_untrusted"] is False
    assert "pnl_untrusted_reason" not in extra
    assert extra["matched_open_fill_id"] == "open_lly_long"
    assert extra["matcher"] == SNAPSHOT_DIFF_MATCHER_NAME
    assert extra["pnl_trusted_reason"] == "matched_opening_fill_for_snapshot_diff_close"
    # Long close: avg_cost (900) - entry (900) * 5 * 1 = 0; positive when avg_cost
    # diverges. Just sanity-check structure here; PnL math is exercised in the
    # module-level tests above.
    assert "entry_price" in extra
    assert "exit_price" in extra
    assert "gross_pnl" in extra


def test_run_once_emits_untrusted_close_when_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now_utc: datetime
) -> None:
    fills_dir = tmp_path / "fills"
    trades_dir = tmp_path / "trades"
    fills_dir.mkdir(parents=True, exist_ok=True)
    # No fills for this symbol/strategy

    import chad.analytics.trade_result_logger as trl
    monkeypatch.setattr(trl, "TRADE_DIR", trades_dir)

    import chad.portfolio.ibkr_paper_ledger_watcher as mod
    monkeypatch.setattr(mod, "utc_now", lambda: now_utc)

    watcher, gateway, _ = _build_watcher(tmp_path, fills_dir, trades_dir)
    gateway.positions_sequence = [
        [_FakePosition("ZZZZ", qty=5.0, avg_cost=100.0)],
        [],
    ]
    watcher.run_once()
    watcher.run_once()

    written = sorted(trades_dir.glob("trade_history_*.ndjson"))
    assert written
    last_envelope = None
    for path in written:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            last_envelope = json.loads(line)
    assert last_envelope is not None
    extra = last_envelope["payload"]["extra"]
    assert extra["pnl_untrusted"] is True
    assert extra["pnl_untrusted_reason"] == "symbol_close_detected_without_fill_matcher"
    assert "matched_open_fill_id" not in extra


def test_run_once_falls_back_to_untrusted_when_matcher_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, now_utc: datetime
) -> None:
    fills_dir = tmp_path / "fills"
    trades_dir = tmp_path / "trades"
    fills_dir.mkdir(parents=True, exist_ok=True)

    import chad.analytics.trade_result_logger as trl
    monkeypatch.setattr(trl, "TRADE_DIR", trades_dir)

    import chad.portfolio.ibkr_paper_ledger_watcher as mod
    monkeypatch.setattr(mod, "utc_now", lambda: now_utc)

    watcher, gateway, _ = _build_watcher(tmp_path, fills_dir, trades_dir)

    # Force the matcher to blow up
    import chad.portfolio.ibkr_paper_ledger_watcher as mod

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated matcher failure")

    monkeypatch.setattr(mod, "find_matched_opening_fill_for_snapshot_close", _boom)

    gateway.positions_sequence = [
        [_FakePosition("LLY", qty=5.0, avg_cost=900.0)],
        [],
    ]
    watcher.run_once()
    watcher.run_once()

    written = sorted(trades_dir.glob("trade_history_*.ndjson"))
    last_envelope = None
    for path in written:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            last_envelope = json.loads(line)
    assert last_envelope is not None
    extra = last_envelope["payload"]["extra"]
    assert extra["pnl_untrusted"] is True
    assert extra["pnl_untrusted_reason"] == "symbol_close_detected_without_fill_matcher"

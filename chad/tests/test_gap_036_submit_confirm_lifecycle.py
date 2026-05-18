"""GAP-036 (Phase-53/54) — submit→confirm lifecycle fix unit tests.

Covers:
  - _SQLiteIdempotencyStore.claim_or_reclaim() state machine S1–S5
    (fresh / terminal-positive / terminal-negative / non-terminal-not-stale /
    non-terminal-stale-with-probe ACTIVE/ABSENT/TERMINAL).
  - trade.statusEvent handler promoting PendingSubmit → Filled and
    PendingSubmit → Cancelled into the SQLite store.
  - FakeIB end-to-end submit→Filled (single shot) and submit→Cancelled→retry.
  - MANDATORY: test_absent_but_executed_blocks_no_double_submit — stale
    non-terminal row, openTrades=ABSENT BUT executions/fills shows a fill for
    that broker_order_id → row promoted Filled, claim_or_reclaim returns False
    (NO reclaim, NO double submit). Closes the restart-gap double-submit hole.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

import pytest

# CHAD_SKIP_IB_CONNECT keeps any transitive import of live_loop safe.
os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

from chad.execution.ibkr_adapter import (  # noqa: E402
    IbkrAdapter,
    IbkrConfig,
    _SQLiteIdempotencyStore,
    _PROBE_ABSENT,
    _PROBE_STILL_ACTIVE,
    _PROBE_TERMINAL_PREFIX,
    _classify_idempotency_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEY = "test-key-abc"
PAYLOAD = {"symbol": "TEST", "side": "BUY", "quantity": 1.0}


def _now() -> datetime:
    return datetime(2026, 5, 18, 17, 0, 0, tzinfo=timezone.utc)


def _store(tmp_path: Path) -> _SQLiteIdempotencyStore:
    return _SQLiteIdempotencyStore(tmp_path / "exec_state.sqlite3")


def _seed_row(
    store: _SQLiteIdempotencyStore,
    *,
    status: str,
    age_seconds: float = 0.0,
    broker_order_id: Optional[int] = None,
    now: Optional[datetime] = None,
) -> datetime:
    """Insert a row with the given status, then back-date updated_at_utc."""
    base_now = now or _now()
    store.claim(KEY, PAYLOAD, base_now)
    if status.lower() != "claimed":
        store.mark(
            KEY,
            status=status,
            broker_order_id=broker_order_id,
            result={"order_status": {"status": status}},
            now=base_now,
        )
    if age_seconds > 0.0:
        old = base_now - timedelta(seconds=age_seconds)
        # Directly rewrite updated_at_utc to simulate staleness.
        with store._lock, store._connect() as conn:  # type: ignore[attr-defined]
            conn.execute(
                "UPDATE ibkr_exec_state SET updated_at_utc = ? WHERE idempotency_key = ?",
                (old.isoformat(), KEY),
            )
    return base_now


# ---------------------------------------------------------------------------
# Probe stubs
# ---------------------------------------------------------------------------

def probe_always_active(_broker_order_id: Optional[int]) -> str:
    return _PROBE_STILL_ACTIVE


def probe_always_absent(_broker_order_id: Optional[int]) -> str:
    return _PROBE_ABSENT


def probe_terminal_filled(_broker_order_id: Optional[int]) -> str:
    return _PROBE_TERMINAL_PREFIX + "Filled"


def probe_terminal_cancelled(_broker_order_id: Optional[int]) -> str:
    return _PROBE_TERMINAL_PREFIX + "Cancelled"


# ---------------------------------------------------------------------------
# 1) claim_or_reclaim — fresh row
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_fresh_returns_true(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_active
    ) is True
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "claimed"


# ---------------------------------------------------------------------------
# 2) claim_or_reclaim — non-terminal, not stale → block
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_pending_non_stale_returns_false(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_row(store, status="PendingSubmit", age_seconds=30.0)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_absent
    ) is False
    # Row preserved (not reclaimed).
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "pendingsubmit"


# ---------------------------------------------------------------------------
# 3) claim_or_reclaim — Filled UNCONDITIONALLY blocks
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_filled_always_blocks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_row(store, status="Filled", age_seconds=3600.0, broker_order_id=42)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_absent
    ) is False
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "filled"


# ---------------------------------------------------------------------------
# 4) claim_or_reclaim — terminal-negative (Cancelled) permits retry
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_cancelled_permits_retry(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_row(store, status="Cancelled", age_seconds=10.0, broker_order_id=99)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_active
    ) is True
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "claimed"
    # Old broker_order_id is cleared by the fresh INSERT.
    assert row["broker_order_id"] in (None, 0)


# ---------------------------------------------------------------------------
# 5) claim_or_reclaim — stale non-terminal + probe STILL_ACTIVE → block
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_stale_pending_probe_active_blocks(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_row(store, status="PendingSubmit", age_seconds=700.0, broker_order_id=123)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_active
    ) is False
    # updated_at bumped — row preserved as PendingSubmit but timestamp advanced.
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "pendingsubmit"


# ---------------------------------------------------------------------------
# 6) claim_or_reclaim — stale non-terminal + probe ABSENT → reclaim
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_stale_pending_probe_absent_reclaims(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_row(store, status="PendingSubmit", age_seconds=900.0, broker_order_id=555)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_absent
    ) is True
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "claimed"


# ---------------------------------------------------------------------------
# 7) claim_or_reclaim — stale non-terminal + probe TERMINAL_AT_BROKER:Filled
# → row promoted to Filled, returns False (no reclaim)
# ---------------------------------------------------------------------------

def test_claim_or_reclaim_stale_pending_probe_terminal_filled_blocks(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _seed_row(store, status="PendingSubmit", age_seconds=900.0, broker_order_id=777)
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_terminal_filled
    ) is False
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "filled"


# ---------------------------------------------------------------------------
# 8) trade.statusEvent handler promotes PendingSubmit → Filled
# ---------------------------------------------------------------------------

@dataclass
class _FakeOrder:
    orderId: int = 101
    permId: int = 0


@dataclass
class _FakeOrderStatus:
    status: str = "PendingSubmit"


@dataclass
class _FakeStatusEvent:
    subscribers: List[Callable[[Any], None]] = field(default_factory=list)

    def __iadd__(self, handler: Callable[[Any], None]) -> "_FakeStatusEvent":
        self.subscribers.append(handler)
        return self

    def emit(self, trade: Any) -> None:
        for h in list(self.subscribers):
            h(trade)


@dataclass
class _FakeTrade:
    order: _FakeOrder = field(default_factory=_FakeOrder)
    orderStatus: _FakeOrderStatus = field(default_factory=_FakeOrderStatus)
    fills: List[Any] = field(default_factory=list)
    statusEvent: _FakeStatusEvent = field(default_factory=_FakeStatusEvent)
    commissionReport: List[Any] = field(default_factory=list)


def _make_adapter_with_store(tmp_path: Path) -> IbkrAdapter:
    cfg = IbkrConfig(
        dry_run=False,
        state_db_path=tmp_path / "exec_state.sqlite3",
        terminal_wait_s=0.0,  # tests drive transitions explicitly
        initial_status_wait_s=0.0,
    )
    # Bypass _lazy_import_ib_factory which raises by design.
    adapter = IbkrAdapter(
        config=cfg,
        ib_factory=lambda: None,  # type: ignore[arg-type]
    )
    return adapter


def test_orderstatus_handler_promotes_pending_to_filled(tmp_path: Path) -> None:
    adapter = _make_adapter_with_store(tmp_path)
    store = adapter._idempotency
    assert store is not None
    store.claim(KEY, PAYLOAD, _now())
    trade = _FakeTrade()
    adapter._install_trade_status_handler(trade, KEY)
    # PendingSubmit event.
    trade.orderStatus.status = "PendingSubmit"
    trade.statusEvent.emit(trade)
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "pendingsubmit"
    # Filled event.
    trade.orderStatus.status = "Filled"
    trade.statusEvent.emit(trade)
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "filled"


# ---------------------------------------------------------------------------
# 9) handler promotes PendingSubmit → Cancelled and a fresh claim_or_reclaim
#    is permitted afterwards (terminal-negative retry path)
# ---------------------------------------------------------------------------

def test_orderstatus_handler_promotes_pending_to_cancelled(tmp_path: Path) -> None:
    adapter = _make_adapter_with_store(tmp_path)
    store = adapter._idempotency
    assert store is not None
    store.claim(KEY, PAYLOAD, _now())
    trade = _FakeTrade()
    adapter._install_trade_status_handler(trade, KEY)
    trade.orderStatus.status = "Cancelled"
    trade.statusEvent.emit(trade)
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "cancelled"
    # A subsequent claim_or_reclaim succeeds (retry allowed).
    assert store.claim_or_reclaim(
        KEY, PAYLOAD, _now(), stale_threshold_s=600.0, ib_probe=probe_always_active
    ) is True


# ---------------------------------------------------------------------------
# 10) FakeIB end-to-end: submit → Filled, second submit blocks duplicate
# ---------------------------------------------------------------------------

@dataclass
class _FakeIB:
    """Minimal IB stand-in for end-to-end probe + handler tests."""

    open_trades: List[_FakeTrade] = field(default_factory=list)
    all_trades: List[_FakeTrade] = field(default_factory=list)
    fills_list: List[Any] = field(default_factory=list)

    def openTrades(self) -> List[_FakeTrade]:
        return list(self.open_trades)

    def trades(self) -> List[_FakeTrade]:
        return list(self.all_trades)

    def fills(self) -> List[Any]:
        return list(self.fills_list)

    def sleep(self, _s: float) -> None:
        return None


def test_submit_to_filled_lifecycle_end_to_end(tmp_path: Path) -> None:
    """Submit → trade emits Filled → row terminal-positive → re-claim blocks."""
    adapter = _make_adapter_with_store(tmp_path)
    store = adapter._idempotency
    assert store is not None
    fake_ib = _FakeIB()

    # First "submit": claim_or_reclaim returns True (fresh), then statusEvent
    # promotes to Filled.
    claimed_1 = store.claim_or_reclaim(
        KEY,
        PAYLOAD,
        _now(),
        stale_threshold_s=600.0,
        ib_probe=lambda boid: adapter._ib_probe(fake_ib, boid),
    )
    assert claimed_1 is True
    trade = _FakeTrade()
    adapter._install_trade_status_handler(trade, KEY)
    trade.orderStatus.status = "Filled"
    trade.statusEvent.emit(trade)

    # Second "submit": Filled → unconditional block.
    claimed_2 = store.claim_or_reclaim(
        KEY,
        PAYLOAD,
        _now() + timedelta(seconds=5),
        stale_threshold_s=600.0,
        ib_probe=lambda boid: adapter._ib_probe(fake_ib, boid),
    )
    assert claimed_2 is False
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "filled"


# ---------------------------------------------------------------------------
# 11) FakeIB end-to-end: submit → Cancelled → retry succeeds
# ---------------------------------------------------------------------------

def test_submit_to_cancelled_then_retry(tmp_path: Path) -> None:
    adapter = _make_adapter_with_store(tmp_path)
    store = adapter._idempotency
    assert store is not None
    fake_ib = _FakeIB()

    claimed_1 = store.claim_or_reclaim(
        KEY,
        PAYLOAD,
        _now(),
        stale_threshold_s=600.0,
        ib_probe=lambda boid: adapter._ib_probe(fake_ib, boid),
    )
    assert claimed_1 is True
    trade = _FakeTrade()
    adapter._install_trade_status_handler(trade, KEY)
    trade.orderStatus.status = "Cancelled"
    trade.statusEvent.emit(trade)

    claimed_2 = store.claim_or_reclaim(
        KEY,
        PAYLOAD,
        _now() + timedelta(seconds=5),
        stale_threshold_s=600.0,
        ib_probe=lambda boid: adapter._ib_probe(fake_ib, boid),
    )
    assert claimed_2 is True
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "claimed"


# ---------------------------------------------------------------------------
# 12) MANDATORY (Phase-54): ABSENT-but-executed blocks reclaim
# ---------------------------------------------------------------------------

@dataclass
class _FakeExecution:
    orderId: int = 0
    permId: int = 0


@dataclass
class _FakeFill:
    execution: _FakeExecution = field(default_factory=_FakeExecution)


def test_absent_but_executed_blocks_no_double_submit(tmp_path: Path) -> None:
    """Stale non-terminal row + openTrades/trades empty (ABSENT) BUT
    ib.fills() carries an execution for the same broker_order_id. The probe
    MUST detect the execution and return TERMINAL_AT_BROKER:Filled rather
    than ABSENT — claim_or_reclaim then promotes to Filled and refuses to
    reclaim. Closes the restart-gap double-submit hole.
    """
    adapter = _make_adapter_with_store(tmp_path)
    store = adapter._idempotency
    assert store is not None

    broker_order_id = 31337
    # Seed a stale PendingSubmit row tied to broker_order_id=31337.
    _seed_row(
        store,
        status="PendingSubmit",
        age_seconds=900.0,
        broker_order_id=broker_order_id,
    )

    # Fake IB: openTrades + trades are EMPTY (the order is "gone" from open
    # state), but fills() carries an execution for orderId=31337.
    fake_ib = _FakeIB(
        open_trades=[],
        all_trades=[],
        fills_list=[_FakeFill(execution=_FakeExecution(orderId=broker_order_id))],
    )

    # The probe must return TERMINAL_AT_BROKER:Filled, NOT ABSENT.
    probe_result = adapter._ib_probe(fake_ib, broker_order_id)
    assert probe_result == _PROBE_TERMINAL_PREFIX + "Filled", probe_result

    # claim_or_reclaim with this probe must return False (no double submit).
    claimed = store.claim_or_reclaim(
        KEY,
        PAYLOAD,
        _now(),
        stale_threshold_s=600.0,
        ib_probe=lambda boid: adapter._ib_probe(fake_ib, boid),
    )
    assert claimed is False
    row = store.get(KEY)
    assert row is not None
    assert (row["status"] or "").lower() == "filled"


# ---------------------------------------------------------------------------
# 13) classify helper — sanity of the lower-case bucketing
# ---------------------------------------------------------------------------

def test_classify_idempotency_status_buckets() -> None:
    assert _classify_idempotency_status("Filled") == "terminal_positive"
    assert _classify_idempotency_status("filled") == "terminal_positive"
    assert _classify_idempotency_status("Cancelled") == "terminal_negative"
    assert _classify_idempotency_status("ApiCancelled") == "terminal_negative"
    assert _classify_idempotency_status("Rejected") == "terminal_negative"
    assert _classify_idempotency_status("Inactive") == "terminal_negative"
    assert _classify_idempotency_status("error") == "terminal_negative"
    assert _classify_idempotency_status("PendingSubmit") == "non_terminal"
    assert _classify_idempotency_status("PreSubmitted") == "non_terminal"
    assert _classify_idempotency_status("Submitted") == "non_terminal"
    assert _classify_idempotency_status("") == "non_terminal"
    assert _classify_idempotency_status(None) == "non_terminal"

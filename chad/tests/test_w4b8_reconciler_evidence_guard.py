"""W4B-8b (INCIDENT-0723 D2) — apply_close_intents never writes non-fill rows.

The reconciler close path used to write fill evidence for WHATEVER status the
adapter returned. On 2026-07-23 that spammed 465 market_closed + 15 dry_run
rows into data/fills/FILLS_20260723.ndjson (530/540 rows that day were
non-fill exhaust); the dry_run rows then fed FIFO netting and false-flatted
the position guard. These tests pin the writer-side refusal: a close that
never reached the broker produces NO money-ledger row, NO price scan, and NO
guard mutation — the guard stays open and the next cycle retries (same
contract as the PR-02b price-abstain branch).
"""

from __future__ import annotations

import logging
from typing import List

import pytest

from chad.core import position_reconciler as pr


CLOSE_INTENT = {
    "symbol": "IWM",
    "action": "CLOSE",
    "open_side": "BUY",
    "close_side": "SELL",
    "quantity": 10.0,
    "reason": "reconciler_flip_test",
    "position_key": "gamma|IWM",
    "strategy": "gamma",
}


def _fake_order(status: str):
    class _FakeOrder:
        symbol = "IWM"
        side = "SELL"
        quantity = 10.0
        submitted_at = None
        asset_class = "EQUITY"

    o = _FakeOrder()
    o.status = status
    return o


def _fake_adapter(status: str):
    class _FakeAdapter:
        def submit_strategy_trade_intents(self, intents):
            return [_fake_order(status)]

    return _FakeAdapter()


@pytest.fixture()
def capture_writes(monkeypatch):
    written: List[str] = []

    def _fake_write(ev):
        written.append(str(getattr(ev, "status", "")))
        return {"fill_id": "fake_fill", "fills_path": ""}

    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.write_paper_exec_evidence",
        _fake_write,
        raising=True,
    )
    # The evidence normalizer reads price_cache — irrelevant here; identity.
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.normalize_paper_fill_evidence",
        lambda ev: ev,
        raising=True,
    )
    # A resolvable price so only the STATUS gate decides the outcome.
    monkeypatch.setattr(pr, "_resolve_close_fill_price", lambda s: 295.0,
                        raising=True)
    return written


@pytest.fixture()
def guard_recorder(monkeypatch):
    mutations: List[str] = []
    monkeypatch.setattr(
        "chad.core.position_guard.write_position_guard",
        lambda state: mutations.append("write"),
        raising=True,
    )
    monkeypatch.setattr(
        "chad.core.position_guard._load_state",
        lambda: {"gamma|IWM": {"open": True}},
        raising=True,
    )
    return mutations


@pytest.mark.parametrize("status", sorted(pr._EVIDENCE_SKIP_FILL_STATUSES))
def test_nonfill_status_writes_nothing(status, capture_writes, guard_recorder,
                                       caplog):
    caplog.set_level(logging.WARNING, logger="chad.core.position_reconciler")
    pr.apply_close_intents([dict(CLOSE_INTENT)], _fake_adapter(status))
    assert capture_writes == [], (
        f"status={status!r} must not reach the money ledger"
    )
    assert guard_recorder == [], (
        f"status={status!r} must not mutate the position guard"
    )
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "RECONCILER_SKIP_EVIDENCE_NONFILL_STATUS" in msgs


def test_mixed_case_pseudo_status_is_skipped(capture_writes, guard_recorder):
    """The gate normalizes case — an adapter quirk cannot sneak a row in."""
    pr.apply_close_intents([dict(CLOSE_INTENT)], _fake_adapter("Dry_Run"))
    assert capture_writes == []
    assert guard_recorder == []


def test_genuine_fill_status_still_writes(capture_writes, guard_recorder):
    """Control: a real fill status flows through the writer and (with the
    fake fill_id confirmed) the guard close proceeds as before."""
    pr.apply_close_intents([dict(CLOSE_INTENT)], _fake_adapter("paper_fill"))
    assert capture_writes == ["paper_fill"]
    assert guard_recorder == ["write"]


def test_broker_reject_keeps_forensic_row(capture_writes, guard_recorder,
                                          caplog):
    """A genuine broker terminal answer (rejected) still writes its forensic
    row — downstream defense-in-depth excludes it from money truth — but the
    guard must NOT be mutated (ISSUE-29 reject branch)."""
    caplog.set_level(logging.WARNING, logger="chad.core.position_reconciler")
    pr.apply_close_intents([dict(CLOSE_INTENT)], _fake_adapter("rejected"))
    assert capture_writes == ["rejected"]
    assert guard_recorder == []
    assert "ISSUE29_GUARD_SKIP" in " | ".join(
        r.getMessage() for r in caplog.records
    )

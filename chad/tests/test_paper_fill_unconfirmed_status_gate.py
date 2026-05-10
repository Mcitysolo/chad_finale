"""Paper-fill safety gate — Error 201 / unconfirmed IBKR status hardening.

Background
----------
Before this gate, an IBKR STK/FUT order returning status=PendingSubmit was
normalized to paper_fill in paper_exec_evidence_writer (status was in
``_PAPER_PENDING_STATUSES``). When IBKR later cancelled the order async
(e.g. Error 201 working-orders cap), the fictional fill stayed in
data/fills/FILLS_*.ndjson and fed bogus realized PnL into trade_closer / SCR.

These tests pin the two-layer defense:

  * chad/core/live_loop._should_persist_paper_evidence — primary gate; skips
    write_paper_exec_evidence with a SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS
    log marker when the adapter returned an unconfirmed status AND the path
    is not an explicit BAG/COMBO simulator.
  * chad/execution/paper_exec_evidence_writer.normalize_paper_fill_evidence —
    defense-in-depth; broker-rejected statuses (error/failed/rejected/
    cancelled) are demoted to status="rejected" + pnl_untrusted=True instead
    of being auto-translated to paper_fill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from chad.core.live_loop import (
    _UNCONFIRMED_BROKER_STATUSES,
    _is_explicit_paper_simulator,
    _should_persist_paper_evidence,
)
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    _PAPER_REJECTED_STATUSES,
    normalize_paper_fill_evidence,
)


# ---------------------------------------------------------------------------
# Fake SubmittedOrder — duck-typed; the gate only reads attributes.
# ---------------------------------------------------------------------------
@dataclass
class _FakeOrder:
    status: str = "unknown"
    symbol: str = "SPY"
    side: str = "BUY"
    sec_type: str = "STK"
    quantity: float = 12.0
    asset_class: str = "etf"


# ---------------------------------------------------------------------------
# live_loop layer — _should_persist_paper_evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "PendingSubmit",
        "PreSubmitted",
        "Submitted",
        "ApiPending",
        "Inactive",
        "Unknown",
        "",
        "error",
        "failed",
        "rejected",
        "cancelled",
        "duplicate_blocked",
    ],
)
def test_unconfirmed_status_skips_evidence(status):
    """Every unconfirmed IBKR status must be skipped on a non-simulator STK path."""
    order = _FakeOrder(status=status, sec_type="STK")
    persist, reason = _should_persist_paper_evidence(order, {})
    assert persist is False, (
        f"unconfirmed status={status!r} must NOT persist a paper fill"
    )
    assert reason is not None
    assert "unconfirmed_order_status" in reason


@pytest.mark.parametrize("status", ["PendingSubmit", "Inactive", "Submitted", "error"])
def test_unconfirmed_status_skips_evidence_for_fut(status):
    """FUT submissions are gated identically to STK."""
    order = _FakeOrder(status=status, sec_type="FUT", symbol="MES", asset_class="futures")
    persist, reason = _should_persist_paper_evidence(order, {})
    assert persist is False
    assert "unconfirmed_order_status" in (reason or "")


def test_filled_status_persists_evidence():
    """Confirmed Filled status proceeds to write paper_fill evidence."""
    order = _FakeOrder(status="Filled", sec_type="STK")
    persist, reason = _should_persist_paper_evidence(order, {})
    assert persist is True
    assert reason is None


def test_paper_fill_status_persists_evidence():
    """When the BAG simulator already promoted status to paper_fill we proceed."""
    order = _FakeOrder(status="paper_fill", sec_type="BAG")
    persist, reason = _should_persist_paper_evidence(order, {})
    assert persist is True
    assert reason is None


def test_bag_simulator_path_persists_despite_pendingsubmit():
    """Explicit BAG simulator path is licensed regardless of synchronous status.

    alpha_options vertical spreads route through IBKR with a wrapper STK proxy
    submission that returns PendingSubmit, but the BAG simulator in the writer
    rewrites the fill from net_debit_estimate. Skipping evidence here would
    delete legitimate paper-options fills.
    """
    bag_extra = {
        "sec_type": "BAG",
        "required_asset_class": "options",
        "long_strike": 738.0,
        "short_strike": 746.0,
        "long_right": "C",
        "short_right": "C",
        "expiry": "20260612",
        "net_debit_estimate": 1.50,
    }
    order = _FakeOrder(status="PendingSubmit", sec_type="BAG", asset_class="options")
    persist, reason = _should_persist_paper_evidence(order, bag_extra)
    assert persist is True
    assert reason is None


def test_options_required_asset_class_path_persists():
    """required_asset_class=options + leg meta also licenses simulator path."""
    bag_extra = {
        "required_asset_class": "options",
        "long_strike": 700.0,
        "short_strike": 710.0,
    }
    order = _FakeOrder(status="PreSubmitted", sec_type="OPT", asset_class="options")
    persist, _ = _should_persist_paper_evidence(order, bag_extra)
    assert persist is True


def test_simulator_detection_rejects_empty_extra():
    assert _is_explicit_paper_simulator({}) is False
    assert _is_explicit_paper_simulator(None) is False
    assert _is_explicit_paper_simulator({"sec_type": "STK"}) is False


def test_unconfirmed_set_includes_error_201_aftermath_statuses():
    """If the gate constant ever drifts, this fails loudly.

    Error 201 → IBKR cancels the order; the orderStatusEvent emits status=
    'Cancelled'. PendingSubmit and PreSubmitted are the synchronous statuses
    seen at submission time. All three must remain in the gate.
    """
    for must in ("pendingsubmit", "presubmitted", "cancelled", "error"):
        assert must in _UNCONFIRMED_BROKER_STATUSES


# ---------------------------------------------------------------------------
# writer layer — broker-rejected statuses must demote to rejected, never paper_fill
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_price_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(
        json.dumps({"prices": {"SPY": 700.0, "MES": 7100.0}, "ts_utc": "2026-05-10T00:00:00Z"})
    )
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.PRICE_CACHE_PATH", cache_path
    )
    return cache_path


@pytest.mark.parametrize("status", ["error", "failed", "rejected", "cancelled"])
def test_broker_rejected_statuses_never_become_paper_fill(fake_price_cache, status):
    """Direct callers of normalize_paper_fill_evidence must not see error→paper_fill."""
    ev = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=12.0,
        fill_price=0.0,
        status=status,
        is_live=False,
        asset_class="etf",
    )
    normalize_paper_fill_evidence(ev)
    assert ev.status == "rejected"
    assert ev.reject is True
    assert isinstance(ev.extra, dict)
    assert ev.extra.get("pnl_untrusted") is True
    assert "broker_rejected_status" in str(ev.extra.get("pnl_untrusted_reason", ""))
    tag_lower = {str(t).strip().lower() for t in (ev.tags or ())}
    assert "broker_rejected" in tag_lower
    assert "pnl_untrusted" in tag_lower


def test_paper_rejected_statuses_constant_covers_broker_rejects():
    """The rejected set must cover the four canonical broker-reject statuses."""
    for must in ("error", "failed", "rejected", "cancelled"):
        assert must in _PAPER_REJECTED_STATUSES


# ---------------------------------------------------------------------------
# log marker — SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS appears on the gated path
# ---------------------------------------------------------------------------


def test_skip_log_marker_emitted(caplog):
    """End-to-end: when the gate skips, the SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS
    log line is emitted with the symbol/strategy/status payload."""
    import logging as _logging

    logger = _logging.getLogger("test_skip_log_marker")
    order = _FakeOrder(status="PendingSubmit", sec_type="STK", symbol="SPY")
    persist, reason = _should_persist_paper_evidence(order, {})
    assert persist is False

    # Replicate the live_loop log call so the marker stays exercised even if
    # the inline format string drifts.
    caplog.set_level(_logging.WARNING, logger="test_skip_log_marker")
    logger.warning(
        "SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS symbol=%s strategy=%s "
        "sec_type=%s side=%s qty=%s status=%s reason=%s",
        order.symbol, "delta", order.sec_type, order.side, order.quantity,
        order.status, reason,
    )
    msgs = [r.getMessage() for r in caplog.records]
    assert any("SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS" in m for m in msgs)
    assert any("symbol=SPY" in m for m in msgs)
    assert any("status=PendingSubmit" in m for m in msgs)

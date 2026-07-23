"""W4B-8f (drill follow-up 3) — pin every FILLS-consumer exclusion site.

INCIDENT-0723 proved a status=dry_run row in data/fills/FILLS_*.ndjson can
move real money if even ONE consumer trusts it. This file pins the full
consumer census (audits/INCIDENT_20260723_DRILL_EXHAUST_FALSE_FLAT.md §2)
so a future blessing of a rehearsal status anywhere fails loudly:

  1. trade_closer FIFO ingest        — chad/execution/trade_closer.py
     (root cause; replay pinned in test_incident_0723_dry_run_exclusion.py)
  2. reconciler evidence writer      — chad/core/position_reconciler.py
     (writer-side skip-set; pinned in test_w4b8_reconciler_evidence_guard.py)
  3. reconciler tier-1 price cascade — chad/core/position_reconciler.py
     (pinned in test_w4b8_price_cascade_status_gate.py)
  4. Stage-2 harness adapter         — chad/validation/trade_log_adapter.py
  5. paper ledger watcher            — chad/portfolio/ibkr_paper_ledger_watcher.py
  6. guard close confirmation        — chad/core/position_guard.py

SCR (trade_stats_engine) and expectancy_tracker read trade_history rows the
trade closer DERIVES — site 1 is their protection; the cross-site invariant
tests at the bottom pin that no trusted set anywhere re-admits a status the
writer-side skip-set names as exhaust.
"""

from __future__ import annotations

from datetime import datetime, timezone

from chad.core import position_reconciler as pr
from chad.core.position_guard import (
    _CONFIRMED_FILL_STATUSES,
    is_fill_confirmed,
)
from chad.execution import trade_closer as tc
from chad.portfolio import ibkr_paper_ledger_watcher as watcher
from chad.validation import trade_log_adapter as harness

EXHAUST_STATUSES = ("dry_run", "market_closed", "duplicate_blocked",
                    "PendingSubmit", "PendingCancel", "Submitted")


def _fill_payload(status: str) -> dict:
    return {
        "fill_id": "abc123",
        "strategy": "gamma",
        "symbol": "PSQ",
        "side": "SELL",
        "quantity": 5.0,
        "fill_price": 26.2,
        "status": status,
        "reject": False,
        "fill_time_utc": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Site 4 — Stage-2 harness adapter (chad/validation/trade_log_adapter.py)
# --------------------------------------------------------------------------- #

def test_harness_trusted_set_excludes_rehearsal_statuses():
    assert "dry_run" not in harness.TRUSTED_FILL_STATUSES
    assert harness.TRUSTED_FILL_STATUSES == frozenset({"paper_fill", "fill"})


def test_harness_rejects_dry_run_record():
    reason = harness.trust_exclusion({"payload": _fill_payload("dry_run")})
    assert reason == "non_fill_status"


def test_harness_admits_paper_fill_record():
    assert harness.trust_exclusion({"payload": _fill_payload("paper_fill")}) is None


# --------------------------------------------------------------------------- #
# Site 5 — paper ledger watcher (chad/portfolio/ibkr_paper_ledger_watcher.py)
# --------------------------------------------------------------------------- #

def _qualify(status: str) -> bool:
    ok, _ = watcher._fill_qualifies_as_open(
        _fill_payload(status),
        strategy="gamma", symbol="PSQ", expected_open_side="SELL",
        close_time=datetime.now(timezone.utc), consumed_ids=set(),
    )
    return ok


def test_watcher_rejects_dry_run_open_fill():
    assert _qualify("dry_run") is False


def test_watcher_admits_paper_fill_open_fill():
    assert _qualify("paper_fill") is True


# --------------------------------------------------------------------------- #
# Site 6 — guard close confirmation (chad/core/position_guard.py)
# --------------------------------------------------------------------------- #

def test_guard_confirmation_rejects_every_exhaust_status():
    for status in EXHAUST_STATUSES:
        assert not is_fill_confirmed(
            {"fill_id": "abc123", "status": status}
        ), f"is_fill_confirmed must reject {status!r}"


# --------------------------------------------------------------------------- #
# Cross-site invariants — a re-blessing anywhere fails here
# --------------------------------------------------------------------------- #

def test_no_trusted_set_intersects_the_writer_skip_set():
    """The reconciler's skip-set is the canonical list of never-money
    statuses. No consumer's trusted set may re-admit any of them."""
    exhaust = pr._EVIDENCE_SKIP_FILL_STATUSES
    for name, trusted in (
        ("trade_closer", tc._TRUSTED_FILL_STATUSES),
        ("harness", harness.TRUSTED_FILL_STATUSES),
        ("guard_confirmation", _CONFIRMED_FILL_STATUSES),
        ("price_cascade", pr._PRICE_ELIGIBLE_FILL_STATUSES),
    ):
        overlap = {s.lower() for s in trusted} & exhaust
        assert not overlap, f"{name} trusts exhaust statuses: {overlap}"


def test_dry_run_named_in_writer_skip_set():
    assert "dry_run" in pr._EVIDENCE_SKIP_FILL_STATUSES
    assert "market_closed" in pr._EVIDENCE_SKIP_FILL_STATUSES

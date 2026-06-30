"""Tests for the proven-flat broker-authority truth path.

Covers the lifecycle_truth_publisher.build_positions_truth() truth_ok /
broker_authority logic for a legitimately flat, broker-confirmed account, while
proving the existing non-empty reconciliation gate is unchanged.

All fixtures only — no broker, no network. We write plain JSON runtime files
into a tmp runtime_dir; build_positions_truth() reads them best-effort.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from chad.ops.lifecycle_truth_publisher import (
    BrokerEventsEvidence,
    LedgerEvidence,
    build_positions_truth,
)

# Evidence objects do not influence truth_ok / broker_authority_status (those
# depend only on upstream GREEN, ledger_state, snapshot/ledger counts, and
# cash). Minimal "absent" evidence keeps the fixtures honest.
_EMPTY_BROKER_EVENTS = BrokerEventsEvidence(
    exists=False,
    newest_file=None,
    newest_mtime_unix=None,
    last_event_ts_utc=None,
    event_count_hint=0,
)
_EMPTY_LEDGER_EVIDENCE = LedgerEvidence(
    exists=False,
    newest_file=None,
    newest_mtime_unix=None,
    line_count_hint=0,
)

_CASH_CONFIRMED = {
    "CAD": {
        "TotalCashValue": 1000000.0,
        "NetLiquidation": 1000133.17,
        "AvailableFunds": 1000133.17,
    }
}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _seed_runtime(
    runtime_dir: Path,
    *,
    positions: list,
    cash: Optional[Dict[str, Any]],
    reconciliation_status: str,
    ledger_open: Dict[str, Any],
) -> None:
    snapshot: Dict[str, Any] = {
        "positions": positions,
        "positions_count": len(positions),
    }
    if cash is not None:
        snapshot["cash"] = cash
    _write_json(runtime_dir / "positions_snapshot.json", snapshot)
    _write_json(
        runtime_dir / "reconciliation_state.json",
        {"status": reconciliation_status, "ts_utc": "2026-06-30T00:00:00Z"},
    )
    _write_json(
        runtime_dir / "ibkr_paper_ledger_state.json",
        {"open": ledger_open, "last_run_utc": "2026-06-30T00:00:00Z"},
    )


def _build(runtime_dir: Path) -> Dict[str, Any]:
    return build_positions_truth(
        repo_root=runtime_dir.parent,
        runtime_dir=runtime_dir,
        data_dir=runtime_dir.parent / "data",
        evidence=_EMPTY_BROKER_EVENTS,
        fills_evidence=_EMPTY_LEDGER_EVIDENCE,
        fees_evidence=_EMPTY_LEDGER_EVIDENCE,
    )


def test_a_flat_cash_confirmed_upstream_green_is_proven_flat(tmp_path: Path) -> None:
    """(a) flat + cash-confirmed + upstream GREEN => GREEN via flat-proven."""
    runtime_dir = tmp_path / "runtime"
    _seed_runtime(
        runtime_dir,
        positions=[],
        cash=_CASH_CONFIRMED,
        reconciliation_status="GREEN",
        ledger_open={},
    )

    payload = _build(runtime_dir)

    assert payload["truth_ok"] is True
    assert payload["broker_authority_status"] == "GREEN"
    assert payload["truth_source"] == "BROKER_AUTHORITY_FLAT_PROVEN"
    assert payload["broker_authority_reason"] == (
        "BROKER_AUTHORITY_GREEN: flat_proven cash_confirmed snapshot=0 ledger=0"
    )
    # Replay diagnostics must not block truth on the flat-proven path.
    assert payload["replay_diagnostic_blocks_truth"] is False
    # Cash is surfaced in the truth artifact.
    assert payload["cash"] == _CASH_CONFIRMED


def test_b_flat_but_cash_empty_is_red(tmp_path: Path) -> None:
    """(b) flat + cash EMPTY => RED (broker did not confirm cash)."""
    runtime_dir = tmp_path / "runtime"
    _seed_runtime(
        runtime_dir,
        positions=[],
        cash={},
        reconciliation_status="GREEN",
        ledger_open={},
    )

    payload = _build(runtime_dir)

    assert payload["truth_ok"] is False
    assert payload["broker_authority_status"] == "RED"
    assert payload["truth_source"] != "BROKER_AUTHORITY_FLAT_PROVEN"


def test_c_flat_cash_confirmed_upstream_not_green_is_red(tmp_path: Path) -> None:
    """(c) flat + cash-confirmed + upstream NOT GREEN => RED."""
    runtime_dir = tmp_path / "runtime"
    _seed_runtime(
        runtime_dir,
        positions=[],
        cash=_CASH_CONFIRMED,
        reconciliation_status="RED",
        ledger_open={},
    )

    payload = _build(runtime_dir)

    assert payload["truth_ok"] is False
    assert payload["broker_authority_status"] == "RED"
    assert payload["truth_source"] != "BROKER_AUTHORITY_FLAT_PROVEN"


def test_d_nonempty_reconciled_counts_match_is_green_unchanged(tmp_path: Path) -> None:
    """(d) non-empty reconciled (snapshot==ledger) => GREEN via ledger path."""
    runtime_dir = tmp_path / "runtime"
    _seed_runtime(
        runtime_dir,
        positions=[{"symbol": "AAPL", "position": 10.0, "secType": "STK"}],
        cash=_CASH_CONFIRMED,
        reconciliation_status="GREEN",
        ledger_open={"o1": {"symbol": "AAPL", "qty": 10.0}},
    )

    payload = _build(runtime_dir)

    assert payload["truth_ok"] is True
    assert payload["broker_authority_status"] == "GREEN"
    # The non-empty path is unchanged: it is NOT the flat-proven source.
    assert payload["truth_source"].startswith("BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER")
    assert payload["truth_source"] != "BROKER_AUTHORITY_FLAT_PROVEN"
    assert payload["replay_diagnostic_blocks_truth"] is False


def test_e_nonempty_count_mismatch_is_red_unchanged(tmp_path: Path) -> None:
    """(e) non-empty count-mismatch (snapshot=1, ledger=2) => RED unchanged."""
    runtime_dir = tmp_path / "runtime"
    _seed_runtime(
        runtime_dir,
        positions=[{"symbol": "AAPL", "position": 10.0, "secType": "STK"}],
        cash=_CASH_CONFIRMED,
        reconciliation_status="GREEN",
        ledger_open={
            "o1": {"symbol": "AAPL", "qty": 10.0},
            "o2": {"symbol": "MSFT", "qty": 5.0},
        },
    )

    payload = _build(runtime_dir)

    assert payload["truth_ok"] is False
    assert payload["broker_authority_status"] == "RED"
    assert payload["truth_source"] != "BROKER_AUTHORITY_FLAT_PROVEN"
    assert "count_mismatch" in payload["broker_authority_reason"]

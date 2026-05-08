"""Tests for GAP-A009: positions_truth reconciliation_status classifier.

The classifier in build_positions_truth consumes upstream reconciliation_state.json
and lifecycle_replay_coverage.json to produce a truthful reconciliation_status:

  RED    = upstream RED, or qty/symbol mismatches detected.
  YELLOW = scope mismatch OR replay not confirmed.
  GREEN  = upstream GREEN AND replay confirmed AND no scope/qty mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.ops.lifecycle_truth_publisher import (
    BrokerEventsEvidence,
    LedgerEvidence,
    build_positions_truth,
)


def _empty_evidence() -> BrokerEventsEvidence:
    return BrokerEventsEvidence(
        exists=False,
        newest_file=None,
        newest_mtime_unix=None,
        last_event_ts_utc=None,
        event_count_hint=0,
    )


def _empty_ledger_ev() -> LedgerEvidence:
    return LedgerEvidence(exists=False, newest_file=None, newest_mtime_unix=None, line_count_hint=0)


def _setup(
    runtime_dir: Path,
    *,
    snapshot_positions: list,
    upstream_status: str,
    replay_coverage_status: str,
    replay_positions: dict,
    ledger_open: dict,
    qty_mismatches: list = None,
    missing_from_replay: list = None,
) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "positions_snapshot.json").write_text(
        json.dumps({"positions": snapshot_positions, "cash": {}}), encoding="utf-8"
    )
    (runtime_dir / "reconciliation_state.json").write_text(
        json.dumps({"status": upstream_status, "ts_utc": "2026-05-08T00:00:00Z"}),
        encoding="utf-8",
    )
    (runtime_dir / "ibkr_paper_ledger_state.json").write_text(
        json.dumps({"open": ledger_open}), encoding="utf-8"
    )
    (runtime_dir / "lifecycle_replay_state.json").write_text(
        json.dumps({"positions": replay_positions, "inputs": {}}), encoding="utf-8"
    )
    (runtime_dir / "lifecycle_replay_coverage.json").write_text(
        json.dumps(
            {
                "status": replay_coverage_status,
                "summary": {},
                "missing_from_replay": missing_from_replay or [],
                "replay_only": [],
                "qty_mismatches": qty_mismatches or [],
            }
        ),
        encoding="utf-8",
    )


def _build(runtime_dir: Path) -> dict:
    return build_positions_truth(
        repo_root=runtime_dir.parent,
        runtime_dir=runtime_dir,
        data_dir=runtime_dir.parent / "data",
        evidence=_empty_evidence(),
        fills_evidence=_empty_ledger_ev(),
        fees_evidence=_empty_ledger_ev(),
    )


def test_scope_mismatch_reports_yellow(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}] * 4,
        upstream_status="GREEN",
        replay_coverage_status="SCOPE_MISMATCH_MANUAL_VS_PAPER_EXEC",
        replay_positions={"A": 1, "B": 1, "C": 1},
        ledger_open={},
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "YELLOW"
    assert "SCOPE_MISMATCH_MANUAL_VS_PAPER_EXEC" in ev["reconciliation_status_reason"]
    assert "snapshot=4" in ev["reconciliation_status_reason"]
    assert "replay=3" in ev["reconciliation_status_reason"]
    assert "ledger=0" in ev["reconciliation_status_reason"]
    # Upstream value is preserved.
    assert ev["reconciliation_status_upstream"] == "GREEN"
    assert ev["reconciliation_green"] is False


def test_replay_not_confirmed_reports_yellow(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}],
        upstream_status="GREEN",
        replay_coverage_status="REPLAY_PENDING",  # not REPLAY_MATCH_CONFIRMED, not scope mismatch
        replay_positions={"A": 1},
        ledger_open={"A": {}},
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "YELLOW"
    assert "REPLAY_NOT_CONFIRMED" in ev["reconciliation_status_reason"]


def test_clean_confirmed_replay_remains_green(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}],
        upstream_status="GREEN",
        replay_coverage_status="REPLAY_MATCH_CONFIRMED",
        replay_positions={"SPY": 1},
        ledger_open={"SPY": {}},
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "GREEN"
    assert ev["reconciliation_green"] is True
    assert "GREEN_CONFIRMED" in ev["reconciliation_status_reason"]


def test_qty_mismatch_reports_red(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}],
        upstream_status="GREEN",
        replay_coverage_status="REPLAY_MATCH_CONFIRMED",
        replay_positions={"SPY": 1},
        ledger_open={"SPY": {}},
        qty_mismatches=[{"symbol": "SPY", "snapshot": 1, "replay": 2}],
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "RED"
    assert "QTY_OR_SYMBOL_MISMATCH" in ev["reconciliation_status_reason"]


def test_missing_from_replay_reports_red(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}],
        upstream_status="GREEN",
        replay_coverage_status="REPLAY_MATCH_CONFIRMED",
        replay_positions={"SPY": 1},
        ledger_open={"SPY": {}},
        missing_from_replay=["AAPL"],
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "RED"
    assert "QTY_OR_SYMBOL_MISMATCH" in ev["reconciliation_status_reason"]


def test_upstream_red_reports_red(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}],
        upstream_status="RED",
        replay_coverage_status="REPLAY_MATCH_CONFIRMED",
        replay_positions={"SPY": 1},
        ledger_open={"SPY": {}},
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "RED"
    assert "UPSTREAM_RED" in ev["reconciliation_status_reason"]


def test_upstream_unknown_reports_yellow(tmp_path: Path):
    runtime = tmp_path / "runtime"
    _setup(
        runtime,
        snapshot_positions=[{"symbol": "SPY", "position": 1.0}],
        upstream_status="",  # unknown
        replay_coverage_status="REPLAY_MATCH_CONFIRMED",
        replay_positions={"SPY": 1},
        ledger_open={"SPY": {}},
    )
    payload = _build(runtime)
    ev = payload["evidence"]
    assert ev["reconciliation_status"] == "YELLOW"
    assert "UPSTREAM_NOT_GREEN" in ev["reconciliation_status_reason"]

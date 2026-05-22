"""NEW-GAP-041 — live-readiness reconciliation must read resolved truth.

Pre-fix, `ops/live_readiness_publish.py::reconciliation_ok()` only checked
`runtime/reconciliation_state.json::status == "GREEN"`. The publisher's
GREEN status excludes broker-only drifts and strategy-attribution side
mismatches (those land in `runtime/position_guard_drift.json` instead),
so live-readiness could falsely report `reconciliation=GREEN` while
real divergence existed.

This file pins the post-fix contract:
  * publisher GREEN + position_guard_drift drift_count>0 → ready_for_live=false
  * publisher GREEN + drifts array on reconciliation_state non-empty → false
  * publisher GREEN + position_guard_drift unreadable → UNKNOWN → false
  * publisher RED → always false
  * publisher YELLOW (or any non-GREEN) → false
  * resolved GREEN with both surfaces clean → reconciliation gate passes
  * reconciliation_state missing → UNKNOWN → false

The fix MUST NOT authorize live trading; it can only narrow the GREEN gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops import live_readiness_publish as lrp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_recon(
    runtime: Path,
    *,
    status: str,
    drifts: list | None = None,
    mismatches: list | None = None,
) -> Path:
    p = runtime / "reconciliation_state.json"
    payload = {
        "status": status,
        "worst_diff": 0.0,
        "mismatches": list(mismatches or []),
        "drifts": list(drifts or []),
        "counts": {"chad_open": 0, "chad_strategy_open": 0, "broker_positions": 0},
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_guard_drift(runtime: Path, *, drift_count: int) -> Path:
    p = runtime / "position_guard_drift.json"
    drifts = [
        {
            "key": f"alpha|SYM{i}",
            "strategy": "alpha",
            "symbol": f"SYM{i}",
            "guard_side": "SELL",
            "broker_side": "BUY",
            "broker_present": True,
            "drift_kind": "side_mismatch",
        }
        for i in range(drift_count)
    ]
    payload = {
        "schema_version": "position_guard_drift.v1",
        "ts_utc": "2026-05-19T20:00:00Z",
        "ttl_seconds": 360,
        "drift_count": drift_count,
        "drifts": drifts,
    }
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _resolved_reconciliation_status — direct pure-function pins
# ---------------------------------------------------------------------------


def test_resolved_returns_green_only_when_both_surfaces_clean(tmp_path: Path) -> None:
    recon = _write_recon(tmp_path, status="GREEN", drifts=[], mismatches=[])
    drift = _write_guard_drift(tmp_path, drift_count=0)
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "GREEN", reason
    assert "GREEN" in reason


def test_upstream_green_but_guard_drift_count_positive_resolves_red(tmp_path: Path) -> None:
    """The headline NEW-GAP-041 reproducer: publisher says GREEN, but the
    per-strategy guard-vs-broker advisory shows real divergence. Resolved
    truth MUST be RED so ready_for_live=false."""
    recon = _write_recon(tmp_path, status="GREEN", drifts=[], mismatches=[])
    drift = _write_guard_drift(tmp_path, drift_count=2)
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "RED"
    assert "position_guard_drift_count=2" in reason


def test_upstream_green_with_recon_drifts_array_resolves_red(tmp_path: Path) -> None:
    """Publisher status=GREEN but its own `drifts` advisory list is
    non-empty (broker-only movement CHAD didn't initiate). Live-readiness
    must not paper over that — resolve to RED."""
    recon = _write_recon(
        tmp_path,
        status="GREEN",
        drifts=[{"symbol": "M6E", "chad": 2.0, "broker": 4.0, "diff": 2.0}],
        mismatches=[],
    )
    drift = _write_guard_drift(tmp_path, drift_count=0)
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "RED"
    assert "reconciliation_drifts=1" in reason


def test_upstream_red_always_resolves_red(tmp_path: Path) -> None:
    recon = _write_recon(tmp_path, status="RED", drifts=[], mismatches=[])
    drift = _write_guard_drift(tmp_path, drift_count=0)
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "RED"
    assert "RED" in reason


def test_upstream_yellow_resolves_red(tmp_path: Path) -> None:
    """YELLOW is publisher's mid-band status — for live-readiness purposes
    treat it as RED (fail-closed, not a soft pass)."""
    recon = _write_recon(tmp_path, status="YELLOW", drifts=[], mismatches=[])
    drift = _write_guard_drift(tmp_path, drift_count=0)
    status, _ = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "RED"


def test_missing_recon_file_resolves_unknown(tmp_path: Path) -> None:
    """If reconciliation_state.json is absent / unreadable, return UNKNOWN
    so the gate fails closed at the reconciliation_ok wrapper."""
    drift = _write_guard_drift(tmp_path, drift_count=0)
    nonexistent = tmp_path / "no_such_recon.json"
    status, reason = lrp._resolved_reconciliation_status(nonexistent, drift)
    assert status == "UNKNOWN"
    assert "unreadable" in reason


def test_missing_guard_drift_file_resolves_unknown(tmp_path: Path) -> None:
    """Per task spec: 'missing resolved field fails closed'. If the resolved
    advisory artifact is absent we cannot claim resolved=GREEN."""
    recon = _write_recon(tmp_path, status="GREEN", drifts=[], mismatches=[])
    nonexistent = tmp_path / "no_such_guard_drift.json"
    status, reason = lrp._resolved_reconciliation_status(recon, nonexistent)
    assert status == "UNKNOWN"
    assert "unreadable" in reason


def test_guard_drift_wrong_schema_resolves_unknown(tmp_path: Path) -> None:
    """An incorrectly-schemaed advisory file is just as untrustworthy as
    a missing one — fail-closed UNKNOWN, not silent GREEN."""
    recon = _write_recon(tmp_path, status="GREEN", drifts=[], mismatches=[])
    bad = tmp_path / "position_guard_drift.json"
    bad.write_text(json.dumps({"schema_version": "old.v0", "drift_count": 0}), encoding="utf-8")
    status, reason = lrp._resolved_reconciliation_status(recon, bad)
    assert status == "UNKNOWN"
    assert "schema_invalid" in reason


# ---------------------------------------------------------------------------
# reconciliation_ok wrapper — pin that ready_for_live cannot pass on red/unknown
# ---------------------------------------------------------------------------


@pytest.fixture
def _runtime_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect module-level RECON_PATH / GUARD_DRIFT_PATH at a tmp dir so
    the wrapper resolves through the same code path as production."""
    monkeypatch.setattr(lrp, "RECON_PATH", tmp_path / "reconciliation_state.json")
    monkeypatch.setattr(lrp, "GUARD_DRIFT_PATH", tmp_path / "position_guard_drift.json")
    return tmp_path


def test_reconciliation_ok_returns_true_only_on_resolved_green(_runtime_paths: Path) -> None:
    _write_recon(_runtime_paths, status="GREEN", drifts=[], mismatches=[])
    _write_guard_drift(_runtime_paths, drift_count=0)
    ok, reason = lrp.reconciliation_ok()
    assert ok is True
    assert "GREEN" in reason


def test_reconciliation_ok_false_when_upstream_green_but_guard_drift_positive(_runtime_paths: Path) -> None:
    """The fix's core promise: live-readiness no longer accepts upstream-GREEN
    when the guard-vs-broker advisory is non-zero. ready_for_live=false."""
    _write_recon(_runtime_paths, status="GREEN", drifts=[], mismatches=[])
    _write_guard_drift(_runtime_paths, drift_count=1)
    ok, reason = lrp.reconciliation_ok()
    assert ok is False
    assert "position_guard_drift_count=1" in reason


def test_reconciliation_ok_false_on_red(_runtime_paths: Path) -> None:
    _write_recon(_runtime_paths, status="RED", drifts=[], mismatches=[])
    _write_guard_drift(_runtime_paths, drift_count=0)
    ok, _ = lrp.reconciliation_ok()
    assert ok is False


def test_reconciliation_ok_false_when_guard_drift_missing(_runtime_paths: Path) -> None:
    """No guard_drift file at all => UNKNOWN resolved => ready_for_live=false."""
    _write_recon(_runtime_paths, status="GREEN", drifts=[], mismatches=[])
    # Deliberately do not write position_guard_drift.json.
    ok, reason = lrp.reconciliation_ok()
    assert ok is False
    assert "unreadable" in reason


def test_reconciliation_ok_false_when_recon_missing(_runtime_paths: Path) -> None:
    _write_guard_drift(_runtime_paths, drift_count=0)
    ok, reason = lrp.reconciliation_ok()
    assert ok is False
    assert "unreadable" in reason


def test_patch_cannot_widen_to_authorize_live(_runtime_paths: Path) -> None:
    """Defense against accidental loosening: the post-fix function must
    never return True unless BOTH surfaces are clean. Iterate degraded
    matrices to be sure none produce ok=True."""
    cases = [
        # (recon_status, recon_drifts, guard_drift_count)
        ("RED",    [],                                              0),
        ("YELLOW", [],                                              0),
        ("GREEN",  [{"symbol": "X", "chad": 1, "broker": 2, "diff": 1}], 0),
        ("GREEN",  [],                                              1),
        ("GREEN",  [{"symbol": "X", "chad": 1, "broker": 2, "diff": 1}], 5),
    ]
    for status, drifts, drift_count in cases:
        _write_recon(_runtime_paths, status=status, drifts=drifts, mismatches=[])
        _write_guard_drift(_runtime_paths, drift_count=drift_count)
        ok, _ = lrp.reconciliation_ok()
        assert ok is False, f"ok=True for degraded state ({status=}, drifts={len(drifts)}, drift_count={drift_count})"

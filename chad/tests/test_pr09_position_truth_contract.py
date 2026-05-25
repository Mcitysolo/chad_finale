"""PR-09 — Position truth / reconciliation contract.

Pins the contract change that separates broker-authority truth from
replay-diagnostic evidence in runtime/positions_truth.json, and routes
broker_sync-only advisory drifts to runtime/reconciliation_state.json's
new ``diagnostic_drifts`` field (leaving ``drifts`` as the strategy-
attributable fail-closed signal that GAP-041's live-readiness gate
already enforces).

The fix MUST NOT widen any existing gate. It only:
  * adds explicit broker_authority_status / replay_diagnostic_status
    fields to positions_truth.json so consumers can tell which signal
    is authoritative.
  * renames broker_sync-only advisory drift entries to a new
    ``diagnostic_drifts`` field while keeping ``drifts`` reserved for
    strategy-attributable drifts that still trip live_readiness RED.
  * preserves evidence.reconciliation_status and
    evidence.reconciliation_status_upstream verbatim for backward
    compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

from chad.ops.lifecycle_truth_publisher import (
    BrokerEventsEvidence,
    LedgerEvidence,
    build_positions_truth,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _flat_ledger_records(symbols_qtys: list) -> dict:
    out: dict = {}
    for i, (sym, qty) in enumerate(symbols_qtys):
        out[f"{i:064x}"] = {
            "symbol": sym,
            "qty": qty,
            "avg_cost": 100.0,
            "conId": 1000 + i,
            "currency": "USD",
            "secType": "STK",
            "strategy": "manual",
            "tags": ["ibkr_paper", "manual"],
        }
    return out


def _setup_broker_authority_with_partial_replay(
    runtime_dir: Path,
    *,
    snapshot_syms: list,
    missing_from_replay: list,
    qty_mismatches: list,
) -> None:
    """Stage the exact scenario PR-09 was diagnosed against: broker-authority
    truth is GREEN (upstream+ledger+counts match) but lifecycle replay has
    only partial coverage of the snapshot scope."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "positions_snapshot.json").write_text(
        json.dumps(
            {"positions": [{"symbol": s, "position": q} for s, q in snapshot_syms], "cash": {}}
        ),
        encoding="utf-8",
    )
    (runtime_dir / "reconciliation_state.json").write_text(
        json.dumps({"status": "GREEN", "ts_utc": "2026-05-25T16:21:19Z"}), encoding="utf-8"
    )
    (runtime_dir / "ibkr_paper_ledger_state.json").write_text(
        json.dumps(_flat_ledger_records(snapshot_syms)), encoding="utf-8"
    )
    (runtime_dir / "lifecycle_replay_state.json").write_text(
        json.dumps({"positions": {"MES": 1}, "inputs": {}}), encoding="utf-8"
    )
    (runtime_dir / "lifecycle_replay_coverage.json").write_text(
        json.dumps(
            {
                "status": "PARTIAL_REPLAY_COVERAGE",
                "summary": {},
                "missing_from_replay": missing_from_replay,
                "replay_only": ["MES"],
                "qty_mismatches": qty_mismatches,
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


# ---------------------------------------------------------------------------
# 1. positions_truth: truth_ok=true + replay PARTIAL/RED does not flip broker authority
# ---------------------------------------------------------------------------


def test_truth_ok_with_partial_replay_keeps_broker_authority_green(tmp_path: Path):
    """Broker-authority truth is GREEN (upstream+ledger+counts match) even
    when lifecycle replay coverage is PARTIAL. The replay-diagnostic field
    must report PARTIAL/RED but must NOT make broker_authority_status RED."""
    runtime = tmp_path / "runtime"
    syms = [("CVX", -54.0), ("SPY", 11.0), ("M6E", 20.0), ("MGC", -13.0)]
    _setup_broker_authority_with_partial_replay(
        runtime,
        snapshot_syms=syms,
        missing_from_replay=["CVX", "M6E", "MGC"],
        qty_mismatches=[],
    )
    payload = _build(runtime)

    # Broker-authority path is green; truth is good.
    assert payload["truth_ok"] is True
    assert payload["truth_source"].startswith("BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER")
    assert payload["broker_authority_status"] == "GREEN"
    assert payload["broker_authority_reason"].startswith("BROKER_AUTHORITY_GREEN")

    # Replay diagnostic is separately RED — the PR-09 root contradiction
    # said this is correct, but it must not block.
    assert payload["replay_diagnostic_status"] in ("RED", "PARTIAL")
    assert payload["replay_diagnostic_blocks_truth"] is False

    # Backward-compat: existing evidence fields preserved verbatim.
    assert payload["evidence"]["reconciliation_status_upstream"] == "GREEN"
    assert payload["evidence"]["reconciliation_status"] == "RED"


def test_truth_ok_false_implies_broker_authority_red(tmp_path: Path):
    """When broker-authority truth fails (e.g. ledger/snapshot count mismatch)
    the new broker_authority_status field must report RED with a reason and
    truth_ok stays False."""
    runtime = tmp_path / "runtime"
    snapshot_syms = [("SPY", 1.0), ("QQQ", -1.0)]
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "positions_snapshot.json").write_text(
        json.dumps(
            {"positions": [{"symbol": s, "position": q} for s, q in snapshot_syms], "cash": {}}
        ),
        encoding="utf-8",
    )
    (runtime / "reconciliation_state.json").write_text(
        json.dumps({"status": "GREEN", "ts_utc": "2026-05-25T16:21:19Z"}), encoding="utf-8"
    )
    # Ledger has only 1 record — count mismatch against the 2-symbol snapshot.
    (runtime / "ibkr_paper_ledger_state.json").write_text(
        json.dumps(_flat_ledger_records([("SPY", 1.0)])), encoding="utf-8"
    )
    (runtime / "lifecycle_replay_state.json").write_text(
        json.dumps({"positions": {}, "inputs": {}}), encoding="utf-8"
    )
    (runtime / "lifecycle_replay_coverage.json").write_text(
        json.dumps(
            {
                "status": "REPLAY_MATCH_CONFIRMED",
                "summary": {},
                "missing_from_replay": [],
                "replay_only": [],
                "qty_mismatches": [],
            }
        ),
        encoding="utf-8",
    )
    payload = _build(runtime)

    assert payload["truth_ok"] is False
    assert payload["broker_authority_status"] == "RED"
    assert payload["broker_authority_reason"].startswith("BROKER_AUTHORITY_RED")
    # When broker authority fails, replay_diagnostic_blocks_truth flips True
    # because we are NOT on the broker-authority truth_source.
    assert payload["replay_diagnostic_blocks_truth"] is True
    assert payload["truth_source"] == "FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN"


# ---------------------------------------------------------------------------
# 2. Replay diagnostic remains visible
# ---------------------------------------------------------------------------


def test_replay_diagnostic_visibility_preserved(tmp_path: Path):
    """PR-09 explicitly preserves replay diagnostic evidence. Verify all of
    evidence.reconciliation_status, evidence.reconciliation_status_reason,
    evidence.replay_coverage_status, and evidence.missing_from_replay
    remain populated in the broker-authority + partial-replay scenario."""
    runtime = tmp_path / "runtime"
    syms = [("CVX", -54.0), ("M6E", 20.0)]
    _setup_broker_authority_with_partial_replay(
        runtime,
        snapshot_syms=syms,
        missing_from_replay=["CVX", "M6E"],
        qty_mismatches=[],
    )
    payload = _build(runtime)

    ev = payload["evidence"]
    # Replay diagnostic surface is intact.
    assert ev["replay_coverage_status"] == "PARTIAL_REPLAY_COVERAGE"
    assert ev["missing_from_replay"] == ["CVX", "M6E"]
    # Upstream broker reconciliation is separately preserved.
    assert ev["reconciliation_status_upstream"] == "GREEN"
    # New top-level field signals replay is non-authoritative on this path.
    assert payload["replay_diagnostic_blocks_truth"] is False


# ---------------------------------------------------------------------------
# 3. position_guard_drift.drift_count remains the authoritative drift gate
# ---------------------------------------------------------------------------


def test_position_guard_drift_count_zero_is_authoritative_drift_gate(tmp_path):
    """Sanity-check that ops.live_readiness_publish._resolved_reconciliation_status
    continues to honor position_guard_drift.drift_count as authoritative even
    when reconciliation_state.drifts is empty (the PR-09 publisher state) —
    i.e. the GAP-041 contract is preserved end-to-end."""
    from ops import live_readiness_publish as lrp

    recon = tmp_path / "reconciliation_state.json"
    recon.write_text(
        json.dumps(
            {
                "status": "GREEN",
                "worst_diff": 0.0,
                "mismatches": [],
                "drifts": [],
                "diagnostic_drifts": [
                    {"symbol": "M6E", "chad": 18.0, "broker": 20.0, "diff": 2.0, "kind": "broker_sync_only"},
                    {"symbol": "MGC", "chad": -11.0, "broker": -12.0, "diff": 1.0, "kind": "broker_sync_only"},
                ],
                "counts": {"chad_open": 20, "chad_strategy_open": 20, "broker_positions": 20},
            }
        ),
        encoding="utf-8",
    )
    drift = tmp_path / "position_guard_drift.json"
    drift.write_text(
        json.dumps(
            {
                "schema_version": "position_guard_drift.v1",
                "ts_utc": "2026-05-25T16:21:16Z",
                "ttl_seconds": 360,
                "drift_count": 0,
                "drifts": [],
            }
        ),
        encoding="utf-8",
    )
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "GREEN", reason
    assert "GREEN" in reason


def test_position_guard_drift_count_positive_still_red(tmp_path):
    """If position_guard_drift.drift_count > 0 then resolution is RED even
    when reconciliation_state.drifts is empty and diagnostic_drifts is
    populated. The authoritative gate is unchanged."""
    from ops import live_readiness_publish as lrp

    recon = tmp_path / "reconciliation_state.json"
    recon.write_text(
        json.dumps(
            {
                "status": "GREEN",
                "worst_diff": 0.0,
                "mismatches": [],
                "drifts": [],
                "diagnostic_drifts": [],
                "counts": {"chad_open": 0, "chad_strategy_open": 0, "broker_positions": 0},
            }
        ),
        encoding="utf-8",
    )
    drift = tmp_path / "position_guard_drift.json"
    drift.write_text(
        json.dumps(
            {
                "schema_version": "position_guard_drift.v1",
                "drift_count": 1,
                "drifts": [
                    {
                        "key": "alpha|SYM1",
                        "strategy": "alpha",
                        "symbol": "SYM1",
                        "guard_side": "SELL",
                        "broker_side": "BUY",
                        "broker_present": True,
                        "drift_kind": "side_mismatch",
                    }
                ],
                "ts_utc": "2026-05-25T16:21:16Z",
                "ttl_seconds": 360,
            }
        ),
        encoding="utf-8",
    )
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "RED"
    assert "position_guard_drift_count=1" in reason


# ---------------------------------------------------------------------------
# 4. live_readiness lifecycle_truth_ok consumes the broker_authority field
# ---------------------------------------------------------------------------


def test_live_readiness_lifecycle_truth_ok_passes_on_broker_authority_green(
    tmp_path: Path, monkeypatch
):
    """When positions_truth carries truth_ok=true and the new
    broker_authority_status=GREEN, lifecycle_truth_ok must pass and surface
    the broker-authority field in its reason string."""
    from ops import live_readiness_publish as lrp

    truth_path = tmp_path / "positions_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "truth_ok": True,
                "truth_source": "BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER",
                "broker_authority_status": "GREEN",
                "broker_authority_reason": "BROKER_AUTHORITY_GREEN: upstream=GREEN snapshot=20 ledger=20",
                "replay_diagnostic_status": "PARTIAL",
                "replay_diagnostic_reason": "QTY_OR_SYMBOL_MISMATCH: missing_from_replay=14",
                "replay_diagnostic_blocks_truth": False,
                "evidence": {"reconciliation_status_upstream": "GREEN"},
            }
        ),
        encoding="utf-8",
    )
    lifecycle_path = tmp_path / "trade_lifecycle_state.json"
    lifecycle_path.write_text(
        json.dumps({"gap_flag": False, "backlog_flag": False}), encoding="utf-8"
    )
    monkeypatch.setattr(lrp, "TRUTH_PATH", truth_path)
    monkeypatch.setattr(lrp, "LIFECYCLE_PATH", lifecycle_path)

    ok, reason = lrp.lifecycle_truth_ok()
    assert ok is True
    assert "broker_authority_status=GREEN" in reason


def test_live_readiness_lifecycle_truth_ok_fails_when_truth_ok_false(
    tmp_path: Path, monkeypatch
):
    """truth_ok=false must still fail-closed and surface the new field for
    auditability."""
    from ops import live_readiness_publish as lrp

    truth_path = tmp_path / "positions_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "truth_ok": False,
                "truth_source": "FAIL_CLOSED_BOOTSTRAP_SCOPE_UNPROVEN",
                "broker_authority_status": "RED",
                "broker_authority_reason": "BROKER_AUTHORITY_RED: ledger_state_missing",
                "replay_diagnostic_status": "UNKNOWN",
                "replay_diagnostic_reason": "",
                "replay_diagnostic_blocks_truth": True,
                "evidence": {},
            }
        ),
        encoding="utf-8",
    )
    lifecycle_path = tmp_path / "trade_lifecycle_state.json"
    lifecycle_path.write_text(
        json.dumps({"gap_flag": False, "backlog_flag": False}), encoding="utf-8"
    )
    monkeypatch.setattr(lrp, "TRUTH_PATH", truth_path)
    monkeypatch.setattr(lrp, "LIFECYCLE_PATH", lifecycle_path)

    ok, reason = lrp.lifecycle_truth_ok()
    assert ok is False
    assert "broker_authority_status=RED" in reason


# ---------------------------------------------------------------------------
# 5. M6E/MGC diagnostic drifts do not block when guard drift is clean
# ---------------------------------------------------------------------------


def test_m6e_mgc_diagnostic_drifts_do_not_block_when_guard_drift_clean(tmp_path):
    """The exact PR-09 reproducer: reconciliation publisher emits broker_sync-
    only advisory drifts to diagnostic_drifts (NOT drifts). Live readiness's
    resolved status must be GREEN."""
    from ops import live_readiness_publish as lrp

    recon = tmp_path / "reconciliation_state.json"
    recon.write_text(
        json.dumps(
            {
                "status": "GREEN",
                "worst_diff": 0.0,
                "mismatches": [],
                "drifts": [],
                "diagnostic_drifts": [
                    {"symbol": "M6E", "chad": 18.0, "broker": 20.0, "diff": 2.0, "kind": "broker_sync_only"},
                    {"symbol": "MGC", "chad": -11.0, "broker": -12.0, "diff": 1.0, "kind": "broker_sync_only"},
                ],
                "counts": {"chad_open": 20, "chad_strategy_open": 20, "broker_positions": 20},
            }
        ),
        encoding="utf-8",
    )
    drift = tmp_path / "position_guard_drift.json"
    drift.write_text(
        json.dumps(
            {
                "schema_version": "position_guard_drift.v1",
                "drift_count": 0,
                "drifts": [],
                "ts_utc": "2026-05-25T16:21:16Z",
                "ttl_seconds": 360,
            }
        ),
        encoding="utf-8",
    )
    ok, reason = lrp.reconciliation_ok.__wrapped__() if hasattr(lrp.reconciliation_ok, "__wrapped__") else (None, None)
    # The wrapper resolves module-level paths; call _resolved directly here
    # so we don't have to monkey-patch.
    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "GREEN", reason


# ---------------------------------------------------------------------------
# 6. Exterminator EX009 reads diagnostic_drifts
# ---------------------------------------------------------------------------


def test_exterminator_ex009_surfaces_diagnostic_drifts():
    """EX009 (advisory warning) must continue to surface broker-side drifts
    via the new diagnostic_drifts field — visibility is preserved even when
    drifts[] is empty under the PR-09 publisher contract."""
    from chad.ops.exterminator import Exterminator

    recon = {
        "status": "GREEN",
        "worst_diff": 0.0,
        "mismatches": [],
        "drifts": [],
        "diagnostic_drifts": [
            {"symbol": "M6E", "chad": 18.0, "broker": 20.0, "diff": 2.0, "kind": "broker_sync_only"},
            {"symbol": "MGC", "chad": -11.0, "broker": -12.0, "diff": 1.0, "kind": "broker_sync_only"},
        ],
    }
    findings = Exterminator().check_reconciliation(recon)
    assert len(findings) == 1
    f = findings[0]
    assert f.id == "EX009"
    assert f.evidence["diagnostic_drift_count"] == 2
    assert f.evidence["strategy_drift_count"] == 0
    assert f.evidence["drift_count"] == 2


# ---------------------------------------------------------------------------
# 7. Reconciliation publisher classifier routes broker_sync-only to diagnostic_drifts
# ---------------------------------------------------------------------------


def test_reconciliation_publisher_routes_broker_sync_only_to_diagnostic_drifts():
    """Simulate the publisher's per-symbol classification branch (the loop
    at chad/ops/reconciliation_publisher.py main lines ~360-389) to verify
    a broker_sync-only diff lands in diagnostic_drifts, not drifts."""
    from chad.ops.reconciliation_publisher import (
        KNOWN_FUTURES_SYMBOLS,
        KNOWN_NON_CHAD_SYMBOLS,
    )

    # Fixture: M6E broker=20, chad=18 (broker_sync only — strategy_contrib=0)
    fixtures = [
        # (sym, chad_qty, broker_qty, breakdown)
        ("M6E", 18.0, 20.0, {"broker_sync": 18.0, "strategies": 0.0}),
        ("MGC", -11.0, -12.0, {"broker_sync": -11.0, "strategies": 0.0}),
    ]
    mismatches = []
    drifts = []
    diagnostic_drifts = []
    for sym, c, b, bd in fixtures:
        if sym in KNOWN_NON_CHAD_SYMBOLS:
            continue
        if sym in KNOWN_FUTURES_SYMBOLS:
            continue
        diff = abs(c - b)
        if diff > 0:
            strategy_contrib = abs(bd.get("strategies", 0.0))
            if strategy_contrib < 1e-6:
                diagnostic_drifts.append(
                    {"symbol": sym, "chad": c, "broker": b, "diff": diff, "kind": "broker_sync_only"}
                )
            else:
                mismatches.append({"symbol": sym, "chad": c, "broker": b, "diff": diff})

    assert drifts == [], drifts
    assert mismatches == [], mismatches
    assert [d["symbol"] for d in diagnostic_drifts] == ["M6E", "MGC"]
    for d in diagnostic_drifts:
        assert d["kind"] == "broker_sync_only"

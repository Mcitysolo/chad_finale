"""WKF U3 — like-with-like guard↔broker drift semantics.

Covers the F3 fix in three layers:
  * the pure detector chad.core.position_guard.detect_guard_vs_broker_drift_v2:
    aggregate guard qty per symbol vs broker symbol totals; broker truth read
    from the recorded broker_sync `quantity` (NOT gated on `open`) — this is the
    exact false-positive class F3 named; each drift_kind classified correctly;
    single-snapshot atomicity stamped as snapshot_generation;
  * the publisher chad.ops.reconciliation_publisher._emit_position_guard_drift:
    emits schema v2, read-only w.r.t. position_guard.json;
  * the consumer ops.live_readiness_publish._resolved_reconciliation_status:
    accepts both v1 and v2, drift_count>0 still gates RED.

No live broker, no network, no runtime writes — all I/O in tmp_path.
"""
from __future__ import annotations

import json

from chad.core.position_guard import detect_guard_vs_broker_drift_v2


# ---------------------------------------------------------------------------
# Fixtures — a single position_guard.json-shaped snapshot (guard + broker_sync)
# ---------------------------------------------------------------------------

def _guard(symbol, qty, side="BUY", strategy="gamma", open_=True):
    return {
        "open": open_, "strategy": strategy, "symbol": symbol,
        "side": side, "quantity": qty, "last_state": "OPEN" if open_ else "CLOSED",
    }


def _broker_sync(symbol, qty, side="BUY", open_=False):
    # The broker-truth-rebuild writes broker_sync entries carrying the last-known
    # broker quantity and marks them open=False (closed_by strategy_ownership_assumed).
    return {
        "open": open_, "strategy": "broker_sync", "symbol": symbol,
        "side": side, "quantity": qty, "source": "broker_truth_rebuild",
        "closed_by": "strategy_ownership_assumed",
    }


def _snapshot():
    """Mirrors the real 2026-07-11 runtime picture (F3)."""
    return {
        "_version": 1783806455690,
        "_written_by": "position_guard",
        # phantom guard entries — broker holds nothing
        "gamma|BAC": _guard("BAC", 11.0),
        "gamma|SPY": _guard("SPY", 13.0),
        # qty mismatches — broker holds far more than the ledger tracks
        "gamma|TLT": _guard("TLT", 20.0),
        "gamma|UNH": _guard("UNH", 6.0),
        "broker_sync|TLT": _broker_sync("TLT", 420.0),
        "broker_sync|UNH": _broker_sync("UNH", 207.0),
        # broker-untracked accumulation — no guard entry at all
        "broker_sync|IWM": _broker_sync("IWM", 200.0),
        # exact agreement — must NOT drift
        "gamma|MSFT": _guard("MSFT", 15.0),
        "broker_sync|MSFT": _broker_sync("MSFT", 15.0),
    }


# ---------------------------------------------------------------------------
# Detector — drift_kind classification
# ---------------------------------------------------------------------------

def test_each_drift_kind_classified_correctly():
    out = detect_guard_vs_broker_drift_v2(_snapshot())
    by_symbol = {r["symbol"]: r for r in out["drifts"]}

    # phantom_guard_entry: guard open, broker holds nothing
    assert by_symbol["BAC"]["drift_kind"] == "phantom_guard_entry"
    assert by_symbol["BAC"]["guard_qty"] == 11.0
    assert by_symbol["BAC"]["broker_qty"] == 0.0
    assert by_symbol["SPY"]["drift_kind"] == "phantom_guard_entry"

    # broker_untracked_position: broker holds, no guard entry — surfaced, not fixed
    assert by_symbol["IWM"]["drift_kind"] == "broker_untracked_position"
    assert by_symbol["IWM"]["guard_qty"] == 0.0
    assert by_symbol["IWM"]["broker_qty"] == 200.0
    assert by_symbol["IWM"]["guard_keys"] == []

    # qty_mismatch: both hold, totals differ (the TLT 420 / UNH 207 accumulation)
    assert by_symbol["TLT"]["drift_kind"] == "qty_mismatch"
    assert by_symbol["TLT"]["guard_qty"] == 20.0
    assert by_symbol["TLT"]["broker_qty"] == 420.0
    assert by_symbol["TLT"]["qty_delta"] == 20.0 - 420.0
    assert by_symbol["UNH"]["drift_kind"] == "qty_mismatch"

    # exact-agreement symbol must NOT appear
    assert "MSFT" not in by_symbol

    assert out["drift_count"] == 5
    assert out["counts_by_kind"] == {
        "phantom_guard_entry": 2,
        "broker_untracked_position": 1,
        "qty_mismatch": 2,
    }


def test_broker_truth_read_from_quantity_not_open_flag():
    """F3 false-positive class: broker_sync entries are open=False but carry the
    broker quantity. The v1 detector gated on open and saw zero broker truth,
    falsely flagging broker_truth_missing. v2 must read the quantity."""
    state = {
        "_version": 1,
        "gamma|LLY": _guard("LLY", 4.0),
        "broker_sync|LLY": _broker_sync("LLY", 130.0, open_=False),  # closed flag
    }
    out = detect_guard_vs_broker_drift_v2(state)
    assert out["drift_count"] == 1
    rec = out["drifts"][0]
    # broker truth is 130 (from quantity), so this is a qty_mismatch — NOT a
    # phantom/broker_truth_missing. The open=False flag must be ignored.
    assert rec["drift_kind"] == "qty_mismatch"
    assert rec["broker_qty"] == 130.0


def test_side_flip_shows_as_signed_qty_mismatch():
    state = {
        "_version": 1,
        "alpha|SYM": _guard("SYM", 10.0, side="BUY", strategy="alpha"),
        "broker_sync|SYM": _broker_sync("SYM", 10.0, side="SELL"),
    }
    out = detect_guard_vs_broker_drift_v2(state)
    assert out["drift_count"] == 1
    rec = out["drifts"][0]
    assert rec["drift_kind"] == "qty_mismatch"
    assert rec["guard_qty"] == 10.0     # BUY 10 → +10
    assert rec["broker_qty"] == -10.0   # SELL 10 → -10


def test_snapshot_generation_stamped_atomically():
    """Atomicity: guard + broker truth come from ONE state dict bearing ONE
    _version generation, echoed onto the payload and every record."""
    state = _snapshot()
    out = detect_guard_vs_broker_drift_v2(state)
    assert out["snapshot_generation"] == state["_version"]
    assert out["written_by"] == "position_guard"
    assert all(r["snapshot_generation"] == state["_version"] for r in out["drifts"])


def test_multi_strategy_guard_qty_aggregated_per_symbol():
    state = {
        "_version": 1,
        "alpha|NVDA": _guard("NVDA", 3.0, strategy="alpha"),
        "beta|NVDA": _guard("NVDA", 5.0, strategy="beta"),
        "broker_sync|NVDA": _broker_sync("NVDA", 8.0),  # 3+5 == 8 → no drift
    }
    out = detect_guard_vs_broker_drift_v2(state)
    assert out["drift_count"] == 0, "aggregate guard 3+5 must net against broker 8"


def test_closed_guard_entries_ignored():
    state = {
        "_version": 1,
        "gamma|X": _guard("X", 9.0, open_=False),   # closed guard → not counted
        "broker_sync|X": _broker_sync("X", 0.0),
    }
    out = detect_guard_vs_broker_drift_v2(state)
    assert out["drift_count"] == 0


def test_non_mapping_state_is_safe():
    out = detect_guard_vs_broker_drift_v2(None)  # type: ignore[arg-type]
    assert out["drift_count"] == 0
    assert out["drifts"] == []
    assert out["snapshot_generation"] is None


# ---------------------------------------------------------------------------
# Publisher — emits v2, read-only
# ---------------------------------------------------------------------------

def test_publisher_emits_v2_and_is_read_only(tmp_path, monkeypatch):
    from chad.ops import reconciliation_publisher as pub
    from chad.core import position_guard

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    guard_path = runtime_dir / "position_guard.json"
    drift_path = runtime_dir / "position_guard_drift.json"

    monkeypatch.setattr(pub, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(pub, "GUARD_PATH", guard_path)
    monkeypatch.setattr(pub, "DRIFT_OUT_PATH", drift_path)
    monkeypatch.setattr(position_guard, "STATE_PATH", guard_path)

    guard_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    pre_mtime = guard_path.stat().st_mtime_ns

    n = pub._emit_position_guard_drift()
    assert n == 5
    payload = json.loads(drift_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "position_guard_drift.v2"
    assert payload["drift_count"] == 5
    assert payload["counts_by_kind"]["broker_untracked_position"] == 1
    # read-only invariant
    assert guard_path.stat().st_mtime_ns == pre_mtime


# ---------------------------------------------------------------------------
# Consumer — v2 accepted, drift_count>0 still gates RED
# ---------------------------------------------------------------------------

def test_consumer_accepts_v2_and_gates_red_on_drift(tmp_path):
    from ops import live_readiness_publish as lrp

    recon = tmp_path / "reconciliation_state.json"
    recon.write_text(json.dumps({
        "status": "GREEN", "worst_diff": 0.0, "mismatches": [],
        "drifts": [], "diagnostic_drifts": [],
        "counts": {"chad_open": 0, "chad_strategy_open": 0, "broker_positions": 0},
    }), encoding="utf-8")

    drift = tmp_path / "position_guard_drift.json"
    drift.write_text(json.dumps({
        "schema_version": "position_guard_drift.v2",
        "ts_utc": "2026-07-11T21:00:00Z", "ttl_seconds": 360,
        "drift_count": 3,
        "counts_by_kind": {"phantom_guard_entry": 1, "broker_untracked_position": 1,
                           "qty_mismatch": 1},
        "drifts": [{"symbol": "IWM", "drift_kind": "broker_untracked_position",
                    "guard_qty": 0.0, "broker_qty": 200.0}],
    }), encoding="utf-8")

    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "RED"
    assert "position_guard_drift_count=3" in reason


def test_consumer_accepts_v2_green_when_drift_zero(tmp_path):
    from ops import live_readiness_publish as lrp

    recon = tmp_path / "reconciliation_state.json"
    recon.write_text(json.dumps({
        "status": "GREEN", "worst_diff": 0.0, "mismatches": [],
        "drifts": [], "diagnostic_drifts": [],
        "counts": {"chad_open": 0, "chad_strategy_open": 0, "broker_positions": 0},
    }), encoding="utf-8")

    drift = tmp_path / "position_guard_drift.json"
    drift.write_text(json.dumps({
        "schema_version": "position_guard_drift.v2",
        "ts_utc": "2026-07-11T21:00:00Z", "ttl_seconds": 360,
        "drift_count": 0, "counts_by_kind": {}, "drifts": [],
    }), encoding="utf-8")

    status, reason = lrp._resolved_reconciliation_status(recon, drift)
    assert status == "GREEN", reason

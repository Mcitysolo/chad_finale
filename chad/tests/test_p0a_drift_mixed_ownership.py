"""P0A-A5 — drift comparator: operator-owned symbols are mixed-ownership INFO.

For symbols in the reconciliation exclusion policy that CHAD also touches
(BAC/SPY today, plus LLY/MSFT), the v2 comparator flagged the operator's
pre-existing broker holdings as actionable broker_untracked_position /
qty_mismatch, inflating drift_count (9 records on 2026-07-13, of which 4 were
operator-owned). The broker total on an excluded symbol is a MIX of the
operator's shares and any CHAD lots — not like-with-like against CHAD-tracked
lots — so it must be reported as informational mixed_ownership_info and kept
OUT of drift_count (never flip live-readiness RED for the operator's book).
"""

from __future__ import annotations

import json

from chad.core.position_guard import detect_guard_vs_broker_drift_v2


def _guard(symbol, qty, side="BUY", strategy="gamma", open_=True):
    return {
        "open": open_, "strategy": strategy, "symbol": symbol,
        "side": side, "quantity": qty,
    }


def _broker_sync(symbol, qty, side="BUY", open_=False):
    return {
        "open": open_, "strategy": "broker_sync", "symbol": symbol,
        "side": side, "quantity": qty, "source": "broker_truth_rebuild",
    }


def _todays_snapshot():
    """The exact 2026-07-13 runtime picture: 9 broker_untracked symbols,
    4 of them operator-owned (BAC/SPY/LLY/MSFT)."""
    return {
        "_version": 1,
        "_written_by": "position_guard",
        # operator-excluded (broker holds operator's pre-existing shares)
        "broker_sync|BAC": _broker_sync("BAC", 154.0),
        "broker_sync|SPY": _broker_sync("SPY", 182.0),
        "broker_sync|LLY": _broker_sync("LLY", 174.0),
        "broker_sync|MSFT": _broker_sync("MSFT", 23.0),
        # CHAD symbols (actionable)
        "broker_sync|IWM": _broker_sync("IWM", 200.0),
        "broker_sync|SVXY": _broker_sync("SVXY", 156.0),
        "broker_sync|TLT": _broker_sync("TLT", 1340.0, side="SELL"),
        "broker_sync|UNH": _broker_sync("UNH", 207.0),
        "broker_sync|V": _broker_sync("V", 154.0),
    }


EXCLUDED = {"AAPL", "BAC", "CVX", "LLY", "MSFT", "NVDA", "PEP", "QQQ", "SPY"}


def test_backward_compatible_without_excluded_arg():
    """No excluded_symbols -> identical behaviour to the pre-A5 v2 detector."""
    r = detect_guard_vs_broker_drift_v2(_todays_snapshot())
    # all 9 are broker_untracked; drift_count counts all of them.
    assert r["drift_count"] == 9
    assert r["counts_by_kind"]["broker_untracked_position"] == 9
    assert r["counts_by_kind"]["mixed_ownership_info"] == 0
    assert r["info_count"] == 0


def test_excluded_symbols_become_mixed_ownership_info():
    r = detect_guard_vs_broker_drift_v2(_todays_snapshot(), excluded_symbols=EXCLUDED)
    # 4 operator-owned -> mixed_ownership_info (BAC/SPY/LLY/MSFT)
    # 5 CHAD symbols   -> broker_untracked_position (IWM/SVXY/TLT/UNH/V)
    assert r["counts_by_kind"]["mixed_ownership_info"] == 4
    assert r["counts_by_kind"]["broker_untracked_position"] == 5
    # drift_count is ACTIONABLE only — excludes the 4 operator-owned records.
    assert r["drift_count"] == 5
    assert r["info_count"] == 4

    by_symbol = {d["symbol"]: d for d in r["drifts"]}
    for sym in ("BAC", "SPY", "LLY", "MSFT"):
        assert by_symbol[sym]["drift_kind"] == "mixed_ownership_info"
        assert by_symbol[sym]["is_excluded"] is True
    for sym in ("IWM", "SVXY", "TLT", "UNH", "V"):
        assert by_symbol[sym]["drift_kind"] == "broker_untracked_position"
        assert by_symbol[sym]["is_excluded"] is False


def test_bac_spy_exact_case_not_counted_red():
    """Today's exact BAC/SPY case: broker holds operator shares, guard=0."""
    r = detect_guard_vs_broker_drift_v2(_todays_snapshot(), excluded_symbols=EXCLUDED)
    by_symbol = {d["symbol"]: d for d in r["drifts"]}
    bac, spy = by_symbol["BAC"], by_symbol["SPY"]
    assert bac["broker_qty"] == 154.0 and bac["guard_qty"] == 0.0
    assert spy["broker_qty"] == 182.0 and spy["guard_qty"] == 0.0
    # No operator baseline recorded -> delta honestly unattributable.
    assert bac["operator_baseline"] is None
    assert bac["net_broker_qty"] is None
    assert bac["chad_vs_net_broker_delta"] is None


def test_operator_baseline_nets_out_when_recorded():
    """If a signed operator baseline IS supplied, the residual is netted."""
    # BAC broker 154 = operator 154 baseline + CHAD 0 -> net residual 0.
    snap = {
        "_version": 1,
        "broker_sync|BAC": _broker_sync("BAC", 154.0),
        "gamma|BAC": _guard("BAC", 12.0),  # CHAD holds 12 lots too
    }
    r = detect_guard_vs_broker_drift_v2(
        snap, excluded_symbols={"BAC"}, operator_baselines={"BAC": 154.0}
    )
    d = r["drifts"][0]
    assert d["drift_kind"] == "mixed_ownership_info"
    assert d["operator_baseline"] == 154.0
    assert d["net_broker_qty"] == 0.0                  # 154 - 154
    assert d["chad_vs_net_broker_delta"] == 12.0        # guard 12 - net 0
    assert r["drift_count"] == 0                          # never actionable
    assert r["info_count"] == 1


def test_excluded_symbol_both_flat_no_record():
    snap = {"_version": 1, "broker_sync|BAC": _broker_sync("BAC", 0.0)}
    r = detect_guard_vs_broker_drift_v2(snap, excluded_symbols={"BAC"})
    assert r["drifts"] == []
    assert r["drift_count"] == 0 and r["info_count"] == 0


def test_publisher_emits_v3_with_excluded(tmp_path, monkeypatch):
    """End-to-end: reconciliation_publisher emits schema v3, drift_count drops
    to the actionable subset, and mixed-ownership records are informational."""
    from chad.ops import reconciliation_publisher as rp

    guard_path = tmp_path / "position_guard.json"
    guard_path.write_text(json.dumps(_todays_snapshot()), encoding="utf-8")
    out_path = tmp_path / "position_guard_drift.json"
    monkeypatch.setattr(rp, "GUARD_PATH", guard_path)
    monkeypatch.setattr(rp, "DRIFT_OUT_PATH", out_path)
    monkeypatch.setattr(rp, "RUNTIME_DIR", tmp_path)

    count = rp._emit_position_guard_drift()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "position_guard_drift.v3"
    # BAC/SPY/LLY/MSFT are in the real exclusion policy -> mixed_ownership_info.
    assert payload["counts_by_kind"]["mixed_ownership_info"] >= 4
    assert count == payload["drift_count"]
    assert count < 9  # the operator-owned records no longer inflate the count
    assert "BAC" in payload["excluded_symbols"]


def test_live_readiness_accepts_v3(tmp_path):
    """The live-readiness gate must accept v3 (else it returns UNKNOWN)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "live_readiness_publish",
        "/home/ubuntu/chad_finale/ops/live_readiness_publish.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    recon = tmp_path / "reconciliation_state.json"
    recon.write_text(json.dumps({"status": "GREEN", "drifts": []}), encoding="utf-8")
    drift = tmp_path / "position_guard_drift.json"

    # v3 with drift_count=0 -> GREEN, not schema-invalid UNKNOWN.
    drift.write_text(
        json.dumps({"schema_version": "position_guard_drift.v3", "drift_count": 0}),
        encoding="utf-8",
    )
    status, reason = mod._resolved_reconciliation_status(recon, drift)
    assert status == "GREEN", (status, reason)

    # v3 with actionable drift -> RED.
    drift.write_text(
        json.dumps({"schema_version": "position_guard_drift.v3", "drift_count": 5}),
        encoding="utf-8",
    )
    status, reason = mod._resolved_reconciliation_status(recon, drift)
    assert status == "RED", (status, reason)

#!/usr/bin/env python3
"""
Tests for the RED-window mechanical classifier
(chad/ops/soak/red_window_classifier.py), enforcing the LOCKED rule
ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md
(sha256 prefix 800480a5).

Coverage (per the build objective VERIFY section):
  - Rule-sha integrity (asserts 800480a5 prefix + live-file match).
  - Fully-clean RED window where all 8 conditions are satisfied -> EXPLAINED.
  - Flip each condition individually -> that condition reports FAIL and the
    window classifies FAIL.
  - Sub-60s and unbounded/missing-evidence windows -> NOT_COUNTABLE.
  - §5 schema conformance (every mandated field present, exact names).
  - Determinism (identical inputs -> byte-identical output file).
  - No-stray-write proof (only --out is created/modified; nothing else under a
    sentinel live runtime path or the evidence dirs changes).

All writes go to pytest tmp_path; no live runtime/ or data/soak/ path is touched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chad.ops.soak import red_window_classifier as rwc


# ---------------------------------------------------------------------------
# fixture builders (synthetic, in tmp)
# ---------------------------------------------------------------------------
DATE = "20260620"


def _row(
    ts,
    broker,
    *,
    recon_ts,
    recon_status="GREEN",
    sizing="sha256:STABLE",
    dc_ts="2026-06-20T12:59:00Z",  # caps published BEFORE the window
    scr="CAUTIOUS",
    pl_ts="2026-06-20T10:00:00Z",
    tier_ts="2026-06-20T04:00:00Z",
    sb_active=False,
    sb_trig="",
    sb_clear="2026-05-28T18:38:41Z",
):
    return {
        "schema_version": "soak_status_history.v1",
        "ts_utc": ts,
        "broker_authority_status": broker,
        "reconciliation_status": recon_status,
        "reconciliation_ts_utc": recon_ts,
        "sizing_digest": sizing,
        "dynamic_caps_ts_utc": dc_ts,
        "scr_band": scr,
        "profit_lock_ts_utc": pl_ts,
        "tier_ts_utc": tier_ts,
        "stop_bus_active": sb_active,
        "stop_bus_triggered_at": sb_trig,
        "stop_bus_cleared_at": sb_clear,
    }


def _clean_status_rows(red_minutes=frozenset({1, 2, 3, 4, 5})):
    """30 rows at 60s cadence, minutes 13:00..13:29.

    RED during ``red_minutes``. Reconciliation cycles bucket every 5 minutes
    (13:00, 13:05, 13:10, ...). With the default RED 13:01..13:05 the window is
    [13:01:00, 13:06:00) (duration 300s), and the next reconciliation cycle after
    window_close is 13:10 (carried by the minute-10 row, broker GREEN).
    """
    rows = []
    for m in range(0, 30):
        broker = "RED" if m in red_minutes else "GREEN"
        bucket = (m // 5) * 5
        rows.append(
            _row(
                f"2026-06-20T13:{m:02d}:00Z",
                broker,
                recon_ts=f"2026-06-20T13:{bucket:02d}:00Z",
            )
        )
    return rows


def _write_evidence(
    tmp: Path,
    *,
    status_rows,
    signal_rows=(),
    entry_rows=(),
    trade_rows=(),
    date=DATE,
):
    """Write the four dated evidence files under tmp/data and return data_dir."""
    data_dir = tmp / "data"
    soak = data_dir / "soak"
    trades = data_dir / "trades"
    soak.mkdir(parents=True, exist_ok=True)
    trades.mkdir(parents=True, exist_ok=True)

    def _dump(path: Path, rows):
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )

    _dump(soak / f"status_history_{date}.ndjson", status_rows)
    if signal_rows:
        _dump(soak / f"signal_router_emissions_{date}.ndjson", signal_rows)
    if entry_rows:
        _dump(soak / f"entry_intent_audit_{date}.ndjson", entry_rows)
    if trade_rows:
        _dump(trades / f"trade_history_{date}.ndjson", trade_rows)
    return data_dir


def _classify(tmp: Path, **evidence):
    """Run the classifier end-to-end against tmp evidence with a PINNED evaluator
    identity + ts (so output is deterministic) and return the parsed doc."""
    data_dir = _write_evidence(tmp, **evidence)
    runtime_dir = tmp / "runtime"
    out_path = runtime_dir / "session_explanations.json"
    return rwc.run(
        date=DATE,
        data_dir=data_dir,
        runtime_dir=runtime_dir,
        out_path=out_path,
        evaluator_identity="pytest-evaluator",
        evaluated_at_utc="2026-06-20T23:00:00Z",
    )


# ---------------------------------------------------------------------------
# 1. Rule-sha integrity
# ---------------------------------------------------------------------------
def test_rule_sha_prefix_is_800480a5():
    assert rwc.RULE_SHA256.startswith("800480a5")


def test_rule_sha_matches_live_file():
    repo = rwc._repo_root()
    live = hashlib.sha256((repo / rwc.RULE_PATH).read_bytes()).hexdigest()
    assert live == rwc.RULE_SHA256, "locked rule contract has DRIFTED"
    ok, got = rwc._verify_rule_sha(repo)
    assert ok is True
    assert got == rwc.RULE_SHA256


# ---------------------------------------------------------------------------
# 2. RED-window detection
# ---------------------------------------------------------------------------
def test_detect_single_red_window():
    windows = rwc.detect_red_windows(_clean_status_rows())
    assert len(windows) == 1
    w = windows[0]
    assert w["window_open_utc"] == "2026-06-20T13:01:00Z"
    assert w["window_close_utc"] == "2026-06-20T13:06:00Z"  # first GREEN recovery
    assert w["bounded"] is True
    assert [r["ts_utc"] for r in w["red_rows"]] == [
        f"2026-06-20T13:0{m}:00Z" for m in range(1, 6)
    ]


def test_detect_two_windows_and_unbounded_tail():
    rows = _clean_status_rows(red_minutes=frozenset({1, 2, 28, 29}))
    windows = rwc.detect_red_windows(rows)
    assert len(windows) == 2
    assert windows[0]["window_open_utc"] == "2026-06-20T13:01:00Z"
    assert windows[0]["bounded"] is True
    # the 28..29 RED run reaches EOF -> unbounded (no recovery sample)
    assert windows[1]["window_open_utc"] == "2026-06-20T13:28:00Z"
    assert windows[1]["bounded"] is False
    assert windows[1]["window_close_utc"] is None


def test_detect_no_red_windows():
    rows = _clean_status_rows(red_minutes=frozenset())
    assert rwc.detect_red_windows(rows) == []


# ---------------------------------------------------------------------------
# 3a. Clean window -> EXPLAINED
# ---------------------------------------------------------------------------
def test_clean_window_is_explained(tmp_path):
    doc = _classify(tmp_path, status_rows=_clean_status_rows())
    assert doc["counts"] == {"EXPLAINED": 1, "FAIL": 0, "NOT_COUNTABLE": 0}
    w = doc["windows"][0]
    assert w["classification"] == "EXPLAINED"
    assert w["countability"] == {"countable": True, "reason": "all_eight_conditions_pass"}
    # every one of the eight conditions PASSed
    for k in (str(i) for i in range(1, 9)):
        assert w["criteria"][k]["result"] == "PASS", (k, w["criteria"][k])
    assert w["duration_seconds"] == 300


# ---------------------------------------------------------------------------
# 3b. Flip each condition -> that condition FAILs and the window FAILs
# ---------------------------------------------------------------------------
def _signal_row(ts="2026-06-20T13:03:00Z"):
    return {"schema_version": "soak_signal_router.v1", "ts_utc": ts, "symbol": "AAPL",
            "side": "BUY", "primary_strategy": "alpha", "net_size": 100.0}


def _entry_row(ts="2026-06-20T13:03:00Z", intent_type="ENTRY"):
    return {"schema_version": "soak_entry_intent.v1", "ts_utc": ts,
            "intent_type": intent_type, "symbol": "AAPL", "side": "BUY",
            "strategy": "alpha", "admitted": True}


def _trade_row(exit_ts="2026-06-20T13:03:00Z"):
    return {"timestamp_utc": exit_ts, "record_hash": "x", "prev_hash": "y",
            "sequence_id": 1, "payload": {"exit_time_utc": exit_ts, "pnl": -12.0,
                                          "symbol": "MES", "side": "BUY"}}


def test_flip_condition_1_duration(tmp_path):
    # RED 13:01..13:17 -> recovery 13:18 -> duration 1020s > 900 -> crit 1 FAIL
    doc = _classify(tmp_path, status_rows=_clean_status_rows(frozenset(range(1, 18))))
    w = doc["windows"][0]
    assert w["criteria"]["1"]["result"] == "FAIL"
    assert w["duration_seconds"] == 1020
    assert w["classification"] == "FAIL"


def test_flip_condition_2_signals(tmp_path):
    doc = _classify(tmp_path, status_rows=_clean_status_rows(), signal_rows=[_signal_row()])
    w = doc["windows"][0]
    assert w["criteria"]["2"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


def test_flip_condition_3_entries(tmp_path):
    doc = _classify(tmp_path, status_rows=_clean_status_rows(), entry_rows=[_entry_row()])
    w = doc["windows"][0]
    assert w["criteria"]["3"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


def test_entry_row_non_entry_intent_does_not_fail(tmp_path):
    # An EXIT intent inside the window is NOT a fresh entry (crit 3 counts ENTRY only).
    doc = _classify(
        tmp_path,
        status_rows=_clean_status_rows(),
        entry_rows=[_entry_row(intent_type="EXIT")],
    )
    assert doc["windows"][0]["criteria"]["3"]["result"] == "PASS"
    assert doc["windows"][0]["classification"] == "EXPLAINED"


def test_flip_condition_4_sizing(tmp_path):
    rows = _clean_status_rows()
    # caps republished mid-window (branch A fails) AND digest changed (branch B fails)
    rows[3]["dynamic_caps_ts_utc"] = "2026-06-20T13:03:00Z"
    rows[3]["sizing_digest"] = "sha256:CHANGED"
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["criteria"]["4"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


def test_flip_condition_5_scr(tmp_path):
    rows = _clean_status_rows()
    rows[3]["scr_band"] = "CONFIDENT"  # band changed mid-window
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["criteria"]["5"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


def test_flip_condition_6_risk_state(tmp_path):
    rows = _clean_status_rows()
    rows[3]["profit_lock_ts_utc"] = "2026-06-20T13:03:00Z"  # risk-state mutated
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["criteria"]["6"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


def test_flip_condition_6_stop_bus_proxy(tmp_path):
    # stop_bus has no ts_utc; a triggered_at change across the window must FAIL crit 6
    rows = _clean_status_rows()
    rows[3]["stop_bus_triggered_at"] = "2026-06-20T13:03:00Z"
    rows[3]["stop_bus_active"] = True
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["criteria"]["6"]["result"] == "FAIL"
    # basis surfaces the phantom-field flag honestly
    assert "no ts_utc" in w["criteria"]["6"]["basis"]
    assert w["classification"] == "FAIL"


def test_flip_condition_7_pnl(tmp_path):
    doc = _classify(tmp_path, status_rows=_clean_status_rows(), trade_rows=[_trade_row()])
    w = doc["windows"][0]
    assert w["criteria"]["7"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


def test_trade_outside_window_does_not_fail(tmp_path):
    # A close realized AFTER the window (13:09) must not fail crit 7.
    doc = _classify(
        tmp_path,
        status_rows=_clean_status_rows(),
        trade_rows=[_trade_row(exit_ts="2026-06-20T13:09:00Z")],
    )
    assert doc["windows"][0]["criteria"]["7"]["result"] == "PASS"
    assert doc["windows"][0]["classification"] == "EXPLAINED"


def test_flip_condition_8_reconciler(tmp_path):
    # Next reconciliation cycle after the window (13:10) is still RED -> crit 8 FAIL.
    rows = _clean_status_rows()
    for r in rows:
        if r["reconciliation_ts_utc"] == "2026-06-20T13:10:00Z":
            r["broker_authority_status"] = "RED"
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]  # the 13:01..05 window
    assert w["window_open_utc"] == "2026-06-20T13:01:00Z"
    assert w["criteria"]["8"]["result"] == "FAIL"
    assert w["classification"] == "FAIL"


# ---------------------------------------------------------------------------
# 3c. Sub-60s / unbounded -> NOT_COUNTABLE
# ---------------------------------------------------------------------------
def test_sub_60s_window_is_not_countable(tmp_path):
    rows = [
        _row("2026-06-20T13:00:00Z", "GREEN", recon_ts="2026-06-20T13:00:00Z"),
        _row("2026-06-20T13:01:00Z", "RED", recon_ts="2026-06-20T13:00:00Z"),
        _row("2026-06-20T13:01:30Z", "GREEN", recon_ts="2026-06-20T13:00:00Z"),  # 30s later
        _row("2026-06-20T13:05:00Z", "GREEN", recon_ts="2026-06-20T13:05:00Z"),
    ]
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["duration_seconds"] == 30
    assert w["classification"] == "NOT_COUNTABLE"
    assert w["countability"] == {"countable": False, "reason": "sub_cadence_window"}


def test_unbounded_window_is_not_countable(tmp_path):
    # RED never recovers (still RED at EOF) -> missing recovery evidence.
    rows = _clean_status_rows(red_minutes=frozenset(range(1, 30)))
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["window_close_utc"] is None
    assert w["classification"] == "NOT_COUNTABLE"
    assert w["countability"] == {"countable": False, "reason": "unbounded_window"}
    # the unboundable conditions are UNVERIFIABLE, not silently PASS
    for k in ("1", "2", "3", "7", "8"):
        assert w["criteria"][k]["result"] == "UNVERIFIABLE"


def test_unverifiable_without_fail_is_not_countable(tmp_path):
    # A bounded, long-enough window whose next reconciliation cycle is absent
    # (no recon cycle strictly after window_close) -> crit 8 UNVERIFIABLE, no FAIL
    # -> NOT_COUNTABLE (the "cannot be verified" disposition).
    rows = [
        _row("2026-06-20T13:00:00Z", "GREEN", recon_ts="2026-06-20T13:00:00Z"),
        _row("2026-06-20T13:01:00Z", "RED", recon_ts="2026-06-20T13:00:00Z"),
        _row("2026-06-20T13:02:00Z", "RED", recon_ts="2026-06-20T13:00:00Z"),
        _row("2026-06-20T13:03:00Z", "GREEN", recon_ts="2026-06-20T13:00:00Z"),
    ]
    doc = _classify(tmp_path, status_rows=rows)
    w = doc["windows"][0]
    assert w["criteria"]["8"]["result"] == "UNVERIFIABLE"
    assert w["classification"] == "NOT_COUNTABLE"
    assert w["countability"]["reason"] == "unverifiable_criteria"


# ---------------------------------------------------------------------------
# 4. §5 schema conformance
# ---------------------------------------------------------------------------
def test_section5_schema_conformance(tmp_path):
    doc = _classify(tmp_path, status_rows=_clean_status_rows())

    # top-level
    for key in (
        "schema_version", "rule_path", "rule_sha256", "date",
        "evaluator_identity", "evaluated_at_utc",
        "artifact_paths", "artifact_sha256", "window_count", "counts", "windows",
    ):
        assert key in doc, f"missing top-level key {key}"
    assert doc["schema_version"] == "session_explanations.v1"
    assert doc["rule_sha256"] == rwc.RULE_SHA256
    assert doc["evaluator_identity"] == "pytest-evaluator"
    assert doc["evaluated_at_utc"] == "2026-06-20T23:00:00Z"

    w = doc["windows"][0]
    assert w["classification"] == "EXPLAINED"
    # rule §5 mandated per-EXPLAINED-window fields
    assert "window_open_utc" in w
    assert "window_close_utc" in w
    assert "duration_seconds" in w
    assert "evaluator_identity" in w           # §5: evaluator process identity
    assert "evaluated_at_utc" in w             # §5: evaluator ts_utc
    # §5: the artifact paths used to prove each criterion (1..8)
    assert set(w["artifact_paths"].keys()) == {str(i) for i in range(1, 9)}
    assert all(w["artifact_paths"][k] for k in w["artifact_paths"])
    # §5: the evaluator's sha256 of each artifact at evaluation time
    for art in ("status_history", "signal_router_emissions", "entry_intent_audit",
                "trade_history"):
        assert art in w["artifact_sha256"]
    # the status-history artifact actually existed -> sha present
    assert w["artifact_sha256"]["status_history"].startswith("sha256:")
    # per-criterion structure
    for k in (str(i) for i in range(1, 9)):
        c = w["criteria"][k]
        assert set(c.keys()) == {"result", "basis", "rule_section", "artifact"}
        assert c["rule_section"].startswith("§5.")


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------
def test_determinism_byte_identical(tmp_path):
    rows = _clean_status_rows()
    data_dir = _write_evidence(tmp_path, status_rows=rows)
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"
    common = dict(
        date=DATE, data_dir=data_dir, runtime_dir=tmp_path / "rt",
        evaluator_identity="ev", evaluated_at_utc="2026-06-20T23:00:00Z",
    )
    rwc.run(out_path=out1, **common)
    rwc.run(out_path=out2, **common)
    assert out1.read_bytes() == out2.read_bytes()


# ---------------------------------------------------------------------------
# 6. No-stray-write proof
# ---------------------------------------------------------------------------
def _tree_snapshot(root: Path):
    """Map of relative path -> sha256(content) for every file under root."""
    snap = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def test_no_stray_writes(tmp_path):
    # a sentinel "live" runtime dir that must NOT be touched
    live_runtime = tmp_path / "live_runtime"
    live_runtime.mkdir()
    sentinel = live_runtime / "session_explanations.json"
    sentinel.write_text('{"sentinel": "do-not-touch"}', encoding="utf-8")

    data_dir = _write_evidence(tmp_path, status_rows=_clean_status_rows(),
                               signal_rows=[_signal_row("2026-06-20T20:00:00Z")])
    out_path = tmp_path / "out" / "session_explanations.json"

    before = _tree_snapshot(tmp_path)
    rwc.run(
        date=DATE, data_dir=data_dir, runtime_dir=tmp_path / "classifier_rt",
        out_path=out_path, evaluator_identity="ev",
        evaluated_at_utc="2026-06-20T23:00:00Z",
    )
    after = _tree_snapshot(tmp_path)

    # the sentinel live-runtime file is byte-identical (untouched)
    assert sentinel.read_text(encoding="utf-8") == '{"sentinel": "do-not-touch"}'
    # the ONLY new path is the out file; nothing pre-existing changed
    new_paths = set(after) - set(before)
    assert new_paths == {str(out_path.relative_to(tmp_path))}
    for path, digest in before.items():
        assert after[path] == digest, f"stray write to pre-existing file {path}"


# ---------------------------------------------------------------------------
# 7. CLI end-to-end (python -m ... ) + rule-sha gate
# ---------------------------------------------------------------------------
def test_cli_main_writes_default_out(tmp_path):
    data_dir = _write_evidence(tmp_path, status_rows=_clean_status_rows())
    runtime_dir = tmp_path / "runtime"
    rc = rwc.main([
        "--date", DATE,
        "--data-dir", str(data_dir),
        "--runtime-dir", str(runtime_dir),
        "--evaluator-identity", "cli-test",
        "--evaluated-at", "2026-06-20T23:00:00Z",
    ])
    assert rc == 0
    out = runtime_dir / "session_explanations.json"  # default out under runtime-dir
    assert out.is_file()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["counts"]["EXPLAINED"] == 1
    assert doc["evaluator_identity"] == "cli-test"


def test_cli_explicit_out_path(tmp_path):
    data_dir = _write_evidence(tmp_path, status_rows=_clean_status_rows())
    out = tmp_path / "custom" / "explain.json"
    rc = rwc.main([
        "--date", DATE, "--data-dir", str(data_dir),
        "--runtime-dir", str(tmp_path / "rt"), "--out", str(out),
        "--evaluated-at", "2026-06-20T23:00:00Z",
    ])
    assert rc == 0
    assert out.is_file()

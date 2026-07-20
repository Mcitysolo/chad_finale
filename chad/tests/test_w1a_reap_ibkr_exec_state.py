"""
W1A-6 — status-aware, gated reaper for ibkr_adapter_state.sqlite3::ibkr_exec_state.

Locks: detection is read-only (byte-identical DB), dry-run is default, token
discipline (rc 2 without/with wrong --confirm), selectivity (deletes only stale
NON-terminal rows; preserves Filled/Cancelled evidence + recent rows), archive
-before-mutate, idempotent rerun, gate refusals before any write, the wrong-store
hard-target guard, the incumbent-policy diff, and classifier parity with the
adapter. Every test uses tmp_path sqlite — the reaper never touches real runtime.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "reap_ibkr_exec_state", REPO / "scripts" / "reap_ibkr_exec_state.py")
reap = importlib.util.module_from_spec(_spec)
sys.modules["reap_ibkr_exec_state"] = reap
_spec.loader.exec_module(reap)

NOW = dt.datetime(2026, 7, 18, 0, 0, 0, tzinfo=dt.timezone.utc)


def _seed_db(path: Path, now: dt.datetime = NOW) -> None:
    old = (now - dt.timedelta(days=60)).isoformat()
    recent = (now - dt.timedelta(days=1)).isoformat()
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE ibkr_exec_state (idempotency_key TEXT PRIMARY KEY, status TEXT NOT NULL,
           created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL, broker_order_id INTEGER,
           payload_json TEXT NOT NULL, result_json TEXT)"""
    )
    rows = [
        ("k_old_pending", "PendingSubmit", old),      # stale non-terminal -> DELETE
        ("k_old_submitted", "Submitted", old),         # stale non-terminal -> DELETE
        ("k_old_validation", "ValidationError", old),  # stale non-terminal -> DELETE
        ("k_old_filled", "Filled", old),               # stale TERMINAL -> PRESERVE
        ("k_old_cancelled", "Cancelled", old),         # stale TERMINAL -> PRESERVE
        ("k_recent_pending", "PendingSubmit", recent),  # recent non-terminal -> retain
    ]
    for k, s, u in rows:
        conn.execute("INSERT INTO ibkr_exec_state VALUES (?,?,?,?,?,?,?)", (k, s, u, u, None, "{}", None))
    conn.commit()
    conn.close()


def _db(tmp_path: Path) -> Path:
    p = tmp_path / "ibkr_adapter_state.sqlite3"
    _seed_db(p)
    return p


def _pass_gates(monkeypatch) -> None:
    monkeypatch.setattr(reap, "run_gates",
                        lambda rd: (True, {"exec_mode": "paper", "scr": "CONFIDENT", "reconciliation": "GREEN"}))


# ---------------------------------------------------------------------------
# Detection (read-only) + incumbent-diff
# ---------------------------------------------------------------------------


def test_detection_is_read_only_and_counts_correct(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sha_before = reap._sha256_file(db)
    rep = reap.analyze(db, older_than_days=30, now=NOW)

    assert rep["total_rows"] == 6
    assert rep["delete_candidate_count"] == 3  # the 3 stale non-terminal rows
    assert rep["retain_count"] == 3            # 2 terminal + 1 recent
    assert reap._sha256_file(db) == sha_before  # detection mutates nothing


def test_incumbent_diff_exposes_evidence_the_age_only_reaper_destroys(tmp_path: Path) -> None:
    db = _db(tmp_path)
    rep = reap.analyze(db, older_than_days=30, now=NOW)

    # Incumbent (age-only, status-blind) would delete ALL 5 old rows...
    assert rep["incumbent_age_only_would_delete_count"] == 5
    # ...including the 2 terminal evidence rows (Filled + Cancelled) that
    # status-aware logic preserves — the case for the harden-incumbent PA.
    assert rep["preserved_by_status_awareness_count"] == 2
    statuses = {r["status"] for r in rep["preserved_by_status_awareness"]}
    assert statuses == {"Filled", "Cancelled"}


def test_main_detection_returns_zero_no_mutation(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sha_before = reap._sha256_file(db)
    rc = reap.main(["--db", str(db), "--older-than-days", "30"])
    assert rc == 0
    assert reap._sha256_file(db) == sha_before


# ---------------------------------------------------------------------------
# Token discipline
# ---------------------------------------------------------------------------


def test_execute_without_confirm_refuses_rc2(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sha_before = reap._sha256_file(db)
    assert reap.main(["--db", str(db), "--execute"]) == 2
    assert reap._sha256_file(db) == sha_before


def test_execute_wrong_token_refuses_rc2(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sha_before = reap._sha256_file(db)
    assert reap.main(["--db", str(db), "--execute", "--confirm", "WRONG"]) == 2
    assert reap._sha256_file(db) == sha_before


def test_execute_correct_token_applies_and_is_selective(tmp_path: Path, monkeypatch) -> None:
    db = _db(tmp_path)
    _pass_gates(monkeypatch)
    rc = reap.main(["--db", str(db), "--runtime-dir", str(tmp_path),
                    "--older-than-days", "30", "--execute", "--confirm", reap.CONFIRM_TOKEN])
    assert rc == 0

    conn = sqlite3.connect(str(db))
    remaining = {r[0] for r in conn.execute("SELECT idempotency_key FROM ibkr_exec_state")}
    conn.close()
    # Deleted: the 3 stale non-terminal rows.
    assert "k_old_pending" not in remaining
    assert "k_old_submitted" not in remaining
    assert "k_old_validation" not in remaining
    # Preserved: terminal evidence + the recent non-terminal row.
    assert {"k_old_filled", "k_old_cancelled", "k_recent_pending"} <= remaining
    # Archive-before-mutate produced a .bak_reap_* alongside the DB.
    assert list(tmp_path.glob("ibkr_adapter_state.sqlite3.bak_reap_*"))


def test_idempotent_rerun_deletes_nothing_more(tmp_path: Path, monkeypatch) -> None:
    db = _db(tmp_path)
    _pass_gates(monkeypatch)
    first = reap.purge(db, older_than_days=30, runtime_dir=tmp_path, now=NOW)
    assert first["rows_deleted"] == 3
    second = reap.purge(db, older_than_days=30, runtime_dir=tmp_path, now=NOW)
    assert second["applied"] is True
    assert second["rows_deleted"] == 0  # nothing stale-non-terminal remains


# ---------------------------------------------------------------------------
# Gate refusals — refuse BEFORE any write (real run_gates, temp runtime files)
# ---------------------------------------------------------------------------


def _write_runtime(runtime: Path, scr_state: str, recon_status: str) -> None:
    import json
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "scr_state.json").write_text(json.dumps({"state": scr_state}), encoding="utf-8")
    (runtime / "reconciliation_state.json").write_text(json.dumps({"status": recon_status}), encoding="utf-8")


def test_gate_refusal_scr_paused_no_write(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sha_before = reap._sha256_file(db)
    runtime = tmp_path / "runtime"
    _write_runtime(runtime, scr_state="PAUSED", recon_status="GREEN")

    result = reap.purge(db, older_than_days=30, runtime_dir=runtime, now=NOW)

    assert result["refused"] is True
    assert result["applied"] is False
    assert reap._sha256_file(db) == sha_before                 # no mutation
    assert not list(tmp_path.glob("*.bak_reap_*"))             # no archive either


def test_gate_refusal_reconciliation_red_no_write(tmp_path: Path) -> None:
    db = _db(tmp_path)
    sha_before = reap._sha256_file(db)
    runtime = tmp_path / "runtime"
    _write_runtime(runtime, scr_state="CONFIDENT", recon_status="RED")

    result = reap.purge(db, older_than_days=30, runtime_dir=runtime, now=NOW)

    assert result["refused"] is True
    assert reap._sha256_file(db) == sha_before


# ---------------------------------------------------------------------------
# Wrong-store hard-target guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["exec_state_paper.sqlite3", "ibkr_exec_state.db", "something_else.sqlite3"])
def test_wrong_store_name_refused_rc4(tmp_path: Path, name: str) -> None:
    p = tmp_path / name
    p.write_bytes(b"")  # exists, but wrong name
    assert reap.main(["--db", str(p)]) == 4


def test_missing_target_db_refused_rc4(tmp_path: Path) -> None:
    assert reap.main(["--db", str(tmp_path / "ibkr_adapter_state.sqlite3")]) == 4  # correct name, absent


def test_non_positive_older_than_days_refused_rc3(tmp_path: Path) -> None:
    db = _db(tmp_path)
    assert reap.main(["--db", str(db), "--older-than-days", "0"]) == 3


# ---------------------------------------------------------------------------
# Classifier parity with the adapter (locks divergence)
# ---------------------------------------------------------------------------


def test_classifier_vocabulary_matches_adapter() -> None:
    from chad.execution.ibkr_adapter import (
        _IDEMPOTENCY_TERMINAL_POSITIVE, _IDEMPOTENCY_TERMINAL_NEGATIVE,
    )
    assert reap._TERMINAL_POSITIVE == _IDEMPOTENCY_TERMINAL_POSITIVE
    assert reap._TERMINAL_NEGATIVE == _IDEMPOTENCY_TERMINAL_NEGATIVE

"""W2A-2 — scripts/ghost_scrub_pff1.py acceptance tests.

Covers the safety rails and the deterministic scrub DELTA on a fixture ledger:
  * dry-run mutates nothing; execute writes exactly the 6-record manifest + a report;
  * fail-closed verification (missing hash / mismatched qty-pnl / pnl_untrusted row);
  * idempotent NOOP re-run; refuses to overwrite a different manifest without --idempotent-ok;
  * wrong confirm token refused;
  * BOTH scorekeepers then drop the 6: Stage-2 admitted -6 (trusted-pnl += 145.79) and SCR's
    get_exclusion_sets pins the 6 record_hashes.

The live SCR effective 73->67 / total_pnl -375.60 is a Phase-3 verification (gated on the SCR
shadow server reloading PFF1-Q2, per D2); here the DELTA is asserted on the fixture so it is
deterministic and env-free.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ghost_scrub_pff1", REPO / "scripts" / "ghost_scrub_pff1.py")
gs = importlib.util.module_from_spec(_spec)
sys.modules["ghost_scrub_pff1"] = gs
_spec.loader.exec_module(gs)

from chad.utils.quarantine import get_exclusion_sets  # SCR exclusion authority
from chad.validation.trade_log_adapter import run_adapter


# --------------------------------------------------------------------------- #
# Fixture ledger: the 6 real phantom records + seq1 untrusted seed + 1 unrelated clean row.
# --------------------------------------------------------------------------- #
def _phantom_row(spec: dict) -> dict:
    qty, pnl, side = spec["qty"], spec["pnl"], spec["side"]
    return {
        "payload": {
            "schema_version": "closed_trade.v1",
            "strategy": "gamma", "symbol": "UNH", "side": side,
            "pnl": pnl, "fill_price": 425.0, "quantity": qty,
            "notional": abs(qty * 425.0), "broker": "paper_exec", "is_live": False,
            "entry_time_utc": "2026-07-20T13:50:56Z", "exit_time_utc": "2026-07-20T13:55:45Z",
            "tags": ["paper", "closed", "gamma"], "extra": {},
        },
        "sequence_id": 1, "record_hash": spec["record_hash"],
    }


def _seed_untrusted_row() -> dict:
    return {
        "payload": {
            "schema_version": "closed_trade.v1",
            "strategy": "gamma", "symbol": "UNH", "side": "BUY",
            "pnl": 625.17, "fill_price": 423.0, "quantity": 273.0, "notional": 114853.83,
            "broker": "paper_exec", "is_live": False,
            "entry_time_utc": "2026-07-15T17:36:18Z", "exit_time_utc": "2026-07-20T13:50:55Z",
            "tags": ["paper", "closed", "gamma", "pnl_untrusted", "scoring_excluded"],
            "extra": {"pnl_untrusted": True, "scoring_excluded": True},
        },
        "sequence_id": 0,
        "record_hash": "cf302a4b7e74b71ec91cd4ad4cd4c13741d441d515f4c5c70ec66e28644287ff",
    }


def _unrelated_clean_row() -> dict:
    return {
        "payload": {
            "schema_version": "closed_trade.v1",
            "strategy": "alpha", "symbol": "SPY", "side": "SELL",
            "pnl": 10.0, "fill_price": 500.0, "quantity": 3.0, "notional": 1500.0,
            "broker": "paper_exec", "is_live": False,
            "entry_time_utc": "2026-07-20T14:00:00Z", "exit_time_utc": "2026-07-20T14:05:00Z",
            "tags": ["paper", "closed", "alpha"], "extra": {},
        },
        "sequence_id": 99, "record_hash": "a" * 64,
    }


def _make_repo(tmp_path: Path, *, mutate=None) -> Path:
    trades = tmp_path / "data" / "trades"
    trades.mkdir(parents=True, exist_ok=True)
    rows = [_seed_untrusted_row()] + [_phantom_row(s) for s in gs.EXPECTED_RECORDS] + [_unrelated_clean_row()]
    if mutate:
        mutate(rows)
    with (trades / "trade_history_20260720.ndjson").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _paper_mode(monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")


def _manifest_path(repo: Path) -> Path:
    return repo / "runtime" / gs.MANIFEST_NAME


# --------------------------------------------------------------------------- #
# dry-run / execute / idempotency / gates
# --------------------------------------------------------------------------- #
def test_dry_run_mutates_nothing(tmp_path):
    repo = _make_repo(tmp_path)
    rc = gs.main(["--repo-root", str(repo)])
    assert rc == 0
    assert not _manifest_path(repo).exists()
    assert not (repo / "reports").exists()


def test_execute_writes_manifest_and_report(tmp_path):
    repo = _make_repo(tmp_path)
    rc = gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN])
    assert rc == 0
    mp = _manifest_path(repo)
    assert mp.is_file()
    doc = json.loads(mp.read_text(encoding="utf-8"))
    pinned = {e["record_hash"] for e in doc["invalid_trades"]}
    assert pinned == {r["record_hash"] for r in gs.EXPECTED_RECORDS}
    assert len(pinned) == 6
    # a signed report was written.
    reports = list((repo / "reports").glob("ghost_scrub_pff1_*.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text(encoding="utf-8"))
    assert rep["sum_qty"] == 233.0
    assert rep["sum_pnl"] == -145.79
    assert rep["expected_effect"]["scr_effective_delta"] == -6


def test_idempotent_rerun_is_noop(tmp_path):
    repo = _make_repo(tmp_path)
    assert gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN]) == 0
    before = _manifest_path(repo).read_text(encoding="utf-8")
    # second run: NOOP (already-current manifest), no new backup, content unchanged.
    assert gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN]) == 0
    assert _manifest_path(repo).read_text(encoding="utf-8") == before
    assert not list((repo / "runtime").glob("*.bak_ghost_scrub_*"))


def test_wrong_confirm_token_refused(tmp_path):
    repo = _make_repo(tmp_path)
    rc = gs.main(["--repo-root", str(repo), "--execute", "--confirm", "WRONG"])
    assert rc == 2
    assert not _manifest_path(repo).exists()


def test_refuses_missing_record_hash(tmp_path):
    def drop_one(rows):
        # remove the 3rd phantom (seq4) from the ledger -> its hash won't be found.
        del rows[3]
    repo = _make_repo(tmp_path, mutate=drop_one)
    rc = gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN])
    assert rc == 2
    assert not _manifest_path(repo).exists()


def test_refuses_qty_mismatch(tmp_path):
    def bump_qty(rows):
        rows[2]["payload"]["quantity"] = 999.0  # first phantom row, wrong qty
    repo = _make_repo(tmp_path, mutate=bump_qty)
    rc = gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN])
    assert rc == 2
    assert not _manifest_path(repo).exists()


def test_refuses_pnl_untrusted_target(tmp_path):
    def taint(rows):
        rows[2]["payload"]["extra"]["pnl_untrusted"] = True  # a target now looks like Q2's seed
    repo = _make_repo(tmp_path, mutate=taint)
    rc = gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN])
    assert rc == 2


def test_refuses_overwrite_different_manifest_without_flag(tmp_path):
    repo = _make_repo(tmp_path)
    mp = _manifest_path(repo)
    mp.write_text(json.dumps({"invalid_trades": [{"record_hash": "deadbeef"}]}), encoding="utf-8")
    rc = gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN])
    assert rc == 2  # different content present, no --idempotent-ok
    # with the flag it overwrites, taking a .bak first.
    rc2 = gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN, "--idempotent-ok"])
    assert rc2 == 0
    assert list((repo / "runtime").glob("*.bak_ghost_scrub_*"))


# --------------------------------------------------------------------------- #
# The DELTA: both scorekeepers drop the 6 once the manifest is written.
# --------------------------------------------------------------------------- #
def test_both_scorekeepers_drop_the_six(tmp_path):
    repo = _make_repo(tmp_path)
    trades = repo / "data" / "trades"
    runtime = repo / "runtime"

    # BEFORE scrub: Stage-2 admits the 6 phantoms + the 1 unrelated clean row (seed is untrusted).
    before = run_adapter(trades_dir=trades, runtime_dir=runtime, generated_at="2026-07-20T00:00:00Z")
    assert before.manifest.admitted == 7
    assert before.manifest.excluded_by_reason["quarantined"] == 0
    before_pnl = sum(t.gross_pnl for t in before.admitted)

    # apply the scrub.
    assert gs.main(["--repo-root", str(repo), "--execute", "--confirm", gs.CONFIRM_TOKEN]) == 0

    # AFTER: Stage-2 admits only the unrelated clean row; the 6 are excluded_quarantined.
    after = run_adapter(trades_dir=trades, runtime_dir=runtime, generated_at="2026-07-20T00:00:00Z")
    assert after.manifest.admitted == 1
    assert after.manifest.excluded_by_reason["quarantined"] == 6
    after_pnl = sum(t.gross_pnl for t in after.admitted)

    # trusted-pnl DELTA: removing the 6 (net -145.79) raises the trusted total by +145.79.
    assert after_pnl - before_pnl == pytest.approx(145.79, abs=1e-6)

    # SCR authority: get_exclusion_sets pins exactly the 6 record_hashes.
    _fills, trade_hashes = get_exclusion_sets(
        runtime_dir=runtime, fills_dir=repo / "data" / "fills", trades_dir=trades
    )
    assert {r["record_hash"] for r in gs.EXPECTED_RECORDS} <= trade_hashes

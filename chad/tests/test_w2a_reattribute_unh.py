"""W2A-4 — scripts/reattribute_unh_pff1.py acceptance tests.

Covers the safety rails and the trusted re-attribution write:
  * dry-run mutates nothing; execute writes ONE trusted gamma|UNH=228 @424.97 FIFO lot +
    the matching guard entry, leaving broker_sync|UNH UNTOUCHED (D7);
  * fail-closed broker-truth verification (guard mirror != 228 / stale snapshot / missing snap
    / snapshot qty mismatch / wrong confirm token);
  * idempotent NOOP re-run; refuses an unexpected pre-existing gamma UNH queue;
  * the written lot is TRUSTED — not flagged by the seed-lot untrust backstop, so its close is
    scoreable (the point of a trusted vs UNATTRIBUTED_EPOCH3 re-attribution, D3);
  * integration with the D7 proof: after the write, detect_guard_vs_broker_drift_v2 sees UNH
    matched (guard 228 == broker 228), NOT a 456 over-count.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "reattribute_unh_pff1", REPO / "scripts" / "reattribute_unh_pff1.py")
ra = importlib.util.module_from_spec(_spec)
sys.modules["reattribute_unh_pff1"] = ra
_spec.loader.exec_module(ra)

from chad.core.position_guard import detect_guard_vs_broker_drift_v2
from chad.utils.quarantine import get_untrusted_fill_ids_from_fifo_lots


_FIXED_NOW = _dt.datetime(2026, 7, 20, 20, 15, 0, tzinfo=_dt.timezone.utc)
_REBUY_TS = "2026-07-20T13:55:46Z"


@pytest.fixture(autouse=True)
def _paper_and_clock(monkeypatch):
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    monkeypatch.setattr(ra, "_utcnow", lambda: _FIXED_NOW)


def _make_runtime(tmp_path: Path, *, guard_qty=228.0, guard_open=True, guard_side="BUY",
                  snap_qty=228.0, snap_age_s=30, snap_present=True, gamma_unh_lots=None) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    guard = {
        "_version": 7,
        "_written_by": "test",
        "broker_sync|UNH": {
            "open": guard_open, "strategy": "broker_sync", "symbol": "UNH",
            "side": guard_side, "quantity": guard_qty, "source": "broker_truth_rebuild",
        },
        "alpha|SPY": {"open": True, "strategy": "alpha", "symbol": "SPY", "side": "BUY", "quantity": 3.0},
    }
    (runtime / "position_guard.json").write_text(json.dumps(guard), encoding="utf-8")

    if snap_present:
        snap_ts = _FIXED_NOW - _dt.timedelta(seconds=snap_age_s)
        snap = {
            "ts_utc": snap_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "ttl_seconds": 300,
            "positions": [
                {"symbol": "UNH", "position": snap_qty, "avgCost": 424.98, "secType": "STK"},
                {"symbol": "SPY", "position": 3.0, "avgCost": 500.0, "secType": "STK"},
            ],
        }
        (runtime / "positions_snapshot.json").write_text(json.dumps(snap), encoding="utf-8")

    queues = [
        {"strategy": "broker_sync", "symbol": "UNH", "lots": [
            {"fill_id": "c400", "side": "BUY", "quantity": 40.0, "fill_price": 424.96, "ts_utc": _REBUY_TS, "multiplier": 1.0, "meta": {}},
            {"fill_id": "be0e", "side": "BUY", "quantity": 23.0, "fill_price": 424.95, "ts_utc": _REBUY_TS, "multiplier": 1.0, "meta": {}},
        ]},
        {"strategy": "gamma", "symbol": "SVXY", "lots": [
            {"fill_id": "s1", "side": "BUY", "quantity": 156.0, "fill_price": 57.62, "ts_utc": "2026-07-13T18:37:49Z", "multiplier": 1.0, "meta": {}},
        ]},
    ]
    if gamma_unh_lots is not None:
        queues.append({"strategy": "gamma", "symbol": "UNH", "lots": gamma_unh_lots})
    fifo = {"processed_fill_ids": ["old-1", "old-2"], "queues": queues, "saved_at_utc": "2026-07-20T13:55:46Z"}
    (runtime / "trade_closer_state.json").write_text(json.dumps(fifo), encoding="utf-8")
    return tmp_path


def _read(runtime: Path, name: str) -> dict:
    return json.loads((runtime / "runtime" / name).read_text(encoding="utf-8"))


def _gamma_unh_row(fifo: dict):
    for row in fifo["queues"]:
        if row["strategy"] == "gamma" and row["symbol"] == "UNH":
            return row
    return None


# --------------------------------------------------------------------------- #
# dry-run / execute / idempotency
# --------------------------------------------------------------------------- #
def test_dry_run_mutates_nothing(tmp_path):
    repo = _make_runtime(tmp_path)
    before_fifo = _read(repo, "trade_closer_state.json")
    before_guard = _read(repo, "position_guard.json")
    rc = ra.main(["--repo-root", str(repo)])
    assert rc == 0
    assert _read(repo, "trade_closer_state.json") == before_fifo
    assert _read(repo, "position_guard.json") == before_guard
    assert not (repo / "reports").exists()


def test_execute_writes_trusted_lot_and_guard(tmp_path):
    repo = _make_runtime(tmp_path)
    rc = ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN])
    assert rc == 0

    fifo = _read(repo, "trade_closer_state.json")
    row = _gamma_unh_row(fifo)
    assert row is not None and len(row["lots"]) == 1
    lot = row["lots"][0]
    assert lot["side"] == "BUY"
    assert lot["quantity"] == 228.0
    assert lot["fill_price"] == 424.97          # D4 broker-truth VWAP
    assert lot["ts_utc"] == _REBUY_TS           # real re-buy establishment time
    assert lot["fill_id"].startswith("PFF1_REATTR_UNH_")
    # TRUSTED: no pnl_untrusted / scoring_excluded on the lot meta.
    assert lot["meta"].get("pnl_untrusted") is not True
    assert lot["meta"].get("scoring_excluded") is not True
    assert lot["meta"]["basis"] == "broker_truth_vwap"
    # processed_fill_ids retained + our fill marked.
    assert "old-1" in fifo["processed_fill_ids"] and lot["fill_id"] in fifo["processed_fill_ids"]

    guard = _read(repo, "position_guard.json")
    assert guard["gamma|UNH"]["open"] is True
    assert guard["gamma|UNH"]["quantity"] == 228.0
    assert guard["gamma|UNH"]["side"] == "BUY"
    # broker_sync|UNH UNTOUCHED (D7 truth anchor).
    assert guard["broker_sync|UNH"]["quantity"] == 228.0
    assert guard["broker_sync|UNH"]["source"] == "broker_truth_rebuild"

    # signed report + .bak of both files.
    assert len(list((repo / "reports").glob("reattribute_unh_pff1_*.json"))) == 1
    assert list((repo / "runtime").glob("trade_closer_state.json.bak_reattr_unh_*"))
    assert list((repo / "runtime").glob("position_guard.json.bak_reattr_unh_*"))


def test_written_lot_is_trusted_not_seed_untrusted(tmp_path):
    """The re-attribution lot must NOT be flagged by the seed-lot untrust backstop (it is a
    real broker position at a real basis) — so its eventual close is scoreable."""
    repo = _make_runtime(tmp_path)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 0
    untrusted = get_untrusted_fill_ids_from_fifo_lots(runtime_dir=repo / "runtime")
    fifo = _read(repo, "trade_closer_state.json")
    our_fid = _gamma_unh_row(fifo)["lots"][0]["fill_id"]
    assert our_fid not in untrusted


def test_idempotent_rerun_is_noop(tmp_path):
    repo = _make_runtime(tmp_path)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 0
    fifo_after_first = _read(repo, "trade_closer_state.json")
    # second run: NOOP (a gamma|UNH re-attribution lot already exists).
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 0
    assert _read(repo, "trade_closer_state.json") == fifo_after_first


def test_after_reattribution_drift_v2_is_matched(tmp_path):
    """Integration with the D7 proof: post-write the guard shows UNH matched, not over-counted."""
    repo = _make_runtime(tmp_path)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 0
    guard = _read(repo, "position_guard.json")
    d = detect_guard_vs_broker_drift_v2(guard)
    assert [r for r in d["drifts"] if r["symbol"] == "UNH"] == []  # matched (228==228), no 456


# --------------------------------------------------------------------------- #
# fail-closed
# --------------------------------------------------------------------------- #
def test_refuses_wrong_confirm(tmp_path):
    repo = _make_runtime(tmp_path)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", "WRONG"]) == 2
    assert _gamma_unh_row(_read(repo, "trade_closer_state.json")) is None


def test_refuses_guard_mirror_not_228(tmp_path):
    repo = _make_runtime(tmp_path, guard_qty=200.0)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 2


def test_refuses_snapshot_qty_mismatch(tmp_path):
    repo = _make_runtime(tmp_path, snap_qty=200.0)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 2


def test_refuses_stale_snapshot(tmp_path):
    repo = _make_runtime(tmp_path, snap_age_s=100_000)  # > ttl*3
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 2


def test_refuses_missing_snapshot(tmp_path):
    repo = _make_runtime(tmp_path, snap_present=False)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 2


def test_refuses_unexpected_existing_gamma_unh_queue(tmp_path):
    # a gamma UNH queue with a NON-reattribution lot -> refuse (do not overwrite).
    stray = [{"fill_id": "stray", "side": "BUY", "quantity": 100.0, "fill_price": 400.0,
              "ts_utc": _REBUY_TS, "multiplier": 1.0, "meta": {}}]
    repo = _make_runtime(tmp_path, gamma_unh_lots=stray)
    assert ra.main(["--repo-root", str(repo), "--execute", "--confirm", ra.CONFIRM_TOKEN]) == 2

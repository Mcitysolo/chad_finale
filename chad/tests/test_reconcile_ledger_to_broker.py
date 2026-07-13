"""D4 tests — scripts/reconcile_ledger_to_broker.py.

Covers: dispositions (ADOPT / REBASELINE_EXCLUDED / REBASELINE_PHANTOM); dry-run mutates nothing;
execute converges (guard = broker truth, FIFO sum = broker truth, drift->0 for CHAD symbols) while
PRESERVING every removed lot in the reconciliation ledger (relabel, never delete); broker_sync is
never touched; idempotent second run is a no-op; and the synthetic records are provably excluded
by the real Stage-2 trust filter (trade_stats_engine._is_untrusted).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "reconcile_ledger_to_broker", REPO / "scripts" / "reconcile_ledger_to_broker.py")
rec = importlib.util.module_from_spec(_spec)
sys.modules["reconcile_ledger_to_broker"] = rec  # dataclass processing needs this registered
_spec.loader.exec_module(rec)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _lot(fid, qty, price=100.0):
    return {"fill_id": fid, "side": "BUY", "quantity": qty, "fill_price": price,
            "ts_utc": "2026-07-10T13:30:00Z", "multiplier": 1.0, "meta": {}}


def _guard_state():
    # broker truth: TLT 420 (CHAD), LLY 130 (operator-excluded); SPY has NO broker_sync (phantom).
    return {
        "_version": 1,
        "broker_sync|TLT": {"open": False, "symbol": "TLT", "side": "BUY", "quantity": 420.0,
                            "strategy": "broker_sync", "closed_by": "strategy_ownership_assumed"},
        "broker_sync|LLY": {"open": False, "symbol": "LLY", "side": "BUY", "quantity": 130.0,
                            "strategy": "broker_sync", "closed_by": "strategy_ownership_assumed"},
    }


def _fifo_state():
    return {
        "saved_at_utc": "2026-07-13T00:00:00Z",
        "processed_fill_ids": ["old-fid-1", "old-fid-2"],
        "queues": [
            {"strategy": "gamma", "symbol": "TLT",
             "lots": [_lot("t1", 500), _lot("t2", 400), _lot("t3", 360)]},   # FIFO 1260 vs broker 420
            {"strategy": "gamma", "symbol": "LLY",
             "lots": [_lot("l1", 200), _lot("l2", 106)]},                      # excluded -> rebaseline
            {"strategy": "gamma", "symbol": "SPY",
             "lots": [_lot("s1", 300)]},                                       # phantom (no broker) -> rebaseline
        ],
    }


def _exclusions():
    return {"LLY", "AAPL", "BAC"}  # LLY operator-excluded; SPY deliberately NOT excluded (isolate phantom path)


def _marks():
    return {"TLT": 84.0}


def _plan(fifo=None, guard=None, excl=None, marks=None):
    fifo = fifo or _fifo_state()
    guard = guard or _guard_state()
    return rec.compute_plan(
        queues=fifo["queues"],
        broker_signed=rec.broker_signed_by_symbol(guard),
        exclusions=excl if excl is not None else _exclusions(),
        marks=marks if marks is not None else _marks(),
    )


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #
def test_compute_plan_dispositions():
    plan = {p.symbol: p for p in _plan()}
    assert plan["TLT"].disposition == rec.DISP_ADOPT
    assert plan["TLT"].seed_qty == 420.0 and plan["TLT"].target_strategy == "gamma"
    assert plan["TLT"].fifo_qty_before == 1260.0 and plan["TLT"].mark == 84.0
    assert plan["LLY"].disposition == rec.DISP_REBASELINE_EXCLUDED and plan["LLY"].excluded
    assert plan["SPY"].disposition == rec.DISP_REBASELINE_PHANTOM and plan["SPY"].broker_qty == 0.0


def test_broker_signed_reads_broker_sync_regardless_of_open():
    signed = rec.broker_signed_by_symbol(_guard_state())
    assert signed == {"TLT": 420.0, "LLY": 130.0}


# --------------------------------------------------------------------------- #
# apply: convergence + evidence preservation + broker_sync untouched
# --------------------------------------------------------------------------- #
def test_apply_converges_and_preserves_every_lot():
    guard, fifo = _guard_state(), _fifo_state()
    plan = _plan(fifo, guard)
    res = rec.apply_plan(plan=plan, guard_state=guard, fifo_state=fifo,
                         now_iso="2026-07-13T16:00:00Z", stamp="20260713T160000Z")

    # FIFO: TLT collapses to a single seed lot at broker truth; LLY + SPY rows removed.
    q = {r["symbol"]: r for r in res.new_fifo["queues"]}
    assert set(q) == {"TLT"}
    assert rec._row_qty(q["TLT"]) == 420.0
    seed = rec._row_lots(q["TLT"])[0]
    assert seed["meta"]["reconciled"] is True and seed["meta"]["pnl_untrusted"] is True
    assert seed["meta"]["provenance"] == rec.PROV_ADOPT

    # guard: gamma|TLT opened at broker truth; broker_sync entries byte-identical (untouched).
    assert res.new_guard["gamma|TLT"]["open"] is True
    assert res.new_guard["gamma|TLT"]["quantity"] == 420.0
    assert res.new_guard["broker_sync|TLT"] == guard["broker_sync|TLT"]
    assert res.new_guard["broker_sync|LLY"] == guard["broker_sync|LLY"]

    # drift check: after recon, guard OPEN strategy qty for TLT == broker truth -> no drift.
    open_tlt = res.new_guard["gamma|TLT"]["quantity"] if res.new_guard["gamma|TLT"]["open"] else 0
    assert open_tlt == rec.broker_signed_by_symbol(res.new_guard)["TLT"]

    # EVIDENCE: every original lot is preserved in a ledger record (nothing hard-deleted).
    original_lots = sum(len(rec._row_lots(r)) for r in fifo["queues"])  # 3+2+1 = 6
    preserved = sum(r["removed_lot_count"] for r in res.ledger_records)
    assert preserved == original_lots == 6
    # all removed lot bodies carried in the ledger records
    preserved_fids = {f for r in res.ledger_records for f in r["removed_fill_ids"]}
    assert preserved_fids == {"t1", "t2", "t3", "l1", "l2", "s1"}
    # every ledger record is trust-filter-excludable
    assert all(r["pnl_untrusted"] is True and "pnl_untrusted" in r["tags"] for r in res.ledger_records)

    # processed_fill_ids RETAINED + adoption fid added (never re-open old fills).
    assert "old-fid-1" in res.new_fifo["processed_fill_ids"]
    assert "RECON_ADOPT_TLT_20260713T160000Z" in res.new_fifo["processed_fill_ids"]


def test_apply_does_not_mutate_inputs():
    guard, fifo = _guard_state(), _fifo_state()
    before = json.dumps(fifo, sort_keys=True)
    rec.apply_plan(plan=_plan(fifo, guard), guard_state=guard, fifo_state=fifo,
                   now_iso="2026-07-13T16:00:00Z", stamp="s")
    assert json.dumps(fifo, sort_keys=True) == before  # inputs untouched (deepcopy)


# --------------------------------------------------------------------------- #
# idempotency
# --------------------------------------------------------------------------- #
def test_idempotent_second_run_is_noop():
    guard, fifo = _guard_state(), _fifo_state()
    res = rec.apply_plan(plan=_plan(fifo, guard), guard_state=guard, fifo_state=fifo,
                         now_iso="2026-07-13T16:00:00Z", stamp="s")
    # Re-plan against the reconciled FIFO — TLT already seeded at broker truth -> no ADOPT row.
    plan2 = rec.compute_plan(
        queues=res.new_fifo["queues"],
        broker_signed=rec.broker_signed_by_symbol(res.new_guard),
        exclusions=_exclusions(), marks=_marks())
    assert plan2 == []  # nothing left to do


# --------------------------------------------------------------------------- #
# Stage-2 trust filter: synthetic rows provably excluded (adapter test)
# --------------------------------------------------------------------------- #
def test_synthetic_records_excluded_by_real_trust_filter():
    from chad.analytics.trade_stats_engine import _is_untrusted
    guard, fifo = _guard_state(), _fifo_state()
    res = rec.apply_plan(plan=_plan(fifo, guard), guard_state=guard, fifo_state=fifo,
                         now_iso="2026-07-13T16:00:00Z", stamp="s")
    # every reconciliation ledger record is excluded by the canonical SCR trust predicate
    for r in res.ledger_records:
        assert _is_untrusted(r["tags"], {"pnl_untrusted": r["pnl_untrusted"]}) is True
    # the adoption seed lot's meta also trips the extra.pnl_untrusted gate
    seed_meta = rec._row_lots({"lots": rec._row_lots(
        [r for r in res.new_fifo["queues"] if r["symbol"] == "TLT"][0])})[0]["meta"]
    assert _is_untrusted([], seed_meta) is True


# --------------------------------------------------------------------------- #
# gates + confirmation
# --------------------------------------------------------------------------- #
def _write_runtime(tmp: Path, scr="CONFIDENT", recon_status="GREEN"):
    rt = tmp / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    (rt / "scr_state.json").write_text(json.dumps({"state": scr}))
    (rt / "reconciliation_state.json").write_text(json.dumps({"status": recon_status, "exclusion_policy": {"LLY": {}}}))
    (rt / "price_cache.json").write_text(json.dumps({"prices": {"TLT": 84.0}, "ts_utc": "2026-07-13T16:00:00Z", "ttl_seconds": 300}))
    (rt / "position_guard.json").write_text(json.dumps(_guard_state()))
    (rt / "trade_closer_state.json").write_text(json.dumps(_fifo_state()))
    return rt


def test_run_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_gate_exec_mode_paper", lambda: (True, "exec_mode=paper"))
    rt = _write_runtime(tmp_path)
    ok, reasons = rec.run_gates(rt)
    assert ok, reasons
    rt_bad = _write_runtime(tmp_path / "bad", scr="UNKNOWN")
    ok2, reasons2 = rec.run_gates(rt_bad)
    assert not ok2 and "scr" in reasons2
    rt_red = _write_runtime(tmp_path / "red", recon_status="RED")
    ok3, _ = rec.run_gates(rt_red)
    assert not ok3


# --------------------------------------------------------------------------- #
# main(): dry-run mutates nothing; execute converges + writes report/backups
# --------------------------------------------------------------------------- #
def test_main_dry_run_mutates_nothing(tmp_path):
    _write_runtime(tmp_path)
    guard_before = (tmp_path / "runtime" / "position_guard.json").read_text()
    fifo_before = (tmp_path / "runtime" / "trade_closer_state.json").read_text()
    code = rec.main(["--repo-root", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "runtime" / "position_guard.json").read_text() == guard_before
    assert (tmp_path / "runtime" / "trade_closer_state.json").read_text() == fifo_before
    assert not list((tmp_path / "reports").glob("*.json")) if (tmp_path / "reports").exists() else True


def test_main_execute_converges_and_writes_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_gate_exec_mode_paper", lambda: (True, "exec_mode=paper"))
    rt = _write_runtime(tmp_path)
    # refuses without the exact confirm token
    assert rec.main(["--repo-root", str(tmp_path), "--execute"]) == 2
    assert rec.main(["--repo-root", str(tmp_path), "--execute", "--confirm", "WRONG"]) == 2
    # applies with the token
    code = rec.main(["--repo-root", str(tmp_path), "--execute", "--confirm", rec.CONFIRM_TOKEN])
    assert code == 0
    fifo = json.loads((rt / "trade_closer_state.json").read_text())
    q = {r["symbol"]: r for r in fifo["queues"]}
    assert set(q) == {"TLT"} and rec._row_qty(q["TLT"]) == 420.0
    guard = json.loads((rt / "position_guard.json").read_text())
    assert guard["gamma|TLT"]["open"] is True and guard["gamma|TLT"]["quantity"] == 420.0
    assert guard["broker_sync|TLT"]["quantity"] == 420.0  # broker truth untouched
    # backups + report + ledger written
    assert list(rt.glob("position_guard.json.bak_ledger_recon_*"))
    assert list(rt.glob("trade_closer_state.json.bak_ledger_recon_*"))
    assert list(rt.glob("ledger_reconciliation_*.ndjson"))
    assert list((tmp_path / "reports").glob("ledger_recon_*.json"))
    # second execute is idempotent -> plan empty -> converged, no new backup churn error
    assert rec.main(["--repo-root", str(tmp_path), "--execute", "--confirm", rec.CONFIRM_TOKEN]) == 0

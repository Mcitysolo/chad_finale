from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from chad.execution import exit_only_consumer_paper as m


def _write_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def _minimal_plan() -> dict:
    return {
        "schema_version": "phase9.1.exit_only_plan.v1",
        "ts_utc": "2026-01-30T00:03:49Z",
        "exits_count": 1,
        "exits": [
            {
                "symbol": "AAPL",
                "side": "SELL",
                "qty": 1.0,
                "currency": "USD",
                "asset_class": "equity",
                "reason": "exit_only_flatten",
            }
        ],
    }


def test_preview_does_not_write_reports(tmp_path: Path, monkeypatch) -> None:
    # point runtime/report dirs to temp
    rt = tmp_path / "runtime"
    rp = tmp_path / "reports"
    rt.mkdir()
    (rp / "exit_only").mkdir(parents=True, exist_ok=True)

    plan_path = rt / "exit_only_plan.json"
    stop_path = rt / "stop_state.json"
    _write_json(plan_path, _minimal_plan())
    _write_json(stop_path, {"stop": False, "reason": "ok", "ts_utc": "x", "ttl_seconds": 999})

    monkeypatch.setattr(m, "RUNTIME_DIR", rt)
    monkeypatch.setattr(m, "PLAN_PATH", plan_path)
    monkeypatch.setattr(m, "STOP_PATH", stop_path)
    monkeypatch.setattr(m, "REPORTS_DIR", rp / "exit_only")

    before = list((rp / "exit_only").glob("EXIT_ONLY_RECEIPTS_*.json"))
    rc = m.run(execute=False)
    after = list((rp / "exit_only").glob("EXIT_ONLY_RECEIPTS_*.json"))

    assert rc == 0
    assert before == after, "preview must not write report files"


def test_receipt_id_is_stable() -> None:
    plan = _minimal_plan()
    ph = m._plan_hash(plan)
    rid1 = m._receipt_id(ph, "AAPL", "SELL", 1.0)
    rid2 = m._receipt_id(ph, "AAPL", "SELL", 1.0)
    assert rid1 == rid2


def test_execute_blocks_when_live_gate_not_exit_only(monkeypatch, tmp_path: Path) -> None:
    # stop false + plan exists, but live-gate denies -> must block
    rt = tmp_path / "runtime"
    rp = tmp_path / "reports"
    rt.mkdir()
    (rp / "exit_only").mkdir(parents=True, exist_ok=True)

    plan_path = rt / "exit_only_plan.json"
    stop_path = rt / "stop_state.json"
    _write_json(plan_path, _minimal_plan())
    _write_json(stop_path, {"stop": False, "reason": "ok", "ts_utc": "x", "ttl_seconds": 999})

    monkeypatch.setattr(m, "PLAN_PATH", plan_path)
    monkeypatch.setattr(m, "STOP_PATH", stop_path)
    monkeypatch.setattr(m, "REPORTS_DIR", rp / "exit_only")

    # deny gate
    monkeypatch.setattr(
        m,
        "_fetch_live_gate",
        lambda: m.GateSnapshot(
            operator_mode="ALLOW_LIVE",
            allow_exits_only=False,
            allow_ibkr_paper=True,
            allow_ibkr_live=False,
        ),
    )

    rc = m.run(execute=True)
    assert rc == 0
    # execute mode should write a blocked report
    files = list((rp / "exit_only").glob("EXIT_ONLY_RECEIPTS_*.json"))
    assert files, "blocked execute must still emit a report"

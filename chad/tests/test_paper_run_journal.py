from __future__ import annotations

from pathlib import Path

from chad.execution.paper_run_journal import PaperRunJournal


def test_start_finish_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite3"
    j = PaperRunJournal(db)

    r1 = j.start(run_id="R1", mode="PREVIEW", planned_hash="H")
    assert r1.ok is True
    assert r1.inserted is True
    assert r1.reason == "inserted"

    r2 = j.start(run_id="R1", mode="PREVIEW", planned_hash="H")
    assert r2.ok is True
    assert r2.inserted is False
    assert r2.reason == "duplicate"

    f = j.finish(run_id="R1", outcome="blocked", error="", notes={"x": 1})
    assert f.ok is True
    assert f.updated is True
    assert f.reason == "updated"

    rec = j.get("R1")
    assert rec is not None
    assert rec["run_id"] == "R1"
    assert rec["mode"] == "PREVIEW"
    assert rec["planned_hash"] == "H"
    assert rec["outcome"] == "blocked"
    assert isinstance(rec["notes"], dict)


def test_finish_missing_run(tmp_path: Path) -> None:
    db = tmp_path / "runs.sqlite3"
    j = PaperRunJournal(db)
    f = j.finish(run_id="NOPE", outcome="error", error="x")
    assert f.ok is True
    assert f.updated is False
    assert f.reason == "missing"

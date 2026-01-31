from __future__ import annotations

from pathlib import Path

from chad.execution.idempotency_store import IdempotencyStore


def test_mark_once_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "exec_state_paper.sqlite3"
    store = IdempotencyStore(db)

    r1 = store.mark_once("T1", "H1", meta={"a": 1})
    assert r1.inserted is True
    assert r1.reason == "inserted"

    r2 = store.mark_once("T1", "H1", meta={"a": 2})
    assert r2.inserted is False
    assert r2.reason == "duplicate"

    rec = store.get("T1")
    assert rec is not None
    assert rec["trade_id"] == "T1"
    assert rec["payload_hash"] == "H1"
    assert isinstance(rec["meta"], dict)


def test_mark_once_rejects_empty_inputs(tmp_path: Path) -> None:
    db = tmp_path / "x.sqlite3"
    store = IdempotencyStore(db)

    assert store.mark_once("", "H").inserted is False
    assert store.mark_once("T", "").inserted is False
    assert store.get("") is None

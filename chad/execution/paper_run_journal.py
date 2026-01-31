from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class StartResult:
    ok: bool
    inserted: bool
    reason: str  # inserted | duplicate | error
    run_id: str


@dataclass(frozen=True)
class FinishResult:
    ok: bool
    updated: bool
    reason: str  # updated | missing | error
    run_id: str


class PaperRunJournal:
    """
    SQLite-backed run journal for paper shadow runner.

    Goals:
    - Crash-safe run evidence: start/end/outcome/error
    - Exactly-once per run_id (PRIMARY KEY)
    - Fail-soft public API: never raises
    - WAL mode for concurrent readers
    """

    def __init__(self, db_path: Path, *, table: str = "paper_runs", timeout_s: float = 3.0) -> None:
        self.db_path = Path(db_path)
        self.table = str(table).strip() or "paper_runs"
        self.timeout_s = float(timeout_s)
        self._ensure_db()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout_s,
            isolation_level=None,  # autocommit
        )
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        return con

    def _ensure_db(self) -> None:
        try:
            con = self._connect()
            try:
                con.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        run_id TEXT PRIMARY KEY,
                        started_utc TEXT NOT NULL,
                        ended_utc TEXT,
                        mode TEXT NOT NULL,
                        planned_hash TEXT,
                        outcome TEXT,
                        error TEXT,
                        notes_json TEXT NOT NULL
                    );
                    """
                )
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_started ON {self.table}(started_utc);")
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_ended ON {self.table}(ended_utc);")
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_outcome ON {self.table}(outcome);")
            finally:
                con.close()
        except Exception:
            # Fail-soft: caller will see error in API methods if DB unusable.
            pass

    def start(self, *, run_id: str, mode: str, planned_hash: str = "", notes: Optional[Dict[str, Any]] = None) -> StartResult:
        rid = str(run_id).strip()
        if not rid:
            return StartResult(ok=False, inserted=False, reason="error", run_id="")
        md = str(mode).strip().upper() or "UNKNOWN"
        ph = str(planned_hash).strip()
        notes_obj = notes if isinstance(notes, dict) else {}
        notes_json = json.dumps(notes_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        try:
            con = self._connect()
            try:
                cur = con.execute(
                    f"""
                    INSERT OR IGNORE INTO {self.table}
                      (run_id, started_utc, ended_utc, mode, planned_hash, outcome, error, notes_json)
                    VALUES (?, ?, NULL, ?, ?, NULL, NULL, ?);
                    """,
                    (rid, utc_now_iso(), md, ph, notes_json),
                )
                inserted = (cur.rowcount == 1)
                return StartResult(ok=True, inserted=inserted, reason=("inserted" if inserted else "duplicate"), run_id=rid)
            finally:
                con.close()
        except Exception:
            return StartResult(ok=False, inserted=False, reason="error", run_id=rid)

    def finish(self, *, run_id: str, outcome: str, error: str = "", notes: Optional[Dict[str, Any]] = None) -> FinishResult:
        rid = str(run_id).strip()
        if not rid:
            return FinishResult(ok=False, updated=False, reason="error", run_id="")
        oc = str(outcome).strip().lower() or "unknown"
        err = str(error).strip()
        notes_obj = notes if isinstance(notes, dict) else {}
        notes_json = json.dumps(notes_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        try:
            con = self._connect()
            try:
                cur = con.execute(
                    f"""
                    UPDATE {self.table}
                    SET ended_utc=?, outcome=?, error=?, notes_json=?
                    WHERE run_id=?;
                    """,
                    (utc_now_iso(), oc, err, notes_json, rid),
                )
                updated = (cur.rowcount == 1)
                if not updated:
                    return FinishResult(ok=True, updated=False, reason="missing", run_id=rid)
                return FinishResult(ok=True, updated=True, reason="updated", run_id=rid)
            finally:
                con.close()
        except Exception:
            return FinishResult(ok=False, updated=False, reason="error", run_id=rid)

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        rid = str(run_id).strip()
        if not rid:
            return None
        try:
            con = self._connect()
            try:
                cur = con.execute(
                    f"SELECT run_id, started_utc, ended_utc, mode, planned_hash, outcome, error, notes_json FROM {self.table} WHERE run_id=?;",
                    (rid,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                notes_json = row[7] if isinstance(row[7], str) else "{}"
                try:
                    notes = json.loads(notes_json)
                    if not isinstance(notes, dict):
                        notes = {}
                except Exception:
                    notes = {}
                return {
                    "run_id": row[0],
                    "started_utc": row[1],
                    "ended_utc": row[2],
                    "mode": row[3],
                    "planned_hash": row[4],
                    "outcome": row[5],
                    "error": row[6],
                    "notes": notes,
                }
            finally:
                con.close()
        except Exception:
            return None

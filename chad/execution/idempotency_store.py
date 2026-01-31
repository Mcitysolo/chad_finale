from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class MarkResult:
    inserted: bool
    reason: str  # inserted | duplicate | error
    trade_id: str


class IdempotencyStore:
    """
    SQLite-backed idempotency store.

    Guarantees:
    - Exactly-once marking for a given trade_id (PRIMARY KEY)
    - Safe across concurrent processes (SQLite locking)
    - Fail-soft: never raises in public methods
    - WAL mode for robustness under concurrent readers

    Intended use:
        store = IdempotencyStore(Path("runtime/exec_state_paper.sqlite3"), table="paper_trade_ids")
        res = store.mark_once(trade_id, payload_hash, meta={"source": "paper_shadow_runner"})
        if not res.inserted:
            # skip writing duplicate ledger record
    """

    def __init__(self, db_path: Path, *, table: str = "paper_trade_ids", timeout_s: float = 3.0) -> None:
        self.db_path = Path(db_path)
        self.table = str(table).strip() or "paper_trade_ids"
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
                        trade_id TEXT PRIMARY KEY,
                        first_seen_utc TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        meta_json TEXT NOT NULL
                    );
                    """
                )
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_first_seen ON {self.table}(first_seen_utc);")
            finally:
                con.close()
        except Exception:
            # Fail-soft: store may be unusable, but callers will see error result.
            pass

    @staticmethod
    def _utc_now_iso() -> str:
        # seconds precision is enough; caller can include more if desired
        import time
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def mark_once(self, trade_id: str, payload_hash: str, meta: Optional[Dict[str, Any]] = None) -> MarkResult:
        """
        Attempt to mark trade_id as seen.

        Returns inserted=True only on first successful insert.
        """
        tid = str(trade_id).strip()
        ph = str(payload_hash).strip()
        if not tid:
            return MarkResult(inserted=False, reason="error", trade_id="")
        if not ph:
            return MarkResult(inserted=False, reason="error", trade_id=tid)

        meta_obj = meta if isinstance(meta, dict) else {}
        meta_json = json.dumps(meta_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        try:
            con = self._connect()
            try:
                cur = con.execute(
                    f"""
                    INSERT OR IGNORE INTO {self.table} (trade_id, first_seen_utc, payload_hash, meta_json)
                    VALUES (?, ?, ?, ?);
                    """,
                    (tid, self._utc_now_iso(), ph, meta_json),
                )
                inserted = (cur.rowcount == 1)
                return MarkResult(inserted=inserted, reason=("inserted" if inserted else "duplicate"), trade_id=tid)
            finally:
                con.close()
        except Exception:
            return MarkResult(inserted=False, reason="error", trade_id=tid)

    def get(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Best-effort readback. Returns dict if present else None. Never raises.
        """
        tid = str(trade_id).strip()
        if not tid:
            return None
        try:
            con = self._connect()
            try:
                cur = con.execute(
                    f"SELECT trade_id, first_seen_utc, payload_hash, meta_json FROM {self.table} WHERE trade_id=?;",
                    (tid,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                meta_json = row[3] if isinstance(row[3], str) else "{}"
                try:
                    meta = json.loads(meta_json)
                    if not isinstance(meta, dict):
                        meta = {}
                except Exception:
                    meta = {}
                return {
                    "trade_id": row[0],
                    "first_seen_utc": row[1],
                    "payload_hash": row[2],
                    "meta": meta,
                }
            finally:
                con.close()
        except Exception:
            return None


def default_paper_db_path(repo_root: Path) -> Path:
    """
    Canonical location for paper idempotency state in this repo layout.
    """
    root = Path(repo_root)
    return root / "runtime" / "exec_state_paper.sqlite3"

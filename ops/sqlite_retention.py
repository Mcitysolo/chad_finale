#!/usr/bin/env python3
"""
SQLite retention pruner.
Removes ibkr_exec_state rows older than RETENTION_DAYS.
Runs VACUUM after deletion.
GAP-A019 / UNK-06 remediation.
"""
import sqlite3, datetime, logging, sys
from pathlib import Path

RETENTION_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 14
DB = Path(__file__).parent.parent / "runtime" / "ibkr_adapter_state.sqlite3"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("chad.sqlite_retention")

def run() -> None:
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)).isoformat()
    con = sqlite3.connect(str(DB))
    before = con.execute("SELECT count(*) FROM ibkr_exec_state").fetchone()[0]
    con.execute("DELETE FROM ibkr_exec_state WHERE updated_at_utc < ?", (cutoff,))
    con.commit()
    con.execute("VACUUM")
    after = con.execute("SELECT count(*) FROM ibkr_exec_state").fetchone()[0]
    con.close()
    log.info("ibkr_exec_state pruned: %d removed, %d retained (cutoff=%s)", before - after, after, cutoff)

if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
SQLite retention pruner.
Removes ibkr_exec_state rows older than the configured cutoff.
Runs VACUUM after deletion.
GAP-A019 / UNK-06 remediation.

Cutoff modes (mutually exclusive):
  - Legacy positional integer argument  → cutoff = utcnow - timedelta(days=N)
      Backwards-compatible. Default N=14 when no argument is given.
      Example:  python3 ops/sqlite_retention.py 1
  - Explicit UTC cutoff timestamp        → cutoff = the provided ISO-8601 timestamp
      Requires the explicit --cutoff flag. Recommended for boundary-precise
      cleanup (e.g. clearing pre-restart debt at a known restart timestamp).
      Example:  python3 ops/sqlite_retention.py --cutoff 2026-05-19T01:25:06Z

Dry-run:
  - Adding --dry-run prints the projected delete/retain counts and the
    full list of rows that WOULD be deleted, then exits without mutating
    the database. Safe to compose with either cutoff mode.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Optional

DB = Path(__file__).parent.parent / "runtime" / "ibkr_adapter_state.sqlite3"
TARGET_TABLE = "ibkr_exec_state"
DEFAULT_RETENTION_DAYS = 14

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("chad.sqlite_retention")


def _parse_iso_cutoff(value: str) -> str:
    """Normalize an ISO-8601 cutoff string to the column's storage format.

    Accepts trailing 'Z' or '+00:00'. Raises ValueError on unparseable input.
    Returns a string suitable for direct string-comparison against
    `updated_at_utc` in SQLite (column type TEXT, stored as ISO-8601).
    """
    s = str(value).strip()
    if not s:
        raise ValueError("empty cutoff")
    normalized = s.replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc).isoformat()


def _resolve_cutoff(
    legacy_days_arg: Optional[str],
    explicit_cutoff: Optional[str],
) -> str:
    """Pick the cutoff string based on which flag the caller passed.

    Returns the cutoff in column-comparison form. Mutually exclusive args
    are enforced by argparse upstream; here we only resolve which mode.
    """
    if explicit_cutoff is not None:
        return _parse_iso_cutoff(explicit_cutoff)
    days = int(legacy_days_arg) if legacy_days_arg is not None else DEFAULT_RETENTION_DAYS
    now = datetime.datetime.now(datetime.timezone.utc)
    return (now - datetime.timedelta(days=days)).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prune old rows from ibkr_exec_state by age cutoff.",
    )
    # Legacy positional: still accepted, still meaning RETENTION_DAYS.
    p.add_argument(
        "retention_days",
        nargs="?",
        default=None,
        help="Legacy: integer days of retention. cutoff = utcnow - N days. "
             f"Default {DEFAULT_RETENTION_DAYS}. Mutually exclusive with --cutoff.",
    )
    p.add_argument(
        "--cutoff",
        dest="cutoff",
        default=None,
        help="Explicit UTC cutoff timestamp (ISO-8601, e.g. 2026-05-19T01:25:06Z). "
             "Rows with updated_at_utc < cutoff are deleted.",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Compute and print what would be deleted, but do not mutate the DB.",
    )
    return p


def run(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cutoff is not None and args.retention_days is not None:
        log.error("REFUSE: pass either positional retention_days OR --cutoff, not both.")
        return 2

    try:
        cutoff = _resolve_cutoff(args.retention_days, args.cutoff)
    except ValueError as exc:
        log.error("REFUSE: invalid --cutoff value: %s", exc)
        return 3

    if not DB.is_file():
        log.error("REFUSE: SQLite DB not found at %s", DB)
        return 4

    con = sqlite3.connect(str(DB))
    try:
        before = con.execute(f"SELECT count(*) FROM {TARGET_TABLE}").fetchone()[0]
        would_delete = con.execute(
            f"SELECT count(*) FROM {TARGET_TABLE} WHERE updated_at_utc < ?",
            (cutoff,),
        ).fetchone()[0]
        would_retain = before - would_delete

        if args.dry_run:
            log.info(
                "DRY_RUN cutoff=%s would_delete=%d would_retain=%d (before=%d)",
                cutoff, would_delete, would_retain, before,
            )
            # List the rows that would be deleted (status + timestamp only — payload omitted to keep audit log compact).
            rows = con.execute(
                f"SELECT idempotency_key, status, updated_at_utc "
                f"FROM {TARGET_TABLE} WHERE updated_at_utc < ? "
                f"ORDER BY updated_at_utc",
                (cutoff,),
            ).fetchall()
            for k, s, u in rows:
                log.info("DRY_RUN_DELETE_CANDIDATE key=%s status=%s updated=%s", k[:16], s, u)
            return 0

        con.execute(
            f"DELETE FROM {TARGET_TABLE} WHERE updated_at_utc < ?",
            (cutoff,),
        )
        con.commit()
        con.execute("VACUUM")
        after = con.execute(f"SELECT count(*) FROM {TARGET_TABLE}").fetchone()[0]
        log.info(
            "%s pruned: %d removed, %d retained (cutoff=%s)",
            TARGET_TABLE, before - after, after, cutoff,
        )
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

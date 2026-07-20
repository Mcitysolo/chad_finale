#!/usr/bin/env python3
"""scripts/reap_ibkr_exec_state.py — status-aware, gated reaper for CHAD's own
IBKR idempotency/dedup ledger.

Target (hard-locked): ``runtime/ibkr_adapter_state.sqlite3`` :: table
``ibkr_exec_state`` — the adapter's persisted duplicate-submission guard
(chad/execution/ibkr_adapter.py). Rows are ``(idempotency_key, status,
created_at_utc, updated_at_utc, broker_order_id, payload_json, result_json)``.
This is CHAD's last-observed snapshot, NOT live broker truth.

Why this exists (P1-8 / GAP-036): the adapter only cleans a key lazily when the
SAME key re-submits (``claim_or_reclaim``). An orphaned non-terminal key that is
never re-submitted lingers forever. The incumbent ``ops/sqlite_retention.py``
does prune, but AGE-ONLY and STATUS-BLIND (it ``DELETE``s ``Filled`` evidence
too), un-gated, real-delete-by-default. This reaper is the opposite: status-aware
(deletes only stale NON-TERMINAL rows, preserves terminal evidence + recent
rows), dry-run by default, and fail-closed gated for ``--execute``.

Modes
-----
* Detection (default, READ-ONLY): classify rows by the adapter's own status
  vocabulary; report delete-candidates + retain set; mutate NOTHING. Also prints
  the INCUMBENT-DIFF: what the age-only reaper WOULD delete vs what status-aware
  logic PRESERVES (the evidence for a future harden-the-incumbent PA).
* Purge (``--execute`` + ``--confirm REAP-IBKR-EXEC-STATE``): fail-closed gates
  (paper exec mode, SCR safe, reconciliation not RED — reused from
  scripts/reconcile_ledger_to_broker.run_gates), archive-before-mutate
  (``.bak_reap_<UTC>`` + sha256), then delete only stale NON-TERMINAL rows.

Safety
------
DRY-RUN default. Hard-targets ``ibkr_adapter_state.sqlite3::ibkr_exec_state`` and
refuses any other DB basename (explicitly the wrong-stores
``exec_state_paper.sqlite3`` — Kraken trusted-lot FIFO — and the dead
``ibkr_exec_state.db``). NEVER run ``--execute`` against real runtime here; that
is an operator step, gated by
ops/pending_actions/W1A_reaper_purge_authorization.md. Performs NO broker I/O:
Wave-1 purge is age-margin only (an ``--broker-probe`` cross-check that keeps
only broker-ABSENT keys is a noted follow-up requiring a live connection).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# F1 — make `chad` and the sibling gate module importable regardless of CWD.
# `python3 scripts/reap_ibkr_exec_state.py` puts scripts/ on sys.path[0], not the
# repo root; `chad` is not pip-installed. Prepend both once at import time.
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the gold-standard fail-closed gates (exec_mode paper, SCR safe,
# reconciliation not RED). Imported (not reimplemented) so the reaper cannot
# drift from the reconcile tool's gate semantics.
from reconcile_ledger_to_broker import run_gates  # noqa: E402

LOG = logging.getLogger("chad.reap_ibkr_exec_state")

CONFIRM_TOKEN = "REAP-IBKR-EXEC-STATE"
MARKER = "IBKR_EXEC_STATE_REAPED"
TARGET_DB_NAME = "ibkr_adapter_state.sqlite3"
TARGET_TABLE = "ibkr_exec_state"
DEFAULT_OLDER_THAN_DAYS = 30  # conservative: > the incumbent's 14d, safely past any working order

# Known wrong-stores that must NEVER be reaped by this script.
_FORBIDDEN_DB_NAMES = {
    "exec_state_paper.sqlite3": "Kraken trusted-lot FIFO + Stage-2 evidence store",
    "ibkr_exec_state.db": "dead 0-byte legacy file",
}

# Mirror of chad/execution/ibkr_adapter.py's idempotency classifier vocabulary
# (ibkr_adapter.py:790). Kept local so the reaper stays broker-import-free; a
# parity test (test_w1a_reap_ibkr_exec_state.py) locks it against the adapter.
_TERMINAL_POSITIVE = frozenset({"filled"})
_TERMINAL_NEGATIVE = frozenset({"cancelled", "apicancelled", "rejected", "inactive", "error"})


def classify(status: Optional[str]) -> str:
    """'terminal_positive' | 'terminal_negative' | 'non_terminal' (adapter parity)."""
    raw = (status or "").strip().lower()
    if raw in _TERMINAL_POSITIVE:
        return "terminal_positive"
    if raw in _TERMINAL_NEGATIVE:
        return "terminal_negative"
    return "non_terminal"


def _is_terminal(status: Optional[str]) -> bool:
    return classify(status) != "non_terminal"


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _stamp(now: _dt.datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def _cutoff_iso(now: _dt.datetime, older_than_days: int) -> str:
    return (now - _dt.timedelta(days=older_than_days)).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_db_path(db_path: Path) -> Tuple[bool, str]:
    """Hard-target guard. Refuse any DB that is not the idempotency store."""
    name = db_path.name
    if name in _FORBIDDEN_DB_NAMES:
        return False, f"REFUSE: {name} is the {_FORBIDDEN_DB_NAMES[name]}, not the idempotency store"
    if name != TARGET_DB_NAME:
        return False, f"REFUSE: DB basename {name!r} != required {TARGET_DB_NAME!r}"
    if not db_path.is_file():
        return False, f"REFUSE: DB not found at {db_path}"
    return True, "ok"


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    """Open the DB strictly read-only (URI mode=ro)."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _require_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TARGET_TABLE,)
    ).fetchone()
    return row is not None


def analyze(db_path: Path, *, older_than_days: int, now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """READ-ONLY classification + incumbent-diff. Opens the DB read-only."""
    now = now or _utcnow()
    cutoff = _cutoff_iso(now, older_than_days)
    conn = _connect_ro(db_path)
    try:
        if not _require_table(conn):
            raise ValueError(f"table {TARGET_TABLE} not present in {db_path.name}")
        rows = conn.execute(
            f"SELECT idempotency_key, status, updated_at_utc FROM {TARGET_TABLE}"
        ).fetchall()
    finally:
        conn.close()

    by_class = {"terminal_positive": 0, "terminal_negative": 0, "non_terminal": 0}
    delete_candidates: List[Dict[str, Any]] = []   # status-aware: stale AND non-terminal
    retain: List[Dict[str, Any]] = []
    incumbent_would_delete: List[Dict[str, Any]] = []  # age-only, status-blind
    preserved_by_status_awareness: List[Dict[str, Any]] = []  # old AND terminal (evidence saved)

    for key, status, updated in rows:
        cls = classify(status)
        by_class[cls] = by_class.get(cls, 0) + 1
        is_old = (updated or "") < cutoff
        rec = {"key": (key or "")[:16], "status": status, "updated_at_utc": updated, "class": cls}

        if is_old:
            incumbent_would_delete.append(rec)  # the incumbent deletes ALL old rows, status-blind
        if is_old and cls == "non_terminal":
            delete_candidates.append(rec)
        else:
            retain.append(rec)
        if is_old and _is_terminal(status):
            preserved_by_status_awareness.append(rec)  # incumbent destroys this evidence; we keep it

    return {
        "schema_version": "reap_ibkr_exec_state.report.v1",
        "db_path": str(db_path),
        "target_table": TARGET_TABLE,
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "older_than_days": older_than_days,
        "cutoff_utc": cutoff,
        "total_rows": len(rows),
        "counts_by_class": by_class,
        "delete_candidate_count": len(delete_candidates),
        "retain_count": len(retain),
        "delete_candidates": delete_candidates,
        "retain": retain,
        # W1A-6 incumbent-diff: the case for the harden-the-incumbent PA.
        "incumbent_age_only_would_delete_count": len(incumbent_would_delete),
        "preserved_by_status_awareness_count": len(preserved_by_status_awareness),
        "preserved_by_status_awareness": preserved_by_status_awareness,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print(f"[reap_ibkr_exec_state] db={report['db_path']} table={report['target_table']}")
    print(f"  older_than_days={report['older_than_days']} cutoff_utc={report['cutoff_utc']}")
    print(f"  total_rows={report['total_rows']} counts_by_class={report['counts_by_class']}")
    print(f"  STATUS-AWARE delete_candidates={report['delete_candidate_count']} "
          f"retain={report['retain_count']}")
    print(f"  INCUMBENT (age-only, status-blind) would_delete="
          f"{report['incumbent_age_only_would_delete_count']}")
    print(f"  => PRESERVED by status-awareness (terminal evidence the incumbent destroys)="
          f"{report['preserved_by_status_awareness_count']}")
    for rec in report["delete_candidates"][:50]:
        print(f"    DELETE_CANDIDATE key={rec['key']} status={rec['status']} "
              f"updated={rec['updated_at_utc']}")
    for rec in report["preserved_by_status_awareness"][:50]:
        print(f"    PRESERVED_EVIDENCE key={rec['key']} status={rec['status']} "
              f"updated={rec['updated_at_utc']} class={rec['class']}")


def purge(
    db_path: Path,
    *,
    older_than_days: int,
    runtime_dir: Path,
    now: Optional[_dt.datetime] = None,
) -> Dict[str, Any]:
    """Gated destructive purge. Assumes caller already validated the token.

    Fail-closed: runs run_gates() and REFUSES (no write) unless all pass.
    Archives the whole DB to ``.bak_reap_<UTC>`` (with sha256) before any DELETE.
    Deletes only stale NON-TERMINAL rows; terminal evidence + recent rows stay.
    """
    now = now or _utcnow()
    ok, reasons = run_gates(runtime_dir)
    if not ok:
        return {"applied": False, "refused": True, "reason": "gates_failed", "gates": reasons}

    report = analyze(db_path, older_than_days=older_than_days, now=now)
    cutoff = report["cutoff_utc"]

    # Archive-before-mutate.
    stamp = _stamp(now)
    bak = db_path.with_name(db_path.name + f".bak_reap_{stamp}")
    shutil.copy2(db_path, bak)
    backup_sha = _sha256_file(bak)

    # Delete ONLY stale non-terminal rows: age-old AND classified non_terminal.
    # Expressed in SQL by excluding the terminal statuses (the adapter vocabulary).
    terminal_all = sorted(_TERMINAL_POSITIVE | _TERMINAL_NEGATIVE)
    placeholders = ",".join("?" for _ in terminal_all)
    conn = sqlite3.connect(str(db_path))
    try:
        before = conn.execute(f"SELECT count(*) FROM {TARGET_TABLE}").fetchone()[0]
        conn.execute(
            f"DELETE FROM {TARGET_TABLE} "
            f"WHERE updated_at_utc < ? AND lower(trim(status)) NOT IN ({placeholders})",
            (cutoff, *terminal_all),
        )
        conn.commit()
        conn.execute("VACUUM")
        after = conn.execute(f"SELECT count(*) FROM {TARGET_TABLE}").fetchone()[0]
    finally:
        conn.close()

    return {
        "applied": True,
        "refused": False,
        "marker": MARKER,
        "gates": reasons,
        "backup": str(bak),
        "backup_sha256": backup_sha,
        "rows_before": before,
        "rows_after": after,
        "rows_deleted": before - after,
        "cutoff_utc": cutoff,
        "preserved_terminal_evidence": report["preserved_by_status_awareness_count"],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Status-aware, gated reaper for ibkr_adapter_state.sqlite3::ibkr_exec_state "
                    "(dry-run default; NEVER --execute against real runtime here).",
    )
    p.add_argument("--db", default=None,
                   help=f"DB path (default runtime/{TARGET_DB_NAME}). Must be the idempotency store.")
    p.add_argument("--runtime-dir", default=None,
                   help="Runtime dir for the fail-closed gates (default repo runtime/).")
    p.add_argument("--older-than-days", type=int, default=DEFAULT_OLDER_THAN_DAYS,
                   help=f"Age threshold for stale non-terminal rows (default {DEFAULT_OLDER_THAN_DAYS}).")
    p.add_argument("--execute", action="store_true",
                   help="Apply the purge (mutates the DB). Default is read-only detection.")
    p.add_argument("--confirm", default="",
                   help=f"Required with --execute: the exact token {CONFIRM_TOKEN}.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)

    if args.older_than_days <= 0:
        LOG.error("REFUSE: --older-than-days must be positive (got %s)", args.older_than_days)
        return 3

    db_path = Path(args.db).resolve() if args.db else (_REPO_ROOT / "runtime" / TARGET_DB_NAME)
    runtime_dir = Path(args.runtime_dir).resolve() if args.runtime_dir else (_REPO_ROOT / "runtime")

    ok, msg = _validate_db_path(db_path)
    if not ok:
        LOG.error(msg)
        return 4

    if not args.execute:
        report = analyze(db_path, older_than_days=args.older_than_days)
        _print_report(report)
        print(json.dumps({k: report[k] for k in (
            "total_rows", "delete_candidate_count", "retain_count",
            "incumbent_age_only_would_delete_count", "preserved_by_status_awareness_count",
        )}, sort_keys=True))
        return 0

    # --execute path: typed confirmation is mandatory.
    if args.confirm != CONFIRM_TOKEN:
        LOG.error("REFUSE: --execute requires --confirm %s (got %r)", CONFIRM_TOKEN, args.confirm)
        return 2

    result = purge(db_path, older_than_days=args.older_than_days, runtime_dir=runtime_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("refused"):
        LOG.error("REFUSE: gates failed: %s", result.get("gates"))
        return 5
    LOG.info("%s deleted=%d retained=%d backup=%s",
             MARKER, result["rows_deleted"], result["rows_after"], result["backup"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

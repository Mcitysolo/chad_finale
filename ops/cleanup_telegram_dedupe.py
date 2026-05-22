#!/usr/bin/env python3
"""
ops/cleanup_telegram_dedupe.py

BOX-042 / GAP-012 / NEW-GAP-049 — bounded cleanup of stale Telegram dedupe state.

Background
----------
``chad/utils/telegram_notify.py`` writes one file per unique dedupe_key:
``runtime/telegram_dedupe_<safe_key>.json`` containing
``{"last_sent_unix": <float>}``. The file is overwritten on every successful
send (atomic ``tmp.replace(path)``) so the number of files equals the
cardinality of dedupe_keys, never the number of messages.

Some callers (``chad/ops/health_monitor.py:311`` for example) construct
dedupe_keys from variable finding titles, which causes the file count to
grow with rule-title cardinality. The active-suppression window is
``TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS`` (default 900s). Files older than that
no longer suppress anything — they are dead state.

Retention rule (BOX-042 policy)
-------------------------------
- An "active" dedupe file = mtime within ``TTL * safety_multiplier`` seconds
  of now. These are NEVER touched.
- A "stale" dedupe file = mtime older than the safety window. These may be
  archived to ``_archive/telegram_dedupe/YYYY/MM/`` (preferred) or deleted
  (with ``--delete-instead-of-archive``).
- Default mode is **dry-run** — files are listed but not touched.
- ``--apply`` is required to perform any filesystem mutation.

Safety invariants
-----------------
1. Only files matching ``runtime/telegram_dedupe_*.json`` are considered.
   No other runtime artifact is ever in scope.
2. The pattern explicitly requires the ``telegram_dedupe_`` prefix and the
   ``.json`` suffix; no other extension or compound suffix is touched.
3. Active dedupe state (mtime within the safety window) is **never** in
   the deletion set regardless of flags.
4. Archive mode preserves the original filename inside a YYYY/MM tree so
   forensic history is retrievable.
5. The tool refuses to operate if the safety multiplier is < 1.0.
6. The tool emits an NDJSON audit log of every move/delete (or, in
   dry-run, every intended move/delete) to
   ``runtime/cleanup_telegram_dedupe.audit.ndjson`` (or a caller-specified
   path; see ``--audit-log``).

Usage
-----

    # Default: dry-run, list what would be archived
    python3 ops/cleanup_telegram_dedupe.py

    # Override TTL window (e.g., 1-hour safety even if env TTL is shorter)
    python3 ops/cleanup_telegram_dedupe.py --ttl-seconds 3600

    # Safety multiplier (default 4): keep anything within TTL*N
    python3 ops/cleanup_telegram_dedupe.py --safety-multiplier 4

    # Actually move files to _archive/telegram_dedupe/YYYY/MM/
    python3 ops/cleanup_telegram_dedupe.py --apply

    # Apply but delete instead of archive (NOT recommended; loses history)
    python3 ops/cleanup_telegram_dedupe.py --apply --delete-instead-of-archive

The script never restarts services, never mutates SQLite, never touches
trading/order state, and never authorizes live execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "runtime"
ARCHIVE_BASE_DIR = REPO_ROOT / "_archive" / "telegram_dedupe"
DEFAULT_AUDIT_LOG = RUNTIME_DIR / "cleanup_telegram_dedupe.audit.ndjson"

DEDUPE_PREFIX = "telegram_dedupe_"
DEDUPE_SUFFIX = ".json"


@dataclass(frozen=True)
class CleanupConfig:
    runtime_dir: Path
    archive_dir: Path
    audit_log: Path
    ttl_seconds: int
    safety_multiplier: float
    apply: bool
    delete_instead_of_archive: bool
    now_unix: float


@dataclass(frozen=True)
class CleanupResult:
    scanned: int
    active: int
    stale: int
    archived: int
    deleted: int
    skipped: int
    safety_window_seconds: float
    targets: tuple  # tuple of (path, mtime, action)


def _is_dedupe_file(path: Path) -> bool:
    name = path.name
    return (
        path.is_file()
        and name.startswith(DEDUPE_PREFIX)
        and name.endswith(DEDUPE_SUFFIX)
        and not name.endswith(".tmp")
        and ".tmp." not in name
    )


def _iter_dedupe_files(runtime_dir: Path) -> Iterable[Path]:
    if not runtime_dir.is_dir():
        return []
    return sorted(p for p in runtime_dir.iterdir() if _is_dedupe_file(p))


def _archive_target(archive_base: Path, src: Path, mtime: float) -> Path:
    t = time.gmtime(mtime)
    sub = f"{t.tm_year:04d}/{t.tm_mon:02d}"
    return archive_base / sub / src.name


def _append_audit(audit_log: Path, record: dict) -> None:
    try:
        audit_log.parent.mkdir(parents=True, exist_ok=True)
        with audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:
        # Never fail cleanup because audit log failed; emit to stderr only.
        print(f"WARN audit_log_write_failed err={exc}", file=sys.stderr)


def run_cleanup(cfg: CleanupConfig) -> CleanupResult:
    if cfg.safety_multiplier < 1.0:
        raise ValueError(
            f"safety_multiplier must be >= 1.0 (got {cfg.safety_multiplier}) — "
            "refusing to operate (would risk deleting active dedupe state)"
        )
    if cfg.ttl_seconds <= 0:
        raise ValueError(
            f"ttl_seconds must be > 0 (got {cfg.ttl_seconds}) — refusing"
        )

    safety_window = float(cfg.ttl_seconds) * float(cfg.safety_multiplier)
    cutoff_unix = float(cfg.now_unix) - safety_window

    scanned = 0
    active = 0
    stale = 0
    archived = 0
    deleted = 0
    skipped = 0
    targets: list = []

    for src in _iter_dedupe_files(cfg.runtime_dir):
        scanned += 1
        try:
            mtime = src.stat().st_mtime
        except FileNotFoundError:
            skipped += 1
            continue

        if mtime >= cutoff_unix:
            # Active — never touch.
            active += 1
            continue

        stale += 1
        action_planned = "delete" if cfg.delete_instead_of_archive else "archive"
        dst = (
            None
            if cfg.delete_instead_of_archive
            else _archive_target(cfg.archive_dir, src, mtime)
        )
        targets.append((str(src), mtime, action_planned))

        if not cfg.apply:
            _append_audit(
                cfg.audit_log,
                {
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": "dry_run",
                    "action": action_planned,
                    "src": str(src),
                    "dst": str(dst) if dst else None,
                    "src_mtime": mtime,
                    "ttl_seconds": cfg.ttl_seconds,
                    "safety_multiplier": cfg.safety_multiplier,
                    "safety_window_seconds": safety_window,
                },
            )
            continue

        try:
            if cfg.delete_instead_of_archive:
                src.unlink()
                deleted += 1
            else:
                assert dst is not None
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                archived += 1
            _append_audit(
                cfg.audit_log,
                {
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": "apply",
                    "action": action_planned,
                    "src": str(src),
                    "dst": str(dst) if dst else None,
                    "src_mtime": mtime,
                    "ttl_seconds": cfg.ttl_seconds,
                    "safety_multiplier": cfg.safety_multiplier,
                    "safety_window_seconds": safety_window,
                },
            )
        except Exception as exc:
            skipped += 1
            _append_audit(
                cfg.audit_log,
                {
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": "apply",
                    "action": "skip_error",
                    "src": str(src),
                    "err": f"{type(exc).__name__}: {exc}",
                },
            )

    return CleanupResult(
        scanned=scanned,
        active=active,
        stale=stale,
        archived=archived,
        deleted=deleted,
        skipped=skipped,
        safety_window_seconds=safety_window,
        targets=tuple(targets),
    )


def _resolve_ttl_default() -> int:
    raw = os.environ.get("TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS", "").strip()
    if not raw:
        return 900
    try:
        return max(1, int(raw))
    except Exception:
        return 900


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Safe cleanup for stale Telegram dedupe sentinels"
    )
    p.add_argument(
        "--runtime-dir",
        type=Path,
        default=RUNTIME_DIR,
        help="Directory containing telegram_dedupe_*.json (default: runtime/)",
    )
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=ARCHIVE_BASE_DIR,
        help="Base archive dir; YYYY/MM is appended (default: _archive/telegram_dedupe/)",
    )
    p.add_argument(
        "--audit-log",
        type=Path,
        default=DEFAULT_AUDIT_LOG,
        help="NDJSON audit log path (default: runtime/cleanup_telegram_dedupe.audit.ndjson)",
    )
    p.add_argument(
        "--ttl-seconds",
        type=int,
        default=_resolve_ttl_default(),
        help="Dedupe TTL window in seconds (default: env TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS or 900)",
    )
    p.add_argument(
        "--safety-multiplier",
        type=float,
        default=4.0,
        help="Files newer than TTL * multiplier are NEVER touched (default: 4)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually move/delete; without this, the tool is dry-run only",
    )
    p.add_argument(
        "--delete-instead-of-archive",
        action="store_true",
        help="Delete stale files outright instead of moving to _archive/",
    )
    p.add_argument(
        "--json-summary",
        action="store_true",
        help="Print a single JSON line at the end (machine-readable)",
    )
    args = p.parse_args(argv)

    cfg = CleanupConfig(
        runtime_dir=args.runtime_dir.resolve(),
        archive_dir=args.archive_dir.resolve(),
        audit_log=args.audit_log.resolve(),
        ttl_seconds=int(args.ttl_seconds),
        safety_multiplier=float(args.safety_multiplier),
        apply=bool(args.apply),
        delete_instead_of_archive=bool(args.delete_instead_of_archive),
        now_unix=time.time(),
    )

    try:
        result = run_cleanup(cfg)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    mode = "APPLY" if cfg.apply else "DRY-RUN"
    summary = {
        "mode": mode,
        "runtime_dir": str(cfg.runtime_dir),
        "archive_dir": str(cfg.archive_dir),
        "ttl_seconds": cfg.ttl_seconds,
        "safety_multiplier": cfg.safety_multiplier,
        "safety_window_seconds": result.safety_window_seconds,
        "scanned": result.scanned,
        "active_preserved": result.active,
        "stale_in_scope": result.stale,
        "archived": result.archived,
        "deleted": result.deleted,
        "skipped": result.skipped,
        "delete_instead_of_archive": cfg.delete_instead_of_archive,
    }

    if args.json_summary:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"[{mode}] telegram_dedupe cleanup")
        print(f"  runtime_dir          : {cfg.runtime_dir}")
        print(f"  archive_dir          : {cfg.archive_dir}")
        print(f"  ttl_seconds          : {cfg.ttl_seconds}")
        print(f"  safety_multiplier    : {cfg.safety_multiplier}")
        print(f"  safety_window_secs   : {result.safety_window_seconds:.0f}")
        print(f"  scanned              : {result.scanned}")
        print(f"  active (preserved)   : {result.active}")
        print(f"  stale (in scope)     : {result.stale}")
        if cfg.apply:
            print(f"  archived             : {result.archived}")
            print(f"  deleted              : {result.deleted}")
            print(f"  skipped (errors)     : {result.skipped}")
        else:
            print(f"  would_archive        : {result.stale - (result.deleted)}")
        print(f"  audit_log            : {cfg.audit_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
ops/cleanup_runtime_artifacts.py

BOX-043 — Unified, safe, dry-run-first cleanup of growing runtime/log artifact
families that are NOT already covered by chad-disk-guard, logrotate, or the
Box 041 / Box 042 / sqlite_retention tools.

Categories handled here
-----------------------
- ``runtime_proofs`` — ``runtime/proofs/*.json``
  Burn-in / rebuild proofs written by chad-burnin-check.timer (every 10 min,
  ~13k+ files). Keep the latest N days (default 30); older proofs are
  archived to ``_archive/runtime_proofs/YYYY/MM/<basename>`` where the
  existing ``_archive/`` 30-day rolling delete in ``chad-disk-guard.sh``
  then provides the long-term bound.
- ``config_snapshots`` — ``data/config_snapshots/snapshot_*.json``
  Per-minute hash snapshots written by mutation_state_publisher
  (~65k+ files). Keep the latest N days (default 30); older snapshots
  archive to ``_archive/config_snapshots/YYYY/MM/``.
- ``backups`` — ``backups/chad_backup_*.tar.gz`` + sidecars
  Nightly tarballs (~150 MB/day). Keep the latest K tarballs (default 14)
  along with their ``.sha256`` and ``.manifest.json`` sidecars; older
  triplets are DELETED (size is too large to archive sensibly).
- ``claude_logs`` — ``logs/claude/calls_*.ndjson``
  Per-day Claude API call logs. Keep the latest N days (default 30);
  older days are gzipped in place (``calls_YYYYMMDD.ndjson.gz``) and the
  uncompressed original is removed.

Out of scope (handled elsewhere or intentionally retained)
----------------------------------------------------------
- ``data/fills``, ``data/trades``, ``data/fees``, ``data/broker_events``,
  ``data/traces`` — audit-critical retention; NEVER pruned here.
- ``runtime/completion_matrix_evidence`` — audit evidence retention;
  NEVER pruned here.
- ``runtime/telegram_dedupe_*.json`` — see ``ops/cleanup_telegram_dedupe.py``.
- ``runtime/dynamic_caps*`` operator backups — see Box 041 policy.
- ``runtime/*.tmp``, ``runtime/*.tmp.*`` — handled by chad-disk-guard.sh.
- ``data/feeds/*``, ``_archive/*`` — handled by chad-disk-guard.sh.
- ``logs/backend-uvicorn.log``, ``logs/polygon_stocks.log``,
  ``data/feeds/polygon_stocks/POLYGON_STOCKS_*.ndjson`` — handled by
  ``/etc/logrotate.d/chad-*``.
- ``runtime/ibkr_adapter_state.sqlite3`` rows — handled by
  ``ops/sqlite_retention.py``.

Safety invariants
-----------------
1. Default mode is **dry-run**. ``--apply`` is required to perform any
   filesystem mutation.
2. Per category, files newer than the category's retention window are
   ALWAYS preserved — no flag overrides this.
3. The tool refuses to operate when any retention parameter is non-positive.
4. Scope is restricted by category-specific glob patterns; no other path
   can be touched, even with hand-edited config.
5. Every action (real or dry-run) is logged as a JSON line to the audit
   log (default ``runtime/cleanup_runtime_artifacts.audit.ndjson``).
6. The audit-critical roots (fills, trades, fees, broker_events, traces,
   completion_matrix_evidence) are not listed in any category and are not
   reachable by the cleanup loop.

Usage
-----

    # Default: dry-run every category, no mutation
    python3 ops/cleanup_runtime_artifacts.py

    # Apply (move/gzip/delete according to per-category rule)
    python3 ops/cleanup_runtime_artifacts.py --apply

    # Restrict to a subset of categories
    python3 ops/cleanup_runtime_artifacts.py --categories runtime_proofs,backups

    # Override retention per category (days for time-based, K for count-based)
    python3 ops/cleanup_runtime_artifacts.py --keep-proof-days 14 --keep-backups 7

The tool never restarts services, never mutates SQLite, never touches
trading state, and never authorizes live execution.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "runtime"
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"
BACKUPS_DIR = REPO_ROOT / "backups"
ARCHIVE_BASE_DIR = REPO_ROOT / "_archive"
DEFAULT_AUDIT_LOG = RUNTIME_DIR / "cleanup_runtime_artifacts.audit.ndjson"

# Strict allowlist: every category is a (name, source_dir, filename_regex,
# retention_kind, retention_value, action) tuple.
#
# retention_kind:
#   "days_old"     — files with mtime older than `retention_value` days
#                    are out-of-window; newer files are active.
#   "keep_latest"  — keep the latest `retention_value` files by mtime;
#                    older files are out-of-window.
#
# action:
#   "archive"  — move to _archive/<category>/YYYY/MM/<basename>
#   "delete"   — unlink in place
#   "gzip"     — gzip in place and unlink the original


@dataclass(frozen=True)
class Category:
    name: str
    source_dir: Path
    filename_re: re.Pattern
    retention_kind: str  # "days_old" or "keep_latest"
    retention_value: int
    action: str  # "archive" | "delete" | "gzip"
    sidecar_suffixes: tuple = ()  # extra suffixes that travel with the primary


# Audit-critical roots that must NEVER appear as a source_dir.
_FORBIDDEN_ROOTS = (
    DATA_DIR / "fills",
    DATA_DIR / "trades",
    DATA_DIR / "fees",
    DATA_DIR / "broker_events",
    DATA_DIR / "traces",
    RUNTIME_DIR / "completion_matrix_evidence",
)


def default_categories(
    *,
    keep_proof_days: int,
    keep_snapshot_days: int,
    keep_backups: int,
    keep_claude_log_days: int,
) -> list[Category]:
    cats = [
        Category(
            name="runtime_proofs",
            source_dir=RUNTIME_DIR / "proofs",
            filename_re=re.compile(r"^[A-Z][A-Z0-9_]+_\d{8}T\d{6}Z.*\.(json|txt)$"),
            retention_kind="days_old",
            retention_value=int(keep_proof_days),
            action="archive",
        ),
        Category(
            name="config_snapshots",
            source_dir=DATA_DIR / "config_snapshots",
            filename_re=re.compile(r"^snapshot_\d{8}T\d{6}Z\.json$"),
            retention_kind="days_old",
            retention_value=int(keep_snapshot_days),
            action="archive",
        ),
        Category(
            name="backups",
            source_dir=BACKUPS_DIR,
            filename_re=re.compile(r"^chad_backup_.+_\d{8}T\d{6}Z\.tar\.gz$"),
            retention_kind="keep_latest",
            retention_value=int(keep_backups),
            action="delete",
            sidecar_suffixes=(".sha256", ".manifest.json"),
        ),
        Category(
            name="claude_logs",
            source_dir=LOGS_DIR / "claude",
            filename_re=re.compile(r"^calls_\d{8}\.ndjson$"),
            retention_kind="days_old",
            retention_value=int(keep_claude_log_days),
            action="gzip",
        ),
    ]
    for c in cats:
        for forbidden in _FORBIDDEN_ROOTS:
            try:
                c.source_dir.resolve().relative_to(forbidden.resolve())
                raise RuntimeError(
                    f"Category {c.name!r} source_dir resolves inside "
                    f"audit-critical root {forbidden} — refusing"
                )
            except ValueError:
                pass
        if c.retention_value <= 0:
            raise ValueError(
                f"Category {c.name!r} has non-positive retention "
                f"({c.retention_value}); refusing"
            )
    return cats


@dataclass
class CategoryResult:
    name: str
    scanned: int = 0
    active: int = 0
    stale: int = 0
    archived: int = 0
    deleted: int = 0
    gzipped: int = 0
    skipped: int = 0
    targets: list = field(default_factory=list)


@dataclass
class CleanupConfig:
    categories: list
    audit_log: Path
    archive_base: Path
    apply: bool
    now_unix: float


def _archive_target(archive_base: Path, category: str, src: Path, mtime: float) -> Path:
    t = time.gmtime(mtime)
    sub = f"{t.tm_year:04d}/{t.tm_mon:02d}"
    return archive_base / category / sub / src.name


def _append_audit(audit_log: Path, record: dict) -> None:
    try:
        audit_log.parent.mkdir(parents=True, exist_ok=True)
        with audit_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception as exc:
        print(f"WARN audit_log_write_failed err={exc}", file=sys.stderr)


def _gzip_in_place(src: Path) -> Path:
    """Compress src to src.with_suffix('.ndjson.gz') and unlink src.

    Returns the gz path. Atomic via .tmp + os.replace.
    """
    dst = src.with_suffix(src.suffix + ".gz")
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    try:
        with src.open("rb") as f_in, gzip.open(tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.replace(tmp, dst)
        src.unlink()
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise
    return dst


def _classify(category: Category, files: list[Path], now_unix: float) -> tuple[list, list]:
    """Return (active_files, stale_files) for a category."""
    if category.retention_kind == "days_old":
        cutoff = float(now_unix) - float(category.retention_value) * 86400.0
        active = []
        stale = []
        for p in files:
            try:
                mtime = p.stat().st_mtime
            except FileNotFoundError:
                continue
            if mtime >= cutoff:
                active.append((p, mtime))
            else:
                stale.append((p, mtime))
        return active, stale
    if category.retention_kind == "keep_latest":
        decorated = []
        for p in files:
            try:
                mtime = p.stat().st_mtime
            except FileNotFoundError:
                continue
            decorated.append((p, mtime))
        decorated.sort(key=lambda x: x[1], reverse=True)
        active = decorated[: category.retention_value]
        stale = decorated[category.retention_value :]
        return active, stale
    raise ValueError(f"Unknown retention_kind: {category.retention_kind}")


def _collect_category_files(category: Category) -> list[Path]:
    if not category.source_dir.is_dir():
        return []
    out = []
    for p in category.source_dir.iterdir():
        if not p.is_file():
            continue
        if category.filename_re.match(p.name):
            out.append(p)
    return sorted(out)


def run_one_category(
    category: Category,
    *,
    archive_base: Path,
    audit_log: Path,
    apply: bool,
    now_unix: float,
) -> CategoryResult:
    res = CategoryResult(name=category.name)
    primary = _collect_category_files(category)
    res.scanned = len(primary)
    active, stale = _classify(category, primary, now_unix)
    res.active = len(active)
    res.stale = len(stale)

    for src, mtime in stale:
        action_planned = category.action
        if category.action == "archive":
            dst = _archive_target(archive_base, category.name, src, mtime)
        else:
            dst = None
        res.targets.append((str(src), mtime, action_planned))

        # Sidecars travel with the primary file.
        siblings: list[Path] = []
        for suf in category.sidecar_suffixes:
            sib = src.with_name(src.name + suf)
            if sib.exists():
                siblings.append(sib)

        record_common = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "category": category.name,
            "src": str(src),
            "dst": str(dst) if dst else None,
            "src_mtime": mtime,
            "action": action_planned,
            "siblings": [str(s) for s in siblings],
        }

        if not apply:
            _append_audit(audit_log, {**record_common, "mode": "dry_run"})
            continue

        try:
            if category.action == "archive":
                assert dst is not None
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                for sib in siblings:
                    sib_dst = dst.parent / sib.name
                    shutil.move(str(sib), str(sib_dst))
                res.archived += 1
            elif category.action == "delete":
                src.unlink()
                for sib in siblings:
                    try:
                        sib.unlink()
                    except FileNotFoundError:
                        pass
                res.deleted += 1
            elif category.action == "gzip":
                _gzip_in_place(src)
                res.gzipped += 1
            else:
                res.skipped += 1
                continue
            _append_audit(audit_log, {**record_common, "mode": "apply"})
        except Exception as exc:
            res.skipped += 1
            _append_audit(
                audit_log,
                {
                    **record_common,
                    "mode": "apply",
                    "action": "skip_error",
                    "err": f"{type(exc).__name__}: {exc}",
                },
            )

    return res


def run_cleanup(cfg: CleanupConfig) -> list[CategoryResult]:
    return [
        run_one_category(
            c,
            archive_base=cfg.archive_base,
            audit_log=cfg.audit_log,
            apply=cfg.apply,
            now_unix=cfg.now_unix,
        )
        for c in cfg.categories
    ]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified safe cleanup for CHAD runtime artifacts")
    p.add_argument("--apply", action="store_true", help="Actually mutate; default is dry-run")
    p.add_argument("--keep-proof-days", type=int, default=30)
    p.add_argument("--keep-snapshot-days", type=int, default=30)
    p.add_argument("--keep-backups", type=int, default=14)
    p.add_argument("--keep-claude-log-days", type=int, default=30)
    p.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma-separated subset (default: all)",
    )
    p.add_argument("--audit-log", type=Path, default=DEFAULT_AUDIT_LOG)
    p.add_argument("--archive-base", type=Path, default=ARCHIVE_BASE_DIR)
    p.add_argument("--json-summary", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        all_cats = default_categories(
            keep_proof_days=args.keep_proof_days,
            keep_snapshot_days=args.keep_snapshot_days,
            keep_backups=args.keep_backups,
            keep_claude_log_days=args.keep_claude_log_days,
        )
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.categories.strip():
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        unknown = wanted - {c.name for c in all_cats}
        if unknown:
            print(f"FAILED: unknown categories: {sorted(unknown)}", file=sys.stderr)
            return 2
        all_cats = [c for c in all_cats if c.name in wanted]

    cfg = CleanupConfig(
        categories=all_cats,
        audit_log=args.audit_log.resolve(),
        archive_base=args.archive_base.resolve(),
        apply=bool(args.apply),
        now_unix=time.time(),
    )

    results = run_cleanup(cfg)
    mode = "APPLY" if cfg.apply else "DRY-RUN"
    summary = {
        "mode": mode,
        "apply": cfg.apply,
        "archive_base": str(cfg.archive_base),
        "audit_log": str(cfg.audit_log),
        "now_unix": cfg.now_unix,
        "categories": [],
    }
    for r in results:
        summary["categories"].append(
            {
                "name": r.name,
                "scanned": r.scanned,
                "active_preserved": r.active,
                "stale_in_scope": r.stale,
                "archived": r.archived,
                "deleted": r.deleted,
                "gzipped": r.gzipped,
                "skipped": r.skipped,
            }
        )

    if args.json_summary:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"[{mode}] cleanup_runtime_artifacts")
        for entry in summary["categories"]:
            print(
                f"  {entry['name']:18s} "
                f"scanned={entry['scanned']:6d} "
                f"active={entry['active_preserved']:6d} "
                f"stale={entry['stale_in_scope']:6d} "
                f"archived={entry['archived']:6d} "
                f"deleted={entry['deleted']:6d} "
                f"gzipped={entry['gzipped']:6d} "
                f"skipped={entry['skipped']:6d}"
            )
        print(f"  audit_log: {cfg.audit_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

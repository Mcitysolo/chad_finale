"""
BOX-043 — tests for ops/cleanup_runtime_artifacts.py.

These tests run entirely in tmp directories. They prove:

1. Each category preserves files within its retention window.
2. ``runtime_proofs`` archives stale BURNIN/RUNTIME_REBUILD files to
   ``_archive/runtime_proofs/YYYY/MM/<basename>``.
3. ``config_snapshots`` archives stale snapshot_*.json files.
4. ``backups`` keeps the latest K tarballs and deletes older ones along
   with their .sha256 and .manifest.json sidecars.
5. ``claude_logs`` gzips old per-day NDJSON logs in place.
6. Dry-run performs NO filesystem mutation, but DOES log intended actions
   to the NDJSON audit log.
7. Files not matching the per-category regex are NOT touched.
8. Audit-critical roots (fills, trades, fees, broker_events, traces,
   completion_matrix_evidence) are explicitly forbidden as source dirs.
9. Non-positive retention values are refused.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from pathlib import Path

import pytest

from ops.cleanup_runtime_artifacts import (
    Category,
    CategoryResult,
    CleanupConfig,
    _FORBIDDEN_ROOTS,
    default_categories,
    run_cleanup,
    run_one_category,
)


def _touch(path: Path, *, mtime: float, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "{}", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _mk_runtime_proofs_category(src: Path, *, days: int = 30) -> Category:
    return Category(
        name="runtime_proofs",
        source_dir=src,
        filename_re=re.compile(r"^[A-Z][A-Z0-9_]+_\d{8}T\d{6}Z.*\.(json|txt)$"),
        retention_kind="days_old",
        retention_value=days,
        action="archive",
    )


def _mk_config_snapshots_category(src: Path, *, days: int = 30) -> Category:
    return Category(
        name="config_snapshots",
        source_dir=src,
        filename_re=re.compile(r"^snapshot_\d{8}T\d{6}Z\.json$"),
        retention_kind="days_old",
        retention_value=days,
        action="archive",
    )


def _mk_backups_category(src: Path, *, k: int = 14) -> Category:
    return Category(
        name="backups",
        source_dir=src,
        filename_re=re.compile(r"^chad_backup_.+_\d{8}T\d{6}Z\.tar\.gz$"),
        retention_kind="keep_latest",
        retention_value=k,
        action="delete",
        sidecar_suffixes=(".sha256", ".manifest.json"),
    )


def _mk_claude_logs_category(src: Path, *, days: int = 30) -> Category:
    return Category(
        name="claude_logs",
        source_dir=src,
        filename_re=re.compile(r"^calls_\d{8}\.ndjson$"),
        retention_kind="days_old",
        retention_value=days,
        action="gzip",
    )


def _run(cat: Category, *, tmp_path: Path, apply: bool) -> CategoryResult:
    audit = tmp_path / "audit.ndjson"
    archive = tmp_path / "_archive"
    return run_one_category(
        cat,
        archive_base=archive,
        audit_log=audit,
        apply=apply,
        now_unix=time.time(),
    )


# ---------------------------------------------------------------------------
# Category: runtime_proofs
# ---------------------------------------------------------------------------


def test_runtime_proofs_active_files_preserved_dry_run(tmp_path: Path) -> None:
    src = tmp_path / "runtime_proofs"
    now = time.time()
    fresh = src / "BURNIN_CHECK_20260520T113244Z.json"
    _touch(fresh, mtime=now - 60.0)
    cat = _mk_runtime_proofs_category(src, days=30)

    res = _run(cat, tmp_path=tmp_path, apply=False)

    assert fresh.exists()
    assert res.scanned == 1
    assert res.active == 1
    assert res.stale == 0


def test_runtime_proofs_stale_files_archived_to_year_month(tmp_path: Path) -> None:
    src = tmp_path / "runtime_proofs"
    now = time.time()
    stale_mtime = now - 60 * 86400.0  # 60 days ago
    stale = src / "BURNIN_CHECK_20260320T113244Z.json"
    _touch(stale, mtime=stale_mtime)
    cat = _mk_runtime_proofs_category(src, days=30)

    res = _run(cat, tmp_path=tmp_path, apply=True)

    assert not stale.exists()
    t = time.gmtime(stale_mtime)
    expected = (
        tmp_path
        / "_archive"
        / "runtime_proofs"
        / f"{t.tm_year:04d}"
        / f"{t.tm_mon:02d}"
        / stale.name
    )
    assert expected.exists()
    assert res.archived == 1
    assert res.active == 0


def test_runtime_proofs_unrelated_files_not_touched(tmp_path: Path) -> None:
    src = tmp_path / "runtime_proofs"
    now = time.time()
    stale_mtime = now - 60 * 86400.0
    stale = src / "BURNIN_CHECK_20260320T113244Z.json"
    _touch(stale, mtime=stale_mtime)
    # Wrong filename shape — must be left alone.
    not_a_proof = src / "manifest.json"
    _touch(not_a_proof, mtime=stale_mtime)
    other = src / "LATEST_PAPER_CYCLE_DIR.txt"  # not matching the timestamped regex
    _touch(other, mtime=stale_mtime)

    cat = _mk_runtime_proofs_category(src, days=30)
    res = _run(cat, tmp_path=tmp_path, apply=True)

    assert not stale.exists()
    assert not_a_proof.exists()
    assert other.exists()
    assert res.scanned == 1
    assert res.archived == 1


# ---------------------------------------------------------------------------
# Category: config_snapshots
# ---------------------------------------------------------------------------


def test_config_snapshots_active_preserved(tmp_path: Path) -> None:
    src = tmp_path / "config_snapshots"
    now = time.time()
    fresh = src / "snapshot_20260520T113325Z.json"
    _touch(fresh, mtime=now - 600)
    cat = _mk_config_snapshots_category(src, days=30)
    res = _run(cat, tmp_path=tmp_path, apply=True)
    assert fresh.exists()
    assert res.active == 1
    assert res.archived == 0


def test_config_snapshots_stale_archived(tmp_path: Path) -> None:
    src = tmp_path / "config_snapshots"
    now = time.time()
    stale_mtime = now - 45 * 86400.0
    stale = src / "snapshot_20260403T022958Z.json"
    _touch(stale, mtime=stale_mtime)
    cat = _mk_config_snapshots_category(src, days=30)
    res = _run(cat, tmp_path=tmp_path, apply=True)
    assert not stale.exists()
    t = time.gmtime(stale_mtime)
    expected = (
        tmp_path
        / "_archive"
        / "config_snapshots"
        / f"{t.tm_year:04d}"
        / f"{t.tm_mon:02d}"
        / stale.name
    )
    assert expected.exists()
    assert res.archived == 1


# ---------------------------------------------------------------------------
# Category: backups (keep_latest with sidecars)
# ---------------------------------------------------------------------------


def test_backups_keep_latest_k_with_sidecars(tmp_path: Path) -> None:
    src = tmp_path / "backups"
    src.mkdir(parents=True)
    now = time.time()
    # Build 5 daily backups (oldest first), each with .sha256 + .manifest.json
    days = [
        ("20260501", now - 19 * 86400.0),
        ("20260502", now - 18 * 86400.0),
        ("20260503", now - 17 * 86400.0),
        ("20260504", now - 16 * 86400.0),
        ("20260505", now - 15 * 86400.0),
    ]
    for d, t in days:
        primary = src / f"chad_backup_test_{d}T033000Z.tar.gz"
        _touch(primary, mtime=t, content="x")
        _touch(src / f"chad_backup_test_{d}T033000Z.tar.gz.sha256", mtime=t)
        _touch(src / f"chad_backup_test_{d}T033000Z.tar.gz.manifest.json", mtime=t)

    cat = _mk_backups_category(src, k=2)
    res = _run(cat, tmp_path=tmp_path, apply=True)

    surviving_primaries = sorted(p.name for p in src.glob("chad_backup_*.tar.gz"))
    assert surviving_primaries == [
        "chad_backup_test_20260504T033000Z.tar.gz",
        "chad_backup_test_20260505T033000Z.tar.gz",
    ]
    # Older primaries deleted along with their sidecars.
    assert not (src / "chad_backup_test_20260501T033000Z.tar.gz").exists()
    assert not (src / "chad_backup_test_20260501T033000Z.tar.gz.sha256").exists()
    assert not (src / "chad_backup_test_20260501T033000Z.tar.gz.manifest.json").exists()
    # Surviving primaries still have their sidecars.
    assert (src / "chad_backup_test_20260505T033000Z.tar.gz.sha256").exists()
    assert (src / "chad_backup_test_20260505T033000Z.tar.gz.manifest.json").exists()

    assert res.scanned == 5
    assert res.active == 2
    assert res.stale == 3
    assert res.deleted == 3


def test_backups_dry_run_preserves_everything(tmp_path: Path) -> None:
    src = tmp_path / "backups"
    src.mkdir(parents=True)
    now = time.time()
    paths = []
    for i, age in enumerate([10, 20, 30]):
        p = src / f"chad_backup_test_2026010{i+1}T033000Z.tar.gz"
        _touch(p, mtime=now - age * 86400.0)
        paths.append(p)
    cat = _mk_backups_category(src, k=1)
    res = _run(cat, tmp_path=tmp_path, apply=False)

    for p in paths:
        assert p.exists()  # nothing deleted in dry-run
    assert res.scanned == 3
    assert res.deleted == 0
    assert res.stale == 2
    # Audit log received 2 dry_run entries for the 2 stale candidates.
    audit_lines = [
        json.loads(line)
        for line in (tmp_path / "audit.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(audit_lines) == 2
    assert all(ln["mode"] == "dry_run" for ln in audit_lines)
    assert all(ln["category"] == "backups" for ln in audit_lines)


# ---------------------------------------------------------------------------
# Category: claude_logs (gzip in place)
# ---------------------------------------------------------------------------


def test_claude_logs_old_files_gzipped_in_place(tmp_path: Path) -> None:
    src = tmp_path / "logs_claude"
    now = time.time()
    fresh = src / "calls_20260519.ndjson"
    _touch(fresh, mtime=now - 86400.0, content='{"x":1}\n')
    stale = src / "calls_20260415.ndjson"
    _touch(stale, mtime=now - 35 * 86400.0, content='{"y":2}\n')
    # Already compressed: must not be touched (does not match regex).
    already_gz = src / "calls_20260301.ndjson.gz"
    _touch(already_gz, mtime=now - 60 * 86400.0)

    cat = _mk_claude_logs_category(src, days=30)
    res = _run(cat, tmp_path=tmp_path, apply=True)

    assert fresh.exists()
    assert not stale.exists()
    assert (src / "calls_20260415.ndjson.gz").exists()
    assert already_gz.exists()
    # Verify content was actually compressed and matches the original.
    with gzip.open(src / "calls_20260415.ndjson.gz", "rb") as f:
        assert f.read() == b'{"y":2}\n'
    assert res.gzipped == 1


# ---------------------------------------------------------------------------
# Safety: forbidden roots and bad config
# ---------------------------------------------------------------------------


def test_default_categories_never_point_at_audit_critical_roots() -> None:
    cats = default_categories(
        keep_proof_days=30,
        keep_snapshot_days=30,
        keep_backups=14,
        keep_claude_log_days=30,
    )
    for c in cats:
        for forbidden in _FORBIDDEN_ROOTS:
            assert not str(c.source_dir).startswith(str(forbidden))


def test_non_positive_retention_is_refused() -> None:
    with pytest.raises(ValueError, match="retention"):
        default_categories(
            keep_proof_days=0,
            keep_snapshot_days=30,
            keep_backups=14,
            keep_claude_log_days=30,
        )


def test_empty_source_dir_is_handled(tmp_path: Path) -> None:
    cat = _mk_runtime_proofs_category(tmp_path / "missing", days=30)
    res = _run(cat, tmp_path=tmp_path, apply=True)
    assert res.scanned == 0
    assert res.archived == 0
    assert res.deleted == 0


# ---------------------------------------------------------------------------
# End-to-end: full config runs cleanly with empty fixture
# ---------------------------------------------------------------------------


def test_run_cleanup_end_to_end_empty(tmp_path: Path) -> None:
    cats = [
        _mk_runtime_proofs_category(tmp_path / "p", days=30),
        _mk_config_snapshots_category(tmp_path / "s", days=30),
        _mk_backups_category(tmp_path / "b", k=14),
        _mk_claude_logs_category(tmp_path / "c", days=30),
    ]
    cfg = CleanupConfig(
        categories=cats,
        audit_log=tmp_path / "audit.ndjson",
        archive_base=tmp_path / "_archive",
        apply=True,
        now_unix=time.time(),
    )
    results = run_cleanup(cfg)
    assert [r.name for r in results] == [
        "runtime_proofs",
        "config_snapshots",
        "backups",
        "claude_logs",
    ]
    assert all(r.scanned == 0 for r in results)

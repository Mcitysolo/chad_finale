"""
BOX-042 / GAP-012 / NEW-GAP-049 — tests for ops/cleanup_telegram_dedupe.py.

These tests prove the four safety invariants required for box closure:

1. Active dedupe state (mtime within TTL * safety_multiplier of now) is never
   selected for archive/delete.
2. Stale dedupe state (mtime older than safety window) is selected and
   archived to ``_archive/telegram_dedupe/YYYY/MM/`` by default.
3. Dry-run performs NO filesystem mutation but DOES log intended targets to
   the NDJSON audit log.
4. ``--apply`` only touches files explicitly classified as stale; unrelated
   files in the same directory are not touched.
5. Refuses to run with ``safety_multiplier < 1.0``.

The tests run entirely in tmp directories — no real ``runtime/`` artifact
is ever touched.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ops.cleanup_telegram_dedupe import CleanupConfig, run_cleanup


def _touch_dedupe(path: Path, mtime: float) -> None:
    path.write_text(json.dumps({"last_sent_unix": mtime}) + "\n", encoding="utf-8")
    import os

    os.utime(path, (mtime, mtime))


def _mk_cfg(
    tmp_path: Path,
    *,
    ttl_seconds: int = 900,
    safety_multiplier: float = 4.0,
    apply: bool = False,
    delete_instead_of_archive: bool = False,
    now: float | None = None,
) -> CleanupConfig:
    rt = tmp_path / "runtime"
    rt.mkdir(parents=True, exist_ok=True)
    arc = tmp_path / "_archive" / "telegram_dedupe"
    return CleanupConfig(
        runtime_dir=rt,
        archive_dir=arc,
        audit_log=tmp_path / "audit.ndjson",
        ttl_seconds=ttl_seconds,
        safety_multiplier=safety_multiplier,
        apply=apply,
        delete_instead_of_archive=delete_instead_of_archive,
        now_unix=time.time() if now is None else now,
    )


def test_active_dedupe_state_is_never_touched_dry_run(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path)
    now = cfg.now_unix
    # Just-written file (mtime = now). Must always be preserved.
    active = cfg.runtime_dir / "telegram_dedupe_stop_bus_triggered.json"
    _touch_dedupe(active, now)

    result = run_cleanup(cfg)

    assert active.exists(), "Active file removed in dry-run — invariant 1 violated"
    assert result.active == 1
    assert result.stale == 0
    assert result.archived == 0
    assert result.deleted == 0


def test_active_dedupe_state_is_never_touched_apply(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, apply=True)
    now = cfg.now_unix
    # Within the TTL window — actively suppressing.
    just_sent = cfg.runtime_dir / "telegram_dedupe_ibkr_down.json"
    _touch_dedupe(just_sent, now - 60.0)
    # Within TTL*safety_multiplier — still preserved per policy.
    recent = cfg.runtime_dir / "telegram_dedupe_scr_milestone_CONFIDENT.json"
    _touch_dedupe(recent, now - (cfg.ttl_seconds * cfg.safety_multiplier - 60))

    result = run_cleanup(cfg)

    assert just_sent.exists(), "TTL-window file removed — invariant 1 violated"
    assert recent.exists(), "Safety-window file removed — invariant 1 violated"
    assert result.active == 2
    assert result.archived == 0
    assert result.deleted == 0


def test_stale_files_are_archived_to_year_month_tree(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, apply=True)
    now = cfg.now_unix
    # Older than safety window (TTL*safety_multiplier seconds ago).
    old_mtime = now - (cfg.ttl_seconds * cfg.safety_multiplier + 86400.0)
    stale = cfg.runtime_dir / "telegram_dedupe_health_R13_SCRgap999.json"
    _touch_dedupe(stale, old_mtime)

    result = run_cleanup(cfg)

    assert not stale.exists(), "Stale file should have been moved"
    # Archive path uses YYYY/MM from the file's mtime.
    t = time.gmtime(old_mtime)
    expected_dst = (
        cfg.archive_dir
        / f"{t.tm_year:04d}"
        / f"{t.tm_mon:02d}"
        / stale.name
    )
    assert expected_dst.exists(), f"Expected archive at {expected_dst}, not found"
    assert result.stale == 1
    assert result.archived == 1
    assert result.deleted == 0
    assert result.active == 0


def test_dry_run_performs_no_mutation_but_logs_targets(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, apply=False)
    now = cfg.now_unix
    stale1 = cfg.runtime_dir / "telegram_dedupe_health_oldA.json"
    stale2 = cfg.runtime_dir / "telegram_dedupe_health_oldB.json"
    active = cfg.runtime_dir / "telegram_dedupe_now.json"
    _touch_dedupe(stale1, now - (cfg.ttl_seconds * cfg.safety_multiplier + 100))
    _touch_dedupe(stale2, now - (cfg.ttl_seconds * cfg.safety_multiplier + 200))
    _touch_dedupe(active, now)

    result = run_cleanup(cfg)

    # Nothing was moved or deleted.
    assert stale1.exists()
    assert stale2.exists()
    assert active.exists()
    assert result.archived == 0
    assert result.deleted == 0
    assert result.stale == 2
    assert result.active == 1
    # But the dry-run audit log lists exactly those two as intended targets.
    lines = [
        json.loads(line)
        for line in cfg.audit_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    paths = {ln["src"] for ln in lines}
    assert str(stale1) in paths
    assert str(stale2) in paths
    for ln in lines:
        assert ln["mode"] == "dry_run"
        assert ln["action"] == "archive"


def test_apply_only_touches_telegram_dedupe_files(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, apply=True)
    now = cfg.now_unix
    old_mtime = now - (cfg.ttl_seconds * cfg.safety_multiplier + 600)
    stale = cfg.runtime_dir / "telegram_dedupe_obsolete.json"
    _touch_dedupe(stale, old_mtime)
    # Unrelated runtime files that share the runtime dir — must NOT be touched.
    sibling_json = cfg.runtime_dir / "live_readiness.json"
    sibling_json.write_text("{}", encoding="utf-8")
    sibling_other = cfg.runtime_dir / "telegram_dedupe_obsolete.json.tmp"
    sibling_other.write_text("{}", encoding="utf-8")  # ".tmp" excluded by pattern
    sibling_misc = cfg.runtime_dir / "stop_bus.json"
    sibling_misc.write_text("{}", encoding="utf-8")

    result = run_cleanup(cfg)

    assert not stale.exists(), "Stale dedupe file should have been archived"
    assert sibling_json.exists(), "Unrelated runtime JSON must not be touched"
    assert sibling_other.exists(), ".tmp sidecar must not be in scope"
    assert sibling_misc.exists(), "Unrelated state file must not be touched"
    assert result.scanned == 1
    assert result.archived == 1


def test_delete_mode_unlinks_instead_of_archiving(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, apply=True, delete_instead_of_archive=True)
    now = cfg.now_unix
    stale = cfg.runtime_dir / "telegram_dedupe_gone.json"
    _touch_dedupe(stale, now - (cfg.ttl_seconds * cfg.safety_multiplier + 10))

    result = run_cleanup(cfg)

    assert not stale.exists(), "Stale file should have been deleted"
    assert not cfg.archive_dir.exists(), "Archive tree must not be created in delete mode"
    assert result.deleted == 1
    assert result.archived == 0


def test_safety_multiplier_below_one_is_refused(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, safety_multiplier=0.5)
    with pytest.raises(ValueError, match="safety_multiplier"):
        run_cleanup(cfg)


def test_zero_or_negative_ttl_is_refused(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl_seconds"):
        run_cleanup(cfg)


def test_empty_runtime_dir_is_handled(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path)
    result = run_cleanup(cfg)
    assert result.scanned == 0
    assert result.active == 0
    assert result.stale == 0
    assert result.archived == 0

"""GAP-032 preventive lint guard — unit tests.

Covers:
  (i)   all-symlinks fixture → ok=true, exit 0
  (ii)  planted regular file (chad-*) → ok=false, exit 2, entry reported
  (iii) monotonic-only-no-anchor synthetic case → warning emitted, exit 0
  (iv)  atomic-write of the lint artifact
  (v)   idempotent rerun

No real systemctl is invoked. All systemctl calls are mocked via the
runner-injection points on `_detect_monotonic_only`.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import List, Sequence

import pytest

from chad.ops import systemd_wants_lint as lint


def _make_wants_root(tmp_path: Path) -> Path:
    root = tmp_path / "systemd_root"
    root.mkdir()
    return root


def _make_wants_dir(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir()
    return d


def _link_unit(wants_dir: Path, link_name: str, target_dir: Path) -> Path:
    target = target_dir / link_name
    target.write_text("[Unit]\nDescription=fixture\n", encoding="utf-8")
    link = wants_dir / link_name
    os.symlink(target, link)
    return link


def _plant_regular(wants_dir: Path, name: str, body: str = "[Unit]\nDescription=corrupted\n") -> Path:
    p = wants_dir / name
    p.write_text(body, encoding="utf-8")
    return p


# -----------------------------------------------------------------------------
# (i) all-symlinks
# -----------------------------------------------------------------------------

def test_all_symlinks_ok_exit_zero(tmp_path: Path) -> None:
    root = _make_wants_root(tmp_path)
    wants = _make_wants_dir(root, "timers.target.wants")
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    _link_unit(wants, "chad-foo.service", units_dir)
    _link_unit(wants, "chad-bar.timer", units_dir)

    result = lint.scan(wants_root=root, enable_monotonic_check=False)
    assert result["ok"] is True
    assert result["regular_file_count"] == 0
    assert result["chad_regular_file_count"] == 0
    assert result["entries"] == []

    out = tmp_path / "lint.json"
    rc = lint.main(["--wants-root", str(root), "--output", str(out),
                    "--no-monotonic-check", "--quiet"])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["schema_version"] == lint.SCHEMA_VERSION


# -----------------------------------------------------------------------------
# (ii) planted regular file (chad-* flips exit 2; OS-* does not)
# -----------------------------------------------------------------------------

def test_chad_regular_file_flips_exit_two(tmp_path: Path) -> None:
    root = _make_wants_root(tmp_path)
    wants = _make_wants_dir(root, "timers.target.wants")
    _plant_regular(wants, "chad-corrupted.timer")

    result = lint.scan(wants_root=root, enable_monotonic_check=False)
    assert result["ok"] is False
    assert result["chad_regular_file_count"] == 1
    assert result["entries"][0]["kind"] == "chad"
    assert result["entries"][0]["parent_target"] == "timers.target.wants"

    out = tmp_path / "lint.json"
    rc = lint.main(["--wants-root", str(root), "--output", str(out),
                    "--no-monotonic-check", "--quiet"])
    assert rc == 2


def test_os_only_regular_file_stays_exit_zero(tmp_path: Path) -> None:
    root = _make_wants_root(tmp_path)
    wants = _make_wants_dir(root, "timers.target.wants")
    _plant_regular(wants, "apt-daily.timer")  # OS scope

    result = lint.scan(wants_root=root, enable_monotonic_check=False)
    # ok is keyed to chad scope; OS regulars are informational.
    assert result["ok"] is True
    assert result["chad_regular_file_count"] == 0
    assert result["os_regular_file_count"] == 1
    assert result["entries"][0]["kind"] == "os"

    rc = lint.main(["--wants-root", str(root),
                    "--output", str(tmp_path / "x.json"),
                    "--no-monotonic-check", "--quiet"])
    assert rc == 0


# -----------------------------------------------------------------------------
# (iii) monotonic-only-no-anchor synthetic case
# -----------------------------------------------------------------------------

def test_monotonic_only_warning_emitted(tmp_path: Path) -> None:
    root = _make_wants_root(tmp_path)
    wants = _make_wants_dir(root, "timers.target.wants")
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    _link_unit(wants, "chad-monoonly.timer", units_dir)

    def fake_show(args: Sequence[str]):
        # Empty elapse values — timer has never armed.
        return (
            "Id=chad-monoonly.timer\n"
            "NextElapseUSecRealtime=\n"
            "NextElapseUSecMonotonic=infinity\n"
        )

    def fake_cat(args: Sequence[str]):
        return (
            "[Unit]\nDescription=monoonly fixture\n"
            "[Timer]\nOnUnitActiveSec=30min\n"
            "[Install]\nWantedBy=timers.target\n"
        )

    result = lint.scan(wants_root=root,
                       show_runner=fake_show, cat_runner=fake_cat)
    assert result["ok"] is True  # no chad regulars
    assert len(result["monotonic_no_calendar_warnings"]) == 1
    warn = result["monotonic_no_calendar_warnings"][0]
    assert warn["unit"] == "chad-monoonly.timer"
    assert warn["reason"] == "monotonic_only_no_anchor"


def test_timer_with_oncalendar_not_warned(tmp_path: Path) -> None:
    root = _make_wants_root(tmp_path)
    wants = _make_wants_dir(root, "timers.target.wants")
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    _link_unit(wants, "chad-cal.timer", units_dir)

    def fake_show(args: Sequence[str]):
        return (
            "Id=chad-cal.timer\n"
            "NextElapseUSecRealtime=\nNextElapseUSecMonotonic=\n"
        )

    def fake_cat(args: Sequence[str]):
        return (
            "[Timer]\nOnUnitActiveSec=30min\nOnCalendar=*-*-* 12:00:00\n"
        )

    result = lint.scan(wants_root=root,
                       show_runner=fake_show, cat_runner=fake_cat)
    assert result["monotonic_no_calendar_warnings"] == []


# -----------------------------------------------------------------------------
# (iv) atomic write of the artifact
# -----------------------------------------------------------------------------

def test_artifact_written_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_wants_root(tmp_path)
    _make_wants_dir(root, "timers.target.wants")

    out = tmp_path / "deep" / "nested" / "lint.json"

    seen_replaces: List[tuple] = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        seen_replaces.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(lint.os, "replace", tracking_replace)

    rc = lint.main(["--wants-root", str(root), "--output", str(out),
                    "--no-monotonic-check", "--quiet"])
    assert rc == 0
    assert out.is_file()
    assert seen_replaces, "expected at least one os.replace call"
    tmp_src, dst = seen_replaces[-1]
    assert dst == str(out)
    assert tmp_src.endswith(".tmp")
    # And the temp file must not linger after a successful write.
    assert not Path(tmp_src).exists()


# -----------------------------------------------------------------------------
# (v) idempotent rerun — second invocation produces the same payload shape
# -----------------------------------------------------------------------------

def test_idempotent_rerun(tmp_path: Path) -> None:
    root = _make_wants_root(tmp_path)
    wants = _make_wants_dir(root, "timers.target.wants")
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    _link_unit(wants, "chad-foo.service", units_dir)
    out = tmp_path / "lint.json"

    rc1 = lint.main(["--wants-root", str(root), "--output", str(out),
                     "--no-monotonic-check", "--quiet"])
    p1 = json.loads(out.read_text(encoding="utf-8"))
    rc2 = lint.main(["--wants-root", str(root), "--output", str(out),
                     "--no-monotonic-check", "--quiet"])
    p2 = json.loads(out.read_text(encoding="utf-8"))

    assert rc1 == 0 and rc2 == 0
    # Stable scalar fields (ts_utc will change, so exclude it)
    for key in ("ok", "regular_file_count", "chad_regular_file_count",
                "os_regular_file_count", "entries",
                "monotonic_no_calendar_warnings", "schema_version"):
        assert p1[key] == p2[key], f"field {key} drifted across reruns"


# -----------------------------------------------------------------------------
# Defensive: refusal of non-read systemctl args
# -----------------------------------------------------------------------------

def test_run_systemctl_refuses_mutations(monkeypatch: pytest.MonkeyPatch) -> None:
    called: List[Sequence[str]] = []

    def fake_run(*args, **kwargs):  # pragma: no cover - should not run
        called.append(args[0])
        raise AssertionError("subprocess.run must not be invoked for mutations")

    monkeypatch.setattr(lint.subprocess, "run", fake_run)
    assert lint._run_systemctl(["enable", "chad-foo.timer"]) is None
    assert lint._run_systemctl(["start", "chad-foo.service"]) is None
    assert called == []

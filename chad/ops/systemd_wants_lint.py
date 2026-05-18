#!/usr/bin/env python3
"""
chad/ops/systemd_wants_lint.py — GAP-032 preventive lint guard.

Scans /etc/systemd/system/*.wants/ for the GAP-032 corruption signature
(regular files where systemd expects enable-symlinks) and the Phase-27
monotonic-only-timer fragility (chad *.timer with OnUnitActiveSec /
OnBootSec, no OnCalendar, AND empty NextElapseUSecRealtime AND empty
NextElapseUSecMonotonic — i.e. timer that has never armed an
elapse).

Strictly READ-ONLY at runtime:
- no sudo
- no systemctl enable/disable/start/stop/rm
- only os.scandir + systemctl show/cat (read)

Importable and CLI:
- ``scan() -> dict``         — pure inspection, returns structured result.
- ``main(argv) -> int``      — writes runtime/systemd_wants_lint.json
                              atomically and returns:
                                0 → clean OR monotonic-only warnings only
                                2 → chad-scoped regular-file corruption

The exit code is keyed to **chad scope only**. OS-side wants entries
(e.g. apt-daily.timer, fwupd-refresh.timer) are commonly distributed as
regular files by their packages; they are reported as informational
entries but do NOT flip exit 2. A chad-* regular file is by contrast a
genuine GAP-032 regression and DOES flip exit 2.
"""
from __future__ import annotations

import json
import logging
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LOG = logging.getLogger("chad.ops.systemd_wants_lint")

DEFAULT_WANTS_ROOT = Path("/etc/systemd/system")
DEFAULT_OUTPUT_PATH = Path("/home/ubuntu/chad_finale/runtime/systemd_wants_lint.json")
SCHEMA_VERSION = "systemd_wants_lint.v1"
CHAD_PREFIX = "chad-"
SYSTEMCTL_TIMEOUT_SEC = 5


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _classify(name: str) -> str:
    """chad-* entries are operational scope; everything else is OS scope."""
    return "chad" if name.startswith(CHAD_PREFIX) else "os"


def _scan_wants_dirs(wants_root: Path) -> List[Dict[str, Any]]:
    """Return entries inside *.wants/ subdirs that are regular files (not symlinks).

    A symlink with a missing target is still classified by the type of its
    link entry (i.e. symlink), not the target, so we use os.scandir with
    follow_symlinks=False on the entry's stat.
    """
    entries: List[Dict[str, Any]] = []
    if not wants_root.is_dir():
        return entries
    try:
        with os.scandir(str(wants_root)) as it:
            parents = sorted(
                (e for e in it if e.is_dir(follow_symlinks=False) and e.name.endswith(".wants")),
                key=lambda e: e.name,
            )
    except OSError:
        return entries
    for parent in parents:
        try:
            with os.scandir(parent.path) as it:
                children = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for child in children:
            try:
                st = child.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            entries.append(
                {
                    "path": child.path,
                    "parent_target": parent.name,
                    "kind": _classify(child.name),
                }
            )
    return entries


def _run_systemctl(args: Sequence[str]) -> Optional[str]:
    """Run a READ-ONLY systemctl command. Returns stdout or None on failure.

    Only `show` and `cat` subcommands are permitted here; anything else is
    refused defensively so a future refactor cannot accidentally mutate state.
    """
    if not args or args[0] not in ("show", "cat"):
        LOG.warning("refused non-read systemctl invocation: %s", args)
        return None
    try:
        res = subprocess.run(
            ["systemctl", *args],
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_TIMEOUT_SEC,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return res.stdout or ""


def _list_chad_timers(wants_root: Path) -> List[str]:
    """Enumerate chad-*.timer unit basenames present under wants_root.

    We avoid `systemctl list-unit-files` to keep the surface narrow and
    deterministic. A timer that is not symlinked into a .wants/ tree is
    considered not-enabled and out of scope for the monotonic warning.
    """
    seen: List[str] = []
    if not wants_root.is_dir():
        return seen
    try:
        with os.scandir(str(wants_root)) as it:
            parents = [e for e in it if e.is_dir(follow_symlinks=False) and e.name.endswith(".wants")]
    except OSError:
        return seen
    for parent in parents:
        try:
            with os.scandir(parent.path) as it:
                for child in it:
                    name = child.name
                    if name.startswith(CHAD_PREFIX) and name.endswith(".timer") and name not in seen:
                        seen.append(name)
        except OSError:
            continue
    return sorted(seen)


def _parse_show_props(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _parse_timer_section(unit_text: str) -> Dict[str, List[str]]:
    """Extract [Timer] directives from `systemctl cat <unit>` output.

    Returns a dict like {"OnUnitActiveSec": ["30min"], "OnCalendar": [...]}.
    Multiple lines with the same key are preserved.
    """
    out: Dict[str, List[str]] = {}
    in_timer = False
    for raw in unit_text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_timer = line.lower() == "[timer]"
            continue
        if not in_timer or not line or line.startswith("#") or line.startswith(";"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out.setdefault(k.strip(), []).append(v.strip())
    return out


_INFINITY_TOKENS = {"", "0", "infinity", "n/a"}


def _is_empty_elapse(value: Optional[str]) -> bool:
    if value is None:
        return True
    return value.strip().lower() in _INFINITY_TOKENS


def _detect_monotonic_only(
    timer_units: Sequence[str],
    *,
    show_runner=_run_systemctl,
    cat_runner=_run_systemctl,
) -> List[Dict[str, str]]:
    """Flag chad timers with monotonic-only directives and no armed elapse.

    The runner injection points exist purely for unit testing — production
    callers should use the defaults.
    """
    warnings: List[Dict[str, str]] = []
    for unit in timer_units:
        show_out = show_runner(["show", unit, "-p", "Id", "-p", "NextElapseUSecRealtime", "-p", "NextElapseUSecMonotonic"])
        cat_out = cat_runner(["cat", unit])
        if show_out is None or cat_out is None:
            continue
        props = _parse_show_props(show_out)
        timer = _parse_timer_section(cat_out)
        has_calendar = bool(timer.get("OnCalendar"))
        has_monotonic = bool(timer.get("OnUnitActiveSec") or timer.get("OnBootSec") or timer.get("OnUnitInactiveSec"))
        if not has_monotonic or has_calendar:
            continue
        real_empty = _is_empty_elapse(props.get("NextElapseUSecRealtime"))
        mono_empty = _is_empty_elapse(props.get("NextElapseUSecMonotonic"))
        if real_empty and mono_empty:
            warnings.append(
                {
                    "unit": unit,
                    "reason": "monotonic_only_no_anchor",
                }
            )
    return warnings


def scan(
    *,
    wants_root: Path = DEFAULT_WANTS_ROOT,
    enable_monotonic_check: bool = True,
    show_runner=_run_systemctl,
    cat_runner=_run_systemctl,
) -> Dict[str, Any]:
    """Run a single read-only lint pass and return a structured result."""
    entries = _scan_wants_dirs(wants_root)
    chad_regulars = [e for e in entries if e["kind"] == "chad"]
    if enable_monotonic_check:
        timers = _list_chad_timers(wants_root)
        monotonic_warnings = _detect_monotonic_only(
            timers,
            show_runner=show_runner,
            cat_runner=cat_runner,
        )
    else:
        monotonic_warnings = []
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": not chad_regulars,
        "regular_file_count": len(entries),
        "chad_regular_file_count": len(chad_regulars),
        "os_regular_file_count": len(entries) - len(chad_regulars),
        "entries": entries,
        "monotonic_no_calendar_warnings": monotonic_warnings,
        "ts_utc": _utc_now_iso(),
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Read-only lint for systemd .wants/ directories (GAP-032 + Phase-27)."
    )
    parser.add_argument(
        "--wants-root",
        default=str(DEFAULT_WANTS_ROOT),
        help="Root dir whose *.wants/ subdirs are scanned.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Artifact path for the lint result JSON.",
    )
    parser.add_argument(
        "--no-monotonic-check",
        action="store_true",
        help="Skip the Phase-27 monotonic-only-timer check.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    result = scan(
        wants_root=Path(args.wants_root),
        enable_monotonic_check=not args.no_monotonic_check,
    )
    try:
        _atomic_write_json(Path(args.output), result)
    except OSError as exc:
        LOG.warning("failed to write %s: %s", args.output, exc)

    if not args.quiet:
        LOG.info(
            "systemd_wants_lint chad_regular=%d os_regular=%d monotonic_warnings=%d ok=%s",
            result["chad_regular_file_count"],
            result["os_regular_file_count"],
            len(result["monotonic_no_calendar_warnings"]),
            result["ok"],
        )

    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

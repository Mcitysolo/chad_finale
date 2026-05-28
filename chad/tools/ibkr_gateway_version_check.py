#!/usr/bin/env python3
"""Read-only IBKR Gateway version audit (Fix C / Channel 2).

Detects the installed IB Gateway build by inspecting the on-disk install tree
under ``/home/ubuntu/Jts/ibgateway/`` and compares it against a target build
(default 1045 = 10.45, per external IBKR operator-community recommendation as
of 2026-05-28). It classifies staleness and prints a structured verdict.

The tool is strictly READ-ONLY:
  * It never downloads anything and makes NO network calls.
  * It never mutates the Gateway install tree.
  * It writes to disk ONLY when ``--output <path>`` is given (atomic replace).
    The default invocation prints to stdout and nothing else.

Detection strategies (first hit per install dir wins; highest build across
dirs is selected):
  1. directory_name — the build dir is named after the build (e.g. "1037").
  2. version_txt    — a version.txt inside the build dir (rare on Linux).
  3. jar_filename   — jts4launch-<BUILD>.jar / twslaunch-<BUILD>.jar in jars/.

CLI:
    python -m chad.tools.ibkr_gateway_version_check [--target-build N]
        [--json] [--quiet] [--output PATH]

Exit codes:
    0  info     (installed build >= target)
    2  warning  (behind by <= 5 builds) OR stale (behind by > 5 builds)
    3  unknown  (detection failed)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "ibkr_gateway_version_check.v1"

DEFAULT_JTS_ROOT = Path("/home/ubuntu/Jts")
DEFAULT_TARGET_BUILD = 1045
TARGET_SOURCE = "external_ibkr_operator_recommendation_2026-05-28"
STALE_DELTA_THRESHOLD = 5  # behind by > this many builds → "stale"

EXIT_INFO = 0
EXIT_STALE_OR_WARNING = 2
EXIT_UNKNOWN = 3

_JAR_RE = re.compile(r"(?:jts4launch|twslaunch)-(\d{3,5})\.jar$")


def build_to_display(build: int) -> str:
    """Render a build int as a dotted version, e.g. 1037 -> '10.37'."""
    return f"{build // 100}.{build % 100:02d}"


def _parse_version_txt(path: Path) -> Optional[int]:
    """Parse a build number from the first line of a version.txt."""
    try:
        first = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    if first.isdigit():
        return int(first)
    m = re.search(r"(\d+)\.(\d+)", first)
    if m:
        return int(m.group(1)) * 100 + int(m.group(2))
    return None


def _scan_install_dir(build_dir: Path) -> Optional[tuple[int, str]]:
    """Return (build, detection_source) for one install dir, trying each
    strategy in order; None if no build can be derived."""
    # Strategy 1 — directory name is the build number.
    if build_dir.name.isdigit():
        return int(build_dir.name), "directory_name"
    # Strategy 2 — version.txt inside the build dir.
    vtxt = build_dir / "version.txt"
    if vtxt.is_file():
        build = _parse_version_txt(vtxt)
        if build is not None:
            return build, "version_txt"
    # Strategy 3 — jar filename inside jars/.
    jars = build_dir / "jars"
    if jars.is_dir():
        for jar in sorted(jars.iterdir()):
            m = _JAR_RE.search(jar.name)
            if m:
                return int(m.group(1)), "jar_filename"
    return None


def detect_installed(jts_root: Path = DEFAULT_JTS_ROOT) -> dict[str, Any]:
    """Inspect <jts_root>/ibgateway/ and return the highest installed build."""
    gateway_root = jts_root / "ibgateway"
    none_result = {
        "build": None,
        "display": None,
        "install_path": str(gateway_root),
        "detection_source": "none",
        "detection_error": None,
    }
    if not gateway_root.is_dir():
        none_result["detection_error"] = "detection_failed: ibgateway dir not found"
        return none_result

    candidates: list[tuple[int, str, Path]] = []
    try:
        subdirs = [p for p in gateway_root.iterdir() if p.is_dir()]
    except OSError as exc:
        none_result["detection_error"] = f"detection_failed: {exc}"
        return none_result

    for sub in subdirs:
        scanned = _scan_install_dir(sub)
        if scanned is not None:
            build, source = scanned
            candidates.append((build, source, sub))

    if not candidates:
        none_result["detection_error"] = "detection_failed: no build derivable from install tree"
        return none_result

    # Highest build wins; on a tie, directory_name source is preferred.
    _source_rank = {"directory_name": 0, "version_txt": 1, "jar_filename": 2}
    build, source, path = max(
        candidates, key=lambda c: (c[0], -_source_rank.get(c[1], 9))
    )
    return {
        "build": build,
        "display": build_to_display(build),
        "install_path": str(path),
        "detection_source": source,
        "detection_error": None,
    }


def classify(installed_build: Optional[int], target_build: int) -> dict[str, Any]:
    """Compare installed vs target and return the comparison + severity."""
    if installed_build is None:
        return {
            "comparison": {"is_current": False, "build_delta": None, "severity": "unknown"},
            "recommendation": "investigate; detection failed",
            "exit_code": EXIT_UNKNOWN,
        }
    behind = target_build - installed_build  # positive when stale
    is_current = installed_build >= target_build
    if is_current:
        severity, recommendation, exit_code = "info", "no action", EXIT_INFO
    elif behind <= STALE_DELTA_THRESHOLD:
        severity, recommendation, exit_code = "warning", "schedule upgrade", EXIT_STALE_OR_WARNING
    else:
        severity, recommendation, exit_code = "stale", "upgrade priority", EXIT_STALE_OR_WARNING
    return {
        "comparison": {"is_current": is_current, "build_delta": behind, "severity": severity},
        "recommendation": recommendation,
        "exit_code": exit_code,
    }


def build_report(
    jts_root: Path = DEFAULT_JTS_ROOT,
    target_build: int = DEFAULT_TARGET_BUILD,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Assemble the full version-check report (no I/O side effects)."""
    ts = (now_utc or datetime.now(timezone.utc)).isoformat()
    installed = detect_installed(jts_root)
    verdict = classify(installed["build"], target_build)
    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": ts,
        "installed": installed,
        "target": {
            "build": target_build,
            "display": build_to_display(target_build),
            "source": TARGET_SOURCE,
        },
        "comparison": verdict["comparison"],
        "recommendation": verdict["recommendation"],
    }


def _exit_code_for(report: dict[str, Any]) -> int:
    sev = report["comparison"]["severity"]
    if sev == "unknown":
        return EXIT_UNKNOWN
    if sev == "info":
        return EXIT_INFO
    return EXIT_STALE_OR_WARNING


def write_output_atomic(report: dict[str, Any], out_path: Path) -> None:
    """Write the report JSON atomically (tmp + os.replace)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, out_path)


def _human_summary(report: dict[str, Any]) -> str:
    inst = report["installed"]
    comp = report["comparison"]
    return (
        f"IBKR Gateway version check [{comp['severity'].upper()}]\n"
        f"  installed : build={inst['build']} ({inst['display']}) "
        f"source={inst['detection_source']} path={inst['install_path']}\n"
        f"  target    : build={report['target']['build']} ({report['target']['display']})\n"
        f"  comparison: is_current={comp['is_current']} build_delta={comp['build_delta']}\n"
        f"  recommend : {report['recommendation']}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point: build the report, optionally persist, print, return exit code."""
    parser = argparse.ArgumentParser(description="Read-only IBKR Gateway version audit.")
    parser.add_argument("--json", action="store_true", help="Output structured JSON only.")
    parser.add_argument("--quiet", action="store_true", help="Suppress human output, JSON only.")
    parser.add_argument("--target-build", type=int, default=DEFAULT_TARGET_BUILD,
                        help=f"Build to compare against (default {DEFAULT_TARGET_BUILD}).")
    parser.add_argument("--output", type=str, default=None,
                        help="Also write the JSON to this path (atomic). Default: stdout only.")
    args = parser.parse_args(argv)

    report = build_report(target_build=args.target_build)

    if args.output:
        write_output_atomic(report, Path(args.output))

    json_only = args.json or args.quiet
    if not json_only:
        print(_human_summary(report), file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))

    return _exit_code_for(report)


if __name__ == "__main__":
    sys.exit(main())

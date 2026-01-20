"""
CHAD Phase 10 — Reports Index (Latest Artifacts)

Purpose
-------
Create a single source-of-truth index pointing to the latest generated report artifacts.

Scans:
  /home/ubuntu/CHAD FINALE/reports/ops/

Writes:
  reports/ops/REPORTS_INDEX_<ts>.json
  reports/ops/REPORTS_INDEX_LATEST.json (symlink)

This is read-only and safe:
- No secrets
- No broker calls
- No runtime mutation beyond index files
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path("/home/ubuntu/CHAD FINALE")
OPS_DIR = REPO_ROOT / "reports" / "ops"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def newest(glob_pat: str) -> Optional[Path]:
    files = sorted(OPS_DIR.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_symlink(target: Path, link: Path) -> None:
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
    except FileNotFoundError:
        pass
    link.symlink_to(target.name)  # relative symlink within same directory


def run() -> Tuple[Path, Path]:
    OPS_DIR.mkdir(parents=True, exist_ok=True)

    items = {
        "daily_ops_json": newest("DAILY_OPS_REPORT_*.json"),
        "daily_ops_md": newest("DAILY_OPS_REPORT_*.md"),
        "daily_perf_json": newest("DAILY_PERF_REPORT_*.json"),
        "daily_perf_md": newest("DAILY_PERF_REPORT_*.md"),
        "daily_exec_json": newest("DAILY_EXEC_REPORT_*.json"),
        "daily_exec_md": newest("DAILY_EXEC_REPORT_*.md"),
        "weekly_investor_json": newest("WEEKLY_INVESTOR_REPORT_*.json"),
        "weekly_investor_md": newest("WEEKLY_INVESTOR_REPORT_*.md"),
    }

    payload: Dict[str, Any] = {
        "generated_utc": utc_now_iso(),
        "host": socket.gethostname(),
        "ops_dir": str(OPS_DIR),
        "latest": {k: (str(v) if v else None) for k, v in items.items()},
    }

    ts = utc_now_compact()
    out = OPS_DIR / f"REPORTS_INDEX_{ts}.json"
    atomic_write_json(out, payload)

    latest_link = OPS_DIR / "REPORTS_INDEX_LATEST.json"
    update_symlink(out, latest_link)

    print(str(out))
    print(str(latest_link))
    return out, latest_link


if __name__ == "__main__":
    run()

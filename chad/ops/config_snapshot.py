"""
chad.ops.config_snapshot

Creates SHA-256 snapshots of all config/*.json files for mutation governance
and tamper detection.

Writes timestamped snapshots to data/config_snapshots/ with per-file hashes
and a composite hash for fast drift comparison.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_now_file_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def create_config_snapshot(
    *,
    repo_root: Path,
    trigger: str = "manual",
    config_dir: Optional[Path] = None,
    snapshots_dir: Optional[Path] = None,
) -> Path:
    """
    Snapshot all config/*.json files.

    Returns the path to the written snapshot file.
    """
    cfg_dir = config_dir or (repo_root / "config")
    snap_dir = snapshots_dir or (repo_root / "data" / "config_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, Dict[str, Any]] = {}
    for p in sorted(cfg_dir.glob("*.json")):
        if not p.is_file():
            continue
        files[p.name] = {
            "sha256": _sha256_file(p),
            "size_bytes": p.stat().st_size,
        }

    # Composite hash: deterministic concat of all individual hashes
    composite_input = "".join(
        f"{name}:{info['sha256']}" for name, info in sorted(files.items())
    )
    composite_hash = hashlib.sha256(composite_input.encode("utf-8")).hexdigest()

    ts = _utc_now_iso()
    payload = {
        "schema_version": "config_snapshot.v1",
        "ts_utc": ts,
        "trigger": trigger,
        "composite_hash": composite_hash,
        "files_count": len(files),
        "files": files,
    }

    filename = f"snapshot_{_utc_now_file_tag()}.json"
    out_path = snap_dir / filename
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, out_path)

    return out_path


def get_current_composite_hash(*, repo_root: Path, config_dir: Optional[Path] = None) -> str:
    """Compute the current composite hash without writing a snapshot."""
    cfg_dir = config_dir or (repo_root / "config")
    hashes = []
    for p in sorted(cfg_dir.glob("*.json")):
        if not p.is_file():
            continue
        hashes.append(f"{p.name}:{_sha256_file(p)}")
    return hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()

#!/usr/bin/env python3
"""
CHAD — Execution Environment Publisher (SSOT v5)

Publishes runtime/execution_environment.json from the active process
environment. The mode value is read through the canonical
chad.execution.execution_config.get_execution_mode() reader (no direct
os.environ peek), so the published file always reflects the same truth that
LiveGate, IBKR adapter, and Kraken adapter consume.

Schema (preserved from prior bootstrap file):
    schema_version : "execution_environment.v1"
    ts_utc         : ISO8601 UTC, injected by write_runtime_state_json
    ttl_seconds    : 60
    ok             : True when publish succeeds
    exec_mode      : "dry_run" | "paper" | "live"
    live_enabled   : exec_mode == "live"
    ibkr_enabled   : True (CHAD always enables the IBKR adapter)
    ibkr_dry_run   : True unless exec_mode == "live"
    kraken_enabled : truthy CHAD_KRAKEN_ENABLED
    source         : "systemd_env"
    notes          : short description of the truth source

This module exposes publish_once() so callers (LiveGate, the systemd
oneshot CLI, tests) can refresh the file safely. All file I/O is atomic
via chad.utils.runtime_json.write_runtime_state_json.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from chad.execution.execution_config import ExecutionMode, get_execution_mode
from chad.utils.runtime_json import write_runtime_state_json

LOG = logging.getLogger("chad.ops.execution_environment_publisher")

SCHEMA_VERSION = "execution_environment.v1"
DEFAULT_TTL_SECONDS = 60
DEFAULT_SOURCE = "systemd_env"
DEFAULT_NOTES = (
    "Published from active CHAD_EXECUTION_MODE via "
    "chad.execution.execution_config canonical reader."
)


def _repo_root() -> Path:
    root = str(os.environ.get("CHAD_ROOT", "")).strip()
    if root:
        p = Path(root).expanduser()
        if p.is_dir():
            return p.resolve()
    return Path(__file__).resolve().parents[2]


def _runtime_dir() -> Path:
    rd = str(os.environ.get("CHAD_RUNTIME_DIR", "")).strip()
    if rd:
        return Path(rd).expanduser().resolve()
    return (_repo_root() / "runtime").resolve()


def _truthy(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _exec_mode_label(mode: ExecutionMode) -> str:
    if mode == ExecutionMode.IBKR_LIVE:
        return "live"
    if mode == ExecutionMode.IBKR_PAPER:
        return "paper"
    return "dry_run"


def build_payload(
    *,
    source: str = DEFAULT_SOURCE,
    notes: str = DEFAULT_NOTES,
) -> Dict[str, Any]:
    mode = get_execution_mode()
    label = _exec_mode_label(mode)

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "exec_mode": label,
        "live_enabled": bool(label == "live"),
        "ibkr_enabled": _truthy("IBKR_ENABLED", True),
        "ibkr_dry_run": bool(label != "live"),
        "kraken_enabled": _truthy("CHAD_KRAKEN_ENABLED", _truthy("KRAKEN_ENABLED", True)),
        "source": str(source),
        "notes": str(notes),
    }


def publish_once(
    runtime_dir: Optional[Path] = None,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    source: str = DEFAULT_SOURCE,
    notes: str = DEFAULT_NOTES,
) -> Dict[str, Any]:
    """Atomically publish runtime/execution_environment.json.

    Returns the dict that was written (with ts_utc + ttl_seconds injected).
    Raises only on disk-write failures; missing or malformed prior file is
    irrelevant because the publisher overwrites unconditionally.
    """
    rd = runtime_dir if runtime_dir is not None else _runtime_dir()
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / "execution_environment.json"
    payload = build_payload(source=source, notes=notes)
    return write_runtime_state_json(out, payload, ttl_seconds=int(ttl_seconds), inject_ts=True)


def publish_once_safe(
    runtime_dir: Optional[Path] = None,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    source: str = DEFAULT_SOURCE,
    notes: str = DEFAULT_NOTES,
) -> bool:
    """Fail-soft variant for callers like LiveGate.

    Returns True on success, False on any error. Never raises.
    """
    try:
        publish_once(runtime_dir, ttl_seconds=ttl_seconds, source=source, notes=notes)
        return True
    except Exception as exc:
        LOG.warning("execution_environment publish failed: %r", exc)
        return False


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="chad.ops.execution_environment_publisher")
    p.add_argument("--once", action="store_true", help="Publish a single snapshot and exit (default).")
    p.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    written = publish_once(ttl_seconds=int(args.ttl_seconds), source=str(args.source))
    LOG.info(
        "execution_environment published exec_mode=%s source=%s ts_utc=%s",
        written.get("exec_mode"),
        written.get("source"),
        written.get("ts_utc"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

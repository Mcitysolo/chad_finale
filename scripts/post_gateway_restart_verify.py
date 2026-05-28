#!/usr/bin/env python3
"""Post-restart health verifier for chad-ibgateway.service (Fix B / Channel 2).

Invoked as the final ``ExecStartPost=`` step of
``chad-ibgateway-nightly-restart.service`` after the nightly Gateway restart
and a 60s warm-up sleep. It performs a small set of bounded, read-only checks
to confirm the Gateway came back healthy and emits a structured alert artifact
under ``reports/gateway_restart_log/<UTC_TS>.json``.

This verifier is strictly READ-ONLY with respect to system state:
  * It does NOT mutate any ``runtime/*`` JSON.
  * It does NOT mutate ``stop_bus``.
  * It does NOT place, cancel, or query broker orders.
  * Its only write is the alert artifact under ``reports/gateway_restart_log/``.

Checks (in order, each bounded):
  1. Port 4002 listening              — 60s budget, retry every 5s.
  2. ibkr_status.json mtime refreshed — 120s budget, retry every 5s.
  3. Latency below sanity threshold   — single read, healthy if < 5000ms.
  4. recovery_state not wedged        — fail only if state==above_threshold
                                         AND latency_ms > 5000ms.

Exit codes:
    0  healthy (all checks pass)
    2  unhealthy (any check failed)
"""

from __future__ import annotations

import json
import os
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "gateway_restart_verify.v1"

EXIT_OK = 0
EXIT_UNHEALTHY = 2

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS_PATH = REPO_ROOT / "runtime" / "ibkr_status.json"
DEFAULT_LOG_DIR = REPO_ROOT / "reports" / "gateway_restart_log"

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 4002

PORT_BUDGET_S = 60.0
ARTIFACT_BUDGET_S = 120.0
RETRY_INTERVAL_S = 5.0
LATENCY_SANITY_MS = 5000.0


def _real_connect(host: str, port: int, timeout: float = 3.0) -> None:
    """Open and immediately close a TCP connection. Raises OSError on failure."""
    with socket.create_connection((host, port), timeout=timeout):
        pass


@dataclass
class Deps:
    """Injectable seams so the verifier is fully testable without a real Gateway."""

    status_path: Path = DEFAULT_STATUS_PATH
    log_dir: Path = DEFAULT_LOG_DIR
    host: str = GATEWAY_HOST
    port: int = GATEWAY_PORT
    now_utc: Callable[[], datetime] = field(
        default_factory=lambda: (lambda: datetime.now(timezone.utc))
    )
    monotonic: Callable[[], float] = field(
        default_factory=lambda: __import__("time").monotonic
    )
    sleep: Callable[[float], None] = field(
        default_factory=lambda: __import__("time").sleep
    )
    connect: Callable[[str, int], None] = _real_connect


def check_port_listening(deps: Deps) -> dict[str, Any]:
    """Check 1 — Gateway accepting connections on its API port (60s budget)."""
    t0 = deps.monotonic()
    attempts = 0
    ok = False
    while True:
        attempts += 1
        try:
            deps.connect(deps.host, deps.port)
            ok = True
            break
        except OSError:
            ok = False
        if deps.monotonic() - t0 >= PORT_BUDGET_S:
            break
        deps.sleep(RETRY_INTERVAL_S)
    return {
        "ok": ok,
        "elapsed_seconds": round(deps.monotonic() - t0, 3),
        "attempts": attempts,
    }


def check_artifact_fresh(deps: Deps, start_epoch: float) -> dict[str, Any]:
    """Check 2 — ibkr_status.json rewritten since the verifier started (120s)."""
    t0 = deps.monotonic()
    ok = False
    mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(deps.status_path)
        except OSError:
            mtime = 0.0
        if mtime >= start_epoch:
            ok = True
            break
        if deps.monotonic() - t0 >= ARTIFACT_BUDGET_S:
            ok = False
            break
        deps.sleep(RETRY_INTERVAL_S)
    mtime_iso = (
        datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        if mtime > 0
        else ""
    )
    return {
        "ok": ok,
        "elapsed_seconds": round(deps.monotonic() - t0, 3),
        "mtime_iso": mtime_iso,
    }


def _read_status(deps: Deps) -> dict[str, Any]:
    """Read the ibkr_status.json document, returning {} on any error."""
    try:
        with open(deps.status_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def check_latency_healthy(status: dict[str, Any]) -> dict[str, Any]:
    """Check 3 — latency below the sanity ceiling (single read)."""
    raw = status.get("latency_ms")
    try:
        latency_ms = float(raw) if raw is not None else float("inf")
    except (TypeError, ValueError):
        latency_ms = float("inf")
    return {
        "ok": latency_ms < LATENCY_SANITY_MS,
        "latency_ms": latency_ms,
        "threshold_ms": LATENCY_SANITY_MS,
    }


def check_recovery_state(status: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    """Check 4 — wedge persistence: above_threshold AND high latency is unhealthy."""
    state = str(status.get("current_recovery_state", "unknown"))
    wedged = state == "above_threshold" and latency_ms > LATENCY_SANITY_MS
    return {"ok": not wedged, "state": state}


def run_verification(deps: Deps | None = None) -> dict[str, Any]:
    """Execute all checks and build the alert artifact payload (no I/O side effects)."""
    deps = deps or Deps()
    start_dt = deps.now_utc()
    start_epoch = start_dt.timestamp()

    port = check_port_listening(deps)
    artifact = check_artifact_fresh(deps, start_epoch)
    status = _read_status(deps)
    latency = check_latency_healthy(status)
    recovery = check_recovery_state(status, latency["latency_ms"])

    overall_ok = bool(port["ok"] and artifact["ok"] and latency["ok"] and recovery["ok"])
    exit_code = EXIT_OK if overall_ok else EXIT_UNHEALTHY

    if not port["ok"]:
        severity = "CRITICAL"
    elif not overall_ok:
        severity = "WARNING"
    else:
        severity = "INFO"

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": deps.now_utc().isoformat(),
        "verifier_pid": os.getpid(),
        "checks": {
            "port_listening": port,
            "artifact_fresh": artifact,
            "latency_healthy": latency,
            "recovery_state": recovery,
        },
        "overall_ok": overall_ok,
        "exit_code": exit_code,
        "alert_severity": severity,
    }


def write_artifact(payload: dict[str, Any], deps: Deps) -> Path:
    """Persist the alert artifact under reports/gateway_restart_log/<UTC_TS>.json."""
    deps.log_dir.mkdir(parents=True, exist_ok=True)
    ts = deps.now_utc().strftime("%Y%m%dT%H%M%SZ")
    out_path = deps.log_dir / f"{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def _summary_line(payload: dict[str, Any], artifact_path: Path) -> str:
    """One-line structured summary for the journal."""
    c = payload["checks"]
    return (
        f"[{payload['alert_severity']}] gateway_restart_verify "
        f"overall_ok={payload['overall_ok']} exit={payload['exit_code']} "
        f"port={c['port_listening']['ok']} "
        f"artifact_fresh={c['artifact_fresh']['ok']} "
        f"latency_ms={c['latency_healthy']['latency_ms']} "
        f"latency_ok={c['latency_healthy']['ok']} "
        f"recovery_state={c['recovery_state']['state']} "
        f"artifact={artifact_path}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: run checks, write artifact, print summary, return exit code."""
    deps = Deps()
    payload = run_verification(deps)
    artifact_path = write_artifact(payload, deps)
    print(_summary_line(payload, artifact_path), file=sys.stderr)
    return int(payload["exit_code"])


if __name__ == "__main__":
    sys.exit(main())

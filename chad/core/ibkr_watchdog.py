from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


RUNTIME_DIR = "/home/ubuntu/CHAD FINALE/runtime"
STATUS_PATH = os.path.join(RUNTIME_DIR, "ibkr_status.json")


@dataclass
class IbkrStatus:
    """Represents the current IBKR connectivity / health status.

    This file is written by a oneshot watchdog, typically invoked via systemd.
    It is designed to be read by other CHAD components (API gateway, coach,
    shadow snapshot, etc.) to know whether IBKR is healthy and when it last was.
    """

    ok: bool
    last_checked_at: float
    last_checked_iso: str
    last_ok_at: Optional[float]
    last_ok_iso: Optional[str]
    consecutive_failures: int
    last_error: Optional[str]
    last_health_raw: Optional[Dict[str, Any]]

    @classmethod
    def default(cls) -> "IbkrStatus":
        """Create a default status when no prior file exists."""
        now = time.time()
        iso_now = _iso_utc(now)
        return cls(
            ok=False,
            last_checked_at=now,
            last_checked_iso=iso_now,
            last_ok_at=None,
            last_ok_iso=None,
            consecutive_failures=0,
            last_error="no prior status",
            last_health_raw=None,
        )

    @classmethod
    def load(cls, path: str) -> "IbkrStatus":
        """Load a previous status from disk, falling back to default on error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return cls.default()
        except Exception as exc:  # noqa: BLE001
            # Corrupt or unreadable file; start fresh but record the issue.
            status = cls.default()
            status.last_error = f"failed to load previous status: {exc!r}"
            return status

        try:
            return cls(
                ok=bool(data.get("ok", False)),
                last_checked_at=float(data.get("last_checked_at", time.time())),
                last_checked_iso=str(data.get("last_checked_iso", _iso_utc(time.time()))),
                last_ok_at=(
                    float(data["last_ok_at"])
                    if data.get("last_ok_at") is not None
                    else None
                ),
                last_ok_iso=(
                    str(data["last_ok_iso"])
                    if data.get("last_ok_iso") is not None
                    else None
                ),
                consecutive_failures=int(data.get("consecutive_failures", 0)),
                last_error=data.get("last_error"),
                last_health_raw=(
                    dict(data["last_health_raw"])
                    if isinstance(data.get("last_health_raw"), dict)
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            status = cls.default()
            status.last_error = f"failed to parse previous status: {exc!r}"
            return status

    def update_from_healthcheck(
        self,
        health: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> "IbkrStatus":
        """Return a new status updated with the latest healthcheck result.

        - `health` is the parsed JSON output of `chad.core.ibkr_healthcheck --json`,
          or None if the call failed.
        - `error` is a human-readable error string if the call failed, else None.
        """
        now = time.time()
        iso_now = _iso_utc(now)

        # Start from a copy of current self
        last_ok_at = self.last_ok_at
        last_ok_iso = self.last_ok_iso
        consecutive_failures = self.consecutive_failures
        ok = False
        last_error = error
        last_health_raw: Optional[Dict[str, Any]] = None

        if health is not None:
            last_health_raw = health
            ok_flag = bool(health.get("ok", False))

            if ok_flag:
                ok = True
                last_ok_at = now
                last_ok_iso = iso_now
                consecutive_failures = 0
                last_error = None
            else:
                ok = False
                consecutive_failures += 1
                # Try to extract a meaningful error message from health payload
                error_msg = health.get("error") or health.get("reason") or "ibkr healthcheck reported ok=false"
                last_error = str(error_msg)
        else:
            # No health payload at all → treat as failure.
            ok = False
            consecutive_failures += 1
            if last_error is None:
                last_error = "ibkr_healthcheck invocation failed"

        return IbkrStatus(
            ok=ok,
            last_checked_at=now,
            last_checked_iso=iso_now,
            last_ok_at=last_ok_at,
            last_ok_iso=last_ok_iso,
            consecutive_failures=consecutive_failures,
            last_error=last_error,
            last_health_raw=last_health_raw,
        )

    def to_json(self) -> str:
        """Serialize status to a stable JSON string."""
        return json.dumps(asdict(self), sort_keys=True, indent=2)


def _iso_utc(ts: float) -> str:
    """Return an ISO-8601 UTC timestamp string for a given epoch seconds."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _run_ibkr_healthcheck() -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Invoke `python -m chad.core.ibkr_healthcheck --json` as a subprocess.

    This avoids importing implementation details and leverages the same path
    that systemd services already use. Any failure is captured and returned as
    an error string.
    """
    cmd = [
        sys.executable,
        "-m",
        "chad.core.ibkr_healthcheck",
        "--json",
    ]

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to invoke ibkr_healthcheck: {exc!r}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        msg = f"ibkr_healthcheck exit code {proc.returncode}"
        if stderr:
            msg = f"{msg}; stderr={stderr}"
        elif stdout:
            msg = f"{msg}; stdout={stdout}"
        return None, msg

    if not stdout:
        return None, "ibkr_healthcheck produced no stdout"

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return None, f"failed to parse ibkr_healthcheck JSON: {exc!r}; raw={stdout[:256]!r}"

    return payload, None


def ensure_runtime_dir(path: str) -> None:
    """Make sure the runtime directory for the status file exists."""
    directory = os.path.dirname(path)
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)


def write_status(path: str, status: IbkrStatus) -> None:
    """Atomically write the status JSON to disk."""
    ensure_runtime_dir(path)
    tmp_path = f"{path}.tmp"
    data = status.to_json()

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def main() -> int:
    """Entry point for the IBKR watchdog.

    Behavior:
    - Loads previous status file (if present).
    - Runs the standard ibkr_healthcheck as a subprocess.
    - Updates status (including consecutive failure count).
    - Writes an updated JSON file to STATUS_PATH.
    - Prints a one-line summary to stdout for systemd logs.
    """
    previous = IbkrStatus.load(STATUS_PATH)
    health, error = _run_ibkr_healthcheck()
    status = previous.update_from_healthcheck(health=health, error=error)

    try:
        write_status(STATUS_PATH, status)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[ibkr_watchdog] ERROR: failed to write status file {STATUS_PATH!r}: {exc!r}",
            file=sys.stderr,
        )
        # Even if writing failed, we still return non-zero to signal systemd.
        return 1

    summary = (
        f"[ibkr_watchdog] ok={status.ok} "
        f"consecutive_failures={status.consecutive_failures} "
        f"last_ok_iso={status.last_ok_iso!r} "
        f"last_error={status.last_error!r}"
    )
    print(summary)

    # If IBKR is unhealthy, return non-zero so systemd can track failures.
    return 0 if status.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
chad.core.ibkr_healthcheck

Production-grade IBKR healthcheck (ib_insync) with CSB-compliant runtime state.

What it does
------------
- Connects to IBKR Gateway/TWS using ib_insync
- Calls reqCurrentTime() to prove the API is responsive
- Writes runtime/ibkr_status.json as a TTL-stamped runtime STATE file:
    - ts_utc
    - ttl_seconds
- Prints JSON to stdout when --json is provided

Why ib_insync
-------------
Your repo's last known-good healthcheck uses ib_insync and is aligned to how your
systemd unit injects IBKR_HOST/IBKR_PORT/IBKR_CLIENT_ID. Your recent ibapi-based
rewrite times out in this deployment.

Safety / ops posture
--------------------
- Default exit code is 0 even when ok=false (health is *status*, not a crash).
- Use --strict-exit to return non-zero on failure if you explicitly want that.
- Never writes secrets.
- Downstream gates (LiveGate) must fail-closed when status is stale/invalid.

Env vars (systemd injects these already)
----------------------------------------
- IBKR_HOST (default 127.0.0.1)
- IBKR_PORT (default 4002)
- IBKR_CLIENT_ID (default 9001)

"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from chad.utils.runtime_json import stable_json_dumps, write_runtime_state_json


@dataclass(frozen=True)
class IBKRHealthStatus:
    ok: bool
    error: Optional[str]
    host: str
    port: int
    client_id: int
    latency_ms: Optional[float]
    server_time_iso: Optional[str]


def _env_or_default(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    if val is not None and str(val).strip() != "":
        return str(val)
    if default is not None:
        return default
    raise RuntimeError(f"Missing required env var: {name}")


def _fmt_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc!r}"


def _repo_root() -> "os.PathLike[str]":
    # chad/core/ibkr_healthcheck.py -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _runtime_path() -> str:
    return os.path.join(_repo_root(), "runtime", "ibkr_status.json")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHAD IBKR healthcheck (connect + reqCurrentTime).")

    # Optional CLI overrides (defaults come from env or safe fallbacks)
    p.add_argument("--host", default=None, help="Override IBKR host (else env IBKR_HOST or 127.0.0.1).")
    p.add_argument("--port", type=int, default=None, help="Override IBKR port (else env IBKR_PORT or 4002).")
    p.add_argument("--client-id", type=int, default=None, help="Override IBKR client id (else env IBKR_CLIENT_ID or 9001).")

    p.add_argument("--timeout-seconds", type=float, default=8.0, help="Total timeout for healthcheck (default 8s).")
    p.add_argument("--ttl-seconds", type=int, default=120, help="TTL for runtime/ibkr_status.json (default 120s).")

    p.add_argument("--json", action="store_true", help="Print JSON payload to stdout.")
    p.add_argument("--strict-exit", action="store_true", help="Exit non-zero when ok=false (default false).")
    p.add_argument("--quiet-ibinsync", action="store_true", default=True, help="Suppress ib_insync stderr noise.")

    return p.parse_args()


def _healthcheck(host: str, port: int, client_id: int, timeout_s: float, quiet_ibinsync: bool) -> IBKRHealthStatus:
    """
    Perform the healthcheck using ib_insync.

    Returns ok=false with error on failure; never raises.
    """
    start = time.perf_counter()

    try:
        from ib_insync import IB  # type: ignore[import]
        from ib_insync import ib as ib_mod  # type: ignore[import]
    except Exception as exc:
        return IBKRHealthStatus(
            ok=False,
            error=f"import_error: {_fmt_exc(exc)}",
            host=host,
            port=port,
            client_id=client_id,
            latency_ms=None,
            server_time_iso=None,
        )

    # Silence ib_insync stderr spam (optional)
    stderr_buf: Optional[io.StringIO] = None
    stderr_cm: Any
    if quiet_ibinsync:
        stderr_buf = io.StringIO()
        stderr_cm = contextlib.redirect_stderr(stderr_buf)
    else:
        stderr_cm = contextlib.nullcontext()

    with stderr_cm:
        ib = IB()

        try:
            # ib.connect uses socket handshake + API negotiation
            ib.connect(host, port, clientId=client_id, timeout=float(timeout_s))
        except Exception as exc:
            return IBKRHealthStatus(
                ok=False,
                error=f"connect_failed: {_fmt_exc(exc)}",
                host=host,
                port=port,
                client_id=client_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                server_time_iso=None,
            )

        try:
            # Prove API responsiveness
            server_time = ib.reqCurrentTime()
            if server_time is None:
                return IBKRHealthStatus(
                    ok=False,
                    error="reqCurrentTime_returned_none",
                    host=host,
                    port=port,
                    client_id=client_id,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    server_time_iso=None,
                )

            # server_time is a datetime
            server_time_iso = server_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")  # type: ignore[name-defined]

            return IBKRHealthStatus(
                ok=True,
                error=None,
                host=host,
                port=port,
                client_id=client_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                server_time_iso=server_time_iso,
            )

        except Exception as exc:
            return IBKRHealthStatus(
                ok=False,
                error=f"reqCurrentTime_failed: {_fmt_exc(exc)}",
                host=host,
                port=port,
                client_id=client_id,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                server_time_iso=None,
            )

        finally:
            try:
                ib.disconnect()
            except Exception:
                pass


def _as_payload(st: IBKRHealthStatus) -> Dict[str, Any]:
    return {
        "ok": bool(st.ok),
        "error": st.error,
        "host": st.host,
        "port": int(st.port),
        "client_id": int(st.client_id),
        "latency_ms": st.latency_ms,
        "server_time_iso": st.server_time_iso,
    }


def main() -> int:
    args = _parse_args()

    # Resolve config from CLI overrides or env with safe defaults
    host = args.host or os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(args.port or os.getenv("IBKR_PORT", "4002"))
    client_id = int(args.client_id or os.getenv("IBKR_CLIENT_ID", "9001"))
    timeout_s = float(args.timeout_seconds)
    ttl_s = int(args.ttl_seconds)

    # Run the check
    st = _healthcheck(host, port, client_id, timeout_s, bool(args.quiet_ibinsync))
    payload = _as_payload(st)

    # Write CSB-compliant runtime STATE with TTL
    path = Path(_runtime_path())
    written = write_runtime_state_json(path, payload, ttl_seconds=ttl_s, inject_ts=True)

    # Optional stdout JSON
    if args.json:
        print(stable_json_dumps(written))

    # Default: status-only exit=0
    if args.strict_exit and not bool(payload.get("ok", False)):
        return 2
    return 0


if __name__ == "__main__":
    from pathlib import Path
    from datetime import timezone  # needed for server_time conversion

    raise SystemExit(main())

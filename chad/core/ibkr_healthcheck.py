from __future__ import annotations

"""
chad/core/ibkr_healthcheck.py

IBKR healthcheck CLI for CHAD.

Purpose
-------
This module provides a small, production-grade health probe for the IBKR API
running behind IB Gateway on this host. It is designed to be:

- Safe: it DOES NOT place orders or modify account state.
- Lightweight: a single connect + reqCurrentTime() call, then disconnect.
- Env-driven: uses IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID.
- Machine-friendly: can emit JSON, suitable for systemd/alerts.

Usage
-----
From the CHAD venv:

    # Human-readable check (default)
    python -m chad.core.ibkr_healthcheck

    # JSON output for monitoring / scripts
    python -m chad.core.ibkr_healthcheck --json

Exit codes
----------
0  -> healthy (connected and reqCurrentTime() succeeded within latency threshold)
1  -> configuration or environment error (missing env, missing ib_insync, etc.)
2  -> connectivity timeout or failure to connect
3  -> request failure (reqCurrentTime() raised or returned invalid)
4  -> latency too high (if --max-latency-ms is provided and exceeded)

Environment
-----------
IBKR_HOST       (default "127.0.0.1")
IBKR_PORT       (default "4002")
IBKR_CLIENT_ID  (no default; REQUIRED)

This module does NOT look at execution mode (DRY_RUN vs LIVE). It is strictly
about connectivity and basic API responsiveness.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class IBKRHealthStatus:
    ok: bool
    error: Optional[str]
    host: str
    port: int
    client_id: int
    latency_ms: Optional[float]
    server_time_iso: Optional[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _env_or_default(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    if val is not None:
        return val
    if default is not None:
        return default
    raise RuntimeError(f"Missing required env var: {name}")


def _build_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CHAD IBKR healthcheck (connect + reqCurrentTime)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds (default: 10.0).",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=None,
        help="If set, fail healthcheck if latency exceeds this many milliseconds.",
    )
    return parser.parse_args(argv)


def _load_config() -> Dict[str, Any]:
    host = _env_or_default("IBKR_HOST", "127.0.0.1")
    port = int(_env_or_default("IBKR_PORT", "4002"))
    client_id = int(_env_or_default("IBKR_CLIENT_ID"))
    return {"host": host, "port": port, "client_id": client_id}


def _check_ibkr(config: Dict[str, Any], timeout: float) -> IBKRHealthStatus:
    try:
        from ib_insync import IB  # type: ignore[import]
    except ImportError as exc:
        return IBKRHealthStatus(
            ok=False,
            error=f"ib_insync not installed: {exc}",
            host=config["host"],
            port=config["port"],
            client_id=config["client_id"],
            latency_ms=None,
            server_time_iso=None,
        )

    ib = IB()
    start = time.monotonic()
    try:
        ib.connect(
            host=config["host"],
            port=config["port"],
            clientId=config["client_id"],
            timeout=timeout,
        )
    except Exception as exc:
        return IBKRHealthStatus(
            ok=False,
            error=f"connect_failed: {exc}",
            host=config["host"],
            port=config["port"],
            client_id=config["client_id"],
            latency_ms=None,
            server_time_iso=None,
        )

    try:
        server_time = ib.reqCurrentTime()
        latency_ms = (time.monotonic() - start) * 1000.0
        iso_ts = None
        try:
            iso_ts = server_time.isoformat()  # type: ignore[attr-defined]
        except Exception:
            iso_ts = str(server_time)

        return IBKRHealthStatus(
            ok=True,
            error=None,
            host=config["host"],
            port=config["port"],
            client_id=config["client_id"],
            latency_ms=latency_ms,
            server_time_iso=iso_ts,
        )
    except Exception as exc:
        return IBKRHealthStatus(
            ok=False,
            error=f"reqCurrentTime_failed: {exc}",
            host=config["host"],
            port=config["port"],
            client_id=config["client_id"],
            latency_ms=None,
            server_time_iso=None,
        )
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_args(argv)

    try:
        config = _load_config()
    except Exception as exc:
        status = IBKRHealthStatus(
            ok=False,
            error=f"config_error: {exc}",
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "4002")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "0") or 0),
            latency_ms=None,
            server_time_iso=None,
        )
        if args.json:
            print(status.to_json())
        else:
            print(f"[IBKR HEALTH] CONFIG ERROR: {status.error}", file=sys.stderr)
        return 1

    status = _check_ibkr(config=config, timeout=args.timeout)

    # Apply latency threshold if configured
    exit_code = 0
    if not status.ok:
        exit_code = 2
    elif args.max_latency_ms is not None and status.latency_ms is not None:
        if status.latency_ms > args.max_latency_ms:
            status.ok = False
            status.error = (
                f"latency_too_high: {status.latency_ms:.2f}ms > {args.max_latency_ms:.2f}ms"
            )
            exit_code = 4

    if args.json:
        print(status.to_json())
    else:
        if status.ok:
            print(
                "[IBKR HEALTH] OK - "
                f"host={status.host} port={status.port} client_id={status.client_id} "
                f"latency_ms={status.latency_ms:.2f} server_time={status.server_time_iso}"
            )
        else:
            print(
                "[IBKR HEALTH] ERROR - "
                f"host={status.host} port={status.port} client_id={status.client_id} "
                f"error={status.error}",
                file=sys.stderr,
            )

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

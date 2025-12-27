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


def _fmt_exc(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc!r}"


def _build_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHAD IBKR healthcheck (connect + reqCurrentTime).")
    p.add_argument("--json", action="store_true", help="Emit JSON output.")
    p.add_argument("--timeout", type=float, default=10.0, help="IB connect timeout seconds (default 10).")
    p.add_argument("--max-latency-ms", type=float, default=None, help="Fail if latency exceeds this ms.")
    p.add_argument("--quiet-ibinsync", action="store_true", default=True, help="Suppress ib_insync stderr noise.")
    return p.parse_args(argv)


def _load_config() -> Dict[str, Any]:
    host = _env_or_default("IBKR_HOST", "127.0.0.1")
    port = int(_env_or_default("IBKR_PORT", "4002"))
    client_id = int(_env_or_default("IBKR_CLIENT_ID"))
    return {"host": host, "port": port, "client_id": client_id}


def _check_ibkr(config: Dict[str, Any], *, timeout: float, quiet: bool) -> IBKRHealthStatus:
    try:
        from ib_insync import IB  # type: ignore[import]
        from ib_insync import ib as ib_mod  # type: ignore[import]
    except Exception as exc:
        return IBKRHealthStatus(
            ok=False,
            error=f"import_failed: {_fmt_exc(exc)}",
            host=config["host"],
            port=config["port"],
            client_id=config["client_id"],
            latency_ms=None,
            server_time_iso=None,
        )

    # Silence ib_insync stderr spam (e.g., "completed orders request timed out").
    stderr_buf: io.StringIO = io.StringIO()
    stderr_cm = contextlib.redirect_stderr(stderr_buf) if quiet else contextlib.nullcontext()

    # Patch IB.reqCompletedOrdersAsync during connect so connect doesn't stall ~10s waiting for it.
    IBClass = getattr(ib_mod, "IB", None)
    orig_req_completed = getattr(IBClass, "reqCompletedOrdersAsync", None)

    def _patched_req_completed_orders_async(self, apiOnly: bool = False):  # noqa: ANN001
        async def _noop():
            return []
        return _noop()

    start = time.monotonic()
    ib = IB()

    try:
        if IBClass is not None and callable(orig_req_completed):
            setattr(IBClass, "reqCompletedOrdersAsync", _patched_req_completed_orders_async)

        with stderr_cm:
            ib.connect(
                host=config["host"],
                port=config["port"],
                clientId=config["client_id"],
                timeout=float(timeout),
            )

        with stderr_cm:
            server_time = ib.reqCurrentTime()

        latency_ms = (time.monotonic() - start) * 1000.0
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
            error=f"healthcheck_failed: {_fmt_exc(exc)}",
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
        # Restore original method no matter what.
        try:
            if IBClass is not None and callable(orig_req_completed):
                setattr(IBClass, "reqCompletedOrdersAsync", orig_req_completed)
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_args(argv)

    try:
        config = _load_config()
    except Exception as exc:
        status = IBKRHealthStatus(
            ok=False,
            error=f"config_error: {_fmt_exc(exc)}",
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

    status = _check_ibkr(config=config, timeout=float(args.timeout), quiet=bool(args.quiet_ibinsync))

    exit_code = 0
    if not status.ok:
        exit_code = 2
    elif args.max_latency_ms is not None and status.latency_ms is not None:
        if status.latency_ms > float(args.max_latency_ms):
            status.ok = False
            status.error = f"latency_too_high: {status.latency_ms:.2f}ms > {float(args.max_latency_ms):.2f}ms"
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
                f"error={status.error}"
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

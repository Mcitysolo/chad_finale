#!/usr/bin/env python3
"""
CHAD — Kraken Futures Authenticated Smoke-Test Scaffold (Phase C Item 1C).

Scaffold only. This tool is NOT wired into strategies, execution routing,
live_loop, or any systemd unit. It exists so the operator can verify the
presence and basic shape of Kraken Futures private credentials without
risking any order placement.

Behavior
--------
- Default mode (no flags):
    * Detect credential presence.
    * If missing -> ``status=missing_credentials``, exit 2.
    * If present -> ``status=credentials_present_endpoint_not_certified``,
      exit 3, because no read-only private endpoint has been certified yet.

- ``--dry-run``:
    * Validate that the tool can import.
    * Construct a ``KrakenFuturesClient`` with ``dry_run=True`` and no creds.
    * Confirm the client reports no credentials and is in dry-run.
    * Print ``status=dry_run_ok``, exit 0.
    * No network activity.

- ``--live-readonly``:
    * Requires credentials. If missing -> exit 2 ``missing_credentials``.
    * Calls ``perform_private_readonly_probe(client)`` (injectable for
      testing). The default implementation refuses to make a speculative
      call against an uncertified private endpoint and returns
      ``{"status": "not_certified", ...}``.
    * On ``not_certified`` -> exit 3
      ``credentials_present_endpoint_not_certified``.
    * On a future certified ``ok`` result -> exit 0 ``live_readonly_ok``.
    * Never places orders. Never calls ``submit_order``.

Exit codes
----------
- 0  dry-run ok, or (eventually) certified live-readonly ok
- 1  unexpected error
- 2  missing credentials
- 3  endpoint not certified / not implemented
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Callable, Dict, Optional

from chad.exchanges.kraken_futures_client import (
    KrakenFuturesClient,
    load_credentials_from_env,
)

_PREFIX = "KRAKEN_FUTURES_AUTH_SMOKE"


def _emit(status: str, *, credentials_present: bool, dry_run: bool, live_readonly: bool) -> None:
    """Emit machine-readable status lines. No secrets."""
    print(f"{_PREFIX} status={status}")
    print(f"credentials_present={'true' if credentials_present else 'false'}")
    print(f"dry_run={'true' if dry_run else 'false'}")
    print(f"live_readonly={'true' if live_readonly else 'false'}")


def perform_private_readonly_probe(client: KrakenFuturesClient) -> Dict[str, Any]:
    """Default read-only probe placeholder.

    No Kraken Futures private read-only endpoint has been certified for use
    by this scaffold yet, so this function REFUSES to make a speculative
    network call. Tests and future operator-driven smoke runs may inject a
    different implementation once an endpoint is certified.
    """
    return {
        "status": "not_certified",
        "error": "kraken_futures_readonly_endpoint_not_certified",
        "endpoint": None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chad.tools.kraken_futures_auth_smoke",
        description=(
            "Read-only Kraken Futures authenticated smoke-test scaffold. "
            "Never places orders."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate construction and fail-closed behavior only. No network.",
    )
    parser.add_argument(
        "--live-readonly",
        action="store_true",
        help=(
            "Attempt a read-only private call. Requires credentials. Fails "
            "closed with exit 3 if no read-only endpoint is certified."
        ),
    )
    return parser


def run(
    argv: Optional[list] = None,
    *,
    probe: Callable[[KrakenFuturesClient], Dict[str, Any]] = perform_private_readonly_probe,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    dry_run_flag = bool(args.dry_run)
    live_readonly_flag = bool(args.live_readonly)

    if dry_run_flag and live_readonly_flag:
        _emit(
            "conflicting_flags",
            credentials_present=False,
            dry_run=dry_run_flag,
            live_readonly=live_readonly_flag,
        )
        return 1

    if dry_run_flag:
        client = KrakenFuturesClient(credentials=None, dry_run=True)
        if client.has_credentials() or not client.dry_run:
            _emit(
                "dry_run_construction_failed",
                credentials_present=client.has_credentials(),
                dry_run=client.dry_run,
                live_readonly=False,
            )
            return 1
        _emit(
            "dry_run_ok",
            credentials_present=False,
            dry_run=True,
            live_readonly=False,
        )
        return 0

    creds = load_credentials_from_env()
    credentials_present = creds is not None

    if not credentials_present:
        _emit(
            "missing_credentials",
            credentials_present=False,
            dry_run=False,
            live_readonly=live_readonly_flag,
        )
        return 2

    if not live_readonly_flag:
        # Credentials present but operator did not explicitly request a
        # live-readonly probe. Refuse to do anything speculative.
        _emit(
            "credentials_present_endpoint_not_certified",
            credentials_present=True,
            dry_run=False,
            live_readonly=False,
        )
        return 3

    client = KrakenFuturesClient(credentials=creds, dry_run=False)
    result = probe(client)
    status_val = str(result.get("status", "") or "")

    if status_val == "ok":
        _emit(
            "live_readonly_ok",
            credentials_present=True,
            dry_run=False,
            live_readonly=True,
        )
        return 0

    if status_val == "not_certified":
        _emit(
            "credentials_present_endpoint_not_certified",
            credentials_present=True,
            dry_run=False,
            live_readonly=True,
        )
        return 3

    _emit(
        "live_readonly_failed",
        credentials_present=True,
        dry_run=False,
        live_readonly=True,
    )
    return 1


def main(argv: Optional[list] = None) -> int:
    try:
        return run(argv)
    except SystemExit:
        raise
    except BaseException:
        # Print a redacted traceback so operators see *what* class of error
        # occurred without leaking secrets or stack data into logs.
        sys.stderr.write(
            f"{_PREFIX} unexpected_error type="
            f"{type(sys.exc_info()[1]).__name__}\n"
        )
        traceback.print_exc(limit=2, file=sys.stderr)
        _emit(
            "unexpected_error",
            credentials_present=False,
            dry_run=False,
            live_readonly=False,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

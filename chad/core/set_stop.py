from __future__ import annotations

import argparse

from chad.core.stop_state import STOP_TTL_SECONDS_DEFAULT, StopState, write_stop_state


def main(argv: list[str] | None = None) -> int:
    """
    Backwards-compatible STOP setter CLI.

    This module exists as a thin wrapper for operator convenience and legacy
    scripts. The SSOT implementation is chad.core.stop_state.

    Usage:
      python -m chad.core.set_stop --enable  --reason "panic"  [--ttl-seconds N]
      python -m chad.core.set_stop --disable --reason "resume" [--ttl-seconds N]
    """
    p = argparse.ArgumentParser(description="CHAD STOP toggle (DENY_ALL freeze).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--enable", action="store_true", help="Enable STOP (DENY_ALL).")
    g.add_argument("--disable", action="store_true", help="Disable STOP.")
    p.add_argument("--reason", required=True, help="Human reason (required).")
    p.add_argument(
        "--ttl-seconds",
        type=int,
        default=STOP_TTL_SECONDS_DEFAULT,
        help=f"TTL seconds for runtime/stop_state.json (default {STOP_TTL_SECONDS_DEFAULT}).",
    )
    args = p.parse_args(argv)

    enabled = bool(args.enable)
    reason = str(args.reason)
    ttl_seconds = int(args.ttl_seconds)

    payload = write_stop_state(stop=enabled, reason=reason, ttl_seconds=ttl_seconds)

    # Keep a friendly stdout contract (useful for systemd logs).
    st = StopState(stop=bool(payload.get("stop", enabled)), reason=str(payload.get("reason", reason)))
    print("OK: wrote runtime/stop_state.json")
    print(f"stop={st.stop} reason={st.reason!r} ttl_seconds={payload.get('ttl_seconds')} ts_utc={payload.get('ts_utc')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from chad.core.stop_state import StopState, save_stop_state


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CHAD STOP toggle (DENY_ALL freeze).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--enable", action="store_true", help="Enable STOP (DENY_ALL).")
    g.add_argument("--disable", action="store_true", help="Disable STOP.")
    p.add_argument("--reason", type=str, default="operator_stop", help="Audit reason.")
    args = p.parse_args(argv)

    enabled = bool(args.enable)
    state = StopState(stop=enabled, reason=str(args.reason), updated_at_utc=_utc_now_iso())
    save_stop_state(state)

    print("OK: wrote runtime/stop_state.json")
    print(f"stop={state.stop} reason={state.reason!r} updated_at_utc={state.updated_at_utc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

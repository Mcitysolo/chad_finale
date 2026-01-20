"""
CHAD Live Mode Setter (runtime/live_mode.json)

This module is the authoritative writer for runtime/live_mode.json.

Contract (must satisfy)
-----------------------
- Writes a TTL-stamped runtime file with top-level:
    live: bool
    reason: str
    ts_utc: str
    ttl_seconds: int
- Atomic write (tmp + fsync + replace)
- No secrets
- CLI is deterministic and safe-by-default

Notes
-----
This file is *operator intent only*. In Phase 7/DRY_RUN builds, adapters remain hard-locked to DRY_RUN
regardless of this intent. LiveGate combines this with CHAD_MODE, SCR, STOP, caps, broker health, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_TTL_SECONDS = 86400  # 24h
MAX_TTL_SECONDS = 7 * 86400  # 7d hard cap


@dataclass(frozen=True)
class LiveModeState:
    live: bool
    reason: str
    ts_utc: str
    ttl_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "live": bool(self.live),
            "reason": str(self.reason),
            "ts_utc": str(self.ts_utc),
            "ttl_seconds": int(self.ttl_seconds),
        }


class LiveModeError(RuntimeError):
    pass


def _repo_root() -> Path:
    """
    Resolve repo root deterministically from this file location.

    Expected path:
      <repo>/chad/core/set_live_mode.py
    """
    p = Path(__file__).resolve()
    root = p.parents[2]  # .../chad/core/file.py -> repo
    if not (root / "runtime").exists():
        raise LiveModeError(f"Cannot locate repo root runtime/ directory from: {p}")
    return root


def _runtime_path() -> Path:
    return _repo_root() / "runtime" / "live_mode.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_reason(reason: str) -> str:
    r = (reason or "").strip()
    if not r:
        raise LiveModeError("Reason is required and cannot be empty.")
    return r[:240]


def _normalize_ttl(ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        raise LiveModeError("ttl_seconds must be > 0.")
    if ttl_seconds > MAX_TTL_SECONDS:
        raise LiveModeError(f"ttl_seconds too large (max {MAX_TTL_SECONDS}).")
    return int(ttl_seconds)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise LiveModeError(f"Failed to parse JSON at {path}: {type(e).__name__}: {e}") from e


def load_state() -> Tuple[Optional[LiveModeState], Dict[str, Any]]:
    """
    Load current live_mode.json. Returns (typed_state_or_None, raw_dict).
    If file is missing or malformed, typed_state_or_None may be None.
    """
    p = _runtime_path()
    raw = _read_json(p) if p.exists() else {}
    if not raw:
        return None, {}
    try:
        st = LiveModeState(
            live=bool(raw.get("live", False)),
            reason=str(raw.get("reason", "")),
            ts_utc=str(raw.get("ts_utc", "")),
            ttl_seconds=int(raw.get("ttl_seconds", 0)),
        )
        return st, raw
    except Exception:
        return None, raw


def write_state(live: bool, reason: str, ttl_seconds: int) -> LiveModeState:
    st = LiveModeState(
        live=bool(live),
        reason=_normalize_reason(reason),
        ts_utc=_utc_now_iso(),
        ttl_seconds=_normalize_ttl(ttl_seconds),
    )
    _atomic_write_json(_runtime_path(), st.to_dict())
    return st


def _print_state(st: LiveModeState) -> None:
    print("=== CHAD Live Mode State ===")
    print(f"live   : {bool(st.live)}")
    print(f"reason : {st.reason}")
    print("")
    print("Notes:")
    print("  • In this Phase 7 build, adapters and execution remain hard-locked")
    print("    to DRY_RUN regardless of this flag.")
    print("  • This state reflects operator intent only and will be combined")
    print("    with CHAD_MODE, SCR, LiveGate, caps, and STOP flags in Phase 8.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHAD Live Mode Setter (operator intent only).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_show = sub.add_parser("show", help="Show current live-mode state.")

    s_live = sub.add_parser("live", help="Set live-mode intent to LIVE (still DRY_RUN in this build).")
    s_live.add_argument("--reason", default="operator_live", help="Audit reason (required, default provided).")
    s_live.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS, help="TTL seconds (max 7 days).")

    s_stop = sub.add_parser("stop", help="Set live-mode intent to STOP / DRY_RUN.")
    s_stop.add_argument("--reason", default="operator_stop", help="Audit reason (required, default provided).")
    s_stop.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS, help="TTL seconds (max 7 days).")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "show":
            st, raw = load_state()
            if st is None:
                if not raw:
                    print("=== CHAD Live Mode State ===")
                    print(f"path   : {_runtime_path()}")
                    print("live   : False")
                    print("reason : (missing file)")
                    return 0
                print("=== CHAD Live Mode State ===")
                print(f"path   : {_runtime_path()}")
                print("live   : False")
                print("reason : (malformed file)")
                return 0
            # If file exists, display using our standard view
            _print_state(st)
            return 0

        if args.cmd == "live":
            st = write_state(True, args.reason, args.ttl_seconds)
            print("Updated live-mode state (LIVE intent):")
            _print_state(st)
            return 0

        if args.cmd == "stop":
            st = write_state(False, args.reason, args.ttl_seconds)
            print("Updated live-mode state (STOP/DRY_RUN intent):")
            _print_state(st)
            return 0

        raise LiveModeError(f"Unknown command: {args.cmd}")

    except LiveModeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

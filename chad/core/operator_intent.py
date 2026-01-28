"""
CHAD Operator Intent (runtime/operator_intent.json)

Purpose
-------
This module provides a single, auditable, TTL-stamped operator intent file that LiveGate
consumes as a deny-by-default safety input.

Design constraints (non-negotiable)
-----------------------------------
- Atomic writes (tmp + fsync + replace) so readers never see partial JSON.
- TTL-stamped runtime file: top-level ts_utc + ttl_seconds.
- Strict allowlist for mode: EXIT_ONLY | ALLOW | DENY_ALL.
- No secrets, ever.
- CLI is deterministic and safe-by-default.

LiveGate compatibility
----------------------
LiveGate currently loads runtime/operator_intent.json and reads:
- operator_mode OR mode
- operator_reason OR reason

This writer uses the preferred keys:
- operator_mode
- operator_reason
and also includes:
- ts_utc
- ttl_seconds

Usage
-----
Show current operator intent:
  python -m chad.core.operator_intent show --json

Set EXIT_ONLY (safe default):
  python -m chad.core.operator_intent set --mode EXIT_ONLY --reason "warmup_exit_only" --ttl-seconds 86400

Set ALLOW:
  python -m chad.core.operator_intent set --mode ALLOW --reason "explicit_allow_live" --ttl-seconds 3600

Set DENY_ALL:
  python -m chad.core.operator_intent set --mode DENY_ALL --reason "emergency_deny_all" --ttl-seconds 3600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ALLOWED_MODES = ("EXIT_ONLY", "ALLOW", "DENY_ALL")
DEFAULT_MODE = "EXIT_ONLY"
DEFAULT_TTL_SECONDS = 86400  # 24h, operator-friendly default


@dataclass(frozen=True)
class OperatorIntentState:
    operator_mode: str
    operator_reason: str
    ts_utc: str
    ttl_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OperatorIntentError(RuntimeError):
    pass


def _repo_root() -> Path:
    """
    Resolve repo root deterministically from this file location.

    Expected path:
      <repo>/chad/core/operator_intent.py
    """
    p = Path(__file__).resolve()
    # .../chad/core/operator_intent.py -> parents[0]=core, [1]=chad, [2]=repo
    root = p.parents[2]
    if not (root / "runtime").exists():
        # Do not guess further; fail loudly.
        raise OperatorIntentError(f"Cannot locate repo root runtime/ directory from: {p}")
    return root


def _runtime_dir() -> Path:
    return _repo_root() / "runtime"


def _state_path() -> Path:
    return _runtime_dir() / "operator_intent.json"


def _utc_now_iso() -> str:
    # ISO 8601 UTC with Z suffix, microsecond precision
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_mode(mode: str) -> str:
    m = (mode or "").strip().upper()
    if m not in ALLOWED_MODES:
        raise OperatorIntentError(f"Invalid mode {mode!r}. Allowed: {', '.join(ALLOWED_MODES)}")
    return m


def _normalize_reason(reason: str) -> str:
    r = (reason or "").strip()
    if not r:
        raise OperatorIntentError("Reason is required and cannot be empty.")
    # Keep it short to avoid bloating traces/logs
    return r[:240]


def _normalize_ttl(ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        raise OperatorIntentError("ttl_seconds must be > 0.")
    # Hard safety: prevent absurdly large TTLs that hide stale intent indefinitely.
    # 7 days max. If you want longer, re-apply intent periodically (auditable).
    if ttl_seconds > 7 * 86400:
        raise OperatorIntentError("ttl_seconds too large (max 604800 = 7 days).")
    return int(ttl_seconds)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Atomic JSON write:
      write tmp -> fsync -> replace
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"
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
        raise OperatorIntentError(f"Failed to parse JSON at {path}: {type(e).__name__}: {e}") from e


def load_state() -> Tuple[Optional[OperatorIntentState], Dict[str, Any]]:
    """
    Load and validate current state file.

    Returns:
      (typed_state_or_None, raw_dict)
    """
    p = _state_path()
    raw = _read_json(p) if p.exists() else {}
    if not raw:
        return None, {}
    mode = raw.get("operator_mode", raw.get("mode", DEFAULT_MODE))
    reason = raw.get("operator_reason", raw.get("reason", ""))
    ts_utc = raw.get("ts_utc")
    ttl = raw.get("ttl_seconds")
    try:
        st = OperatorIntentState(
            operator_mode=_normalize_mode(str(mode)),
            operator_reason=_normalize_reason(str(reason)),
            ts_utc=str(ts_utc),
            ttl_seconds=int(ttl),
        )
        return st, raw
    except Exception:
        # If the file exists but is malformed, do not crash callers; return raw for debugging.
        return None, raw


def write_state(mode: str, reason: str, ttl_seconds: int) -> OperatorIntentState:
    st = OperatorIntentState(
        operator_mode=_normalize_mode(mode),
        operator_reason=_normalize_reason(reason),
        ts_utc=_utc_now_iso(),
        ttl_seconds=_normalize_ttl(ttl_seconds),
    )
    _atomic_write_json(_state_path(), st.to_dict())
    return st


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHAD operator intent writer (TTL-stamped runtime state).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_show = sub.add_parser("show", help="Print current operator intent state.")
    s_show.add_argument("--json", action="store_true", help="Print JSON only (default).")
    s_show.add_argument("--raw", action="store_true", help="If file is malformed, print raw JSON too.")

    s_set = sub.add_parser("set", help="Set operator intent (writes runtime/operator_intent.json).")
    s_set.add_argument(
        "--mode",
        required=True,
        choices=list(ALLOWED_MODES),
        help="Operator intent mode. EXIT_ONLY is safest; ALLOW permits entries if all other gates are green.",
    )
    s_set.add_argument("--reason", required=True, help="Required human reason for audit trail.")
    s_set.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS, help="TTL in seconds (max 7 days).")
    s_set.add_argument("--quiet", action="store_true", help="Do not print the state JSON to stdout.")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "show":
            st, raw = load_state()
            if st is not None:
                payload = st.to_dict()
                print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
                return 0
            # No state file or malformed
            if not raw:
                print(json.dumps({"present": False, "path": str(_state_path())}, separators=(",", ":"), sort_keys=True))
                return 0
            # Malformed file
            out: Dict[str, Any] = {"present": True, "path": str(_state_path()), "parsed_ok": False}
            if args.raw:
                out["raw"] = raw
            print(json.dumps(out, separators=(",", ":"), sort_keys=True))
            return 0

        if args.cmd == "set":
            st = write_state(args.mode, args.reason, args.ttl_seconds)
            if not args.quiet:
                print(json.dumps(st.to_dict(), separators=(",", ":"), sort_keys=True))
            return 0

        raise OperatorIntentError(f"Unknown command: {args.cmd}")

    except OperatorIntentError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""
CHAD Shadow State Writer (runtime/shadow_state.json)

Why this exists
---------------
LiveGate consumes "shadow state" as a safety input (SCR / paper-only gating).
In your repo, LiveGate's fallback path expects a runtime JSON with keys:
  - state
  - sizing_factor
  - paper_only
  - reasons
and it must be TTL-stamped with top-level:
  - ts_utc
  - ttl_seconds

This module is the authoritative writer for runtime/shadow_state.json and provides a
deterministic CLI to set safe defaults and record an audit reason.

Non-negotiables
---------------
- Atomic writes (tmp + fsync + replace)
- TTL-stamped runtime state (ts_utc + ttl_seconds)
- No secrets
- Safe-by-default semantics (paper_only=True, WARMUP state)

CLI
---
Show current shadow state:
  python -m chad.core.shadow_state show

Set safe default WARMUP paper-only (recommended baseline):
  python -m chad.core.shadow_state set --state WARMUP --paper-only true --sizing-factor 0.1 \
    --reason "baseline_shadow_state" --ttl-seconds 3600

Mark confident and allow live (still gated by other systems; only if explicitly intended):
  python -m chad.core.shadow_state set --state CONFIDENT --paper-only false --sizing-factor 1.0 \
    --reason "scr_confident" --ttl-seconds 900
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_TTL_SECONDS = 3600  # 1h default for shadow state freshness
MAX_TTL_SECONDS = 7 * 86400  # hard cap: 7 days

ALLOWED_STATES = ("WARMUP", "CONFIDENT", "CAUTIOUS", "PAUSED", "UNKNOWN")


@dataclass(frozen=True)
class ShadowState:
    state: str
    sizing_factor: float
    paper_only: bool
    reasons: List[str]
    ts_utc: str
    ttl_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "sizing_factor": self.sizing_factor,
            "paper_only": self.paper_only,
            "reasons": list(self.reasons),
            "ts_utc": self.ts_utc,
            "ttl_seconds": self.ttl_seconds,
        }


class ShadowStateError(RuntimeError):
    pass


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    root = p.parents[2]
    if not (root / "runtime").exists():
        raise ShadowStateError(f"Cannot locate repo root runtime/ directory from: {p}")
    return root


def _path() -> Path:
    return _repo_root() / "runtime" / "shadow_state.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_state(state: str) -> str:
    s = (state or "").strip().upper()
    if s not in ALLOWED_STATES:
        raise ShadowStateError(f"Invalid state {state!r}. Allowed: {', '.join(ALLOWED_STATES)}")
    return s


def _normalize_sizing_factor(x: float) -> float:
    try:
        v = float(x)
    except Exception as e:  # noqa: BLE001
        raise ShadowStateError(f"sizing_factor must be float: {e}") from e
    # Hard safety bounds
    if not (0.0 <= v <= 2.0):
        raise ShadowStateError("sizing_factor must be within [0.0, 2.0].")
    return v


def _normalize_bool(s: str) -> bool:
    v = (s or "").strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ShadowStateError("paper_only must be a boolean string: true/false")


def _normalize_reason(reason: str) -> str:
    r = (reason or "").strip()
    if not r:
        raise ShadowStateError("reason is required and cannot be empty.")
    return r[:240]


def _normalize_ttl(ttl_seconds: int) -> int:
    if ttl_seconds <= 0:
        raise ShadowStateError("ttl_seconds must be > 0.")
    if ttl_seconds > MAX_TTL_SECONDS:
        raise ShadowStateError(f"ttl_seconds too large (max {MAX_TTL_SECONDS}).")
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
        raise ShadowStateError(f"Failed to parse JSON at {path}: {type(e).__name__}: {e}") from e


def load_state() -> Tuple[Optional[ShadowState], Dict[str, Any]]:
    p = _path()
    raw = _read_json(p) if p.exists() else {}
    if not raw:
        return None, {}
    try:
        st = ShadowState(
            state=str(raw.get("state", "UNKNOWN")).upper(),
            sizing_factor=float(raw.get("sizing_factor", 0.1)),
            paper_only=bool(raw.get("paper_only", True)),
            reasons=[str(x) for x in (raw.get("reasons", []) or [])][:20],
            ts_utc=str(raw.get("ts_utc", "")),
            ttl_seconds=int(raw.get("ttl_seconds", 0)),
        )
        return st, raw
    except Exception:
        return None, raw


def write_state(state: str, sizing_factor: float, paper_only: bool, reason: str, ttl_seconds: int) -> ShadowState:
    st = ShadowState(
        state=_normalize_state(state),
        sizing_factor=_normalize_sizing_factor(sizing_factor),
        paper_only=bool(paper_only),
        reasons=[_normalize_reason(reason)],
        ts_utc=_utc_now_iso(),
        ttl_seconds=_normalize_ttl(ttl_seconds),
    )
    _atomic_write_json(_path(), st.to_dict())
    return st


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHAD shadow state writer (TTL-stamped runtime state).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="Show current shadow_state.json (if present).")

    s_set = sub.add_parser("set", help="Set shadow state (writes runtime/shadow_state.json).")
    s_set.add_argument("--state", required=True, choices=list(ALLOWED_STATES))
    s_set.add_argument("--sizing-factor", required=True, type=float)
    s_set.add_argument("--paper-only", required=True, help="true/false")
    s_set.add_argument("--reason", required=True)
    s_set.add_argument("--ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.cmd == "show":
            st, raw = load_state()
            if st is None:
                if not raw:
                    print(json.dumps({"present": False, "path": str(_path())}, separators=(",", ":"), sort_keys=True))
                    return 0
                print(json.dumps({"present": True, "path": str(_path()), "parsed_ok": False, "raw": raw},
                                 separators=(",", ":"), sort_keys=True))
                return 0
            print(json.dumps(st.to_dict(), separators=(",", ":"), sort_keys=True))
            return 0

        if args.cmd == "set":
            paper_only = _normalize_bool(str(args.paper_only))
            st = write_state(
                state=str(args.state),
                sizing_factor=float(args.sizing_factor),
                paper_only=paper_only,
                reason=str(args.reason),
                ttl_seconds=int(args.ttl_seconds),
            )
            print(json.dumps(st.to_dict(), separators=(",", ":"), sort_keys=True))
            return 0

        raise ShadowStateError(f"Unknown command: {args.cmd}")

    except ShadowStateError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

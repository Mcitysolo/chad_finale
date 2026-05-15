"""Phase B Item 5 — entry-only futures roll gate reader.

Reads ``runtime/futures_roll_state.json`` (published by
``chad.market_data.futures_roll_publisher``) and reports whether new entries
for a given symbol should be blocked because the contract is inside its roll
warning window.

Fail-open by design:
  - Missing, stale, or unreadable file -> never block.
  - Symbol not in the file -> never block.
  - Symbol marked ``roll_supported=False`` -> never block.

Only ``block_new_entries=True`` combined with ``roll_supported=True`` triggers
a block, and only ever for new entries (callers must not invoke this on the
exit / stop-loss path).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_RUNTIME_DIR: Path = Path("/home/ubuntu/chad_finale/runtime")
ROLL_FILENAME: str = "futures_roll_state.json"
ROLL_FILE_TTL: int = 172800


@dataclass(frozen=True)
class RollGateResult:
    blocked: bool
    days_to_expiry: Optional[int]
    roll_pattern: Optional[str]
    roll_supported: bool
    block_reason: Optional[str]


def _fail_open() -> RollGateResult:
    return RollGateResult(
        blocked=False,
        days_to_expiry=None,
        roll_pattern=None,
        roll_supported=False,
        block_reason=None,
    )


def _parse_iso_z(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = ts.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_stale(payload: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    ttl = payload.get("ttl_seconds")
    try:
        ttl_s = int(ttl) if ttl is not None else ROLL_FILE_TTL
    except Exception:
        ttl_s = ROLL_FILE_TTL
    if ttl_s <= 0:
        return True
    ts = _parse_iso_z(str(payload.get("ts_utc") or ""))
    if ts is None:
        return True
    cur = now or datetime.now(timezone.utc)
    age = (cur - ts).total_seconds()
    return age > ttl_s


def _opt_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def check_roll_gate(
    symbol: str,
    *,
    runtime_dir: Optional[Path] = None,
) -> RollGateResult:
    """Return the entry-only roll-gate decision for ``symbol``.

    Fail-open on any error path. Never hard-blocks unsupported symbols.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return _fail_open()

    rd = runtime_dir if runtime_dir is not None else DEFAULT_RUNTIME_DIR
    path = Path(rd) / ROLL_FILENAME

    try:
        if not path.is_file():
            return _fail_open()
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return _fail_open()
    if not isinstance(payload, dict):
        return _fail_open()
    if _is_stale(payload):
        return _fail_open()

    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return _fail_open()
    rec = symbols.get(sym)
    if not isinstance(rec, dict):
        return _fail_open()

    roll_supported = bool(rec.get("roll_supported", False))
    roll_pattern = rec.get("roll_pattern")
    roll_pattern_str: Optional[str] = (
        str(roll_pattern) if isinstance(roll_pattern, str) and roll_pattern else None
    )
    days_to_expiry = _opt_int(rec.get("days_to_expiry"))

    if not roll_supported:
        return RollGateResult(
            blocked=False,
            days_to_expiry=days_to_expiry,
            roll_pattern=roll_pattern_str,
            roll_supported=False,
            block_reason=None,
        )

    block = bool(rec.get("block_new_entries", False)) and roll_supported
    return RollGateResult(
        blocked=block,
        days_to_expiry=days_to_expiry,
        roll_pattern=roll_pattern_str,
        roll_supported=True,
        block_reason="ROLL_WARNING_WINDOW" if block else None,
    )


__all__ = [
    "ROLL_FILE_TTL",
    "RollGateResult",
    "check_roll_gate",
]

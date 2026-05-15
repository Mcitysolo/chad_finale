"""Phase B Item 5 — futures contract roll calendar publisher.

Publishes ``runtime/futures_roll_state.json`` with a conservative, fail-open
contract roll calendar. v1 actively gates only audited quarterly equity-index
micro futures (MES, MNQ, MYM, M2K). All other known and unknown symbols are
emitted as informational ``unsupported_v1`` records with
``block_new_entries=False``.

The publisher performs no broker I/O, no external API calls, and no strategy
imports. The calendar is fully static (third-Friday quarterly months).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION: str = "futures_roll_state.v1"
DEFAULT_TTL_SECONDS: int = 86400
ROLL_WARNING_DAYS: int = 5
ROLL_CRITICAL_DAYS: int = 2

DEFAULT_RUNTIME_DIR: Path = Path("/home/ubuntu/chad_finale/runtime")
RUNTIME_FILENAME: str = "futures_roll_state.json"

_SUPPORTED_QUARTERLY_EQUITY_INDEX: frozenset[str] = frozenset(
    {"MES", "MNQ", "MYM", "M2K"}
)
_UNSUPPORTED_V1: frozenset[str] = frozenset(
    {"MCL", "MGC", "ZN", "ZB", "M6E"}
)

# Default publish universe — superset of the alpha_futures active universe so
# that operators see roll state for symbols that may be added later. Strategy
# config is NOT imported here to avoid any import-time side effects.
_DEFAULT_SYMBOLS: Tuple[str, ...] = (
    "MES", "MNQ", "MCL", "MGC", "MYM", "M2K", "M6E", "ZN", "ZB",
)

_QUARTERLY_MONTHS: Tuple[int, ...] = (3, 6, 9, 12)


def third_friday(year: int, month: int) -> date:
    """Return the third-Friday date of (year, month)."""
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + timedelta(days=days_to_friday)
    return first_friday + timedelta(weeks=2)


def next_quarterly_expiry(today: date) -> date:
    """Return the next quarterly third-Friday on or after ``today``."""
    candidates: List[date] = []
    for y in (today.year, today.year + 1):
        for m in _QUARTERLY_MONTHS:
            candidates.append(third_friday(y, m))
    for c in candidates:
        if c >= today:
            return c
    return candidates[-1]


def _quarterly_expiry_after(expiry: date) -> date:
    """Return the quarterly third-Friday strictly after ``expiry``."""
    return next_quarterly_expiry(expiry + timedelta(days=1))


def _unsupported_record() -> Dict[str, Any]:
    return {
        "current_expiry": None,
        "days_to_expiry": None,
        "roll_warning": False,
        "roll_critical": False,
        "roll_pattern": "unsupported_v1",
        "roll_supported": False,
        "next_expiry": None,
        "block_new_entries": False,
    }


def build_symbol_record(symbol: str, today: date) -> Dict[str, Any]:
    """Build the per-symbol roll record.

    Supported quarterly equity-index micros get a real third-Friday calendar
    with a roll-warning window. All other symbols return an informational
    ``unsupported_v1`` record that never blocks entries.
    """
    sym = (symbol or "").strip().upper()
    if sym not in _SUPPORTED_QUARTERLY_EQUITY_INDEX:
        return _unsupported_record()

    current = next_quarterly_expiry(today)
    nxt = _quarterly_expiry_after(current)
    days_to_expiry = (current - today).days
    roll_warning = days_to_expiry <= ROLL_WARNING_DAYS
    roll_critical = days_to_expiry <= ROLL_CRITICAL_DAYS

    return {
        "current_expiry": current.isoformat(),
        "days_to_expiry": int(days_to_expiry),
        "roll_warning": bool(roll_warning),
        "roll_critical": bool(roll_critical),
        "roll_pattern": "quarterly_3rd_friday",
        "roll_supported": True,
        "next_expiry": nxt.isoformat(),
        "block_new_entries": bool(roll_warning),
    }


def build_payload(
    symbols: Optional[List[str]] = None,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Build the full publisher payload."""
    if symbols is None:
        sym_list: List[str] = list(_DEFAULT_SYMBOLS)
    else:
        sym_list = [str(s).strip().upper() for s in symbols if str(s).strip()]

    as_of = today if today is not None else datetime.now(timezone.utc).date()

    records: Dict[str, Dict[str, Any]] = {}
    supported_count = 0
    unsupported_count = 0
    roll_warning_count = 0
    roll_critical_count = 0
    blocked_symbols: List[str] = []

    for sym in sym_list:
        rec = build_symbol_record(sym, as_of)
        records[sym] = rec
        if rec.get("roll_supported"):
            supported_count += 1
        else:
            unsupported_count += 1
        if rec.get("roll_warning"):
            roll_warning_count += 1
        if rec.get("roll_critical"):
            roll_critical_count += 1
        if rec.get("block_new_entries"):
            blocked_symbols.append(sym)

    return {
        "schema_version": SCHEMA_VERSION,
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ttl_seconds": DEFAULT_TTL_SECONDS,
        "symbols": records,
        "summary": {
            "symbols_tracked": len(records),
            "supported_count": supported_count,
            "unsupported_count": unsupported_count,
            "roll_warning_count": roll_warning_count,
            "roll_critical_count": roll_critical_count,
            "blocked_symbols": sorted(blocked_symbols),
        },
        "source": {
            "provider": "static_cme_calendar",
            "provider_status": "real",
            "scope": "equity_index_micro_quarterly_v1",
        },
        "status": "ok",
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def publish(
    runtime_dir: Path,
    dry_run: bool = False,
    today: Optional[date] = None,
    symbols: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Build payload and (optionally) write it atomically.

    Returns ``(payload, wrote)`` where ``wrote`` is False on dry-run.
    """
    payload = build_payload(symbols=symbols, today=today)
    wrote = False
    if not dry_run:
        out_path = Path(runtime_dir) / RUNTIME_FILENAME
        _atomic_write_json(out_path, payload)
        wrote = True
    return payload, wrote


def _parse_iso_date(raw: str) -> date:
    return datetime.strptime(raw.strip(), "%Y-%m-%d").date()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Publish runtime/futures_roll_state.json (Phase B Item 5).",
    )
    ap.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help="Runtime directory to write futures_roll_state.json into.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build payload and print to stdout; do not write the runtime file.",
    )
    ap.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Override 'today' for deterministic testing (YYYY-MM-DD).",
    )
    args = ap.parse_args(argv)

    today: Optional[date] = None
    if args.as_of_date:
        try:
            today = _parse_iso_date(args.as_of_date)
        except Exception as exc:
            print(f"invalid --as-of-date: {exc}", file=sys.stderr)
            return 2

    payload, _wrote = publish(
        runtime_dir=args.runtime_dir,
        dry_run=bool(args.dry_run),
        today=today,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_TTL_SECONDS",
    "ROLL_WARNING_DAYS",
    "ROLL_CRITICAL_DAYS",
    "third_friday",
    "next_quarterly_expiry",
    "build_symbol_record",
    "build_payload",
    "publish",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())

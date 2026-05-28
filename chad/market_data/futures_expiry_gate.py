"""Futures expiry gate (FUTURES-ROLL-1).

Bar-provider gate that skips polling expired futures contracts.

The audit observed ~588 Error 162 ("HMDS query returned no data: SILK6@COMEX
Trades") per day after the Silver futures front month (SILK6) expired
2026-05-27 but the bar-provider universe iterator continued to poll it.

This module:
  * Reads ``runtime/futures_roll_state.json`` (schema futures_roll_state.v1).
  * For each futures symbol in the bar-provider universe, decides whether the
    contract is expired *for trading bars today*. If it is, returns a verdict
    telling the caller to skip + (optionally) use a mapped active contract.
  * Tracks a per-process "one warning per (symbol, session) day" log dedupe.

It is pure-read; never writes to ``runtime/``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
ROLL_STATE_PATH = RUNTIME_DIR / "futures_roll_state.json"


@dataclass
class ExpiryVerdict:
    """One verdict per (symbol, day) gate decision."""

    symbol: str
    skip: bool
    reason: str
    current_expiry: str | None
    mapped_symbol: str | None  # if a roll target is published (audit §4 §5.2 acceptance)
    next_expiry: str | None
    roll_state_present: bool


_WARN_LOCK = threading.Lock()
_WARNED_KEYS: set[tuple[str, str]] = set()


def reset_warning_dedupe() -> None:
    """Test hook — clears the per-session warning dedupe set."""
    with _WARN_LOCK:
        _WARNED_KEYS.clear()


def should_warn_once(symbol: str, day_iso: str) -> bool:
    """Return True iff this (symbol, day) has not been warned yet in this
    process. Marks the key as warned on the first True return.
    """
    key = (symbol.upper(), day_iso)
    with _WARN_LOCK:
        if key in _WARNED_KEYS:
            return False
        _WARNED_KEYS.add(key)
        return True


def _load_roll_state(path: Path = ROLL_STATE_PATH) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_iso_date(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def evaluate_symbol(
    symbol: str,
    *,
    roll_state: dict[str, Any] | None = None,
    now: datetime | None = None,
    roll_path: Path = ROLL_STATE_PATH,
) -> ExpiryVerdict:
    """Decide whether to skip polling a futures symbol.

    Logic:
      * If no entry in roll_state: warn-and-skip ("no roll mapping").
      * If ``current_expiry`` is in the past (strictly < today UTC): skip with
        reason ``expired``. If ``next_expiry`` is present, surface it as the
        proposed roll target (informational only; caller decides whether to
        substitute).
      * If ``current_expiry`` is today and the market session for that
        contract is over: same handling as expired (today's polling will
        generate Error 162). The gate is conservative — it skips on the
        expiry day to avoid intraday spam.
      * Otherwise: do not skip.
    """
    now = now or datetime.now(timezone.utc)
    today = now.date()
    if roll_state is None:
        roll_state = _load_roll_state(roll_path)
    symbols = roll_state.get("symbols") or {} if isinstance(roll_state, dict) else {}
    has_state = bool(symbols)
    entry = symbols.get(symbol.upper()) if isinstance(symbols, dict) else None

    if not isinstance(entry, dict):
        return ExpiryVerdict(
            symbol=symbol.upper(),
            skip=False,           # absent entry is not authoritative for skip
            reason="no_roll_mapping",
            current_expiry=None,
            mapped_symbol=None,
            next_expiry=None,
            roll_state_present=has_state,
        )

    current_expiry = entry.get("current_expiry") if isinstance(entry.get("current_expiry"), str) else None
    next_expiry = entry.get("next_expiry") if isinstance(entry.get("next_expiry"), str) else None
    block_new = bool(entry.get("block_new_entries"))

    parsed = _parse_iso_date(current_expiry)
    if parsed is None:
        # roll_pattern unsupported or expiry not declared: don't skip, but
        # caller may still log this once.
        return ExpiryVerdict(
            symbol=symbol.upper(),
            skip=False,
            reason="expiry_unknown",
            current_expiry=current_expiry,
            mapped_symbol=None,
            next_expiry=next_expiry,
            roll_state_present=has_state,
        )

    if parsed.date() < today:
        return ExpiryVerdict(
            symbol=symbol.upper(),
            skip=True,
            reason="expired",
            current_expiry=current_expiry,
            mapped_symbol=None,  # roll-mapping substitution is operator-domain
            next_expiry=next_expiry,
            roll_state_present=has_state,
        )

    if parsed.date() == today:
        # Expiry day: skip if block_new_entries is set OR we are past the
        # standard CME session close (futures trade overnight; the gate is
        # conservative and treats the entire expiry day as "skip").
        return ExpiryVerdict(
            symbol=symbol.upper(),
            skip=True,
            reason="expires_today",
            current_expiry=current_expiry,
            mapped_symbol=None,
            next_expiry=next_expiry,
            roll_state_present=has_state,
        )

    return ExpiryVerdict(
        symbol=symbol.upper(),
        skip=False,
        reason="not_expired",
        current_expiry=current_expiry,
        mapped_symbol=None,
        next_expiry=next_expiry,
        roll_state_present=has_state,
    )


def filter_universe(
    futures_universe: list[str],
    *,
    roll_state: dict[str, Any] | None = None,
    now: datetime | None = None,
    roll_path: Path = ROLL_STATE_PATH,
    log_callback=None,
) -> tuple[list[str], list[ExpiryVerdict]]:
    """Return (kept_symbols, skip_verdicts).

    Bar provider calls this once per polling cycle. ``log_callback(verdict)``
    is invoked only on the first session-day occurrence of a skip per symbol
    (warning dedupe), so the journal does not flood with one log per cycle.
    """
    now = now or datetime.now(timezone.utc)
    day_iso = now.strftime("%Y-%m-%d")
    kept: list[str] = []
    skipped: list[ExpiryVerdict] = []
    if roll_state is None:
        roll_state = _load_roll_state(roll_path)
    for sym in futures_universe:
        v = evaluate_symbol(sym, roll_state=roll_state, now=now, roll_path=roll_path)
        if v.skip:
            skipped.append(v)
            if log_callback is not None and should_warn_once(sym, day_iso):
                try:
                    log_callback(v)
                except Exception:
                    pass
        else:
            kept.append(sym.upper())
    return kept, skipped

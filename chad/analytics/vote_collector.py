"""
chad/analytics/vote_collector.py

Phase-8 Session 5 (S1): signal stacking via cross-cycle intent voting.

Intents are accumulated in a per-process queue keyed by (symbol, side).
An intent is released when, inside the rolling window, at least
`min_votes` DISTINCT signal_family values have submitted the same
(symbol, side) pair. Otherwise the intent waits for the window to
expire, at which point it is discarded.

Default min_votes = 1. With that default every submit() returns the
intent immediately — no behavior change over the pre-Session-5 pipeline.
Operators can edit config/signal_stacking_config.json to raise the
threshold once enough strategies tag their intents with signal_family.

Safety:
  * In-memory state only — a process restart discards pending intents
    (no stale votes after a deploy).
  * Never raises on malformed input — failures degrade to "allow through".
  * Flush is idempotent: calling it with no pending intents is a no-op.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "signal_stacking_config.json"

DEFAULT_MIN_VOTES: int = 1
DEFAULT_WINDOW_SECONDS: int = 60


@dataclass
class _PendingIntent:
    intent: Any
    submitted_at: float
    family: str


def _intent_key(intent: Any) -> Tuple[str, str]:
    symbol = getattr(intent, "symbol", None) or getattr(intent, "pair", None) or ""
    side = getattr(intent, "side", "") or ""
    # Normalize side case — IBKR uses "BUY"/"SELL", Kraken uses "buy"/"sell".
    side = str(side).upper()
    return str(symbol).upper(), side


def _intent_family(intent: Any) -> str:
    fam = getattr(intent, "signal_family", "") or ""
    return str(fam or "unknown").lower()


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Return the signal-stacking config, or defaults on any error."""
    defaults = {
        "min_votes": DEFAULT_MIN_VOTES,
        "window_seconds": DEFAULT_WINDOW_SECONDS,
    }
    if not path.is_file():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(data, dict):
        return defaults
    out = dict(defaults)
    for k in ("min_votes", "window_seconds"):
        if k in data:
            try:
                out[k] = int(data[k])
            except (TypeError, ValueError):
                continue
    return out


class VoteCollector:
    """Thread-safe collector for cross-cycle signal stacking.

    With min_votes=1, submit() is effectively pass-through — the intent
    is released on first submission and behavior matches the pre-S5
    pipeline exactly.
    """

    def __init__(
        self,
        min_votes: int = DEFAULT_MIN_VOTES,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.min_votes = max(1, int(min_votes))
        self.window_seconds = max(1, int(window_seconds))
        self._lock = threading.Lock()
        # {(symbol, side): [PendingIntent, ...]}
        self._pending: Dict[Tuple[str, str], List[_PendingIntent]] = {}

    # ------------------------------------------------------------------
    # submit / flush
    # ------------------------------------------------------------------

    def submit(self, intent: Any, now: Optional[float] = None) -> List[Any]:
        """Submit an intent and return the list of intents ready to ship.

        * Below quorum → returns [] (intent is held internally).
        * At/above quorum → returns all pending intents for this (symbol,
          side) key and clears that bucket.
        """
        if intent is None:
            return []
        ts = float(now) if now is not None else time.time()
        key = _intent_key(intent)
        family = _intent_family(intent)
        entry = _PendingIntent(intent=intent, submitted_at=ts, family=family)

        with self._lock:
            bucket = self._pending.setdefault(key, [])
            # Purge this bucket's expired entries in the same pass so stale
            # votes never count toward a fresh quorum.
            bucket[:] = [
                p for p in bucket if (ts - p.submitted_at) <= self.window_seconds
            ]
            bucket.append(entry)

            unique_families = {p.family for p in bucket}
            if len(unique_families) >= self.min_votes:
                released = [p.intent for p in bucket]
                # Remove the bucket entirely on release so later submissions
                # start a fresh quorum.
                self._pending.pop(key, None)
                return released
            return []

    def flush_expired(self, now: Optional[float] = None) -> List[Any]:
        """Drop intents whose window has expired; return the dropped set.

        Returned intents are NOT released for execution — the caller
        should treat them as 'expired' and log accordingly. Releasing
        them would defeat the purpose of the vote threshold.
        """
        ts = float(now) if now is not None else time.time()
        dropped: List[Any] = []
        with self._lock:
            for key, bucket in list(self._pending.items()):
                fresh = [p for p in bucket if (ts - p.submitted_at) <= self.window_seconds]
                expired = [p for p in bucket if (ts - p.submitted_at) > self.window_seconds]
                if expired:
                    dropped.extend(p.intent for p in expired)
                if fresh:
                    self._pending[key] = fresh
                else:
                    self._pending.pop(key, None)
        return dropped

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._pending.values())

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy) — most callers use get_default_collector().
# ---------------------------------------------------------------------------

_DEFAULT: Optional[VoteCollector] = None


def get_default_collector() -> VoteCollector:
    global _DEFAULT
    if _DEFAULT is None:
        cfg = load_config()
        _DEFAULT = VoteCollector(
            min_votes=cfg.get("min_votes", DEFAULT_MIN_VOTES),
            window_seconds=cfg.get("window_seconds", DEFAULT_WINDOW_SECONDS),
        )
    return _DEFAULT


def reset_default_collector() -> None:
    """Test helper — rebuilds the singleton with fresh config."""
    global _DEFAULT
    _DEFAULT = None


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_MIN_VOTES",
    "DEFAULT_WINDOW_SECONDS",
    "VoteCollector",
    "get_default_collector",
    "load_config",
    "reset_default_collector",
]

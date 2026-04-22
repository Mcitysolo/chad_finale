"""
chad/analytics/slippage_tracker.py

Phase-8 Session 3 (E3): per-trade slippage ledger + rolling stats.

Slippage is recorded as a signed per-share value adjusted for side so that
positive slippage always represents an adverse move against the trader:

    BUY:  slippage_per_share = fill_price - expected_price
    SELL: slippage_per_share = expected_price - fill_price

A positive number therefore means the fill was worse than the expectation,
regardless of side. The raw unsigned diff (fill - expected) is retained in
the ledger for downstream tooling that wants the non-side-adjusted value.

Per-fill records are appended to data/slippage/SLIPPAGE_YYYYMMDD.ndjson.
Rolling stats are maintained in-memory (per process) and can be queried
by symbol and/or strategy.

Degradation: if expected_price is 0.0 (the intent-schema default, meaning
the strategy did not populate it), the record is written with slip=None
and rolling stats are not updated. This matches the survey's requirement
that missing expected_price never crashes the tracker.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
SLIPPAGE_DIR = ROOT / "data" / "slippage"

DEFAULT_ROLLING_WINDOW = 200


def _ymd(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%d")


def _iso_z(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


class SlippageTracker:
    """Record per-fill slippage and expose rolling stats by symbol/strategy.

    Thread-safe for a single process; cross-process coordination is not
    needed because fills are serialized by the executor.
    """

    def __init__(
        self,
        ledger_dir: Optional[Path] = None,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ) -> None:
        self._ledger_dir = Path(ledger_dir) if ledger_dir is not None else SLIPPAGE_DIR
        self._window = int(max(1, rolling_window))
        self._lock = threading.Lock()
        # keyed by (symbol, strategy); deque bounded by _window
        self._rolling: Dict[tuple, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        # per-symbol and per-strategy marginals
        self._by_symbol: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._by_strategy: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        self._all: Deque[float] = deque(maxlen=self._window)

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        strategy: str,
        side: str,
        expected_price: float,
        fill_price: float,
        quantity: float,
        intent_id: str = "",
        fill_timestamp: str = "",
    ) -> Dict[str, Any]:
        """Record a single fill and update rolling stats.

        Returns the ledger record that was appended (useful for callers
        that want to inline-log it). If expected_price is 0.0 the record
        is still written but with slippage_per_share=None.
        """
        sym = (symbol or "UNKNOWN").upper()
        strat = strategy or "unknown"
        side_upper = (side or "BUY").upper()
        exp = _safe_float(expected_price)
        fp = _safe_float(fill_price)
        qty = _safe_float(quantity)
        ts = fill_timestamp or _iso_z()

        if exp > 0.0 and fp > 0.0:
            raw_diff = fp - exp
            if side_upper in ("SELL", "SHORT"):
                slippage_per_share: Optional[float] = -raw_diff
            else:
                slippage_per_share = raw_diff
            slippage_bps = (
                (slippage_per_share / exp) * 10_000.0 if exp > 0 else None
            )
            total_slippage = (
                slippage_per_share * abs(qty) if qty != 0.0 else 0.0
            )
        else:
            raw_diff = None
            slippage_per_share = None
            slippage_bps = None
            total_slippage = None

        record: Dict[str, Any] = {
            "schema_version": "slippage.v1",
            "fill_timestamp": ts,
            "symbol": sym,
            "strategy": strat,
            "side": side_upper,
            "expected_price": exp,
            "fill_price": fp,
            "quantity": qty,
            "raw_price_diff": raw_diff,
            "slippage_per_share": slippage_per_share,
            "slippage_bps": slippage_bps,
            "total_slippage": total_slippage,
            "intent_id": intent_id or "",
        }

        self._append_to_ledger(record)

        if slippage_per_share is not None:
            with self._lock:
                self._rolling[(sym, strat)].append(slippage_per_share)
                self._by_symbol[sym].append(slippage_per_share)
                self._by_strategy[strat].append(slippage_per_share)
                self._all.append(slippage_per_share)

        return record

    # ------------------------------------------------------------------
    # stats
    # ------------------------------------------------------------------

    def get_rolling_stats(
        self,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return {n, mean, std, min, max} filtered by symbol/strategy.

        Filter precedence:
          - both symbol and strategy: exact (symbol, strategy) bucket
          - symbol only:   per-symbol marginal
          - strategy only: per-strategy marginal
          - neither:       global rolling deque
        last_n optionally caps the returned sample size (most recent).
        """
        with self._lock:
            if symbol and strategy:
                samples = list(self._rolling.get((symbol.upper(), strategy), ()))
            elif symbol:
                samples = list(self._by_symbol.get(symbol.upper(), ()))
            elif strategy:
                samples = list(self._by_strategy.get(strategy, ()))
            else:
                samples = list(self._all)

        if last_n is not None and last_n > 0:
            samples = samples[-int(last_n):]

        n = len(samples)
        if n == 0:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = statistics.fmean(samples)
        std = statistics.pstdev(samples) if n >= 2 else 0.0
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": min(samples),
            "max": max(samples),
        }

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _ledger_path(self, now: Optional[datetime] = None) -> Path:
        return self._ledger_dir / f"SLIPPAGE_{_ymd(now)}.ndjson"

    def _append_to_ledger(self, record: Dict[str, Any]) -> None:
        path = self._ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
                fh.write("\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
        except OSError:
            # Fail-open: the in-memory stats still update even if disk is read-only.
            return

    def read_ledger(self, day: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read back the ledger for a given YYYYMMDD (default: today)."""
        day = day or _ymd()
        path = self._ledger_dir / f"SLIPPAGE_{day}.ndjson"
        if not path.is_file():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


# Module-level singleton for production callers.
_DEFAULT_TRACKER: Optional[SlippageTracker] = None


def get_default_tracker() -> SlippageTracker:
    global _DEFAULT_TRACKER
    if _DEFAULT_TRACKER is None:
        _DEFAULT_TRACKER = SlippageTracker()
    return _DEFAULT_TRACKER


def record_fill(**kwargs: Any) -> Dict[str, Any]:
    """Module-level convenience forwarder to the default tracker."""
    return get_default_tracker().record_fill(**kwargs)


__all__ = [
    "SLIPPAGE_DIR",
    "DEFAULT_ROLLING_WINDOW",
    "SlippageTracker",
    "get_default_tracker",
    "record_fill",
]

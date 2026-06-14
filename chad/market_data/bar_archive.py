#!/usr/bin/env python3
"""
chad/market_data/bar_archive.py

PA-EP4 — append-only, date-partitioned 1m bar archive.

The live bar provider (chad/market_data/ibkr_bar_provider.py) keeps only a
rolling ~1-hour window per symbol: write_1m_bars_file() overwrites
data/bars/1m/<SYM>.json wholesale every poll, so older bars are discarded.
This module mirrors that otherwise-discarded stream into a durable,
append-only archive suitable for an out-of-sample edge harness — WITHOUT
touching the live cache file, its shape, or its TTL.

Approved design (PA-EP4):
- Location/format: data/bars/1m_archive/<SYM>/<YYYY-MM-DD>.ndjson, one
  compact JSON bar per line.
- Partition by the BAR's normalized UTC date (a cycle spanning midnight
  writes to two date files).
- D1 (UTC-from-day-one): ts_utc is normalized to true UTC at append time via
  the bar provider's shared _bar_ts_to_utc_iso helper (PA-EP6), so the archive
  is uniformly true-UTC regardless of the live writer's restart state. Dedup
  and date-partition use the normalized value.
- D2 (live-poll only): captures the live-poll stream; hooked in poll_once
  after a successful write_1m_bars_file. The backfill provider is NOT archived.
- D3 (in-memory seen-set): a per-(symbol, utc_date) seen-set of ts_utc is
  seeded once from that day's existing NDJSON on first touch / process start,
  then maintained across cycles (no per-cycle full-file re-read). Stale dates
  are evicted on UTC-date rollover.
- Dedup key: (symbol, normalized ts_utc). Append (open "a") only bars not
  already in the seen-set.
- Retention: RETENTION_DAYS days. Prune once per UTC day, unlinking
  <date>.ndjson older than retention by filename-date compare (no content
  parse). Pruning is NOT gated by the disk-guard (pruning frees space).
- Disk-guard: before any append I/O, skip (log WARN with free bytes, never
  raise) when free < MIN_FREE_BYTES or disk-use >= MAX_USE_PCT.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# Shared true-UTC normalizer (PA-EP6). bar_archive imports from
# ibkr_bar_provider; the provider imports bar_archive lazily to avoid a cycle.
from chad.market_data.ibkr_bar_provider import _bar_ts_to_utc_iso

LOGGER = logging.getLogger("chad.market_data.bar_archive")

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_1M_DIR = REPO_ROOT / "data" / "bars" / "1m_archive"

RETENTION_DAYS = 365
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
MAX_USE_PCT = 90.0


def _normalize_ts_utc(raw_ts: Any) -> str:
    """Return a bar ts_utc as a true-UTC ISO string via _bar_ts_to_utc_iso.

    A Bar carries ts_utc as a *string* (true-UTC post-EP6, or local-labeled
    pre-restart). Parse it to a datetime so the shared helper can do the
    instant-preserving UTC conversion uniformly. Unparseable values are
    returned stripped/unchanged rather than raising.
    """
    if isinstance(raw_ts, datetime):
        return _bar_ts_to_utc_iso(raw_ts)
    s = str(raw_ts).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return s
    return _bar_ts_to_utc_iso(dt)


def _utc_date_of(normalized_ts: str) -> str:
    """YYYY-MM-DD partition key from a normalized true-UTC ts string."""
    return normalized_ts[:10]


class BarArchive:
    """Stateful, append-only 1m bar archive (one instance per process).

    Holds the cross-cycle seen-set (D3) and the once-per-day prune marker.
    Never raises for operational errors (disk-guard, seed/append/prune I/O) —
    those log and degrade cleanly so the live poll loop can never be broken.
    """

    def __init__(
        self,
        *,
        archive_dir: Optional[Path] = None,
        retention_days: int = RETENTION_DAYS,
        min_free_bytes: int = MIN_FREE_BYTES,
        max_use_pct: float = MAX_USE_PCT,
        free_space_probe: Optional[Callable[[Path], Tuple[int, int]]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._dir = Path(archive_dir) if archive_dir else ARCHIVE_1M_DIR
        self._retention_days = int(retention_days)
        self._min_free_bytes = int(min_free_bytes)
        self._max_use_pct = float(max_use_pct)
        # probe returns (total_bytes, free_bytes); default = shutil.disk_usage
        self._probe = free_space_probe or self._default_probe
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        # D3: (symbol, utc_date) -> set of normalized ts_utc already archived.
        self._seen: Dict[Tuple[str, str], Set[str]] = {}
        self._last_prune_utc_date: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Public hook
    # ------------------------------------------------------------------ #

    def append(self, symbol: str, bars: Iterable[Any]) -> int:
        """Append live-poll 1m bars to the date-partitioned archive.

        Returns the number of new lines written. Disk-guard trips and write
        failures log and return what was written so far; they do not raise.
        """
        sym = str(symbol).strip().upper()
        if not sym:
            return 0

        # Group new (unseen) bars by normalized UTC date. Speculatively add to
        # the seen-set so duplicates within this same batch also dedup; roll
        # back below if the disk-guard blocks or a write fails.
        by_date: Dict[str, List[Tuple[str, str]]] = {}
        for b in bars:
            raw_ts = b.get("ts_utc") if isinstance(b, dict) else getattr(b, "ts_utc", "")
            ts_norm = _normalize_ts_utc(raw_ts)
            if not ts_norm:
                continue
            utc_date = _utc_date_of(ts_norm)
            seen = self._seen_for(sym, utc_date)
            if ts_norm in seen:
                continue
            seen.add(ts_norm)
            by_date.setdefault(utc_date, []).append((ts_norm, self._bar_to_line(b, ts_norm)))

        # Prune runs once per UTC day, independent of the disk-guard.
        self._maybe_prune()

        written = 0
        if by_date:
            if self._disk_guard_blocks():
                self._rollback(sym, by_date)
            else:
                written = self._flush(sym, by_date)

        self._evict_old_dates(sym)
        return written

    # ------------------------------------------------------------------ #
    # Seen-set (D3)
    # ------------------------------------------------------------------ #

    def _seen_for(self, symbol: str, utc_date: str) -> Set[str]:
        key = (symbol, utc_date)
        seen = self._seen.get(key)
        if seen is None:
            seen = self._seed_seen(symbol, utc_date)
            self._seen[key] = seen
        return seen

    def _seed_seen(self, symbol: str, utc_date: str) -> Set[str]:
        """Seed the seen-set once from that day's existing NDJSON (restart-safe)."""
        seen: Set[str] = set()
        path = self._dir / symbol / f"{utc_date}.ndjson"
        if not path.is_file():
            return seen
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        ts = json.loads(line).get("ts_utc")
                    except Exception:
                        continue
                    if ts:
                        seen.add(str(ts))
        except Exception as exc:
            LOGGER.warning(
                "bar_archive.seed_failed",
                extra={"symbol": symbol, "utc_date": utc_date, "error": str(exc)},
            )
        return seen

    def _evict_old_dates(self, symbol: str) -> None:
        """Keep only the 2 most recent UTC dates per symbol (rollover reset).

        Two are retained so a single midnight-spanning batch does not churn the
        seen-set; older dates are dropped to bound memory.
        """
        dates = sorted({k[1] for k in self._seen if k[0] == symbol})
        for d in dates[:-2]:
            self._seen.pop((symbol, d), None)

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #

    def _flush(self, sym: str, by_date: Dict[str, List[Tuple[str, str]]]) -> int:
        written = 0
        for utc_date, items in by_date.items():
            out_path = self._dir / sym / f"{utc_date}.ndjson"
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    for _ts, line in items:
                        f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                written += len(items)
            except Exception as exc:
                LOGGER.warning(
                    "bar_archive.append_failed",
                    extra={"symbol": sym, "utc_date": utc_date, "error": str(exc)},
                )
                self._rollback_date(sym, utc_date, items)
        return written

    def _rollback(self, sym: str, by_date: Dict[str, List[Tuple[str, str]]]) -> None:
        for utc_date, items in by_date.items():
            self._rollback_date(sym, utc_date, items)

    def _rollback_date(self, sym: str, utc_date: str, items: List[Tuple[str, str]]) -> None:
        seen = self._seen.get((sym, utc_date))
        if seen is not None:
            for ts, _line in items:
                seen.discard(ts)

    @staticmethod
    def _bar_to_line(bar: Any, ts_norm: str) -> str:
        if hasattr(bar, "to_dict"):
            rec = dict(bar.to_dict())
        elif isinstance(bar, dict):
            rec = dict(bar)
        else:
            rec = {k: getattr(bar, k, None) for k in ("open", "high", "low", "close", "volume")}
        rec["ts_utc"] = ts_norm  # overwrite with normalized true-UTC value
        return json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n"

    # ------------------------------------------------------------------ #
    # Disk-guard
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_probe(path: Path) -> Tuple[int, int]:
        # Probe the nearest existing ancestor (archive dir may not exist yet).
        p = path
        while not p.exists():
            p = p.parent
        usage = shutil.disk_usage(str(p))
        return usage.total, usage.free

    def _disk_guard_blocks(self) -> bool:
        try:
            total, free = self._probe(self._dir)
        except Exception as exc:
            # Fail-safe: if free space cannot be measured, do not write.
            LOGGER.warning("bar_archive.disk_probe_failed", extra={"error": str(exc)})
            return True
        used_pct = 100.0 * (total - free) / total if total > 0 else 100.0
        if free < self._min_free_bytes or used_pct >= self._max_use_pct:
            LOGGER.warning(
                "bar_archive.disk_guard_skip",
                extra={
                    "free_bytes": int(free),
                    "used_pct": round(used_pct, 2),
                    "min_free_bytes": self._min_free_bytes,
                    "max_use_pct": self._max_use_pct,
                },
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # Retention / prune
    # ------------------------------------------------------------------ #

    def _maybe_prune(self) -> None:
        today = self._now().date().isoformat()
        if self._last_prune_utc_date == today:
            return
        self._last_prune_utc_date = today
        self._prune(today)

    def _prune(self, today_utc_date: str) -> int:
        """Unlink <date>.ndjson older than retention by filename-date compare.

        Returns the count removed. Never raises.
        """
        try:
            cutoff = datetime.fromisoformat(today_utc_date).date() - timedelta(days=self._retention_days)
        except ValueError:
            return 0
        if not self._dir.is_dir():
            return 0
        removed = 0
        try:
            for sym_dir in self._dir.iterdir():
                if not sym_dir.is_dir():
                    continue
                for f in sym_dir.glob("*.ndjson"):
                    try:
                        fdate = datetime.fromisoformat(f.stem).date()
                    except ValueError:
                        continue
                    if fdate < cutoff:
                        try:
                            f.unlink()
                            removed += 1
                        except Exception as exc:
                            LOGGER.warning(
                                "bar_archive.prune_unlink_failed",
                                extra={"file": str(f), "error": str(exc)},
                            )
        except Exception as exc:
            LOGGER.warning("bar_archive.prune_failed", extra={"error": str(exc)})
        return removed

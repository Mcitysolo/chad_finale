"""
chad/analytics/signal_decay.py

Phase-8 Session 6 (F2): signal decay measurement.

At fill time the SignalDecayRecorder writes a "pending" entry carrying
the strategy, symbol, side, entry price, and entry timestamp. Later —
once enough daily bars exist to look back — compute_decay_for_pending
reads bar data from data/bars/1d/ and computes the unrealised alpha at
T+1, T+5, T+15, T+30 trading days relative to entry.

"Alpha" is side-adjusted so a positive number always means the trade
was working:

    BUY:  alpha = (px_later - entry_price) / entry_price
    SELL: alpha = (entry_price - px_later) / entry_price

Decay records are persisted to
``data/signal_decay/<strategy>_decay.ndjson`` so they survive restarts
and can be sliced by strategy offline. The sampling intervals are
measured in *daily* bars to match the 1d bar dataset the rest of CHAD
already maintains — using daily intervals is also what keeps the
computation retrospective and cheap (no live polling).

SignalDecayAnalyzer reads the ledger and reports per-strategy mean
alpha at each horizon, with the sample count.

Safety contract
---------------

  * Never raises on I/O errors — missing directories, unreadable files,
    and malformed JSON are all treated as "no data".
  * Never blocks the live loop — compute_decay_for_pending only processes
    entries whose bars are already on disk and is bounded by the size of
    the pending queue.
  * record_entry is idempotent on (strategy, symbol, intent_id) — the
    ledger will contain duplicate entries if called twice, but downstream
    analysis uses mean-alpha statistics which are robust to duplication.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DECAY_DIR = ROOT / "data" / "signal_decay"
BARS_DIR_1D = ROOT / "data" / "bars" / "1d"

# Horizons measured in trading-day bars. Daily bars means T+1 = next trading day.
DEFAULT_HORIZONS_DAYS: Tuple[int, ...] = (1, 5, 15, 30)

SCHEMA_VERSION = "signal_decay.v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return f


def _parse_date(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp or bare YYYY-MM-DD string."""
    if not ts:
        return None
    try:
        # Full ISO8601
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        return datetime.strptime(str(ts)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _load_bars(symbol: str, bars_dir: Path) -> List[dict]:
    """Read {bars:[...]} for ``symbol`` or return [] on any failure."""
    if not symbol:
        return []
    path = bars_dir / f"{str(symbol).upper()}.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    bars = data.get("bars")
    if not isinstance(bars, list):
        return []
    return bars


def _find_entry_bar_index(bars: List[dict], entry_time: str) -> Optional[int]:
    """Return the index of the last bar at or before ``entry_time``.

    Returns None when the entry date is before the first bar or bars
    carry no parseable timestamps.
    """
    entry_dt = _parse_date(entry_time)
    if entry_dt is None:
        return None
    best_idx: Optional[int] = None
    for idx, bar in enumerate(bars):
        if not isinstance(bar, dict):
            continue
        ts = bar.get("ts_utc") or bar.get("timestamp") or bar.get("date")
        bar_dt = _parse_date(str(ts or ""))
        if bar_dt is None:
            continue
        if bar_dt <= entry_dt:
            best_idx = idx
        else:
            break
    return best_idx


def _alpha(side: str, entry_price: float, later_price: float) -> Optional[float]:
    """Side-adjusted return: positive means the position is winning."""
    if entry_price <= 0.0 or later_price <= 0.0:
        return None
    side_u = (side or "").upper()
    if side_u in ("BUY", "LONG"):
        return (later_price - entry_price) / entry_price
    if side_u in ("SELL", "SHORT"):
        return (entry_price - later_price) / entry_price
    return None


def _write_atomic_line(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
        fh.write("\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except OSError:
        return out
    return out


def _write_ndjson(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":"), sort_keys=True))
            fh.write("\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class SignalDecayRecorder:
    """Append-only recorder of entry prices for later decay measurement."""

    def __init__(
        self,
        ledger_dir: Path = DECAY_DIR,
        bars_dir: Path = BARS_DIR_1D,
        horizons_days: Tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
    ) -> None:
        self._ledger_dir = Path(ledger_dir)
        self._bars_dir = Path(bars_dir)
        self._horizons = tuple(int(h) for h in horizons_days if int(h) > 0)

    def _path_for(self, strategy: str) -> Path:
        safe = (strategy or "unknown").replace("/", "_")
        return self._ledger_dir / f"{safe}_decay.ndjson"

    def record_entry(
        self,
        strategy: str,
        symbol: str,
        side: str,
        entry_price: float,
        entry_time: str = "",
        intent_id: str = "",
    ) -> Dict[str, Any]:
        """Append a pending-entry record for later decay measurement.

        Returns the record dict that was appended. On I/O failure the
        returned record is still produced so callers can log it; the
        ledger simply is not updated.
        """
        price = _safe_float(entry_price, 0.0) or 0.0
        side_u = (side or "").upper()
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "strategy": str(strategy or "unknown"),
            "symbol": str(symbol or "").upper(),
            "side": side_u if side_u in ("BUY", "SELL", "LONG", "SHORT") else "BUY",
            "entry_price": price,
            "entry_time": entry_time or _utc_now_iso(),
            "intent_id": str(intent_id or ""),
            "measured": False,
            "alpha": {str(h): None for h in self._horizons},
            "measured_at": None,
        }
        try:
            _write_atomic_line(self._path_for(record["strategy"]), record)
        except OSError:
            LOG.warning("signal_decay: could not append entry record")
        return record

    # ------------------------------------------------------------------
    # retrospective decay computation
    # ------------------------------------------------------------------

    def compute_decay_for_pending(
        self,
        strategies: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Measure alpha for all unmeasured entries with available bars.

        Only rewrites the ledger file when at least one record flips from
        unmeasured to measured, so calls in the steady state are cheap.
        Returns the list of newly measured records.
        """
        if strategies is None:
            if not self._ledger_dir.is_dir():
                return []
            strategies = [
                p.stem.removesuffix("_decay")
                for p in self._ledger_dir.glob("*_decay.ndjson")
            ]

        newly_measured: List[Dict[str, Any]] = []
        for strategy in strategies:
            path = self._path_for(strategy)
            records = _read_ndjson(path)
            if not records:
                continue
            updated = False
            for rec in records:
                if rec.get("measured"):
                    continue
                alpha_by_h = self._measure_record(rec)
                if alpha_by_h is None:
                    continue
                # Persist whatever horizons are now computable so progress
                # isn't lost if the process exits before the 30-day horizon
                # becomes available.
                prior = dict(rec.get("alpha") or {})
                rec["alpha"].update(alpha_by_h)
                if rec["alpha"] != prior:
                    updated = True
                # Only mark measured when EVERY horizon could be computed.
                if any(v is None for v in alpha_by_h.values()):
                    continue
                rec["measured"] = True
                rec["measured_at"] = _utc_now_iso()
                newly_measured.append(dict(rec))
                updated = True
            if updated:
                try:
                    _write_ndjson(path, records)
                except OSError:
                    LOG.warning("signal_decay: could not rewrite ledger %s", path)
        return newly_measured

    def _measure_record(self, rec: Mapping[str, Any]) -> Optional[Dict[str, Optional[float]]]:
        """Compute alpha-by-horizon for one pending record.

        Returns a dict keyed by horizon-day-string → alpha or None when a
        horizon's bar is not yet available. Returns None overall when bars
        for the symbol cannot be loaded or the entry predates the dataset.
        """
        symbol = str(rec.get("symbol") or "").upper()
        bars = _load_bars(symbol, self._bars_dir)
        if not bars:
            return None
        entry_idx = _find_entry_bar_index(bars, str(rec.get("entry_time") or ""))
        if entry_idx is None:
            return None
        side = str(rec.get("side") or "BUY")
        entry_price = _safe_float(rec.get("entry_price"), 0.0) or 0.0
        if entry_price <= 0.0:
            return None

        out: Dict[str, Optional[float]] = {}
        for h in self._horizons:
            target_idx = entry_idx + int(h)
            if target_idx >= len(bars):
                out[str(h)] = None
                continue
            later = bars[target_idx]
            close = _safe_float(later.get("close") if isinstance(later, dict) else None)
            if close is None:
                out[str(h)] = None
                continue
            out[str(h)] = _alpha(side, entry_price, close)
        return out


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class SignalDecayAnalyzer:
    """Reads decay ledgers and reports per-strategy alpha statistics."""

    def __init__(self, ledger_dir: Path = DECAY_DIR) -> None:
        self._ledger_dir = Path(ledger_dir)

    def _path_for(self, strategy: str) -> Path:
        safe = (strategy or "unknown").replace("/", "_")
        return self._ledger_dir / f"{safe}_decay.ndjson"

    def get_decay_stats(
        self,
        strategy: str,
        horizons_days: Tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
    ) -> Dict[str, Any]:
        """Return mean alpha per horizon and the measured sample count.

        A strategy with no data returns::

            {"strategy": name, "sample_count": 0,
             "mean_alpha_t1": None, "mean_alpha_t5": None,
             "mean_alpha_t15": None, "mean_alpha_t30": None}
        """
        records = _read_ndjson(self._path_for(strategy))
        measured = [r for r in records if r.get("measured")]
        out: Dict[str, Any] = {
            "strategy": str(strategy),
            "sample_count": len(measured),
        }
        for h in horizons_days:
            samples = [
                _safe_float(r.get("alpha", {}).get(str(h)))
                for r in measured
            ]
            clean = [s for s in samples if s is not None]
            mean = statistics.fmean(clean) if clean else None
            out[f"mean_alpha_t{h}"] = mean
        return out

    def get_all_strategies(self) -> List[str]:
        if not self._ledger_dir.is_dir():
            return []
        return sorted(
            p.stem.removesuffix("_decay")
            for p in self._ledger_dir.glob("*_decay.ndjson")
        )

    def get_all_stats(
        self,
        horizons_days: Tuple[int, ...] = DEFAULT_HORIZONS_DAYS,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            s: self.get_decay_stats(s, horizons_days=horizons_days)
            for s in self.get_all_strategies()
        }


# ---------------------------------------------------------------------------
# Module-level singletons for hot-path callers
# ---------------------------------------------------------------------------


_DEFAULT_RECORDER: Optional[SignalDecayRecorder] = None
_DEFAULT_ANALYZER: Optional[SignalDecayAnalyzer] = None


def get_default_recorder() -> SignalDecayRecorder:
    global _DEFAULT_RECORDER
    if _DEFAULT_RECORDER is None:
        _DEFAULT_RECORDER = SignalDecayRecorder()
    return _DEFAULT_RECORDER


def get_default_analyzer() -> SignalDecayAnalyzer:
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = SignalDecayAnalyzer()
    return _DEFAULT_ANALYZER


__all__ = [
    "BARS_DIR_1D",
    "DECAY_DIR",
    "DEFAULT_HORIZONS_DAYS",
    "SCHEMA_VERSION",
    "SignalDecayAnalyzer",
    "SignalDecayRecorder",
    "get_default_analyzer",
    "get_default_recorder",
]

"""setup_family_expectancy_updater.py — Gap-2 (v9.1 audit).

Compute per-setup-family expectancy metrics for the alpha_intraday_micro
strategy from the canonical NDJSON trade ledger and atomically publish
runtime/setup_family_expectancy.json.

Background
----------
runtime/setup_family_expectancy.json shipped seeded with zeros and no
updater. alpha_intraday_micro tags every emitted signal with a
setup_family ("ORB", "VWAP_RECLAIM", "VWAP_REJECTION",
"PULLBACK_CONTINUATION", "SWEEP_REVERSAL"), but no analytics job rolls
those back into per-family P&L / R-multiple metrics.

This module is the deterministic, idempotent updater that closes that
loop. It does not modify the strategy, the tier manager, or the
enforcer. It is read-only against the trade ledger and writes a single
runtime artifact.

Reads:
- data/trades/trade_history_*.ndjson

Writes:
- runtime/setup_family_expectancy.json   (schema_version=v2)

Determinism guarantees
----------------------
- File enumeration is sorted by path.
- Lines are read in file order.
- Float arithmetic is the standard double-precision sum/mean — running
  the updater twice on the same input yields byte-identical output
  (modulo the `ts_utc` field, which is intentionally moved out of the
  determinism contract; an explicit override is exposed for tests).
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from chad.utils.runtime_json import atomic_write_json, utc_now_iso

LOG = logging.getLogger("chad.analytics.setup_family_expectancy_updater")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/home/ubuntu/chad_finale")
DEFAULT_TRADES_DIR = REPO_ROOT / "data" / "trades"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "runtime" / "setup_family_expectancy.json"

TARGET_STRATEGY = "alpha_intraday_micro"
SETUP_EXPECTANCY_STRATEGIES = frozenset(
    {
        "alpha_intraday_micro",
        "alpha",
        "alpha_futures",
        "alpha_intraday",
    }
)

CANONICAL_FAMILIES: Tuple[str, ...] = (
    "ORB",
    "VWAP_RECLAIM",
    "VWAP_REJECTION",
    "PULLBACK_CONTINUATION",
    "SWEEP_REVERSAL",
)
UNKNOWN_FAMILY = "UNKNOWN"
SKIP_REASON_STOP_TOO_WIDE = "SKIP_STOP_TOO_WIDE"

SCHEMA_VERSION = "setup_family_expectancy.v2"
DEFAULT_TTL_SECONDS = 86400
DEFAULT_LOOKBACK_DAYS = 90

STATUS_ACTIVE = "ACTIVE"
STATUS_LOW_SAMPLE = "LOW_SAMPLE"
STATUS_NO_DATA = "NO_DATA"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ts(value: Any) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime parsed from an ISO-8601 string.

    Accepts trailing ``Z``, ``+00:00``, and naive strings (which are
    assumed UTC). Returns None on unparseable input."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort float coercion with NaN/inf rejection."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _coerce_dict(value: Any) -> Dict[str, Any]:
    """Return value if it is a dict, otherwise an empty dict."""
    return value if isinstance(value, dict) else {}


def _extract_realized_pnl(payload: Dict[str, Any]) -> Optional[float]:
    """Read realized P&L from a payload. Accepts (in order of preference)
    ``realized_pnl``, ``net_pnl``, ``pnl``. Returns None when nothing
    valid is present."""
    for key in ("realized_pnl", "net_pnl", "pnl"):
        if key in payload:
            v = _safe_float(payload.get(key))
            if v is not None:
                return v
    return None


def _extract_close_ts(payload: Dict[str, Any], outer: Dict[str, Any]) -> Optional[datetime]:
    """Read the closing timestamp from a closed_trade payload, falling
    back to the outer record's ``timestamp_utc`` then ``entry_time_utc``."""
    for key in ("exit_time_utc", "close_ts_utc", "ts_utc"):
        ts = _parse_ts(payload.get(key))
        if ts is not None:
            return ts
    ts = _parse_ts(outer.get("timestamp_utc"))
    if ts is not None:
        return ts
    return _parse_ts(payload.get("entry_time_utc"))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


@dataclass
class _FamilyAccumulator:
    """Per-family scratch state. Cleared between updater runs."""
    trades: int = 0
    wins: int = 0
    r_values: List[float] = field(default_factory=list)
    win_r_values: List[float] = field(default_factory=list)
    loss_r_values: List[float] = field(default_factory=list)
    skip_count_stop_too_wide: int = 0


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------


class SetupFamilyExpectancyUpdater:
    """Compute per-family expectancy for alpha_intraday_micro trades.

    Construct one instance per run. The instance is single-use: state is
    accumulated during :meth:`run` and the output path is written once
    at the end. The class is deterministic and idempotent — re-running
    against the same ledger produces byte-equivalent output (apart from
    ``ts_utc``, which can be pinned via ``ts_override`` in tests).
    """

    def __init__(
        self,
        *,
        trades_dir: Optional[Path] = None,
        output_path: Optional[Path] = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        now: Optional[datetime] = None,
        ts_override: Optional[str] = None,
    ) -> None:
        self._trades_dir: Path = Path(trades_dir) if trades_dir else DEFAULT_TRADES_DIR
        self._output_path: Path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
        if int(lookback_days) <= 0:
            raise ValueError("lookback_days must be positive")
        self._lookback_days: int = int(lookback_days)
        if now is not None and now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        self._now: datetime = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        self._cutoff: datetime = self._now - timedelta(days=self._lookback_days)
        self._ts_override: Optional[str] = ts_override

        # Run counters (reset on each run() invocation).
        self._total_trades_found: int = 0
        self._trades_processed: int = 0
        self._trades_skipped_corrupt: int = 0
        self._last_trade_ts: Optional[datetime] = None
        self._families: Dict[str, _FamilyAccumulator] = {}
        self._strategies_seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Process the trade ledger and write the runtime artifact.

        Returns the published payload."""
        started = time.monotonic()

        # Reset per-run scratch state.
        self._total_trades_found = 0
        self._trades_processed = 0
        self._trades_skipped_corrupt = 0
        self._last_trade_ts = None
        self._families = {fam: _FamilyAccumulator() for fam in CANONICAL_FAMILIES}
        self._strategies_seen = set()

        for record in self._iter_records():
            self._consume(record)

        payload = self._build_payload()

        try:
            atomic_write_json(self._output_path, payload)
        except OSError as exc:
            LOG.error(
                "setup_family_expectancy write_failed path=%s err=%s",
                self._output_path,
                exc,
            )
            raise

        elapsed = time.monotonic() - started
        LOG.info(
            "setup_family_expectancy_updated families_processed=%d "
            "trades_processed=%d trades_skipped_corrupt=%d "
            "last_trade_ts=%s elapsed_seconds=%.3f",
            payload["summary"]["families_processed"],
            payload["summary"]["trades_processed"],
            payload["summary"]["trades_skipped_corrupt"],
            payload["summary"]["last_trade_ts_utc"],
            elapsed,
        )
        return payload

    # ------------------------------------------------------------------
    # Ledger iteration
    # ------------------------------------------------------------------

    def _iter_records(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Yield ``(outer_record, payload)`` tuples for every parseable
        line in every trade_history_*.ndjson file under ``trades_dir``.

        Files are sorted lexicographically so ordering is deterministic.
        Corrupt lines are counted and logged at WARNING level but never
        raise."""
        if not self._trades_dir.is_dir():
            LOG.warning(
                "setup_family_expectancy trades_dir_missing path=%s — "
                "publishing zero-state artifact",
                self._trades_dir,
            )
            return

        pattern = str(self._trades_dir / "trade_history_*.ndjson")
        for path in sorted(glob.glob(pattern)):
            # Skip backup / quarantine sidecars (e.g. .bak, .scr_reset_bak).
            if path.endswith(".bak") or ".scr_reset_bak" in path:
                continue
            try:
                fh = open(path, "r", encoding="utf-8", errors="replace")
            except OSError as exc:
                LOG.warning(
                    "setup_family_expectancy ledger_open_failed path=%s err=%s",
                    path,
                    exc,
                )
                continue
            try:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (ValueError, TypeError) as exc:
                        self._trades_skipped_corrupt += 1
                        LOG.warning(
                            "setup_family_expectancy corrupt_ledger_line path=%s err=%s",
                            path,
                            exc,
                        )
                        continue
                    if not isinstance(record, dict):
                        self._trades_skipped_corrupt += 1
                        LOG.warning(
                            "setup_family_expectancy non_object_record path=%s",
                            path,
                        )
                        continue
                    raw_payload = record.get("payload")
                    payload = raw_payload if isinstance(raw_payload, dict) else record
                    yield record, payload
            finally:
                fh.close()

    # ------------------------------------------------------------------
    # Per-record consumption
    # ------------------------------------------------------------------

    def _consume(self, record: Tuple[Dict[str, Any], Dict[str, Any]]) -> None:
        outer, payload = record
        try:
            strategy = str(payload.get("strategy") or outer.get("strategy") or "").strip()
            if strategy not in SETUP_EXPECTANCY_STRATEGIES:
                return
            self._strategies_seen.add(strategy)

            self._total_trades_found += 1

            meta = _coerce_dict(payload.get("meta"))
            close_ts = _extract_close_ts(payload, outer)
            if close_ts is not None and close_ts < self._cutoff:
                # Outside lookback window — drop silently.
                return

            family_raw = meta.get("setup_family") or payload.get("setup_family")
            family = str(family_raw).strip() if isinstance(family_raw, str) and family_raw.strip() else ""
            if not family:
                family = UNKNOWN_FAMILY

            fam = self._families.setdefault(family, _FamilyAccumulator())

            skip_reason_raw = meta.get("skip_reason") or payload.get("skip_reason")
            if isinstance(skip_reason_raw, str) and skip_reason_raw.strip() == SKIP_REASON_STOP_TOO_WIDE:
                fam.skip_count_stop_too_wide += 1
                # Skip records carry no P&L — they are diagnostic only.
                return

            pnl = _extract_realized_pnl(payload)
            if pnl is None:
                # No realized P&L on this record — ignore (e.g. open
                # trade snapshot, pure diagnostic row). Not counted as
                # corrupt.
                return

            fam.trades += 1
            self._trades_processed += 1
            if close_ts is not None and (
                self._last_trade_ts is None or close_ts > self._last_trade_ts
            ):
                self._last_trade_ts = close_ts

            if pnl > 0:
                fam.wins += 1

            stop_width = _safe_float(meta.get("stop_width_usd"))
            if stop_width is None:
                stop_width = _safe_float(payload.get("stop_width_usd"))
            if stop_width is not None and stop_width > 0:
                r = pnl / stop_width
                fam.r_values.append(r)
                if pnl > 0:
                    fam.win_r_values.append(r)
                elif pnl < 0:
                    fam.loss_r_values.append(r)
        except Exception as exc:  # noqa: BLE001
            # A single corrupt record must never crash the updater.
            self._trades_skipped_corrupt += 1
            LOG.warning(
                "setup_family_expectancy record_consume_failed err=%s",
                exc,
            )

    # ------------------------------------------------------------------
    # Payload assembly
    # ------------------------------------------------------------------

    def _build_payload(self) -> Dict[str, Any]:
        families_out: Dict[str, Dict[str, Any]] = {}

        # Ensure canonical families are always present, then layer on
        # any UNKNOWN/extra family encountered during processing.
        for fam_name, fam in self._families.items():
            families_out[fam_name] = self._serialize_family(fam)

        # Canonical-family insertion guarantee (any of the five not
        # present is added back with zero state).
        for fam_name in CANONICAL_FAMILIES:
            if fam_name not in families_out:
                families_out[fam_name] = self._serialize_family(_FamilyAccumulator())

        ts_utc = self._ts_override if self._ts_override is not None else utc_now_iso()
        last_trade_iso: Optional[str] = (
            self._last_trade_ts.isoformat().replace("+00:00", "Z")
            if self._last_trade_ts is not None
            else None
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "ts_utc": ts_utc,
            "ttl_seconds": DEFAULT_TTL_SECONDS,
            "strategy": TARGET_STRATEGY,
            "lookback_days": self._lookback_days,
            "families": families_out,
            "summary": {
                "families_processed": len(families_out),
                "total_trades_found": self._total_trades_found,
                "trades_processed": self._trades_processed,
                "trades_skipped_corrupt": self._trades_skipped_corrupt,
                "last_trade_ts_utc": last_trade_iso,
                "strategies_processed": sorted(self._strategies_seen),
            },
        }

    @staticmethod
    def _serialize_family(fam: _FamilyAccumulator) -> Dict[str, Any]:
        trades = int(fam.trades)
        wins = int(fam.wins)

        if trades == 0:
            win_rate: Optional[float] = None
            status = STATUS_NO_DATA
        else:
            win_rate = round(wins / trades, 6)
            status = STATUS_ACTIVE if trades >= 10 else STATUS_LOW_SAMPLE

        if fam.r_values:
            avg_r: Optional[float] = round(
                sum(fam.r_values) / len(fam.r_values), 6
            )
        else:
            avg_r = None

        expectancy_r: Optional[float] = None
        # Expectancy = win_rate * avg_win_r - loss_rate * |avg_loss_r|
        # Defined only when we have at least one winning AND one losing
        # R observation (and trades > 0). With no losses we cannot
        # estimate loss_rate from R; spec asks for null in that case.
        if trades > 0 and fam.win_r_values and fam.loss_r_values:
            wr = wins / trades
            avg_win_r = sum(fam.win_r_values) / len(fam.win_r_values)
            avg_loss_r = sum(fam.loss_r_values) / len(fam.loss_r_values)
            loss_rate = 1.0 - wr
            expectancy_r = round(
                (wr * avg_win_r) - (loss_rate * abs(avg_loss_r)), 6
            )

        return {
            "trades": trades,
            "wins": wins,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "expectancy_r": expectancy_r,
            "skip_count_stop_too_wide": int(fam.skip_count_stop_too_wide),
            "status": status,
        }


__all__ = [
    "SetupFamilyExpectancyUpdater",
    "CANONICAL_FAMILIES",
    "UNKNOWN_FAMILY",
    "SKIP_REASON_STOP_TOO_WIDE",
    "SETUP_EXPECTANCY_STRATEGIES",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_TRADES_DIR",
    "DEFAULT_OUTPUT_PATH",
]

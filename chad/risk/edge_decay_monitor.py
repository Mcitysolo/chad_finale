"""
chad/risk/edge_decay_monitor.py

Phase-8 Session 5 (F4): edge decay auto-reduce.

If a strategy's most recent closed trades show `consecutive_threshold`
losses in a row AND it has accumulated at least `min_trades` overall,
its capital allocation is set to 0 in
`runtime/strategy_allocations.json`. A manual step (edit the file or
run scripts/clear_edge_decay.py) is required to un-halt.

The monitor is intentionally conservative:

  * min_trades (default 20) prevents halting strategies that simply
    haven't traded much yet.
  * Only trades with a *numeric* pnl are counted — records flagged as
    pnl_untrusted or historical_pre_rebuild are skipped (matches the
    exclusion list used by expectancy_tracker.py).
  * A single positive-pnl trade resets the streak back to zero.
  * Writes are atomic (tmp + rename); reads tolerate missing/malformed
    files (fail-open → no strategy appears halted).

Storage
-------
`runtime/strategy_allocations.json` uses this schema:

    {
      "schema_version": "strategy_allocations.v1",
      "updated_at": <ISO8601 UTC>,
      "allocations": {
         "<strategy>": {
            "halted": bool,
            "halt_reason": str,
            "halted_at": <ISO8601> | "",
            "cleared_at": <ISO8601> | null,
            "cleared_by": str,
            "consecutive_negative": int
         },
         ...
      }
    }

Only halted strategies appear in the dict. The live_loop's activation
filter treats any absent strategy as fully allowed, which matches the
G2 fail-open behavior.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRADES_GLOB = str(ROOT / "data" / "trades" / "trade_history_*.ndjson")
ALLOCATIONS_PATH = ROOT / "runtime" / "strategy_allocations.json"
CONFIG_PATH = ROOT / "config" / "edge_decay_config.json"

# Audit-O calibration (2026-04-22): consecutive_threshold default
# lowered 10→5 to match observed max streak of 3. Loaded at module
# import from config/edge_decay_config.json; falls back to the value
# below if the file is missing or malformed.
DEFAULT_CONSECUTIVE_THRESHOLD: int = 5
DEFAULT_MIN_TRADES: int = 20


def _load_config_values() -> tuple:
    """Return (consecutive_threshold, min_trades) from JSON config.

    Missing / malformed config returns the hardcoded defaults.
    """
    if not CONFIG_PATH.is_file():
        return DEFAULT_CONSECUTIVE_THRESHOLD, DEFAULT_MIN_TRADES
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_CONSECUTIVE_THRESHOLD, DEFAULT_MIN_TRADES
    if not isinstance(data, dict):
        return DEFAULT_CONSECUTIVE_THRESHOLD, DEFAULT_MIN_TRADES
    try:
        ct = int(data.get("consecutive_threshold", DEFAULT_CONSECUTIVE_THRESHOLD))
    except (TypeError, ValueError):
        ct = DEFAULT_CONSECUTIVE_THRESHOLD
    try:
        mt = int(data.get("min_trades", DEFAULT_MIN_TRADES))
    except (TypeError, ValueError):
        mt = DEFAULT_MIN_TRADES
    return ct, mt


SCHEMA_VERSION = "strategy_allocations.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Trade-history iteration (mirrors expectancy_tracker's exclusions)
# ---------------------------------------------------------------------------


def _load_quarantine_exclusion_sets() -> Tuple[Set[str], Set[str]]:
    """Return ``(invalid_fill_ids, invalid_trade_hashes)`` for quarantine
    filtering. Mirrors the trade_stats_engine._load_all_trades pattern so
    the same exclusion union (operator manifests + live untrusted-fill
    scan + sidecars) is honoured here. Fail-safe: any error returns
    empty sets so the monitor degrades to its pre-quarantine behaviour
    rather than crashing.
    """
    try:
        from chad.utils.quarantine import get_exclusion_sets

        return get_exclusion_sets()
    except Exception as exc:  # noqa: BLE001 — must never break the monitor
        LOG.warning("edge_decay_quarantine_load_failed err=%s — using empty sets", exc)
        return set(), set()


def _iter_trade_payloads(
    glob_pattern: str = TRADES_GLOB,
    invalid_fill_ids: Optional[Set[str]] = None,
    invalid_trade_hashes: Optional[Set[str]] = None,
) -> Iterable[dict]:
    """Yield trade payloads in chronological file order, oldest first.

    When ``invalid_fill_ids`` / ``invalid_trade_hashes`` are provided,
    records matching any of:
      * top-level ``record_hash`` ∈ invalid_trade_hashes
      * ``payload.fill_id`` ∈ invalid_fill_ids
      * any element of ``payload.fill_ids`` ∈ invalid_fill_ids
    are skipped before the existing pnl_untrusted / historical_pre_rebuild
    filters run. This keeps the streak counter from re-halting strategies
    whose quarantined evidence has already been excluded by SCR.
    """
    bad_fills = invalid_fill_ids or set()
    bad_hashes = invalid_trade_hashes or set()
    for path in sorted(glob.glob(glob_pattern)):
        if ".scr_reset_bak" in path or path.endswith(".bak"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    rh = rec.get("record_hash")
                    if isinstance(rh, str) and rh in bad_hashes:
                        continue
                    payload = rec.get("payload", rec)
                    if not isinstance(payload, dict):
                        continue
                    fid = payload.get("fill_id")
                    if isinstance(fid, str) and fid in bad_fills:
                        continue
                    fids = payload.get("fill_ids")
                    if isinstance(fids, list) and any(
                        isinstance(f, str) and f in bad_fills for f in fids
                    ):
                        continue
                    if payload.get("pnl_untrusted") is True:
                        continue
                    if payload.get("historical_pre_rebuild") is True:
                        continue
                    tags = payload.get("tags") or []
                    if isinstance(tags, list) and (
                        "pnl_untrusted" in tags or "historical_pre_rebuild" in tags
                    ):
                        continue
                    yield payload
        except OSError:
            continue


def _pnl_of(payload: Mapping[str, Any]) -> Optional[float]:
    v = payload.get("pnl")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_recent_trades_by_strategy(
    glob_pattern: str = TRADES_GLOB,
    invalid_fill_ids: Optional[Set[str]] = None,
    invalid_trade_hashes: Optional[Set[str]] = None,
) -> Dict[str, List[float]]:
    """Return {strategy: [pnl_oldest, ..., pnl_newest]} trimmed to a
    manageable tail per strategy (1000 trades is plenty for streak logic).

    When the quarantine sets are not supplied, they are loaded once from
    ``chad.utils.quarantine.get_exclusion_sets`` so the monitor cannot
    re-halt a strategy whose losing tail has already been quarantined
    by an operator manifest (e.g. alpha_options 2026-05-11 phantom BAG
    closes).
    """
    if invalid_fill_ids is None and invalid_trade_hashes is None:
        invalid_fill_ids, invalid_trade_hashes = _load_quarantine_exclusion_sets()
    per_strat: Dict[str, List[float]] = {}
    for payload in _iter_trade_payloads(
        glob_pattern,
        invalid_fill_ids=invalid_fill_ids,
        invalid_trade_hashes=invalid_trade_hashes,
    ):
        strat = str(payload.get("strategy") or "unknown")
        pnl = _pnl_of(payload)
        if pnl is None:
            continue
        per_strat.setdefault(strat, []).append(pnl)
    # Trim to keep memory bounded — newest 1000 are all we ever need.
    for k in list(per_strat.keys()):
        if len(per_strat[k]) > 1000:
            per_strat[k] = per_strat[k][-1000:]
    return per_strat


def count_consecutive_negative(pnls: List[float]) -> int:
    """Count trailing trades (newest first) whose pnl is strictly < 0."""
    streak = 0
    for pnl in reversed(pnls or []):
        if pnl < 0:
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# Allocations state
# ---------------------------------------------------------------------------


def read_allocations(path: Path = ALLOCATIONS_PATH) -> Dict[str, Any]:
    """Return the current allocation state or a safe empty default."""
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": "",
            "allocations": {},
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": "",
            "allocations": {},
        }
    if not isinstance(data, dict):
        return {"schema_version": SCHEMA_VERSION, "updated_at": "", "allocations": {}}
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("allocations", {})
    if not isinstance(data["allocations"], dict):
        data["allocations"] = {}
    return data


def is_strategy_halted(strategy: str, path: Path = ALLOCATIONS_PATH) -> bool:
    """True iff the allocations file records strategy as halted."""
    data = read_allocations(path)
    entry = data.get("allocations", {}).get(str(strategy))
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("halted", False))


def set_strategy_halted(
    strategy: str,
    consecutive_negative: int,
    path: Path = ALLOCATIONS_PATH,
) -> Dict[str, Any]:
    """Mark a strategy halted. Idempotent — re-halting updates counts."""
    state = read_allocations(path)
    allocations = state.get("allocations", {})
    allocations[str(strategy)] = {
        "halted": True,
        "halt_reason": f"consecutive_negative_{int(consecutive_negative)}",
        "halted_at": _utc_now_iso(),
        "cleared_at": None,
        "cleared_by": "",
        "consecutive_negative": int(consecutive_negative),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now_iso(),
        "allocations": allocations,
    }
    _write_atomic(path, payload)
    LOG.warning(
        "EDGE_DECAY_HALT strategy=%s consecutive_negative=%d path=%s",
        strategy, consecutive_negative, path,
    )
    return payload


def clear_strategy_halt(
    strategy: str,
    cleared_by: str = "operator",
    path: Path = ALLOCATIONS_PATH,
    clear_reason: str = "manual_operator_clear",
) -> Dict[str, Any]:
    """Remove the halt flag (or strategy entry) and record who cleared it.

    GAP-018 / NEW-GAP-051 halt-clear semantics: when an operator clears
    a halt, the persisted ``consecutive_negative`` counter is reset to 0
    so downstream readers of ``strategy_allocations.json`` cannot be
    misled by a stale pre-clear streak value. The pre-clear value is
    preserved as ``previous_consecutive_negative`` for the audit trail.
    Raw trade history in ``data/trades/`` is not touched; the next
    ``EdgeDecayMonitor.check_strategy`` pass recomputes the streak from
    the ledger, so any real (unquarantined) losing run will re-halt the
    strategy with a fresh count rather than silently being hidden.
    """
    state = read_allocations(path)
    allocations = state.get("allocations", {})
    existing = allocations.get(str(strategy))
    if isinstance(existing, dict):
        prior_streak = int(existing.get("consecutive_negative", 0) or 0)
        existing.update({
            "halted": False,
            "halt_reason": "",
            "cleared_at": _utc_now_iso(),
            "cleared_by": str(cleared_by or ""),
            "clear_reason": str(clear_reason or ""),
            "previous_consecutive_negative": prior_streak,
            "consecutive_negative": 0,
        })
        allocations[str(strategy)] = existing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now_iso(),
        "allocations": allocations,
    }
    _write_atomic(path, payload)
    LOG.warning(
        "EDGE_DECAY_CLEARED strategy=%s cleared_by=%s reason=%s prior_streak=%d",
        strategy,
        cleared_by,
        clear_reason,
        int((existing or {}).get("previous_consecutive_negative", 0) or 0)
        if isinstance(existing, dict) else 0,
    )
    return payload


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------


def _config_consecutive_threshold() -> int:
    return _load_config_values()[0]


def _config_min_trades() -> int:
    return _load_config_values()[1]


@dataclass
class EdgeDecayMonitor:
    """Per-strategy streak monitor.

    Usage (typical):

        monitor = EdgeDecayMonitor()
        results = monitor.check_all()
        for strat, verdict in results.items():
            if verdict["halted"]:
                logger.warning("strategy %s halted: %s", strat, verdict["reason"])

    Audit-O note (2026-04-22): consecutive_threshold / min_trades now
    default to values read from config/edge_decay_config.json via the
    default_factory lambdas below. Callers can still override with
    explicit kwargs.
    """

    consecutive_threshold: int = field(default_factory=_config_consecutive_threshold)
    min_trades: int = field(default_factory=_config_min_trades)
    allocations_path: Path = ALLOCATIONS_PATH
    trades_glob: str = TRADES_GLOB

    def check_strategy(
        self,
        strategy: str,
        pnls: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single strategy and persist a halt if warranted.

        `pnls` is optional — if omitted, the monitor reads the ledger itself.
        Tests pass a list explicitly to avoid filesystem dependence.
        """
        if pnls is None:
            ledger = collect_recent_trades_by_strategy(self.trades_glob)
            pnls = ledger.get(strategy, [])

        total = len(pnls)
        if total < int(self.min_trades):
            return {
                "halted": False,
                "reason": "insufficient_trades",
                "consecutive_neg": count_consecutive_negative(pnls),
                "total_trades": total,
            }

        streak = count_consecutive_negative(pnls)
        if streak >= int(self.consecutive_threshold):
            set_strategy_halted(
                strategy,
                consecutive_negative=streak,
                path=self.allocations_path,
            )
            return {
                "halted": True,
                "reason": f"consecutive_negative_{streak}",
                "consecutive_neg": streak,
                "total_trades": total,
            }

        return {
            "halted": False,
            "reason": "ok",
            "consecutive_neg": streak,
            "total_trades": total,
        }

    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate every strategy found in the trade ledger."""
        ledger = collect_recent_trades_by_strategy(self.trades_glob)
        results: Dict[str, Dict[str, Any]] = {}
        for strategy, pnls in ledger.items():
            results[strategy] = self.check_strategy(strategy, pnls=pnls)
        return results


__all__ = [
    "ALLOCATIONS_PATH",
    "DEFAULT_CONSECUTIVE_THRESHOLD",
    "DEFAULT_MIN_TRADES",
    "SCHEMA_VERSION",
    "EdgeDecayMonitor",
    "collect_recent_trades_by_strategy",
    "count_consecutive_negative",
    "clear_strategy_halt",
    "is_strategy_halted",
    "read_allocations",
    "set_strategy_halted",
]

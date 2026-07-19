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

# W1B-3 self-clear defaults (D3: ship the LOGIC + conservative in-code defaults
# as code; the config VALUES land as a Pending Action because
# config/edge_decay_config.json is governed strategy config -- see
# docs/PA_W1B3_edge_decay_config_v2.md). A v1 config (which lacks these keys)
# therefore runs unchanged on these defaults.
#   halt_ttl_days: auto-clear a monitor-imposed halt only after this many days
#     AND only when the trusted ledger shows no fresh losing evidence. <=0
#     disables TTL clears.
#   clear_on_recovery: when a ledger-resident halted strategy's trusted streak
#     falls below consecutive_threshold, persist the recovery (halted:False).
DEFAULT_HALT_TTL_DAYS: int = 14
DEFAULT_CLEAR_ON_RECOVERY: bool = True

# Provenance stamped on halts the edge-decay mechanism imposes (both the
# monitor and strategy_throttle_gate route through set_strategy_halted). Only
# halts carrying this provenance (or, for legacy records, the monitor's own
# consecutive_negative_* reason signature) may be auto-cleared -- operator
# manual halts are never touched (rider iv).
MONITOR_HALT_PROVENANCE: str = "edge_decay_monitor"


def _load_config_values() -> tuple:
    """Return (consecutive_threshold, min_trades, halt_ttl_days,
    clear_on_recovery) from JSON config.

    Missing / malformed config -- or a v1 config that predates the W1B-3
    self-clear keys -- falls back to the hardcoded defaults per key, so the
    self-clear logic ships and runs on in-code defaults regardless of the
    config file version.
    """
    ct, mt = DEFAULT_CONSECUTIVE_THRESHOLD, DEFAULT_MIN_TRADES
    ttl, cor = DEFAULT_HALT_TTL_DAYS, DEFAULT_CLEAR_ON_RECOVERY
    if not CONFIG_PATH.is_file():
        return ct, mt, ttl, cor
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ct, mt, ttl, cor
    if not isinstance(data, dict):
        return ct, mt, ttl, cor
    try:
        ct = int(data.get("consecutive_threshold", ct))
    except (TypeError, ValueError):
        pass
    try:
        mt = int(data.get("min_trades", mt))
    except (TypeError, ValueError):
        pass
    try:
        ttl = int(data.get("halt_ttl_days", ttl))
    except (TypeError, ValueError):
        pass
    raw_cor = data.get("clear_on_recovery", cor)
    if isinstance(raw_cor, bool):
        cor = raw_cor
    return ct, mt, ttl, cor


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 UTC timestamp (as written by ``_utc_now_iso``) back to
    an aware datetime, or ``None`` if absent/unparseable. A missing/unparseable
    ``halted_at`` is treated as "age unknown" so TTL never clears on it."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _is_monitor_imposed(entry: Mapping[str, Any]) -> bool:
    """Rider (iv): only halts the edge-decay mechanism imposed may auto-clear.

    Forward records carry ``halted_by == MONITOR_HALT_PROVENANCE``. Legacy
    records (written before W1B-3, no ``halted_by``) are inferred from the
    monitor's own ``consecutive_negative_*`` reason signature. Any explicit,
    non-monitor provenance (an operator manual halt) is protected.
    """
    hb = entry.get("halted_by")
    if isinstance(hb, str) and hb:
        return hb == MONITOR_HALT_PROVENANCE
    reason = str(entry.get("halt_reason") or "")
    return reason.startswith("consecutive_negative_")


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
        # W1B-3 provenance (rider iv): stamps this halt as edge-decay-imposed so
        # the self-clear path may release it. Operator manual halts never carry
        # this and are therefore protected.
        "halted_by": MONITOR_HALT_PROVENANCE,
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


def _emit_auto_clear_notification(
    strategy: str,
    *,
    reason: str,
    streak: int,
    total: int,
    ttl_days: int,
    halted_at: Any,
) -> None:
    """Rider (iii): every self-clear emits a greppable marker AND a
    coach-voiced operator NOTIFY, so an operator sees each release.

    Fully fail-soft: a notification/transport failure is logged and swallowed
    so it can never break the monitor (which runs every live-loop cycle). The
    marker is always emitted first, before the best-effort telegram push.
    """
    LOG.warning(
        "EDGE_DECAY_AUTO_CLEARED strategy=%s reason=%s streak=%d total=%d "
        "ttl_days=%s halted_at=%s cleared_by=edge_decay_monitor_auto",
        strategy, reason, int(streak), int(total), ttl_days, halted_at,
    )
    try:
        facts = {
            "title": f"Edge-decay halt lifted: {strategy}",
            "strategy": strategy,
            "reason": reason,
            "summary": (
                f"The automatic edge-decay halt on {strategy} has been "
                "released — the trusted trade ledger shows no fresh losing "
                "streak."
            ),
        }
        msg: Optional[str] = None
        try:
            from chad.utils.coach_voice import format_alert

            rendered = format_alert("edge_decay_cleared", facts)
            if isinstance(rendered, str) and rendered.strip():
                msg = rendered
        except Exception:  # noqa: BLE001 — presentation is optional
            msg = None
        if not msg:
            msg = (
                f"Edge-decay halt auto-cleared for {strategy} ({reason}); "
                "no fresh trusted losing evidence."
            )
        from chad.utils.telegram_notify import notify

        notify(
            msg,
            severity="info",
            dedupe_key=f"edge_decay_auto_clear:{strategy}",
            raise_on_fail=False,
        )
    except Exception as exc:  # noqa: BLE001 — NOTIFY is best-effort, never fatal
        LOG.warning(
            "edge_decay_auto_clear_notify_failed strategy=%s err=%s",
            strategy, exc,
        )


def _config_consecutive_threshold() -> int:
    return _load_config_values()[0]


def _config_min_trades() -> int:
    return _load_config_values()[1]


def _config_halt_ttl_days() -> int:
    return _load_config_values()[2]


def _config_clear_on_recovery() -> bool:
    return _load_config_values()[3]


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
    halt_ttl_days: int = field(default_factory=_config_halt_ttl_days)
    clear_on_recovery: bool = field(default_factory=_config_clear_on_recovery)
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

        W1B-3: the two non-halt branches (``insufficient_trades`` and ``ok``)
        now also run the self-clear path so a stale monitor-imposed halt can
        actually be released and persisted, rather than the historical
        in-memory-only ``halted:False`` that never reached disk.
        """
        if pnls is None:
            ledger = collect_recent_trades_by_strategy(self.trades_glob)
            pnls = ledger.get(strategy, [])

        total = len(pnls)
        streak = count_consecutive_negative(pnls)

        if total < int(self.min_trades):
            # Sparse / ledger-absent (e.g. alpha_crypto, whose fills are all
            # pnl_untrusted/validate_only and excluded from the trusted ledger):
            # eligible for a TTL-based clear once the halt is old enough and
            # there is no fresh trusted losing evidence.
            self._maybe_self_clear(strategy, streak=streak, total=total)
            return {
                "halted": False,
                "reason": "insufficient_trades",
                "consecutive_neg": streak,
                "total_trades": total,
            }

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

        # Ledger-resident and recovered (streak below threshold): eligible for a
        # recovery clear.
        self._maybe_self_clear(strategy, streak=streak, total=total)
        return {
            "halted": False,
            "reason": "ok",
            "consecutive_neg": streak,
            "total_trades": total,
        }

    def _maybe_self_clear(self, strategy: str, *, streak: int, total: int) -> Optional[str]:
        """Release a stale monitor-imposed halt when the TRUSTED ledger shows no
        fresh losing evidence. Persists ``halted:False`` (rider ii) and emits a
        marker + coach-voiced NOTIFY (rider iii). Returns the clear-reason if it
        cleared, else ``None``.

        Rider compliance:
          (i)  every decision uses only the trusted-ledger ``streak``/``total``
               (collect_recent_trades_by_strategy already excludes
               pnl_untrusted / validate_only / quarantined rows), so untrusted
               rows can neither keep nor clear a halt;
          (iv) only monitor-imposed halts (see ``_is_monitor_imposed``) are
               touched — operator manual halts are protected, and operator-
               *cleared* entries are ``halted:False`` so they short-circuit here.
        """
        state = read_allocations(self.allocations_path)
        entry = (state.get("allocations") or {}).get(str(strategy))
        if not isinstance(entry, dict) or not entry.get("halted"):
            return None  # not halted -> nothing to clear (also skips operator-cleared)
        if not _is_monitor_imposed(entry):
            return None  # rider (iv): operator-imposed halt is untouched
        # Fresh trusted losing evidence keeps the halt — never clear an active decay.
        if streak >= int(self.consecutive_threshold):
            return None

        resident = total >= int(self.min_trades)
        if resident:
            # Mechanism 2 — ledger-resident strategy has recovered.
            if not self.clear_on_recovery:
                return None
            reason = "auto_recovery_streak_below_threshold"
        else:
            # Mechanism 1 — sparse/absent: require TTL elapsed on a real
            # halted_at timestamp (a missing/unparseable stamp is "age unknown"
            # and never clears).
            ttl = int(self.halt_ttl_days)
            if ttl <= 0:
                return None
            halted_at = entry.get("halted_at")
            dt = _parse_iso(halted_at)
            if dt is None:
                return None
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
            if age_days <= float(ttl):
                return None
            reason = "auto_ttl_no_trusted_losing_evidence"

        clear_strategy_halt(
            strategy,
            cleared_by="edge_decay_monitor_auto",
            path=self.allocations_path,
            clear_reason=reason,
        )
        _emit_auto_clear_notification(
            strategy,
            reason=reason,
            streak=int(streak),
            total=int(total),
            ttl_days=int(self.halt_ttl_days),
            halted_at=entry.get("halted_at"),
        )
        return reason

    def check_all(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate every strategy in the trusted ledger UNION every strategy
        currently halted in the store.

        W1B-3 Mechanism 1: without the union, a strategy that is halted but
        absent from the trusted ledger (all of its fills excluded as
        pnl_untrusted/validate_only) is never re-evaluated, so its halt can
        never self-clear — the roach motel. Unioning the halted-store keys in
        lets ``check_strategy`` run the self-clear path for them each cycle.
        """
        ledger = collect_recent_trades_by_strategy(self.trades_glob)
        state = read_allocations(self.allocations_path)
        halted_in_store = {
            str(s)
            for s, v in (state.get("allocations") or {}).items()
            if isinstance(v, dict) and v.get("halted")
        }
        results: Dict[str, Dict[str, Any]] = {}
        for strategy in set(ledger.keys()) | halted_in_store:
            results[strategy] = self.check_strategy(strategy, pnls=ledger.get(strategy, []))
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

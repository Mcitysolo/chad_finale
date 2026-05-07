"""
chad/ops/strategy_routing_diagnostics.py

GAP-025: per-strategy routing diagnostics.

Writes runtime/strategy_routing_diagnostics.json each cycle so an operator
can answer "this strategy generated signals — why didn't it fill?". Pure
observation: this module never decides whether a signal proceeds.

Per-strategy fields:

    signals_generated_this_cycle      int | null
    signals_after_regime_gate         int | null
    signals_after_net_exposure_gate   int | null
    signals_after_spam_governor       int | null
    signals_after_strategy_throttle   int | null
    blocked_reasons                   {reason: count}
    current_cap                       float | null   (from dynamic_caps)
    halted                            bool
    halt_reason                       str | null
    last_fill_at                      ISO8601 | null
    last_closed_trade_at              ISO8601 | null
    zero_fill_epoch2                  bool
    notes                             [str, ...]

If a count cannot be measured at the chosen point, it is null with a
reason recorded under `null_reasons` (rather than fabricating a value).

The artifact is the only allowed runtime write; routing is unaffected.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from chad.utils.runtime_json import atomic_write_json, utc_now_iso

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
DATA_DIR = REPO_ROOT / "data"
DIAGNOSTICS_PATH = RUNTIME_DIR / "strategy_routing_diagnostics.json"

EPOCH_STATE_PATH = RUNTIME_DIR / "epoch_state.json"
DYNAMIC_CAPS_PATH = RUNTIME_DIR / "dynamic_caps.json"
ALLOCATIONS_PATH = RUNTIME_DIR / "strategy_allocations.json"

KNOWN_STRATEGIES: tuple = (
    "alpha",
    "alpha_crypto",
    "alpha_forex",
    "alpha_futures",
    "alpha_intraday",
    "alpha_options",
    "beta",
    "beta_trend",
    "delta",
    "delta_pairs",
    "gamma",
    "gamma_futures",
    "gamma_reversion",
    "omega",
    "omega_macro",
    "omega_momentum_options",
    "omega_vol",
)


def _signal_strategy(sig: Any) -> str:
    """Best-effort strategy-name extraction (mirrors live_loop logic)."""
    try:
        v = getattr(sig, "strategy", None)
        v = getattr(v, "value", v)
        if v is None and isinstance(sig, Mapping):
            v = sig.get("strategy")
        return str(v or "").strip().lower()
    except Exception:
        return ""


def _parse_iso(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_epoch_start() -> Optional[datetime]:
    """Epoch 2 start timestamp; None if not readable."""
    obj = _read_json(EPOCH_STATE_PATH)
    if not obj:
        return None
    return _parse_iso(obj.get("epoch_started_at_utc"))


def _scan_fills_for_strategy(
    epoch_start: Optional[datetime],
) -> Dict[str, dict]:
    """
    Scan data/fills/FILLS_*.ndjson for last fill per strategy and a
    boolean for whether any fill exists in the active epoch.
    """
    last_fill: Dict[str, datetime] = {}
    any_in_epoch: Dict[str, bool] = {}
    fills_dir = DATA_DIR / "fills"
    if not fills_dir.is_dir():
        return {}

    for path in sorted(fills_dir.glob("FILLS_*.ndjson")):
        try:
            for line in path.read_text(errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                payload = rec.get("payload", rec)
                strat = str(payload.get("strategy") or "").strip().lower()
                if not strat:
                    continue
                ts = (
                    _parse_iso(payload.get("fill_time_utc"))
                    or _parse_iso(payload.get("ts_utc"))
                    or _parse_iso(payload.get("time_utc"))
                )
                if ts is not None:
                    cur = last_fill.get(strat)
                    if cur is None or ts > cur:
                        last_fill[strat] = ts
                    if epoch_start is not None and ts >= epoch_start:
                        any_in_epoch[strat] = True
        except Exception:
            continue

    out: Dict[str, dict] = {}
    for strat in set(list(last_fill.keys()) + list(any_in_epoch.keys())):
        out[strat] = {
            "last_fill_at": (
                last_fill[strat].isoformat() if strat in last_fill else None
            ),
            "any_fill_in_epoch": bool(any_in_epoch.get(strat, False)),
        }
    return out


def _scan_trades_for_strategy() -> Dict[str, Optional[str]]:
    """Last closed-trade timestamp per strategy."""
    last_closed: Dict[str, datetime] = {}
    trades_dir = DATA_DIR / "trades"
    if not trades_dir.is_dir():
        return {}
    for path in sorted(trades_dir.glob("trade_history_*.ndjson")):
        try:
            for line in path.read_text(errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                payload = rec.get("payload", rec)
                strat = str(payload.get("strategy") or "").strip().lower()
                if not strat:
                    continue
                ts = (
                    _parse_iso(payload.get("exit_time_utc"))
                    or _parse_iso(payload.get("closed_at_utc"))
                    or _parse_iso(payload.get("ts_utc"))
                )
                if ts is None:
                    continue
                cur = last_closed.get(strat)
                if cur is None or ts > cur:
                    last_closed[strat] = ts
        except Exception:
            continue
    return {k: v.isoformat() for k, v in last_closed.items()}


def _read_dynamic_caps() -> Dict[str, Optional[float]]:
    obj = _read_json(DYNAMIC_CAPS_PATH) or {}
    strategies = obj.get("strategies") or {}
    out: Dict[str, Optional[float]] = {}
    for name, info in strategies.items():
        if not isinstance(info, Mapping):
            continue
        cap = info.get("dollar_cap")
        try:
            out[str(name).strip().lower()] = (
                float(cap) if cap is not None else None
            )
        except (TypeError, ValueError):
            out[str(name).strip().lower()] = None
    return out


def _read_halts() -> Dict[str, dict]:
    obj = _read_json(ALLOCATIONS_PATH) or {}
    allocs = obj.get("allocations") or {}
    out: Dict[str, dict] = {}
    for name, info in allocs.items():
        if not isinstance(info, Mapping):
            continue
        out[str(name).strip().lower()] = {
            "halted": bool(info.get("halted", False)),
            "halt_reason": info.get("halt_reason") or None,
        }
    return out


class RoutingDiagnostics:
    """
    Optional in-cycle tracker. live_loop creates one per cycle and
    feeds it observations; the writer translates these into the per-
    strategy artifact. Tracker construction and every observe call
    is failure-soft from the caller's perspective.

    Stage population model:
      * stage_counts[stage] is *only* populated when observe_signals
        was called for that stage. Missing entry → stage was not
        observed this cycle (null with reason).
      * mark_stage_not_present(stage) records that the gate isn't
        wired into this code path at all → null with reason.
      * observe_drop(before, after, reason) attributes the per-strategy
        delta between two signal lists to a block reason.
    """

    STAGES: tuple = (
        "signals_generated_this_cycle",
        "signals_after_edge_decay_or_halt_filter",
        "signals_after_loss_guard_report_only_or_enforced",
        "signals_after_spam_governor",
        "signals_after_net_exposure_gate",
        "signals_after_strategy_throttle",
        "signals_after_regime_gate",
    )

    def __init__(self) -> None:
        self.stage_counts: Dict[str, Dict[str, int]] = {}
        self.stages_not_present: Set[str] = set()
        self.blocked_reasons: Dict[str, Counter] = defaultdict(Counter)
        self.notes: List[str] = []
        self.cycle_started_at: str = utc_now_iso()

    def observe_signals(self, stage: str, signals: Iterable[Any]) -> None:
        if stage not in self.STAGES:
            return
        try:
            counts = self.stage_counts.setdefault(stage, {})
            for sig in signals or ():
                strat = _signal_strategy(sig)
                if strat:
                    counts[strat] = counts.get(strat, 0) + 1
        except Exception:
            return

    def mark_stage_not_present(self, stage: str) -> None:
        """Record that a gate is not wired into this code path."""
        if stage in self.STAGES:
            self.stages_not_present.add(stage)

    def observe_drop(
        self,
        before: Iterable[Any],
        after: Iterable[Any],
        reason: str,
    ) -> None:
        """
        Attribute per-strategy drops between two signal lists to a
        block reason. Failure-soft.
        """
        try:
            r = (reason or "unknown").strip()
            if not r:
                return
            before_counts: Counter = Counter()
            for sig in (before or ()):
                strat = _signal_strategy(sig)
                if strat:
                    before_counts[strat] += 1
            after_counts: Counter = Counter()
            for sig in (after or ()):
                strat = _signal_strategy(sig)
                if strat:
                    after_counts[strat] += 1
            for strat, b in before_counts.items():
                a = int(after_counts.get(strat, 0))
                if a < b:
                    self.blocked_reasons[strat][r] += b - a
        except Exception:
            return

    def observe_block(
        self, strategy: str, reason: str, count: int = 1
    ) -> None:
        try:
            s = (strategy or "").strip().lower()
            r = (reason or "unknown").strip()
            if not s:
                return
            self.blocked_reasons[s][r] += int(count)
        except Exception:
            return

    def add_note(self, note: str) -> None:
        try:
            if note:
                self.notes.append(str(note))
        except Exception:
            return


def build_diagnostics(
    tracker: Optional[RoutingDiagnostics] = None,
    *,
    extra_strategies: Optional[Iterable[str]] = None,
    now: Optional[datetime] = None,
) -> dict:
    """
    Compose the diagnostic dict from a tracker (optional) plus disk state.
    Pure function — does not touch routing or runtime state outside the
    diagnostic file. Always returns a dict (failure-soft).
    """
    epoch_start = _read_epoch_start()
    fills_index = _scan_fills_for_strategy(epoch_start)
    trades_index = _scan_trades_for_strategy()
    caps = _read_dynamic_caps()
    halts = _read_halts()

    universe = set(KNOWN_STRATEGIES)
    universe.update(fills_index.keys())
    universe.update(trades_index.keys())
    universe.update(caps.keys())
    universe.update(halts.keys())
    if extra_strategies:
        universe.update(s.strip().lower() for s in extra_strategies if s)
    universe.discard("")
    pseudo = {"broker_sync", "reconciler", "paper_exec", "warmup_sim",
              "unknown"}

    null_reasons: Dict[str, str] = {}
    recorded_stages: Set[str] = set()
    not_present_stages: Set[str] = set()

    if tracker is None:
        for stage in RoutingDiagnostics.STAGES:
            null_reasons[stage] = "tracker_not_attached_for_this_cycle"
    else:
        recorded_stages = set(tracker.stage_counts.keys())
        not_present_stages = set(tracker.stages_not_present)
        for stage in RoutingDiagnostics.STAGES:
            if stage in not_present_stages:
                null_reasons[stage] = "stage_not_present"
            elif stage not in recorded_stages:
                null_reasons[stage] = "stage_not_observed_this_cycle"

    strategies_out: Dict[str, dict] = {}
    for strat in sorted(universe):
        if strat in pseudo:
            continue
        fill_info = fills_index.get(strat, {})
        cap_value = caps.get(strat)
        halt_info = halts.get(strat, {})
        zero_fill_epoch2 = not bool(fill_info.get("any_fill_in_epoch", False))

        per_stage: Dict[str, Any] = {}
        for stage in RoutingDiagnostics.STAGES:
            if tracker is None:
                per_stage[stage] = None
            elif stage in not_present_stages:
                per_stage[stage] = None
            elif stage in recorded_stages:
                per_stage[stage] = int(
                    tracker.stage_counts[stage].get(strat, 0)
                )
            else:
                per_stage[stage] = None

        blocked_for_strat = (
            dict(tracker.blocked_reasons.get(strat, {}))
            if tracker is not None
            else {}
        )

        strategies_out[strat] = {
            **per_stage,
            "blocked_reasons": blocked_for_strat,
            "current_cap": cap_value,
            "halted": bool(halt_info.get("halted", False)),
            "halt_reason": halt_info.get("halt_reason"),
            "last_fill_at": fill_info.get("last_fill_at"),
            "last_closed_trade_at": trades_index.get(strat),
            "zero_fill_epoch2": zero_fill_epoch2,
        }

    out = {
        "schema_version": "strategy_routing_diagnostics.v1",
        "ts_utc": utc_now_iso(),
        "epoch_started_at_utc": (
            epoch_start.isoformat() if epoch_start else None
        ),
        "tracker_attached": tracker is not None,
        "stages_observed": sorted(recorded_stages),
        "stages_not_present": sorted(not_present_stages),
        "null_reasons": null_reasons,
        "notes": list(tracker.notes) if tracker is not None else [],
        "strategies": strategies_out,
    }
    return out


def write_diagnostics(
    tracker: Optional[RoutingDiagnostics] = None,
    *,
    path: Optional[Path] = None,
    extra_strategies: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """
    Build and atomically write the diagnostics artifact. Returns the
    path written (or None on failure). Never raises — failure is
    silent so the caller's hot path is unaffected.
    """
    target = Path(path) if path is not None else DIAGNOSTICS_PATH
    try:
        payload = build_diagnostics(
            tracker, extra_strategies=extra_strategies
        )
        atomic_write_json(target, payload)
        return target
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "strategy_routing_diagnostics write failed (non-fatal): %s",
            exc,
        )
        return None


__all__ = [
    "RoutingDiagnostics",
    "build_diagnostics",
    "write_diagnostics",
    "DIAGNOSTICS_PATH",
    "KNOWN_STRATEGIES",
]

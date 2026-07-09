#!/usr/bin/env python3
"""
chad/core/live_loop.py

Autonomous live loop for CHAD — runtime behavior layer summary (SSOT v6.4).

Cadence
-------
One cycle every LOOP_INTERVAL_SECONDS (default 60s).  Each cycle is
independent: no cross-cycle state except position_guard.json,
signal_guard.json, and the bridge file last_route_decision.json.

Full cycle sequence
-------------------
1. Broker truth rebuild — reconcile position_guard.json against actual
   broker positions.  Closes stale local entries, adds unknown broker
   positions.  Runs FIRST so all downstream guards see broker reality.

2. Strategy routing — evaluate all strategy handlers, produce signals.
   Two modes controlled by CHAD_ALWAYS_ACTIVE_ROUTING:
     OFF (default): single-winner via choose_strategy_route() — highest
         weight+preference strategy with signals wins; others rejected.
     ON:  always-active via evaluate_all_strategies() — all strategies
         with signals contribute.
   RouteDecision / AllStrategiesDecision written to
   runtime/last_route_decision.json for orchestrator DecisionTrace pickup.

3. Intent planning — DecisionPipeline runs ALL strategies through
   StrategyEngine → PolicyEngine (affordability, sizing) → SignalRouter
   (merge same-symbol-same-side) → ExecutionPlan → IBKR intents.
   Zero-net-size buckets produce PlanRejection(reason="zero_net_size").

4. Per-intent guard evaluation (in priority order):
   a. Strategy attribution — attach strategy label from routed_signal_map.
   b. Position guard — is_same_side_open() blocks same-side duplicates
      (SuppressionReason.SAME_SIDE_POSITION_OPEN).
   c. Signal dedup — should_emit_signal() enforces 10-min cooldown per
      (strategy, symbol, side, size) fingerprint
      (SuppressionReason.COOLDOWN_ACTIVE).  Skipped for flip signals.

5. Execution — surviving intents are executed:
   - Flip: replace_position() (PositionState.FLIPPED)
   - New:  mark_position_open() (PositionState.OPEN)

6. Idle detection — if no signals from any strategy, cycle emits
   CYCLE_IDLE with SuppressionReason.NO_SIGNAL and strategies_checked
   counts proving all strategies were evaluated.

Evidence logging
----------------
- Paper execution evidence: hash-chained NDJSON in data/fills/, data/fees/,
  data/execution_metrics/ via paper_exec_evidence_writer.
- DecisionTrace: NDJSON in data/traces/decision_trace_YYYYMMDD.ndjson
  (schema v3, includes strategy_detail from last_route_decision.json).
- Signal/position guard state: runtime/signal_guard.json,
  runtime/position_guard.json (last_state field tracks lifecycle enum).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from chad.core.live_execution_router import (
    build_live_signals,
    build_all_live_signals,
    is_always_active_routing,
)
from chad.core.ibkr_execution_runner import _build_plan_and_intents
from chad.core.suppression import SuppressionReason
from chad.core.broker_position_sync import BrokerPositionSync
from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig, resolve_asset_class
from chad.execution.futures_gate import is_futures_sec_type
from chad.execution.ibkr_client_ids import (
    LIVE_LOOP as _IBKR_LIVE_LOOP_CLIENT_ID,
    EXECUTION as _IBKR_EXECUTION_CLIENT_ID,
)
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    StrategyAttributionError,
    normalize_paper_fill_evidence,
    write_paper_exec_evidence,
)

# Paper-mode fill statuses that indicate the order was submitted to IBKR paper
# but no fill confirmation arrived synchronously. In paper mode the correct
# action is to record a simulated fill at the last-known price — IBKR paper
# fills are near-instantaneous and waiting indefinitely causes the fill to
# be permanently recorded with status="PendingSubmit", which SCR excludes as
# untrusted.
_PAPER_PENDING_STATUSES = frozenset({
    "pendingsubmit", "presubmitted", "submitted", "apipending",
    "inactive", "unknown", "", "error",
})
from ib_async import IB, util

# ib_insync requires asyncio. patchAsyncio() applies nest_asyncio so the
# main-thread loop is reentrant. Worker threads (e.g. Redis state-bus
# listeners in chad/core/state_bus.py) that end up invoking IB calls have
# no loop of their own; _ensure_thread_event_loop() below installs one
# before each submission to avoid "no current event loop in thread ..."
# from asyncio's default policy.
util.patchAsyncio()

ib = IB()
# Monkey-patch: skip reqExecutionsAsync on connect to avoid
# cold-start hang when gateway has large execution backlog.
# The fill harvester (clientId=79) handles execution history separately.
import ib_async.ib as _ib_module
async def _noop_executions(self, *a, **kw): return []
_ib_module.IB.reqExecutionsAsync = _noop_executions
# ISSUE-29 / test-import safety: tests that import this module must NOT
# attempt to claim the LIVE_LOOP client id — the running live_loop process
# holds it and the connect would TimeoutError (Error 326).
# CHAD_SKIP_IB_CONNECT=1 in the pytest environment skips the connect; the
# live runner leaves it unset and connects normally.
if os.environ.get("CHAD_SKIP_IB_CONNECT", "").strip().lower() not in ("1", "true", "yes"):
    ib.connect("127.0.0.1", 4002, clientId=_IBKR_LIVE_LOOP_CLIENT_ID, timeout=120)


def _ensure_thread_event_loop() -> None:
    if threading.current_thread() is threading.main_thread():
        return
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

position_sync = BrokerPositionSync(ib)
from chad.execution.execution_config import (
    ExecutionMode as _ExecMode,
    get_execution_mode as _get_exec_mode,
)

# ---------------------------------------------------------------------------
# L1-CLD U7 activation — execution-connection ownership.
#
# The cross-loop-deadlock fix (PA L1_CLD_cross_loop_deadlock_fix_2026-07-08)
# roots the deadlock: the shared `ib` above is connected on MainThread and its
# reader is never pumped, so broker request/response calls dispatched onto other
# loops hang uninterruptibly. The fix homes the EXECUTION connection on a
# dedicated broker-owner event loop (chad/execution/broker_loop.py) whose reader
# runs continuously.
#
# To activate it in production the execution adapter must own its OWN connection
# (params-mode) instead of adopting the pre-connected MainThread `ib`. The shared
# `ib` stays exactly as-is for position_sync + market-data; only the execution
# adapter migrates. params-mode is the production default; set
# CHAD_EXECUTION_OWN_CONNECTION=0 for an instant rollback to legacy adoption
# WITHOUT a code revert. Fail-closed: if boot homing fails, live_loop starts in
# a NO-EXECUTION state (BROKER_LOOP_DOWN) and never falls back to a MainThread
# connect (see _home_execution_connection + the submit gate in run_once).
# ---------------------------------------------------------------------------

# Set True by _home_execution_connection() when boot homing fails; the submit
# path in run_once() reads it to stay fail-closed (no orders, no fallback).
_EXECUTION_DISABLED: bool = False


def _execution_owns_connection() -> bool:
    """True (default) when the execution adapter owns a DEDICATED IB connection
    homed on the broker owner loop. CHAD_EXECUTION_OWN_CONNECTION=0/false/no/off
    forces legacy shared-`ib` adoption (the pre-U7 dormant behavior)."""
    return (
        os.environ.get("CHAD_EXECUTION_OWN_CONNECTION", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )


def _execution_client_id() -> int:
    """Dedicated execution clientId. Env CHAD_EXECUTION_CLIENT_ID overrides the
    canonical registry default (ibkr_client_ids.EXECUTION); a non-int value
    falls back to the safe default. Never LIVE_LOOP's shared-connection id."""
    raw = os.environ.get("CHAD_EXECUTION_CLIENT_ID", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            logging.getLogger("chad.live_loop").warning(
                "CHAD_EXECUTION_CLIENT_ID=%r is not an int; using registry "
                "default clientId=%s",
                raw,
                _IBKR_EXECUTION_CLIENT_ID,
            )
    return _IBKR_EXECUTION_CLIENT_ID


if _execution_owns_connection():
    # params-mode (L1-CLD activation, production default): the adapter creates
    # an UNCONNECTED IB and connectAsync's it on the broker owner loop with the
    # dedicated execution clientId. No pre-connected MainThread `ib` is injected.
    _paper_adapter = IbkrAdapter(
        config=IbkrConfig(
            dry_run=(_get_exec_mode() != _ExecMode.IBKR_PAPER),
            client_id=_execution_client_id(),
        ),
        ib_factory=lambda: IB(),
    )
else:
    # Legacy adoption (rollback): reuse the shared MainThread-connected `ib`.
    # The L1-CLD owner-loop routing stays OFF (dormant) on this path.
    _paper_adapter = IbkrAdapter(
        config=IbkrConfig(
            dry_run=(_get_exec_mode() != _ExecMode.IBKR_PAPER),
        ),
        ib_factory=lambda: ib,
    )


def _home_execution_connection(logger: logging.Logger) -> bool:
    """Boot-time homing of the dedicated execution connection on the broker
    owner loop (L1-CLD U7). FAIL-CLOSED: on any failure mark execution disabled,
    emit the BROKER_LOOP_DOWN marker, and start in a NO-EXECUTION state — never
    silently fall back to a MainThread connection. Returns True iff homed.

    No-ops (returns False) in legacy-adoption mode and when CHAD_SKIP_IB_CONNECT
    is set (import/test parity with the shared-`ib` connect guard above)."""
    global _EXECUTION_DISABLED
    if not _execution_owns_connection():
        return False
    if os.environ.get("CHAD_SKIP_IB_CONNECT", "").strip().lower() in ("1", "true", "yes"):
        return False
    try:
        _paper_adapter.ensure_connected(force=True)
        _EXECUTION_DISABLED = False
        logger.info(
            "EXECUTION_OWNER_LOOP_ENGAGED clientId=%s — dedicated execution "
            "connection homed on the broker owner loop (L1-CLD U7 active)",
            _execution_client_id(),
        )
        return True
    except BaseException as exc:  # noqa: BLE001 - fail-closed on ANY boot error
        _EXECUTION_DISABLED = True
        logger.error(
            "BROKER_LOOP_DOWN — execution owner-loop homing FAILED at boot; "
            "starting in NO-EXECUTION state (fail-closed, no MainThread "
            "fallback): %s",
            exc,
            extra={"marker": "BROKER_LOOP_DOWN"},
        )
        try:
            from chad.utils.telegram_notify import notify
            notify(
                "🚨 CHAD LIVE LOOP — execution owner-loop homing FAILED at boot. "
                "Starting in NO-EXECUTION state (BROKER_LOOP_DOWN); no orders "
                "will be placed until the execution connection is restored.",
                severity="critical",
                dedupe_key="exec_broker_loop_down",
            )
        except Exception:
            pass
        return False
LOOP_INTERVAL_SECONDS = 60
_ROUTE_DECISION_PATH = Path("/home/ubuntu/chad_finale/runtime/last_route_decision.json")

# Gap-1 (v9.1 audit): tier-level allowlist enforcement on raw TradeSignals
# emitted by strategies. Strategies (e.g. alpha_crypto) carry internal
# universes and do not consult the active tier; this gate is the canonical
# pipeline-level enforcement point. Instance is created once at module
# load and reused each cycle; the gate re-reads tier_state.json per call.
from chad.execution.tier_instrument_gate import TierInstrumentGate as _TierInstrumentGate
_TIER_INSTRUMENT_GATE = _TierInstrumentGate(
    runtime_dir=Path("/home/ubuntu/chad_finale/runtime"),
)

# Redis stop signal flag — set by state bus subscriber
_redis_stop_flag = False
_redis_stop_reason = ""

# Redis dynamic_caps — advisory cache for reducing JSON poll lag
_redis_dynamic_caps: Optional[Dict] = None

# Redis live_gate — last received timestamp for latency monitoring
_redis_live_gate_ts: Optional[str] = None

# Edge-decay edge-trigger tracking — strategies that already received
# a Telegram halt alert this process lifetime. Cleared when a strategy
# leaves the halted set so re-halts after a clear can re-alert.
_EDGE_DECAY_ALERTED: set = set()

# SIGTERM-driven clean shutdown flag — set by _handle_sigterm so the
# main run_loop() can break between cycles instead of being killed
# mid-trade by systemd.
_SHUTDOWN_REQUESTED: bool = False


def _handle_sigterm(signum: int, frame) -> None:
    """Handle SIGTERM from systemd gracefully."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    logger = logging.getLogger("chad.live_loop")
    logger.warning("SIGTERM received — requesting clean shutdown")
    try:
        from chad.utils.telegram_notify import notify
        notify(
            "⚠️ CHAD LIVE LOOP — SIGTERM received. Clean shutdown requested.",
            severity="warning",
            dedupe_key="live_loop_sigterm",
        )
    except Exception:
        pass


from chad.core.kraken_execution import execute_kraken_intents as _execute_kraken_intents
from chad.risk.symbol_performance_blocker import is_symbol_blocked


_KRAKEN_BALANCE_PROVIDER = None  # lazy singleton


def _futures_execution_disabled(env: Mapping[str, str]) -> bool:
    """True when any futures-disable flag is set. Reversible env stopgap.

    Thin wrapper preserved for the existing call sites / tests; the flag logic
    lives in exactly one place — chad.execution.futures_gate — which the
    broker-submit chokepoints import as the same single source of truth.
    """
    from chad.execution.futures_gate import futures_execution_disabled

    return futures_execution_disabled(env)


# ---------------------------------------------------------------------------
# Bug B Fix A: cumulative broker-truth per-symbol futures position cap.
#
# The same-side dedupe is strategy-scoped and ledger-backed; its attribution
# window is why the env gate above could not be removed. This cap is the
# durable guard: it bounds every FUT open against the BROKER-TRUTH net
# position for the symbol (any strategy attribution), consults NO env flag,
# and fail-closes when broker truth cannot be verified. Exits/flips always
# pass (same classification as the env gate).
# ---------------------------------------------------------------------------

_POSITIONS_TRUTH_PATH = Path("/home/ubuntu/chad_finale/runtime/positions_truth.json")


def _build_futures_position_caps() -> Dict[str, int]:
    """Per-symbol cumulative position caps, derived from the strategy
    instrument spec tables (single source of truth), min() on conflict."""
    try:
        from chad.strategies.alpha_futures import DEFAULT_SPECS as _alpha_specs
        from chad.strategies.omega_macro import OMEGA_MACRO_SPECS as _omega_specs

        caps: Dict[str, int] = {}
        for table in (_alpha_specs, _omega_specs):
            for sym, spec in table.items():
                cap = int(getattr(spec, "max_contracts", 0) or 0)
                key = str(sym).strip().upper()
                caps[key] = min(caps[key], cap) if key in caps else cap
        return caps
    except Exception:
        # FAIL-CLOSED FALLBACK — not the source of truth; tune caps in the
        # spec tables (alpha_futures.DEFAULT_SPECS / OMEGA_MACRO_SPECS),
        # never here. Snapshot of the derived table as of 2026-06-03.
        return {
            "MES": 5,
            "MNQ": 5,
            "MCL": 2,
            "MGC": 5,
            "MYM": 5,
            "M2K": 5,
            "ZN": 3,
            "ZB": 2,
            "M6E": 3,
        }


_FUTURES_POSITION_CAPS: Dict[str, int] = _build_futures_position_caps()


def _read_broker_truth_for_caps(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load positions_truth.json fresh + broker-authority GREEN, else None.

    Uses the ttl-aware reader (live_gate.read_runtime_state_json): missing,
    unparseable, or stale-beyond-ttl all return None. Additionally requires
    broker_authority_status == GREEN and truth_ok is True — fail-closed on
    any doubt.
    """
    from chad.core.live_gate import read_runtime_state_json

    truth_path = path or _POSITIONS_TRUTH_PATH
    try:
        obj, _fr = read_runtime_state_json(truth_path, default_ttl_s=120)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if str(obj.get("broker_authority_status", "")).strip().upper() != "GREEN":
        return None
    if obj.get("truth_ok") is not True:
        return None
    return obj


def _futures_net_from_truth(truth_obj: Mapping[str, Any], symbol: str) -> Optional[float]:
    """Signed net broker position for *symbol* from a loaded truth object.

    Sums `position` over FUT entries matching the symbol. No matching entry
    means flat (0.0). A malformed position value returns None (fail-closed —
    cannot verify).
    """
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    net = 0.0
    for entry in truth_obj.get("positions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("secType", "")).strip().upper() != "FUT":
            continue
        if str(entry.get("symbol", "")).strip().upper() != sym:
            continue
        try:
            net += float(entry.get("position"))
        except (TypeError, ValueError):
            return None
    return net


def _futures_net_broker_position(symbol: str, path: Optional[Path] = None) -> Optional[float]:
    """Signed net broker position for a futures symbol, from broker truth.

    Returns None when positions_truth is unreadable / stale / not
    broker-authority GREEN, or carries a malformed position — callers must
    treat None as "cannot verify" and refuse futures opens (fail-closed).
    """
    truth_obj = _read_broker_truth_for_caps(path)
    if truth_obj is None:
        return None
    return _futures_net_from_truth(truth_obj, symbol)


def _futures_cap_check(
    symbol: str,
    side: str,
    quantity: float,
    broker_net: Optional[float],
    pending_adds: Dict[str, float],
    caps: Optional[Mapping[str, int]] = None,
) -> Tuple[str, float, float, int]:
    """Cap decision for one FUT open intent (exits/flips never reach this).

    Returns (verdict, net_incl_pending, projected, cap) where verdict is:
      - "unverified": broker_net is None — fail-closed, refuse.
      - "block": the trade would GROW |net| beyond the per-symbol cap.
      - "allow": within cap, or a reduce — the signed quantity is registered
        into *pending_adds* so later intents this cycle project cumulatively.

    Unknown symbols resolve to cap 0 (a futures symbol absent from the spec
    tables must not open uncapped).
    """
    sym = str(symbol or "").strip().upper()
    cap_table: Mapping[str, int] = _FUTURES_POSITION_CAPS if caps is None else caps
    cap = int(cap_table.get(sym, 0))
    if broker_net is None:
        return "unverified", 0.0, 0.0, cap
    signed = float(quantity or 0.0)
    if str(side or "").strip().upper() == "SELL":
        signed = -signed
    net_incl_pending = float(broker_net) + float(pending_adds.get(sym, 0.0))
    projected = net_incl_pending + signed
    if abs(projected) > cap and abs(projected) > abs(net_incl_pending):
        return "block", net_incl_pending, projected, cap
    pending_adds[sym] = pending_adds.get(sym, 0.0) + signed
    return "allow", net_incl_pending, projected, cap


def _refresh_kraken_balance_snapshot(logger: logging.Logger) -> None:
    """
    Refresh runtime/kraken_balances.json on a 5-minute interval.

    Reads the current price cache to value crypto + fiat balances in USD.
    The provider's own throttle keeps actual API calls capped at one per
    DEFAULT_REFRESH_INTERVAL_SECONDS regardless of how often this is invoked.
    """
    global _KRAKEN_BALANCE_PROVIDER
    if _KRAKEN_BALANCE_PROVIDER is None:
        from chad.market_data.kraken_balance_provider import KrakenBalanceProvider
        _KRAKEN_BALANCE_PROVIDER = KrakenBalanceProvider()

    prices: Dict[str, float] = {}
    try:
        pc_path = Path("/home/ubuntu/chad_finale/runtime/price_cache.json")
        if pc_path.is_file():
            pc = json.loads(pc_path.read_text(encoding="utf-8"))
            raw_prices = pc.get("prices", {}) if isinstance(pc, dict) else {}
            if isinstance(raw_prices, dict):
                for k, v in raw_prices.items():
                    try:
                        prices[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
    except Exception:
        prices = {}

    snap = _KRAKEN_BALANCE_PROVIDER.maybe_refresh_snapshot(prices)
    if snap is not None:
        logger.info(
            "KRAKEN_BALANCE_SNAPSHOT ok=%s usd_eq=%.2f assets=%s",
            snap.ok, snap.usd_equivalent, list(snap.balances.keys()),
        )


def _write_route_decision(detail: Dict[str, Any]) -> None:
    """Write strategy_detail bridge file for orchestrator DecisionTrace pickup."""
    try:
        _ROUTE_DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ROUTE_DECISION_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")
        tmp.replace(_ROUTE_DECISION_PATH)
    except Exception:
        pass


def _log_throttle_audit(event: str, reason: str, thr_data: dict) -> None:
    """SS05: append-only audit log for signal throttle state transitions.

    Each event (activate / expire / manual_clear) is written as a single
    NDJSON line so a tail can recover the throttle's full lifecycle without
    parsing the live runtime file. Failures are intentionally silent —
    audit logging must never break the trading loop.
    """
    try:
        from datetime import datetime, timezone
        audit_path = Path(
            "/home/ubuntu/chad_finale/runtime/signal_throttle_audit.json"
        )
        entry = {
            "event": event,
            "reason": reason,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "throttle_data": thr_data,
        }
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def _load_optional_metric(name: str) -> Optional[float]:
    """Best-effort lookup of a regime classifier input.

    Scans a small set of runtime/* JSON files for the named metric.
    Missing keys or unreadable files return None — the classifier then
    treats the input as unavailable and downgrades to 'unknown' if no
    other signal is present.
    """
    candidate_paths = (
        Path("/home/ubuntu/chad_finale/runtime/market_metrics.json"),
        Path("/home/ubuntu/chad_finale/runtime/regime_inputs.json"),
        Path("/home/ubuntu/chad_finale/runtime/macro_state.json"),
    )
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if name in data and data[name] is not None:
            try:
                return float(data[name])
            except (TypeError, ValueError):
                continue
    return None


def _build_stop_bus_snapshot(
    logger: logging.Logger,
    ibkr_status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Best-effort snapshot of halt-trigger inputs.

    Missing inputs are simply omitted — the aggregator skips any trigger
    whose keys aren't present, so partial snapshots are fine.

    ``ibkr_status_path`` is overridable for tests; production uses the canonical
    runtime artifact. Fix A activation: this snapshot now also carries the
    sustained-breach hysteresis inputs (consecutive_cycles_above_stop_threshold,
    last_above_threshold_at, breach_streak_started_at) when the tracker has
    published them. A missing/old artifact simply omits those keys, so the
    broker_latency trigger falls back to legacy single-cycle behaviour.
    """
    snap: Dict[str, Any] = {}

    # Daily loss trigger: pnl_state.json carries today's realized PnL.
    try:
        pnl_path = Path("/home/ubuntu/chad_finale/runtime/pnl_state.json")
        if pnl_path.is_file():
            pnl = json.loads(pnl_path.read_text(encoding="utf-8"))
            if isinstance(pnl, dict):
                realized = pnl.get("realized_pnl", pnl.get("daily_realized_pnl"))
                if realized is not None:
                    snap["realized_pnl"] = float(realized)
    except Exception:
        pass

    # Daily loss limit — operator-tuned env var, default generous so the
    # trigger stays dormant unless explicitly configured.
    try:
        limit = os.environ.get("CHAD_DAILY_LOSS_LIMIT")
        if limit is not None:
            snap["daily_loss_limit"] = float(limit)
    except (TypeError, ValueError):
        pass

    # Broker latency: ibkr_status.json publishes the latency measurement plus
    # the reliability-tracker hysteresis fields (Fix A activation). All are
    # read best-effort; a missing/old artifact leaves the hysteresis keys
    # absent and the broker_latency trigger uses legacy single-cycle behaviour.
    try:
        status_path = ibkr_status_path or Path(
            "/home/ubuntu/chad_finale/runtime/ibkr_status.json"
        )
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict):
                # Prefer api_ms (true API round-trip); latency_ms includes connect_ms (SSOT: connect_ms != api_ms).
                lat = status.get("api_ms") or status.get("latency_ms") or status.get("avg_latency_ms")
                if lat is not None:
                    snap["avg_latency_ms"] = float(lat)
                if "consecutive_cycles_above_stop_threshold" in status:
                    snap["consecutive_cycles_above_stop_threshold"] = status.get(
                        "consecutive_cycles_above_stop_threshold"
                    )
                if "last_above_threshold_at" in status:
                    snap["last_above_threshold_at"] = status.get("last_above_threshold_at")
                if "breach_streak_started_at" in status:
                    snap["breach_streak_started_at"] = status.get("breach_streak_started_at")
    except Exception:
        pass

    # Reject-rate and data-staleness snapshots require windowed counters
    # that the current codebase does not yet publish; they are left
    # unpopulated and the aggregator skips them cleanly.
    return snap


def _build_stop_bus_config() -> Dict[str, Any]:
    """Assemble the stop-bus trigger config from operator env vars (Fix A
    activation). Threads the broker_latency hysteresis knobs introduced in
    f3ab3d8 so they are tunable without a code change; each falls back to the
    module default baked into chad/risk/stop_bus_triggers.py.
    """
    from chad.risk.stop_bus_triggers import (
        DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED,
        DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS,
        DEFAULT_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED,
    )

    cfg: Dict[str, Any] = {}
    try:
        cfg["broker_latency_trip_consecutive_required"] = int(
            os.environ.get(
                "CHAD_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED",
                DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED,
            )
        )
    except (TypeError, ValueError):
        cfg["broker_latency_trip_consecutive_required"] = (
            DEFAULT_BROKER_LATENCY_TRIP_CONSECUTIVE_REQUIRED
        )
    try:
        cfg["broker_latency_trip_min_breach_seconds"] = float(
            os.environ.get(
                "CHAD_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS",
                DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS,
            )
        )
    except (TypeError, ValueError):
        cfg["broker_latency_trip_min_breach_seconds"] = (
            DEFAULT_BROKER_LATENCY_TRIP_MIN_BREACH_SECONDS
        )
    raw = os.environ.get("CHAD_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED")
    if raw is None:
        cfg["broker_latency_trip_hysteresis_enabled"] = (
            DEFAULT_BROKER_LATENCY_TRIP_HYSTERESIS_ENABLED
        )
    else:
        cfg["broker_latency_trip_hysteresis_enabled"] = (
            str(raw).strip().lower() in ("1", "true", "yes")
        )
    return cfg


def _build_router_signal_map(signals: List[object]) -> Dict[Tuple[str, str, str, float], object]:
    out: Dict[Tuple[str, str, str, float], object] = {}

    for sig in signals:
        strategy = getattr(getattr(sig, "strategy", None), "value", getattr(sig, "strategy", None))
        symbol = getattr(sig, "symbol", None)
        side = getattr(getattr(sig, "side", None), "value", getattr(sig, "side", None))
        size = float(getattr(sig, "size", 0.0) or 0.0)

        key = (str(strategy), str(symbol), str(side), size)
        out[key] = sig

    return out


def _attach_strategy_to_intent(intent: object, routed_signal_map: Dict[Tuple[str, str, str, float], object]) -> None:
    strategy = getattr(intent, "strategy", None)
    if strategy not in (None, "", "None"):
        return

    symbol = getattr(intent, "symbol", None)
    side = getattr(intent, "side", None)
    quantity = float(getattr(intent, "quantity", 0.0) or 0.0)

    exact_key = ("alpha_futures", str(symbol), str(side), quantity)
    if exact_key in routed_signal_map:
        setattr(intent, "strategy", exact_key[0])
        return

    for (sig_strategy, sig_symbol, sig_side, _sig_size), _sig in routed_signal_map.items():
        if sig_symbol == str(symbol) and sig_side == str(side):
            setattr(intent, "strategy", sig_strategy)
            return


def _is_paper_mode() -> bool:
    """Detect paper / dry_run posture via canonical execution_config reader."""
    from chad.execution.execution_config import is_paper_mode
    return is_paper_mode()


_TRADE_CLOSER_STATE_PATH = Path("/home/ubuntu/chad_finale/runtime/trade_closer_state.json")


def _rebuild_guard_from_paper_ledger(logger: logging.Logger) -> None:
    """
    Paper-mode reconciliation. Reads runtime/trade_closer_state.json and
    rewrites paper-strategy entries in position_guard.json so the doom loop
    cannot wipe simulated positions every cycle.

    Rules:
    - For each (strategy, symbol) with a non-empty FIFO queue → guard entry
      open=True with side from the head lot, quantity = sum of lot sizes.
    - For each existing guard entry that is NOT a broker_sync entry and is
      NOT present in the trade_closer queues → mark closed
      (closed_by="paper_ledger_rebuild").
    - broker_sync|* entries (real broker truth from prior live runs) are
      left untouched.

    The IB Gateway is NOT queried in paper mode.
    """
    from datetime import datetime, timezone
    from chad.core.position_guard import _load_state, save_state

    queues_by_key: Dict[Tuple[str, str], list] = {}
    if _TRADE_CLOSER_STATE_PATH.is_file():
        try:
            data = json.loads(_TRADE_CLOSER_STATE_PATH.read_text(encoding="utf-8"))
            for entry in data.get("queues", []) or []:
                strategy = str(entry.get("strategy", "")).strip()
                symbol = str(entry.get("symbol", "")).strip().upper()
                lots = [
                    lot for lot in (entry.get("lots") or [])
                    if float(lot.get("quantity", 0) or 0) > 0
                ]
                if strategy and symbol and lots:
                    queues_by_key[(strategy, symbol)] = lots
        except Exception as exc:
            logger.warning("PAPER_GUARD_RECONCILE: failed to read trade_closer_state: %s", exc)
            return

    guard_state = _load_state()
    now_iso = datetime.now(timezone.utc).isoformat()
    open_count = 0
    closed_count = 0

    for (strategy, symbol), lots in queues_by_key.items():
        head = lots[0]
        side = str(head.get("side", "")).strip().upper()
        total_qty = sum(float(lot.get("quantity", 0) or 0) for lot in lots)
        key = f"{strategy}|{symbol}"
        # ISSUE-56 v2: when a strategy queue claims a symbol, reduce the
        # matching broker_sync|<symbol> anchor by the strategy's quantity.
        # Same side + residual > 0 → keep open at residual (partial attribution).
        # Same side + residual <= 0 → soft-close (full attribution).
        # Opposite side → leave untouched (flip intent; reconciliation surfaces drift).
        if strategy != "broker_sync":
            broker_sync_key = f"broker_sync|{symbol}"
            bs_entry = guard_state.get(broker_sync_key)
            if bs_entry and bs_entry.get("open"):
                bs_side = str(bs_entry.get("side", "") or "").upper()
                if side == bs_side:
                    bs_qty = abs(float(bs_entry.get("quantity", 0) or 0))
                    residual = bs_qty - abs(float(total_qty or 0))
                    bs_entry["updated_at_utc"] = now_iso
                    if residual <= 0:
                        bs_entry["open"] = False
                        bs_entry["closed_by"] = "strategy_ownership_assumed"
                    else:
                        bs_entry["quantity"] = residual
                        bs_entry["open"] = True
                        bs_entry["closed_by"] = "partial_attribution_residual"
        prior = guard_state.get(key) or {}
        prior_opened = None
        if prior.get("open") is True and str(prior.get("side", "")).upper() == side:
            prior_opened = prior.get("opened_at")
        guard_state[key] = {
            "open": True,
            "opened_at": prior_opened or now_iso,
            "updated_at_utc": now_iso,
            "strategy": strategy,
            "symbol": symbol,
            "side": side,
            "quantity": total_qty,
            "last_state": "OPEN",
            "source": "paper_ledger_rebuild",
        }
        open_count += 1

    for key, entry in list(guard_state.items()):
        if str(key).startswith("_") or not isinstance(entry, dict):
            continue
        strategy = str(entry.get("strategy", ""))
        symbol = str(entry.get("symbol", "")).upper()
        if strategy == "broker_sync":
            continue
        if (strategy, symbol) in queues_by_key:
            continue
        if entry.get("open"):
            entry["open"] = False
            entry["updated_at_utc"] = now_iso
            entry["closed_by"] = "paper_ledger_rebuild"
            closed_count += 1

    save_state(guard_state)
    logger.info(
        "PAPER_GUARD_RECONCILE: rebuilt from trade_closer_state, %d positions open, %d newly closed",
        open_count,
        closed_count,
        extra={
            "paper_open": open_count,
            "paper_closed": closed_count,
            "source": "trade_closer_state",
        },
    )


# Roots used to collapse contract-month tickers (e.g. "MESM6", "MESZ6",
# "MESH7") down to a stable root key for broker_sync guard entries. Without
# this, a contract roll silently changes the guard key and the prior month's
# entry is left as a zombie in position_guard.json.
_BROKER_SYNC_FUTURES_ROOTS: tuple = (
    "MES", "MNQ", "MCL", "MGC", "M2K", "MYM", "M6E", "ZN", "ZB",
)
_CME_MONTH_CODES: frozenset = frozenset("FGHJKMNQUVXZ")


def _normalize_broker_sync_symbol(symbol: str) -> str:
    """Collapse a futures contract-month ticker to its root.

    "MESM6" -> "MES", "M2KZ7" -> "M2K". Non-futures (equities, ETFs) and
    unrecognised symbols pass through unchanged. Sorts roots longest-first
    so "MES" wins over any shorter prefix.
    """
    if not symbol:
        return symbol
    for root in sorted(_BROKER_SYNC_FUTURES_ROOTS, key=len, reverse=True):
        if symbol == root or not symbol.startswith(root):
            continue
        tail = symbol[len(root):]
        if 2 <= len(tail) <= 3 and tail[0] in _CME_MONTH_CODES and tail[1:].isdigit():
            return root
    return symbol


def _rebuild_guard_from_broker(logger: logging.Logger) -> None:
    """
    Reconcile position_guard.json against actual broker positions.

    Called at the top of every cycle so the guard state reflects broker truth
    before any signal evaluation or entry logic runs.

    Local guard reset conditions (SSOT v6.4):
    ──────────────────────────────────────────
    COVERED:
    1. Stale local memory — guard entry shows open but broker has no position
       for that symbol → entry closed, closed_by="broker_truth_rebuild".
    2. Broker-truth contradiction — broker holds a position the guard doesn't
       know about → new entry created with strategy="broker_sync".
    3. Broker reconnect (implicit) — this function runs every cycle; after a
       broker disconnect/reconnect, the next successful cycle reconciles.

    KNOWN GAPS (not yet implemented):
    4. Symbol normalization mismatch — comparison is plain string equality;
       variants like "BRK.B" vs "BRK B" are not normalized.
    5. Reconciliation repair tracking — no concept of marking a previously-
       broken entry as "repaired" vs "newly corrected".
    """
    import json
    from chad.core.position_guard import _load_state, save_state

    broker_positions = position_sync.fetch_positions()
    guard_state = _load_state()

    # Collapse contract-month futures tickers ("MESM6") to their stable root
    # ("MES") so guard keys survive contract rolls. Equities pass through.
    broker_symbols = {
        _normalize_broker_sync_symbol(sym)
        for sym, pos in broker_positions.items()
        if abs(pos.quantity) > 1e-9
    }
    corrections_closed = 0
    corrections_opened = 0

    # Close guard entries that the broker no longer holds (compare on
    # normalized symbol so a roll from MESM6 -> MESZ6 does not look like
    # the position vanished).
    for key, entry in list(guard_state.items()):
        if str(key).startswith("_") or not isinstance(entry, dict):
            continue
        if not entry.get("open"):
            continue
        symbol = entry.get("symbol", "")
        if symbol and _normalize_broker_sync_symbol(symbol) not in broker_symbols:
            entry["open"] = False
            entry["closed_by"] = "broker_truth_rebuild"
            corrections_closed += 1

    # Add or update broker_sync positions to reflect broker truth.
    # broker_sync key always tracks IBKR position exactly. Other strategy
    # keys (alpha|SPY, delta|SPY) track CHAD-initiated trades separately.
    for sym, bp in broker_positions.items():
        if abs(bp.quantity) < 1e-9:
            continue
        norm_sym = _normalize_broker_sync_symbol(sym)
        side = "BUY" if bp.quantity > 0 else "SELL"
        fallback_key = f"broker_sync|{norm_sym}"
        now_iso = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        existing = guard_state.get(fallback_key, {})
        prior_opened = existing.get("opened_at") if existing.get("open") else None
        guard_state[fallback_key] = {
            "open": True,
            "opened_at": prior_opened or now_iso,
            "updated_at_utc": now_iso,
            "strategy": "broker_sync",
            "symbol": norm_sym,
            "side": side,
            "quantity": abs(bp.quantity),
            "source": "broker_truth_rebuild",
        }
        if not existing.get("open") or float(existing.get("quantity", 0)) != abs(bp.quantity):
            corrections_opened += 1

    if corrections_closed or corrections_opened:
        save_state(guard_state)
        logger.info(
            "BROKER_TRUTH_REBUILD",
            extra={
                "corrections_closed": corrections_closed,
                "corrections_opened": corrections_opened,
                "broker_position_count": len(broker_symbols),
                "guard_open_count": sum(1 for e in guard_state.values() if isinstance(e, dict) and e.get("open")),
            },
        )
    else:
        logger.info(
            "BROKER_TRUTH_REBUILD",
            extra={
                "corrections_closed": 0,
                "corrections_opened": 0,
                "broker_position_count": len(broker_symbols),
                "guard_open_count": sum(1 for e in guard_state.values() if isinstance(e, dict) and e.get("open")),
                "status": "no_corrections_needed",
            },
        )


_INTELLIGENCE_CACHE_PATH = Path("/home/ubuntu/chad_finale/runtime/strategy_intelligence_cache.json")
_INTELLIGENCE_CACHE_MAX_AGE_SEC = 300  # 5 minutes
_MIN_CONFIDENCE_THRESHOLD = 0.20


def _apply_intelligence_bias(intents: list, logger: logging.Logger) -> list:
    """
    Apply strategy intelligence confidence bias to intents.

    Reads cached bias from runtime/strategy_intelligence_cache.json.
    If cache is fresh (< 5 min) and symbol has a bias entry, adjusts confidence.
    Suppresses signals where adjusted confidence falls below threshold.

    Fail-open: if cache is missing, stale, or unreadable, all intents pass through.
    """
    try:
        if not _INTELLIGENCE_CACHE_PATH.exists():
            return intents

        cache_data = json.loads(_INTELLIGENCE_CACHE_PATH.read_text(encoding="utf-8"))
        confidence_cache = cache_data.get("confidence", {})
        if not confidence_cache:
            return intents
    except Exception as _bias_exc:
        logger.warning(
            "live_loop_intelligence_bias_failed err=%s — proceeding without bias",
            _bias_exc,
        )
        return intents

    from datetime import datetime, timezone

    surviving = []
    for intent in intents:
        symbol = str(getattr(intent, "symbol", "") or "")
        strategy = str(getattr(intent, "strategy", "") or "")

        # Look up cache entry
        cache_key = f"{symbol}|{strategy}"
        entry = confidence_cache.get(cache_key)

        if not isinstance(entry, dict):
            surviving.append(intent)
            continue

        # Check freshness
        ts_str = entry.get("ts_utc", "")
        try:
            cached_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age > _INTELLIGENCE_CACHE_MAX_AGE_SEC:
                surviving.append(intent)
                continue
        except Exception:
            surviving.append(intent)
            continue

        adjustment = float(entry.get("adjustment", 0.0))
        base_confidence = float(getattr(intent, "confidence", 0.5) or 0.5)
        adjusted = base_confidence + adjustment

        if adjusted < _MIN_CONFIDENCE_THRESHOLD:
            logger.info(
                "SUPPRESSED_BY_INTELLIGENCE %s %s base=%.3f adj=%.3f threshold=%.3f reason=%s",
                symbol, strategy, base_confidence, adjusted,
                _MIN_CONFIDENCE_THRESHOLD, entry.get("reason", ""),
            )
            continue

        surviving.append(intent)

    return surviving


# ---------------------------------------------------------------------------
# Paper-fill safety gate — guards FILLS_*.ndjson against unconfirmed IBKR
# submissions that may be rejected asynchronously (e.g. Error 201 open-order
# cap). Without this gate, an IBKR STK/FUT order returning status=PendingSubmit
# is normalized to status=paper_fill in paper_exec_evidence_writer and the
# fake fill persists even after IBKR cancels the order.
# ---------------------------------------------------------------------------
_UNCONFIRMED_BROKER_STATUSES = frozenset({
    "pendingsubmit", "presubmitted", "submitted", "apipending",
    "inactive", "unknown", "", "error", "failed", "rejected",
    "cancelled", "duplicate_blocked",
    "duplicate_open_order", "suppressed_open_orders_cap",
})


def _is_explicit_paper_simulator(bag_extra: Any) -> bool:
    """True when the intent carries explicit BAG/options-spread metadata that
    licenses paper-fill evidence regardless of IBKR order acceptance.

    The alpha_options BAG simulator in paper_exec_evidence_writer rewrites the
    fill from the strategy-supplied net_debit_estimate and never depends on a
    confirmed IBKR fill, so its evidence is trustworthy even when the wrapping
    STK proxy submission returns PendingSubmit.
    """
    if not isinstance(bag_extra, dict) or not bag_extra:
        return False
    sec = str(bag_extra.get("sec_type", "") or "").strip().upper()
    if sec in ("BAG", "COMBO"):
        return True
    req_ac = str(bag_extra.get("required_asset_class", "") or "").strip().lower()
    if req_ac == "options" and any(
        k in bag_extra for k in ("long_strike", "short_strike", "net_debit_estimate")
    ):
        return True
    return False


def _should_persist_paper_evidence(order: Any, bag_extra: Any) -> Tuple[bool, Optional[str]]:
    """Decide whether a SubmittedOrder result warrants a paper_fill record.

    Returns ``(persist, skip_reason)``. ``skip_reason`` is None when persist=True.

    Rule: IBKR submissions whose synchronous adapter status is unconfirmed
    (PendingSubmit/PreSubmitted/Submitted/Inactive/Unknown/Error/...) MUST NOT
    produce paper_fill evidence — IBKR may asynchronously reject or cancel
    the order (e.g. Error 201 open-order cap) and the fake fill would persist
    in FILLS_*.ndjson and feed bogus realized PnL into SCR / trade_closer.

    Exception: explicit paper-only simulators (alpha_options BAG/COMBO with
    leg metadata) own their own fill semantics and are licensed to write
    paper_fill regardless of the synchronous adapter status.
    """
    raw_status = str(getattr(order, "status", "") or "").strip().lower()
    if raw_status in _UNCONFIRMED_BROKER_STATUSES:
        if _is_explicit_paper_simulator(bag_extra):
            return True, None
        return False, f"unconfirmed_order_status:{raw_status or 'empty'}"
    return True, None


def run_once(logger: logging.Logger) -> None:
    """
    Execute one CHAD live cycle.

    Guard precedence order (SSOT v6.4):
    1. Broker truth rebuild — reconcile local guard against broker positions
    2. Strategy routing — evaluate and select strategy signals
    3. Intent planning — pipeline: strategies → policy → routing → plan → intents
    4. Position guard — block same-side duplicate positions (MAINTAINED state)
    5. Signal dedup — cooldown check on non-flip signals (COOLDOWN_BLOCKED state)
    6. Execution — flip (replace_position / FLIPPED) or open (mark_position_open / OPEN)

    This ordering is load-bearing: broker truth MUST run before any guard
    check so that stale local state does not suppress valid signals or
    allow phantom positions.
    """
    # GAP-039 (Phase-58/59): evaluate halt triggers FIRST, then check the
    # early-return. The previous order (early-return BEFORE evaluate)
    # caused GAP-034's auto-recovery counter to be unreachable once the
    # bus was latched — the cycle returned before evaluate_and_persist
    # ever ran, so the clean-streak hysteresis never incremented and the
    # bus stayed latched indefinitely.
    #
    # Under the new order:
    #   - A freshly-firing trigger writes runtime/stop_bus.json here and
    #     is then halted by the is_stop_bus_active() early-return below.
    #   - A latched-but-clean bus is auto-cleared here by the GAP-034
    #     hysteresis helper inside evaluate_and_persist, and the
    #     subsequent is_stop_bus_active() check falls through so the
    #     cycle resumes normally.
    #   - A still-latched bus stays active and is halted by the same
    #     early-return.
    #
    # Snapshot inputs (pnl_state.json, CHAD_DAILY_LOSS_LIMIT env,
    # ibkr_status.json) are written by independent publishers and do
    # NOT depend on the guard-rebuild that runs later in this cycle.
    try:
        from chad.risk.stop_bus_state import evaluate_and_persist
        snapshot = _build_stop_bus_snapshot(logger)
        sb_config = _build_stop_bus_config()
        sb_result = evaluate_and_persist(
            snapshot=snapshot,
            config=sb_config,
            triggered_by="live_loop.run_once",
        )
        if sb_result.get("any_active"):
            logger.warning(
                "STOP_BUS_TRIGGERED cycle=halted triggers=%s reason=%s",
                sb_result.get("active_triggers", []),
                sb_result.get("reason", ""),
            )
            # Fire Telegram alert — dedupe key in telegram_notify suppresses
            # repeat sends for the same STOP event within the TTL window.
            try:
                from chad.utils.telegram_notify import send_stop_bus_alert
                _sb_reason = sb_result.get("reason") or ",".join(
                    str(t) for t in (sb_result.get("active_triggers") or [])
                )
                send_stop_bus_alert(_sb_reason or "stop_bus_triggered")
            except Exception:
                pass
            # Fall through to the single is_stop_bus_active early-return
            # below — it observes the just-persisted bus state and halts.
    except Exception as _sbt_err:  # noqa: BLE001
        logger.warning("stop_bus evaluation failed (non-fatal): %s", _sbt_err)

    # Phase-8 Session 4 (R2): honour an operator- or trigger-set STOP bus.
    # A truthy runtime/stop_bus.json halts the cycle before any rebuild or
    # routing so that reconciliation side-effects are not applied while
    # trading is stopped. clear_stop_bus() (from chad.risk.stop_bus_state
    # or the CLI script) is required to resume. Single halt path — both
    # freshly-fired and still-latched buses exit here.
    try:
        from chad.risk.stop_bus_state import is_stop_bus_active, read_stop_bus
        if is_stop_bus_active():
            state = read_stop_bus()
            logger.warning(
                "STOP_BUS_ACTIVE reason=%s triggered_by=%s triggered_at=%s — skipping cycle",
                state.get("reason", ""),
                state.get("triggered_by", ""),
                state.get("triggered_at", ""),
            )
            return
    except Exception as _sb_err:  # noqa: BLE001
        logger.warning("stop_bus check failed (non-fatal): %s", _sb_err)

    # P1-3: Rebuild position guard before any signal evaluation. In paper /
    # dry_run mode reconcile against the local trade_closer ledger so the
    # simulated positions survive across cycles; only query IB Gateway when
    # actually live.
    try:
        if _is_paper_mode():
            _rebuild_guard_from_paper_ledger(logger)
            # Also reconcile broker_sync entries against actual IBKR
            # paper positions. Strategy positions live in the paper
            # ledger; broker_sync entries must reflect IBKR truth so
            # reconciliation reports accurate drift.
            try:
                _rebuild_guard_from_broker(logger)
            except Exception as bexc:
                logger.warning("Broker guard rebuild failed in paper mode (non-fatal): %s", bexc)
        else:
            _rebuild_guard_from_broker(logger)
    except Exception as exc:
        logger.warning("Guard rebuild failed (non-fatal): %s", exc)

    strategy_detail: Dict[str, Any] = {
        "available_strategies": {},
        "rejected_strategies": {},
        "selected_strategy": None,
        "selected_strategy_reason": None,
        "affordability_rejections": [],
        "guard_rejections": [],
    }

    # GAP-025: routing diagnostics tracker. Observational only — never
    # affects routing decisions. Construction failure-soft so any error
    # leaves the cycle unaffected (writer falls back to disk-only data).
    _routing_diag = None
    try:
        from chad.ops.strategy_routing_diagnostics import (
            RoutingDiagnostics as _RoutingDiagnostics,
        )
        _routing_diag = _RoutingDiagnostics()
        # CHAD has no dedicated per-strategy spam governor; the
        # signal_throttle global trim is observed at its own stage but
        # is not strategy-specific, so the spam_governor stage is not
        # present in this code path.
        _routing_diag.mark_stage_not_present("signals_after_spam_governor")
    except Exception:  # noqa: BLE001
        _routing_diag = None

    # Phase-8 Session 5 (F4): edge decay monitor — halt strategies whose
    # recent trades show a run of consecutive losses. Writes
    # runtime/strategy_allocations.json atomically. Non-fatal — a broken
    # monitor must not stop the cycle. Runs BEFORE stage-4 signal
    # building so the halt set can filter newly-emitted signals
    # (FINDING-2 fix).
    _halted_strategies: set = set()
    try:
        from chad.risk.edge_decay_monitor import EdgeDecayMonitor, read_allocations as _edm_read
        decay_results = EdgeDecayMonitor().check_all()
        halted = [s for s, v in decay_results.items() if v.get("halted")]
        # Refresh halted set from the persisted file so previously-halted
        # strategies (written in earlier cycles, no new trade since) are
        # still enforced this cycle.
        try:
            _edm_state = _edm_read()
            for _k, _v in (_edm_state.get("allocations") or {}).items():
                if isinstance(_v, dict) and _v.get("halted"):
                    halted_set_member = str(_k).strip().lower()
                    if halted_set_member:
                        _halted_strategies.add(halted_set_member)
        except Exception:
            pass
        for _h in halted:
            _halted_strategies.add(str(_h).strip().lower())
        if halted:
            logger.warning(
                "EDGE_DECAY_REPORT halted_strategies=%s (total_evaluated=%d)",
                halted, len(decay_results),
            )
            # Edge-trigger-only Telegram alerts (FINDING-4 fix): only
            # fire on strategies not already alerted this process
            # lifetime. The notify helper still dedupes by strategy
            # name as a second line of defence.
            try:
                from chad.utils.telegram_notify import send_edge_decay_alert
                for _strat in halted:
                    _info = decay_results.get(_strat) or {}
                    _losses = int(
                        _info.get("consecutive_neg")
                        or _info.get("consecutive_losses")
                        or _info.get("streak")
                        or 0
                    )
                    if _strat not in _EDGE_DECAY_ALERTED:
                        send_edge_decay_alert(str(_strat), _losses)
                        _EDGE_DECAY_ALERTED.add(_strat)
            except Exception:
                pass
        # Clear alert-tracking entries for strategies that are no longer
        # halted (operator cleared via scripts/clear_edge_decay.py),
        # so subsequent re-halts can re-alert.
        # Use intersection_update (in-place mutation) rather than `-=`:
        # augmented assignment would mark the module-global as a function
        # local for the entire run_once frame, raising UnboundLocalError
        # on the earlier reads at lines 800/802 above.
        _currently_halted = set(halted)
        _EDGE_DECAY_ALERTED.intersection_update(_currently_halted)
    except Exception as _decay_err:  # noqa: BLE001
        logger.warning("edge_decay_monitor failed (non-fatal): %s", _decay_err)

    def _signal_strategy_name(sig) -> str:
        v = getattr(sig, "strategy", None)
        v = getattr(v, "value", v)
        return str(v or "").strip().lower()

    # Signal throttle — written by health_monitor when churn detected.
    # Detected here so routing runs on the full signal set; trim happens
    # AFTER stage-4 so only submission is capped.
    _max_signals_this_cycle = None
    try:
        _throttle_path = Path("/home/ubuntu/chad_finale/runtime/signal_throttle.json")
        if _throttle_path.exists():
            _thr = json.loads(_throttle_path.read_text(encoding="utf-8"))
            if _thr.get("active"):
                _expires_raw = _thr.get("auto_expires_at_utc", "")
                _expired = False
                if _expires_raw:
                    try:
                        from datetime import datetime, timezone
                        _exp_dt = datetime.fromisoformat(
                            str(_expires_raw).replace("Z", "+00:00")
                        )
                        if datetime.now(timezone.utc) > _exp_dt:
                            _expired = True
                            _thr["active"] = False
                            _throttle_path.write_text(
                                json.dumps(_thr, indent=2),
                                encoding="utf-8",
                            )
                            # SS05: emit audit-grade warning + append-only log
                            # so post-mortems can reconstruct exactly when the
                            # throttle deactivated and why.
                            logger.warning(
                                "SIGNAL_THROTTLE_DEACTIVATED "
                                "reason=expired "
                                "activated_at=%s "
                                "deactivated_at=%s "
                                "trades_during_throttle=UNKNOWN",
                                _thr.get("activated_at_utc", "?"),
                                datetime.now(timezone.utc).isoformat(),
                            )
                            _log_throttle_audit(
                                event="deactivated",
                                reason="expired",
                                thr_data=_thr,
                            )
                    except Exception:
                        pass
                if not _expired:
                    _max_signals_this_cycle = int(
                        _thr.get("max_signals_per_cycle", 3)
                    )
                    logger.warning(
                        "SIGNAL_THROTTLE_ACTIVE max=%d reason=%s",
                        _max_signals_this_cycle,
                        _thr.get("reason", "unknown"),
                    )
    except Exception:
        pass

    # Weekend market hours gate — equity strategies do not trade on
    # Saturday/Sunday (UTC). Crypto strategies (alpha_crypto, omega_vol,
    # kraken) trade 24/7. Setup is here; filter applied after stage-4.
    from datetime import datetime, timezone
    _now_utc = datetime.now(timezone.utc)
    _is_weekend = _now_utc.weekday() >= 5  # 5=Saturday, 6=Sunday
    CRYPTO_STRATEGIES = {"alpha_crypto", "omega_vol", "kraken", "crypto"}
    EQUITY_BLOCKED_ON_WEEKEND = True

    try:
        if is_always_active_routing():
            all_result = build_all_live_signals(logger)
            routed_signals = all_result.all_signals
            # GAP-025: record initial (pre-halt-filter) signal count.
            if _routing_diag is not None:
                try:
                    _routing_diag.observe_signals(
                        "signals_generated_this_cycle", routed_signals
                    )
                except Exception:
                    pass
            _pre_halt_signals = list(routed_signals or [])
            if _halted_strategies and routed_signals:
                _before_count = len(routed_signals)
                routed_signals = [
                    s for s in routed_signals
                    if _signal_strategy_name(s) not in _halted_strategies
                ]
                _dropped = _before_count - len(routed_signals)
                if _dropped:
                    logger.warning(
                        "EDGE_DECAY_FILTERED dropped=%d halted=%s",
                        _dropped, sorted(_halted_strategies),
                    )
            # GAP-025: post-halt-filter count + drop attribution.
            if _routing_diag is not None:
                try:
                    _routing_diag.observe_drop(
                        _pre_halt_signals, routed_signals or [], "edge_decay"
                    )
                    _routing_diag.observe_signals(
                        "signals_after_edge_decay_or_halt_filter",
                        routed_signals,
                    )
                except Exception:
                    pass
            routed_signal_map = _build_router_signal_map(list(routed_signals or []))
            decision = all_result.decision
            strategy_detail["available_strategies"] = decision.available_counts
            strategy_detail["rejected_strategies"] = decision.rejected_strategies
            strategy_detail["selected_strategy"] = decision.primary_strategy
            strategy_detail["selected_strategy_reason"] = decision.reason
            _write_route_decision(strategy_detail)

            if not routed_signals:
                logger.info(
                    "CYCLE_IDLE suppression=%s strategies_checked=%s",
                    SuppressionReason.NO_SIGNAL.value,
                    strategy_detail["available_strategies"],
                    extra={
                        "suppression_reason": SuppressionReason.NO_SIGNAL.value,
                        "routing_mode": "always_active",
                        "strategies_checked": strategy_detail["available_strategies"],
                    },
                )
                return
        else:
            result = build_live_signals(logger)
            routed_signals = result.signals
            # GAP-025: record initial (pre-halt-filter) signal count.
            if _routing_diag is not None:
                try:
                    _routing_diag.observe_signals(
                        "signals_generated_this_cycle", routed_signals
                    )
                except Exception:
                    pass
            _pre_halt_signals = list(routed_signals or [])
            if _halted_strategies and routed_signals:
                _before_count = len(routed_signals)
                routed_signals = [
                    s for s in routed_signals
                    if _signal_strategy_name(s) not in _halted_strategies
                ]
                _dropped = _before_count - len(routed_signals)
                if _dropped:
                    logger.warning(
                        "EDGE_DECAY_FILTERED dropped=%d halted=%s",
                        _dropped, sorted(_halted_strategies),
                    )
            # GAP-025: post-halt-filter count + drop attribution.
            if _routing_diag is not None:
                try:
                    _routing_diag.observe_drop(
                        _pre_halt_signals, routed_signals or [], "edge_decay"
                    )
                    _routing_diag.observe_signals(
                        "signals_after_edge_decay_or_halt_filter",
                        routed_signals,
                    )
                except Exception:
                    pass
            routed_signal_map = _build_router_signal_map(list(routed_signals or []))
            decision = result.decision
            strategy_detail["available_strategies"] = decision.available_counts
            strategy_detail["rejected_strategies"] = decision.rejected_strategies or {}
            strategy_detail["selected_strategy"] = decision.selected_strategy
            strategy_detail["selected_strategy_reason"] = decision.reason
            _write_route_decision(strategy_detail)

            if not routed_signals:
                logger.info(
                    "CYCLE_IDLE suppression=%s strategies_checked=%s",
                    SuppressionReason.NO_SIGNAL.value,
                    strategy_detail["available_strategies"],
                    extra={
                        "suppression_reason": SuppressionReason.NO_SIGNAL.value,
                        "routing_mode": "single_winner",
                        "strategies_checked": strategy_detail["available_strategies"],
                    },
                )
                return
    except Exception as _stage4_exc:
        logger.error(
            "live_loop_stage4_failed cycle=%s err=%s — skipping execution",
            '?',
            _stage4_exc,
        )
        return []

    # ------------------------------------------------------------------
    # Gap-1 (v9.1 audit): tier instrument allowlist enforcement. Runs as
    # the first gate after raw TradeSignal aggregation and BEFORE all
    # downstream gates and intent building. Strategies whose internal
    # universe ignores the active tier (alpha_crypto BTC/ETH/SOL) can no
    # longer reach the router on a tier that excludes their symbols.
    # Fail-soft: any exception leaves routed_signals unchanged. The gate
    # itself fails open on tier_state.json read/parse errors.
    # ------------------------------------------------------------------
    try:
        _pre_tier_count = len(routed_signals or [])
        _allowed_sigs, _blocked_sigs = _TIER_INSTRUMENT_GATE.filter_signals(
            list(routed_signals or [])
        )
        routed_signals = _allowed_sigs
        if _blocked_sigs:
            logger.warning(
                "TIER_INSTRUMENT_GATE blocked=%d remaining=%d symbols=%s",
                len(_blocked_sigs),
                len(routed_signals),
                sorted({
                    str(getattr(s, "symbol", "") or "")
                    for s in _blocked_sigs
                }),
            )
        if _routing_diag is not None:
            try:
                _routing_diag.observe_signals(
                    "signals_after_tier_instrument_gate", routed_signals
                )
            except Exception:
                pass
    except Exception as _tier_gate_err:  # noqa: BLE001
        logger.warning(
            "tier_instrument_gate_failed err=%s — proceeding without filter",
            _tier_gate_err,
        )

    # Weekend gate — drop equity signals on Saturday/Sunday UTC; crypto
    # strategies pass through unchanged. Pipeline B (equity intent builder
    # downstream) does NOT consult this gate, so the log line is explicitly
    # named WEEKEND_GATE_A_FILTERED to make the A/B asymmetry visible to
    # operators reading strategy diagnostics.
    if _is_weekend and routed_signals:
        _pre_weekend = list(routed_signals)
        routed_signals = [
            s for s in routed_signals
            if _signal_strategy_name(s) in CRYPTO_STRATEGIES
            or not EQUITY_BLOCKED_ON_WEEKEND
        ]
        _filtered_count = len(_pre_weekend) - len(routed_signals)
        if _filtered_count > 0:
            _kept_ids = {id(s) for s in routed_signals}
            _dropped_symbols = sorted({
                str(getattr(s, "symbol", "") or "")
                for s in _pre_weekend if id(s) not in _kept_ids
            })
            logger.info(
                "WEEKEND_GATE_A_FILTERED dropped=%d symbols=%s "
                "(pipeline_a only; pipeline_b equity intents not gated here)",
                _filtered_count, _dropped_symbols,
            )

    # ------------------------------------------------------------------
    # GAP-026: per-strategy daily realized PnL guard. Report-only by
    # default — every cycle a WARNING line is logged for any strategy
    # whose today-window realized PnL is at/below its configured loss
    # limit. CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1 turns this into a
    # suppression that drops fresh entry signals from breached
    # strategies; exits / closes / reductions / hedges always pass.
    # The today-window is anchored to the active epoch start so prior
    # epoch losses cannot trigger current-day suppression. Failure-soft.
    # ------------------------------------------------------------------
    _loss_guard = None
    try:
        from chad.risk.per_strategy_loss_guard import (
            PerStrategyLossGuard as _PerStrategyLossGuard,
        )
        _loss_guard = _PerStrategyLossGuard()
        _loss_guard.report(logger)
        _pre_lg_signals = list(routed_signals or [])
        if _loss_guard.enforce and routed_signals:
            _pre_lg = len(routed_signals)
            routed_signals, _ = _loss_guard.filter_signals(
                list(routed_signals), logger
            )
            _lg_suppressed = _pre_lg - len(routed_signals)
            if _lg_suppressed:
                logger.warning(
                    "PER_STRATEGY_LOSS_GUARD enforce_suppressed=%d remaining=%d",
                    _lg_suppressed, len(routed_signals),
                )
        # GAP-025: post-loss-guard count + drop attribution (only attributes
        # drops when enforce mode actually filtered). Always recorded so the
        # stage shows up as observed in the diagnostic, even in report-only
        # mode (with zero drops).
        if _routing_diag is not None:
            try:
                _routing_diag.observe_drop(
                    _pre_lg_signals,
                    routed_signals or [],
                    "per_strategy_loss_limit",
                )
                _routing_diag.observe_signals(
                    "signals_after_loss_guard_report_only_or_enforced",
                    routed_signals,
                )
            except Exception:
                pass
    except Exception as _lg_err:  # noqa: BLE001
        logger.debug(
            "per_strategy_loss_guard failed (non-fatal): %s", _lg_err
        )

    # Signal throttle — apply trim AFTER routing decisions are made so
    # only submission is capped. Exits/risk-reducing signals are NEVER
    # trimmed; only fresh entries are capped at _max_signals_this_cycle.
    if _max_signals_this_cycle is not None:
        try:
            if routed_signals:
                _EXIT_INTENT_TYPES = {
                    "exit", "stop_loss", "reduce", "liquidation",
                    "hedge", "close", "risk_reduction",
                }
                _protected = []
                _fresh_entries = []
                for _sig in routed_signals:
                    _sig_meta = getattr(_sig, "meta", {}) or {}
                    _sig_tags = getattr(_sig, "tags", None) or ()
                    _meta_tags = (
                        _sig_meta.get("tags", _sig_meta.get("signal_tags", []))
                        if isinstance(_sig_meta, dict) else []
                    )
                    if not isinstance(_meta_tags, list):
                        _meta_tags = []
                    _all_tags = list(_sig_tags) + list(_meta_tags)
                    _is_exit = (
                        (isinstance(_sig_meta, dict) and (
                            _sig_meta.get("exit")
                            or _sig_meta.get("reason") == "max_hold_exit"
                            or _sig_meta.get("intent") in (
                                "exit", "close", "reduce", "hedge"
                            )
                        ))
                        or any(
                            str(t).lower() in _EXIT_INTENT_TYPES
                            for t in _all_tags
                        )
                    )
                    if _is_exit:
                        _protected.append(_sig)
                    else:
                        _fresh_entries.append(_sig)
                # Sort fresh entries deterministically:
                # confidence DESC, created_at ASC
                _fresh_entries.sort(
                    key=lambda s: (
                        -float(getattr(s, "confidence", 0.5) or 0.5),
                        str(getattr(s, "created_at", "") or ""),
                    )
                )
                _trimmed = _fresh_entries[:_max_signals_this_cycle]
                if len(_fresh_entries) > _max_signals_this_cycle:
                    logger.warning(
                        "SIGNAL_THROTTLE_TRIMMED entries=%d→%d protected=%d",
                        len(_fresh_entries), len(_trimmed), len(_protected),
                    )
                routed_signals = _protected + _trimmed
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Net Exposure Conflict Gate — runs after throttle trim, before
    # intent building. Prevents strategies from fighting each other
    # while preserving hedges, reductions, and reversals. Failure-soft:
    # any error allows all signals through.
    # ------------------------------------------------------------------
    try:
        from chad.execution.net_exposure_gate import run_gate as _run_net_exposure_gate
        _pre_gate_count = len(routed_signals or [])
        _pre_neg_signals = list(routed_signals or [])
        routed_signals, _gate_decisions = _run_net_exposure_gate(list(routed_signals or []))
        _blocked = sum(
            1 for d in _gate_decisions
            if d.action.value == "BLOCK"
        )
        if _blocked:
            logger.warning(
                "NET_EXPOSURE_GATE blocked=%d remaining=%d",
                _blocked, len(routed_signals),
            )
        # GAP-025: post-net-exposure count + drop attribution.
        if _routing_diag is not None:
            try:
                _routing_diag.observe_drop(
                    _pre_neg_signals,
                    routed_signals or [],
                    "net_exposure",
                )
                _routing_diag.observe_signals(
                    "signals_after_net_exposure_gate", routed_signals
                )
            except Exception:
                pass
    except Exception as _gate_err:
        logger.debug(
            "net_exposure_gate_failed err=%s — proceeding without gate",
            _gate_err,
        )
        _gate_decisions = []

    # ------------------------------------------------------------------
    # BG11 — FLIP_ALLOWED close-first / open-second enforcement.
    # For every FLIP_ALLOWED gate decision, submit the close intent for
    # the conflicting position FIRST. The new flipped entry is only
    # permitted if the close is broker-confirmed; otherwise it is
    # dropped and position_guard is left untouched (no false-flat).
    # ------------------------------------------------------------------
    try:
        from chad.core.flip_executor import enforce_flip_close_first
        _flip_pre_count = len(routed_signals or [])
        routed_signals, _flip_audit = enforce_flip_close_first(
            list(routed_signals or []),
            _gate_decisions or [],
            _paper_adapter,
        )
        _flip_dropped = _flip_pre_count - len(routed_signals)
        if _flip_dropped:
            logger.warning(
                "BG11_FLIP_EXECUTOR dropped=%d remaining=%d",
                _flip_dropped, len(routed_signals),
            )
        if _flip_audit:
            try:
                _audit_path = Path(
                    "/home/ubuntu/chad_finale/runtime/flip_executor_audit.json"
                )
                with open(_audit_path, "a", encoding="utf-8") as _fh:
                    for _row in _flip_audit:
                        _fh.write(json.dumps(_row, default=str) + "\n")
            except Exception:
                pass
    except Exception as _flip_err:
        logger.debug(
            "flip_executor_failed err=%s — flipped entries kept; gate "
            "still enforces priority. Manual review required.",
            _flip_err,
        )

    # ------------------------------------------------------------------
    # Smart Strategy Throttle Gate — performance-aware time-window
    # throttle. Sits between net exposure gate and intent building.
    # Winning strategies pass unrestricted; losing strategies are
    # progressively throttled. Exits/protectives never blocked.
    # Failure-soft.
    # ------------------------------------------------------------------
    try:
        from chad.execution.strategy_throttle_gate import (
            run_throttle_gate as _run_throttle_gate,
        )
        _pre_throttle_signals = list(routed_signals or [])
        routed_signals, _throttle_decisions = _run_throttle_gate(
            list(routed_signals or [])
        )
        _throttled = sum(
            1 for d in _throttle_decisions
            if d.level.value not in ("ALLOW", "HALT_DEFER_TO_EDGE_DECAY")
        )
        if _throttled:
            logger.warning(
                "STRATEGY_THROTTLE_GATE throttled=%d remaining=%d",
                _throttled, len(routed_signals),
            )
        # GAP-025: post-strategy-throttle count + drop attribution.
        if _routing_diag is not None:
            try:
                _routing_diag.observe_drop(
                    _pre_throttle_signals,
                    routed_signals or [],
                    "strategy_throttle",
                )
                _routing_diag.observe_signals(
                    "signals_after_strategy_throttle", routed_signals
                )
            except Exception:
                pass
    except Exception as _throttle_err:
        logger.debug(
            "strategy_throttle_gate_failed err=%s — proceeding",
            _throttle_err,
        )

    # ------------------------------------------------------------------
    # Position reconciler — close open positions when the net strategy
    # signal direction flips. Runs BEFORE intent planning so close
    # intents bypass the pipeline's netting step and cannot be
    # suppressed by same-symbol opposite-side new signals.
    # ------------------------------------------------------------------
    try:
        from chad.core.position_reconciler import (
            apply_close_intents,
            load_open_positions,
            reconcile_positions_with_signals,
        )

        _open_positions = load_open_positions()
        _reconciler_closes = reconcile_positions_with_signals(
            open_positions=_open_positions,
            routed_signals=list(routed_signals or []),
            prices={},
        )
        if _reconciler_closes:
            logger.info(
                "reconciler_close_intents count=%d symbols=%s",
                len(_reconciler_closes),
                [c["symbol"] for c in _reconciler_closes],
            )
            apply_close_intents(_reconciler_closes, _paper_adapter)
    except Exception as _rec_err:  # noqa: BLE001
        logger.warning("position_reconciler failed (non-fatal): %s", _rec_err)

    # ------------------------------------------------------------------
    # Phase-8 Session 6 (G1 feed): publish runtime/market_metrics.json from
    # current daily bars so the regime classifier sees real vol / ADX /
    # trend_slope / breadth inputs rather than defaulting to 'unknown'.
    # Non-fatal — classifier already degrades gracefully without inputs.
    # ------------------------------------------------------------------
    try:
        from chad.analytics.market_metrics_publisher import MarketMetricsPublisher
        MarketMetricsPublisher().compute_and_publish()
    except Exception as _mmp_err:  # noqa: BLE001
        logger.warning("market_metrics_publisher failed (non-fatal): %s", _mmp_err)

    # ------------------------------------------------------------------
    # Phase-8 Session 4 (G1 + G3): regime classification and transition-
    # driven position reduction. Both are non-fatal — a broken classifier
    # must not stop the trading cycle.
    # ------------------------------------------------------------------
    try:
        from chad.analytics.regime_classifier import (
            classify_regime,
            read_regime_state,
            write_regime_state,
        )
        from chad.risk.regime_reduction import handle_regime_transition

        previous_state = read_regime_state()
        previous_regime = previous_state.get("regime")

        # Inputs are best-effort — missing inputs degrade to 'unknown'.
        regime_result = classify_regime(
            realized_vol_percentile=_load_optional_metric("realized_vol_percentile"),
            adx=_load_optional_metric("adx"),
            trend_slope=_load_optional_metric("trend_slope"),
            market_breadth=_load_optional_metric("market_breadth"),
        )
        new_state = write_regime_state(regime_result, source="live_loop.run_once", ttl_seconds=360)
        logger.info(
            "REGIME_CLASSIFIED regime=%s confidence=%.2f inputs=%s previous=%s",
            new_state.get("regime"),
            float(new_state.get("confidence") or 0.0),
            new_state.get("inputs_used"),
            new_state.get("previous_regime"),
        )

        if previous_regime and previous_regime != new_state.get("regime"):
            transition = handle_regime_transition(
                from_regime=previous_regime,
                to_regime=new_state.get("regime"),
                open_positions=load_open_positions(),
            )
            close_intents = transition.get("close_intents") or []
            if close_intents:
                apply_close_intents(close_intents, _paper_adapter)
    except Exception as _reg_err:  # noqa: BLE001
        logger.warning("regime classifier/reduction failed (non-fatal): %s", _reg_err)

    # ------------------------------------------------------------------
    # Choppy regime defensive gate. Per-signal asset-class scoping (BG13):
    # crypto/forex are always exempt; equity/options/futures evaluate the
    # choppy overlay against their own asset_class. Stale state applies a
    # conservative confidence floor (only to non-exempt asset classes).
    # Exits and protectives always pass.
    # ------------------------------------------------------------------
    if routed_signals:
        try:
            from chad.analytics.choppy_regime_detector import get_choppy_state
            _pre_choppy = len(routed_signals)
            _choppy_allowed = []
            for _sig in routed_signals:
                _sig_meta = getattr(_sig, "meta", {}) or {}
                _is_exit = (
                    isinstance(_sig_meta, dict)
                    and (
                        _sig_meta.get("exit")
                        or _sig_meta.get("reason") == "max_hold_exit"
                        or _sig_meta.get("intent") in ("exit", "close", "reduce", "hedge")
                    )
                )
                if _is_exit:
                    _choppy_allowed.append(_sig)
                    continue

                # BG13: derive asset class per-signal with safe fallbacks.
                _sig_ac = getattr(_sig, "asset_class", None)
                if _sig_ac is not None and hasattr(_sig_ac, "value"):
                    _sig_ac = _sig_ac.value
                if not _sig_ac and isinstance(_sig_meta, dict):
                    _sig_ac = (
                        _sig_meta.get("asset_class")
                        or _sig_meta.get("required_asset_class")
                    )
                _sig_ac_str = str(_sig_ac or "equity").lower()

                # Per-signal choppy state — crypto/forex short-circuit to
                # choppy_exempt=True inside get_choppy_state.
                try:
                    _sig_choppy = get_choppy_state(asset_class=_sig_ac_str)
                except Exception:
                    _sig_choppy = {"choppy_active": False, "choppy_score": 0.0}

                if _sig_choppy.get("choppy_exempt"):
                    _choppy_allowed.append(_sig)
                    continue

                # Stale → conservative floor; active → normal floor; else
                # the gate is a no-op for this signal.
                if _sig_choppy.get("stale"):
                    _conf_add = 0.10
                elif _sig_choppy.get("choppy_active"):
                    _conf_add = 0.15
                else:
                    _choppy_allowed.append(_sig)
                    continue

                _sig_conf = float(getattr(_sig, "confidence", 0.5) or 0.5)
                _required = 0.5 + _conf_add
                if _sig_conf < _required:
                    logger.info(
                        "CHOPPY_GATE_BLOCK symbol=%s strategy=%s asset=%s conf=%.2f required=%.2f",
                        getattr(_sig, "symbol", "?"),
                        getattr(_sig, "primary_strategy", None)
                        or getattr(_sig, "strategy", "?"),
                        _sig_ac_str,
                        _sig_conf,
                        _required,
                    )
                    continue
                _choppy_allowed.append(_sig)
            routed_signals = _choppy_allowed
            if len(routed_signals) < _pre_choppy:
                logger.warning(
                    "CHOPPY_GATE filtered=%d remaining=%d",
                    _pre_choppy - len(routed_signals),
                    len(routed_signals),
                )
        except Exception as _cg_err:  # noqa: BLE001
            logger.debug("choppy_gate_failed err=%s — proceeding without gate", _cg_err)

    # Edge decay check moved before stage-4 signal building (FINDING-2 fix)

    # Phase-8 Session 6 (F2): compute retrospective signal decay for any
    # pending entries whose bars are now on disk. Lightweight — only
    # processes entries that have not yet been measured. Non-fatal.
    try:
        from chad.analytics.signal_decay import get_default_recorder as _get_decay_recorder
        _measured = _get_decay_recorder().compute_decay_for_pending()
        if _measured:
            logger.info("SIGNAL_DECAY_MEASURED count=%d", len(_measured))
    except Exception as _sd_err:  # noqa: BLE001
        logger.warning("signal_decay measurement failed (non-fatal): %s", _sd_err)

    # Phase-8 Session 6 (F3): composite strategy-health scorer. Writes
    # runtime/strategy_health.json each cycle and feeds very-low-health
    # strategies back to the edge_decay monitor as an additional halt
    # signal. Non-fatal — a scoring error must not block trading.
    try:
        from chad.analytics import expectancy_tracker as _expectancy
        from chad.analytics.slippage_tracker import get_default_tracker as _get_slippage_tracker
        from chad.analytics.strategy_health import StrategyHealthScorer
        from chad.analytics.regime_classifier import read_regime_state as _read_regime

        regime_now = _read_regime().get("regime", "unknown")
        expectancy_state = _expectancy.compute()
        strategy_names = sorted((expectancy_state.get("strategies") or {}).keys())
        if strategy_names:
            scorer = StrategyHealthScorer()
            health = scorer.compute_all(
                strategy_names=strategy_names,
                expectancy_tracker=expectancy_state,
                slippage_tracker=_get_slippage_tracker(),
                regime_state=str(regime_now),
            )
            low_health = [
                s for s, v in health.items()
                if isinstance(v, dict) and float(v.get("health_score") or 0.5) < 0.2
            ]
            if low_health:
                logger.warning(
                    "STRATEGY_HEALTH_LOW strategies=%s (scored=%d)",
                    low_health, len(health),
                )
    except Exception as _sh_err:  # noqa: BLE001
        logger.warning("strategy_health scorer failed (non-fatal): %s", _sh_err)

    try:
        _ctx, _plan, intents, kraken_intents = _build_plan_and_intents(logger)
    except Exception as _stage11_exc:
        logger.error(
            "live_loop_stage11_failed err=%s — returning zero intents",
            _stage11_exc,
        )
        intents = []
        kraken_intents = []

    try:
        from chad.utils.telegram_notify import send_drawdown_alert
        _portfolio = getattr(_ctx, 'portfolio', None) if '_ctx' in locals() else None
        _equity = float(getattr(_portfolio, 'equity', 0.0) or 0.0)
        _extra = getattr(_portfolio, 'extra', {}) or {}
        _peak = float(_extra.get('equity_peak', _equity) or _equity)
        if _peak > 0 and _equity < _peak:
            _dd_pct = (_equity - _peak) / _peak * 100
            if _dd_pct <= -5.0:
                send_drawdown_alert(_dd_pct, 5.0)
    except Exception:
        pass

    # 2026-04-22 Audit-O fix: gate intents through the regime activation
    # matrix. Fail-open on any error — if the matrix is missing/malformed
    # or the regime label unknown, all intents pass through (current
    # pre-wiring behavior). Applied to both IBKR and Kraken lanes so the
    # matrix governs crypto too.
    try:
        from chad.portfolio.regime_activation import filter_intents_by_regime
        from chad.analytics.regime_classifier import read_regime_state
        _regime_now = str(read_regime_state().get("regime", "unknown"))
        _pre_regime_intents = list(intents or []) + list(kraken_intents or [])
        _kept_ibkr, _dropped_ibkr = filter_intents_by_regime(intents, _regime_now)
        _kept_kr, _dropped_kr = filter_intents_by_regime(kraken_intents, _regime_now)
        if _dropped_ibkr or _dropped_kr:
            logger.info(
                "REGIME_GATE regime=%s ibkr_kept=%d ibkr_dropped=%d kraken_kept=%d kraken_dropped=%d",
                _regime_now,
                len(_kept_ibkr), len(_dropped_ibkr),
                len(_kept_kr), len(_dropped_kr),
            )
            for _intent, _reason in (_dropped_ibkr + _dropped_kr)[:8]:
                logger.info(
                    "REGIME_GATE_DROP strategy=%s symbol=%s reason=%s",
                    getattr(_intent, "strategy", None),
                    getattr(_intent, "symbol", None) or getattr(_intent, "pair", None),
                    _reason,
                )
        intents = list(_kept_ibkr)
        kraken_intents = list(_kept_kr)
        # GAP-025: post-regime-gate count + drop attribution. Stage runs
        # on intents (not raw signals) — the strategy attribute is still
        # populated, so per-strategy attribution is preserved.
        if _routing_diag is not None:
            try:
                _post_regime = list(intents) + list(kraken_intents)
                _routing_diag.observe_drop(
                    _pre_regime_intents, _post_regime, "regime_inactive"
                )
                _routing_diag.observe_signals(
                    "signals_after_regime_gate", _post_regime
                )
            except Exception:
                pass
    except Exception as _rg_err:  # noqa: BLE001
        logger.warning("regime_gate failed (fail-open, non-fatal): %s", _rg_err)

    # Throttled (5-min) refresh of runtime/kraken_balances.json so the
    # advisory engine, daily report, and alpha_crypto's CAD lane all see
    # the live Kraken account state. Non-fatal on any error.
    try:
        _refresh_kraken_balance_snapshot(logger)
    except Exception as bex:  # noqa: BLE001
        logger.warning("Kraken balance snapshot refresh failed (non-fatal): %s", bex)

    # ------------------------------------------------------------------
    # Kraken (CRYPTO) execution lane — gated by LiveGate.kraken_enabled
    # and CHAD_KRAKEN_MODE in {live, paper_kraken}.
    # paper_kraken routes through KrakenExecutor with validate_only=True
    # so fills are simulated against real Kraken prices/spec without
    # placing real orders.
    # ------------------------------------------------------------------
    if kraken_intents:
        try:
            _execute_kraken_intents(logger, kraken_intents)
        except Exception as kex:  # noqa: BLE001
            logger.warning("Kraken execution lane failed (non-fatal): %s", kex)

    if not intents:
        logger.info("No executable IBKR intents.")
        return

    # ------------------------------------------------------------------
    # Phase 9: Optional strategy intelligence confidence bias
    # ------------------------------------------------------------------
    if os.environ.get("CHAD_STRATEGY_INTELLIGENCE_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        intents = _apply_intelligence_bias(intents, logger)

    logger.info("Executing %d intents", len(intents))

    from chad.core.signal_guard import should_emit_signal
    from chad.core.position_guard import (
        is_same_side_open,
        is_flip_signal,
        mark_position_closed,
        mark_position_open,
        replace_position,
    )

    emitted = 0

    # Bug B Fix A: broker truth is read ONCE per run_once() cycle (lazily, on
    # the first FUT open intent) and reused across all intents this cycle.
    # _futures_pending_adds tracks signed quantities of FUT opens APPROVED
    # earlier in this same cycle, so two strategies adding to one symbol in a
    # single cycle project against the cap cumulatively, not independently.
    _futures_truth_obj: Optional[Dict[str, Any]] = None
    _futures_truth_loaded = False
    _futures_pending_adds: Dict[str, float] = {}

    for intent in intents:
        try:
            _attach_strategy_to_intent(intent, routed_signal_map)

            # --- SCR gate (P3-1: hard-block on PAUSED before any state mutation) ---
            try:
                _scr_path = Path("/home/ubuntu/chad_finale/runtime/scr_state.json")
                _scr_raw = json.loads(_scr_path.read_text(encoding="utf-8"))
                _scr_state_val = str(_scr_raw.get("state", "")).upper()
                _scr_sizing = float(_scr_raw.get("sizing_factor", 0.0) or 0.0)
                _scr_reasons = list(_scr_raw.get("reasons") or [])[:3]
                if _scr_state_val == "PAUSED":
                    # P3-3: EXIT bypass — flip intents close existing positions and
                    # must always pass through regardless of SCR state.
                    # A flip signal reverses an open position — it is a closing trade.
                    _is_exit_intent = is_flip_signal(intent)
                    if _is_exit_intent:
                        logger.info(
                            "SCR_EXIT_BYPASS state=PAUSED symbol=%s side=%s qty=%s strategy=%s — exit allowed",
                            getattr(intent, "symbol", None),
                            getattr(intent, "side", None),
                            getattr(intent, "quantity", None),
                            getattr(intent, "strategy", None),
                        )
                    else:
                        logger.warning(
                            "SCR_HARD_BLOCK state=PAUSED sizing_factor=%.3f symbol=%s side=%s qty=%s strategy=%s reasons=%s",
                            _scr_sizing,
                            getattr(intent, "symbol", None),
                            getattr(intent, "side", None),
                            getattr(intent, "quantity", None),
                            getattr(intent, "strategy", None),
                            _scr_reasons,
                        )
                        continue
            except Exception as _scr_err:
                logger.warning("SCR_GATE_READ_FAILED (fail-open): %s", _scr_err)
            # --- end SCR gate ---

            # P3-2: CAUTIOUS scaling — apply sizing_factor to quantity when SCR
            # is CAUTIOUS. Respects minimum quantity per market type.
            # Futures: minimum 1 contract (no fractional). Equities: minimum 1 share.
            try:
                if _scr_state_val == "CAUTIOUS" and _scr_sizing > 0.0:
                    _raw_qty = float(getattr(intent, "quantity", 0.0) or 0.0)
                    _sec_type = str(getattr(intent, "sec_type", "") or "").upper()
                    _scaled_qty: float
                    if _sec_type == "FUT":
                        # Futures: round to nearest whole contract, minimum 1
                        _scaled_qty = max(1.0, round(_raw_qty * _scr_sizing))
                    else:
                        # Equities and other: floor to whole shares, minimum 1
                        import math as _math
                        _scaled_qty = max(1.0, float(_math.floor(_raw_qty * _scr_sizing)))
                    if _scaled_qty != _raw_qty:
                        logger.info(
                            "SCR_CAUTIOUS_SCALE symbol=%s raw_qty=%.2f sizing_factor=%.3f scaled_qty=%.2f",
                            getattr(intent, "symbol", None),
                            _raw_qty,
                            _scr_sizing,
                            _scaled_qty,
                        )
                        try:
                            object.__setattr__(intent, "quantity", _scaled_qty)
                        except (AttributeError, TypeError):
                            intent.quantity = _scaled_qty
            except Exception as _p3_err:
                logger.warning("SCR_CAUTIOUS_SCALE_FAILED (skipped): %s", _p3_err)
            # --- end P3-2 ---

            class _IntentSignalAdapter:
                def __init__(self, obj: object) -> None:
                    self.strategy = getattr(obj, "strategy", None)
                    self.symbol = getattr(obj, "symbol", None)
                    self.side = getattr(obj, "side", None)
                    self.size = float(getattr(obj, "quantity", 0.0) or 0.0)

            adapted = _IntentSignalAdapter(intent)

            if is_same_side_open(intent):
                logger.info(
                    "SKIP suppression=%s strategy=%s → %s %s %s qty=%s",
                    SuppressionReason.SAME_SIDE_POSITION_OPEN.value,
                    getattr(intent, "strategy", None),
                    getattr(intent, "symbol", None),
                    getattr(intent, "sec_type", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
                    extra={
                        "suppression_reason": SuppressionReason.SAME_SIDE_POSITION_OPEN.value,
                        "strategy": getattr(intent, "strategy", None),
                    },
                )
                continue

            # Per-symbol performance blocker — suppress new entries on symbols
            # with 3 consecutive losses. Flip (exit-and-reverse) and explicit
            # EXIT/CLOSE intents always pass through. Fail-open.
            _side_str = str(getattr(intent, "side", "") or "").upper()
            _is_exit_intent = is_flip_signal(intent) or _side_str in {"EXIT", "CLOSE"}
            if not _is_exit_intent and is_symbol_blocked(getattr(intent, "symbol", "") or ""):
                logger.info(
                    "SKIP suppression=symbol_performance_blocked → %s %s",
                    getattr(intent, "symbol", None),
                    getattr(intent, "side", None),
                )
                continue

            if not is_flip_signal(intent):
                if not should_emit_signal(adapted):
                    logger.info(
                        "SKIP suppression=%s → %s %s %s qty=%s",
                        SuppressionReason.COOLDOWN_ACTIVE.value,
                        getattr(intent, "symbol", None),
                        getattr(intent, "sec_type", None),
                        getattr(intent, "side", None),
                        getattr(intent, "quantity", None),
                        extra={"suppression_reason": SuppressionReason.COOLDOWN_ACTIVE.value},
                    )
                    continue

            # --- ML veto loop (shadow always on; hard veto behind flag) ---
            try:
                from chad.analytics.ml_veto_predictor import (
                    score_intent as _ml_score_intent,
                    ACTION_VETO as _ML_ACTION_VETO,
                    INTENT_ENTRY as _ML_INTENT_ENTRY,
                )
                _ml_ctx = {
                    "regime": {"regime": locals().get("_regime_now", "unknown")},
                    "prices": {},
                    "scr": {"sizing_factor": locals().get("_scr_sizing", 0.25)},
                    "strategy_health": {},
                    "portfolio": {
                        "total_equity": float(
                            getattr(getattr(_ctx, "portfolio", None), "equity", 0.0) or 0.0
                        ),
                    },
                    # Pre-computed flip flag — predictor uses this to
                    # classify the intent as protective so enforcement
                    # never blocks an exit, even if tags are missing.
                    "is_flip": bool(is_flip_signal(intent)),
                }
                _shadow = _ml_score_intent(intent, _ml_ctx)
                # Always log at INFO so the shadow soak is captured by
                # default journald collection (DEBUG was previously
                # filtered out in production).
                logger.info(
                    "ML_SHADOW symbol=%s strategy=%s intent_class=%s "
                    "model_version=%s manifest_hash=%s loss_prob=%.3f "
                    "threshold=%.2f would_veto=%s final_action=%s reason=%s",
                    getattr(intent, "symbol", "?"),
                    getattr(intent, "strategy", "?"),
                    _shadow.intent_class,
                    _shadow.model_version,
                    _shadow.manifest_hash,
                    _shadow.prediction,
                    _shadow.threshold,
                    _shadow.would_veto,
                    _shadow.final_action,
                    _shadow.reason,
                )
                if _shadow.final_action == _ML_ACTION_VETO:
                    # Defense in depth: predictor already enforces the
                    # protective-intent + canary + manifest gates, but
                    # we still re-check intent_class here so a future
                    # predictor regression cannot accidentally block
                    # an exit.
                    if _shadow.intent_class == _ML_INTENT_ENTRY:
                        continue
            except Exception:
                pass  # veto failure never blocks trade
            # --- end ML veto loop ---

            emitted += 1

            # GAP-A001: detect exit-intent (max_hold_exit / explicit close)
            # by inspecting the originating signal's meta. Exit-intents must
            # NOT mark the guard open or flip it — they will mark_position_closed
            # AFTER a trusted fill evidence record is written below. This
            # prevents a max_hold SELL from being treated as a flip (which
            # would leave the guard open with the reversed side) and is the
            # close-side counterpart to mark_position_open/replace_position.
            _intent_is_exit = False
            try:
                _exit_intent_strategy = str(getattr(intent, "strategy", "") or "")
                _exit_intent_symbol = str(getattr(intent, "symbol", "") or "")
                _exit_intent_side = str(getattr(intent, "side", "") or "")
                _exit_orig_sig = None
                for _ek, _esig in (routed_signal_map or {}).items():
                    _eks, _eksy, _eksi, _ = _ek
                    if (
                        _eks == _exit_intent_strategy
                        and _eksy == _exit_intent_symbol
                        and _eksi == _exit_intent_side
                    ):
                        _exit_orig_sig = _esig
                        break
                _exit_meta = getattr(_exit_orig_sig, "meta", None) if _exit_orig_sig else None
                if isinstance(_exit_meta, dict):
                    _intent_is_exit = bool(
                        _exit_meta.get("exit")
                        or _exit_meta.get("reason") == "max_hold_exit"
                        or _exit_meta.get("intent") in ("exit", "close", "reduce")
                    )
            except Exception:
                _intent_is_exit = False

            _gate_sec_type = str(getattr(intent, "sec_type", "") or "").upper()

            # --- Bug B Fix A: cumulative broker-truth futures position cap ---
            # PRIMARY guard for futures opens. Consults NO env flag — binds
            # with the env gate (below, redundant backup) ON or OFF. Exits
            # and flips always pass, same classification as the gate. Refuses
            # when the projected net (broker truth + this cycle's approved
            # adds + this intent) would GROW the position beyond the
            # per-symbol cap; reduces pass. Fail-closed when broker truth
            # cannot be verified.
            if (
                _gate_sec_type == "FUT"
                and not _intent_is_exit
                and not is_flip_signal(intent)
            ):
                _cap_symbol = str(getattr(intent, "symbol", "") or "").strip().upper()
                if not _futures_truth_loaded:
                    _futures_truth_obj = _read_broker_truth_for_caps()
                    _futures_truth_loaded = True
                _broker_net = (
                    _futures_net_from_truth(_futures_truth_obj, _cap_symbol)
                    if _futures_truth_obj is not None
                    else None
                )
                _cap_verdict, _cap_net, _cap_projected, _cap_value = _futures_cap_check(
                    _cap_symbol,
                    str(getattr(intent, "side", "") or ""),
                    float(getattr(intent, "quantity", 0.0) or 0.0),
                    _broker_net,
                    _futures_pending_adds,
                )
                if _cap_verdict == "unverified":
                    logger.warning(
                        "FUTURES_POSITION_CAP_UNVERIFIED symbol=%s strategy=%s side=%s "
                        "qty=%s reason=positions_truth_unavailable_or_stale (fail-closed)",
                        _cap_symbol,
                        getattr(intent, "strategy", "?"),
                        getattr(intent, "side", "?"),
                        getattr(intent, "quantity", "?"),
                    )
                    continue
                if _cap_verdict == "block":
                    logger.warning(
                        "FUTURES_POSITION_CAP_BLOCK symbol=%s strategy=%s side=%s qty=%s "
                        "net=%.1f projected=%.1f cap=%d reason=cumulative_cap",
                        _cap_symbol,
                        getattr(intent, "strategy", "?"),
                        getattr(intent, "side", "?"),
                        getattr(intent, "quantity", "?"),
                        _cap_net,
                        _cap_projected,
                        _cap_value,
                    )
                    continue
                # "allow": _futures_cap_check registered the signed qty into
                # _futures_pending_adds — later intents this cycle project
                # against the cap cumulatively.
            # --- end Bug B Fix A cap ---

            # Futures execution off-switch (reversible env stopgap). OPERATOR
            # DECISION: block BOTH sides and ALL intent classes — NO carve-out
            # for exits/flips. Placed before the position-guard mutations below
            # (mark_position_open / replace_position) so a gated FUT/FOP intent
            # can never leave a phantom guard entry. The ibkr_adapter submit
            # chokepoint enforces the same shared predicate as the hard
            # backstop; this early skip merely avoids needless downstream work.
            if (
                _futures_execution_disabled(os.environ)
                and is_futures_sec_type(_gate_sec_type)
            ):
                logger.warning(
                    "FUTURES_EXECUTION_DISABLED_SKIP symbol=%s strategy=%s sec_type=%s "
                    "side=%s qty=%s reason=env_guard",
                    getattr(intent, "symbol", "?"),
                    getattr(intent, "strategy", "?"),
                    _gate_sec_type,
                    getattr(intent, "side", "?"),
                    getattr(intent, "quantity", "?"),
                )
                continue

            if _intent_is_exit:
                logger.info(
                    "EXIT intent → %s %s %s qty=%s — guard close deferred to fill confirmation",
                    getattr(intent, "symbol", None),
                    getattr(intent, "sec_type", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
                )
            elif is_flip_signal(intent):
                logger.info(
                    "FLIP intent → %s %s %s qty=%s",
                    getattr(intent, "symbol", None),
                    getattr(intent, "sec_type", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
                )
                replace_position(intent)
            else:
                logger.info(
                    "INTENT → %s %s %s qty=%s",
                    getattr(intent, "symbol", None),
                    getattr(intent, "sec_type", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
                )
                mark_position_open(intent)

            # Soak ENTRY-intent audit (PA SOAK_STATUS_HISTORY_WRITER 2026-06-20,
            # companion #3). Best-effort + isolated observer placed just before
            # the broker submit: classify the intent exactly as the branch tree
            # above did, then append one soak_entry_intent.v1 row (admitted=True
            # — it reached submission). Writes only under data/soak/; the writer
            # never raises into the host and this try/except is redundant cover
            # so a failure can never block the submit.
            try:
                from chad.ops.soak.evidence_writers import (
                    classify_intent_type as _soak_classify_intent_type,
                    emit_entry_intent_audit as _soak_emit_entry_intent,
                )
                _soak_emit_entry_intent(
                    intent_type=_soak_classify_intent_type(
                        is_exit=_intent_is_exit,
                        is_flip=is_flip_signal(intent),
                        side=_side_str,
                    ),
                    symbol=getattr(intent, "symbol", None),
                    side=getattr(intent, "side", None),
                    strategy=getattr(intent, "strategy", None),
                    admitted=True,
                )
            except Exception:
                pass

            # --- Submit to IBKR adapter and record paper evidence ---
            try:
                _ensure_thread_event_loop()
                if _EXECUTION_DISABLED:
                    # L1-CLD U7 fail-closed: boot homing failed (BROKER_LOOP_DOWN).
                    # Skip the submit entirely rather than risk a fallback path;
                    # no orders are placed until the execution connection is
                    # restored (operator action).
                    logger.warning(
                        "SUBMIT_SKIPPED_NO_EXECUTION %s %s — execution disabled "
                        "(BROKER_LOOP_DOWN at boot)",
                        getattr(intent, "symbol", ""),
                        getattr(intent, "side", ""),
                    )
                    submitted = []
                else:
                    submitted = _paper_adapter.submit_strategy_trade_intents([intent])
                for order in submitted:
                    if str(order.status or "").strip().lower() in {"error", "failed", "rejected"}:
                        logger.warning(
                            "SUBMIT_FAILED %s %s %s qty=%s status=%s dry_run=%s",
                            order.symbol, order.side, order.sec_type,
                            order.quantity, order.status, order.dry_run,
                        )
                    else:
                        logger.info(
                            "SUBMITTED %s %s %s qty=%s status=%s dry_run=%s",
                            order.symbol, order.side, order.sec_type,
                            order.quantity, order.status, order.dry_run,
                        )
                    try:
                        # Thread expected_price (limit_price fallback) so the
                        # slippage tracker can compute real slippage. Asset_class,
                        # fill_price, and status normalization are all delegated
                        # to normalize_paper_fill_evidence — single chokepoint
                        # that every paper-mode writer (live_loop, position
                        # reconciler, timer-driven executor) shares so
                        # PendingSubmit / error / unknown can never leak into
                        # FILLS_*.ndjson.
                        _expected_px = float(getattr(intent, "expected_price", 0.0) or 0.0)
                        if _expected_px <= 0.0:
                            _lp = getattr(intent, "limit_price", None)
                            try:
                                _expected_px = float(_lp) if _lp is not None else 0.0
                            except (TypeError, ValueError):
                                _expected_px = 0.0

                        # Look up the original routed signal so its meta —
                        # including BAG/COMBO leg structure for alpha_options
                        # vertical spreads — can be threaded into ev.extra.
                        # The paper-fill simulator in paper_exec_evidence_writer
                        # uses these fields to rewrite fill_price as the
                        # net_debit_estimate (per-contract debit) instead of
                        # the underlying ETF price.
                        _bag_extra: dict = {}
                        try:
                            _intent_strategy = str(getattr(intent, "strategy", "") or "")
                            _intent_symbol = str(getattr(intent, "symbol", "") or "")
                            _intent_side = str(getattr(intent, "side", "") or "")
                            _intent_qty = float(getattr(intent, "quantity", 0.0) or 0.0)
                            _orig_sig = None
                            for _key, _sig in (routed_signal_map or {}).items():
                                _ks, _ksy, _ksi, _ = _key
                                if (
                                    _ks == _intent_strategy
                                    and _ksy == _intent_symbol
                                    and _ksi == _intent_side
                                ):
                                    _orig_sig = _sig
                                    break
                            _sm = getattr(_orig_sig, "meta", None) if _orig_sig else None
                            if isinstance(_sm, dict):
                                for _k in (
                                    "sec_type", "spread_id", "spread_type", "expiry",
                                    "long_strike", "short_strike", "long_right", "short_right",
                                    "dte", "max_loss_per_contract", "net_debit_estimate",
                                    "contracts", "required_asset_class", "engine",
                                ):
                                    if _k in _sm and _sm[_k] is not None:
                                        _bag_extra[_k] = _sm[_k]
                        except Exception:
                            _bag_extra = {}

                        # Paper-fill safety gate (see _should_persist_paper_evidence):
                        # IBKR returns PendingSubmit/PreSubmitted synchronously and
                        # may reject async (e.g. Error 201 open-order cap). Skip the
                        # fill record unless the adapter status is confirmed OR the
                        # path is an explicit paper simulator (alpha_options BAG).
                        _persist_ev, _skip_reason = _should_persist_paper_evidence(order, _bag_extra)
                        if not _persist_ev:
                            logger.warning(
                                "SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS "
                                "symbol=%s strategy=%s sec_type=%s side=%s "
                                "qty=%s status=%s reason=%s — IBKR submission "
                                "not yet confirmed; no paper fill written",
                                getattr(order, "symbol", "?"),
                                getattr(intent, "strategy", "?"),
                                getattr(order, "sec_type", "?"),
                                getattr(order, "side", "?"),
                                getattr(order, "quantity", "?"),
                                getattr(order, "status", "?"),
                                _skip_reason,
                            )
                            # OPS-OMEGA-01 Pattern C: revert the cooldown-arming
                            # write that should_emit_signal performed pre-submit,
                            # so an unconfirmed/duplicate result does not consume
                            # the 10-minute cooldown.
                            try:
                                from chad.core.signal_guard import (
                                    revert_emission_for_unconfirmed,
                                )
                                if revert_emission_for_unconfirmed(adapted):
                                    logger.info(
                                        "COOLDOWN_NOT_REARMED_UNCONFIRMED_STATUS "
                                        "symbol=%s strategy=%s side=%s qty=%s status=%s",
                                        getattr(order, "symbol", "?"),
                                        getattr(intent, "strategy", "?"),
                                        getattr(order, "side", "?"),
                                        getattr(order, "quantity", "?"),
                                        getattr(order, "status", "?"),
                                    )
                            except Exception as _cd_err:
                                logger.warning(
                                    "COOLDOWN_REVERT_FAILED (non-fatal): %s", _cd_err
                                )
                            continue

                        ev = PaperExecEvidence(
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            fill_price=0.0,  # resolved by normalizer from price_cache
                            expected_price=_expected_px,
                            # PA-EP3: thread the canonical intent identifier so
                            # slippage.v1 / signal_decay records carry a real
                            # join key (idempotency_key, trace_id fallback —
                            # same precedence as routing_gates.py:437).
                            execution_id=(
                                getattr(intent, "idempotency_key", "")
                                or getattr(intent, "trace_id", "")
                                or ""
                            ),
                            strategy=getattr(intent, "strategy", "") or "",
                            source_strategies=[getattr(intent, "strategy", "") or ""],
                            broker="ibkr_paper",
                            status=order.status or "",
                            asset_class=getattr(order, "asset_class", "") or "",
                            is_live=False,
                            fill_time_utc=order.submitted_at.isoformat() if order.submitted_at else "",
                            extra=_bag_extra,
                        )
                        normalize_paper_fill_evidence(ev)
                        if str(getattr(order, 'status', '') or '').strip().lower() == 'error':
                            logger.warning(
                                "SUBMIT_FAILED_SKIP_EVIDENCE symbol=%s strategy=%s "
                                "status=%s — order not placed; skipping fill record",
                                getattr(order, 'symbol', '?'),
                                getattr(intent, 'strategy', '?'),
                                order.status,
                            )
                            continue
                        paths = write_paper_exec_evidence(ev)
                        logger.info(
                            "EVIDENCE_WRITTEN symbol=%s status=%s price=%s ac=%s fills=%s",
                            ev.symbol, ev.status, ev.fill_price, ev.asset_class,
                            paths.get("fills_path", ""),
                        )

                        # GAP-A001: close-confirmation for exit-intents.
                        # When the originating signal is an exit (max_hold or
                        # explicit close) AND the resulting fill is trusted
                        # (status=paper_fill, fill_id present, not pnl_untrusted),
                        # close the guard entry now. Untrusted closes leave the
                        # guard open per ISSUE-29 — no phantom closes.
                        if _intent_is_exit:
                            try:
                                _ev_extra = ev.extra if isinstance(ev.extra, dict) else {}
                                _close_evidence = {
                                    "fill_id": str(paths.get("fill_id", "")) if isinstance(paths, dict) else "",
                                    "status": str(getattr(ev, "status", "") or "").strip().lower(),
                                    "pnl_untrusted": bool(_ev_extra.get("pnl_untrusted")),
                                    "reject": bool(getattr(ev, "reject", False)),
                                    "tags": list(ev.tags) if ev.tags else [],
                                    "extra": dict(_ev_extra),
                                }
                                _closed = mark_position_closed(intent, _close_evidence)
                                if _closed:
                                    logger.info(
                                        "EXIT_GUARD_CLOSED strategy=%s symbol=%s side=%s "
                                        "fill_id=%s — alpha_options BAG SELL close confirmed",
                                        getattr(intent, "strategy", "?"),
                                        getattr(intent, "symbol", "?"),
                                        getattr(intent, "side", "?"),
                                        _close_evidence["fill_id"],
                                    )
                                else:
                                    logger.warning(
                                        "EXIT_GUARD_NOT_CLOSED strategy=%s symbol=%s "
                                        "status=%s pnl_untrusted=%s reject=%s — "
                                        "guard left open; will retry on next cycle",
                                        getattr(intent, "strategy", "?"),
                                        getattr(intent, "symbol", "?"),
                                        _close_evidence["status"],
                                        _close_evidence["pnl_untrusted"],
                                        _close_evidence["reject"],
                                    )
                            except Exception as _close_err:
                                logger.warning(
                                    "EXIT_GUARD_CLOSE_FAILED (non-fatal): %s",
                                    _close_err,
                                )

#                         # Real-time Telegram trade alert — best effort only,
#                         # must never block execution or evidence persistence.
#                         try:
#                             from chad.utils.telegram_notify import send_trade_alert
#                             _alert_qty = float(order.quantity or 0.0)
#                             _alert_price = float(ev.fill_price or 0.0)
#                             _alert_notional = abs(_alert_qty) * _alert_price
#                             send_trade_alert(
#                                 symbol=order.symbol,
#                                 side=str(order.side or ""),
#                                 quantity=_alert_qty,
#                                 price=_alert_price,
#                                 strategy=str(getattr(intent, "strategy", "") or ""),
#                                 notional=_alert_notional,
#                                 is_live=False,
#                             )
#                         except Exception:
#                             pass
                    except StrategyAttributionError as attr_err:
                        logger.warning("Evidence attribution failed (non-fatal): %s", attr_err)
                    except Exception as ev_err:
                        logger.warning("Evidence write failed (non-fatal): %s", ev_err)
            except Exception as submit_err:
                logger.error(
                    "SUBMIT_FAILED %s %s qty=%s: %s",
                    getattr(intent, "symbol", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
                    submit_err,
                )

        except Exception as _intent_exc:
            logger.exception(
                "live_loop_intent_failed symbol=%s strategy=%s err=%s — skipping intent",
                getattr(intent, 'symbol', '?'),
                getattr(intent, 'strategy', '?'),
                _intent_exc,
            )
            continue
    if emitted == 0:
        logger.info("All intents skipped by signal/position guard.")

    # ------------------------------------------------------------------
    # GAP-025: routing diagnostics. Pure observer — writes
    # runtime/strategy_routing_diagnostics.json from disk-derived state
    # (fills, closed trades, allocations, dynamic_caps) PLUS the
    # per-cycle stage counts and block-reason attribution recorded by
    # the tracker that observed each routing/filter step above.
    # Failure-soft; never affects routing or runtime state outside the
    # diagnostic artifact.
    # ------------------------------------------------------------------
    try:
        from chad.ops.strategy_routing_diagnostics import (
            write_diagnostics as _write_routing_diagnostics,
        )
        _write_routing_diagnostics(_routing_diag)
    except Exception as _diag_err:  # noqa: BLE001
        logger.debug(
            "strategy_routing_diagnostics failed (non-fatal): %s",
            _diag_err,
        )


def _init_redis_stop_subscriber(logger: logging.Logger) -> None:
    """Subscribe to Redis stop signal for immediate propagation."""
    global _redis_stop_flag, _redis_stop_reason
    try:
        from chad.core.state_bus import get_subscriber

        def _on_stop(data: dict) -> None:
            global _redis_stop_flag, _redis_stop_reason
            _redis_stop_flag = True
            _redis_stop_reason = str(data.get("reason", "redis_stop_signal"))
            logger.warning(
                "STOP_SIGNAL_RECEIVED via Redis: %s", _redis_stop_reason,
            )

        get_subscriber().on_stop(_on_stop)
        logger.info("Redis stop signal subscriber active")
    except Exception as exc:
        logger.info("Redis stop subscriber not available (non-fatal): %s", exc)


def _init_redis_dynamic_caps_subscriber(logger: logging.Logger) -> None:
    """Subscribe to Redis dynamic_caps for real-time cap updates."""
    global _redis_dynamic_caps
    try:
        from chad.core.state_bus import get_subscriber

        def _on_dynamic_caps(data: dict) -> None:
            global _redis_dynamic_caps
            _redis_dynamic_caps = data
            logger.debug(
                "DYNAMIC_CAPS_RECEIVED via Redis: %d keys",
                len(data) if isinstance(data, dict) else 0,
            )

        get_subscriber().on_dynamic_caps(_on_dynamic_caps)
        logger.info("Redis dynamic_caps subscriber active")
    except Exception as exc:
        logger.info("Redis dynamic_caps subscriber not available (non-fatal): %s", exc)


def _init_redis_live_gate_subscriber(logger: logging.Logger) -> None:
    """Subscribe to Redis live_gate for observability."""
    global _redis_live_gate_ts
    try:
        from chad.core.state_bus import get_subscriber

        def _on_live_gate(data: dict) -> None:
            global _redis_live_gate_ts
            _redis_live_gate_ts = data.get("ts_utc") or data.get("timestamp", "")
            logger.debug(
                "LIVE_GATE_RECEIVED via Redis: ts=%s", _redis_live_gate_ts,
            )

        get_subscriber().on_live_gate(_on_live_gate)
        logger.info("Redis live_gate subscriber active")
    except Exception as exc:
        logger.info("Redis live_gate subscriber not available (non-fatal): %s", exc)


def run_loop() -> None:
    global _redis_stop_flag, _redis_stop_reason
    logger = logging.getLogger("chad.live_loop")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.info("Starting CHAD live loop")

    import signal as _signal
    _signal.signal(_signal.SIGTERM, _handle_sigterm)
    _signal.signal(_signal.SIGINT, _handle_sigterm)

    # Subscribe to Redis stop signal for <100ms propagation
    _init_redis_stop_subscriber(logger)
    # Subscribe to Redis dynamic_caps for real-time cap updates
    _init_redis_dynamic_caps_subscriber(logger)
    # Subscribe to Redis live_gate for observability
    _init_redis_live_gate_subscriber(logger)

    # L1-CLD U7: home the dedicated execution connection on the broker owner
    # loop BEFORE the first cycle. Fail-closed — a failure here disables the
    # execution path (BROKER_LOOP_DOWN) rather than letting a later lazy connect
    # fall through to a MainThread path.
    _home_execution_connection(logger)

    while True:
        if _SHUTDOWN_REQUESTED:
            logger.warning("live_loop_shutdown: clean exit on SIGTERM")
            break

        # Check Redis stop flag before each cycle
        if _redis_stop_flag:
            logger.warning(
                "STOP_SIGNAL_ACTIVE via Redis, skipping cycle: %s",
                _redis_stop_reason,
            )
            time.sleep(LOOP_INTERVAL_SECONDS)
            continue

        try:
            run_once(logger)
        except Exception as exc:
            logger.exception("Loop error: %s", exc)
            try:
                from chad.utils.telegram_notify import notify
                notify(
                    f"🚨 LIVE LOOP EXCEPTION — cycle crashed\n{type(exc).__name__}: {exc}",
                    severity="critical",
                    dedupe_key="live_loop_exception",
                )
            except Exception:
                pass

        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_loop()

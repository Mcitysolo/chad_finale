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
from typing import Any, Dict, List, Optional, Tuple

from chad.core.live_execution_router import (
    build_live_signals,
    build_all_live_signals,
    is_always_active_routing,
)
from chad.core.ibkr_execution_runner import _build_plan_and_intents
from chad.core.suppression import SuppressionReason
from chad.core.broker_position_sync import BrokerPositionSync
from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig, resolve_asset_class
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
from ib_insync import IB, util

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
import ib_insync.ib as _ib_module
async def _noop_executions(self, *a, **kw): return []
_ib_module.IB.reqExecutionsAsync = _noop_executions
# ISSUE-29 / test-import safety: tests that import this module must NOT
# attempt to claim clientId=99 — the running live_loop process holds it
# and the connect would TimeoutError (Error 326). CHAD_SKIP_IB_CONNECT=1
# in the pytest environment skips the connect; the live runner leaves it
# unset and connects normally.
if os.environ.get("CHAD_SKIP_IB_CONNECT", "").strip().lower() not in ("1", "true", "yes"):
    ib.connect("127.0.0.1", 4002, clientId=99, timeout=120)


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
_paper_adapter = IbkrAdapter(
    config=IbkrConfig(
        dry_run=(_get_exec_mode() != _ExecMode.IBKR_PAPER),
    ),
    ib_factory=lambda: ib,
)
LOOP_INTERVAL_SECONDS = 60
_ROUTE_DECISION_PATH = Path("/home/ubuntu/chad_finale/runtime/last_route_decision.json")

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


def _build_stop_bus_snapshot(logger: logging.Logger) -> Dict[str, Any]:
    """Best-effort snapshot of halt-trigger inputs.

    Missing inputs are simply omitted — the aggregator skips any trigger
    whose keys aren't present, so partial snapshots are fine.
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

    # Broker latency: ibkr_status.json publishes last latency measurement.
    try:
        status_path = Path("/home/ubuntu/chad_finale/runtime/ibkr_status.json")
        if status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict):
                lat = status.get("latency_ms") or status.get("avg_latency_ms")
                if lat is not None:
                    snap["avg_latency_ms"] = float(lat)
    except Exception:
        pass

    # Reject-rate and data-staleness snapshots require windowed counters
    # that the current codebase does not yet publish; they are left
    # unpopulated and the aggregator skips them cleanly.
    return snap


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

    broker_symbols = {sym for sym, pos in broker_positions.items() if abs(pos.quantity) > 1e-9}
    corrections_closed = 0
    corrections_opened = 0

    # Close guard entries that the broker no longer holds
    for key, entry in list(guard_state.items()):
        if not entry.get("open"):
            continue
        symbol = entry.get("symbol", "")
        if symbol and symbol not in broker_symbols:
            entry["open"] = False
            entry["closed_by"] = "broker_truth_rebuild"
            corrections_closed += 1

    # Add or update broker_sync positions to reflect broker truth.
    # broker_sync key always tracks IBKR position exactly. Other strategy
    # keys (alpha|SPY, delta|SPY) track CHAD-initiated trades separately.
    for sym, bp in broker_positions.items():
        if abs(bp.quantity) < 1e-9:
            continue
        side = "BUY" if bp.quantity > 0 else "SELL"
        fallback_key = f"broker_sync|{sym}"
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
            "symbol": sym,
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
                "guard_open_count": sum(1 for e in guard_state.values() if e.get("open")),
            },
        )
    else:
        logger.info(
            "BROKER_TRUTH_REBUILD",
            extra={
                "corrections_closed": 0,
                "corrections_opened": 0,
                "broker_position_count": len(broker_symbols),
                "guard_open_count": sum(1 for e in guard_state.values() if e.get("open")),
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
    # Phase-8 Session 4 (R2): honour an operator- or trigger-set STOP bus.
    # A truthy runtime/stop_bus.json halts the cycle before any rebuild or
    # routing so that reconciliation side-effects are not applied while
    # trading is stopped. clear_stop_bus() (from chad.risk.stop_bus_state
    # or the CLI script) is required to resume.
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

    # Phase-8 Session 4 (R2): evaluate halt triggers. The snapshot is
    # best-effort from existing runtime state — any trigger whose inputs
    # are missing is simply skipped (the aggregator ignores missing keys).
    # If any trigger fires this persists runtime/stop_bus.json and halts
    # the current cycle.
    try:
        from chad.risk.stop_bus_state import evaluate_and_persist
        snapshot = _build_stop_bus_snapshot(logger)
        sb_result = evaluate_and_persist(
            snapshot=snapshot,
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
            return
    except Exception as _sbt_err:  # noqa: BLE001
        logger.warning("stop_bus evaluation failed (non-fatal): %s", _sbt_err)

    strategy_detail: Dict[str, Any] = {
        "available_strategies": {},
        "rejected_strategies": {},
        "selected_strategy": None,
        "selected_strategy_reason": None,
        "affordability_rejections": [],
        "guard_rejections": [],
    }

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
        _currently_halted = set(halted)
        _EDGE_DECAY_ALERTED -= (set(_EDGE_DECAY_ALERTED) - _currently_halted)
    except Exception as _decay_err:  # noqa: BLE001
        logger.warning("edge_decay_monitor failed (non-fatal): %s", _decay_err)

    def _signal_strategy_name(sig) -> str:
        v = getattr(sig, "strategy", None)
        v = getattr(v, "value", v)
        return str(v or "").strip().lower()

    try:
        if is_always_active_routing():
            all_result = build_all_live_signals(logger)
            routed_signals = all_result.all_signals
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
        new_state = write_regime_state(regime_result, source="live_loop.run_once", ttl_seconds=120)
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
        mark_position_open,
        replace_position,
    )

    emitted = 0

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
                    "SKIP suppression=%s → %s %s %s qty=%s",
                    SuppressionReason.SAME_SIDE_POSITION_OPEN.value,
                    getattr(intent, "symbol", None),
                    getattr(intent, "sec_type", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
                    extra={"suppression_reason": SuppressionReason.SAME_SIDE_POSITION_OPEN.value},
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

            emitted += 1

            if is_flip_signal(intent):
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

            # --- Submit to IBKR adapter and record paper evidence ---
            try:
                _ensure_thread_event_loop()
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

                        ev = PaperExecEvidence(
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            fill_price=0.0,  # resolved by normalizer from price_cache
                            expected_price=_expected_px,
                            strategy=getattr(intent, "strategy", "") or "",
                            source_strategies=[getattr(intent, "strategy", "") or ""],
                            broker="ibkr_paper",
                            status=order.status or "",
                            asset_class=getattr(order, "asset_class", "") or "",
                            is_live=False,
                            fill_time_utc=order.submitted_at.isoformat() if order.submitted_at else "",
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

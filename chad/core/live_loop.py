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

import json
import logging
import os
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
from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=99)

position_sync = BrokerPositionSync(ib)
LOOP_INTERVAL_SECONDS = 60
_ROUTE_DECISION_PATH = Path("/home/ubuntu/chad_finale/runtime/last_route_decision.json")


def _write_route_decision(detail: Dict[str, Any]) -> None:
    """Write strategy_detail bridge file for orchestrator DecisionTrace pickup."""
    try:
        _ROUTE_DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ROUTE_DECISION_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(detail, indent=2, default=str), encoding="utf-8")
        tmp.replace(_ROUTE_DECISION_PATH)
    except Exception:
        pass


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
    from chad.core.position_guard import _load_state, _save_state

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

    # Add broker positions the guard doesn't know about
    for sym, bp in broker_positions.items():
        if abs(bp.quantity) < 1e-9:
            continue
        # Guard keys are strategy|symbol — broker doesn't know strategy,
        # so we check if ANY guard entry covers this symbol as open.
        symbol_covered = any(
            e.get("symbol") == sym and e.get("open")
            for e in guard_state.values()
        )
        if not symbol_covered:
            side = "BUY" if bp.quantity > 0 else "SELL"
            fallback_key = f"broker_sync|{sym}"
            guard_state[fallback_key] = {
                "open": True,
                "updated_at_utc": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "strategy": "broker_sync",
                "symbol": sym,
                "side": side,
                "quantity": abs(bp.quantity),
                "source": "broker_truth_rebuild",
            }
            corrections_opened += 1

    if corrections_closed or corrections_opened:
        _save_state(guard_state)
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
    except Exception:
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
    # P1-3: Rebuild position guard from broker truth before any signal evaluation
    try:
        _rebuild_guard_from_broker(logger)
    except Exception as exc:
        logger.warning("Broker truth rebuild failed (non-fatal): %s", exc)

    strategy_detail: Dict[str, Any] = {
        "available_strategies": {},
        "rejected_strategies": {},
        "selected_strategy": None,
        "selected_strategy_reason": None,
        "affordability_rejections": [],
        "guard_rejections": [],
    }

    if is_always_active_routing():
        all_result = build_all_live_signals(logger)
        routed_signals = all_result.all_signals
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

    _ctx, _plan, intents = _build_plan_and_intents(logger)

    if not intents:
        logger.info("No executable intents.")
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
        _attach_strategy_to_intent(intent, routed_signal_map)

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

    if emitted == 0:
        logger.info("All intents skipped by signal/position guard.")


def run_loop() -> None:
    logger = logging.getLogger("chad.live_loop")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.info("Starting CHAD live loop")

    while True:
        try:
            run_once(logger)
        except Exception as exc:
            logger.exception("Loop error: %s", exc)

        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_loop()

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
from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig
from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    StrategyAttributionError,
    write_paper_exec_evidence,
)
from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=99)

position_sync = BrokerPositionSync(ib)
_paper_adapter = IbkrAdapter(config=IbkrConfig(
    dry_run=(os.environ.get("CHAD_EXECUTION_MODE", "dry_run").strip().lower()
             not in ("paper",)),
))
LOOP_INTERVAL_SECONDS = 60
_ROUTE_DECISION_PATH = Path("/home/ubuntu/chad_finale/runtime/last_route_decision.json")

# Redis stop signal flag — set by state bus subscriber
_redis_stop_flag = False
_redis_stop_reason = ""

# Redis dynamic_caps — advisory cache for reducing JSON poll lag
_redis_dynamic_caps: Optional[Dict] = None

# Redis live_gate — last received timestamp for latency monitoring
_redis_live_gate_ts: Optional[str] = None


from chad.core.kraken_execution import execute_kraken_intents as _execute_kraken_intents


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
    """Detect paper / dry_run posture from env (mirrors live_gate._load_execution_config)."""
    raw = (
        os.environ.get("CHAD_EXECUTION_MODE")
        or os.environ.get("CHAD_EXEC_MODE")
        or os.environ.get("CHAD_MODE")
        or "dry_run"
    ).strip().lower()
    return raw in ("dry_run", "paper")


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
    from chad.core.position_guard import _load_state, _save_state

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
        guard_state[key] = {
            "open": True,
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

    _save_state(guard_state)
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
    # P1-3: Rebuild position guard before any signal evaluation. In paper /
    # dry_run mode reconcile against the local trade_closer ledger so the
    # simulated positions survive across cycles; only query IB Gateway when
    # actually live.
    try:
        if _is_paper_mode():
            _rebuild_guard_from_paper_ledger(logger)
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

    _ctx, _plan, intents, kraken_intents = _build_plan_and_intents(logger)

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
            submitted = _paper_adapter.submit_strategy_trade_intents([intent])
            for order in submitted:
                logger.info(
                    "SUBMITTED %s %s %s qty=%s status=%s dry_run=%s",
                    order.symbol, order.side, order.sec_type,
                    order.quantity, order.status, order.dry_run,
                )
                try:
                    # Inject real fill price from price cache
                    _fill_price = 0.0
                    try:
                        _pc_path = Path("/home/ubuntu/chad_finale/runtime/price_cache.json")
                        _pc = json.loads(_pc_path.read_text(encoding="utf-8"))
                        _prices = _pc.get("prices", {})
                        _fill_price = float(_prices.get(order.symbol, 0.0))
                        if _fill_price == 0.0:
                            logger.warning("PRICE_CACHE_MISS symbol=%s — fill_price defaulting to 0.0", order.symbol)
                    except Exception as pc_err:
                        logger.warning("PRICE_CACHE_READ_FAILED: %s — fill_price defaulting to 0.0", pc_err)

                    ev = PaperExecEvidence(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=_fill_price,
                        strategy=getattr(intent, "strategy", "") or "",
                        source_strategies=[getattr(intent, "strategy", "") or ""],
                        broker="ibkr_paper",
                        status=order.status,
                        asset_class=order.asset_class,
                        is_live=False,
                        fill_time_utc=order.submitted_at.isoformat() if order.submitted_at else "",
                    )
                    paths = write_paper_exec_evidence(ev)
                    logger.info("EVIDENCE_WRITTEN fills=%s", paths.get("fills_path", ""))
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

    # Subscribe to Redis stop signal for <100ms propagation
    _init_redis_stop_subscriber(logger)
    # Subscribe to Redis dynamic_caps for real-time cap updates
    _init_redis_dynamic_caps_subscriber(logger)
    # Subscribe to Redis live_gate for observability
    _init_redis_live_gate_subscriber(logger)

    while True:
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

        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_loop()

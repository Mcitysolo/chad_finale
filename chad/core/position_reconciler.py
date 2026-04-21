#!/usr/bin/env python3
"""
chad/core/position_reconciler.py

Close open positions when the net strategy signal direction flips.

Runs every live cycle BEFORE the execution pipeline's netting step so
that a BUY signal on a symbol with an open SELL position produces an
explicit CLOSE intent that cannot be netted away by other strategies.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

LOG = logging.getLogger("chad.core.position_reconciler")

# Pre-existing paper account positions that were not opened by CHAD.
# These are skipped entirely so they never trigger reconciler closes
# nor show up as reconciliation mismatches.
KNOWN_NON_CHAD_SYMBOLS: frozenset = frozenset({"AAPL", "MSFT"})

_PRICE_CACHE_PATH = Path("/home/ubuntu/chad_finale/runtime/price_cache.json")


def _iter_strategy_signal(routed_signals: Iterable) -> Iterable:
    """
    Yield (strategy, signal) pairs regardless of whether routed_signals
    is a flat list of TradeSignal objects or a list of (strategy, signal)
    tuples. TradeSignal objects carry a .strategy attribute of their own.
    """
    for item in routed_signals or []:
        if isinstance(item, tuple) and len(item) == 2:
            yield item[0], item[1]
        else:
            yield getattr(item, "strategy", None), item


def _signal_size(sig: Any) -> float:
    for attr in ("size", "qty", "quantity", "net_size"):
        v = getattr(sig, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 1.0


def reconcile_positions_with_signals(
    open_positions: Dict[str, dict],
    routed_signals: List,
    prices: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Compare each open position against the net signal direction for its
    symbol. When they conflict, emit a close intent.

    Returns a list of close-intent dicts (not ExecutionIntents). These
    are consumed by apply_close_intents() and bypass the netting layer.
    """
    close_intents: List[dict] = []

    if not open_positions:
        return close_intents

    try:
        pairs = list(_iter_strategy_signal(routed_signals))
    except Exception as exc:  # noqa: BLE001
        LOG.warning("reconciler: failed to iterate routed_signals (%s)", exc)
        return close_intents

    for position_key, position in open_positions.items():
        try:
            if not isinstance(position, dict) or not position.get("open"):
                continue

            symbol = position.get("symbol")
            open_side = str(position.get("side", "")).upper()
            if not symbol or open_side not in ("BUY", "SELL"):
                continue

            if symbol in KNOWN_NON_CHAD_SYMBOLS:
                continue

            # ISSUE-29 fix: skip partial_attribution_residual entries.
            # These are broker_sync lots being gradually claimed by strategies
            # via position_guard._reduce_or_close_broker_sync (ISSUE-56 v2).
            # They are NOT orphans — they will be consumed over subsequent
            # cycles as attribution completes. Flap-closing them destroys
            # the residual that represents unclaimed broker_truth.
            position_meta = position.get("meta") or position.get("extra") or {}
            if (
                position_meta.get("partial_attribution_residual")
                or position.get("source") == "partial_attribution_residual"
                or position.get("closed_by") == "partial_attribution_residual"
            ):
                LOG.debug(
                    "RECONCILER_SKIP partial_attribution_residual entry %s — "
                    "not an orphan, awaiting strategy attribution",
                    position_key,
                )
                continue

            # ISSUE-29 extension: when a strategy entry exists alongside an
            # OPEN broker_sync residual for the same symbol, attribution is
            # in progress. Flap-closing the strategy side leaves the residual
            # alone and the publisher sees chad < broker — looks like drift.
            strategy_name = str(position.get("strategy", "") or "")
            if strategy_name and strategy_name != "broker_sync":
                paired_bs = open_positions.get(f"broker_sync|{symbol}")
                if (
                    isinstance(paired_bs, dict)
                    and paired_bs.get("open")
                    and paired_bs.get("closed_by") == "partial_attribution_residual"
                ):
                    LOG.debug(
                        "RECONCILER_SKIP strategy entry %s — paired broker_sync "
                        "is partial_attribution_residual, attribution in progress",
                        position_key,
                    )
                    continue

            buy_size = 0.0
            sell_size = 0.0
            for _strat, sig in pairs:
                if getattr(sig, "symbol", None) != symbol:
                    continue
                side_str = str(getattr(sig, "side", "")).upper()
                if "BUY" in side_str:
                    buy_size += _signal_size(sig)
                elif "SELL" in side_str:
                    sell_size += _signal_size(sig)

            if buy_size > sell_size:
                net_direction = "BUY"
            elif sell_size > buy_size:
                net_direction = "SELL"
            else:
                net_direction = None

            if net_direction and net_direction != open_side:
                close_side = "BUY" if open_side == "SELL" else "SELL"
                strategy = (
                    position_key.split("|")[0] if "|" in position_key else "unknown"
                )
                close_intents.append({
                    "symbol": symbol,
                    "action": "CLOSE",
                    "open_side": open_side,
                    "close_side": close_side,
                    "quantity": float(position.get("quantity", 0.0) or 0.0),
                    "reason": f"reconciler_flip_{open_side}_to_{net_direction}",
                    "position_key": position_key,
                    "strategy": strategy,
                })
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "reconciler: error evaluating position %s: %s", position_key, exc
            )
            continue

    return close_intents


def _load_price(symbol: str) -> float:
    try:
        data = json.loads(_PRICE_CACHE_PATH.read_text(encoding="utf-8"))
        return float((data.get("prices") or {}).get(symbol, 0.0) or 0.0)
    except Exception:
        return 0.0


def _close_intent_to_ibkr(close: dict):
    """
    Wrap a close-intent dict in a minimal object that matches the
    ExecutionIntent interface expected by IbkrAdapter.submit_strategy_trade_intents.
    """
    from chad.types import AssetClass

    class _ReconcilerIntent:
        __slots__ = (
            "symbol", "side", "quantity", "sec_type", "asset_class",
            "strategy", "order_type", "confidence",
        )

        def __init__(self, c: dict):
            self.symbol = c["symbol"]
            self.side = c["close_side"]
            self.quantity = float(c.get("quantity", 0.0) or 0.0)
            self.sec_type = "STK"
            self.asset_class = AssetClass.EQUITY
            self.strategy = c.get("strategy", "reconciler")
            self.order_type = "MKT"
            self.confidence = 1.0

    return _ReconcilerIntent(close)


def apply_close_intents(close_intents: List[dict], paper_adapter: Any) -> None:
    """
    Submit reconciler close intents directly through the IBKR paper
    adapter. Writes paper fill evidence just like the main live loop.
    Never raises — each intent is isolated so one failure cannot stop
    the rest of the cycle.
    """
    if not close_intents:
        return

    from chad.execution.paper_exec_evidence_writer import (
        PaperExecEvidence,
        StrategyAttributionError,
        write_paper_exec_evidence,
    )
    from chad.types import AssetClass

    for close in close_intents:
        symbol = close.get("symbol")
        close_side = close.get("close_side")
        reason = close.get("reason", "reconciler_flip")
        qty = float(close.get("quantity", 0.0) or 0.0)

        if qty <= 0:
            LOG.warning(
                "RECONCILER_CLOSE_SKIP symbol=%s side=%s reason=%s qty_non_positive=%s",
                symbol, close_side, reason, qty,
            )
            continue

        LOG.info(
            "RECONCILER_CLOSE symbol=%s side=%s reason=%s qty=%s",
            symbol, close_side, reason, qty,
        )

        try:
            intent_obj = _close_intent_to_ibkr(close)
            submitted = paper_adapter.submit_strategy_trade_intents([intent_obj])
            for order in submitted:
                fill_price = _load_price(order.symbol)
                try:
                    ev = PaperExecEvidence(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        strategy=close.get("strategy", "reconciler") or "reconciler",
                        source_strategies=[close.get("strategy", "reconciler") or "reconciler"],
                        broker="ibkr_paper",
                        status=order.status,
                        asset_class=getattr(order, "asset_class", AssetClass.EQUITY),
                        is_live=False,
                        fill_time_utc=(
                            order.submitted_at.isoformat()
                            if getattr(order, "submitted_at", None)
                            else datetime.now(timezone.utc).isoformat()
                        ),
                    )
                    write_paper_exec_evidence(ev)
                except StrategyAttributionError as attr_err:
                    LOG.warning("reconciler evidence attribution failed: %s", attr_err)
                except Exception as ev_err:  # noqa: BLE001
                    LOG.warning("reconciler evidence write failed: %s", ev_err)

            try:
                from chad.core.position_guard import _load_state, _save_state  # type: ignore
                state = _load_state()
                pk = close.get("position_key")
                if pk and pk in state:
                    state[pk]["open"] = False
                    state[pk]["last_state"] = "CLOSED"
                    state[pk]["closed_by"] = "position_reconciler"
                    state[pk]["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                    _save_state(state)
            except Exception as gs_err:  # noqa: BLE001
                LOG.warning("reconciler guard update failed: %s", gs_err)

        except Exception as exc:  # noqa: BLE001
            LOG.error(
                "RECONCILER_CLOSE_FAILED symbol=%s side=%s reason=%s err=%s",
                symbol, close_side, reason, exc,
            )


def load_open_positions() -> Dict[str, dict]:
    """Helper — load position_guard.json directly (no side effects)."""
    try:
        from chad.core.position_guard import STATE_PATH
        if not STATE_PATH.is_file():
            return {}
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {k: v for k, v in raw.items() if isinstance(v, dict) and v.get("open")}
    except Exception:
        return {}

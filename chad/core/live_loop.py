#!/usr/bin/env python3
"""
chad/core/live_loop.py

Autonomous live loop for CHAD.

Current behavior
----------------
- builds routed signals
- builds execution plan/intents
- prevents duplicate repeats
- prevents reopening same-side positions
- allows flip detection (SELL -> BUY, BUY -> SELL)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

from chad.core.live_execution_router import build_live_signals
from chad.core.ibkr_execution_runner import _build_plan_and_intents
from chad.core.broker_position_sync import BrokerPositionSync
from ib_insync import IB

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=99)

position_sync = BrokerPositionSync(ib)
LOOP_INTERVAL_SECONDS = 60


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


def run_once(logger: logging.Logger) -> None:
    routed_signals = build_live_signals(logger)
    routed_signal_map = _build_router_signal_map(list(routed_signals or []))

    if not routed_signals:
        logger.info("No routed signals available. Sleeping.")
        return

    _ctx, _plan, intents = _build_plan_and_intents(logger)

    if not intents:
        logger.info("No executable intents.")
        return

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
                "SKIP open-position intent → %s %s %s qty=%s",
                getattr(intent, "symbol", None),
                getattr(intent, "sec_type", None),
                getattr(intent, "side", None),
                getattr(intent, "quantity", None),
            )
            continue

        if not is_flip_signal(intent):
            if not should_emit_signal(adapted):
                logger.info(
                    "SKIP duplicate intent → %s %s %s qty=%s",
                    getattr(intent, "symbol", None),
                    getattr(intent, "sec_type", None),
                    getattr(intent, "side", None),
                    getattr(intent, "quantity", None),
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

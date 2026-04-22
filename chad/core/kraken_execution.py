"""
chad/core/kraken_execution.py

Kraken execution lane helpers used by live_loop.

Lives in its own module so the unit tests can exercise the gating /
mode-resolution logic without importing chad.core.live_loop (which has
import-time side effects against IB Gateway).

Public API:
    resolve_kraken_mode()       -> str   (one of: 'live', 'paper_kraken', 'off')
    is_kraken_gate_enabled()    -> bool
    get_kraken_executor()       -> KrakenExecutor (lazy singleton)
    log_kraken_fill(payload)    -> None
    execute_kraken_intents(logger, kraken_intents) -> None

Gating semantics:
    LiveGate.kraken_enabled must be True AND CHAD_KRAKEN_MODE must resolve
    to 'live' or 'paper_kraken' for any execution to occur.

      - 'paper_kraken' -> KrakenExecutor invoked with live=False
        (validate_only=True at the router; real prices/spec, no real money)
      - 'live'         -> KrakenExecutor invoked with live=True (real orders)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List


_KRAKEN_EXECUTOR = None  # lazy singleton


def resolve_kraken_mode() -> str:
    """
    Read CHAD_KRAKEN_MODE from env. Falls back to 'off' (no Kraken execution).
    Recognized values: 'live', 'paper_kraken'. Anything else is 'off'.
    If CHAD_KRAKEN_MODE is unset, falls back to CHAD_EXECUTION_MODE for the
    'live' value only (so a fully-live system with kraken_enabled goes live).
    """
    raw = (os.environ.get("CHAD_KRAKEN_MODE") or "").strip().lower()
    if raw in ("live", "paper_kraken"):
        return raw
    fallback = (os.environ.get("CHAD_EXECUTION_MODE") or "").strip().lower()
    if fallback == "live":
        return "live"
    return "off"


def is_kraken_gate_enabled() -> bool:
    """
    Check LiveGate.kraken_enabled. Falls back to KRAKEN_ENABLED env var if the
    live_gate module cannot be evaluated for any reason.

    2026-04-22 fix (Audit-O): LiveGateDecision exposes kraken_enabled on
    decision.context.execution, not on the decision object itself — the
    old getattr(decision, "kraken_enabled", None) always returned None and
    fell through to the env-var path, which defaulted False when unset.
    We now read the nested attribute and only fall back to the env when
    the live_gate module is unavailable. Env fallback also defaults True
    to match _load_execution_config's own default.
    """
    try:
        from chad.core.live_gate import evaluate_live_gate
        decision = evaluate_live_gate()
        ctx = getattr(decision, "context", None)
        exec_cfg = getattr(ctx, "execution", None) if ctx is not None else None
        ke = getattr(exec_cfg, "kraken_enabled", None)
        if ke is None:
            # Defensive fallbacks for older decision shapes.
            ke = getattr(decision, "kraken_enabled", None)
            if ke is None and isinstance(decision, dict):
                ke = decision.get("kraken_enabled")
        if ke is not None:
            return bool(ke)
    except Exception:
        pass
    raw = (os.environ.get("KRAKEN_ENABLED") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return True


def get_kraken_executor():
    """Lazy-instantiate the singleton KrakenExecutor."""
    global _KRAKEN_EXECUTOR
    if _KRAKEN_EXECUTOR is not None:
        return _KRAKEN_EXECUTOR
    from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig
    from chad.execution.kraken_trade_router import KrakenTradeRouter
    from chad.execution.kraken_executor import KrakenExecutor

    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    router = KrakenTradeRouter(client)
    _KRAKEN_EXECUTOR = KrakenExecutor(router=router)
    return _KRAKEN_EXECUTOR


def log_kraken_fill(payload: Dict[str, Any]) -> None:
    """Append a Kraken fill record to data/fills/kraken_fills_YYYYMMDD.ndjson."""
    try:
        ymd = time.strftime("%Y%m%d", time.gmtime())
        out_dir = Path("/home/ubuntu/chad_finale/data/fills")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"kraken_fills_{ymd}.ndjson"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def execute_kraken_intents(logger: logging.Logger, kraken_intents: List[object]) -> None:
    """
    Execute Kraken (CRYPTO) intents through KrakenExecutor.

    No-op if there are no intents, the LiveGate denies kraken, or the mode
    resolves to 'off'. All results (success/error) are logged to
    data/fills/kraken_fills_YYYYMMDD.ndjson.
    """
    if not kraken_intents:
        return

    if not is_kraken_gate_enabled():
        logger.info(
            "KRAKEN_GATE_DENIED kraken_enabled=False intents=%d",
            len(kraken_intents),
        )
        return

    mode = resolve_kraken_mode()
    if mode == "off":
        logger.info(
            "KRAKEN_MODE_OFF intents=%d (set CHAD_KRAKEN_MODE=paper_kraken or live)",
            len(kraken_intents),
        )
        return

    live = (mode == "live")
    try:
        executor = get_kraken_executor()
    except Exception as exc:
        logger.warning("KRAKEN_EXECUTOR_INIT_FAILED: %s", exc)
        return

    logger.info(
        "KRAKEN_EXECUTE mode=%s live=%s intents=%d",
        mode, live, len(kraken_intents),
    )

    for intent in kraken_intents:
        try:
            risk_result, resp = executor.execute_with_risk(intent=intent, live=live)
            txids = list(getattr(resp, "txids", []) or []) if resp is not None else []
            payload = {
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": mode,
                "live": live,
                "strategy": getattr(intent, "strategy", None),
                "pair": getattr(intent, "pair", None),
                "side": getattr(intent, "side", None),
                "ordertype": getattr(intent, "ordertype", None),
                "volume": float(getattr(intent, "volume", 0.0) or 0.0),
                "notional_estimate": float(getattr(intent, "notional_estimate", 0.0) or 0.0),
                "risk_allowed": bool(getattr(risk_result, "allowed", False)),
                "risk_reason": getattr(risk_result, "reason", ""),
                "risk_adjusted_notional": float(getattr(risk_result, "adjusted_notional", 0.0) or 0.0),
                "txids": txids,
                "validate_only": (not live),
            }
            log_kraken_fill(payload)
            logger.info(
                "KRAKEN_RESULT pair=%s side=%s vol=%s allowed=%s txids=%s",
                payload["pair"], payload["side"], payload["volume"],
                payload["risk_allowed"], txids,
            )
        except Exception as exc:
            logger.warning(
                "KRAKEN_INTENT_FAILED pair=%s side=%s vol=%s: %s",
                getattr(intent, "pair", None),
                getattr(intent, "side", None),
                getattr(intent, "volume", None),
                exc,
            )
            log_kraken_fill({
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode": mode,
                "live": live,
                "pair": getattr(intent, "pair", None),
                "side": getattr(intent, "side", None),
                "volume": float(getattr(intent, "volume", 0.0) or 0.0),
                "error": f"{type(exc).__name__}: {exc}",
            })

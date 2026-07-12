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

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_KRAKEN_EXECUTOR = None  # lazy singleton
_PAPER_EVIDENCE_STORE = None  # lazy IdempotencyStore for paper-kraken dedup
_TRUSTED_FILL_ENGINE = None  # lazy CRYPTO-TRUST U1 engine


def _kraken_trusted_fills_enabled() -> bool:
    """CRYPTO-TRUST U1 kill-switch. Default ON (mirrors the L1-CLD
    CHAD_EXECUTION_OWN_CONNECTION idiom): trusted unless explicitly disabled."""
    return (
        os.environ.get("CHAD_KRAKEN_TRUSTED_FILLS", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )


def _kraken_trusted_dedup(key: str) -> bool:
    """At-most-once gate for trusted fills, sharing the paper idempotency store
    (store #4, runtime/exec_state_paper.sqlite3). Returns True if newly inserted."""
    store = _get_paper_evidence_store()
    mark = store.mark_once(
        trade_id=key, payload_hash=key, meta={"source": "kraken_trusted_fill"}
    )
    return bool(mark.inserted)


def _get_trusted_fill_engine():
    """Lazy-instantiate the trusted-fill engine (dedup wired to store #4)."""
    global _TRUSTED_FILL_ENGINE
    if _TRUSTED_FILL_ENGINE is not None:
        return _TRUSTED_FILL_ENGINE
    from chad.core.kraken_trusted_fill_engine import TrustedFillEngine
    _TRUSTED_FILL_ENGINE = TrustedFillEngine(dedup=_kraken_trusted_dedup)
    return _TRUSTED_FILL_ENGINE


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
    from chad.execution.execution_config import is_live_mode
    if is_live_mode():
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
    # CRYPTO-2 (U2): wire the Kraken margin/BP shadow gate at the executor
    # chokepoint. Fail-open: any construction problem => margin_gate=None =>
    # byte-identical legacy behavior (a broken margin config never stops trading).
    margin_gate = None
    try:
        from chad.execution.kraken_margin_gate import build_default_kraken_shadow_gate
        margin_gate = build_default_kraken_shadow_gate()
    except Exception as exc:  # noqa: BLE001
        margin_gate = None
    _KRAKEN_EXECUTOR = KrakenExecutor(router=router, margin_gate=margin_gate)
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


# ---------------------------------------------------------------------------
# Paper-Kraken evidence helpers
# ---------------------------------------------------------------------------
#
# Why this exists:
#   In paper_kraken mode, KrakenExecutor calls Kraken with validate_only=True.
#   Kraken's validate-only response carries no txids, so the live txid-gated
#   trade-history / FILLS writers in kraken_executor.py never fire. SCR /
#   daily reports / strategy_routing_diagnostics read trade_history_*.ndjson
#   and FILLS_*.ndjson — without records there, alpha_crypto stays
#   zero_fill_epoch2=true even though it generated, gated, and validated
#   intents successfully.
#
#   This patch writes standard CHAD paper evidence at the dispatcher layer
#   (kraken_execution.execute_kraken_intents) when a validate-only call
#   succeeded, the risk gate allowed it, and txids were empty. Live
#   semantics are untouched: the existing executor txid path still runs
#   end-to-end for live=True, and we never fabricate live txids.
#
#   Dedup uses the existing IdempotencyStore (SQLite-backed at
#   runtime/exec_state_paper.sqlite3) so paper evidence is written at most
#   once per (strategy, pair, side, volume, price, UTC minute bucket) even
#   across live-loop restarts. Advancing the minute bucket admits a new
#   record, matching the spec.

_PAPER_EVIDENCE_TABLE = "kraken_paper_evidence"


def _get_paper_evidence_store():
    """Lazy-instantiate the IdempotencyStore for paper-kraken dedup."""
    global _PAPER_EVIDENCE_STORE
    if _PAPER_EVIDENCE_STORE is not None:
        return _PAPER_EVIDENCE_STORE
    from chad.execution.idempotency_store import IdempotencyStore, default_paper_db_path
    db_path = default_paper_db_path(Path("/home/ubuntu/chad_finale"))
    _PAPER_EVIDENCE_STORE = IdempotencyStore(db_path, table=_PAPER_EVIDENCE_TABLE)
    return _PAPER_EVIDENCE_STORE


def _kraken_paper_minute_bucket(ts: Optional[float] = None) -> str:
    """UTC minute bucket as YYYYMMDDHHMM. Idempotent within the same minute."""
    return time.strftime("%Y%m%d%H%M", time.gmtime(ts))


def _resolve_paper_evidence_price(intent: object) -> float:
    """Best-effort price for paper evidence: intent.price → notional/volume → expected_price."""
    price = getattr(intent, "price", None)
    try:
        if price is not None:
            p = float(price)
            if p > 0.0:
                return p
    except (TypeError, ValueError):
        pass
    try:
        vol = float(getattr(intent, "volume", 0.0) or 0.0)
        notional = float(getattr(intent, "notional_estimate", 0.0) or 0.0)
        if vol > 0.0 and notional > 0.0:
            return notional / vol
    except (TypeError, ValueError):
        pass
    try:
        exp = float(getattr(intent, "expected_price", 0.0) or 0.0)
        if exp > 0.0:
            return exp
    except (TypeError, ValueError):
        pass
    return 0.0


def _paper_synthetic_txid(intent: object, price_used: float, *, bucket: Optional[str] = None) -> str:
    """Deterministic, paper-only synthetic txid.

    Hash inputs:
        strategy | pair | side | volume | price | UTC minute bucket
    """
    strategy = str(getattr(intent, "strategy", "") or "")
    pair = str(getattr(intent, "pair", "") or "")
    side = str(getattr(intent, "side", "") or "")
    try:
        volume = float(getattr(intent, "volume", 0.0) or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    minute_bucket = bucket if bucket is not None else _kraken_paper_minute_bucket()
    raw = "|".join([
        strategy,
        pair,
        side,
        f"{volume:.10f}",
        f"{float(price_used or 0.0):.10f}",
        minute_bucket,
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"PAPER-KRAKEN-{digest}"


def _kraken_pair_to_canonical(pair: str) -> str:
    """SOLUSD -> SOL-USD, XBTUSD -> BTC-USD, etc. Falls back to the input on miss."""
    try:
        from chad.portfolio.kraken_trade_result_logger import _canonical_symbol_for_pair
        return _canonical_symbol_for_pair(pair)
    except Exception:
        return (pair or "").strip().upper()


def _write_paper_kraken_evidence(
    logger: logging.Logger,
    intent: object,
    risk_result: object,
    resp: object,
) -> Dict[str, Any]:
    """Write trade_history + FILLS paper evidence for one validated intent.

    Idempotent per (strategy, pair, side, volume, price, UTC minute bucket).
    Returns a small status dict for logging / tests.
    """
    price_used = _resolve_paper_evidence_price(intent)
    bucket = _kraken_paper_minute_bucket()
    synthetic_txid = _paper_synthetic_txid(intent, price_used, bucket=bucket)

    # --- Dedup gate ---
    payload_hash = hashlib.sha256(
        "|".join([
            str(getattr(intent, "strategy", "") or ""),
            str(getattr(intent, "pair", "") or ""),
            str(getattr(intent, "side", "") or ""),
            f"{float(getattr(intent, 'volume', 0.0) or 0.0):.10f}",
            f"{price_used:.10f}",
            bucket,
        ]).encode("utf-8")
    ).hexdigest()

    try:
        store = _get_paper_evidence_store()
        mark = store.mark_once(
            trade_id=synthetic_txid,
            payload_hash=payload_hash,
            meta={
                "source": "kraken_paper_evidence",
                "strategy": str(getattr(intent, "strategy", "") or ""),
                "pair": str(getattr(intent, "pair", "") or ""),
                "minute_bucket": bucket,
            },
        )
    except Exception as exc:
        logger.warning("KRAKEN_PAPER_EVIDENCE_DEDUP_FAILED: %s", exc)
        return {"written": False, "reason": "dedup_store_error", "txid": synthetic_txid}

    if not mark.inserted:
        logger.info(
            "KRAKEN_PAPER_EVIDENCE_DEDUP txid=%s reason=%s strategy=%s pair=%s",
            synthetic_txid,
            mark.reason,
            getattr(intent, "strategy", None),
            getattr(intent, "pair", None),
        )
        return {"written": False, "reason": mark.reason, "txid": synthetic_txid}

    # --- Trade history (KrakenTradeEvent paper mode) ---
    trade_history_path = ""
    try:
        from chad.portfolio.kraken_trade_result_logger import (
            KrakenTradeEvent,
            log_kraken_trade_event,
        )
        ev = KrakenTradeEvent(
            strategy=str(getattr(intent, "strategy", "alpha_crypto") or "alpha_crypto"),
            pair=str(getattr(intent, "pair", "") or ""),
            side=str(getattr(intent, "side", "") or ""),
            ordertype=str(getattr(intent, "ordertype", "") or ""),
            volume=float(getattr(intent, "volume", 0.0) or 0.0),
            notional_estimate=float(getattr(intent, "notional_estimate", 0.0) or 0.0),
            txid=synthetic_txid,
            raw=dict(getattr(resp, "raw", {}) or {}),
        )
        trade_history_path = log_kraken_trade_event(
            ev,
            paper=True,
            fill_price=price_used,
            expected_price=price_used,
        )
    except Exception as exc:
        logger.warning("KRAKEN_PAPER_TRADE_HISTORY_FAILED: %s", exc)

    # --- Paper exec evidence (FILLS_/FEES_/EXECUTION_METRICS_) ---
    fills_path = ""
    try:
        from chad.execution.paper_exec_evidence_writer import (
            PaperExecEvidence,
            write_paper_exec_evidence,
        )
        canonical_symbol = _kraken_pair_to_canonical(str(getattr(intent, "pair", "") or ""))
        volume = float(getattr(intent, "volume", 0.0) or 0.0)
        notional = float(getattr(intent, "notional_estimate", 0.0) or 0.0)
        if notional <= 0.0 and volume > 0.0 and price_used > 0.0:
            notional = volume * price_used

        ev = PaperExecEvidence(
            symbol=canonical_symbol,
            side=str(getattr(intent, "side", "") or "BUY").upper(),
            quantity=volume,
            fill_price=price_used,
            notional=notional,
            strategy=str(getattr(intent, "strategy", "alpha_crypto") or "alpha_crypto"),
            broker="kraken_paper",
            venue="kraken_paper",
            account_id="KRAKEN_PAPER",
            is_live=False,
            asset_class="crypto",
            order_type=str(getattr(intent, "ordertype", "SIM") or "SIM"),
            status="paper_fill",
            expected_price=price_used,
            execution_id=synthetic_txid,
            tags=[
                "kraken_paper",
                "validate_only",
                "paper_fill",
                "pnl_untrusted",
            ],
            extra={
                "source": "kraken_paper_evidence",
                "synthetic_txid": synthetic_txid,
                "pair": str(getattr(intent, "pair", "") or ""),
                "minute_bucket": bucket,
                "validate_only": True,
                "pnl_untrusted": True,
                "pnl_untrusted_reason": "kraken_paper_validate_only_no_realized_fill",
                "risk_allowed": bool(getattr(risk_result, "allowed", False)),
                "risk_adjusted_notional": float(
                    getattr(risk_result, "adjusted_notional", 0.0) or 0.0
                ),
                "raw_response": dict(getattr(resp, "raw", {}) or {}),
            },
            source="kraken_paper_evidence",
        )
        meta = write_paper_exec_evidence(ev)
        fills_path = str(meta.get("fills_path", "") or "")
    except Exception as exc:
        logger.warning("KRAKEN_PAPER_FILLS_EVIDENCE_FAILED: %s", exc)

    logger.info(
        "KRAKEN_PAPER_EVIDENCE_WRITTEN txid=%s strategy=%s pair=%s side=%s "
        "volume=%s price=%s bucket=%s trade_history=%s fills=%s",
        synthetic_txid,
        getattr(intent, "strategy", None),
        getattr(intent, "pair", None),
        getattr(intent, "side", None),
        getattr(intent, "volume", None),
        price_used,
        bucket,
        trade_history_path or "<skipped>",
        fills_path or "<skipped>",
    )
    return {
        "written": True,
        "reason": "inserted",
        "txid": synthetic_txid,
        "minute_bucket": bucket,
        "trade_history_path": trade_history_path,
        "fills_path": fills_path,
        "price_used": price_used,
    }


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

            # Paper-Kraken evidence write: when a validate-only call succeeded
            # against a risk-allowed intent and Kraken returned no txids (the
            # expected validate_only=True shape), persist standard CHAD paper
            # evidence so SCR / daily reports / strategy_routing_diagnostics
            # see the activity. Live txid path is unchanged; if txids are
            # present we never duplicate.
            if (
                mode == "paper_kraken"
                and not live
                and bool(getattr(risk_result, "allowed", False))
                and resp is not None
                and not txids
            ):
                try:
                    # CRYPTO-TRUST U1: the trusted fill engine is the evidence
                    # product (validate_only above remains the pre-check). It
                    # marks against the live WS tape, applies the real taker fee,
                    # and realizes FIFO round-trip PnL into the SAME pipeline.
                    # Fail-closed to the legacy untrusted writer when the engine
                    # declines (kill-switch off, stale tape, dedup error).
                    used_trusted = False
                    if _kraken_trusted_fills_enabled():
                        try:
                            result = _get_trusted_fill_engine().process_intent(intent)
                            used_trusted = bool(result.get("trusted"))
                            if not used_trusted:
                                logger.info(
                                    "KRAKEN_TRUSTED_FILL_FALLBACK pair=%s side=%s reason=%s",
                                    getattr(intent, "pair", None),
                                    getattr(intent, "side", None),
                                    result.get("reason"),
                                )
                        except Exception as engine_exc:  # noqa: BLE001
                            # Any engine error -> fall back to the legacy writer
                            # so paper evidence is never silently lost.
                            logger.warning(
                                "KRAKEN_TRUSTED_FILL_ENGINE_ERROR pair=%s side=%s: %s",
                                getattr(intent, "pair", None),
                                getattr(intent, "side", None),
                                engine_exc,
                            )
                            used_trusted = False
                    if not used_trusted:
                        _write_paper_kraken_evidence(logger, intent, risk_result, resp)
                except Exception as exc:
                    logger.warning(
                        "KRAKEN_PAPER_EVIDENCE_WRITE_FAILED pair=%s side=%s: %s",
                        getattr(intent, "pair", None),
                        getattr(intent, "side", None),
                        exc,
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

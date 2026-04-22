"""
chad/execution/kraken_executor.py

Phase 12 upgrade: broker execution hardening with durable exactly-once semantics.

What this module now guarantees (Phase 12 core requirements):
- Exactly-once *broker submission* for a given idempotency_key (durable across restarts)
- Safe retries with bounded backoff + jitter (delegated to router/client; we still track attempts)
- Durable txid writeback to exec_state.sqlite3 as broker_order_id
- Deterministic idempotency_key derivation (prefers upstream-provided keys, else hashes a canonical intent payload)

Notes:
- LiveGate / OperatorIntent / SCR gating is enforced upstream in your safety spine.
- This executor enforces *execution integrity* once called.
- validate_only mode is still supported (live=False) with no broker side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig
from chad.execution.intent_schema import DEFAULT_TTL_SECONDS, utc_now_iso
from chad.execution.kraken_trade_router import KrakenTradeRouter, TradeRequest, TradeResponse
from chad.portfolio.kraken_trade_result_logger import KrakenTradeEvent, log_kraken_trade_event
from chad.risk.dynamic_caps import check_risk, load_dynamic_caps
from chad.utils.runtime_defaults import default_dynamic_caps_path

LOGGER = logging.getLogger("chad.execution.kraken_executor")


# ----------------------------
# Types
# ----------------------------

@dataclass(frozen=True)
class StrategyTradeIntent:
    strategy: str
    pair: str
    side: str          # "buy" | "sell"
    ordertype: str     # "market" | "limit"
    volume: float
    notional_estimate: float
    price: Optional[float] = None

    # Optional upstream idempotency key (preferred if provided)
    idempotency_key: Optional[str] = None

    # Optional upstream metadata (helps make derived key stable across restarts)
    trace_id: Optional[str] = None
    cycle_id: Optional[str] = None

    # Canonical intent schema extensions (Phase-8 Session 1 / audit_m).
    # Defaults keep every existing construction site backward-compatible.
    confidence: float = 0.5
    entry_reason: str = ""
    regime_state: str = "unknown"
    expected_pnl: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    plan_hash: Optional[str] = None

    # Phase-8 Session 3 extensions: see IBKR StrategyTradeIntent for rationale.
    expected_price: float = 0.0
    signal_strength: float = 0.0


@dataclass(frozen=True)
class RiskGateResult:
    allowed: bool
    reason: str
    adjusted_notional: float


# ----------------------------
# Exec-State Durable Store
# ----------------------------

@dataclass(frozen=True)
class ExecStateClaim:
    inserted: bool
    status: str
    broker_order_id: Optional[str]
    submit_attempts: int
    claim_attempts: int


class ExecStateStore:
    """
    Durable, restart-safe execution state store.

    Backed by: data/exec_state/exec_state.sqlite3
    Uses the existing schema you already have:
      exec_state(idempotency_key PRIMARY KEY, status, broker, strategy, symbol, side, quantity, asset_class, broker_order_id, ...)
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            isolation_level=None,
        )
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def claim(
        self,
        *,
        idempotency_key: str,
        broker: str,
        strategy: str,
        symbol: str,
        side: str,
        quantity: float,
        asset_class: str,
        intent_canonical_json: str,
        extra_json: str,
    ) -> ExecStateClaim:
        now = _utc_now_iso()
        qty = float(quantity)

        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE;")
            con.execute(
                """
                INSERT OR IGNORE INTO exec_state (
                    idempotency_key, status,
                    created_at_utc, updated_at_utc,
                    broker, strategy, symbol, side, quantity, asset_class,
                    broker_order_id,
                    intent_canonical_json, extra_json,
                    claim_attempts, submit_attempts, last_error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    idempotency_key,
                    "CLAIMED",
                    now,
                    now,
                    broker,
                    strategy,
                    symbol,
                    side,
                    qty,
                    asset_class,
                    None,
                    intent_canonical_json,
                    extra_json,
                    1,
                    0,
                    None,
                ),
            )

            row = con.execute(
                "SELECT status, broker_order_id, submit_attempts, claim_attempts FROM exec_state WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()

            if row is None:
                con.execute("COMMIT;")
                return ExecStateClaim(False, "ERROR", None, 0, 0)

            status, broker_order_id, submit_attempts, claim_attempts = row
            inserted = (claim_attempts == 1) and (submit_attempts == 0) and (broker_order_id is None) and (status == "CLAIMED")

            if not inserted:
                con.execute(
                    "UPDATE exec_state SET claim_attempts = claim_attempts + 1, updated_at_utc=? WHERE idempotency_key=?",
                    (now, idempotency_key),
                )
                row2 = con.execute(
                    "SELECT status, broker_order_id, submit_attempts, claim_attempts FROM exec_state WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if row2:
                    status, broker_order_id, submit_attempts, claim_attempts = row2

            con.execute("COMMIT;")

        return ExecStateClaim(
            inserted=bool(inserted),
            status=str(status),
            broker_order_id=str(broker_order_id) if broker_order_id is not None else None,
            submit_attempts=int(submit_attempts or 0),
            claim_attempts=int(claim_attempts or 0),
        )

    def bump_submit_attempt(self, *, idempotency_key: str) -> int:
        now = _utc_now_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE;")
            con.execute(
                "UPDATE exec_state SET submit_attempts = submit_attempts + 1, updated_at_utc=? WHERE idempotency_key=?",
                (now, idempotency_key),
            )
            row = con.execute(
                "SELECT submit_attempts FROM exec_state WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            con.execute("COMMIT;")
        return int(row[0]) if row and row[0] is not None else 0

    def mark_submitted(self, *, idempotency_key: str, broker_order_id: str) -> None:
        now = _utc_now_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE;")
            con.execute(
                """
                UPDATE exec_state
                SET status = ?, broker_order_id = ?, updated_at_utc = ?, last_error = NULL
                WHERE idempotency_key=?
                """,
                ("SUBMITTED", str(broker_order_id), now, idempotency_key),
            )
            con.execute("COMMIT;")

    def mark_error(self, *, idempotency_key: str, err: str) -> None:
        now = _utc_now_iso()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE;")
            con.execute(
                "UPDATE exec_state SET status=?, last_error=?, updated_at_utc=? WHERE idempotency_key=?",
                ("ERROR", str(err)[:800], now, idempotency_key),
            )
            con.execute("COMMIT;")


# ----------------------------
# Executor
# ----------------------------

class KrakenExecutor:
    """
    High-level executor that:
      * Reads dynamic_caps.json
      * Checks a StrategyTradeIntent against per-strategy cap
      * If allowed, uses KrakenTradeRouter to place the order (validate-only or live)
      * If live and txids exist, logs TradeResult AND writes txid into exec_state (exactly-once)
    """

    def __init__(self, router: KrakenTradeRouter, caps_path: Optional[Path] = None, exec_db_path: Optional[Path] = None) -> None:
        self._router = router
        self._caps_path = caps_path or default_dynamic_caps_path()
        self._store = ExecStateStore(exec_db_path or _default_exec_state_db_path())

    def execute_with_risk(
        self,
        intent: StrategyTradeIntent,
        live: bool = False,
    ) -> Tuple[RiskGateResult, Optional[TradeResponse]]:
        caps_data = load_dynamic_caps(self._caps_path)
        rr = check_risk(caps_data=caps_data, intent=intent)

        risk_result = RiskGateResult(
            allowed=bool(rr.allowed),
            reason=str(rr.reason),
            adjusted_notional=float(rr.adjusted_notional),
        )

        if not risk_result.allowed:
            return risk_result, None

        # Validation-only mode: do NOT claim exec_state (no broker side effects)
        if not live:
            req = TradeRequest(
                pair=intent.pair,
                side=intent.side,
                ordertype=intent.ordertype,
                volume=float(intent.volume),
                price=float(intent.price) if intent.price is not None else None,
                validate_only=True,
            )
            resp = self._router.execute(req)
            return risk_result, resp

        # LIVE mode: enforce exactly-once
        idem_key, intent_json, extra_json = _build_idempotency_and_payload(intent)
        claim = self._store.claim(
            idempotency_key=idem_key,
            broker="kraken",
            strategy=str(intent.strategy),
            symbol=str(intent.pair),
            side=str(intent.side),
            quantity=float(intent.volume),
            asset_class="CRYPTO",
            intent_canonical_json=intent_json,
            extra_json=extra_json,
        )

        # If already submitted and broker_order_id exists -> suppress duplicate
        if claim.status.upper() == "SUBMITTED" and claim.broker_order_id:
            LOGGER.warning(
                "kraken.duplicate_suppressed_already_submitted",
                extra={
                    "idempotency_key": idem_key,
                    "pair": intent.pair,
                    "side": intent.side,
                    "volume": float(intent.volume),
                    "broker_order_id": claim.broker_order_id,
                    "claim_attempts": claim.claim_attempts,
                    "submit_attempts": claim.submit_attempts,
                },
            )
            # Return a synthetic TradeResponse that reflects already-known txid
            return risk_result, TradeResponse(txids=[str(claim.broker_order_id)], raw={"synthetic": True, "reason": "duplicate_suppressed"})

        # Submit (with attempt tracking; router/client handle low-level retry)
        submit_attempts = self._store.bump_submit_attempt(idempotency_key=idem_key)
        try:
            req = TradeRequest(
                pair=intent.pair,
                side=intent.side,
                ordertype=intent.ordertype,
                volume=float(intent.volume),
                price=float(intent.price) if intent.price is not None else None,
                validate_only=False,
            )
            LOGGER.info(
                "kraken.place_order_attempt",
                extra={
                    "idempotency_key": idem_key,
                    "submit_attempts": submit_attempts,
                    "pair": intent.pair,
                    "side": intent.side,
                    "volume": float(intent.volume),
                },
            )

            resp = self._router.execute(req)

            # If live and txids exist, log them and persist txid to exec_state
            if resp and getattr(resp, "txids", None):
                txid = str(resp.txids[0])
                self._store.mark_submitted(idempotency_key=idem_key, broker_order_id=txid)

                for t in resp.txids:
                    event = KrakenTradeEvent(
                        strategy=str(intent.strategy),
                        pair=str(intent.pair),
                        side=str(intent.side),
                        ordertype=str(intent.ordertype),
                        volume=float(intent.volume),
                        notional_estimate=float(intent.notional_estimate),
                        txid=str(t),
                        raw=dict(resp.raw),
                    )
                    log_kraken_trade_event(event)

            LOGGER.info(
                "EXECUTION_RESULT",
                extra={
                    "symbol": intent.pair,
                    "sec_type": "CRYPTO",
                    "exchange": "KRAKEN",
                    "side": intent.side,
                    "quantity": float(intent.volume),
                    "status": "submitted",
                    "classification": "SUBMITTED",
                    "error": None,
                    "strategy": intent.strategy,
                    "ts_utc": _utc_now_iso(),
                },
            )
            return risk_result, resp

        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            self._store.mark_error(idempotency_key=idem_key, err=err)
            LOGGER.warning(
                "kraken.place_order_failed",
                extra={
                    "idempotency_key": idem_key,
                    "pair": intent.pair,
                    "side": intent.side,
                    "volume": float(intent.volume),
                    "error": err,
                },
            )
            LOGGER.info(
                "EXECUTION_RESULT",
                extra={
                    "symbol": intent.pair,
                    "sec_type": "CRYPTO",
                    "exchange": "KRAKEN",
                    "side": intent.side,
                    "quantity": float(intent.volume),
                    "status": "error",
                    "classification": "FAILED",
                    "error": err,
                    "strategy": intent.strategy,
                    "ts_utc": _utc_now_iso(),
                },
            )
            raise


# ----------------------------
# Idempotency helpers
# ----------------------------

def _default_exec_state_db_path() -> Path:
    return Path("/home/ubuntu/chad_finale/data/exec_state/exec_state.sqlite3").resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Any) -> str:
    def _default(x: Any) -> str:
        return str(x)

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_default,
    )


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="strict")).hexdigest()


def _build_idempotency_and_payload(intent: StrategyTradeIntent) -> Tuple[str, str, str]:
    # Prefer upstream idempotency key if provided
    if intent.idempotency_key and str(intent.idempotency_key).strip():
        idem = str(intent.idempotency_key).strip()
        payload = {
            "broker": "kraken",
            "idempotency_key": idem,
            "pair": str(intent.pair),
            "side": str(intent.side),
            "ordertype": str(intent.ordertype),
            "volume": _fmt_qty(float(intent.volume)),
            "price": float(intent.price) if intent.price is not None else None,
            "strategy": str(intent.strategy),
        }
        return idem, _canonical_json(payload), _canonical_json({"source": "upstream"})

    meta: Dict[str, Any] = {}
    for k in ("trace_id", "cycle_id", "plan_hash"):
        v = getattr(intent, k, None)
        if v is not None and str(v).strip():
            meta[k] = str(v).strip()

    base_payload = {
        "broker": "kraken",
        "pair": str(intent.pair),
        "side": str(intent.side),
        "ordertype": str(intent.ordertype),
        "volume": _fmt_qty(float(intent.volume)),
        "price": float(intent.price) if intent.price is not None else None,
        "strategy": str(intent.strategy),
        "meta": meta,
    }
    intent_json = _canonical_json(base_payload)
    idem = f"kraken:{_sha256_hex(intent_json)}"
    return idem, intent_json, _canonical_json({"meta": meta, "derived_from": "canonical_intent_hash"})


def _fmt_qty(q: float) -> str:
    try:
        return f"{float(q):.10f}"
    except Exception:  # noqa: BLE001
        return "0.0000000000"


# ----------------------------
# CLI
# ----------------------------

def _build_executor_from_env(caps_path: Optional[Path] = None) -> KrakenExecutor:
    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    router = KrakenTradeRouter(client)
    return KrakenExecutor(router=router, caps_path=caps_path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Kraken Executor with CHAD risk caps + Phase 12 exactly-once.\n"
            "Reads runtime/dynamic_caps.json, enforces per-strategy dollar caps, "
            "routes trades through KrakenTradeRouter. Defaults to validate-only."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    order_parser = subparsers.add_parser(
        "test-intent",
        help="Send a single StrategyTradeIntent via CLI (default validate-only).",
    )
    order_parser.add_argument("--strategy", required=True)
    order_parser.add_argument("--pair", required=True)
    order_parser.add_argument("--side", required=True, choices=["buy", "sell"])
    order_parser.add_argument("--ordertype", required=True, choices=["market", "limit"])
    order_parser.add_argument("--volume", required=True, type=float)
    order_parser.add_argument("--notional", required=True, type=float)
    order_parser.add_argument("--price", required=False, type=float)
    order_parser.add_argument("--live", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "test-intent":
        caps_path = default_dynamic_caps_path()
        executor = _build_executor_from_env(caps_path=caps_path)

        intent = StrategyTradeIntent(
            strategy=str(args.strategy),
            pair=str(args.pair),
            side=str(args.side),
            ordertype=str(args.ordertype),
            volume=float(args.volume),
            notional_estimate=float(args.notional),
            price=float(args.price) if args.price is not None else None,
        )

        risk_result, resp = executor.execute_with_risk(intent=intent, live=bool(args.live))

        mode = "LIVE" if bool(args.live) else "VALIDATION ONLY"
        print("[KRAKEN EXECUTOR] Risk gate:")
        print("  allowed:", risk_result.allowed)
        print("  reason: ", risk_result.reason)
        print("  adjusted_notional:", f"{risk_result.adjusted_notional:.2f}")

        if resp is None:
            print(f"[KRAKEN EXECUTOR] No order sent ({mode}).")
            return 0

        print(f"[KRAKEN EXECUTOR] Order result ({mode}):")
        if resp.txids:
            print("  txids:", ", ".join(resp.txids))
        else:
            print("  (No txids returned; see raw below)")
        print("  raw:", resp.raw)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

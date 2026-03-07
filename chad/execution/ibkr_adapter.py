"""
chad/execution/ibkr_adapter.py

Phase 12 upgrade: broker execution hardening with durable exactly-once semantics.

What this module now guarantees (Phase 12 core requirements):
- Exactly-once *broker submission* for a given idempotency_key (durable across restarts)
- Safe retries with bounded backoff + jitter
- Durable broker_order_id writeback to exec_state.sqlite3
- Deterministic idempotency_key derivation (prefers upstream-provided keys, else hashes a canonical intent payload)

Design notes:
- This adapter is intentionally minimal and focused: it converts RoutedSignal -> IBKR order submission.
- LiveGate / SCR gating happens upstream. This adapter enforces *execution integrity* once called.
- DRY_RUN remains side-effect-free (no broker calls); it still returns SubmittedOrder objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Type-only imports (avoid importing ib_insync unless execution paths run)
from typing import TYPE_CHECKING

from chad.types import AssetClass, SignalSide

if TYPE_CHECKING:
    from ib_insync import IB, Contract, Order  # type: ignore
    from chad.analytics.shadow_router import RoutedSignal  # type: ignore

LOGGER = logging.getLogger("chad.execution.ibkr_adapter")


# ----------------------------
# Configuration + Output Types
# ----------------------------

@dataclass(frozen=True)
class IbkrConfig:
    """
    Configuration for IBKR adapter.

    dry_run:
      - True: never connect or submit orders; only logs + returns SubmittedOrder(dry_run=True)
      - False: connects and submits orders (requires upstream LiveGate/SCR gating)
    """
    dry_run: bool = True

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 19

    # Execution hardening
    submit_max_attempts: int = 3
    submit_base_backoff_s: float = 0.5
    submit_backoff_cap_s: float = 5.0
    submit_jitter_frac: float = 0.25

    # Exec-state DB path (durable exactly-once store)
    exec_state_db_path: Optional[str] = None


@dataclass(frozen=True)
class SubmittedOrder:
    symbol: str
    side: str
    quantity: float
    strategy: List[str]
    dry_run: bool
    submitted_at: datetime
    ib_order_id: Optional[int]


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

    Contract:
    - idempotency_key is PRIMARY KEY
    - claim() is INSERT OR IGNORE (exactly-once)
    - mark_submitted() persists broker_order_id and status transition
    - mark_error() persists failure and bumps attempts
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(
            str(self._db_path),
            timeout=5.0,
            isolation_level=None,  # autocommit; we manage transactions explicitly
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

            # Try insert (exactly-once)
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

            # Read row
            row = con.execute(
                """
                SELECT status, broker_order_id, submit_attempts, claim_attempts
                FROM exec_state
                WHERE idempotency_key=?
                """,
                (idempotency_key,),
            ).fetchone()

            if row is None:
                con.execute("COMMIT;")
                # Should not happen given insert+select, but fail closed
                return ExecStateClaim(inserted=False, status="ERROR", broker_order_id=None, submit_attempts=0, claim_attempts=0)

            status, broker_order_id, submit_attempts, claim_attempts = row

            # If this was NOT newly inserted, bump claim_attempts for audit
            inserted = (claim_attempts == 1) and (submit_attempts == 0) and (broker_order_id is None) and (status == "CLAIMED")
            if not inserted:
                con.execute(
                    """
                    UPDATE exec_state
                    SET claim_attempts = claim_attempts + 1,
                        updated_at_utc = ?
                    WHERE idempotency_key=?
                    """,
                    (now, idempotency_key),
                )
                # Refresh claim_attempts
                row2 = con.execute(
                    "SELECT status, broker_order_id, submit_attempts, claim_attempts FROM exec_state WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone()
                if row2:
                    status, broker_order_id, submit_attempts, claim_attempts = row2

            con.execute("COMMIT;")

        return ExecStateClaim(
            inserted=inserted,
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
                """
                UPDATE exec_state
                SET submit_attempts = submit_attempts + 1,
                    updated_at_utc = ?
                WHERE idempotency_key=?
                """,
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
                SET status = ?,
                    broker_order_id = ?,
                    updated_at_utc = ?,
                    last_error = NULL
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
                """
                UPDATE exec_state
                SET status = ?,
                    last_error = ?,
                    updated_at_utc = ?
                WHERE idempotency_key=?
                """,
                ("ERROR", str(err)[:800], now, idempotency_key),
            )
            con.execute("COMMIT;")

    def get(self, *, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT
                    idempotency_key, status, created_at_utc, updated_at_utc,
                    broker, strategy, symbol, side, quantity, asset_class,
                    broker_order_id, intent_canonical_json, extra_json,
                    claim_attempts, submit_attempts, last_error
                FROM exec_state
                WHERE idempotency_key=?
                """,
                (idempotency_key,),
            ).fetchone()
        if not row:
            return None
        keys = [
            "idempotency_key", "status", "created_at_utc", "updated_at_utc",
            "broker", "strategy", "symbol", "side", "quantity", "asset_class",
            "broker_order_id", "intent_canonical_json", "extra_json",
            "claim_attempts", "submit_attempts", "last_error",
        ]
        return {k: row[i] for i, k in enumerate(keys)}


# ----------------------------
# Adapter
# ----------------------------

class IbkrAdapter:
    """
    IBKR execution adapter using ib_insync.

    Phase 12 behavior:
    - Before any live broker submission, claim idempotency_key in exec_state.sqlite3.
    - If the key already exists and broker_order_id is present, do NOT re-submit.
    - On submit, write back broker_order_id and mark status SUBMITTED.
    - Retries are bounded and recorded (submit_attempts + last_error).
    """

    def __init__(self, config: IbkrConfig | None = None) -> None:
        self._config = config or IbkrConfig()
        self._ib: Optional["IB"] = None
        self._store = ExecStateStore(_resolve_exec_state_db_path(self._config.exec_state_db_path))

    # -------------------------
    # Connection management
    # -------------------------

    def ensure_connected(self) -> None:
        """
        Ensure there is an active connection to IBKR.

        In DRY_RUN mode this is a no-op.
        """
        if self._config.dry_run:
            return

        if self._ib is not None and getattr(self._ib, "isConnected", lambda: False)():
            return

        IB = _lazy_import_ib_insync_ib()
        ib = IB()
        ib.connect(
            self._config.host,
            int(self._config.port),
            clientId=int(self._config.client_id),
            timeout=10.0,
        )
        self._ib = ib

    def shutdown(self) -> None:
        try:
            if self._ib is not None:
                self._ib.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._ib = None

    # -------------------------
    # Order submission
    # -------------------------

    def submit_routed_signals(self, signals: Iterable["RoutedSignal"]) -> List[SubmittedOrder]:
        """
        Submit (or log) orders corresponding to routed signals.

        DRY_RUN:
          - no external side effects beyond logging.

        LIVE:
          - exactly-once submission enforced by exec_state store.
        """
        submitted: List[SubmittedOrder] = []

        for r in signals:
            try:
                so = self._submit_single_routed(r)
                if so is not None:
                    submitted.append(so)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception(
                    "ibkr.submit_routed_signal_failed",
                    extra={
                        "symbol": getattr(r, "symbol", ""),
                        "side": getattr(getattr(r, "side", None), "value", ""),
                        "net_size": getattr(r, "net_size", None),
                        "error": str(exc),
                    },
                )

        return submitted

    def _submit_single_routed(self, routed: "RoutedSignal") -> Optional[SubmittedOrder]:
        """
        Handle one RoutedSignal.

        Supported:
          - AssetClass.EQUITY
          - AssetClass.ETF
        """
        if float(getattr(routed, "net_size", 0.0) or 0.0) <= 0.0:
            LOGGER.info(
                "ibkr.skip_non_positive_size",
                extra={"symbol": routed.symbol, "net_size": routed.net_size},
            )
            return None

        asset_class = routed.asset_class
        if asset_class not in (AssetClass.EQUITY, AssetClass.ETF):
            LOGGER.info(
                "ibkr.skip_unsupported_asset_class",
                extra={"symbol": routed.symbol, "asset_class": asset_class.value},
            )
            return None

        side = routed.side
        qty = float(routed.net_size)
        symbol = str(routed.symbol)
        strategies = [s.value for s in routed.source_strategies]

        submitted_at = datetime.now(timezone.utc)

        # DRY_RUN: log only, do not touch exec_state (no broker side effects occur).
        if self._config.dry_run:
            LOGGER.info(
                "ibkr.dry_run_order",
                extra={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": qty,
                    "strategies": strategies,
                },
            )
            return SubmittedOrder(
                symbol=symbol,
                side=side.value.lower(),
                quantity=qty,
                strategy=strategies,
                dry_run=True,
                submitted_at=submitted_at,
                ib_order_id=None,
            )

        # Build canonical intent + idempotency key
        idem_key, intent_json, extra_json = _build_idempotency_and_payload(
            broker="ibkr",
            routed=routed,
            strategies=strategies,
        )

        # Claim exactly-once
        claim = self._store.claim(
            idempotency_key=idem_key,
            broker="ibkr",
            strategy=",".join(strategies) if strategies else "unknown",
            symbol=symbol,
            side=str(side.value).lower(),
            quantity=qty,
            asset_class=str(asset_class.value),
            intent_canonical_json=intent_json,
            extra_json=extra_json,
        )

        # If already submitted and broker_order_id exists -> DO NOT re-submit
        if claim.status.upper() == "SUBMITTED" and claim.broker_order_id:
            LOGGER.warning(
                "ibkr.duplicate_suppressed_already_submitted",
                extra={
                    "idempotency_key": idem_key,
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": qty,
                    "broker_order_id": claim.broker_order_id,
                    "claim_attempts": claim.claim_attempts,
                    "submit_attempts": claim.submit_attempts,
                },
            )
            return SubmittedOrder(
                symbol=symbol,
                side=side.value.lower(),
                quantity=qty,
                strategy=strategies,
                dry_run=False,
                submitted_at=submitted_at,
                ib_order_id=_safe_int(claim.broker_order_id),
            )

        # Live mode: ensure connection and place order with bounded retries
        self.ensure_connected()
        assert self._ib is not None  # for type-checkers

        contract = self._build_stock_contract(symbol)
        order = self._build_market_order(side, qty)

        last_err: Optional[str] = None
        for attempt in range(1, max(1, int(self._config.submit_max_attempts)) + 1):
            # record submit attempt before trying
            cur_attempts = self._store.bump_submit_attempt(idempotency_key=idem_key)

            try:
                LOGGER.info(
                    "ibkr.place_order_attempt",
                    extra={
                        "idempotency_key": idem_key,
                        "attempt": attempt,
                        "submit_attempts": cur_attempts,
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": qty,
                        "strategies": strategies,
                    },
                )

                trade = self._ib.placeOrder(contract, order)

                ib_order_id = getattr(getattr(trade, "order", None), "orderId", None)
                if ib_order_id is None:
                    ib_order_id = getattr(trade, "orderId", None)

                if ib_order_id is None:
                    raise RuntimeError("ibkr_placeOrder_missing_orderId")

                self._store.mark_submitted(idempotency_key=idem_key, broker_order_id=str(int(ib_order_id)))

                LOGGER.info(
                    "ibkr.place_order_submitted",
                    extra={
                        "idempotency_key": idem_key,
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": qty,
                        "broker_order_id": int(ib_order_id),
                    },
                )

                return SubmittedOrder(
                    symbol=symbol,
                    side=side.value.lower(),
                    quantity=qty,
                    strategy=strategies,
                    dry_run=False,
                    submitted_at=submitted_at,
                    ib_order_id=int(ib_order_id),
                )

            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                self._store.mark_error(idempotency_key=idem_key, err=last_err)

                LOGGER.warning(
                    "ibkr.place_order_failed",
                    extra={
                        "idempotency_key": idem_key,
                        "attempt": attempt,
                        "submit_attempts": cur_attempts,
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": qty,
                        "error": last_err,
                    },
                )

                if attempt >= int(self._config.submit_max_attempts):
                    break

                time.sleep(_backoff_s(
                    attempt=attempt,
                    base=float(self._config.submit_base_backoff_s),
                    cap=float(self._config.submit_backoff_cap_s),
                    jitter_frac=float(self._config.submit_jitter_frac),
                ))

        # Final failure: fail closed (no SubmittedOrder returned means upstream can decide)
        raise RuntimeError(f"ibkr_place_order_exhausted_retries: {last_err or 'unknown'}")

    # -------------------------
    # Contract & order helpers
    # -------------------------

    @staticmethod
    def _build_stock_contract(symbol: str) -> "Contract":
        # Using Stock shortcut from ib_insync
        Stock = _lazy_import_ib_insync_stock()
        return Stock(symbol, "SMART", "USD")

    @staticmethod
    def _build_market_order(side: SignalSide, quantity: float) -> "Order":
        Order = _lazy_import_ib_insync_order()
        action = "BUY" if side == SignalSide.BUY else "SELL"
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = float(quantity)
        return order


# ----------------------------
# Helpers
# ----------------------------

def _resolve_exec_state_db_path(override: Optional[str]) -> Path:
    """
    Canonical location for live exec_state DB:
      <repo_root>/data/exec_state/exec_state.sqlite3

    Resolution order:
    1) explicit override (config/env)
    2) walk upwards from this file until we find a repo root marker
    3) fail-closed to /home/ubuntu/chad_finale (SSOT deployment root)
    """
    if override and str(override).strip():
        return Path(str(override)).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "chad").is_dir() and (parent / "data").is_dir():
            return (parent / "data" / "exec_state" / "exec_state.sqlite3").resolve()

    # SSOT fallback (production layout)
    return Path("/home/ubuntu/chad_finale/data/exec_state/exec_state.sqlite3").resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except Exception:  # noqa: BLE001
        return None


def _backoff_s(*, attempt: int, base: float, cap: float, jitter_frac: float) -> float:
    a = max(1, int(attempt))
    b = max(0.01, float(base))
    c = max(b, float(cap))
    j = max(0.0, float(jitter_frac))

    raw = min(c, b * (2 ** (a - 1)))
    jitter = raw * j * (random.random() * 2.0 - 1.0)
    out = raw + jitter
    return float(max(0.05, min(c, out)))


def _canonical_json(obj: Any) -> str:
    """
    Deterministic JSON encoding for hashing/idempotency.
    """
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


def _build_idempotency_and_payload(
    *,
    broker: str,
    routed: "RoutedSignal",
    strategies: Sequence[str],
) -> tuple[str, str, str]:
    """
    Prefer an upstream deterministic idempotency key if present.
    Otherwise derive a stable key from a canonical intent payload.

    IMPORTANT:
    - This key must be stable across restarts for the *same* intended action.
    - Upstream should ideally pass a plan_hash / trace_id; we consume if present.
    """

    # Best-effort harvest of upstream identifiers if present
    upstream_key = getattr(routed, "idempotency_key", None)
    if upstream_key and str(upstream_key).strip():
        idem = str(upstream_key).strip()
        payload = {
            "broker": broker,
            "idempotency_key": idem,
            "symbol": str(routed.symbol),
            "side": str(routed.side.value),
            "quantity": float(routed.net_size),
            "asset_class": str(routed.asset_class.value),
            "strategies": sorted([str(s) for s in strategies]),
        }
        intent_json = _canonical_json(payload)
        return idem, intent_json, _canonical_json({"source": "upstream"})

    meta: Dict[str, Any] = {}
    for k in ("trace_id", "cycle_id", "plan_hash", "execution_plan_hash", "source_hash"):
        v = getattr(routed, k, None)
        if v is not None and str(v).strip():
            meta[k] = str(v).strip()

    base_payload = {
        "broker": broker,
        "symbol": str(routed.symbol),
        "side": str(routed.side.value),
        "quantity": _fmt_qty(float(routed.net_size)),
        "asset_class": str(routed.asset_class.value),
        "strategies": sorted([str(s) for s in strategies]),
        "meta": meta,  # empty dict is ok (still deterministic)
    }
    intent_json = _canonical_json(base_payload)
    idem = f"{broker}:{_sha256_hex(intent_json)}"
    extra = {"meta": meta, "derived_from": "canonical_intent_hash"}
    return idem, intent_json, _canonical_json(extra)


def _fmt_qty(q: float) -> str:
    # stable string formatting to reduce float drift in hashing
    try:
        return f"{float(q):.8f}"
    except Exception:  # noqa: BLE001
        return "0.00000000"


# ----------------------------
# Lazy imports (execution-only)
# ----------------------------

def _lazy_import_ib_insync_ib():
    try:
        from ib_insync import IB  # type: ignore
        return IB
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ib_insync import failed. Ensure ib-insync is installed in the active venv."
        ) from exc


def _lazy_import_ib_insync_stock():
    try:
        from ib_insync import Stock  # type: ignore
        return Stock
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ib_insync import failed (Stock). Ensure ib-insync is installed in the active venv."
        ) from exc


def _lazy_import_ib_insync_order():
    try:
        from ib_insync import Order  # type: ignore
        return Order
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ib_insync import failed (Order). Ensure ib-insync is installed in the active venv."
        ) from exc


# ----------------------------
# NOTE: RoutedSignal import
# ----------------------------
# Imported at bottom to avoid import cycles in some layouts.

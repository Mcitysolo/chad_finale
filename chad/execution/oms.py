"""
chad/execution/oms.py

Phase-8 Session 9 (A1): Order Management System — explicit Protocol +
venue-specific implementations.

The OMS owns **order lifecycle**: idempotent submission, order-state
tracking (SUBMITTED → FILLED / REJECTED / DUPLICATE_BLOCKED / ERROR),
fill recording, cancellation. It does NOT decide pricing, sizing, or
routing — those are EMS concerns (see chad/execution/ems.py).

Pure relocation note
--------------------

This session introduces the OMS **surface** without moving the existing
submission *implementation*. IbkrOMS and KrakenOMS wrap the existing
IbkrAdapter and KrakenExecutor respectively, translating between the
new (OrderRequest → OrderResult) Protocol and the existing
(StrategyTradeIntent → SubmittedOrder) adapter methods.

The wrapping approach guarantees:

  * Every status string currently emitted by IbkrAdapter survives
    byte-for-byte (``"submitted"``, ``"dry_run"``, ``"duplicate_blocked"``,
    ``"what-if"``, ``"error"``, ``"unknown"``).
  * ``paper_exec_evidence_writer`` and ``position_reconciler`` see the
    same ``SubmittedOrder`` dataclass they always have — it is
    preserved on ``OrderResult.raw`` for downstream consumers.
  * Existing call sites (``IbkrAdapter(...).submit_strategy_trade_intents``)
    continue to work unchanged.

Later sessions can move the underlying implementation into this file
without disturbing callers, because the Protocol already pins the
surface.

OrderRequest / OrderResult
--------------------------

These dataclasses are the target seam identified by
reports/audit_n_execution_landscape_20260421.json. They carry the
minimum info the OMS needs (intent + venue + sizing / pricing
parameters already resolved by the EMS) and return a uniform result
regardless of venue.

Status vocabulary (preserved)
-----------------------------

``OrderResult.status`` is the same string the underlying adapter
emitted. Canonical values observed in the codebase:

  * ``"submitted"`` — order accepted by broker
  * ``"dry_run"`` — dry-run mode, no broker call
  * ``"what-if"`` — IBKR whatIfOrder margin-check result
  * ``"duplicate_blocked"`` — idempotency store rejected replay
  * ``"error"`` — exception during submission
  * ``"unknown"`` — SubmittedOrder default (pre-submission)

Tests assert these strings survive any refactor (see
chad/tests/test_oms_ems_separation.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Protocol, runtime_checkable

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preserved status vocabulary (constants mirror string values in use)
# ---------------------------------------------------------------------------

STATUS_SUBMITTED: str = "submitted"
STATUS_DRY_RUN: str = "dry_run"
STATUS_WHAT_IF: str = "what-if"
STATUS_DUPLICATE_BLOCKED: str = "duplicate_blocked"
STATUS_ERROR: str = "error"
STATUS_UNKNOWN: str = "unknown"
STATUS_NO_RESULT: str = "no_result"

# Authoritative set — any OMS implementation MUST be able to emit these
# exact strings. Tests assert set equality on this constant.
PRESERVED_STATUS_STRINGS = frozenset(
    {
        STATUS_SUBMITTED,
        STATUS_DRY_RUN,
        STATUS_WHAT_IF,
        STATUS_DUPLICATE_BLOCKED,
        STATUS_ERROR,
        STATUS_UNKNOWN,
    }
)


# ---------------------------------------------------------------------------
# Data classes — the OMS surface
# ---------------------------------------------------------------------------


@dataclass
class OrderRequest:
    """Everything the OMS needs to submit one order.

    Produced by the EMS (``build_order_request`` on EMSInterface). The
    ``intent`` field carries the venue-specific StrategyTradeIntent so
    the OMS implementation can hand it directly to the underlying
    broker adapter without a second transformation step.
    """

    intent: Any  # IBKRStrategyTradeIntent | KrakenStrategyTradeIntent
    venue: str  # "ibkr" | "kraken" | "simulated"
    order_type: str = "LMT"
    limit_price: float = 0.0
    quantity: int = 0
    aggressive: bool = False
    idempotency_key: Optional[str] = None

    def intent_symbol(self) -> str:
        sym = getattr(self.intent, "symbol", "") or getattr(self.intent, "pair", "")
        return str(sym or "")


@dataclass
class OrderResult:
    """Canonical OMS result — venue-agnostic.

    The original venue-specific result (SubmittedOrder for IBKR,
    TradeResponse for Kraken) is preserved on ``raw`` so legacy
    consumers that already read it (paper_exec_evidence_writer,
    position_reconciler) see no behavioral change.
    """

    order_id: str = ""
    status: str = STATUS_UNKNOWN
    fill_price: float = 0.0
    fill_quantity: int = 0
    submitted_at: str = ""
    filled_at: str = ""
    rejection_reason: str = ""
    venue: str = ""
    raw: Any = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# OMS Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OMSInterface(Protocol):
    """Minimum contract for an Order Management System.

    Implementations are expected to be venue-specific (IbkrOMS,
    KrakenOMS) or mode-specific (future SimulatedOMS for A3 backtest
    unification). Every implementation must preserve the status
    vocabulary in ``PRESERVED_STATUS_STRINGS``.
    """

    def submit(self, request: OrderRequest) -> OrderResult:
        """Submit one order. Returns immediately with the broker's
        initial acceptance or rejection status — not a blocking fill."""
        ...

    def cancel(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True iff the cancel was
        accepted by the venue (idempotent: already-cancelled or
        already-filled orders return True)."""
        ...

    def get_status(self, order_id: str) -> OrderResult:
        """Query the current status of a previously submitted order."""
        ...


# ---------------------------------------------------------------------------
# IbkrOMS — wraps IbkrAdapter without moving its implementation
# ---------------------------------------------------------------------------


class IbkrOMS:
    """OMSInterface over an existing IbkrAdapter instance.

    Pure relocation wrapper: the submit path delegates to
    ``IbkrAdapter.submit_strategy_trade_intents`` so every existing
    behavior (idempotency, dry-run branch, what_if fallback, retries,
    SubmittedOrder structure) is inherited unchanged. Only the *surface*
    is new — OrderRequest → OrderResult — with the original
    SubmittedOrder preserved on ``result.raw``.
    """

    venue = "ibkr"

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def submit(self, request: OrderRequest) -> OrderResult:
        intents = [request.intent]
        try:
            submitted_list = self._adapter.submit_strategy_trade_intents(intents)
        except BaseException as exc:  # noqa: BLE001
            LOG.exception("ibkr_oms.submit_failed symbol=%s", request.intent_symbol())
            return OrderResult(
                order_id="",
                status=STATUS_ERROR,
                rejection_reason=str(exc),
                venue=self.venue,
                raw=None,
            )
        if not submitted_list:
            return OrderResult(status=STATUS_NO_RESULT, venue=self.venue, raw=None)
        submitted = submitted_list[0]
        return _submitted_order_to_result(submitted, venue=self.venue)

    def cancel(self, order_id: str) -> bool:
        """Stub for now — IbkrAdapter does not expose a cancel path in
        Session 9. Returns False so callers can detect the no-op.
        Future session will wire to IB's cancelOrder."""
        return False

    def get_status(self, order_id: str) -> OrderResult:
        """Stub for now — relocation-first. Returns unknown result."""
        return OrderResult(order_id=order_id, status=STATUS_UNKNOWN, venue=self.venue)


def _submitted_order_to_result(submitted: Any, venue: str = "ibkr") -> OrderResult:
    """Translate an IBKR SubmittedOrder dataclass into an OrderResult.

    Never raises — an unexpected SubmittedOrder shape returns a result
    with status='unknown' and the original object on ``raw``.
    """
    try:
        order_id_int = getattr(submitted, "ib_order_id", None)
        order_id = str(order_id_int) if order_id_int is not None else ""
        status = str(getattr(submitted, "status", STATUS_UNKNOWN) or STATUS_UNKNOWN)
        submitted_at_raw = getattr(submitted, "submitted_at", None)
        if isinstance(submitted_at_raw, datetime):
            submitted_at = submitted_at_raw.isoformat()
        else:
            submitted_at = str(submitted_at_raw or "")
        qty_raw = getattr(submitted, "quantity", 0.0)
        try:
            fill_qty = int(qty_raw) if qty_raw else 0
        except (TypeError, ValueError):
            fill_qty = 0
        limit_price = getattr(submitted, "limit_price", None)
        try:
            fill_price = float(limit_price) if limit_price is not None else 0.0
        except (TypeError, ValueError):
            fill_price = 0.0
        rejection = str(getattr(submitted, "error", "") or "")
        return OrderResult(
            order_id=order_id,
            status=status,
            fill_price=fill_price,
            fill_quantity=fill_qty,
            submitted_at=submitted_at,
            rejection_reason=rejection,
            venue=venue,
            raw=submitted,
        )
    except Exception:  # noqa: BLE001
        return OrderResult(status=STATUS_UNKNOWN, venue=venue, raw=submitted)


# ---------------------------------------------------------------------------
# KrakenOMS — wraps KrakenExecutor without moving its implementation
# ---------------------------------------------------------------------------


class KrakenOMS:
    """OMSInterface over a KrakenExecutor instance.

    Pure relocation wrapper. Kraken executor exposes
    ``execute_with_risk(intent)`` which returns a TradeResponse —
    we translate that into an OrderResult preserving the legacy
    semantics.
    """

    venue = "kraken"

    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def submit(self, request: OrderRequest) -> OrderResult:
        try:
            response = self._executor.execute_with_risk(request.intent)
        except BaseException as exc:  # noqa: BLE001
            LOG.exception("kraken_oms.submit_failed pair=%s", request.intent_symbol())
            return OrderResult(
                order_id="",
                status=STATUS_ERROR,
                rejection_reason=str(exc),
                venue=self.venue,
                raw=None,
            )
        if response is None:
            return OrderResult(status=STATUS_NO_RESULT, venue=self.venue, raw=None)
        return _kraken_response_to_result(response, venue=self.venue)

    def cancel(self, order_id: str) -> bool:
        return False

    def get_status(self, order_id: str) -> OrderResult:
        return OrderResult(order_id=order_id, status=STATUS_UNKNOWN, venue=self.venue)


def _kraken_response_to_result(response: Any, venue: str = "kraken") -> OrderResult:
    """Translate a Kraken TradeResponse into an OrderResult."""
    try:
        txid = getattr(response, "broker_order_id", None) or getattr(response, "txid", "")
        order_id = str(txid or "")
        # Kraken executor statuses observed: "submitted", "dry_run", "error",
        # "duplicate_blocked" — mirror the IBKR vocabulary.
        status = str(getattr(response, "status", STATUS_UNKNOWN) or STATUS_UNKNOWN)
        rejection = str(getattr(response, "error", "") or "")
        return OrderResult(
            order_id=order_id,
            status=status,
            rejection_reason=rejection,
            venue=venue,
            raw=response,
        )
    except Exception:  # noqa: BLE001
        return OrderResult(status=STATUS_UNKNOWN, venue=venue, raw=response)


# ---------------------------------------------------------------------------
# Null OMS — convenience for tests / dry-run wiring
# ---------------------------------------------------------------------------


class NullOMS:
    """No-op OMS that returns a predictable dry-run result.

    Useful for tests that want to exercise the EMS path without bringing
    up a broker adapter, and for the Session-10 SimulatedOMS stub to
    inherit from.
    """

    venue = "null"

    def submit(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            order_id="",
            status=STATUS_DRY_RUN,
            fill_price=request.limit_price,
            fill_quantity=request.quantity,
            submitted_at=_utc_now_iso(),
            venue=self.venue,
            raw=None,
        )

    def cancel(self, order_id: str) -> bool:
        return True

    def get_status(self, order_id: str) -> OrderResult:
        return OrderResult(order_id=order_id, status=STATUS_UNKNOWN, venue=self.venue)


__all__ = [
    "IbkrOMS",
    "KrakenOMS",
    "NullOMS",
    "OMSInterface",
    "OrderRequest",
    "OrderResult",
    "PRESERVED_STATUS_STRINGS",
    "STATUS_DRY_RUN",
    "STATUS_DUPLICATE_BLOCKED",
    "STATUS_ERROR",
    "STATUS_NO_RESULT",
    "STATUS_SUBMITTED",
    "STATUS_UNKNOWN",
    "STATUS_WHAT_IF",
]

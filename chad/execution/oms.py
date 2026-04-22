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


# ---------------------------------------------------------------------------
# SimulatedFillLedger + SimulatedOMS — Phase-8 Session 10 (A3 unification)
# ---------------------------------------------------------------------------


class SimulatedFillLedger:
    """In-memory record of simulated fills from backtest/paper-shadow runs.

    The ledger holds one dict per fill with the fields a backtest
    report wants to see: symbol, side, quantity, fill_price, order_id,
    timestamps, strategy, confidence, and the slippage model in use.

    The ledger is intentionally simple — it is NOT a position store.
    Backtest engines that want per-symbol P&L compute it from the
    fills, not from the ledger's internal state.
    """

    def __init__(self, ledger_path: str = "") -> None:
        self._fills: List[dict] = []
        self._rejections: List[dict] = []
        self.ledger_path = ledger_path

    def record(self, request: "OrderRequest", result: "OrderResult") -> None:
        """Append one fill record for a successfully-simulated order."""
        intent = request.intent
        self._fills.append(
            {
                "symbol": str(getattr(intent, "symbol", "") or getattr(intent, "pair", "")),
                "side": str(getattr(intent, "side", "")),
                "quantity": int(result.fill_quantity),
                "fill_price": float(result.fill_price),
                "order_id": result.order_id,
                "submitted_at": result.submitted_at,
                "filled_at": result.filled_at,
                "strategy": str(getattr(intent, "strategy", "")),
                "confidence": float(getattr(intent, "confidence", 0.0) or 0.0),
                "venue": result.venue,
                "slippage_bps_model": getattr(result, "_slippage_bps", None),
            }
        )

    def record_rejection(self, request: "OrderRequest", reason: str) -> None:
        """Append one rejection record (gate failure, size=0, etc.)."""
        intent = request.intent
        self._rejections.append(
            {
                "symbol": str(getattr(intent, "symbol", "") or getattr(intent, "pair", "")),
                "side": str(getattr(intent, "side", "")),
                "strategy": str(getattr(intent, "strategy", "")),
                "reason": str(reason or ""),
                "ts_utc": _utc_now_iso(),
            }
        )

    def get_fills(self) -> List[dict]:
        return list(self._fills)

    def get_rejections(self) -> List[dict]:
        return list(self._rejections)

    def clear(self) -> None:
        self._fills.clear()
        self._rejections.clear()

    @property
    def fill_count(self) -> int:
        return len(self._fills)

    @property
    def rejection_count(self) -> int:
        return len(self._rejections)


import json as _json
from pathlib import Path as _Path

_SIMULATED_OMS_CONFIG_PATH = _Path(__file__).resolve().parents[2] / "config" / "simulated_oms_config.json"


def _load_simulated_oms_config() -> dict:
    """Load config/simulated_oms_config.json; return defaults on any error."""
    defaults = {
        "slippage_bps_by_class": {"equity_etf": 3.0, "futures": 1.5, "crypto": 8.0},
        "default_slippage_bps": 3.0,
        "futures_symbols": ["MCL", "MES", "MNQ", "MGC", "M6E", "ZB", "ZN", "SIL", "ES", "NQ", "CL", "GC"],
    }
    if not _SIMULATED_OMS_CONFIG_PATH.is_file():
        return defaults
    try:
        data = _json.loads(_SIMULATED_OMS_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return defaults
    if not isinstance(data, dict):
        return defaults
    # Merge with defaults so partial configs still work.
    merged = dict(defaults)
    for key, val in data.items():
        merged[key] = val
    return merged


def _classify_symbol(symbol: str, futures_symbols: list) -> str:
    """Return 'crypto' / 'futures' / 'equity_etf' for a symbol.

    Rules:
      * Symbols ending in '-USD' (case-insensitive) → crypto.
      * Symbols in the futures_symbols list → futures.
      * Everything else → equity_etf.
    """
    sym = str(symbol or "").upper().strip()
    if not sym:
        return "equity_etf"
    if sym.endswith("-USD"):
        return "crypto"
    if sym in {str(s).upper() for s in (futures_symbols or [])}:
        return "futures"
    return "equity_etf"


class SimulatedOMS:
    """OMSInterface implementation for backtest + paper-shadow runs.

    Accepts an ``OrderRequest`` the same way IbkrOMS / KrakenOMS do,
    but fills it immediately against a simple slippage model and
    records the fill into a :class:`SimulatedFillLedger`.

    Slippage
    --------

    Side-aware bps model::

        BUY:  fill_price = limit_price * (1 + slippage_bps/10_000)
        SELL: fill_price = limit_price * (1 - slippage_bps/10_000)

    The ``slippage_bps`` passed to the constructor sets the DEFAULT
    rate. When the per-order asset class is detectable from the
    symbol (crypto / futures / equity), the class-specific rate from
    ``config/simulated_oms_config.json`` overrides the default. Pass
    ``use_config_overrides=False`` to disable per-class lookup and use
    the constructor value uniformly — useful for tests that want
    deterministic bps regardless of symbol.

    Status vocabulary
    -----------------

    Emits ``"submitted"`` on every fill so downstream code that expects
    the IbkrOMS vocabulary continues to work. A zero-quantity or
    non-positive-price request emits ``"error"`` with a descriptive
    ``rejection_reason``.
    """

    venue = "simulated"

    def __init__(
        self,
        ledger: Optional["SimulatedFillLedger"] = None,
        slippage_bps: float = 5.0,
        *,
        use_config_overrides: bool = True,
    ) -> None:
        self.ledger = ledger if ledger is not None else SimulatedFillLedger()
        self.slippage_bps = float(max(0.0, slippage_bps))
        self._order_counter = 0
        self._use_config_overrides = bool(use_config_overrides)
        cfg = _load_simulated_oms_config() if use_config_overrides else None
        self._class_bps = dict(cfg.get("slippage_bps_by_class", {})) if cfg else {}
        self._futures_symbols = list(cfg.get("futures_symbols", [])) if cfg else []
        self._default_bps_from_config = (
            float(cfg.get("default_slippage_bps", slippage_bps)) if cfg else slippage_bps
        )

    def _resolve_slippage_bps(self, symbol: str) -> float:
        """Pick the slippage bps for this order based on symbol class."""
        if not self._use_config_overrides or not self._class_bps:
            return self.slippage_bps
        klass = _classify_symbol(symbol, self._futures_symbols)
        try:
            return float(self._class_bps.get(klass, self._default_bps_from_config))
        except (TypeError, ValueError):
            return self.slippage_bps

    def _next_order_id(self) -> str:
        self._order_counter += 1
        return f"SIM_{self._order_counter:06d}"

    def submit(self, request: OrderRequest) -> OrderResult:
        qty = int(max(0, request.quantity))
        ref_price = float(request.limit_price or 0.0)
        if qty <= 0 or ref_price <= 0.0:
            result = OrderResult(
                order_id="",
                status=STATUS_ERROR,
                rejection_reason="simulated_non_positive_qty_or_price",
                venue=self.venue,
                raw=None,
            )
            self.ledger.record_rejection(request, result.rejection_reason)
            return result

        # Resolve per-order slippage from config by asset class.
        effective_bps = self._resolve_slippage_bps(request.intent_symbol())
        side = str(getattr(request.intent, "side", "") or "").upper()
        slip = ref_price * (effective_bps / 10_000.0)
        if side == "BUY":
            fill_price = ref_price + slip
        elif side in ("SELL", "SHORT"):
            fill_price = ref_price - slip
        else:
            fill_price = ref_price  # unknown side — no slippage adjustment

        now = _utc_now_iso()
        result = OrderResult(
            order_id=self._next_order_id(),
            status=STATUS_SUBMITTED,
            fill_price=round(fill_price, 6),
            fill_quantity=qty,
            submitted_at=now,
            filled_at=now,
            venue=self.venue,
            raw=None,
        )
        # Carry the effective bps on the result for the ledger without
        # widening OrderResult's public field surface.
        result._slippage_bps = effective_bps  # type: ignore[attr-defined]
        self.ledger.record(request, result)
        return result

    def cancel(self, order_id: str) -> bool:
        # Simulated orders fill instantly — cancel is always a no-op success.
        return True

    def get_status(self, order_id: str) -> OrderResult:
        # No asynchronous state in the simulator; fills are immediate.
        return OrderResult(
            order_id=order_id,
            status=STATUS_SUBMITTED,
            venue=self.venue,
        )


def compare_backtest_to_paper(
    backtest_fills: List[dict],
    paper_fills: List[dict],
) -> dict:
    """Compare a backtest fill ledger to a paper fill ledger.

    Returns a summary dict with:

        n_backtest, n_paper               — fill counts
        signal_overlap_pct                — fraction of backtest
            (symbol, side) pairs that also appear in paper
        mean_backtest_fill_price,
        mean_paper_fill_price,
        mean_fill_price_diff              — absolute difference in
            mean fill price (positive: backtest paid more)
        mean_slippage_diff_bps            — approximate slippage
            difference as basis points of mean price

    Consumed by operators validating that Session-10 routing did not
    introduce a systematic fill bias between simulation and live.
    """

    def _avg(xs: List[float]) -> Optional[float]:
        clean = [x for x in xs if isinstance(x, (int, float))]
        if not clean:
            return None
        return sum(clean) / len(clean)

    bt_prices = [f.get("fill_price") for f in backtest_fills]
    pp_prices = [f.get("fill_price") for f in paper_fills]

    bt_pairs = {(f.get("symbol"), f.get("side")) for f in backtest_fills}
    pp_pairs = {(f.get("symbol"), f.get("side")) for f in paper_fills}

    overlap = bt_pairs & pp_pairs
    overlap_pct = (len(overlap) / len(bt_pairs)) if bt_pairs else 0.0

    bt_mean = _avg(bt_prices)
    pp_mean = _avg(pp_prices)
    if bt_mean is None or pp_mean is None:
        mean_diff: Optional[float] = None
        slip_bps: Optional[float] = None
    else:
        mean_diff = bt_mean - pp_mean
        base = pp_mean or bt_mean or 1.0
        slip_bps = (mean_diff / base) * 10_000.0 if base else None

    return {
        "n_backtest": len(backtest_fills),
        "n_paper": len(paper_fills),
        "signal_overlap_pct": round(overlap_pct, 4),
        "mean_backtest_fill_price": bt_mean,
        "mean_paper_fill_price": pp_mean,
        "mean_fill_price_diff": mean_diff,
        "mean_slippage_diff_bps": slip_bps,
    }


__all__ = [
    "IbkrOMS",
    "KrakenOMS",
    "NullOMS",
    "OMSInterface",
    "OrderRequest",
    "OrderResult",
    "PRESERVED_STATUS_STRINGS",
    "SimulatedFillLedger",
    "SimulatedOMS",
    "STATUS_DRY_RUN",
    "STATUS_DUPLICATE_BLOCKED",
    "STATUS_ERROR",
    "STATUS_NO_RESULT",
    "STATUS_SUBMITTED",
    "STATUS_UNKNOWN",
    "STATUS_WHAT_IF",
    "compare_backtest_to_paper",
]

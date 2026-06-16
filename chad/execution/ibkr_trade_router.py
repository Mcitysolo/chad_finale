from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import os

from chad.execution.futures_gate import futures_execution_disabled, is_futures_sec_type

LOGGER = logging.getLogger("chad.execution.ibkr_trade_router")

_BROKER_TIMEOUT_S = 10.0


def _call_with_timeout(fn: Callable[..., Any], *args: Any, timeout_s: float = _BROKER_TIMEOUT_S, label: str = "broker_call") -> Any:
    """Run *fn(*args)* in a daemon thread with a hard wall-clock timeout."""
    result_box: list = []
    error_box: list = []

    def _target() -> None:
        try:
            result_box.append(fn(*args))
        except BaseException as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        LOGGER.error(
            "ibkr.broker_call_timeout",
            extra={"label": label, "timeout_s": timeout_s, "failure_class": "TIMEOUT"},
        )
        raise TimeoutError(
            f"Broker call {label!r} exceeded {timeout_s}s liveness deadline — failure_class=TIMEOUT"
        )

    if error_box:
        raise error_box[0]

    return result_box[0] if result_box else None


@dataclass(frozen=True)
class IBKRTradeRequest:
    """
    High-level description of a trade to send via IBKR.

    Fields:
        symbol: Underlying symbol, e.g. "AAPL", "SPY", "EUR.USD".
        sec_type: Security type, e.g. "STK", "CASH", "FUT".
        exchange: Primary exchange or SMART routing, e.g. "SMART".
        currency: Trade currency, e.g. "USD".
        side: "BUY" or "SELL".
        order_type: "MKT" or "LMT".
        quantity: Number of shares/contracts (for STK/FUT) or base amount (for CASH).
        limit_price: Optional limit price (required for LMT orders).
        what_if: If True, send a "what-if" order (no real execution).
    """

    symbol: str
    sec_type: str
    exchange: str
    currency: str
    side: str
    order_type: str
    quantity: float
    limit_price: Optional[float] = None
    what_if: bool = True


@dataclass(frozen=True)
class IBKRTradeResponse:
    """
    Normalized response for an IBKR order.

    Fields:
        order_id: The local order id assigned by IBKR.
        status: High-level status ("what-if", "submitted", etc.).
        raw: Raw ib_insync Order / Trade / CommissionReport info.
    """

    order_id: int
    status: str
    raw: Dict[str, Any]


class IBKRTradeRouter:
    """
    IBKR trade router using ib_insync.

    Responsibilities:
        * Build Contract and Order objects from IBKRTradeRequest.
        * Connect to IB Gateway/TWS using env vars for host/port/clientId.
        * Place a what-if or live order.
        * Return a normalized IBKRTradeResponse.

    This router DOES NOT:
        * Handle per-strategy caps (handled in CHAD's risk layer).
        * Deal with portfolio logic (that lives in the collector/executor).
    """

    def __init__(self, host: str, port: int, client_id: int, *, ib: Any = None) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        # P0-3: Accept an injected, externally managed IB session.
        # When provided, the router reuses it instead of creating a new one.
        self._ib = ib

    def _get_ib(self) -> Any:
        """Return the injected IB session, or raise if none was provided."""
        if self._ib is None:
            raise RuntimeError(
                "No IB session injected — IBKRTradeRouter requires an "
                "externally managed IB session passed via the ib= kwarg"
            )
        if hasattr(self._ib, "isConnected") and not self._ib.isConnected():
            raise RuntimeError(
                "Injected IB session is not connected — caller must "
                "ensure the session is connected before routing trades"
            )
        return self._ib

    def execute(self, req: IBKRTradeRequest) -> IBKRTradeResponse:
        """
        Execute a trade via IBKR. Uses what-if mode by default (no real execution).
        """
        # --- Futures execution off-switch: HARD GATE at this broker chokepoint ---
        # Second FUT-capable broker-submit path (alongside ibkr_adapter). When
        # any futures-disable flag is set, hard-block FUT/FOP submission here —
        # BOTH sides, what-if AND live — before building or routing anything.
        # Fail-closed: never call placeOrder for a gated futures contract;
        # return a clean rejected IBKRTradeResponse (never raise on the order
        # path). Reuses the SINGLE shared predicate (chad.execution.futures_gate)
        # so the flag logic is never duplicated. Non-futures secTypes unaffected.
        if futures_execution_disabled(os.environ) and is_futures_sec_type(req.sec_type):
            LOGGER.warning(
                "FUTURES_EXECUTION_GATE_BLOCKED symbol=%s side=%s sec_type=%s "
                "what_if=%s reason=futures_execution_disabled",
                req.symbol,
                req.side,
                req.sec_type,
                bool(req.what_if),
            )
            return IBKRTradeResponse(
                order_id=0,
                status="futures_execution_disabled",
                raw={
                    "guard": "futures_execution_disabled",
                    "symbol": req.symbol,
                    "side": req.side,
                    "sec_type": req.sec_type,
                    "what_if": bool(req.what_if),
                },
            )

        from ib_async import Contract, Order  # type: ignore[import]

        side = req.side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side: {req.side!r} (expected BUY or SELL)")

        order_type = req.order_type.upper()
        if order_type not in ("MKT", "LMT"):
            raise ValueError(f"Invalid order_type: {req.order_type!r} (expected MKT or LMT)")

        if order_type == "LMT" and req.limit_price is None:
            raise ValueError("Limit order requires limit_price")

        contract = Contract()
        contract.symbol = req.symbol
        contract.secType = req.sec_type
        contract.exchange = req.exchange
        contract.currency = req.currency

        order = Order()
        order.action = side
        order.orderType = order_type
        order.totalQuantity = float(req.quantity)
        if order_type == "LMT":
            order.lmtPrice = float(req.limit_price)
        # what-if mode: IB simulates margin/commission/impact without executing
        order.whatIf = bool(req.what_if)

        ib = self._get_ib()

        # For what-if orders, we use whatIfOrder() which returns a fully
        # simulated Order object with margin/commission details.
        if req.what_if:
            what_if_order = _call_with_timeout(ib.whatIfOrder, contract, order, label="whatIfOrder")
            # No real trade object in what-if mode.
            response_raw: Dict[str, Any] = {
                "what_if_order": what_if_order.__dict__ if what_if_order else {},
            }
            # In what-if mode, IB doesn't allocate a real orderId; use 0.
            LOGGER.info(
                "EXECUTION_RESULT",
                extra={
                    "symbol": req.symbol,
                    "sec_type": req.sec_type,
                    "exchange": req.exchange,
                    "side": req.side,
                    "quantity": req.quantity,
                    "status": "what-if",
                    "classification": "WHAT_IF",
                    "error": None,
                    "strategy": "",
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            return IBKRTradeResponse(
                order_id=getattr(what_if_order, "orderId", 0),
                status="what-if",
                raw=response_raw,
            )

        # LIVE order path
        def _place_and_wait() -> Any:
            t = ib.placeOrder(contract, order)
            ib.sleep(1.0)
            return t
        trade = _call_with_timeout(_place_and_wait, label="placeOrder")

        # Extract order id and status
        live_order = trade.order
        status = trade.orderStatus.status

        response_raw = {
            "order": live_order.__dict__ if live_order else {},
            "status": trade.orderStatus.__dict__,
            "fills": [f.__dict__ for f in trade.fills],
            "commissions": [c.__dict__ for c in trade.commissionReport] if trade.commissionReport else [],
        }

        LOGGER.info(
            "EXECUTION_RESULT",
            extra={
                "symbol": req.symbol,
                "sec_type": req.sec_type,
                "exchange": req.exchange,
                "side": req.side,
                "quantity": req.quantity,
                "status": status,
                "classification": "SUBMITTED",
                "error": None,
                "strategy": "",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return IBKRTradeResponse(
            order_id=getattr(live_order, "orderId", 0),
            status=status,
            raw=response_raw,
        )


# --------------------------------------------------------------------------- #
# CLI helpers                                                                 #
# --------------------------------------------------------------------------- #


def _env_or_default(name: str, default: Optional[str] = None) -> str:
    val = os.getenv(name)
    if val is not None:
        return val
    if default is not None:
        return default
    raise RuntimeError(f"Missing required env var: {name}")


def _build_router_from_env() -> IBKRTradeRouter:
    host = _env_or_default("IBKR_HOST", "127.0.0.1")
    port = int(_env_or_default("IBKR_PORT", "4002"))
    client_id = int(_env_or_default("IBKR_CLIENT_ID"))
    return IBKRTradeRouter(host=host, port=port, client_id=client_id)


def _cmd_test_order(args: argparse.Namespace) -> int:
    router = _build_router_from_env()

    req = IBKRTradeRequest(
        symbol=args.symbol,
        sec_type=args.sec_type,
        exchange=args.exchange,
        currency=args.currency,
        side=args.side,
        order_type=args.order_type,
        quantity=float(args.quantity),
        limit_price=float(args.limit_price) if args.limit_price is not None else None,
        what_if=not args.live,
    )

    try:
        resp = router.execute(req)
    except Exception as exc:  # noqa: BLE001
        print(f"[IBKR ROUTER] ERROR placing order: {exc}")
        return 1

    mode = "LIVE" if args.live else "WHAT-IF"
    print(f"[IBKR ROUTER] Order result ({mode}):")
    print(f"  order_id: {resp.order_id}")
    print(f"  status:   {resp.status}")
    print(f"  raw:      {resp.raw}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "IBKR Trade Router\n"
            "Uses IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID from env to place "
            "what-if or live orders via ib_insync."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    order_parser = subparsers.add_parser(
        "test-order",
        help="Place a single test order (WHAT-IF by default).",
    )
    order_parser.add_argument(
        "--symbol",
        required=True,
        help='Underlying symbol, e.g. "AAPL", "SPY", "EUR.USD".',
    )
    order_parser.add_argument(
        "--sec-type",
        required=True,
        help='Security type, e.g. "STK", "CASH", "FUT".',
    )
    order_parser.add_argument(
        "--exchange",
        required=True,
        help='Exchange or routing, e.g. "SMART".',
    )
    order_parser.add_argument(
        "--currency",
        required=True,
        help='Currency code, e.g. "USD".',
    )
    order_parser.add_argument(
        "--side",
        required=True,
        choices=["BUY", "SELL"],
        help='"BUY" or "SELL".',
    )
    order_parser.add_argument(
        "--order-type",
        required=True,
        choices=["MKT", "LMT"],
        help='"MKT" or "LMT".',
    )
    order_parser.add_argument(
        "--quantity",
        required=True,
        help="Quantity (shares/contracts or base amount).",
    )
    order_parser.add_argument(
        "--limit-price",
        required=False,
        help='Limit price (required for order-type="LMT").',
    )
    order_parser.add_argument(
        "--live",
        action="store_true",
        help="If set, place a LIVE order (default is WHAT-IF).",
    )

    args = parser.parse_args(argv)

    if args.command == "test-order":
        return _cmd_test_order(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

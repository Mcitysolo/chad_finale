from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig, KrakenAPIError, KrakenConfigError


@dataclass(frozen=True)
class TradeRequest:
    """
    High-level description of a trade we want to send to Kraken.

    Fields:
        pair: Kraken asset pair code (e.g., "XBTZCAD").
        side: "buy" or "sell".
        ordertype: "market" or "limit".
        volume: Base asset amount (e.g., 0.001 BTC).
        price: Optional limit price (required for ordertype="limit").
        validate_only: If True, Kraken will validate but not execute the order.
    """

    pair: str
    side: str
    ordertype: str
    volume: float
    price: Optional[float] = None
    validate_only: bool = True


@dataclass(frozen=True)
class TradeResponse:
    """
    Normalized response from Kraken trade router.

    Fields:
        txids: List of transaction IDs for placed orders (if any).
        raw: Raw Kraken result dict for inspection/logging.
    """

    txids: list[str]
    raw: Dict[str, Any]


class KrakenTradeRouter:
    """
    High-level router for sending CHAD trades to Kraken.

    Responsibilities:
        * Take a TradeRequest
        * Call KrakenClient.add_order(...)
        * Return a normalized TradeResponse

    This router does NOT:
        * Enforce per-strategy caps (that lives in CHAD's risk layer).
        * Do any withdrawal or funding operations.
    """

    def __init__(self, client: KrakenClient) -> None:
        self._client = client

    def execute(self, req: TradeRequest) -> TradeResponse:
        """
        Execute a trade request on Kraken.

        This method is safe to use with validate_only=True, in which case
        Kraken validates the order but does not execute it.
        """
        # Basic validation
        side = req.side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {req.side!r} (expected 'buy' or 'sell')")

        ordertype = req.ordertype.lower()
        if ordertype not in ("market", "limit"):
            raise ValueError(f"Invalid ordertype: {req.ordertype!r} (expected 'market' or 'limit')")

        if ordertype == "limit" and req.price is None:
            raise ValueError("Limit order requires a price")

        result = self._client.add_order(
            pair=req.pair,
            side=side,
            ordertype=ordertype,
            volume=req.volume,
            price=req.price,
            validate_only=req.validate_only,
        )

        txids = result.get("txid") or []
        if not isinstance(txids, list):
            txids = [str(txids)]

        return TradeResponse(txids=[str(t) for t in txids], raw=result)


# --------------------------------------------------------------------------- #
# CLI Helpers                                                                 #
# --------------------------------------------------------------------------- #


def _build_router_from_env() -> KrakenTradeRouter:
    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    return KrakenTradeRouter(client)


def _cmd_test_order(args: argparse.Namespace) -> int:
    """
    CLI command: place a single test order (validate-only by default).
    """
    router = _build_router_from_env()

    validate_only = not args.live  # default is validate-only unless --live is set

    try:
        req = TradeRequest(
            pair=args.pair,
            side=args.side,
            ordertype=args.ordertype,
            volume=float(args.volume),
            price=float(args.price) if args.price is not None else None,
            validate_only=validate_only,
        )
    except ValueError as exc:
        print(f"[KRAKEN ROUTER] Invalid arguments: {exc}")
        return 1

    try:
        resp = router.execute(req)
    except (KrakenAPIError, KrakenConfigError, ValueError, Exception) as exc:  # noqa: BLE001
        print(f"[KRAKEN ROUTER] ERROR placing order: {exc}")
        return 1

    mode = "VALIDATION ONLY" if validate_only else "LIVE"
    print(f"[KRAKEN ROUTER] Order result ({mode}):")
    if resp.txids:
        print("  txids:", ", ".join(resp.txids))
    else:
        print("  (No txids returned; check raw result below)")
    print("  raw:", resp.raw)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Kraken Trade Router\n"
            "Uses KRAKEN_API_KEY and KRAKEN_API_SECRET from the environment "
            "to place market/limit orders on Kraken. Defaults to validate-only "
            "mode for safety."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    # test-order subcommand
    order_parser = subparsers.add_parser(
        "test-order",
        help="Place a single test order (validate-only by default).",
    )
    order_parser.add_argument(
        "--pair",
        required=True,
        help='Kraken trading pair, e.g. "XBTZCAD".',
    )
    order_parser.add_argument(
        "--side",
        required=True,
        choices=["buy", "sell"],
        help='"buy" or "sell".',
    )
    order_parser.add_argument(
        "--ordertype",
        required=True,
        choices=["market", "limit"],
        help='"market" or "limit".',
    )
    order_parser.add_argument(
        "--volume",
        required=True,
        help="Base asset amount, e.g. 0.001.",
    )
    order_parser.add_argument(
        "--price",
        required=False,
        help='Limit price (required for ordertype="limit").',
    )
    order_parser.add_argument(
        "--live",
        action="store_true",
        help="If set, actually send a LIVE order (not just validate). "
             "Use with caution.",
    )

    args = parser.parse_args(argv)

    if args.command == "test-order":
        return _cmd_test_order(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

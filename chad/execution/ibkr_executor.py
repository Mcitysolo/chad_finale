from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import os

from chad.execution.ibkr_trade_router import (
    IBKRTradeRouter,
    IBKRTradeRequest,
    IBKRTradeResponse,
)


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyTradeIntent:
    """
    High-level intent for a single CHAD trade on IBKR.

    Fields:
        strategy: Strategy name, e.g. "alpha", "beta", "forex".
        symbol: Underlying, e.g. "AAPL", "SPY", "EUR.USD".
        sec_type: Security type: "STK", "CASH", "FUT".
        exchange: Exchange/routing: typically "SMART" for stocks/forex.
        currency: Trade currency: e.g. "USD", "CAD".
        side: "BUY" or "SELL".
        order_type: "MKT" or "LMT".
        quantity: Number of shares/contracts or base amount.
        notional_estimate: Approximate fiat notional (e.g. USD or CAD) for risk comparison.
        limit_price: Optional limit price (for LMT orders).
    """

    strategy: str
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    side: str
    order_type: str
    quantity: float
    notional_estimate: float
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class RiskGateResult:
    """
    Result of checking a trade against dynamic caps.

    Fields:
        allowed: Whether the trade is allowed at the requested notional.
        adjusted_notional: If we later implement downsizing, this will be <= requested.
        reason: Explanation of the decision.
    """

    allowed: bool
    adjusted_notional: float
    reason: str


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def default_dynamic_caps_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "runtime" / "dynamic_caps.json"


def load_dynamic_caps(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dynamic_caps.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_strategy_cap(caps_data: Dict[str, Any], strategy: str) -> float:
    strategy_caps = caps_data.get("strategy_caps", {})
    if strategy not in strategy_caps:
        raise KeyError(f"Strategy {strategy!r} not found in dynamic caps")
    return float(strategy_caps[strategy])


def check_risk(
    *,
    caps_data: Dict[str, Any],
    intent: StrategyTradeIntent,
) -> RiskGateResult:
    cap = get_strategy_cap(caps_data, intent.strategy)
    requested = float(intent.notional_estimate)

    if requested <= 0.0:
        return RiskGateResult(
            allowed=False,
            adjusted_notional=0.0,
            reason="Requested notional <= 0; nothing to do.",
        )

    if requested <= cap:
        return RiskGateResult(
            allowed=True,
            adjusted_notional=requested,
            reason=f"Requested notional {requested:.2f} <= cap {cap:.2f}",
        )

    # For now, reject if over cap.
    return RiskGateResult(
        allowed=False,
        adjusted_notional=cap,
        reason=(
            f"Requested notional {requested:.2f} exceeds cap {cap:.2f}; "
            f"trade blocked by risk gate."
        ),
    )


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


# --------------------------------------------------------------------------- #
# Executor                                                                    #
# --------------------------------------------------------------------------- #


class IBKRExecutor:
    """
    High-level executor that enforces CHAD risk caps for IBKR trades.

    Flow:
        * Load dynamic_caps.json
        * Check StrategyTradeIntent against its per-strategy cap
        * If allowed, map to IBKRTradeRequest and send via IBKRTradeRouter
        * If not allowed, block and return only the RiskGateResult
    """

    def __init__(self, router: IBKRTradeRouter, caps_path: Optional[Path] = None) -> None:
        self._router = router
        self._caps_path = caps_path or default_dynamic_caps_path()

    def execute_with_risk(
        self,
        intent: StrategyTradeIntent,
        live: bool = False,
    ) -> tuple[RiskGateResult, Optional[IBKRTradeResponse]]:
        caps_data = load_dynamic_caps(self._caps_path)
        risk_result = check_risk(caps_data=caps_data, intent=intent)

        if not risk_result.allowed:
            return risk_result, None

        req = IBKRTradeRequest(
            symbol=intent.symbol,
            sec_type=intent.sec_type,
            exchange=intent.exchange,
            currency=intent.currency,
            side=intent.side,
            order_type=intent.order_type,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            what_if=not live,
        )

        resp = self._router.execute(req)
        return risk_result, resp


def _build_executor_from_env(caps_path: Optional[Path] = None) -> IBKRExecutor:
    router = _build_router_from_env()
    return IBKRExecutor(router=router, caps_path=caps_path)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "IBKR Executor with CHAD risk caps.\n"
            "Reads runtime/dynamic_caps.json, enforces per-strategy dollar caps, "
            "and routes trades through IBKRTradeRouter. Defaults to WHAT-IF mode."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    order_parser = subparsers.add_parser(
        "test-intent",
        help="Send a single StrategyTradeIntent via CLI (WHAT-IF by default).",
    )
    order_parser.add_argument(
        "--strategy",
        required=True,
        help='Strategy name (must exist in dynamic caps), e.g. "alpha", "beta", "forex".',
    )
    order_parser.add_argument(
        "--symbol",
        required=True,
        help='Underlying symbol, e.g. "AAPL".',
    )
    order_parser.add_argument(
        "--sec-type",
        required=True,
        help='Security type, e.g. "STK", "CASH".',
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
        type=float,
        help="Quantity (shares/contracts).",
    )
    order_parser.add_argument(
        "--notional",
        required=True,
        type=float,
        help="Approximate fiat notional for risk comparison (e.g. USD).",
    )
    order_parser.add_argument(
        "--limit-price",
        required=False,
        type=float,
        help='Limit price (required if order-type="LMT").',
    )
    order_parser.add_argument(
        "--live",
        action="store_true",
        help="If set, place a LIVE order (default is WHAT-IF).",
    )

    args = parser.parse_args(argv)

    if args.command == "test-intent":
        caps_path = default_dynamic_caps_path()
        executor = _build_executor_from_env(caps_path=caps_path)

        intent = StrategyTradeIntent(
            strategy=args.strategy,
            symbol=args.symbol,
            sec_type=args.sec_type,
            exchange=args.exchange,
            currency=args.currency,
            side=args.side,
            order_type=args.order_type,
            quantity=args.quantity,
            notional_estimate=args.notional,
            limit_price=args.limit_price,
        )

        risk_result, resp = executor.execute_with_risk(
            intent=intent, live=args.live
        )

        print("[IBKR EXECUTOR] Risk gate:")
        print(f"  allowed: {risk_result.allowed}")
        print(f"  reason:  {risk_result.reason}")
        print(f"  adjusted_notional: {risk_result.adjusted_notional:.2f}")

        if not risk_result.allowed:
            print("[IBKR EXECUTOR] No order sent due to risk gate.")
            return 0

        mode = "LIVE" if args.live else "WHAT-IF"
        print(f"[IBKR EXECUTOR] Order result ({mode}):")

        if resp is None:
            print("  (No IBKRTradeResponse returned; this should not happen if allowed=True)")
            return 1

        print(f"  order_id: {resp.order_id}")
        print(f"  status:   {resp.status}")
        print(f"  raw:      {resp.raw}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

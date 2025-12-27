from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig
from chad.execution.kraken_trade_router import KrakenTradeRouter, TradeRequest, TradeResponse
from chad.risk.dynamic_risk_allocator import DynamicRiskAllocator, StrategyAllocation

from chad.portfolio.kraken_trade_result_logger import KrakenTradeEvent, log_kraken_trade_event


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyTradeIntent:
    """
    High-level intent for a single CHAD crypto trade.

    Fields:
        strategy: Strategy name, e.g. "crypto" or "alpha_crypto".
        pair: Kraken pair, e.g. "XXBTZCAD".
        side: "buy" or "sell".
        ordertype: "market" or "limit".
        volume: Base asset amount (float).
        notional_estimate: Approximate fiat notional (e.g., CAD) for this trade.
                           This is used to enforce per-strategy caps.
        price: Optional limit price (for limit orders).
    """

    strategy: str
    pair: str
    side: str
    ordertype: str
    volume: float
    notional_estimate: float
    price: Optional[float] = None


@dataclass(frozen=True)
class RiskGateResult:
    """
    Result of checking a trade against dynamic caps.
    """

    allowed: bool
    adjusted_notional: float
    reason: str


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def load_dynamic_caps(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"dynamic_caps.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_strategy_cap(caps_data: Dict[str, Any], strategy: str) -> float:
    strategy_caps = caps_data.get("strategy_caps", {})
    if strategy not in strategy_caps:
        raise KeyError(f"Strategy {strategy!r} not found in dynamic caps")
    return float(strategy_caps[strategy])


def check_risk(*, caps_data: Dict[str, Any], intent: StrategyTradeIntent) -> RiskGateResult:
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

    return RiskGateResult(
        allowed=False,
        adjusted_notional=cap,
        reason=(
            f"Requested notional {requested:.2f} exceeds cap {cap:.2f}; "
            f"trade blocked by risk gate."
        ),
    )


def default_dynamic_caps_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "runtime" / "dynamic_caps.json"


# --------------------------------------------------------------------------- #
# Executor                                                                    #
# --------------------------------------------------------------------------- #


class KrakenExecutor:
    """
    High-level executor that:
        * Reads dynamic_caps.json
        * Checks a StrategyTradeIntent against per-strategy cap
        * If allowed, uses KrakenTradeRouter to place the order (validate-only or live)
        * If live and txids exist, logs a TradeResult record to CHAD ledger.
    """

    def __init__(self, router: KrakenTradeRouter, caps_path: Optional[Path] = None) -> None:
        self._router = router
        self._caps_path = caps_path or default_dynamic_caps_path()

    def execute_with_risk(
        self,
        intent: StrategyTradeIntent,
        live: bool = False,
    ) -> tuple[RiskGateResult, Optional[TradeResponse]]:
        caps_data = load_dynamic_caps(self._caps_path)
        risk_result = check_risk(caps_data=caps_data, intent=intent)

        if not risk_result.allowed:
            return risk_result, None

        # Map StrategyTradeIntent to TradeRequest
        req = TradeRequest(
            pair=intent.pair,
            side=intent.side,
            ordertype=intent.ordertype,
            volume=float(intent.volume),
            price=float(intent.price) if intent.price is not None else None,
            validate_only=not live,
        )

        resp = self._router.execute(req)

        # If this was live, and we got txids, log them as TradeResults.
        if live and resp and getattr(resp, "txids", None):
            for txid in resp.txids:
                event = KrakenTradeEvent(
                    strategy=str(intent.strategy),
                    pair=str(intent.pair),
                    side=str(intent.side),
                    ordertype=str(intent.ordertype),
                    volume=float(intent.volume),
                    notional_estimate=float(intent.notional_estimate),
                    txid=str(txid),
                    raw=dict(resp.raw),
                )
                log_kraken_trade_event(event)

        return risk_result, resp


def _build_executor_from_env(caps_path: Optional[Path] = None) -> KrakenExecutor:
    cfg = KrakenClientConfig.from_env()
    client = KrakenClient(cfg)
    router = KrakenTradeRouter(client)
    return KrakenExecutor(router=router, caps_path=caps_path)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Kraken Executor with CHAD risk caps.\n"
            "Reads runtime/dynamic_caps.json, enforces per-strategy dollar caps, "
            "and routes trades through KrakenTradeRouter. Defaults to validate-only."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    order_parser = subparsers.add_parser(
        "test-intent",
        help="Send a single StrategyTradeIntent via CLI (default validate-only).",
    )
    order_parser.add_argument(
        "--strategy",
        required=True,
        help='Strategy name (must exist in dynamic caps), e.g. "crypto".',
    )
    order_parser.add_argument(
        "--pair",
        required=True,
        help='Kraken asset pair, e.g. "XXBTZCAD".',
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
        type=float,
        help="Base asset amount, e.g. 0.0001.",
    )
    order_parser.add_argument(
        "--notional",
        required=True,
        type=float,
        help="Approximate fiat notional (e.g. CAD) for risk comparison.",
    )
    order_parser.add_argument(
        "--price",
        required=False,
        type=float,
        help="Limit price (required if ordertype=limit).",
    )
    order_parser.add_argument(
        "--live",
        action="store_true",
        help="If set, actually send a LIVE order (not just validate).",
    )

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

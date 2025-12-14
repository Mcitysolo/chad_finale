from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from chad.exchanges.kraken_client import KrakenClient, KrakenClientConfig
from chad.execution.kraken_trade_router import KrakenTradeRouter, TradeRequest, TradeResponse
from chad.risk.dynamic_risk_allocator import DynamicRiskAllocator, StrategyAllocation


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyTradeIntent:
    """
    High-level intent for a single CHAD crypto trade.

    Fields:
        strategy: Strategy name, e.g. "crypto" or "alpha".
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

    Fields:
        allowed: Whether the trade is allowed at the requested size.
        adjusted_notional: If downsized, the new notional (<= requested).
        reason: Explanation if rejected or adjusted.
    """

    allowed: bool
    adjusted_notional: float
    reason: str


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def load_dynamic_caps(path: Path) -> Dict[str, Any]:
    """
    Load dynamic_caps.json as produced by DynamicRiskAllocator.
    """
    if not path.is_file():
        raise FileNotFoundError(f"dynamic_caps.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_strategy_cap(caps_data: Dict[str, Any], strategy: str) -> float:
    """
    Look up the per-strategy dollar cap for a given strategy name.

    Raises KeyError if the strategy is unknown.
    """
    strategy_caps = caps_data.get("strategy_caps", {})
    if strategy not in strategy_caps:
        raise KeyError(f"Strategy {strategy!r} not found in dynamic caps")
    return float(strategy_caps[strategy])


def check_risk(
    *,
    caps_data: Dict[str, Any],
    intent: StrategyTradeIntent,
) -> RiskGateResult:
    """
    Enforce per-strategy caps for a trade intent.

    Logic:
        * Look up per-strategy cap.
        * If notional_estimate <= cap: allowed.
        * If notional_estimate > cap: reject (for now) with explanation.

    NOTE:
        If you later want to allow partial sizing, you can change this to
        downsize notional instead of rejecting.
    """
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

    # For now, reject if over cap. We can later implement downsizing.
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
        * If allowed, uses KrakenTradeRouter to place the order (validate or live)
    """

    def __init__(
        self,
        router: KrakenTradeRouter,
        caps_path: Optional[Path] = None,
    ) -> None:
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
            volume=intent.volume,
            price=intent.price,
            validate_only=not live,
        )

        resp = self._router.execute(req)
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
            strategy=args.strategy,
            pair=args.pair,
            side=args.side,
            ordertype=args.ordertype,
            volume=args.volume,
            notional_estimate=args.notional,
            price=args.price,
        )

        risk_result, resp = executor.execute_with_risk(intent=intent, live=args.live)

        print("[KRAKEN EXECUTOR] Risk gate:")
        print(f"  allowed: {risk_result.allowed}")
        print(f"  reason:  {risk_result.reason}")
        print(f"  adjusted_notional: {risk_result.adjusted_notional:.2f}")

        if not risk_result.allowed:
            print("[KRAKEN EXECUTOR] No order sent due to risk gate.")
            return 0

        mode = "LIVE" if args.live else "VALIDATION ONLY"
        print(f"[KRAKEN EXECUTOR] Order result ({mode}):")
        if resp is None:
            print("  (No TradeResponse returned; this should not happen if allowed=True)")
            return 1

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

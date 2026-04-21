from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from chad.execution.ibkr_trade_router import (
    IBKRTradeRequest,
    IBKRTradeResponse,
    IBKRTradeRouter,
    _call_with_timeout,
)
from chad.execution.intent_schema import DEFAULT_TTL_SECONDS, utc_now_iso


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyTradeIntent:
    """
    High-level intent for a single CHAD trade on IBKR.
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

    # Canonical intent schema extensions (Phase-8 Session 1 / audit_m).
    # Defaults keep every existing construction site backward-compatible.
    confidence: float = 0.5
    entry_reason: str = ""
    regime_state: str = "unknown"
    expected_pnl: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)
    ttl_seconds: int = DEFAULT_TTL_SECONDS


@dataclass(frozen=True)
class RiskGateResult:
    """
    Result of checking a trade against dynamic caps, including
    normalization receipt fields for forensic auditability.
    """

    allowed: bool
    adjusted_notional: float
    reason: str
    original_quantity: float = 0.0
    submitted_quantity: float = 0.0
    normalized: bool = False
    normalization_reason: str = ""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


WHOLE_UNIT_SEC_TYPES = frozenset({"STK", "FUT", "OPT"})


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


def _safe_float(value: Any, *, field_name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} must be numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite: {value!r}")
    return out


def _normalized_sec_type(sec_type: str) -> str:
    out = str(sec_type or "").strip().upper()
    if not out:
        raise ValueError("sec_type is required")
    return out


def _normalize_quantity_for_ibkr(
    intent: StrategyTradeIntent,
) -> tuple[StrategyTradeIntent, bool, str, float]:
    """
    Normalize quantity into a broker-acceptable form for IBKR.

    Rules:
      - STK / FUT / OPT => whole units only, rounded DOWN.
      - CASH and other types => keep fractional quantity.
      - If rounding would reduce quantity to 0, reject the trade.
      - Recompute submitted notional proportionally after normalization.
    """
    sec_type = _normalized_sec_type(intent.sec_type)
    qty = _safe_float(intent.quantity, field_name="quantity")
    notional = _safe_float(intent.notional_estimate, field_name="notional_estimate")

    if qty <= 0.0:
        raise ValueError(f"quantity must be > 0, got {qty}")

    if notional < 0.0:
        raise ValueError(f"notional_estimate must be >= 0, got {notional}")

    original_quantity = qty
    submitted_qty = qty
    normalized = False
    normalization_reason = "quantity accepted as-is"

    if sec_type in WHOLE_UNIT_SEC_TYPES:
        floored = float(math.floor(qty))
        if floored < 1.0:
            raise ValueError(
                f"{sec_type} quantity rounds down to zero and cannot be submitted: "
                f"original_qty={qty:.8f}"
            )
        if abs(floored - qty) > 1e-12:
            submitted_qty = floored
            normalized = True
            normalization_reason = (
                f"{sec_type} requires whole units on this IBKR path: "
                f"original_qty={qty:.8f} normalized_qty={submitted_qty:.0f}"
            )

    submitted_notional = notional
    if qty > 0.0 and submitted_qty != qty:
        submitted_notional = float(notional * (submitted_qty / qty))

    normalized_intent = replace(
        intent,
        sec_type=sec_type,
        quantity=float(submitted_qty),
        notional_estimate=float(submitted_notional),
    )
    return normalized_intent, normalized, normalization_reason, float(original_quantity)


def check_risk(
    *,
    caps_data: Dict[str, Any],
    intent: StrategyTradeIntent,
    original_quantity: float,
    normalized: bool,
    normalization_reason: str,
) -> RiskGateResult:
    cap = get_strategy_cap(caps_data, intent.strategy)
    requested = float(intent.notional_estimate)
    submitted_quantity = float(intent.quantity)

    if requested <= 0.0:
        return RiskGateResult(
            allowed=False,
            adjusted_notional=0.0,
            reason="Requested notional <= 0 after IBKR normalization; nothing to do.",
            original_quantity=float(original_quantity),
            submitted_quantity=0.0,
            normalized=bool(normalized),
            normalization_reason=str(normalization_reason or "blocked_nonpositive_notional"),
        )

    if requested <= cap:
        suffix = ""
        if normalized:
            suffix = (
                f" | normalized qty {float(original_quantity):.12g}"
                f"->{submitted_quantity:.12g} ({normalization_reason})"
            )
        return RiskGateResult(
            allowed=True,
            adjusted_notional=float(requested),
            reason=f"Requested notional {requested:.2f} <= cap {cap:.2f}{suffix}",
            original_quantity=float(original_quantity),
            submitted_quantity=float(submitted_quantity),
            normalized=bool(normalized),
            normalization_reason=str(normalization_reason),
        )

    return RiskGateResult(
        allowed=False,
        adjusted_notional=float(cap),
        reason=(
            f"Requested notional {requested:.2f} exceeds cap {cap:.2f}; "
            f"trade blocked by risk gate."
        ),
        original_quantity=float(original_quantity),
        submitted_quantity=0.0,
        normalized=bool(normalized),
        normalization_reason="blocked_over_cap",
    )


# --------------------------------------------------------------------------- #
# Executor                                                                    #
# --------------------------------------------------------------------------- #


class IBKRExecutor:
    """
    High-level executor that enforces CHAD risk caps for IBKR trades.
    """

    def __init__(self, router: IBKRTradeRouter, caps_path: Optional[Path] = None) -> None:
        self._router = router
        self._caps_path = caps_path or default_dynamic_caps_path()

    def execute_with_risk(
        self,
        intent: StrategyTradeIntent,
        live: bool = False,
    ) -> tuple[RiskGateResult, Optional[IBKRTradeResponse]]:
        normalized_intent, normalized, normalization_reason, original_quantity = _normalize_quantity_for_ibkr(intent)

        caps_data = load_dynamic_caps(self._caps_path)
        risk_result = check_risk(
            caps_data=caps_data,
            intent=normalized_intent,
            original_quantity=original_quantity,
            normalized=normalized,
            normalization_reason=normalization_reason,
        )

        if not risk_result.allowed:
            return risk_result, None

        req = IBKRTradeRequest(
            symbol=normalized_intent.symbol,
            sec_type=normalized_intent.sec_type,
            exchange=normalized_intent.exchange,
            currency=normalized_intent.currency,
            side=normalized_intent.side,
            order_type=normalized_intent.order_type,
            quantity=float(risk_result.submitted_quantity),
            limit_price=normalized_intent.limit_price,
            what_if=not live,
        )

        resp = _call_with_timeout(self._router.execute, req, label="IBKRExecutor.execute")
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
    order_parser.add_argument("--strategy", required=True)
    order_parser.add_argument("--symbol", required=True)
    order_parser.add_argument("--sec-type", required=True)
    order_parser.add_argument("--exchange", required=True)
    order_parser.add_argument("--currency", required=True)
    order_parser.add_argument("--side", required=True, choices=["BUY", "SELL"])
    order_parser.add_argument("--order-type", required=True, choices=["MKT", "LMT"])
    order_parser.add_argument("--quantity", required=True, type=float)
    order_parser.add_argument("--notional", required=True, type=float)
    order_parser.add_argument("--limit-price", required=False, type=float)
    order_parser.add_argument("--live", action="store_true")

    args = parser.parse_args(argv)

    if args.command != "test-intent":
        parser.print_help()
        return 1

    intent = StrategyTradeIntent(
        strategy=args.strategy,
        symbol=args.symbol,
        sec_type=args.sec_type,
        exchange=args.exchange,
        currency=args.currency,
        side=args.side,
        order_type=args.order_type,
        quantity=float(args.quantity),
        notional_estimate=float(args.notional),
        limit_price=float(args.limit_price) if args.limit_price is not None else None,
    )

    executor = _build_executor_from_env()
    risk_result, resp = executor.execute_with_risk(intent=intent, live=bool(args.live))

    print("[IBKR EXECUTOR] Risk gate:")
    print(f"  allowed: {risk_result.allowed}")
    print(f"  reason:  {risk_result.reason}")
    print(f"  adjusted_notional: {risk_result.adjusted_notional:.2f}")
    print(f"  original_quantity: {risk_result.original_quantity}")
    print(f"  submitted_quantity: {risk_result.submitted_quantity}")
    print(f"  normalized: {risk_result.normalized}")
    print(f"  normalization_reason: {risk_result.normalization_reason}")

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


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Mapping

from chad.types import TradeSignal, SignalSide, StrategyName, AssetClass
from chad.utils.signal_router import SignalRouter
import chad.policy as policy
from chad.risk.daily_throttle import throttle_signals_from_dynamic_caps


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_demo_signals() -> List[TradeSignal]:
    now = datetime.now(timezone.utc)
    return [
        # Two strategies same symbol same side -> router merge
        TradeSignal(symbol="AAPL", side=SignalSide.BUY, size=10.0, strategy=StrategyName.ALPHA, confidence=0.9, asset_class=AssetClass.EQUITY, created_at=now, meta={"case": "merge"}),
        TradeSignal(symbol="AAPL", side=SignalSide.BUY, size=5.0, strategy=StrategyName.GAMMA, confidence=0.5, asset_class=AssetClass.EQUITY, created_at=now, meta={"case": "merge"}),

        # Opposite side must not net in router
        TradeSignal(symbol="AAPL", side=SignalSide.SELL, size=7.0, strategy=StrategyName.GAMMA, confidence=0.8, asset_class=AssetClass.EQUITY, created_at=now, meta={"case": "no_net"}),

        # Another symbol to potentially trigger caps
        TradeSignal(symbol="SPY", side=SignalSide.BUY, size=50.0, strategy=StrategyName.BETA, confidence=0.7, asset_class=AssetClass.ETF, created_at=now, meta={"case": "caps"}),
    ]


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def json_safe(obj: Any) -> Any:
    """
    Convert nested dataclasses/enums/datetimes into JSON-serializable structures.
    Deterministic and side-effect free.
    """
    from enum import Enum
    from datetime import datetime

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return str(obj)


def main() -> None:
    root = Path("/home/ubuntu/CHAD FINALE")
    env_path = os.environ.get("CHAD_DYNAMIC_CAPS_PATH")
    if env_path:
        dyn_caps_path = Path(env_path)
    else:
        dyn_caps_path = root / "runtime" / "dynamic_caps.json"

    # Prices for notional calcs (deterministic)
    price_map: Mapping[str, float] = {"AAPL": 20.0, "SPY": 10.0}

    signals = build_demo_signals()

    router = SignalRouter()
    routed = router.route(signals)

    # Policy evaluation operates on TradeSignals (per your design)
    eng = policy.PolicyEngine(
        strategy_limits=policy.build_default_strategy_limits(),
        global_limits=policy.build_default_global_limits(),
    )
    evaluated = eng.evaluate_signals(
        signals,
        current_symbol_notional={},
        current_total_notional=0.0,
        prices=price_map,
    )

    # Throttle operates on RoutedSignals using dynamic_caps.json
    throttle_res = throttle_signals_from_dynamic_caps(
        routed,
        price_map,
        dynamic_caps_path=dyn_caps_path,
        today=date.today(),
        log_path=None,
        fail_closed=True,
    )

    out: Dict[str, Any] = {
        "ts_utc": utc_now_iso(),
        "dynamic_caps_path": str(dyn_caps_path),
        "inputs": [asdict(s) for s in signals],
        "routed": [asdict(r) for r in routed],
        "policy": [
            {
                "signal": asdict(es.signal),
                "decision": {
                    "accepted": es.decision.accepted,
                    "reason": es.decision.reason,
                    "adjusted_size": es.decision.adjusted_size,
                },
            }
            for es in evaluated
        ],
        "throttle": {
            "accepted": [asdict(s) for s in throttle_res.accepted],
            "rejected": list(throttle_res.rejected),
            "total_notional": throttle_res.total_notional,
            "symbol_notional": _safe_json(throttle_res.symbol_notional),
            "strategy_notional": _safe_json(throttle_res.strategy_notional),
        },
        "purity_claim": {
            "writes_runtime_files": False,
            "note": "This script only reads runtime/dynamic_caps.json and prints JSON. It does not write runtime state.",
        },
    }

    print(json.dumps(json_safe(out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

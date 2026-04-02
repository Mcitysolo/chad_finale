# execution_intelligence.py
# This file preserves the original Delta execution monitoring logic.

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from chad.types import StrategyConfig, StrategyName


@dataclass(frozen=True)
class ExecutionIntelligenceParams:
    enabled: bool = True
    max_slippage_bps_warn: float = 10.0
    max_spread_bps_warn: float = 15.0
    min_fill_ratio_warn: float = 0.90


def build_execution_intelligence_config() -> StrategyConfig:
    return StrategyConfig(
        name=StrategyName.DELTA,
        enabled=True,
        target_universe=["SPY", "QQQ"],
    )


def _safe_iterable(obj: Any) -> Iterable[Any]:
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.values()
    if isinstance(obj, (list, tuple, set)):
        return obj
    return [obj]


def _extract_execution_records(ctx: Any) -> List[Dict[str, Any]]:
    candidates: List[Any] = []
    for attr in ("executions", "execution_history"):
        candidates.extend(list(_safe_iterable(getattr(ctx, attr, None))))

    records: List[Dict[str, Any]] = []
    for rec in candidates:
        if isinstance(rec, dict):
            records.append(rec)
    return records


def execution_intelligence_handler(ctx: Any, params: ExecutionIntelligenceParams) -> List:
    if not params.enabled:
        return []

    records = _extract_execution_records(ctx)

    slippages = []
    spreads = []
    fills = []

    for r in records:
        try:
            if r.get("slippage_bps") is not None:
                slippages.append(float(r["slippage_bps"]))
            if r.get("spread_bps") is not None:
                spreads.append(float(r["spread_bps"]))
            if r.get("fill_ratio") is not None:
                fills.append(float(r["fill_ratio"]))
        except Exception:
            continue

    avg_slippage = mean(slippages) if slippages else None
    avg_spread = mean(spreads) if spreads else None
    avg_fill = mean(fills) if fills else None

    # No trade signals emitted.
    return []

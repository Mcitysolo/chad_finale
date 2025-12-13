from __future__ import annotations

"""
DeltaBrain – execution intelligence / meta-signal strategy.

Purpose
-------
DeltaBrain is responsible for analysing execution quality, slippage, and
liquidity conditions and surfacing *meta-insights* that can later influence
position sizing, venue selection, or strategy weights.

Phase-7 / DRY_RUN Design
------------------------
- Fully production-safe: emits NO trade signals by default.
- Zero external side effects beyond optional logging.
- Handles unknown / partial context structures defensively.
- Designed for incremental enhancement without breaking callers.

Assumptions
-----------
- StrategyName has a DELTA member.
- StrategyConfig matches the structure used by alpha / beta / gamma.
- The StrategyEngine can register Delta with a handler of the form:
      handler(ctx, params) -> list[StrategySignal]  (we use List for now).
"""

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from chad.types import StrategyConfig, StrategyName

# NOTE:
# If/when you want Delta to emit real StrategySignal objects, you can:
#   - import StrategySignal from chad.types, and
#   - change the return type of delta_handler to List[StrategySignal].
# For now we intentionally return a bare List to avoid coupling to a specific
# signal shape while keeping the implementation production-safe.


@dataclass(frozen=True)
class DeltaParams:
    """
    Configuration for DeltaBrain.

    Attributes
    ----------
    enabled:
        Master on/off switch for the strategy.
    max_slippage_bps_warn:
        Warning threshold for average slippage, in basis points (0.01%).
    max_spread_bps_warn:
        Warning threshold for average quoted spread, in basis points.
    min_fill_ratio_warn:
        Warning threshold for fill ratio (0–1) under which conditions are
        considered degraded.
    """

    enabled: bool = True
    max_slippage_bps_warn: float = 10.0
    max_spread_bps_warn: float = 15.0
    min_fill_ratio_warn: float = 0.90


@dataclass(frozen=True)
class ExecutionInsight:
    """
    Aggregated execution-quality snapshot used internally by DeltaBrain.

    This is intentionally small and immutable; it can be logged, persisted,
    or later transformed into StrategySignals without changing handler logic.
    """

    sample_count: int
    avg_slippage_bps: Optional[float]
    max_slippage_bps: Optional[float]
    avg_spread_bps: Optional[float]
    avg_fill_ratio: Optional[float]
    degraded: bool


def build_delta_config() -> StrategyConfig:
    """
    Build the StrategyConfig for DeltaBrain.

    Delta typically does not own a dedicated symbol universe: it observes
    executions across strategies. To remain compatible with the existing
    StrategyConfig contract, we provide a conservative equity universe.

    Returns
    -------
    StrategyConfig
        Configuration object for registering Delta with the StrategyEngine.
    """
    return StrategyConfig(
        name=StrategyName.DELTA,
        enabled=True,
        target_universe=["SPY", "QQQ"],
    )


def _safe_iterable(obj: Any) -> Iterable[Any]:
    """
    Helper that turns arbitrary objects into an iterable or returns an
    empty list if that is not possible. This protects DeltaBrain from
    unexpected context shapes.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        return obj.values()
    if isinstance(obj, (list, tuple, set)):
        return obj
    # Last resort: single element
    return [obj]


def _extract_execution_records(ctx: Any) -> List[Dict[str, Any]]:
    """
    Attempt to extract execution records from the strategy context.

    Expected (but not required) shapes:
    - ctx.executions: iterable of dict-like objects
    - ctx.execution_history: iterable of dict-like objects

    Each record may contain:
    - 'slippage_bps'   : float
    - 'spread_bps'     : float
    - 'fill_ratio'     : float in [0, 1]

    Any deviations are handled defensively; invalid records are ignored.
    """
    candidates: List[Any] = []

    for attr in ("executions", "execution_history"):
        value = getattr(ctx, attr, None)
        candidates.extend(list(_safe_iterable(value)))

    records: List[Dict[str, Any]] = []
    for rec in candidates:
        if isinstance(rec, dict):
            records.append(rec)
            continue
        # Allow simple obj-with-attributes style
        try:
            maybe_record: Dict[str, Any] = {
                "slippage_bps": getattr(rec, "slippage_bps", None),
                "spread_bps": getattr(rec, "spread_bps", None),
                "fill_ratio": getattr(rec, "fill_ratio", None),
            }
            if any(v is not None for v in maybe_record.values()):
                records.append(maybe_record)
        except Exception:
            # Any unexpected shape is ignored; Delta never raises.
            continue

    return records


def _compute_insight(records: List[Dict[str, Any]], params: DeltaParams) -> ExecutionInsight:
    """
    Compute aggregated execution metrics from raw records.

    All calculations are defensive:
    - Missing or non-numeric fields are ignored.
    - If no valid values are present, corresponding metrics are None.
    """
    slippages: List[float] = []
    spreads: List[float] = []
    fill_ratios: List[float] = []

    for rec in records:
        try:
            s = rec.get("slippage_bps")
            if s is not None:
                slippages.append(float(s))
        except Exception:
            pass

        try:
            sp = rec.get("spread_bps")
            if sp is not None:
                spreads.append(float(sp))
        except Exception:
            pass

        try:
            fr = rec.get("fill_ratio")
            if fr is not None:
                fill_ratios.append(float(fr))
        except Exception:
            pass

    avg_slippage = mean(slippages) if slippages else None
    max_slippage = max(slippages) if slippages else None
    avg_spread = mean(spreads) if spreads else None
    avg_fill_ratio = mean(fill_ratios) if fill_ratios else None

    degraded = False
    if avg_slippage is not None and avg_slippage > params.max_slippage_bps_warn:
        degraded = True
    if avg_spread is not None and avg_spread > params.max_spread_bps_warn:
        degraded = True
    if avg_fill_ratio is not None and avg_fill_ratio < params.min_fill_ratio_warn:
        degraded = True

    return ExecutionInsight(
        sample_count=len(records),
        avg_slippage_bps=avg_slippage,
        max_slippage_bps=max_slippage,
        avg_spread_bps=avg_spread,
        avg_fill_ratio=avg_fill_ratio,
        degraded=degraded,
    )


def delta_handler(ctx: Any, params: DeltaParams) -> List:
    """
    Delta strategy handler (production-safe, meta-analytic).

    Current behaviour (Phase-7 / DRY_RUN baseline)
    ----------------------------------------------
    - If params.enabled is False:
        Returns [].
    - If enabled:
        - Extracts execution records from the context (if present).
        - Computes aggregate execution metrics via ExecutionInsight.
        - Returns [] (no trade signals) to avoid impacting current behaviour.

    Future extension (Phase-8+)
    ---------------------------
    You can evolve this handler to:
    - Emit StrategySignals that down-weight strategies when execution
      quality is degraded.
    - Raise alerts via your telemetry stack when fill ratios drop or
      slippage spikes.
    - Feed these insights into SCR or the DynamicRiskAllocator.

    Parameters
    ----------
    ctx : Any
        Strategy context. Kept untyped here to avoid coupling DeltaBrain to a
        specific context implementation.
    params : DeltaParams
        Configuration for thresholding degradations.

    Returns
    -------
    List
        An empty list in this baseline implementation (no trade signals).
    """
    if not params.enabled:
        return []

    records = _extract_execution_records(ctx)
    _ = _compute_insight(records, params=params)

    # IMPORTANT:
    # We intentionally do not mutate `ctx`, write to disk, or emit signals yet.
    # This guarantees that adding Delta to the registry cannot cause side
    # effects until you explicitly decide how to use ExecutionInsight.
    return []

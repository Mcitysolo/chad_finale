from __future__ import annotations

"""
AlphaForex – Institutional-grade FX Alpha engine (Phase-7 safe baseline).

Design Philosophy
-----------------
AlphaForex targets major FX pairs via your configured broker/feeds. This
baseline implementation is engineered for production safety and future
expansion, but emits NO orders until explicitly activated.

Guarantees
----------
- Deterministic, side-effect free (no network, disk, or shared-state writes).
- Fully typed, with immutable configuration objects.
- Defensive against malformed or partial context.
- Compatible with existing StrategyEngine + StrategyConfig contracts.
- Ready to host serious intraday FX logic (trend, carry, vol filters, etc.).

You can later layer in:
- Volatility-normalised trend following
- Carry / funding-aware filters
- Regime detection (risk-on / risk-off)
- Execution-cost models across venues
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from chad.types import StrategyConfig, StrategyName


FOREX_UNIVERSE_DEFAULT: List[str] = ["EUR-USD", "GBP-USD", "USD-CAD", "USD-JPY"]


# ---------------------------------------------------------------------------
# Parameter Object (Immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaForexParams:
    """
    Tunable parameters for AlphaForex.

    Attributes
    ----------
    enabled:
        Master switch for strategy participation.
    universe:
        List of FX symbols to monitor. Defaults to FOREX_UNIVERSE_DEFAULT.
    min_liquidity_usd:
        Reject assets below this estimated liquidity threshold (placeholder).
    max_volatility:
        Reject signals when volatility exceeds this number (placeholder).
    """

    enabled: bool = True
    universe: Optional[List[str]] = None
    min_liquidity_usd: float = 100_000_000.0
    max_volatility: float = 4.0

    def actual_universe(self) -> List[str]:
        """
        Return a stable, fully materialised FX universe.

        If the operator provides an invalid universe, this method gracefully
        falls back to FOREX_UNIVERSE_DEFAULT.
        """
        if self.universe is None:
            return list(FOREX_UNIVERSE_DEFAULT)
        try:
            return list(self.universe)
        except Exception:
            return list(FOREX_UNIVERSE_DEFAULT)


# ---------------------------------------------------------------------------
# StrategyConfig Factory
# ---------------------------------------------------------------------------

def build_alpha_forex_config() -> StrategyConfig:
    """
    Creates the StrategyConfig required by CHAD’s StrategyEngine.

    Notes
    -----
    - StrategyName.ALPHA_FOREX must exist in chad/types.py.
    - target_universe must be deterministic and side-effect free.
    """
    return StrategyConfig(
        name=StrategyName.ALPHA_FOREX,
        enabled=True,
        target_universe=list(FOREX_UNIVERSE_DEFAULT),
    )


# ---------------------------------------------------------------------------
# Context Extractors (defensive)
# ---------------------------------------------------------------------------

def _safe_iter(obj: Any) -> Iterable[Any]:
    """
    Convert arbitrary objects into a safe iterable.

    If `obj` is None or a scalar, returns an empty iterable or a single-element
    iterable as appropriate. This keeps AlphaForex resilient to context shape
    changes.
    """
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return obj
    return [obj]


def _extract_prices(ctx: Any, universe: List[str]) -> dict:
    """
    Extract spot prices for FX pairs from context.

    Expected:
        ctx.prices[symbol] -> float

    Returns
    -------
    dict: {symbol: price}
    """
    try:
        prices = getattr(ctx, "prices", {}) or {}
        return {s: float(prices.get(s)) for s in universe if prices.get(s) is not None}
    except Exception:
        return {}


def _extract_vol(ctx: Any, universe: List[str]) -> dict:
    """
    Extract volatility estimates if present.

    Expected:
        ctx.volatility[symbol] -> float

    Returns an empty dict if unavailable or malformed.
    """
    try:
        vol = getattr(ctx, "volatility", {}) or {}
        return {s: float(vol.get(s)) for s in universe if vol.get(s) is not None}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Strategy Handler (Phase-7 Safe Mode)
# ---------------------------------------------------------------------------

def alpha_forex_handler(ctx: Any, params: AlphaForexParams) -> List:
    """
    AlphaForex handler (Phase-7, DRY_RUN, zero-orders).

    Behaviour
    ---------
    - If disabled → returns [].
    - Extracts price + volatility context defensively.
    - Optionally applies basic volatility sanity checks.
    - Emits NO signals in this baseline version (production-safe).
    - Always returns [], guaranteeing it cannot affect trading today.

    This allows AlphaForex to be registered in the StrategyEngine immediately,
    with zero impact on LiveGate, SCR, or execution until you introduce actual
    signal logic.

    Parameters
    ----------
    ctx : Any
        StrategyContext built upstream.
    params : AlphaForexParams
        Configuration for universe and basic thresholds.

    Returns
    -------
    List
        No StrategySignal objects yet — by design.
    """

    if not params.enabled:
        return []

    universe = params.actual_universe()

    prices = _extract_prices(ctx, universe)
    vol = _extract_vol(ctx, universe)

    # Phase-7: only sanity checks, no trading logic.
    for symbol in universe:
        _ = prices.get(symbol)  # touch for completeness; currently unused
        v = vol.get(symbol)
        if v is not None and v > params.max_volatility:
            # In a future version, this is where you'd suppress or reshape
            # signals for overly volatile conditions.
            continue

    # Phase-7 guarantee: **no signals emitted**
    return []

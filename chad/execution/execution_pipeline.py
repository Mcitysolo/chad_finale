from __future__ import annotations

"""
chad/execution/execution_pipeline.py

Phase-4 Execution Pipeline (Broker-Agnostic Planning Layer)

This module bridges the *logical* decision layer and the concrete broker
executors (IBKR, Kraken, etc.) by turning RoutedSignal objects + prices into
a normalized set of "planned orders" (ExecutionPlan). It deliberately does
NOT talk to brokers or enforce caps; that work is delegated to the existing
IBKRExecutor / KrakenExecutor modules and the dynamic risk allocator.

Primary responsibilities
------------------------
* Take a batch of RoutedSignal objects (already:
    - produced by DecisionPipeline,
    - policy-filtered,
    - per-symbol netted).
* Look up prices and compute notional estimates.
* Produce a list of PlannedOrder instances that contain:
    - strategy label(s),
    - symbol,
    - side,
    - size,
    - asset_class,
    - unit price,
    - notional.

This keeps execution planning:
* deterministic,
* easily testable,
* decoupled from any specific broker API.

Downstream mapping (Phase 4+)
-----------------------------
* IBKRExecutor consumes StrategyTradeIntent (see chad/execution/ibkr_executor.py).
* KrakenExecutor consumes StrategyTradeIntent (see chad/execution/kraken_executor.py).

This module also provides a helper to convert an ExecutionPlan into a set of
IBKR StrategyTradeIntent objects for equity/ETF symbols. This is still pure
transformation: no sockets, no network I/O.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

from chad.execution.ibkr_executor import StrategyTradeIntent as IBKRStrategyTradeIntent
from chad.types import AssetClass, SignalSide, StrategyName
from chad.utils.signal_router import RoutedSignal


# ---------------------------------------------------------------------------
# Broker-agnostic planning models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedOrder:
    """
    Normalized, broker-agnostic representation of a planned trade.

    Fields
    ------
    symbol:
        Ticker / instrument identifier, e.g. "SPY", "AAPL", "QQQ".

    side:
        BUY or SELL as a SignalSide enum.

    size:
        Net size in natural units (e.g. shares or contracts) as produced by
        the SignalRouter. Positive size for BUY, positive size for SELL
        (direction is carried by `side`).

    asset_class:
        AssetClass enum (e.g. EQUITY, ETF, CRYPTO, FOREX).

    price:
        Latest known unit price from the pricing layer (ContextBuilder), used
        for notional estimation and risk comparisons.

    notional:
        size * price. Always >= 0.0. If price is missing or invalid, the
        RoutedSignal is excluded from the plan.

    primary_strategy:
        The first StrategyName that contributed to this RoutedSignal. This
        is useful for mapping into per-strategy caps (e.g. dynamic_caps.json).

    contributing_strategies:
        All StrategyName values that contributed to this RoutedSignal.
    """

    symbol: str
    side: SignalSide
    size: float
    asset_class: AssetClass
    price: float
    notional: float
    primary_strategy: StrategyName
    contributing_strategies: Sequence[StrategyName]


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Container for a batch of planned orders.

    Fields
    ------
    orders:
        List of PlannedOrder instances.

    Notes
    -----
    This object is deliberately simple; it can be enriched later with metadata
    (e.g. snapshot timestamps, risk context) without affecting the basic
    interface.
    """

    orders: List[PlannedOrder]

    @property
    def total_notional(self) -> float:
        """Total notional across all planned orders."""
        return float(sum(o.notional for o in self.orders))


def build_execution_plan(
    routed_signals: Iterable[RoutedSignal],
    prices: Mapping[str, float],
) -> ExecutionPlan:
    """
    Build an ExecutionPlan from routed signals and current prices.

    Behaviour
    ---------
    * For each RoutedSignal:
        - Look up price in `prices` by symbol.
        - If price is missing or <= 0, skip the signal (cannot compute notional).
        - Compute notional = abs(net_size) * price.
        - Use the first contributing strategy as `primary_strategy`.
        - Preserve the full tuple of contributing strategies.

    * Returns:
        ExecutionPlan(orders=[...]) with one PlannedOrder per RoutedSignal
        that has a valid price.

    This function is *pure* and side-effect free. It does not:
        - talk to brokers,
        - mutate dynamic caps,
        - enforce any caps,
        - write logs or files.
    """

    orders: List[PlannedOrder] = []

    for rs in routed_signals:
        symbol = rs.symbol
        price = prices.get(symbol)

        if price is None or price <= 0.0:
            # Skip symbols with missing/invalid prices; they should have been
            # filtered earlier by policy, but we keep this guard as a final
            # safety net.
            continue

        # Net size is always positive; direction is encoded in `side`.
        size = float(rs.net_size)
        if size <= 0.0:
            # Nothing to do; keep plan free of zero-sized orders.
            continue

        notional = abs(size) * float(price)

        if not rs.source_strategies:
            # This should not normally happen; RoutedSignal is expected to
            # carry at least one contributing StrategyName. We simply skip
            # such entries rather than guessing.
            continue

        primary = rs.source_strategies[0]
        contributing = tuple(rs.source_strategies)

        order = PlannedOrder(
            symbol=symbol,
            side=rs.side,
            size=size,
            asset_class=rs.asset_class,
            price=float(price),
            notional=float(notional),
            primary_strategy=primary,
            contributing_strategies=contributing,
        )
        orders.append(order)

    return ExecutionPlan(orders=orders)


# ---------------------------------------------------------------------------
# IBKR intent builder (Phase 4 mapping layer)
# ---------------------------------------------------------------------------


def build_ibkr_intents_from_plan(
    plan: ExecutionPlan,
    *,
    default_sec_type: str = "STK",
    default_exchange: str = "SMART",
    default_currency: str = "USD",
) -> List[IBKRStrategyTradeIntent]:
    """
    Map an ExecutionPlan into a list of IBKR StrategyTradeIntent objects.

    Rules
    -----
    * Only EQUITY and ETF asset classes are mapped; other classes are skipped.
    * Strategy name passed to IBKR is the lowercase StrategyName value
      (e.g. 'beta', 'alpha').
    * Side is converted from SignalSide enum to 'BUY' / 'SELL'.
    * Quantity comes from PlannedOrder.size.
    * notional_estimate is copied from PlannedOrder.notional.
    * Orders are MARKET by default (MKT) with no limit price.

    Parameters
    ----------
    plan:
        ExecutionPlan to convert.

    default_sec_type:
        Default IBKR secType for equity-like instruments, usually "STK".

    default_exchange:
        Default IBKR exchange/routing, usually "SMART".

    default_currency:
        Default trade currency, e.g. "USD". This should match your IBKR
        account base currency for notional comparisons.

    Returns
    -------
    List[IBKRStrategyTradeIntent]
        One intent per PlannedOrder that is eligible for IBKR.
    """

    intents: List[IBKRStrategyTradeIntent] = []

    for order in plan.orders:
        if order.asset_class not in (AssetClass.EQUITY, AssetClass.ETF):
            # Let other executors (e.g. Kraken, Forex) handle non-equity assets.
            continue

        strategy_name = order.primary_strategy.value  # e.g. "beta"
        side = "BUY" if order.side is SignalSide.BUY else "SELL"

        intent = IBKRStrategyTradeIntent(
            strategy=strategy_name,
            symbol=order.symbol,
            sec_type=default_sec_type,
            exchange=default_exchange,
            currency=default_currency,
            side=side,
            order_type="MKT",
            quantity=order.size,
            notional_estimate=order.notional,
            limit_price=None,
        )
        intents.append(intent)

    return intents

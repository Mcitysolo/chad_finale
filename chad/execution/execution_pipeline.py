from __future__ import annotations

"""
chad/execution/execution_pipeline.py

Professional-grade broker-agnostic execution planning layer for CHAD.

What this module does
---------------------
1. Nets routed signals by symbol.
2. Validates prices and sizes.
3. Produces a deterministic ExecutionPlan of PlannedOrder objects.
4. Converts eligible PlannedOrder objects into IBKR StrategyTradeIntent objects.
5. Supports multi-asset routing for:
   - EQUITY
   - ETF
   - FUTURES
   - FOREX
6. Fails closed for unsupported / malformed instruments.

Design goals
------------
- Pure transformation layer: no network I/O, no broker sockets.
- Deterministic, testable, auditable behavior.
- Strong validation and explicit rejection reasons.
- Extensible instrument resolution via strategy/factory style registry.
- Backwards-compatible public entry points:
    * build_execution_plan(...)
    * build_ibkr_intents_from_plan(...)

Notes
-----
- This module assumes:
    * RoutedSignal objects expose:
        symbol, side, net_size, asset_class, source_strategies
    * IBKRStrategyTradeIntent supports:
        strategy, symbol, sec_type, exchange, currency, side,
        order_type, quantity, notional_estimate, limit_price
- Futures support requires broker-side contract construction elsewhere
  (expiry / localSymbol / multiplier). This module correctly routes FUT intents
  but does not build ib_insync Contract objects itself.
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from enum import Enum
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chad.analytics.signal_confidence import (
    compute_confidence,
    normalize_signal_strength,
    regime_quality_from_state,
    sizing_multiplier,
)
from chad.execution.ibkr_executor import StrategyTradeIntent as IBKRStrategyTradeIntent
from chad.execution.intent_schema import utc_now_iso
from chad.execution.routing_gates import run_all_gates
from chad.types import AssetClass, SignalSide, StrategyName
from chad.utils.signal_router import RoutedSignal

LOG = logging.getLogger(__name__)


# ============================================================================
# Constants / tuning
# ============================================================================

_EPSILON = Decimal("0.000000000001")
_DEFAULT_ORDER_TYPE = "MKT"
_DEFAULT_FOREX_CURRENCY = "USD"
_DEFAULT_FUTURES_CURRENCY = "USD"

# Whole-unit instrument types on standard IBKR paths.
_WHOLE_UNIT_SEC_TYPES = frozenset({"STK", "FUT", "OPT"})

# Phase-8 Session 2: routing gate configuration. All four gates
# (data_freshness/A4, stale_intent/E2, too_late_to_chase/E5, net_ev/R7) use
# these defaults unless overridden by CHAD_ROUTING_GATES_* env vars. Zero-
# config behavior: gates are active with sensible defaults.
_routing_gates_config: Dict[str, float] = {
    "max_bar_age_seconds": 300,
    "price_tolerance_pct": 0.005,
    "degraded_ttl_seconds": 60,
    "estimated_commission": 1.0,
    "estimated_spread": 0.0,
    "min_edge": 0.0,
}

# Reason codes are intentionally short and machine-friendly.
_REASON_UNSUPPORTED_ASSET_CLASS = "unsupported_asset_class"
_REASON_MISSING_PRICE = "missing_price"
_REASON_NON_POSITIVE_PRICE = "non_positive_price"
_REASON_NON_POSITIVE_SIZE = "non_positive_size"
_REASON_MISSING_STRATEGY = "missing_strategy"
_REASON_INVALID_SYMBOL = "invalid_symbol"
_REASON_INVALID_FOREX_SYMBOL = "invalid_forex_symbol"
_REASON_INVALID_FUTURES_SYMBOL = "invalid_futures_symbol"
_REASON_INVALID_QUANTITY = "invalid_quantity"


# ============================================================================
# Data models
# ============================================================================


class PlanRejectionSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class PlanRejection:
    symbol: str
    reason: str
    severity: PlanRejectionSeverity = PlanRejectionSeverity.WARNING
    detail: str = ""


@dataclass(frozen=True)
class PlannedOrder:
    """
    Broker-agnostic normalized order.
    """

    symbol: str
    side: SignalSide
    size: float
    asset_class: AssetClass
    price: float
    notional: float
    primary_strategy: StrategyName
    contributing_strategies: Sequence[StrategyName]
    metadata: Mapping[str, object] = field(default_factory=dict)
    # Phase-8 Session 3: preserve signal-quality inputs through planning so
    # the intent builder can compute confidence and apply the sizing
    # multiplier. Defaults keep existing construction paths compatible.
    confidence: float = 0.5
    signal_strength: float = 0.0
    regime_state: str = "unknown"
    expected_pnl: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Deterministic planning result.

    orders:
        Accepted normalized planned orders.

    rejections:
        Explicitly rejected routed signals / netted symbols, useful for audit.

    Notes:
        This preserves enough state to explain why something did not convert
        into an executable broker intent.
    """

    orders: List[PlannedOrder]
    rejections: List[PlanRejection] = field(default_factory=list)

    @property
    def total_notional(self) -> float:
        return float(sum(o.notional for o in self.orders))

    @property
    def futures_orders_count(self) -> int:
        return sum(1 for o in self.orders if o.asset_class == AssetClass.FUTURES)

    @property
    def equity_like_orders_count(self) -> int:
        return sum(1 for o in self.orders if o.asset_class in (AssetClass.EQUITY, AssetClass.ETF))

    @property
    def forex_orders_count(self) -> int:
        return sum(1 for o in self.orders if o.asset_class == AssetClass.FOREX)

    @property
    def symbols(self) -> List[str]:
        return [o.symbol for o in self.orders]


@dataclass(frozen=True)
class IBKRInstrumentSpec:
    """
    Canonical broker intent mapping for an instrument family.
    """

    sec_type: str
    exchange: str
    currency: str
    quantity_step: Decimal = Decimal("1")
    whole_units: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)


# ============================================================================
# Helpers: numeric hygiene
# ============================================================================


def _to_decimal(value: object, *, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def _is_effectively_zero(value: Decimal) -> bool:
    return abs(value) <= _EPSILON


def _quantize_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("step must be positive")
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return units * step


def _normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


# ============================================================================
# Instrument resolution
# ============================================================================


@lru_cache(maxsize=256)
def _futures_spec_registry() -> Dict[str, IBKRInstrumentSpec]:
    """
    Minimal futures routing registry for currently observed CHAD futures symbols.

    These are routing defaults only.
    Contract expiry / localSymbol / multiplier handling must still be done in
    the broker adapter / contract builder layer.
    """
    return {
        "MES": IBKRInstrumentSpec(
            sec_type="FUT",
            exchange="CME",
            currency="USD",
            quantity_step=Decimal("1"),
            whole_units=True,
            metadata={"family": "equity_index", "underlier": "ES", "micro": True},
        ),
        "MNQ": IBKRInstrumentSpec(
            sec_type="FUT",
            exchange="CME",
            currency="USD",
            quantity_step=Decimal("1"),
            whole_units=True,
            metadata={"family": "equity_index", "underlier": "NQ", "micro": True},
        ),
        "MCL": IBKRInstrumentSpec(
            sec_type="FUT",
            exchange="NYMEX",
            currency="USD",
            quantity_step=Decimal("1"),
            whole_units=True,
            metadata={"family": "energy", "underlier": "CL", "micro": True},
        ),
        "MGC": IBKRInstrumentSpec(
            sec_type="FUT",
            exchange="COMEX",
            currency="USD",
            quantity_step=Decimal("1"),
            whole_units=True,
            metadata={"family": "metals", "underlier": "GC", "micro": True},
        ),
    }


def _resolve_equity_like_spec(
    asset_class: AssetClass,
    *,
    default_exchange: str,
    default_currency: str,
    default_sec_type: str,
) -> IBKRInstrumentSpec:
    sec_type = "STK" if asset_class in (AssetClass.EQUITY, AssetClass.ETF) else default_sec_type
    return IBKRInstrumentSpec(
        sec_type=sec_type,
        exchange=default_exchange,
        currency=default_currency,
        quantity_step=Decimal("1"),
        whole_units=True,
        metadata={"asset_class": asset_class.value},
    )


def _resolve_forex_spec(symbol: str) -> IBKRInstrumentSpec:
    # Accept either EURUSD or EUR-USD style.
    compact = symbol.replace("-", "").replace("/", "")
    if len(compact) != 6 or not compact.isalpha():
        raise ValueError(_REASON_INVALID_FOREX_SYMBOL)

    base = compact[:3]
    quote = compact[3:]

    return IBKRInstrumentSpec(
        sec_type="CASH",
        exchange="IDEALPRO",
        currency=quote,
        quantity_step=Decimal("0.0001"),
        whole_units=False,
        metadata={"base_currency": base, "quote_currency": quote},
    )


def _resolve_futures_spec(symbol: str) -> IBKRInstrumentSpec:
    spec = _futures_spec_registry().get(symbol)
    if spec is None:
        raise ValueError(_REASON_INVALID_FUTURES_SYMBOL)
    return spec


def resolve_ibkr_instrument_spec(
    *,
    symbol: str,
    asset_class: AssetClass,
    default_sec_type: str,
    default_exchange: str,
    default_currency: str,
) -> IBKRInstrumentSpec:
    """
    Factory-style resolver for IBKR instrument mapping.
    """
    if asset_class in (AssetClass.EQUITY, AssetClass.ETF):
        return _resolve_equity_like_spec(
            asset_class,
            default_exchange=default_exchange,
            default_currency=default_currency,
            default_sec_type=default_sec_type,
        )

    if asset_class == AssetClass.FUTURES:
        return _resolve_futures_spec(symbol)

    if asset_class == AssetClass.FOREX:
        return _resolve_forex_spec(symbol)

    raise ValueError(_REASON_UNSUPPORTED_ASSET_CLASS)


# ============================================================================
# Plan building
# ============================================================================


@dataclass(frozen=True)
class _NettedBucket:
    symbol: str
    net_signed_size: Decimal
    buy_rs: Optional[RoutedSignal]
    sell_rs: Optional[RoutedSignal]


def _net_routed_signals(routed_signals: Iterable[RoutedSignal]) -> Dict[str, _NettedBucket]:
    buckets: Dict[str, Dict[str, object]] = {}

    for rs in routed_signals:
        symbol = _normalize_symbol(getattr(rs, "symbol", ""))
        if not symbol:
            continue

        raw_size = _to_decimal(getattr(rs, "net_size", None), default=Decimal("0"))
        if raw_size is None or raw_size <= 0:
            continue

        side = getattr(rs, "side", None)
        if side not in (SignalSide.BUY, SignalSide.SELL):
            continue

        signed_size = raw_size if side is SignalSide.BUY else -raw_size

        if symbol not in buckets:
            buckets[symbol] = {
                "net_signed_size": Decimal("0"),
                "buy_rs": None,
                "sell_rs": None,
            }

        bucket = buckets[symbol]
        bucket["net_signed_size"] = Decimal(bucket["net_signed_size"]) + signed_size

        if side is SignalSide.BUY:
            prev = bucket["buy_rs"]
            prev_size = _to_decimal(getattr(prev, "net_size", None), default=Decimal("0")) if prev else Decimal("0")
            if prev is None or raw_size > prev_size:
                bucket["buy_rs"] = rs
        else:
            prev = bucket["sell_rs"]
            prev_size = _to_decimal(getattr(prev, "net_size", None), default=Decimal("0")) if prev else Decimal("0")
            if prev is None or raw_size > prev_size:
                bucket["sell_rs"] = rs

    out: Dict[str, _NettedBucket] = {}
    for symbol in sorted(buckets.keys()):
        b = buckets[symbol]
        out[symbol] = _NettedBucket(
            symbol=symbol,
            net_signed_size=Decimal(b["net_signed_size"]),
            buy_rs=b["buy_rs"],
            sell_rs=b["sell_rs"],
        )
    return out


def build_execution_plan(
    routed_signals: Iterable[RoutedSignal],
    prices: Mapping[str, float],
) -> ExecutionPlan:
    """
    Pure planning function.

    - Nets opposing same-symbol orders.
    - Drops symbols with missing or invalid prices.
    - Preserves strategy provenance.
    - Returns both accepted orders and explicit rejections.
    """
    buckets = _net_routed_signals(routed_signals)
    orders: List[PlannedOrder] = []
    rejections: List[PlanRejection] = []

    for symbol in sorted(buckets.keys()):
        bucket = buckets[symbol]

        if _is_effectively_zero(bucket.net_signed_size):
            rejections.append(
                PlanRejection(
                    symbol=symbol,
                    reason="zero_net_size",
                    severity=PlanRejectionSeverity.INFO,
                    detail="net_signed_size_effectively_zero",
                )
            )
            continue

        side = SignalSide.BUY if bucket.net_signed_size > 0 else SignalSide.SELL
        size_dec = abs(bucket.net_signed_size)

        survivor = bucket.buy_rs if side is SignalSide.BUY else bucket.sell_rs
        if survivor is None:
            rejections.append(
                PlanRejection(symbol=symbol, reason=_REASON_NON_POSITIVE_SIZE, detail="missing_survivor_signal")
            )
            continue

        asset_class = getattr(survivor, "asset_class", None)
        if not isinstance(asset_class, AssetClass):
            rejections.append(
                PlanRejection(symbol=symbol, reason=_REASON_UNSUPPORTED_ASSET_CLASS, detail=f"asset_class={asset_class!r}")
            )
            continue

        price = prices.get(symbol)
        if price is None:
            rejections.append(
                PlanRejection(symbol=symbol, reason=_REASON_MISSING_PRICE, detail="no_price_in_mapping")
            )
            continue

        price_dec = _to_decimal(price)
        if price_dec is None or price_dec <= 0:
            rejections.append(
                PlanRejection(symbol=symbol, reason=_REASON_NON_POSITIVE_PRICE, detail=f"price={price!r}")
            )
            continue

        strategies = tuple(getattr(survivor, "source_strategies", ()) or ())
        if not strategies:
            rejections.append(
                PlanRejection(symbol=symbol, reason=_REASON_MISSING_STRATEGY, detail="empty_source_strategies")
            )
            continue

        primary_strategy = strategies[0]
        if not isinstance(primary_strategy, StrategyName):
            rejections.append(
                PlanRejection(symbol=symbol, reason=_REASON_MISSING_STRATEGY, detail=f"primary_strategy={primary_strategy!r}")
            )
            continue

        notional_dec = size_dec * price_dec

        metadata = {
            "netted": True,
            "survivor_side": side.value,
            "raw_asset_class": asset_class.value,
            "strategy_count": len(strategies),
        }

        # Preserve signal-quality inputs from the survivor signal so the
        # intent builder can compute confidence and apply S3 sizing.
        survivor_confidence = _safe_float_attr(survivor, "confidence", 0.5)
        survivor_signal_strength = _safe_float_attr(survivor, "signal_strength", 0.0)
        survivor_expected_pnl = _safe_float_attr(survivor, "expected_pnl", 0.0)
        survivor_regime_state = str(getattr(survivor, "regime_state", "unknown") or "unknown")
        survivor_reason = str(getattr(survivor, "reason", "") or "")

        orders.append(
            PlannedOrder(
                symbol=symbol,
                side=side,
                size=float(size_dec),
                asset_class=asset_class,
                price=float(price_dec),
                notional=float(notional_dec),
                primary_strategy=primary_strategy,
                contributing_strategies=strategies,
                metadata=metadata,
                confidence=survivor_confidence,
                signal_strength=survivor_signal_strength,
                regime_state=survivor_regime_state,
                expected_pnl=survivor_expected_pnl,
                reason=survivor_reason,
            )
        )

    return ExecutionPlan(orders=orders, rejections=rejections)


def _safe_float_attr(obj: object, name: str, default: float) -> float:
    """Read a float-like attribute with a fallback — used for optional
    signal metadata that older RoutedSignal builds may not carry."""
    value = getattr(obj, name, None)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# ============================================================================
# IBKR intent mapping
# ============================================================================


def _normalize_quantity_for_spec(quantity: float, spec: IBKRInstrumentSpec) -> float:
    q = _to_decimal(quantity)
    if q is None or q <= 0:
        raise ValueError(_REASON_INVALID_QUANTITY)

    if spec.whole_units or spec.sec_type in _WHOLE_UNIT_SEC_TYPES:
        q = _quantize_down(q, Decimal("1"))
    else:
        q = _quantize_down(q, spec.quantity_step)

    if q <= 0:
        raise ValueError(_REASON_INVALID_QUANTITY)

    return float(q)


def build_ibkr_intents_from_plan(
    plan: ExecutionPlan,
    *,
    default_sec_type: str = "STK",
    default_exchange: str = "SMART",
    default_currency: str = "USD",
) -> List[IBKRStrategyTradeIntent]:
    """
    Convert an ExecutionPlan into IBKR StrategyTradeIntent objects.

    Supported asset classes
    -----------------------
    - EQUITY -> STK / SMART / USD
    - ETF    -> STK / SMART / USD
    - FUTURES -> FUT / exchange per registry / USD
    - FOREX  -> CASH / IDEALPRO / quote currency

    Important
    ---------
    This function routes futures correctly as FUT intents, but broker-side
    contract construction still needs expiry/local-symbol logic elsewhere.
    """
    intents: List[IBKRStrategyTradeIntent] = []

    for order in plan.orders:
        try:
            spec = resolve_ibkr_instrument_spec(
                symbol=order.symbol,
                asset_class=order.asset_class,
                default_sec_type=default_sec_type,
                default_exchange=default_exchange,
                default_currency=default_currency,
            )
        except ValueError:
            # Fail closed. Unsupported instruments stay out of broker intents.
            continue

        side = "BUY" if order.side is SignalSide.BUY else "SELL"
        strategy_name = order.primary_strategy.value

        # Phase-8 Session 3 (S3): compute confidence and apply sizing floor.
        # Attenuation is OPT-IN: only when the strategy has populated
        # signal_strength do we recompute confidence and scale size. Legacy
        # strategies that emit neither signal_strength nor an explicit
        # confidence still ship full-size so this change is backward
        # compatible with every pre-Session-3 construction site.
        raw_strength = float(order.signal_strength or 0.0)
        if raw_strength != 0.0:
            confidence = compute_confidence(
                signal_strength=normalize_signal_strength(raw_strength, method="tanh"),
                regime_quality=regime_quality_from_state(order.regime_state),
                liquidity_quality=1.0,
            )
            size_mult = sizing_multiplier(confidence)
        else:
            confidence = float(order.confidence or 0.5)
            size_mult = 1.0

        attenuated_size = float(order.size) * size_mult
        try:
            quantity = _normalize_quantity_for_spec(attenuated_size, spec)
        except ValueError as _qty_err:
            LOG.warning(
                "intent_skipped_invalid_quantity symbol=%s size=%s reason=%s",
                order.symbol,
                attenuated_size,
                _qty_err,
            )
            continue

        # Symbol normalization:
        # - equities/etfs/futures pass through unchanged
        # - forex becomes compact "EURUSD" for IBKR CASH path
        broker_symbol = order.symbol
        if order.asset_class == AssetClass.FOREX:
            broker_symbol = order.symbol.replace("-", "").replace("/", "").upper()

        # Scale the notional to reflect the attenuated quantity so downstream
        # cost/PnL calculations stay consistent with what was actually sent.
        attenuated_notional = float(order.notional) * size_mult

        intent = IBKRStrategyTradeIntent(
            strategy=strategy_name,
            symbol=broker_symbol,
            sec_type=spec.sec_type,
            exchange=spec.exchange,
            currency=spec.currency,
            side=side,
            order_type=_DEFAULT_ORDER_TYPE,
            quantity=quantity,
            notional_estimate=attenuated_notional,
            limit_price=None,
            confidence=confidence,
            entry_reason=str(order.reason or ""),
            regime_state=str(order.regime_state or "unknown"),
            expected_pnl=float(order.expected_pnl or 0.0),
            created_at=utc_now_iso(),
            expected_price=float(order.price),
            signal_strength=raw_strength,
        )

        # Routing gates (Phase-8 Session 2: A4/E2/E5/R7). Reject intents that
        # fail any pre-OMS validation. current_price equals creation price at
        # build time, so E5 operates in its degraded time-based mode here.
        passed, _reason = run_all_gates(
            intent=intent,
            bar_timestamp=None,
            current_price=float(order.price),
            config=_routing_gates_config,
        )
        if not passed:
            continue

        intents.append(intent)

    return intents


# ============================================================================
# Optional audit helpers
# ============================================================================


def summarize_execution_plan(plan: ExecutionPlan) -> Dict[str, object]:
    """
    Lightweight deterministic summary for logs/tests.
    """
    by_asset: Dict[str, int] = {}
    for order in plan.orders:
        key = order.asset_class.value
        by_asset[key] = by_asset.get(key, 0) + 1

    return {
        "orders_count": len(plan.orders),
        "total_notional": round(plan.total_notional, 8),
        "futures_orders_count": plan.futures_orders_count,
        "equity_like_orders_count": plan.equity_like_orders_count,
        "forex_orders_count": plan.forex_orders_count,
        "rejections_count": len(plan.rejections),
        "orders_by_asset_class": dict(sorted(by_asset.items())),
        "symbols": plan.symbols,
    }


# ============================================================================
# Asset-class router (IBKR vs Kraken)
# ============================================================================


def split_signals_by_asset_class(
    routed_signals: Iterable[RoutedSignal],
) -> Tuple[List[RoutedSignal], List[RoutedSignal]]:
    """
    Split a stream of RoutedSignal into (ibkr_signals, kraken_signals).

    - CRYPTO asset class -> kraken_signals
    - Everything else (or missing/unknown) -> ibkr_signals

    The existing IBKR pipeline is unchanged; only CRYPTO signals are
    diverted off the IBKR path. Order is preserved within each bucket.
    """
    ibkr_signals: List[RoutedSignal] = []
    kraken_signals: List[RoutedSignal] = []
    for rs in routed_signals:
        ac = getattr(rs, "asset_class", None)
        if ac == AssetClass.CRYPTO:
            kraken_signals.append(rs)
        else:
            ibkr_signals.append(rs)
    return ibkr_signals, kraken_signals


# ============================================================================
# Kraken intent builder
# ============================================================================

# Kraken-side minimum order sizes (base currency). Conservative defaults that
# satisfy Kraken's published minima as of 2026.
_KRAKEN_MIN_VOLUMES: Dict[str, Decimal] = {
    "XBT/USD": Decimal("0.0001"),
    "ETH/USD": Decimal("0.001"),
    "SOL/USD": Decimal("0.05"),
    "XBT/CAD": Decimal("0.0001"),
    "ETH/CAD": Decimal("0.001"),
}

# CHAD canonical symbol -> Kraken pair
_KRAKEN_SYMBOL_MAP: Dict[str, str] = {
    # USD-quoted
    "BTC-USD": "XBT/USD",
    "BTCUSD": "XBT/USD",
    "XBT-USD": "XBT/USD",
    "XBTUSD": "XBT/USD",
    "ETH-USD": "ETH/USD",
    "ETHUSD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "SOLUSD": "SOL/USD",
    # CAD-quoted alternates (used when USD buying power is empty and the
    # Kraken account holds ZCAD)
    "BTC-CAD": "XBT/CAD",
    "BTCCAD": "XBT/CAD",
    "XBT-CAD": "XBT/CAD",
    "XBTCAD": "XBT/CAD",
    "ETH-CAD": "ETH/CAD",
    "ETHCAD": "ETH/CAD",
}


def normalize_kraken_pair(symbol: str) -> Optional[str]:
    """
    Map a CHAD canonical crypto symbol (e.g. BTC-USD) to a Kraken pair
    (e.g. XBT/USD). Returns None for unsupported symbols.
    """
    if not symbol:
        return None
    key = _normalize_symbol(symbol)
    return _KRAKEN_SYMBOL_MAP.get(key)


def _build_kraken_intent_from_routed_signal(
    signal: RoutedSignal,
    current_price: float,
    *,
    dynamic_cap_for_crypto: Optional[float] = None,
):
    """
    Convert a RoutedSignal (CRYPTO asset class) into a KrakenStrategyTradeIntent.

    - symbol normalization: BTC-USD -> XBT/USD, ETH-USD -> ETH/USD, SOL-USD -> SOL/USD
    - volume = capped_notional / current_price (base currency quantity)
    - ordertype = "market"
    - notional_estimate = min(signal.notional, dynamic_cap_for_crypto)
    - validates volume against per-pair Kraken minimum

    Returns None on any rejection (unsupported symbol, bad price, below min size).
    """
    # Local import: keep execution_pipeline import-light when Kraken stack
    # is not present in some test contexts.
    from chad.execution.kraken_executor import StrategyTradeIntent as KrakenStrategyTradeIntent

    symbol = _normalize_symbol(getattr(signal, "symbol", ""))
    pair = normalize_kraken_pair(symbol)
    if pair is None:
        return None

    price_dec = _to_decimal(current_price)
    if price_dec is None or price_dec <= 0:
        return None

    side_attr = getattr(signal, "side", None)
    if side_attr not in (SignalSide.BUY, SignalSide.SELL):
        return None
    side = "buy" if side_attr is SignalSide.BUY else "sell"

    raw_notional = _to_decimal(getattr(signal, "notional", None), default=Decimal("0"))
    if raw_notional is None or raw_notional <= 0:
        size_dec = _to_decimal(getattr(signal, "net_size", None), default=Decimal("0"))
        if size_dec is None or size_dec <= 0:
            return None
        raw_notional = size_dec * price_dec

    capped_notional = raw_notional
    if dynamic_cap_for_crypto is not None:
        cap_dec = _to_decimal(dynamic_cap_for_crypto)
        if cap_dec is not None and cap_dec > 0 and cap_dec < capped_notional:
            capped_notional = cap_dec

    if capped_notional <= 0:
        return None

    # Phase-8 Session 3 (S3): attenuate volume by the confidence-based sizing
    # multiplier, OPT-IN via non-zero signal_strength on the routed signal.
    # Strategies that did not populate signal_strength ship at legacy (full)
    # size — this preserves backward compatibility with pre-Session-3 callers.
    raw_strength = _safe_float_attr(signal, "signal_strength", 0.0)
    regime_state = str(getattr(signal, "regime_state", "unknown") or "unknown")
    if raw_strength != 0.0:
        confidence = compute_confidence(
            signal_strength=normalize_signal_strength(raw_strength, method="tanh"),
            regime_quality=regime_quality_from_state(regime_state),
            liquidity_quality=1.0,
        )
        size_mult = Decimal(str(sizing_multiplier(confidence)))
    else:
        confidence = float(getattr(signal, "confidence", 0.5) or 0.5)
        size_mult = Decimal("1")

    volume_dec = (capped_notional / price_dec) * size_mult
    attenuated_notional = capped_notional * size_mult

    min_vol = _KRAKEN_MIN_VOLUMES.get(pair, Decimal("0.0001"))
    if volume_dec < min_vol:
        return None

    strategies = tuple(getattr(signal, "source_strategies", ()) or ())
    if strategies and isinstance(strategies[0], StrategyName):
        primary_strategy_name = strategies[0].value
    elif strategies:
        primary_strategy_name = str(strategies[0])
    else:
        primary_strategy_name = "alpha_crypto"

    return KrakenStrategyTradeIntent(
        strategy=primary_strategy_name,
        pair=pair,
        side=side,
        ordertype="market",
        volume=float(volume_dec),
        notional_estimate=float(attenuated_notional),
        price=None,
        confidence=confidence,
        entry_reason=str(getattr(signal, "reason", "") or ""),
        regime_state=regime_state,
        expected_pnl=float(getattr(signal, "expected_pnl", 0.0) or 0.0),
        created_at=utc_now_iso(),
        expected_price=float(current_price),
        signal_strength=raw_strength,
    )


def build_kraken_intents_from_routed_signals(
    routed_signals: Iterable[RoutedSignal],
    prices: Mapping[str, float],
    *,
    dynamic_cap_for_crypto: Optional[float] = None,
) -> List[object]:
    """
    Build KrakenStrategyTradeIntent objects for the CRYPTO subset of routed_signals.

    Prices are looked up by the original CHAD canonical symbol (e.g. BTC-USD).
    Signals that fail any validation step are silently dropped (fail-closed).
    """
    out: List[object] = []
    for rs in routed_signals:
        if getattr(rs, "asset_class", None) != AssetClass.CRYPTO:
            continue
        symbol = _normalize_symbol(getattr(rs, "symbol", ""))
        price = prices.get(symbol)
        if price is None:
            continue
        intent = _build_kraken_intent_from_routed_signal(
            rs, float(price), dynamic_cap_for_crypto=dynamic_cap_for_crypto
        )
        if intent is None:
            continue

        # Routing gates (Phase-8 Session 2: A4/E2/E5/R7). Same gate set as the
        # IBKR path — covers the Kraken execution lane before OMS submission.
        passed, _reason = run_all_gates(
            intent=intent,
            bar_timestamp=None,
            current_price=float(price),
            config=_routing_gates_config,
        )
        if not passed:
            continue

        out.append(intent)
    return out


__all__ = [
    "PlannedOrder",
    "ExecutionPlan",
    "PlanRejection",
    "PlanRejectionSeverity",
    "IBKRInstrumentSpec",
    "build_execution_plan",
    "build_ibkr_intents_from_plan",
    "resolve_ibkr_instrument_spec",
    "summarize_execution_plan",
    "split_signals_by_asset_class",
    "normalize_kraken_pair",
    "build_kraken_intents_from_routed_signals",
    "_build_kraken_intent_from_routed_signal",
]

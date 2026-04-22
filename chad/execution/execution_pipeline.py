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
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from chad.analytics.signal_confidence import (
    compute_confidence,
    normalize_signal_strength,
    regime_quality_from_state,
    sizing_multiplier,
)
from chad.analytics.timeframe_confirmation import (
    get_higher_tf_bias,
    timeframe_confidence_multiplier,
)
from chad.analytics.vote_collector import get_default_collector as _get_vote_collector
from chad.execution.ibkr_executor import StrategyTradeIntent as IBKRStrategyTradeIntent
from chad.execution.intent_schema import utc_now_iso
from chad.execution.order_type_selector import (
    compute_aggressive_limit_price,
    estimate_spread_pct,
    select_order_type,
)
from chad.execution.routing_gates import run_all_gates
from chad.risk.composite_size_cap import CompositeSizeCap
from chad.risk.correlation_monitor import CorrelationMonitor
from chad.risk.vol_adjusted_sizer import VolAdjustedSizer
from chad.types import AssetClass, SignalSide, StrategyName
from chad.utils.signal_router import RoutedSignal


# Phase-8 Session 5 (S1): default signal family per strategy identifier.
# Used when the routed signal does not carry an explicit signal_family tag
# (the common case today). Consumed only by the S1 vote collector — it
# does not change intent economics.
_STRATEGY_SIGNAL_FAMILY: Dict[str, str] = {
    "alpha": "momentum",
    "alpha_crypto": "momentum",
    "alpha_forex": "trend",
    "alpha_futures": "momentum",
    "alpha_intraday": "momentum",
    "alpha_options": "options",
    "beta": "trend",
    "delta": "trend",
    "delta_pairs": "mean_reversion",
    "gamma": "volatility",
    "gamma_futures": "trend",
    "gamma_reversion": "mean_reversion",
    "omega": "volatility",
    "omega_macro": "macro",
    "omega_momentum_options": "options",
    "omega_vol": "volatility",
}


def _resolve_signal_family(intent: object, fallback_strategy: str = "") -> str:
    """Return the signal family for an intent.

    Order of preference:
      1. explicit intent.signal_family (if a strategy has set it)
      2. _STRATEGY_SIGNAL_FAMILY[strategy name] derived map
      3. 'unknown'
    """
    explicit = getattr(intent, "signal_family", "") or ""
    explicit = str(explicit or "").lower()
    if explicit and explicit != "unknown":
        return explicit
    name = (getattr(intent, "strategy", "") or fallback_strategy or "").lower()
    return _STRATEGY_SIGNAL_FAMILY.get(name, "unknown")

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
# Session 8 note: max_bar_age_seconds is tuned for the current daily-bar
# data source (1d bars; 48h tolerates a weekend). Operators running
# against intraday bars should tighten this via config.
_routing_gates_config: Dict[str, float] = {
    "max_bar_age_seconds": 172800,
    "price_tolerance_pct": 0.005,
    "degraded_ttl_seconds": 60,
    "estimated_commission": 1.0,
    "estimated_spread": 0.0,
    "min_edge": 0.0,
}


# Phase-8 Session 7: lazy singletons for the sizing layer. Building from
# config on first use keeps tests light and avoids a hard dependency on the
# config file at import time.
_VOL_SIZER: Optional[VolAdjustedSizer] = None
_SIZE_CAP: Optional[CompositeSizeCap] = None
_CORR_MONITOR: Optional[CorrelationMonitor] = None
_EVENT_CALENDAR: Any = None


def _get_vol_sizer() -> VolAdjustedSizer:
    global _VOL_SIZER
    if _VOL_SIZER is None:
        _VOL_SIZER = VolAdjustedSizer.from_config()
    return _VOL_SIZER


def _get_size_cap() -> CompositeSizeCap:
    global _SIZE_CAP
    if _SIZE_CAP is None:
        _SIZE_CAP = CompositeSizeCap.from_config()
    return _SIZE_CAP


def _get_correlation_monitor() -> CorrelationMonitor:
    global _CORR_MONITOR
    if _CORR_MONITOR is None:
        _CORR_MONITOR = CorrelationMonitor.from_config()
    return _CORR_MONITOR


def _get_event_calendar() -> Any:
    """Return a cached EventCalendar; None when the module / config is unavailable."""
    global _EVENT_CALENDAR
    if _EVENT_CALENDAR is None:
        try:
            from chad.analytics.event_calendar import EventCalendar
            _EVENT_CALENDAR = EventCalendar()
        except Exception:  # noqa: BLE001
            _EVENT_CALENDAR = False  # sentinel — don't retry each call
    return _EVENT_CALENDAR or None


def _get_open_symbols() -> List[str]:
    """Best-effort read of currently open position symbols from position_guard.

    Returns [] when the guard file is missing or unreadable — the sizers
    treat an empty book as "no correlation pressure".
    """
    try:
        from chad.core.position_guard import _load_state
        state = _load_state()
    except Exception:  # noqa: BLE001
        return []
    out: List[str] = []
    for key, record in (state or {}).items():
        if not isinstance(record, dict) or not record.get("open"):
            continue
        sym = record.get("symbol")
        if not sym:
            # Fall back to "<strategy>|<symbol>" key decomposition.
            if isinstance(key, str) and "|" in key:
                sym = key.split("|", 1)[1]
        if sym:
            out.append(str(sym).upper())
    return out


def _get_account_equity() -> float:
    """Best-effort read of account equity from pnl_state.json."""
    try:
        import json as _json
        from pathlib import Path as _Path
        root = _Path(__file__).resolve().parents[2]
        path = root / "runtime" / "pnl_state.json"
        if not path.is_file():
            return 0.0
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    try:
        return float(data.get("account_equity") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _apply_sizing_layer(
    base_size: float,
    symbol: str,
    reference_price: float,
    open_symbols: List[str],
) -> int:
    """Apply R3 → R5 → R6 to a floating-point base_size and return integer shares.

    Never returns 0 for a positive base_size — the minimum is 1 share so the
    routing-gates decide rejection, not the sizer.
    """
    base_int = max(0, int(base_size))
    if base_int <= 0:
        return 0
    # R3: vol-adjust.
    r3_size = _get_vol_sizer().adjust(base_int, symbol)
    # R5: composite cap.
    r5_size = _get_size_cap().apply(
        vol_adjusted_size=r3_size,
        symbol=symbol,
        account_equity=_get_account_equity(),
        reference_price=reference_price,
    )
    # R6: correlation reducer.
    corr_mult = _get_correlation_monitor().get_size_multiplier(
        open_symbols=open_symbols,
        new_symbol=symbol,
    )
    final_size = max(1, int(r5_size * corr_mult))
    return final_size


def _load_latest_bar_for_symbol(symbol: str) -> Optional[Dict[str, object]]:
    """Best-effort load of the most recent 1d bar for spread estimation.

    Reads data/bars/1d/{SYMBOL}.json. Missing files or parse errors
    return None — callers treat this as "no bar data" and fall back to
    the default spread estimate. No I/O is performed for empty symbols.
    """
    if not symbol:
        return None
    try:
        import json as _json
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[2]
        path = root / "data" / "bars" / "1d" / f"{str(symbol).upper()}.json"
        if not path.is_file():
            return None
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    bars = data.get("bars")
    if not isinstance(bars, list) or not bars:
        return None
    last = bars[-1]
    if not isinstance(last, dict):
        return None
    return last


_STALE_BAR_WARN_DAYS: int = 5
_STALE_BAR_WARNED: set = set()


def load_latest_bar(symbol: str) -> Optional[Dict[str, object]]:
    """Public alias for _load_latest_bar_for_symbol.

    Phase-8 Session 8 (A4/E5 full threading): exposes the bar-loading
    helper for gate wiring. Never raises — returns None on any error.

    2026-04-22 Audit-O addition: emit a STALE_BARS warning (once per
    symbol per process) when the latest bar is older than
    ``_STALE_BAR_WARN_DAYS`` trading days. The warning is non-blocking —
    callers still receive the stale bar so shadow evaluation can
    proceed; the operator just gets a visible signal that data is
    drifting (first observed for IWM/TLT/VXX on 2026-04-22).
    """
    bar = _load_latest_bar_for_symbol(symbol)
    if bar is None:
        return None
    try:
        ts_raw = bar.get("ts_utc") or bar.get("timestamp") or bar.get("date")
        bar_dt = _parse_bar_timestamp(ts_raw) if ts_raw is not None else None
        if bar_dt is not None:
            age_days = (datetime.now(timezone.utc) - bar_dt).total_seconds() / 86400.0
            sym_key = str(symbol or "").upper()
            if age_days > _STALE_BAR_WARN_DAYS and sym_key not in _STALE_BAR_WARNED:
                LOG.warning(
                    "STALE_BARS symbol=%s age_days=%d latest_ts=%s",
                    sym_key, int(age_days), ts_raw,
                )
                _STALE_BAR_WARNED.add(sym_key)
    except Exception:  # noqa: BLE001 — warning path must never raise
        pass
    return bar


def _parse_bar_timestamp(ts_raw: object) -> Optional[datetime]:
    """Parse a bar's ts_utc field into a timezone-aware UTC datetime.

    Accepts:
      * YYYY-MM-DD (daily bars; midnight UTC assumed)
      * full ISO8601 with or without tz

    Returns None for anything unparseable. Used by callers that want to
    pass a concrete datetime into run_all_gates.bar_timestamp.
    """
    if not ts_raw:
        return None
    s = str(ts_raw).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _stop_bus_active() -> bool:
    """Phase-8 Session 4 (R2): pre-submit halt check.

    Reads runtime/stop_bus.json via the risk.stop_bus_state helper. A
    missing / malformed file returns False (fail-open) so a broken bus
    file cannot silently halt trading forever. Import is lazy so pure
    planning tests that never touch disk stay fast.
    """
    try:
        from chad.risk.stop_bus_state import is_stop_bus_active
        return bool(is_stop_bus_active())
    except Exception:
        return False

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
    # Phase-8 Session 6 (E4): passive/aggressive order hint propagated from
    # the routed signal. "normal" is the default; strategies with breakout
    # urgency set "high" to request a marketable limit.
    order_urgency: str = "normal"


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
        # E4: propagate order_urgency ("normal" or "high"). Strategies that
        # do not set this attribute default to "normal" — passive routing.
        survivor_urgency = str(getattr(survivor, "order_urgency", "normal") or "normal").strip().lower()
        if survivor_urgency not in ("normal", "high"):
            survivor_urgency = "normal"

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
                order_urgency=survivor_urgency,
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
    # Phase-8 Session 4 (R2): pre-submit STOP bus check. A truthy bus file
    # halts order building entirely — callers receive an empty intent list
    # and therefore submit nothing.
    if _stop_bus_active():
        LOG.warning("STOP_BUS_ACTIVE — IBKR intent building skipped")
        return []

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
        # Phase-8 Session 5 (S2): higher-timeframe confirmation multiplier.
        # 1.0 if daily trend agrees with the intended side, 0.6 if it
        # disagrees, 0.85 on neutral/missing data. Attenuation only —
        # never blocks an intent and never drops confidence below 0.1
        # (guaranteed by the multiplier's own clamping contract).
        tf_bias = get_higher_tf_bias(order.symbol)
        tf_mult = timeframe_confidence_multiplier(side, tf_bias)

        if raw_strength != 0.0:
            confidence = compute_confidence(
                signal_strength=normalize_signal_strength(raw_strength, method="tanh"),
                regime_quality=regime_quality_from_state(order.regime_state),
                liquidity_quality=1.0,
                tf_multiplier=tf_mult,
            )
            size_mult = sizing_multiplier(confidence)
        else:
            base_conf = float(order.confidence or 0.5)
            # Still apply TF attenuation to legacy confidence values so S2
            # reaches pre-Session-3 strategies — but floor at 0.1 to keep
            # this attenuation gentle rather than disabling.
            confidence = max(0.1, min(1.0, base_conf * tf_mult))
            size_mult = 1.0

        attenuated_size = float(order.size) * size_mult

        # Phase-8 Session 7: risk sizing layer. Apply R3 vol-adjust, then
        # R5 composite cap (per-symbol / sector / liquidity / margin), then
        # R6 correlation reducer — in that order. The integer output feeds
        # back into _normalize_quantity_for_spec which rounds for the
        # instrument type (whole units for STK/FUT/OPT, steps for FX).
        # Sizing is only applied on equity-like asset classes; futures
        # already go through futures_position_sizer upstream and crypto
        # uses the Kraken-specific path.
        if order.asset_class in (AssetClass.EQUITY, AssetClass.ETF):
            sized = _apply_sizing_layer(
                base_size=attenuated_size,
                symbol=order.symbol,
                reference_price=float(order.price),
                open_symbols=_get_open_symbols(),
            )
            if sized <= 0:
                LOG.warning(
                    "intent_skipped_sizing_zero symbol=%s base=%s",
                    order.symbol,
                    attenuated_size,
                )
                continue
            attenuated_size = float(sized)

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

        # Phase-8 Session 6 (E4): passive/aggressive order-type selection.
        # A high-urgency intent or a wide estimated spread forces a marketable
        # limit priced through the market; otherwise we submit a passive LMT
        # at the strategy's reference price.
        # Phase-8 Session 8 (A4/E5 full threading): the same bar feeds
        # bar_timestamp (A4) and current_price (E5) so the routing gates
        # compare against real data rather than degrading to time-based mode.
        latest_bar = _load_latest_bar_for_symbol(order.symbol)
        bar_ts_raw = ""
        bar_close: Optional[float] = None
        if latest_bar is not None:
            bar_ts_raw = str(latest_bar.get("ts_utc") or "")
            try:
                bc = float(latest_bar.get("close") or 0.0)
                if bc > 0.0:
                    bar_close = bc
            except (TypeError, ValueError):
                bar_close = None
        spread_pct = estimate_spread_pct(order.symbol, latest_bar)
        order_params = select_order_type(
            urgency=getattr(order, "order_urgency", "normal"),
            estimated_spread_pct=spread_pct,
        )
        if order_params["aggressive"]:
            e4_limit_price: Optional[float] = compute_aggressive_limit_price(
                side=side,
                reference_price=float(order.price),
                price_offset_pct=float(order_params["price_offset_pct"]),
            )
        else:
            e4_limit_price = float(order.price)
        e4_order_type = order_params["order_type"]

        intent = IBKRStrategyTradeIntent(
            strategy=strategy_name,
            symbol=broker_symbol,
            sec_type=spec.sec_type,
            exchange=spec.exchange,
            currency=spec.currency,
            side=side,
            order_type=e4_order_type,
            quantity=quantity,
            notional_estimate=attenuated_notional,
            limit_price=e4_limit_price,
            confidence=confidence,
            entry_reason=str(order.reason or ""),
            regime_state=str(order.regime_state or "unknown"),
            expected_pnl=float(order.expected_pnl or 0.0),
            created_at=utc_now_iso(),
            expected_price=float(order.price),
            signal_strength=raw_strength,
            signal_family=_resolve_signal_family(
                None, fallback_strategy=strategy_name
            ),
            order_urgency=str(getattr(order, "order_urgency", "normal")),
            bar_timestamp=bar_ts_raw,
        )

        # Routing gates (Phase-8 Session 2: A4/E2/E5/R7 + Session 8 full
        # threading). bar_timestamp and current_price come from the same
        # latest bar loaded above, so A4 and E5 can validate against real
        # data instead of degrading to their None-passthrough / time-based
        # paths.
        gate_bar_ts = _parse_bar_timestamp(bar_ts_raw)
        gate_current_price: Optional[float] = bar_close if bar_close and bar_close > 0 else float(order.price)
        passed, _reason = run_all_gates(
            intent=intent,
            bar_timestamp=gate_bar_ts,
            current_price=gate_current_price,
            config=_routing_gates_config,
            event_calendar=_get_event_calendar(),
        )
        if not passed:
            continue

        # Phase-8 Session 5 (S1): signal stacking vote check. With default
        # min_votes=1 this is pass-through — submit() releases the intent
        # immediately. When the operator raises min_votes, intents are
        # held until distinct signal_family votes accumulate within the
        # rolling window.
        released = _get_vote_collector().submit(intent)
        intents.extend(released)

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
    # Phase-8 Session 5 (S2): higher-timeframe confirmation multiplier.
    # Keyed on the CHAD canonical symbol (e.g. BTC-USD) so it looks up
    # data/bars/1d/BTC-USD.json directly.
    tf_bias = get_higher_tf_bias(symbol)
    ibkr_side_for_tf = "BUY" if side_attr is SignalSide.BUY else "SELL"
    tf_mult = timeframe_confidence_multiplier(ibkr_side_for_tf, tf_bias)
    if raw_strength != 0.0:
        confidence = compute_confidence(
            signal_strength=normalize_signal_strength(raw_strength, method="tanh"),
            regime_quality=regime_quality_from_state(regime_state),
            liquidity_quality=1.0,
            tf_multiplier=tf_mult,
        )
        size_mult = Decimal(str(sizing_multiplier(confidence)))
    else:
        base_conf = float(getattr(signal, "confidence", 0.5) or 0.5)
        confidence = max(0.1, min(1.0, base_conf * tf_mult))
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

    signal_urgency = str(
        getattr(signal, "order_urgency", "normal") or "normal"
    ).strip().lower()
    if signal_urgency not in ("normal", "high"):
        signal_urgency = "normal"

    # Phase-8 Session 8 (E4 Kraken): apply the passive/aggressive order-type
    # selector to the Kraken lane, matching the IBKR path. Kraken's ordertype
    # vocabulary differs — "market" vs "limit" (lowercase) — so we translate
    # the selector's LMT output accordingly. Aggressive: limit priced through
    # the market; passive: limit at the bar's close. Missing bar data keeps
    # the legacy "market" routing so the change cannot break existing callers
    # that don't emit signals with enough metadata.
    kraken_latest_bar = _load_latest_bar_for_symbol(symbol)
    kraken_bar_ts = ""
    if kraken_latest_bar is not None:
        kraken_bar_ts = str(kraken_latest_bar.get("ts_utc") or "")
    kraken_spread_pct = estimate_spread_pct(symbol, kraken_latest_bar)
    kraken_order_params = select_order_type(
        urgency=signal_urgency,
        estimated_spread_pct=kraken_spread_pct,
    )
    if kraken_latest_bar is not None:
        kraken_ordertype = "limit"
        if kraken_order_params["aggressive"]:
            kraken_price: Optional[float] = compute_aggressive_limit_price(
                side=("BUY" if side_attr is SignalSide.BUY else "SELL"),
                reference_price=float(current_price),
                price_offset_pct=float(kraken_order_params["price_offset_pct"]),
            )
        else:
            kraken_price = float(current_price)
    else:
        # No bar data → keep the historical market-order behaviour.
        kraken_ordertype = "market"
        kraken_price = None

    return KrakenStrategyTradeIntent(
        strategy=primary_strategy_name,
        pair=pair,
        side=side,
        ordertype=kraken_ordertype,
        volume=float(volume_dec),
        notional_estimate=float(attenuated_notional),
        price=kraken_price,
        confidence=confidence,
        entry_reason=str(getattr(signal, "reason", "") or ""),
        regime_state=regime_state,
        expected_pnl=float(getattr(signal, "expected_pnl", 0.0) or 0.0),
        created_at=utc_now_iso(),
        expected_price=float(current_price),
        signal_strength=raw_strength,
        signal_family=_STRATEGY_SIGNAL_FAMILY.get(primary_strategy_name, "unknown"),
        order_urgency=signal_urgency,
        bar_timestamp=kraken_bar_ts,
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
    # Phase-8 Session 4 (R2): pre-submit STOP bus check for the Kraken lane.
    if _stop_bus_active():
        LOG.warning("STOP_BUS_ACTIVE — Kraken intent building skipped")
        return []

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

        # Routing gates (Phase-8 Session 2: A4/E2/E5/R7 + Session 7: S5 +
        # Session 8 full threading). Bar timestamp and current_price come
        # from the Kraken intent's source bar.
        kraken_bar_ts_parsed = _parse_bar_timestamp(getattr(intent, "bar_timestamp", ""))
        passed, _reason = run_all_gates(
            intent=intent,
            bar_timestamp=kraken_bar_ts_parsed,
            current_price=float(price),
            config=_routing_gates_config,
            event_calendar=_get_event_calendar(),
        )
        if not passed:
            continue

        # Phase-8 Session 5 (S1): signal stacking. Default min_votes=1
        # releases the intent immediately.
        released = _get_vote_collector().submit(intent)
        out.extend(released)
    return out


# ============================================================================
# Phase-8 Session 10 (A1 completion): thin orchestrator functions
# ----------------------------------------------------------------------------
# Compose EMS → routing_gates → vote_collector → OMS explicitly. These
# are the recommended public entry points going forward — they replace
# the pattern where build_ibkr_intents_from_plan called gates and the
# vote collector inline. The old path is preserved for backward compat;
# new callers (and Session-10's backtest re-routing) should use
# execute_ibkr_cycle / execute_kraken_cycle.
# ============================================================================


def execute_ibkr_cycle(
    plan: "ExecutionPlan",
    oms: "Any",
    *,
    ems: "Any" = None,
    run_gates: bool = True,
    event_calendar: "Any" = None,
    default_sec_type: str = "STK",
    default_exchange: str = "SMART",
    default_currency: str = "USD",
) -> List["Any"]:
    """Compose: IbkrEMS → routing_gates → OMS.submit. Returns OrderResults.

    This is the thin orchestrator recommended by
    reports/audit_n_execution_landscape_20260421.json. It replaces the
    legacy pattern where ``build_ibkr_intents_from_plan`` embedded gates
    and vote-collector calls internally; the builder still exists as an
    implementation helper but callers should prefer this orchestrator.

    Parameters
    ----------
    plan : ExecutionPlan
        Output of ``build_execution_plan``.
    oms : OMSInterface
        Target OMS — IbkrOMS for live, SimulatedOMS for backtest.
    ems : EMSInterface, optional
        Defaults to a fresh IbkrEMS. Inject a custom instance for tests.
    run_gates : bool
        When True (default) every intent passes through the Session-2/7/8
        routing gates before OMS.submit. Backtests may disable this to
        observe the pre-gate fill distribution.
    event_calendar : EventCalendar, optional
        S5 event-risk calendar; falls back to the default singleton.
    """
    from chad.execution.ems import IbkrEMS
    from chad.execution.oms import OrderResult
    from chad.execution.routing_gates import run_all_gates

    _ems = ems if ems is not None else IbkrEMS()
    intents = _ems.build_intents_from_plan(
        plan,
        default_sec_type=default_sec_type,
        default_exchange=default_exchange,
        default_currency=default_currency,
    )

    results: List[OrderResult] = []
    calendar = event_calendar if event_calendar is not None else _get_event_calendar()
    for intent in intents:
        if run_gates:
            bar_ts = _parse_bar_timestamp(getattr(intent, "bar_timestamp", ""))
            try:
                current_price = float(getattr(intent, "expected_price", 0.0) or 0.0) or None
            except (TypeError, ValueError):
                current_price = None
            passed, _reason = run_all_gates(
                intent=intent,
                bar_timestamp=bar_ts,
                current_price=current_price,
                config=_routing_gates_config,
                event_calendar=calendar,
            )
            if not passed:
                continue
        order_request = _ems.build_order_request(intent)
        results.append(oms.submit(order_request))
    return results


def execute_kraken_cycle(
    routed_signals: Iterable[RoutedSignal],
    prices: Mapping[str, float],
    oms: "Any",
    *,
    ems: "Any" = None,
    run_gates: bool = True,
    event_calendar: "Any" = None,
    dynamic_cap_for_crypto: Optional[float] = None,
) -> List["Any"]:
    """Compose: KrakenEMS → routing_gates → OMS.submit for crypto signals."""
    from chad.execution.ems import KrakenEMS
    from chad.execution.oms import OrderResult
    from chad.execution.routing_gates import run_all_gates

    _ems = ems if ems is not None else KrakenEMS()
    intents = _ems.build_intents_from_signals(
        routed_signals, prices, dynamic_cap_for_crypto=dynamic_cap_for_crypto
    )

    results: List[OrderResult] = []
    calendar = event_calendar if event_calendar is not None else _get_event_calendar()
    for intent in intents:
        if run_gates:
            bar_ts = _parse_bar_timestamp(getattr(intent, "bar_timestamp", ""))
            try:
                current_price = float(getattr(intent, "expected_price", 0.0) or 0.0) or None
            except (TypeError, ValueError):
                current_price = None
            passed, _reason = run_all_gates(
                intent=intent,
                bar_timestamp=bar_ts,
                current_price=current_price,
                config=_routing_gates_config,
                event_calendar=calendar,
            )
            if not passed:
                continue
        order_request = _ems.build_order_request(intent)
        results.append(oms.submit(order_request))
    return results


# ============================================================================
# Phase-8 Session 9 (A1): backward-compat re-exports
# ----------------------------------------------------------------------------
# The OMS and EMS Protocols moved to their own modules. Callers that
# previously imported from chad.execution.execution_pipeline keep working
# via these re-exports; new code should import from chad.execution.oms
# and chad.execution.ems directly.
# ============================================================================

from chad.execution.ems import (  # noqa: E402,F401
    EMSInterface,
    IbkrEMS,
    KrakenEMS,
)
from chad.execution.oms import (  # noqa: E402,F401
    IbkrOMS,
    KrakenOMS,
    NullOMS,
    OMSInterface,
    OrderRequest,
    OrderResult,
    PRESERVED_STATUS_STRINGS,
    STATUS_DRY_RUN,
    STATUS_DUPLICATE_BLOCKED,
    STATUS_ERROR,
    STATUS_SUBMITTED,
    STATUS_UNKNOWN,
    STATUS_WHAT_IF,
)


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
    # Session 10 thin-orchestrator entry points
    "execute_ibkr_cycle",
    "execute_kraken_cycle",
    # Session 9 re-exports
    "EMSInterface",
    "IbkrEMS",
    "KrakenEMS",
    "IbkrOMS",
    "KrakenOMS",
    "NullOMS",
    "OMSInterface",
    "OrderRequest",
    "OrderResult",
    "PRESERVED_STATUS_STRINGS",
]

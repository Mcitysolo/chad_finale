
from __future__ import annotations

"""
chad/execution/ibkr_adapter.py

Professional-grade IBKR execution adapter for CHAD.

Why this module exists
----------------------
This adapter is the boundary between CHAD's internal signal/intent world and
Interactive Brokers' contract + order API surface.

Key properties
--------------
* Supports both legacy RoutedSignal submission and higher-level IBKR-style
  trade intents.
* Adds first-class futures + forex handling instead of treating everything as a
  stock.
* Uses dependency injection for the IB client factory and time source.
* Supports safe DRY_RUN mode, optional WHAT-IF simulation, idempotency, retry,
  connection reuse, and explicit contract resolution.
* Fails closed on malformed inputs.
* Keeps the original public shape familiar:
    - IbkrConfig
    - SubmittedOrder
    - IbkrAdapter.ensure_connected()
    - IbkrAdapter.shutdown()
    - IbkrAdapter.submit_routed_signals(...)
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, TypeVar, TYPE_CHECKING

from chad.options.spread_spec import OptionsSpreadSpec
from chad.execution.broker_executor import BrokerCallTimeout, call_with_timeout
from chad.execution.broker_loop import (
    BrokerLoop,
    BrokerLoopDown,
    BrokerLoopTimeout,
)
from chad.execution.futures_gate import futures_execution_disabled, is_futures_sec_type
from chad.execution.margin_shadow_gate import MarginShadowGate, order_view_from_intent
from chad.types import AssetClass, RoutedSignal, SignalSide

LOGGER = logging.getLogger("chad.execution.ibkr")

if TYPE_CHECKING:
    from chad.execution.ibkr_executor import StrategyTradeIntent


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IbkrAdapterError(RuntimeError):
    """Base class for adapter-level failures."""


class ConnectionError(IbkrAdapterError):
    """Raised when the adapter cannot establish an IBKR connection."""


class ValidationError(IbkrAdapterError):
    """Raised when input data is malformed or unsupported."""


class ContractResolutionError(IbkrAdapterError):
    """Raised when a broker contract cannot be resolved safely."""


class SubmissionError(IbkrAdapterError):
    """Raised when order submission fails after retries."""


class BrokerTimeoutError(SubmissionError):
    """Raised when a broker call exceeds the 10-second liveness deadline."""


# ---------------------------------------------------------------------------
# P0-2: Failure classification and 10-second liveness timeout
# ---------------------------------------------------------------------------

FAILURE_CLASSES = ("TIMEOUT", "BLOCKED", "REJECTED", "FAILED", "UNKNOWN")

_BROKER_TIMEOUT_S = 10.0

# L1-CLD: reader-progress watchdog cadence for the connection-owner loop.
_READER_STALL_TIMEOUT_S = 20.0
_READER_WATCHDOG_INTERVAL_S = 2.0


def _is_real_ib(ib: Any) -> bool:
    """True for a genuine ib_async ``IB`` (exposes the async twins), False for a
    sync-only ``IBLike`` test fake. Gates the connection-owner-loop routing
    (L1-CLD) so the existing sync-fake test surface is untouched."""
    return callable(getattr(ib, "connectAsync", None)) and callable(
        getattr(ib, "qualifyContractsAsync", None)
    )


def _call_with_timeout(fn: Callable[..., Any], *args: Any, timeout_s: float = _BROKER_TIMEOUT_S, label: str = "broker_call") -> Any:
    """
    Run *fn(*args)* on the shared bounded broker executor with a hard wall-clock
    timeout (Bug A L-01 root fix — see chad/execution/broker_executor.py).

    Each executor worker holds ONE persistent event loop, so ib_async's sync
    util.run() reuses it instead of minting (and, on timeout, leaking) a fresh
    loop per call — fixing the per-call event-loop/fd leak.

    On timeout: logs TIMEOUT classification and raises BrokerTimeoutError.
    On inner exception: re-raises as-is so the caller can classify.
    """
    try:
        return call_with_timeout(fn, *args, timeout_s=timeout_s, label=label)
    except BrokerCallTimeout as exc:
        LOGGER.error(
            "ibkr.broker_call_timeout",
            extra={"label": label, "timeout_s": timeout_s, "failure_class": "TIMEOUT"},
        )
        raise BrokerTimeoutError(
            f"Broker call {label!r} exceeded {timeout_s}s liveness deadline — failure_class=TIMEOUT"
        ) from exc


# ---------------------------------------------------------------------------
# Qualified-contract cache
# ---------------------------------------------------------------------------

_QUALIFY_CACHE_TTL_ENV = "CHAD_IBKR_QUALIFY_CACHE_TTL_SECONDS"
_QUALIFY_CACHE_TTL_DEFAULT_S = 24 * 3600.0


def _resolve_qualify_cache_ttl_seconds() -> float:
    raw = os.environ.get(_QUALIFY_CACHE_TTL_ENV)
    if raw is None or raw == "":
        return _QUALIFY_CACHE_TTL_DEFAULT_S
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return _QUALIFY_CACHE_TTL_DEFAULT_S


class _QualifyCache:
    """Process-local TTL cache for IBKR qualified contracts.

    Avoids repeated qualifyContracts() round-trips on identical contract
    identities within the configured TTL window. ttl_seconds<=0 disables the
    cache entirely (every lookup is a miss; nothing is stored).

    Cache keys are derived from stable contract identity fields only:
    symbol, secType, exchange, currency, lastTradeDateOrContractMonth,
    strike, right, multiplier. Fail-soft: any error during key derivation
    is treated as an uncacheable contract.
    """

    def __init__(self, *, ttl_seconds: float, now_fn: "NowFn") -> None:
        self._ttl_seconds = float(ttl_seconds)
        self._now_fn = now_fn
        self._lock = threading.Lock()
        self._entries: Dict[Tuple[str, ...], Tuple[float, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._ttl_seconds > 0.0

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def _key(self, contract: Any) -> Optional[Tuple[str, ...]]:
        try:
            return (
                str(getattr(contract, "symbol", "") or "").upper(),
                str(getattr(contract, "secType", "") or "").upper(),
                str(getattr(contract, "exchange", "") or "").upper(),
                str(getattr(contract, "currency", "") or "").upper(),
                str(getattr(contract, "lastTradeDateOrContractMonth", "") or ""),
                str(getattr(contract, "strike", "") or ""),
                str(getattr(contract, "right", "") or "").upper(),
                str(getattr(contract, "multiplier", "") or ""),
            )
        except BaseException:  # noqa: BLE001
            return None

    def get(self, contract: Any) -> Optional[Any]:
        if not self.enabled:
            return None
        key = self._key(contract)
        if key is None:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, qualified = entry
            now_ts = self._now_fn().timestamp()
            if (now_ts - stored_at) > self._ttl_seconds:
                self._entries.pop(key, None)
                return None
            return qualified

    def store(self, contract: Any, qualified: Any) -> None:
        if not self.enabled or qualified is None:
            return
        key = self._key(contract)
        if key is None:
            return
        with self._lock:
            self._entries[key] = (self._now_fn().timestamp(), qualified)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuturesContractSpec:
    """
    Static routing metadata for a futures root symbol.

    The contract month itself is resolved dynamically from IBKR contract details
    when possible.
    """

    symbol: str
    exchange: str
    currency: str = "USD"
    multiplier: Optional[str] = None
    include_expired: bool = False
    local_symbol_prefix: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class IbkrConfig:
    """
    Configuration for the IBKR adapter.

    Safe defaults:
    - dry_run=True
    - simulate_what_if_in_dry_run=False
    - idempotency enabled
    - conservative retry strategy
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 99

    dry_run: bool = True
    simulate_what_if_in_dry_run: bool = False
    validate_contracts_in_dry_run: bool = False

    connect_timeout_s: float = 10.0
    initial_status_wait_s: float = 1.0
    max_connect_retries: int = 3
    max_submit_retries: int = 2
    retry_backoff_s: float = 1.25

    # GAP-036 (Phase-53): submit→confirm lifecycle.
    # terminal_wait_s — bounded wait inside _submit_via_ib for trade.orderStatus
    #   to reach a terminal state (Filled / Cancelled / ...) before returning.
    #   Short-circuits as soon as a terminal status is observed.
    # stale_threshold_s — age beyond which a non-terminal idempotency row is
    #   considered potentially abandoned and eligible for the ib_probe reclaim
    #   path inside _SQLiteIdempotencyStore.claim_or_reclaim().
    terminal_wait_s: float = 5.0
    stale_threshold_s: float = 600.0
    # OPS-OMEGA-01 Pattern A: TTL beyond which a *terminal-positive* (Filled)
    # idempotency row is treated as expired and reclaimable. Without this, a
    # single fill permanently blocks any future submission carrying the same
    # fingerprint (strategy|symbol|side|qty|...), so a strategy that
    # legitimately re-fires hours later is stuck on duplicate_blocked.
    # Read at runtime from CHAD_DUPLICATE_CACHE_TTL_SECONDS when set.
    terminal_positive_ttl_s: float = 900.0

    default_stock_exchange: str = "SMART"
    default_stock_currency: str = "USD"
    default_forex_exchange: str = "IDEALPRO"
    default_forex_quantity_step: str = "0.0001"
    default_whole_unit_sec_types: Tuple[str, ...] = ("STK", "FUT", "OPT", "BAG")

    primary_exchange_by_symbol: Mapping[str, str] = field(
        default_factory=lambda: {
            "SPY": "ARCA",
            "QQQ": "NASDAQ",
            "IVV": "ARCA",
            "VOO": "ARCA",
        }
    )

    futures_contracts: Mapping[str, FuturesContractSpec] = field(
        default_factory=lambda: {
            "MES": FuturesContractSpec(symbol="MES", exchange="CME", currency="USD", multiplier="5", notes="Micro E-mini S&P 500"),
            "MNQ": FuturesContractSpec(symbol="MNQ", exchange="CME", currency="USD", multiplier="2", notes="Micro E-mini Nasdaq-100"),
            "MCL": FuturesContractSpec(symbol="MCL", exchange="NYMEX", currency="USD", multiplier="100", notes="Micro WTI Crude Oil"),
            "MGC": FuturesContractSpec(symbol="MGC", exchange="COMEX", currency="USD", multiplier="10", notes="Micro Gold"),
            "ZN": FuturesContractSpec(symbol="ZN", exchange="CBOT", currency="USD", multiplier="1000", notes="10-Year T-Note"),
            "ZB": FuturesContractSpec(symbol="ZB", exchange="CBOT", currency="USD", multiplier="1000", notes="30-Year T-Bond"),
            "M6E": FuturesContractSpec(symbol="M6E", exchange="CME", currency="USD", multiplier="12500", notes="Micro Euro FX"),
            "M2K": FuturesContractSpec(symbol="M2K", exchange="CME", currency="USD", multiplier="5", notes="Micro E-mini Russell 2000"),
            "MYM": FuturesContractSpec(symbol="MYM", exchange="CBOT", currency="USD", multiplier="0.5", notes="Micro E-mini Dow"),
        }
    )

    enable_idempotency: bool = True
    state_db_path: Optional[Path] = None

    account: Optional[str] = None
    outside_rth: bool = False
    tif: str = "DAY"
    allow_market_orders: bool = True

    # Broker-side open-order guard. Snapshots ib.openTrades() before each
    # placeOrder so cross-strategy submissions on the same broker lane do not
    # saturate IBKR's working-order cap and trigger Error 201.
    enable_open_order_guard: bool = True
    # Per-lane (sec_type/symbol/side) suppression threshold. Kept below IBKR's
    # 15-order cap so we leave headroom for in-flight modifications.
    open_order_cap_per_lane: int = 12

    def resolved_state_db_path(self) -> Path:
        if self.state_db_path is not None:
            return self.state_db_path
        repo_root = Path(__file__).resolve().parents[2]
        runtime_dir = repo_root / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir / "ibkr_adapter_state.sqlite3"


@dataclass(frozen=True)
class SubmittedOrder:
    """
    Representation of an order that was sent, simulated, skipped, or blocked.

    Fields kept for compatibility with the original adapter:
    - symbol
    - side
    - quantity
    - strategy
    - dry_run
    - submitted_at
    - ib_order_id
    """

    symbol: str
    side: str
    quantity: float
    strategy: List[str]
    dry_run: bool
    submitted_at: datetime
    ib_order_id: Optional[int] = None

    status: str = "unknown"
    asset_class: str = "unknown"
    sec_type: str = ""
    exchange: str = ""
    currency: str = ""
    order_type: str = "MKT"
    limit_price: Optional[float] = None
    what_if: bool = False
    perm_id: Optional[int] = None
    idempotency_key: Optional[str] = None
    contract_summary: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedIntent:
    """
    Adapter-internal normalized submission intent.

    This lets the adapter accept both legacy RoutedSignal objects and higher-
    level StrategyTradeIntent-like objects while sharing one submission path.
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
    asset_class: str
    source_strategies: Tuple[str, ...]
    created_at: datetime
    limit_price: Optional[float] = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ResolvedContract:
    contract: Any
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _PreparedOrder:
    order: Any
    quantity: float
    what_if: bool


# ---------------------------------------------------------------------------
# Protocols / dependency injection
# ---------------------------------------------------------------------------


class IBLike(Protocol):
    def isConnected(self) -> bool: ...
    def connect(self, host: str, port: int, clientId: int, timeout: float) -> Any: ...
    def disconnect(self) -> Any: ...
    def managedAccounts(self) -> Sequence[str]: ...
    def qualifyContracts(self, *contracts: Any) -> Sequence[Any]: ...
    def whatIfOrder(self, contract: Any, order: Any) -> Any: ...
    def placeOrder(self, contract: Any, order: Any) -> Any: ...
    def sleep(self, seconds: float) -> Any: ...


IBFactory = Callable[[], IBLike]
NowFn = Callable[[], datetime]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


T = TypeVar("T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _safe_upper(value: Any, default: str = "") -> str:
    return _safe_str(value, default).upper()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return default


def _quantize_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        raise ValueError("step must be > 0")
    units = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return units * step


# IBKR contract-side minTick values for FUT symbols currently routed by CHAD.
# Values match the CME / NYMEX / COMEX / CBOT published "minimum price
# fluctuation" for each contract and are cross-checked against the strategy
# specs in chad/strategies/alpha_futures.py (MES/MNQ/MCL/MGC/ZN/ZB).
# Used only by _snap_lmt_price_to_tick; STK / OPT / BAG / unknown FUT symbols
# fall through unchanged so we do not silently substitute a guessed tick.
_FUT_MIN_TICK_BY_SYMBOL: Dict[str, Decimal] = {
    "MES": Decimal("0.25"),
    "MNQ": Decimal("0.25"),
    "MCL": Decimal("0.01"),
    "MGC": Decimal("0.10"),
    "ZN":  Decimal("0.015625"),
    "ZB":  Decimal("0.03125"),
    "M6E": Decimal("0.0001"),
    "SIL": Decimal("0.005"),
    "MYM": Decimal("1.0"),
    "M2K": Decimal("0.10"),
}


def _snap_lmt_price_to_tick(
    price: float,
    *,
    side: str,
    sec_type: str,
    symbol: str,
) -> Tuple[float, Optional[Decimal]]:
    """Snap a LMT price to the instrument minTick boundary.

    Policy is side-safe / conservative so a rounding correction never makes
    the order MORE aggressive than the strategy intended:
      - BUY  -> floor-to-tick (will not accidentally pay more than the
                originally-computed bid).
      - SELL -> ceil-to-tick  (will not accidentally accept less than the
                originally-computed ask).

    Only FUT symbols listed in _FUT_MIN_TICK_BY_SYMBOL are snapped. STK / OPT
    / BAG and unknown FUT symbols pass through unchanged — callers can detect
    "unknown tick" via the returned tick being None and log accordingly
    instead of silently rounding to an assumed value.

    Returns (snapped_price, tick) where `tick` is the Decimal tick that was
    applied, or None if no snap was performed. The function is pure (no
    logging / no I/O) so it is trivial to unit-test.
    """
    if _safe_upper(sec_type) != "FUT":
        return float(price), None
    sym = _safe_upper(symbol)
    tick = _FUT_MIN_TICK_BY_SYMBOL.get(sym)
    if tick is None:
        return float(price), None
    try:
        price_d = Decimal(str(price))
    except (InvalidOperation, ValueError):
        return float(price), None
    if not price_d.is_finite() or price_d <= 0:
        return float(price), None
    side_u = _safe_upper(side)
    if side_u == "BUY":
        rounding = ROUND_DOWN
    elif side_u == "SELL":
        rounding = ROUND_UP
    else:
        # Defensive: intent.side is validated upstream, but fall back to
        # nearest-tick if a caller ever passes something unexpected.
        rounding = ROUND_HALF_EVEN
    steps = (price_d / tick).to_integral_value(rounding=rounding)
    snapped = steps * tick
    if snapped <= 0:
        # Guard against a positive price snapping down to zero on absurdly
        # small inputs; preserve the original price instead of submitting 0.
        return float(price), tick
    return float(snapped), tick


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value")).strip().lower()
    return str(value).strip().lower()


def _side_to_ib(side: Any) -> str:
    side_value = _enum_value(side)
    if side_value == "buy":
        return "BUY"
    if side_value == "sell":
        return "SELL"
    raise ValidationError(f"Unsupported side: {side!r}")


def _normalize_symbol(symbol: Any) -> str:
    out = _safe_upper(symbol)
    if not out:
        raise ValidationError("symbol is required")
    return out


def _stable_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(data: Mapping[str, Any]) -> str:
    raw = _stable_json(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _asset_class_key(asset_class: Any) -> str:
    return _enum_value(asset_class)


# Explicit symbol → asset_class lookup for the paper-execution evidence path,
# where sec_type is often absent. Used as a fallback when sec_type is missing
# or ambiguous so fills are never recorded with asset_class="unknown".
_FUTURES_SYMBOLS = frozenset({
    "MES", "MNQ", "ES", "NQ", "MGC", "MCL", "MCLK6",
    "ZN", "ZB", "M6E", "SI", "SIL",
    "GC", "CL", "RTY", "YM", "MYM",
    # M2K (Micro E-mini Russell 2000) has a FuturesContractSpec entry below
    # and an EXPIRY_SCHEDULE entry in futures_contract_resolver, but was
    # missing here — resolve_asset_class("M2K") silently fell through to
    # "equity", which made reconciler close intents for the open M2K SELL
    # paper position emit as STK→SMART (GAP-037).
    "M2K",
})
_ETF_SYMBOLS = frozenset({
    "GLD", "TLT", "VWO", "UVXY", "SVXY", "SIL", "PSQ", "SH",
    "DIA", "IEMG",
})
_EQUITY_ETF_TICKERS = frozenset({"SPY", "QQQ", "IWM"})
_EQUITY_SYMBOLS = frozenset({
    "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "AMZN", "META",
    "TSLA", "NFLX", "AMD",
})
_CRYPTO_SYMBOLS = frozenset({
    "BTC", "ETH", "SOL", "BTCUSD", "ETHUSD", "SOLUSD",
    "XBTUSD", "XETHZUSD",
})


def resolve_asset_class(symbol: Any, sec_type: Any = "") -> str:
    """Map a symbol (plus optional IBKR sec_type) to a canonical asset_class.

    Returns one of: "equity", "etf", "futures", "forex", "crypto", "options".
    Never returns "unknown" for a recognizable instrument. Falls back to
    "equity" only as a last resort for a short alpha-only ticker.

    Why this exists: the paper-executor path frequently lacks sec_type, which
    caused fills to be written with asset_class="unknown" and SCR to exclude
    them as untrusted (effective_trades stuck below the CAUTIOUS threshold).
    Symbol pattern matching + explicit lists are authoritative here.
    """
    sym = ""
    if symbol is not None:
        sym = str(symbol).strip().upper()
    stype = ""
    if sec_type is not None:
        stype = str(sec_type).strip().upper()

    # Prefer sec_type when it's a clear IBKR code.
    if stype == "FUT":
        return "futures"
    if stype == "CASH":
        return "forex"
    if stype == "OPT":
        return "options"
    # BAG is IBKR's combo container — used exclusively for multi-leg
    # options orders in CHAD (vertical spreads from alpha_options). Without
    # this branch, SPY/BAG would fall through to the ETF lookup below and
    # silently downgrade options combo fills to "etf".
    if stype in ("BAG", "COMBO"):
        return "options"
    if stype == "CRYPTO":
        return "crypto"
    # STK is ambiguous between equity and ETF — fall through to symbol lookup.

    if not sym:
        return "equity" if stype == "STK" else "unknown"

    # Explicit lists (authoritative).
    if sym in _FUTURES_SYMBOLS:
        return "futures"
    if sym in _ETF_SYMBOLS or sym in _EQUITY_ETF_TICKERS:
        return "etf"
    if sym in _EQUITY_SYMBOLS:
        return "equity"
    if sym in _CRYPTO_SYMBOLS:
        return "crypto"

    # Pattern matching.
    # Futures root with month/year suffix, e.g. "MCLK6", "MESH6", "ESU5".
    if len(sym) >= 3:
        for root in _FUTURES_SYMBOLS:
            if sym.startswith(root) and len(sym) <= len(root) + 3:
                tail = sym[len(root):]
                if tail and tail[0].isalpha():  # month code
                    return "futures"

    # FX pairs: 6 letters, either EURUSD-style or with slash.
    compact = "".join(ch for ch in sym if ch.isalpha())
    if "/" in sym and len(compact) == 6:
        return "forex"
    if len(sym) == 6 and sym.isalpha():
        known_fx_bases = {"EUR", "USD", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
        if sym[:3] in known_fx_bases and sym[3:] in known_fx_bases:
            return "forex"

    # Crypto heuristic: anything ending in USD/USDT with BTC/ETH/SOL-like prefix.
    if sym.endswith(("USD", "USDT", "USDC")):
        prefix = sym[: -3 if sym.endswith("USD") else -4]
        if prefix in _CRYPTO_SYMBOLS or prefix in {"BTC", "ETH", "SOL", "XBT", "XETH"}:
            return "crypto"

    # If IBKR already told us it's a stock, trust that.
    if stype == "STK":
        return "equity"

    # Last resort: return "equity" rather than "unknown" so SCR doesn't
    # silently drop the fill. Short alpha-only tickers are overwhelmingly
    # equities in this codebase.
    if sym.isalpha() and 1 <= len(sym) <= 5:
        return "equity"

    return "unknown"


def _strategy_name(value: Any) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value")).strip()
    return str(value).strip()


def _jsonable(value: Any) -> Any:
    """
    Best-effort JSON serializer for ib_insync / dataclass / enum objects.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _parse_fx_pair(raw_symbol: str) -> Tuple[str, str, str]:
    compact = "".join(ch for ch in raw_symbol.upper() if ch.isalpha())
    if len(compact) != 6:
        raise ValidationError(f"Invalid FX pair symbol: {raw_symbol!r}")
    base = compact[:3]
    quote = compact[3:]
    return compact, base, quote


def _should_connect(config: IbkrConfig, *, force: bool = False) -> bool:
    if force:
        return True
    if config.dry_run and not config.simulate_what_if_in_dry_run and not config.validate_contracts_in_dry_run:
        return False
    return True


# ---------------------------------------------------------------------------
# SQLite idempotency store
# ---------------------------------------------------------------------------


# GAP-036 (Phase-53): canonical state-machine for idempotency rows. The row's
# `status` column is the verbatim IBKR orderStatus string (or the adapter's
# own pseudo-statuses like "claimed", "duplicate_blocked"); classification is
# derived by lower-casing and bucket lookup.
#
# terminal-positive: a successful Fill — re-submitting the same logical intent
#   after this would double-buy or double-sell. UNCONDITIONALLY blocks.
# terminal-negative: a failed terminal outcome — a legitimate retry of the
#   same logical intent is permitted (DELETE + fresh INSERT).
# non-terminal: order is still working at IB (or the adapter hasn't yet seen
#   a transition). Blocks re-submission until either a terminal event arrives
#   or the row goes stale and the ib_probe reclaim path runs.
_IDEMPOTENCY_TERMINAL_POSITIVE: frozenset = frozenset({"filled"})
_IDEMPOTENCY_TERMINAL_NEGATIVE: frozenset = frozenset({
    "cancelled",
    "apicancelled",
    "rejected",
    "inactive",
    "error",
})

# Probe result tags (string sentinels — see IbkrAdapter._ib_probe).
_PROBE_STILL_ACTIVE = "STILL_ACTIVE"
_PROBE_ABSENT = "ABSENT"
_PROBE_TERMINAL_PREFIX = "TERMINAL_AT_BROKER:"  # suffix is the IB status string


def _classify_idempotency_status(status: Optional[str]) -> str:
    """Return one of: 'terminal_positive' | 'terminal_negative' | 'non_terminal'."""
    raw = (status or "").strip().lower()
    if raw in _IDEMPOTENCY_TERMINAL_POSITIVE:
        return "terminal_positive"
    if raw in _IDEMPOTENCY_TERMINAL_NEGATIVE:
        return "terminal_negative"
    return "non_terminal"


def _parse_row_updated_at(value: Any) -> Optional[datetime]:
    """Best-effort parse of the SQLite updated_at_utc string back to datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class _SQLiteIdempotencyStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ibkr_exec_state (
                    idempotency_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    broker_order_id INTEGER,
                    payload_json TEXT NOT NULL,
                    result_json TEXT
                )
                """
            )

    def claim(self, key: str, payload: Mapping[str, Any], now: datetime) -> bool:
        with self._lock, contextlib.closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO ibkr_exec_state (
                    idempotency_key, status, created_at_utc, updated_at_utc,
                    broker_order_id, payload_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    "claimed",
                    now.isoformat(),
                    now.isoformat(),
                    None,
                    _stable_json(payload),
                    None,
                ),
            )
            return cur.rowcount == 1

    def mark(self, key: str, *, status: str, broker_order_id: Optional[int], result: Mapping[str, Any], now: datetime) -> None:
        with self._lock, contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE ibkr_exec_state
                   SET status = ?, updated_at_utc = ?, broker_order_id = ?, result_json = ?
                 WHERE idempotency_key = ?
                """,
                (
                    status,
                    now.isoformat(),
                    broker_order_id,
                    _stable_json(result),
                    key,
                ),
            )

    def get(self, key: str) -> Optional[Mapping[str, Any]]:
        with self._lock, contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT idempotency_key, status, created_at_utc, updated_at_utc,
                       broker_order_id, payload_json, result_json
                  FROM ibkr_exec_state
                 WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "idempotency_key": row[0],
            "status": row[1],
            "created_at_utc": row[2],
            "updated_at_utc": row[3],
            "broker_order_id": row[4],
            "payload_json": row[5],
            "result_json": row[6],
        }

    # ------------------------------------------------------------------
    # GAP-036 (Phase-53): state-machine-aware claim with reclaim policy.
    # ------------------------------------------------------------------
    def claim_or_reclaim(
        self,
        key: str,
        payload: Mapping[str, Any],
        now: datetime,
        *,
        stale_threshold_s: float,
        ib_probe: Optional[Callable[[Optional[int]], str]] = None,
        terminal_positive_ttl_s: Optional[float] = None,
    ) -> bool:
        """State-machine-aware idempotency claim.

        Returns True when the caller MAY proceed to submit; False to block.
        Schema is unchanged — semantics live entirely in this method:

          S1 fresh row → INSERT 'claimed' → True
          S2 existing row status ∈ terminal-positive ({filled}) → False
             (UNCONDITIONAL; re-submitting after a Fill double-trades)
          S3 existing row status ∈ terminal-negative
             ({cancelled, apicancelled, rejected, inactive, error}) → DELETE
             old row + INSERT fresh 'claimed' row → True (retry permitted)
          S4 existing row status is non-terminal AND age < stale_threshold_s
             → False (prior submission still working)
          S5 existing row status is non-terminal AND age ≥ stale_threshold_s
             → invoke ib_probe(broker_order_id):
               STILL_ACTIVE → bump updated_at_utc, return False
               TERMINAL_AT_BROKER:<status> → mark(row, status) then:
                   if classified terminal-positive → False (Filled-block)
                   if classified terminal-negative → DELETE + fresh INSERT
                     → True (legitimate retry)
                   otherwise (still non-terminal report) → bump updated_at,
                     return False
               ABSENT → DELETE old row + INSERT fresh 'claimed' → True
             If no ib_probe is supplied, the stale row is treated
             defensively as STILL_ACTIVE (return False) — the caller must
             have wired the probe in production.
        """
        with self._lock, contextlib.closing(self._connect()) as conn:
            # Step 1 — fresh row.
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO ibkr_exec_state (
                    idempotency_key, status, created_at_utc, updated_at_utc,
                    broker_order_id, payload_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    "claimed",
                    now.isoformat(),
                    now.isoformat(),
                    None,
                    _stable_json(payload),
                    None,
                ),
            )
            if cur.rowcount == 1:
                return True

            # Existing row — examine.
            row = conn.execute(
                """
                SELECT status, updated_at_utc, broker_order_id
                  FROM ibkr_exec_state
                 WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                # Race: the row vanished between INSERT OR IGNORE and SELECT.
                # Retry the fresh-row path conservatively.
                conn.execute(
                    """
                    INSERT INTO ibkr_exec_state (
                        idempotency_key, status, created_at_utc, updated_at_utc,
                        broker_order_id, payload_json, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        "claimed",
                        now.isoformat(),
                        now.isoformat(),
                        None,
                        _stable_json(payload),
                        None,
                    ),
                )
                return True

            existing_status, existing_updated_at, existing_order_id = row
            classification = _classify_idempotency_status(existing_status)

            # Compute row age once; reused by S2-TTL and S4/S5 branches.
            age_dt = _parse_row_updated_at(existing_updated_at)
            if age_dt is None:
                # Unparseable timestamp — treat as stale to drive recovery.
                age_seconds = float("inf")
            else:
                age_seconds = (now - age_dt).total_seconds()

            # S2 — terminal-positive (Filled). UNCONDITIONAL block within the
            # configured TTL window; reclaimable once the row is older than
            # ``terminal_positive_ttl_s`` (OPS-OMEGA-01 Pattern A). The TTL
            # exists because a strategy fingerprint can legitimately re-fire
            # hours after a fill — the cache must not permanently lock it out.
            # When ``terminal_positive_ttl_s`` is None or <= 0 the previous
            # "block forever" semantics are preserved.
            if classification == "terminal_positive":
                ttl_val = terminal_positive_ttl_s
                if ttl_val is None or float(ttl_val) <= 0.0 or age_seconds < float(ttl_val):
                    return False
                LOGGER.info(
                    "ibkr.idempotency_terminal_positive_expired_reclaim",
                    extra={
                        "key": key,
                        "age_seconds": age_seconds,
                        "ttl_seconds": float(ttl_val),
                        "prior_status": existing_status,
                    },
                )
                self._delete_and_reinsert(conn, key, payload, now)
                return True

            # S3 — terminal-negative. DELETE + fresh INSERT.
            if classification == "terminal_negative":
                self._delete_and_reinsert(conn, key, payload, now)
                return True

            # S4 — non-terminal AND not stale → block (still working).
            if age_seconds < float(stale_threshold_s):
                return False

            # S5 — non-terminal AND stale → consult probe.
            broker_order_id: Optional[int] = (
                int(existing_order_id) if existing_order_id is not None else None
            )
            probe_result: Optional[str] = None
            if ib_probe is not None:
                try:
                    probe_result = ib_probe(broker_order_id)
                except BaseException as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "ibkr.idempotency_probe_failed",
                        extra={"key": key, "error": str(exc)},
                    )
                    probe_result = None

            if probe_result is None:
                # No probe wired (or probe raised) — fail closed by bumping
                # updated_at_utc and blocking. The caller MUST install a
                # probe in production to drive the stale-reclaim path.
                conn.execute(
                    "UPDATE ibkr_exec_state SET updated_at_utc = ? "
                    "WHERE idempotency_key = ?",
                    (now.isoformat(), key),
                )
                return False

            if probe_result == _PROBE_STILL_ACTIVE:
                conn.execute(
                    "UPDATE ibkr_exec_state SET updated_at_utc = ? "
                    "WHERE idempotency_key = ?",
                    (now.isoformat(), key),
                )
                return False

            if probe_result.startswith(_PROBE_TERMINAL_PREFIX):
                broker_status = probe_result[len(_PROBE_TERMINAL_PREFIX):]
                # Promote the row to the broker's reported terminal status.
                conn.execute(
                    "UPDATE ibkr_exec_state "
                    "SET status = ?, updated_at_utc = ? "
                    "WHERE idempotency_key = ?",
                    (broker_status, now.isoformat(), key),
                )
                bclass = _classify_idempotency_status(broker_status)
                if bclass == "terminal_positive":
                    return False
                if bclass == "terminal_negative":
                    self._delete_and_reinsert(conn, key, payload, now)
                    return True
                # Probe returned a non-terminal IB status (rare) — block but
                # leave the bumped timestamp so subsequent cycles re-probe.
                return False

            if probe_result == _PROBE_ABSENT:
                # ABSENT means the order has no trace at IB *and* no
                # execution evidence for it (the probe enforces the
                # execution-check hardening before returning ABSENT — see
                # IbkrAdapter._ib_probe).
                self._delete_and_reinsert(conn, key, payload, now)
                return True

            # Unknown probe result — fail closed.
            LOGGER.warning(
                "ibkr.idempotency_probe_unknown_result",
                extra={"key": key, "probe_result": probe_result},
            )
            conn.execute(
                "UPDATE ibkr_exec_state SET updated_at_utc = ? "
                "WHERE idempotency_key = ?",
                (now.isoformat(), key),
            )
            return False

    @staticmethod
    def _delete_and_reinsert(
        conn: "sqlite3.Connection",
        key: str,
        payload: Mapping[str, Any],
        now: datetime,
    ) -> None:
        """Atomic (single-connection) replace of a terminal/stale row with a
        fresh 'claimed' row. Used by S3 / S5 reclaim branches."""
        conn.execute(
            "DELETE FROM ibkr_exec_state WHERE idempotency_key = ?",
            (key,),
        )
        conn.execute(
            """
            INSERT INTO ibkr_exec_state (
                idempotency_key, status, created_at_utc, updated_at_utc,
                broker_order_id, payload_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                "claimed",
                now.isoformat(),
                now.isoformat(),
                None,
                _stable_json(payload),
                None,
            ),
        )


# ---------------------------------------------------------------------------
# Lazy ib_insync imports
# ---------------------------------------------------------------------------


def _lazy_import_ib_factory() -> IBFactory:
    # P0-3: No implicit IB() creation in the hot path.
    # Callers must inject a pre-managed IB session via ib_factory.
    def factory() -> IBLike:
        raise ConnectionError(
            "No IB session injected — hot-path execution requires an "
            "externally managed IB session passed via ib_factory"
        )

    return factory


def _lazy_import_contract_classes() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        from ib_async import Contract, Future, Forex, Order, Stock  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ConnectionError(
            "ib_async is not installed. Install it inside the CHAD venv."
        ) from exc
    return Contract, Future, Forex, Order, Stock


def _lazy_import_option_class() -> Any:
    try:
        from ib_async import Option  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ConnectionError(
            "ib_async is not installed. Install it inside the CHAD venv."
        ) from exc
    return Option


# ---------------------------------------------------------------------------
# Contract resolution
# ---------------------------------------------------------------------------


class _ContractResolver:
    def __init__(
        self,
        config: IbkrConfig,
        now_fn: NowFn,
        *,
        broker_call: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._config = config
        self._now_fn = now_fn
        self._cache: MutableMapping[str, _ResolvedContract] = {}
        # L1-CLD: when the adapter homed the connection on the owner loop it
        # passes its _broker_call so qualify round-trips are marshalled onto
        # that loop. Absent (direct construction / dry-run) -> legacy path.
        self._broker_call = broker_call

    def _qualify(self, ib: IBLike, *contracts: Any, label: str) -> Any:
        """Qualify one or more contracts, routing through the connection-owner
        loop when available (L1-CLD), else the legacy bounded sync executor."""
        if self._broker_call is not None:
            return self._broker_call(ib, "qualifyContracts", *contracts, label=label)
        return _call_with_timeout(ib.qualifyContracts, *contracts, label=label)

    def resolve(self, ib: Optional[IBLike], intent: NormalizedIntent) -> _ResolvedContract:
        cache_key = _hash_payload(
            {
                "symbol": intent.symbol,
                "sec_type": intent.sec_type,
                "exchange": intent.exchange,
                "currency": intent.currency,
                "meta": dict(intent.meta),
            }
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        sec_type = _safe_upper(intent.sec_type)
        if sec_type == "STK":
            resolved = self._resolve_stock(intent)
        elif sec_type == "CASH":
            resolved = self._resolve_forex(intent)
        elif sec_type == "FUT":
            resolved = self._resolve_future(ib, intent)
        elif sec_type == "OPT":
            resolved = self._resolve_option(ib, intent)
        elif sec_type == "BAG":
            resolved = self._resolve_combo(ib, intent)
        else:
            raise ContractResolutionError(f"Unsupported sec_type: {intent.sec_type!r}")

        self._cache[cache_key] = resolved
        return resolved

    def _resolve_stock(self, intent: NormalizedIntent) -> _ResolvedContract:
        _Contract, _Future, _Forex, _Order, Stock = _lazy_import_contract_classes()
        primary_exchange = self._config.primary_exchange_by_symbol.get(intent.symbol)
        kwargs: Dict[str, Any] = {}
        if primary_exchange:
            kwargs["primaryExchange"] = primary_exchange

        # Defensive normalization (IBKR Error 321 fix): every STK contract
        # submitted to IBKR must carry an explicit exchange and currency, or
        # IB rejects with "Missing order exchange". Strategy-level intents
        # (e.g. reconciler close intents) do not always populate these
        # fields, so fall back to intent.meta then to configured defaults
        # (SMART / USD). Explicit non-empty values are preserved verbatim.
        meta_exchange = _safe_upper(intent.meta.get("exchange")) if intent.meta else ""
        meta_currency = _safe_upper(intent.meta.get("currency")) if intent.meta else ""
        intent_exchange = _safe_upper(intent.exchange)
        intent_currency = _safe_upper(intent.currency)
        exchange = (
            intent_exchange
            or meta_exchange
            or _safe_upper(self._config.default_stock_exchange)
            or "SMART"
        )
        currency = (
            intent_currency
            or meta_currency
            or _safe_upper(self._config.default_stock_currency)
            or "USD"
        )
        normalized = (exchange != intent_exchange) or (currency != intent_currency)

        contract = Stock(intent.symbol, exchange, currency, **kwargs)
        summary = {
            "symbol": intent.symbol,
            "sec_type": "STK",
            "exchange": exchange,
            "currency": currency,
            "primary_exchange": primary_exchange,
        }
        if normalized:
            LOGGER.info(
                "IBKR_CONTRACT_NORMALIZED symbol=%s sec_type=STK exchange=%s currency=%s",
                intent.symbol, exchange, currency,
            )
        return _ResolvedContract(contract=contract, summary=summary)

    def _resolve_forex(self, intent: NormalizedIntent) -> _ResolvedContract:
        _Contract, _Future, Forex, _Order, _Stock = _lazy_import_contract_classes()
        pair, base, quote = _parse_fx_pair(intent.symbol)
        contract = Forex(pair, exchange=intent.exchange)
        summary = {
            "symbol": pair,
            "sec_type": "CASH",
            "exchange": intent.exchange,
            "currency": quote,
            "base_currency": base,
            "quote_currency": quote,
        }
        return _ResolvedContract(contract=contract, summary=summary)

    def _resolve_future(self, ib: Optional[IBLike], intent: NormalizedIntent) -> _ResolvedContract:
        _Contract, Future, _Forex, _Order, _Stock = _lazy_import_contract_classes()
        spec = self._config.futures_contracts.get(intent.symbol)
        if spec is None:
            raise ContractResolutionError(f"Unsupported futures symbol: {intent.symbol!r}")

        explicit_month = _safe_str(intent.meta.get("lastTradeDateOrContractMonth") or intent.meta.get("contract_month"))
        multiplier = _safe_str(intent.meta.get("multiplier") or spec.multiplier or "")

        if explicit_month:
            contract = Future(
                symbol=spec.symbol,
                lastTradeDateOrContractMonth=explicit_month,
                exchange=spec.exchange,
                currency=spec.currency,
                multiplier=multiplier or None,
            )
            summary = {
                "symbol": spec.symbol,
                "sec_type": "FUT",
                "exchange": spec.exchange,
                "currency": spec.currency,
                "contract_month": explicit_month,
                "multiplier": multiplier or None,
                "resolution": "explicit",
            }
            return _ResolvedContract(contract=contract, summary=summary)

        if ib is None:
            # In pure dry-run mode we may intentionally avoid network access.
            # We still return a broad futures contract for logging, but this is
            # not sufficient for live submission.
            contract = Future(
                symbol=spec.symbol,
                exchange=spec.exchange,
                currency=spec.currency,
                multiplier=multiplier or None,
            )
            summary = {
                "symbol": spec.symbol,
                "sec_type": "FUT",
                "exchange": spec.exchange,
                "currency": spec.currency,
                "contract_month": None,
                "multiplier": multiplier or None,
                "resolution": "broad_unqualified",
            }
            return _ResolvedContract(contract=contract, summary=summary)

        # P0-1: No live network lookup in hot path.
        # contract_month MUST be provided via intent.meta for futures execution.
        raise ContractResolutionError(
            f"Futures contract resolution for {spec.symbol} on {spec.exchange} "
            f"requires explicit contract_month in intent.meta — "
            f"live reqContractDetails lookup is prohibited in the hot path"
        )

    def _resolve_option(self, ib: Optional[IBLike], intent: NormalizedIntent) -> _ResolvedContract:
        """
        Resolve an options contract from intent metadata.

        Required meta fields:
          - expiry: str (YYYYMMDD format)
          - strike: float
          - right: str ("C" or "P")

        Optional meta fields:
          - exchange: str (default "SMART")
          - multiplier: str (default "100")
        """
        Option = _lazy_import_option_class()

        expiry = _safe_str(intent.meta.get("expiry") or intent.meta.get("lastTradeDateOrContractMonth"))
        if not expiry:
            raise ContractResolutionError(
                f"Options contract for {intent.symbol} requires 'expiry' in intent.meta"
            )

        strike_raw = intent.meta.get("strike")
        if strike_raw is None:
            raise ContractResolutionError(
                f"Options contract for {intent.symbol} requires 'strike' in intent.meta"
            )
        try:
            strike = float(strike_raw)
        except (TypeError, ValueError):
            raise ContractResolutionError(
                f"Options contract for {intent.symbol}: invalid strike value {strike_raw!r}"
            )

        right = _safe_upper(_safe_str(intent.meta.get("right")))
        if right not in ("C", "P"):
            raise ContractResolutionError(
                f"Options contract for {intent.symbol} requires 'right' (C or P) in intent.meta, got {right!r}"
            )

        exchange = _safe_str(intent.meta.get("exchange")) or intent.exchange or "SMART"
        multiplier = _safe_str(intent.meta.get("multiplier")) or "100"

        contract = Option(
            symbol=intent.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=strike,
            right=right,
            exchange=exchange,
            currency=intent.currency,
            multiplier=multiplier,
        )

        # Qualify contract if IB session available (non-hot-path)
        if ib is not None:
            try:
                qualified = self._qualify(ib, contract, label="qualifyContracts.option")
                if qualified:
                    contract = qualified[0]
            except Exception as exc:
                LOGGER.warning(
                    "ibkr_adapter.option_qualify_failed",
                    extra={"symbol": intent.symbol, "expiry": expiry, "strike": strike, "error": str(exc)},
                )

        summary = {
            "symbol": intent.symbol,
            "sec_type": "OPT",
            "expiry": expiry,
            "strike": strike,
            "right": right,
            "exchange": exchange,
            "currency": intent.currency,
            "multiplier": multiplier,
            "resolution": "qualified" if ib is not None else "unqualified",
        }
        return _ResolvedContract(contract=contract, summary=summary)

    def _resolve_combo(self, ib: Optional[IBLike], intent: NormalizedIntent) -> _ResolvedContract:
        """
        Resolve a BAG (combo) contract for vertical spreads.

        Builds an IBKR combo contract with two legs (long + short) from
        intent metadata. Qualifies individual leg contracts to obtain conIds
        when an IB session is available.

        Required meta fields:
          - expiry: str (YYYYMMDD)
          - long_strike: float
          - short_strike: float
          - long_right: str ("C" or "P")
          - short_right: str ("C" or "P")
        """
        _Contract, _Future, _Forex, _Order, _Stock = _lazy_import_contract_classes()
        Option = _lazy_import_option_class()

        meta = dict(intent.meta)

        # Phase D Item 2 Tier 1 — prefer the typed OptionsSpreadSpec when the
        # strategy stamped one under meta["spread_spec"]. Falls back to the
        # legacy string-keyed dict path when the spec is absent or arrives
        # in serialized form. Either path raises the same
        # ContractResolutionError on bad data so downstream behavior (live
        # log markers, qualification skipping, cache invalidation) is
        # preserved exactly.
        spec_obj = meta.get("spread_spec")
        if spec_obj is not None and not isinstance(spec_obj, OptionsSpreadSpec):
            # Tolerate a dict (or any mapping) accidentally placed under
            # the typed key: rebuild via from_legacy_meta so the typed
            # validators still run.
            if isinstance(spec_obj, Mapping):
                try:
                    spec_obj = OptionsSpreadSpec.from_legacy_meta(intent.symbol, spec_obj)
                except Exception as exc:
                    raise ContractResolutionError(
                        f"BAG contract for {intent.symbol}: invalid spread_spec dict ({exc})"
                    ) from exc
            else:
                spec_obj = None

        if spec_obj is None:
            # Pre-validate the legacy fields with the same error messages
            # callers (and tests) already match against, then build the
            # typed spec from the validated values. This preserves the
            # historical error surface while still flowing through the
            # typed dataclass.
            expiry_raw = _safe_str(meta.get("expiry") or meta.get("lastTradeDateOrContractMonth"))
            if not expiry_raw:
                raise ContractResolutionError(
                    f"BAG contract for {intent.symbol} requires 'expiry' in intent.meta"
                )
            ls_raw = meta.get("long_strike")
            ss_raw = meta.get("short_strike")
            if ls_raw is None or ss_raw is None:
                raise ContractResolutionError(
                    f"BAG contract for {intent.symbol} requires 'long_strike' and 'short_strike' in intent.meta"
                )
            try:
                _ = float(ls_raw)
                _ = float(ss_raw)
            except (TypeError, ValueError):
                raise ContractResolutionError(
                    f"BAG contract for {intent.symbol}: invalid strike values"
                )
            lr_raw = _safe_upper(_safe_str(meta.get("long_right")))
            sr_raw = _safe_upper(_safe_str(meta.get("short_right")))
            if lr_raw not in ("C", "P") or sr_raw not in ("C", "P"):
                raise ContractResolutionError(
                    f"BAG contract for {intent.symbol} requires 'long_right' and 'short_right' (C or P)"
                )

            try:
                spec_obj = OptionsSpreadSpec.from_legacy_meta(intent.symbol, meta)
            except ValueError as exc:
                # Any remaining typed-validator failure (e.g. same-strike,
                # malformed expiry that bypassed the regex check above) is
                # surfaced as an invalid-strike / generic resolution error.
                msg = str(exc)
                if "differ" in msg:
                    raise ContractResolutionError(
                        f"BAG contract for {intent.symbol}: invalid strike values"
                    ) from exc
                raise ContractResolutionError(
                    f"BAG contract for {intent.symbol}: {msg}"
                ) from exc

        # Pull the canonical, validated fields off the typed spec.
        expiry = spec_obj.expiry
        long_strike = float(spec_obj.long_strike)
        short_strike = float(spec_obj.short_strike)
        long_right = spec_obj.long_right
        short_right = spec_obj.short_right

        exchange = (
            _safe_str(meta.get("exchange"))
            or spec_obj.exchange
            or intent.exchange
            or "SMART"
        )
        currency = intent.currency or spec_obj.currency or "USD"

        # Build individual leg contracts for qualification
        long_opt = Option(
            symbol=intent.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=long_strike,
            right=long_right,
            exchange=exchange,
            currency=currency,
        )
        short_opt = Option(
            symbol=intent.symbol,
            lastTradeDateOrContractMonth=expiry,
            strike=short_strike,
            right=short_right,
            exchange=exchange,
            currency=currency,
        )

        # Qualify legs to obtain conIds when IB session is available.
        #
        # Both legs MUST resolve to non-zero conIds before the BAG is built —
        # IBKR rejects combo orders whose legs reference conId=0
        # (Error 321: "combo details for leg '0' are invalid"). When ib is
        # None we are in dry-run/preview territory; tolerate conId=0 there
        # because no order is ever submitted on that path.
        long_con_id = 0
        short_con_id = 0
        if ib is not None:
            qualify_exc: Optional[BaseException] = None
            qualified: Sequence[Any] = ()
            try:
                qualified = self._qualify(
                    ib,
                    long_opt,
                    short_opt,
                    label="qualifyContracts.combo_legs",
                ) or ()
            except BaseException as exc:  # noqa: BLE001
                qualify_exc = exc
                LOGGER.warning(
                    "ibkr_adapter.combo_qualify_failed",
                    extra={
                        "symbol": intent.symbol,
                        "expiry": expiry,
                        "long_strike": long_strike,
                        "short_strike": short_strike,
                        "error": str(exc),
                    },
                )

            if qualify_exc is None and qualified and len(qualified) >= 2:
                long_opt = qualified[0]
                short_opt = qualified[1]
                long_con_id = int(getattr(long_opt, "conId", 0) or 0)
                short_con_id = int(getattr(short_opt, "conId", 0) or 0)

            if long_con_id <= 0 or short_con_id <= 0:
                # Refuse to build a BAG with conId=0 legs — IBKR will reject
                # with Error 321 on submit. Surface a clear marker so the
                # operator (and the live_loop log scrape) can attribute the
                # skip directly to leg-qualification failure.
                LOGGER.warning(
                    "BAG_INTENT_SKIPPED_UNQUALIFIED_LEG symbol=%s expiry=%s "
                    "long_strike=%s short_strike=%s long_right=%s short_right=%s "
                    "long_conId=%s short_conId=%s qualify_error=%s",
                    intent.symbol, expiry, long_strike, short_strike,
                    long_right, short_right,
                    long_con_id, short_con_id,
                    str(qualify_exc) if qualify_exc is not None else "none",
                )
                raise ContractResolutionError(
                    "BAG_INTENT_SKIPPED_UNQUALIFIED_LEG "
                    f"symbol={intent.symbol} expiry={expiry} "
                    f"long_strike={long_strike} short_strike={short_strike} "
                    f"long_right={long_right} short_right={short_right} "
                    f"long_conId={long_con_id} short_conId={short_con_id}"
                )

        # Build BAG contract
        try:
            from ib_async import ComboLeg  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ConnectionError("ib_async is not installed") from exc

        bag = _Contract()
        bag.symbol = intent.symbol
        bag.secType = "BAG"
        bag.currency = currency
        bag.exchange = exchange

        long_leg = ComboLeg()
        long_leg.conId = long_con_id
        long_leg.ratio = 1
        long_leg.action = "BUY"
        long_leg.exchange = exchange

        short_leg = ComboLeg()
        short_leg.conId = short_con_id
        short_leg.ratio = 1
        short_leg.action = "SELL"
        short_leg.exchange = exchange

        bag.comboLegs = [long_leg, short_leg]

        summary = {
            "symbol": intent.symbol,
            "sec_type": "BAG",
            "spread_type": meta.get("spread_type", "UNKNOWN"),
            "expiry": expiry,
            "long_strike": long_strike,
            "short_strike": short_strike,
            "long_right": long_right,
            "short_right": short_right,
            "exchange": exchange,
            "currency": currency,
            "long_conId": long_con_id,
            "short_conId": short_con_id,
            "resolution": "qualified" if ib is not None else "unqualified",
        }
        return _ResolvedContract(contract=bag, summary=summary)


# ---------------------------------------------------------------------------
# BAG (combo) LMT discipline
# ---------------------------------------------------------------------------


def _resolve_bag_lmt_discipline(intent: NormalizedIntent) -> Optional[NormalizedIntent]:
    """Enforce LMT-only pricing for BAG (combo) intents before order build.

    Rules:
      A. If sec_type == "BAG" and order_type != "LMT", coerce to LMT and log
         BAG_MKT_COERCED_TO_LMT (with symbol + original order_type).
      B/C. If limit_price is missing or non-positive, hydrate from
         meta["net_debit_estimate"], else meta["spread_spec"].net_debit_estimate
         (typed OptionsSpreadSpec or Mapping). On success, log
         BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE.
      D. If no valid positive limit price can be derived, log
         BAG_INTENT_SKIPPED_NO_LIMIT_PRICE and return None to signal the
         caller to skip submission (mirrors the existing
         ``ibkr.skip_non_positive_size`` skip path).

    Non-BAG intents are returned unchanged.
    """
    if _safe_upper(intent.sec_type) != "BAG":
        return intent

    original_order_type = intent.order_type
    updates: Dict[str, Any] = {}

    if _safe_upper(original_order_type) != "LMT":
        updates["order_type"] = "LMT"
        LOGGER.warning(
            "BAG_MKT_COERCED_TO_LMT symbol=%s original_order_type=%s",
            intent.symbol,
            original_order_type,
        )

    def _positive_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out) or out <= 0.0:
            return None
        return out

    current_lp = _positive_float(intent.limit_price)
    resolved_lp: Optional[float] = current_lp

    if resolved_lp is None:
        meta = intent.meta if isinstance(intent.meta, Mapping) else {}
        candidates: List[Any] = [meta.get("net_debit_estimate")]
        spec = meta.get("spread_spec")
        if isinstance(spec, OptionsSpreadSpec):
            candidates.append(getattr(spec, "net_debit_estimate", None))
        elif isinstance(spec, Mapping):
            candidates.append(spec.get("net_debit_estimate"))

        for cand in candidates:
            hydrated = _positive_float(cand)
            if hydrated is not None:
                resolved_lp = hydrated
                updates["limit_price"] = hydrated
                LOGGER.info(
                    "BAG_LIMIT_PRICE_FROM_DEBIT_ESTIMATE symbol=%s limit_price=%s",
                    intent.symbol,
                    hydrated,
                )
                break

    if resolved_lp is None:
        LOGGER.warning(
            "BAG_INTENT_SKIPPED_NO_LIMIT_PRICE symbol=%s strategy=%s original_order_type=%s",
            intent.symbol,
            intent.strategy,
            original_order_type,
        )
        return None

    if updates:
        return replace(intent, **updates)
    return intent


# ---------------------------------------------------------------------------
# Order factory
# ---------------------------------------------------------------------------


class _OrderFactory:
    def __init__(self, config: IbkrConfig) -> None:
        self._config = config
        self._whole_units = frozenset(_safe_upper(x) for x in config.default_whole_unit_sec_types)
        self._fx_step = _to_decimal(config.default_forex_quantity_step, Decimal("0.0001")) or Decimal("0.0001")

    def build(self, intent: NormalizedIntent, *, what_if: bool) -> _PreparedOrder:
        _Contract, _Future, _Forex, Order, _Stock = _lazy_import_contract_classes()

        side = _side_to_ib(intent.side)
        order_type = _safe_upper(intent.order_type)
        if order_type not in {"MKT", "LMT"}:
            raise ValidationError(f"Unsupported order_type: {intent.order_type!r}")
        if order_type == "MKT" and not self._config.allow_market_orders:
            raise ValidationError("Market orders are disabled by configuration")

        quantity = self._normalize_quantity(intent.quantity, sec_type=intent.sec_type)
        if quantity <= 0:
            raise ValidationError("Quantity rounds down to zero or is non-positive")

        limit_price = _safe_float(intent.limit_price, default=float("nan"))
        if order_type == "LMT" and (math.isnan(limit_price) or limit_price <= 0.0):
            raise ValidationError("Limit order requires positive limit_price")

        order = Order()
        order.action = side
        order.orderType = order_type
        order.totalQuantity = quantity
        order.tif = _safe_upper(self._config.tif, "DAY")
        order.outsideRth = bool(self._config.outside_rth)
        order.whatIf = bool(what_if)
        if self._config.account:
            order.account = self._config.account
        if order_type == "LMT":
            snapped_price, applied_tick = _snap_lmt_price_to_tick(
                limit_price,
                side=side,
                sec_type=intent.sec_type,
                symbol=intent.symbol,
            )
            if applied_tick is not None and snapped_price != float(limit_price):
                LOGGER.info(
                    "LMT_PRICE_SNAPPED symbol=%s side=%s sec_type=%s "
                    "raw=%r snapped=%r tick=%s",
                    _safe_upper(intent.symbol),
                    side,
                    _safe_upper(intent.sec_type),
                    float(limit_price),
                    snapped_price,
                    str(applied_tick),
                )
            order.lmtPrice = float(snapped_price)

        return _PreparedOrder(order=order, quantity=quantity, what_if=what_if)

    def _normalize_quantity(self, raw_quantity: float, *, sec_type: str) -> float:
        q = _to_decimal(raw_quantity)
        if q is None or q <= 0:
            raise ValidationError("quantity must be > 0")

        normalized_sec_type = _safe_upper(sec_type)
        if normalized_sec_type in self._whole_units:
            q = _quantize_down(q, Decimal("1"))
        else:
            q = _quantize_down(q, self._fx_step)

        out = float(q)
        if out <= 0 or math.isnan(out) or math.isinf(out):
            raise ValidationError("quantity normalization produced invalid result")
        return out


# ---------------------------------------------------------------------------
# Broker-side open-order guard
# ---------------------------------------------------------------------------

# Statuses IBKR returns for orders that still occupy a working-order slot.
# `inactive` is included with a remaining-quantity check (see _is_working_open_trade).
_WORKING_OPEN_STATUSES = frozenset({
    "pendingsubmit",
    "presubmitted",
    "submitted",
    "apipending",
})

_INACTIVE_OPEN_STATUS = "inactive"


def _normalize_lmt_price(value: Any) -> Optional[float]:
    """Coerce a limit-price-like value to a finite float or None.

    ib_insync uses ~1.8e308 (UNSET_DOUBLE) for unset numeric Order fields, so
    treat any value above 1e100 as None to avoid mismatching MKT orders.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out) or out <= 0.0 or out > 1e100:
        return None
    return out


def _normalize_strike(value: Any) -> str:
    """Render a strike-like value as a stable string. Empty/zero collapse to ''."""
    raw = _safe_str(value, "")
    if not raw:
        return ""
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return raw
    if f == 0.0 or math.isnan(f) or math.isinf(f):
        return ""
    # Use repr-style truncation so 100 and 100.0 collapse to the same string.
    if f.is_integer():
        return str(int(f))
    return f"{f:.6f}".rstrip("0").rstrip(".")


def _project_lane_for_sec_type(
    sec_type: str,
    *,
    contract_month: str,
    strike: str,
    right: str,
) -> Tuple[str, str, str]:
    """Drop contract identity fields that do not apply to a given sec_type so
    that broker-side and intent-side keys collapse onto the same value when
    compared.
    """
    sec_type = sec_type.upper()
    if sec_type in ("STK", "CASH"):
        return ("", "", "")
    if sec_type == "FUT":
        return (contract_month, "", "")
    if sec_type == "BAG":
        # Combo legs are not exposed on the BAG itself; key on month only.
        return (contract_month, "", "")
    # OPT (or anything that carries strike/right) keeps all three.
    return (contract_month, strike, right)


def _open_trade_lane_key(trade: Any) -> Optional[Tuple[Any, ...]]:
    """Build the broker-side dedupe key for a single live ib.openTrades() trade.

    Returns None when the trade does not currently occupy a working-order slot,
    or the contract/order/status payload is malformed.
    """
    contract = getattr(trade, "contract", None)
    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    if contract is None or order is None or status is None:
        return None

    raw_status = _safe_str(getattr(status, "status", "")).lower()
    remaining = _safe_float(getattr(status, "remaining", 0.0))

    if raw_status in _WORKING_OPEN_STATUSES:
        pass
    elif raw_status == _INACTIVE_OPEN_STATUS and remaining > 0.0:
        # Inactive orders can still occupy a slot until IBKR finalises them.
        pass
    else:
        return None

    sec_type = _safe_upper(getattr(contract, "secType", ""))
    symbol = _safe_upper(getattr(contract, "symbol", ""))
    side = _safe_upper(getattr(order, "action", ""))
    exchange = _safe_upper(getattr(contract, "exchange", ""))
    currency = _safe_upper(getattr(contract, "currency", ""))
    contract_month = _safe_str(getattr(contract, "lastTradeDateOrContractMonth", ""))
    strike = _normalize_strike(getattr(contract, "strike", ""))
    right = _safe_upper(getattr(contract, "right", ""))
    contract_month, strike, right = _project_lane_for_sec_type(
        sec_type,
        contract_month=contract_month,
        strike=strike,
        right=right,
    )
    limit_price = _normalize_lmt_price(getattr(order, "lmtPrice", None))

    return (sec_type, symbol, side, exchange, currency, contract_month, strike, right, limit_price)


def _intent_lane_key(intent: NormalizedIntent, resolved_contract: _ResolvedContract, prepared: _PreparedOrder) -> Tuple[Any, ...]:
    """Build the broker-side dedupe key for the intent we are about to submit.

    Pulls identity from the resolved IBKR contract whenever available so the
    key matches the broker's own view (e.g. front-month FUT expiry filled in
    by qualifyContracts / contract details).
    """
    contract = getattr(resolved_contract, "contract", None)
    order = getattr(prepared, "order", None)

    sec_type = _safe_upper(getattr(contract, "secType", "") or intent.sec_type)
    symbol = _safe_upper(getattr(contract, "symbol", "") or intent.symbol)
    side = _safe_upper(getattr(order, "action", "") or intent.side)
    exchange = _safe_upper(getattr(contract, "exchange", "") or intent.exchange)
    currency = _safe_upper(getattr(contract, "currency", "") or intent.currency)

    contract_month = _safe_str(getattr(contract, "lastTradeDateOrContractMonth", ""))
    strike = _normalize_strike(getattr(contract, "strike", ""))
    right = _safe_upper(getattr(contract, "right", ""))
    contract_month, strike, right = _project_lane_for_sec_type(
        sec_type,
        contract_month=contract_month,
        strike=strike,
        right=right,
    )

    limit_price = _normalize_lmt_price(getattr(order, "lmtPrice", None) if order is not None else None)
    if limit_price is None:
        limit_price = _normalize_lmt_price(intent.limit_price)

    return (sec_type, symbol, side, exchange, currency, contract_month, strike, right, limit_price)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class IbkrAdapter:
    """
    Production-grade IBKR execution adapter using ib_insync.

    Supported submission styles
    ---------------------------
    * submit_routed_signals(signals)
      Legacy CHAD path that accepts RoutedSignal objects directly.

    * submit_strategy_trade_intents(intents)
      Higher-level path for StrategyTradeIntent-like objects created by the
      planning layer.

    Design notes
    ------------
    * DRY_RUN mode never places a live order.
    * Optional WHAT-IF in DRY_RUN mode can still connect to IBKR and request
      contract + commission/margin simulation.
    * Idempotency protects against duplicate live submissions.
    """

    def __init__(
        self,
        config: Optional[IbkrConfig] = None,
        *,
        ib_factory: Optional[IBFactory] = None,
        now_fn: Optional[NowFn] = None,
        margin_gate: Optional[MarginShadowGate] = None,
    ) -> None:
        self._config = config or IbkrConfig()
        self._ib_factory = ib_factory or _lazy_import_ib_factory()
        self._now_fn = now_fn or _utc_now
        # G3C Phase C: optional margin/BP gate wired at the pre-claim chokepoint. Default None
        # → byte-identical existing behavior. Production wires the shadow gate (see live_loop);
        # tests inject a gate (shadow or an injected-enforce config) to exercise both paths.
        self._margin_gate = margin_gate
        self._ib: Optional[IBLike] = None
        self._lock = threading.RLock()
        # L1-CLD: dedicated connection-owner event-loop thread. Created lazily
        # on first real connect. `_owner_loop_homed` is True only once THIS
        # adapter established the connection on that loop — an externally
        # pre-connected IB (legacy live_loop injection) is never re-homed.
        self._broker_loop: Optional[BrokerLoop] = None
        self._owner_loop_homed = False
        self._reader_stall_timeout_s = _READER_STALL_TIMEOUT_S
        self._reader_watchdog_interval_s = _READER_WATCHDOG_INTERVAL_S
        self._resolver = _ContractResolver(
            self._config, self._now_fn, broker_call=self._broker_call
        )
        self._order_factory = _OrderFactory(self._config)
        self._qualify_cache = _QualifyCache(
            ttl_seconds=_resolve_qualify_cache_ttl_seconds(),
            now_fn=self._now_fn,
        )
        self._idempotency = (
            _SQLiteIdempotencyStore(self._config.resolved_state_db_path())
            if self._config.enable_idempotency
            else None
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def ensure_connected(self, *, force: bool = False) -> None:
        """
        Ensure the adapter has an active IB connection.

        In pure DRY_RUN mode, this is a no-op unless `force=True` or the config
        explicitly opts into contract validation / WHAT-IF simulation.
        """
        if not _should_connect(self._config, force=force):
            LOGGER.info(
                "ibkr.dry_run_connection_skipped",
                extra={"host": self._config.host, "port": self._config.port},
            )
            return

        with self._lock:
            if self._ib is None:
                self._ib = self._ib_factory()
            ib = self._ib

            # ---- Connection-owner-loop path (L1-CLD) for a real ib_async IB --
            if _is_real_ib(ib):
                already = False
                try:
                    already = bool(ib.isConnected())
                except BaseException:  # noqa: BLE001
                    already = False
                if already and not self._owner_loop_homed:
                    # An externally pre-connected IB was injected (its socket +
                    # reader are bound to the caller's loop, e.g. live_loop's
                    # MainThread). Re-homing it onto the owner loop would strand
                    # the socket on a foreign loop; instead we adopt it as-is and
                    # leave owner-loop routing OFF, preserving today's behavior.
                    # (Activation homes the adapter's OWN execution connection —
                    # see PA L1_CLD_cross_loop_deadlock_fix_2026-07-08 §4.)
                    LOGGER.info(
                        "ibkr.connection_adopted_external",
                        extra={"owner_loop_homed": False},
                    )
                    return
                if self._owner_loop_homed and already:
                    return
                # Establish (or re-establish) the connection ON the owner loop.
                self._ensure_broker_loop()
                self._connect_on_owner_loop(ib)
                self._owner_loop_homed = True
                self._wire_reader_watchdog(ib)
                return

            # ---- Legacy sync path (sync-only IBLike / test fakes) ------------
            if ib.isConnected():
                return

            last_error: Optional[BaseException] = None
            for attempt in range(1, self._config.max_connect_retries + 1):
                try:
                    LOGGER.info(
                        "ibkr.connecting",
                        extra={
                            "host": self._config.host,
                            "port": self._config.port,
                            "client_id": self._config.client_id,
                            "attempt": attempt,
                        },
                    )
                    ib.connect(
                        self._config.host,
                        self._config.port,
                        clientId=self._config.client_id,
                        timeout=float(self._config.connect_timeout_s),
                    )
                    if ib.isConnected():
                        LOGGER.info(
                            "ibkr.connected",
                            extra={"accounts": list(ib.managedAccounts() or [])},
                        )
                        return
                except BaseException as exc:  # noqa: BLE001
                    last_error = exc
                    LOGGER.warning(
                        "ibkr.connection_attempt_failed",
                        extra={"attempt": attempt, "error": str(exc)},
                    )
                    if attempt < self._config.max_connect_retries:
                        time.sleep(self._config.retry_backoff_s * attempt)

            raise ConnectionError(
                f"Failed to connect to IBKR at {self._config.host}:{self._config.port}"
            ) from last_error

    # ------------------------------------------------------------------
    # Connection-owner loop (L1-CLD) — helpers
    # ------------------------------------------------------------------

    def _ensure_broker_loop(self) -> BrokerLoop:
        """Lazily create + start the dedicated connection-owner loop thread."""
        if self._broker_loop is None:
            self._broker_loop = BrokerLoop(name="chad-broker-loop")
        if not self._broker_loop.is_alive():
            self._broker_loop.start()
        return self._broker_loop

    def _owner_loop_active(self, ib: Any) -> bool:
        return bool(
            self._owner_loop_homed
            and self._broker_loop is not None
            and self._broker_loop.is_alive()
            and _is_real_ib(ib)
        )

    def _broker_call(
        self,
        ib: Any,
        name: str,
        *args: Any,
        timeout_s: float = _BROKER_TIMEOUT_S,
        label: Optional[str] = None,
    ) -> Any:
        """Single routing chokepoint for adapter broker calls.

        When the connection is homed on the owner loop, the call runs there —
        the async twin (``<name>Async``) via ``submit_coro``, or a sync-only
        ib_async method (e.g. ``openTrades``) via ``submit_call`` so it touches
        the IB object on its owning loop. Both are bounded + cancellable, so no
        call can hang uninterruptibly. Otherwise (sync fake / externally-adopted
        IB) the pre-existing bounded sync executor path is preserved exactly.

        Owner-loop timeouts / loop-down are translated to ``BrokerTimeoutError``
        so every caller's exception handling is unchanged.
        """
        label = label or name
        if self._owner_loop_active(ib):
            bl = self._broker_loop
            assert bl is not None
            try:
                async_meth = getattr(ib, name + "Async", None)
                if callable(async_meth):
                    return bl.submit_coro(
                        async_meth(*args), timeout_s=timeout_s, label=label
                    )
                sync_meth = getattr(ib, name)
                return bl.submit_call(
                    sync_meth, *args, timeout_s=timeout_s, label=label
                )
            except BrokerLoopTimeout as exc:
                LOGGER.error(
                    "ibkr.broker_call_timeout",
                    extra={"label": label, "timeout_s": timeout_s, "failure_class": "TIMEOUT"},
                )
                raise BrokerTimeoutError(
                    f"Broker call {label!r} exceeded {timeout_s}s liveness "
                    f"deadline — failure_class=TIMEOUT"
                ) from exc
            except BrokerLoopDown as exc:
                LOGGER.error(
                    "ibkr.broker_loop_down",
                    extra={"label": label, "failure_class": "BLOCKED"},
                )
                raise BrokerTimeoutError(
                    f"Broker owner loop down for {label!r} — failure_class=BLOCKED"
                ) from exc
        # Fallback (unchanged legacy behavior).
        return _call_with_timeout(getattr(ib, name), *args, timeout_s=timeout_s, label=label)

    def _connect_on_owner_loop(self, ib: Any) -> None:
        """Run ``connectAsync`` ON the owner loop, with bounded retries. The sync
        MainThread connect path is never used for an owner-loop-homed IB."""
        bl = self._ensure_broker_loop()
        connect_timeout = float(self._config.connect_timeout_s)
        submit_timeout = connect_timeout + _BROKER_TIMEOUT_S
        last_error: Optional[BaseException] = None
        for attempt in range(1, self._config.max_connect_retries + 1):
            try:
                LOGGER.info(
                    "ibkr.connecting",
                    extra={
                        "host": self._config.host,
                        "port": self._config.port,
                        "client_id": self._config.client_id,
                        "attempt": attempt,
                        "via": "owner_loop",
                    },
                )
                bl.submit_coro(
                    ib.connectAsync(
                        self._config.host,
                        self._config.port,
                        self._config.client_id,
                        connect_timeout,
                    ),
                    timeout_s=submit_timeout,
                    label="connectAsync",
                )
                if bool(bl.submit_call(ib.isConnected, timeout_s=_BROKER_TIMEOUT_S, label="isConnected")):
                    LOGGER.info(
                        "ibkr.connected",
                        extra={"via": "owner_loop"},
                    )
                    return
            except BaseException as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning(
                    "ibkr.connection_attempt_failed",
                    extra={"attempt": attempt, "error": str(exc), "via": "owner_loop"},
                )
                if attempt < self._config.max_connect_retries:
                    time.sleep(self._config.retry_backoff_s * attempt)
        raise ConnectionError(
            f"Failed to connect to IBKR at {self._config.host}:{self._config.port} "
            f"via connection-owner loop"
        ) from last_error

    def _wire_reader_watchdog(self, ib: Any) -> None:
        """Hook the socket-reader event to feed the owner loop's reader-progress
        watchdog, then arm the watchdog (forced reconnect on a stall)."""
        bl = self._broker_loop
        if bl is None:
            return

        def _on_read(*_a: Any, **_k: Any) -> None:
            try:
                bl.mark_reader_progress()
            except BaseException:  # noqa: BLE001
                pass

        try:
            # `ib.client.conn` is created once (persists across reconnects), and
            # `conn.hasData` fires on every raw socket read — the most reliable
            # reader-progress signal. Fall back to `ib.updateEvent`.
            conn = getattr(getattr(ib, "client", None), "conn", None)
            has_data = getattr(conn, "hasData", None)
            if has_data is not None:
                has_data += _on_read
            else:
                upd = getattr(ib, "updateEvent", None)
                if upd is not None:
                    upd += _on_read
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning("ibkr.reader_hook_wire_failed", extra={"error": str(exc)})

        async def _reconnect() -> None:
            # Runs ON the owner loop: force disconnect + reconnect so the socket
            # is re-bound to the owner loop.
            LOGGER.warning("ibkr.reader_stalled_reconnect", extra={"via": "owner_loop"})
            try:
                ib.disconnect()
            except BaseException:  # noqa: BLE001
                pass
            await asyncio.sleep(0.1)
            await ib.connectAsync(
                self._config.host,
                self._config.port,
                self._config.client_id,
                float(self._config.connect_timeout_s),
            )

        try:
            bl.attach_watchdog(
                reconnect=_reconnect,
                stall_timeout_s=float(self._reader_stall_timeout_s),
                interval_s=float(self._reader_watchdog_interval_s),
            )
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning("ibkr.reader_watchdog_attach_failed", extra={"error": str(exc)})

    async def _place_and_wait_async(
        self, ib: Any, qualified_contract: Any, prepared: "_PreparedOrder", idempotency_key: str
    ) -> Any:
        """Owner-loop coroutine mirror of the sync ``_place_and_wait``: place the
        order, pump the reader via ``asyncio.sleep`` (never ``ib.sleep`` on a
        foreign loop), install the status handler, and bounded-wait for a
        terminal status."""
        t = ib.placeOrder(qualified_contract, prepared.order)
        initial_wait = float(self._config.initial_status_wait_s)
        if initial_wait > 0.0:
            await asyncio.sleep(initial_wait)
        self._install_trade_status_handler(t, idempotency_key)
        await self._await_terminal_status_async(
            ib, t, timeout_s=float(self._config.terminal_wait_s)
        )
        return t

    async def _await_terminal_status_async(
        self, ib: Any, trade: Any, *, timeout_s: float
    ) -> None:
        """Async twin of ``_await_terminal_status`` — runs on the owner loop so
        the reader keeps advancing while it waits."""
        if timeout_s <= 0.0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        step = 0.1
        while loop.time() < deadline:
            status_str = ""
            try:
                order_status = getattr(trade, "orderStatus", None)
                status_str = _safe_str(getattr(order_status, "status", ""), "")
            except BaseException:  # noqa: BLE001
                status_str = ""
            klass = _classify_idempotency_status(status_str)
            if klass in ("terminal_positive", "terminal_negative"):
                return
            await asyncio.sleep(step)

    def shutdown(self) -> None:
        """Disconnect cleanly; safe to call multiple times."""
        with self._lock:
            ib = self._ib
            bl = self._broker_loop
            if ib is not None:
                try:
                    if self._owner_loop_homed and bl is not None and bl.is_alive():
                        # Disconnect ON the owner loop that owns the socket.
                        bl.submit_call(ib.disconnect, timeout_s=_BROKER_TIMEOUT_S, label="disconnect")
                    elif ib.isConnected():
                        ib.disconnect()
                except BaseException as exc:  # noqa: BLE001
                    LOGGER.warning("ibkr.disconnect_error", extra={"error": str(exc)})
            if bl is not None:
                try:
                    bl.stop()
                except BaseException as exc:  # noqa: BLE001
                    LOGGER.warning("ibkr.broker_loop_stop_error", extra={"error": str(exc)})
            self._broker_loop = None
            self._owner_loop_homed = False
            self._ib = None

    # ------------------------------------------------------------------
    # Public submission APIs
    # ------------------------------------------------------------------

    def submit_routed_signals(self, signals: Iterable[RoutedSignal]) -> List[SubmittedOrder]:
        submitted: List[SubmittedOrder] = []
        for routed in signals:
            try:
                intent = self._intent_from_routed_signal(routed)
                result = self._submit_intent(intent)
                if result is not None:
                    submitted.append(result)
            except BaseException as exc:  # noqa: BLE001
                LOGGER.exception(
                    "ibkr.submit_routed_signal_failed",
                    extra={
                        "symbol": getattr(routed, "symbol", ""),
                        "side": _enum_value(getattr(routed, "side", "")),
                        "net_size": _safe_float(getattr(routed, "net_size", 0.0)),
                        "error": str(exc),
                    },
                )
                submitted.append(
                    self._build_error_result_from_routed_signal(routed, exc)
                )
        return submitted

    def submit_strategy_trade_intents(self, intents: Iterable["StrategyTradeIntent"]) -> List[SubmittedOrder]:
        submitted: List[SubmittedOrder] = []
        for raw_intent in intents:
            try:
                intent = self._intent_from_trade_intent(raw_intent)
                result = self._submit_intent(intent)
                if result is not None:
                    submitted.append(result)
            except BaseException as exc:  # noqa: BLE001
                LOGGER.exception(
                    "ibkr.submit_trade_intent_failed",
                    extra={"symbol": getattr(raw_intent, "symbol", ""), "error": str(exc)},
                )
                submitted.append(self._build_error_result_from_trade_intent(raw_intent, exc))
        return submitted

    # ------------------------------------------------------------------
    # Intent normalization
    # ------------------------------------------------------------------

    def _intent_from_routed_signal(self, routed: RoutedSignal) -> NormalizedIntent:
        symbol = _normalize_symbol(getattr(routed, "symbol", ""))
        side = _side_to_ib(getattr(routed, "side", ""))
        quantity = _safe_float(getattr(routed, "net_size", 0.0))
        if quantity <= 0.0:
            raise ValidationError("net_size must be > 0")

        asset_class = _asset_class_key(getattr(routed, "asset_class", "unknown"))
        strategies = tuple(_strategy_name(s) for s in (getattr(routed, "source_strategies", ()) or ()))
        if not strategies:
            strategies = ("unknown",)

        created_at = getattr(routed, "created_at", None)
        if not isinstance(created_at, datetime):
            created_at = self._now_fn()

        meta = dict(getattr(routed, "meta", {}) or {})

        if asset_class in {"equity", "etf"}:
            sec_type = "STK"
            exchange = _safe_upper(meta.get("exchange"), self._config.default_stock_exchange)
            currency = _safe_upper(meta.get("currency"), self._config.default_stock_currency)
        elif asset_class == "futures":
            spec = self._config.futures_contracts.get(symbol)
            if spec is None:
                raise ValidationError(f"Unsupported futures symbol: {symbol!r}")
            sec_type = "FUT"
            exchange = spec.exchange
            currency = spec.currency
            meta = {**meta, "multiplier": meta.get("multiplier") or spec.multiplier}
        elif asset_class in {"forex", "cash"}:
            pair, base, quote = _parse_fx_pair(symbol)
            symbol = pair
            sec_type = "CASH"
            exchange = _safe_upper(meta.get("exchange"), self._config.default_forex_exchange)
            currency = quote
            meta = {**meta, "base_currency": base, "quote_currency": quote}
        elif asset_class == "options":
            sec_type = _safe_upper(meta.get("sec_type"), "OPT")
            exchange = _safe_upper(meta.get("exchange"), "SMART")
            currency = _safe_upper(meta.get("currency"), "USD")
            if meta.get("net_debit_estimate") is not None:
                meta = {**meta, "limit_price": meta["net_debit_estimate"]}
        else:
            raise ValidationError(f"Unsupported asset_class for IBKR adapter: {asset_class!r}")

        return NormalizedIntent(
            strategy=strategies[0],
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            side=side,
            order_type=_safe_upper(meta.get("order_type"), "MKT"),
            quantity=quantity,
            notional_estimate=_safe_float(meta.get("notional_estimate"), 0.0),
            asset_class=asset_class,
            source_strategies=strategies,
            created_at=created_at,
            limit_price=_safe_float(meta.get("limit_price"), float("nan")),
            meta=meta,
        )

    def _intent_from_trade_intent(self, raw_intent: "StrategyTradeIntent") -> NormalizedIntent:
        symbol = _normalize_symbol(getattr(raw_intent, "symbol", ""))
        sec_type = _safe_upper(getattr(raw_intent, "sec_type", ""))
        side = _side_to_ib(getattr(raw_intent, "side", ""))
        order_type = _safe_upper(getattr(raw_intent, "order_type", "MKT"))
        quantity = _safe_float(getattr(raw_intent, "quantity", 0.0))
        if quantity <= 0.0:
            raise ValidationError("quantity must be > 0")

        exchange = _safe_upper(getattr(raw_intent, "exchange", ""))
        currency = _safe_upper(getattr(raw_intent, "currency", ""))
        strategy = _safe_str(getattr(raw_intent, "strategy", "unknown")) or "unknown"

        asset_class = resolve_asset_class(symbol, sec_type)

        limit_price_raw = getattr(raw_intent, "limit_price", None)
        limit_price = None if limit_price_raw is None else _safe_float(limit_price_raw, float("nan"))

        raw_meta = getattr(raw_intent, "meta", None)
        meta: Dict[str, Any] = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}

        # IBKR Error 321 fix: ensure STK intents always have an exchange and
        # currency before they reach the resolver. Some intent producers
        # (e.g. position_reconciler close intents) construct minimal objects
        # that omit these fields entirely. Preserve explicit values; fall
        # back to intent.meta; finally default to SMART / USD. Futures and
        # other sec_types are handled by their own resolvers and must not
        # be touched here so contract_month safety is preserved.
        if sec_type == "STK":
            if not exchange:
                exchange = _safe_upper(meta.get("exchange")) or "SMART"
            if not currency:
                currency = _safe_upper(meta.get("currency")) or "USD"

        return NormalizedIntent(
            strategy=strategy,
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency,
            side=side,
            order_type=order_type,
            quantity=quantity,
            notional_estimate=_safe_float(getattr(raw_intent, "notional_estimate", 0.0)),
            asset_class=asset_class,
            source_strategies=(strategy,),
            created_at=self._now_fn(),
            limit_price=limit_price,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _evaluate_margin_gate(
        self, intent: NormalizedIntent, idempotency_key: str, now: datetime
    ) -> Optional[SubmittedOrder]:
        """Evaluate the margin/BP gate for ``intent`` at the pre-claim chokepoint.

        Returns ``None`` to proceed (ALWAYS, in shadow) or, only in ENFORCE mode on a real
        BLOCK, a ``SubmittedOrder(status="margin_blocked")`` — returned BEFORE the idempotency
        claim so a blocked order writes no idempotency row. The gate's own ``evaluate`` never
        raises; this wrapper additionally fails OPEN in shadow (proceed) and CLOSED in enforce
        on the extreme edge that view-building/should_block itself raises (task constraint 1)."""
        gate = self._margin_gate
        if gate is None:
            return None
        try:
            order_view = order_view_from_intent(intent, order_id=idempotency_key)
            verdict = gate.evaluate(order_view, now_epoch=now.timestamp())
            if not gate.should_block(verdict):
                return None
            LOGGER.warning(
                "ibkr.margin_gate_blocked",
                extra={
                    "symbol": intent.symbol,
                    "reason": verdict.reason,
                    "detail": verdict.detail,
                    "mode": verdict.mode,
                    "idempotency_key": idempotency_key,
                },
            )
            return SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                strategy=list(intent.source_strategies),
                dry_run=self._config.dry_run,
                submitted_at=now,
                ib_order_id=None,
                status="margin_blocked",
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                what_if=False,
                idempotency_key=idempotency_key,
                raw={"margin_shadow": verdict.to_dict()},
            )
        except Exception as exc:  # noqa: BLE001 - gate wiring must never break the order path
            enforce = False
            try:
                enforce = bool(gate.config.is_enforce)
            except Exception:  # noqa: BLE001
                enforce = False
            LOGGER.error(
                "ibkr.margin_gate_error",
                extra={"symbol": intent.symbol, "error": str(exc), "enforce": enforce},
            )
            if not enforce:
                return None  # SHADOW fail-open: never stop trading
            # ENFORCE fail-closed on an internal wiring error.
            return SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                strategy=list(intent.source_strategies),
                dry_run=self._config.dry_run,
                submitted_at=now,
                ib_order_id=None,
                status="margin_blocked",
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                what_if=False,
                idempotency_key=idempotency_key,
                raw={"margin_gate_error": f"{type(exc).__name__}: {exc}"},
            )

    def _evaluate_rth_gate(
        self, intent: NormalizedIntent, idempotency_key: str, now: datetime
    ) -> Optional[SubmittedOrder]:
        """WKF U2: block equity/ETF intents submitted outside regular trading hours.

        Returns ``None`` to proceed (gate off, non-equity/ETF asset, or market
        open) or a ``SubmittedOrder(status="market_closed")`` BLOCK returned
        BEFORE the idempotency claim, so a blocked order writes NO idempotency
        row. Futures/crypto lanes are exempt (their sessions differ; crypto
        never reaches this adapter). Fails OPEN on any internal wiring error —
        the gate is an off-hours safety net, not a hard risk control, and must
        never trap a legitimate in-session order.
        """
        try:
            from chad.execution.rth_gate import rth_block_reason, RTH_BLOCK_STATUS

            asset_class = resolve_asset_class(
                getattr(intent, "symbol", ""), getattr(intent, "sec_type", "")
            )
            reason = rth_block_reason(asset_class, now, os.environ)
            if reason is None:
                return None
            _strategy = "/".join(intent.source_strategies) if intent.source_strategies else ""
            LOGGER.warning(
                "RTH_GATE_BLOCK symbol=%s strategy=%s asset_class=%s sec_type=%s side=%s "
                "qty=%s now_utc=%s reason=%s — equity/ETF submit outside regular trading "
                "hours; no order placed, no idempotency row",
                intent.symbol, _strategy, asset_class, intent.sec_type, intent.side,
                intent.quantity,
                now.isoformat() if hasattr(now, "isoformat") else now,
                reason,
            )
            return SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                strategy=list(intent.source_strategies),
                dry_run=self._config.dry_run,
                submitted_at=now,
                ib_order_id=None,
                status=RTH_BLOCK_STATUS,
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                what_if=False,
                idempotency_key=idempotency_key,
                raw={"rth_gate": {"reason": reason, "asset_class": asset_class}},
            )
        except Exception as exc:  # noqa: BLE001 - gate wiring must never trap a live order
            LOGGER.warning("ibkr.rth_gate_error (fail-open): %r", exc)
            return None

    def _submit_intent(self, intent: NormalizedIntent) -> Optional[SubmittedOrder]:
        if intent.quantity <= 0.0:
            LOGGER.info(
                "ibkr.skip_non_positive_size",
                extra={"symbol": intent.symbol, "quantity": intent.quantity},
            )
            return None

        now = self._now_fn()
        idempotency_key = self._compute_idempotency_key(intent)

        # G3C Phase C: margin/BP gate at the pre-claim chokepoint (design v2.2 Part 7/8). In
        # SHADOW this only evaluates + logs a MARGIN_SHADOW marker + records evidence, and
        # ALWAYS returns None (proceed) — provably side-effect-free on the submission outcome.
        # In ENFORCE a real BLOCK returns here, BEFORE the idempotency claim below, so a
        # blocked order writes NO idempotency row.
        margin_blocked = self._evaluate_margin_gate(intent, idempotency_key, now)
        if margin_blocked is not None:
            return margin_blocked

        # WKF U2: market-hours (RTH) gate — same pre-claim chokepoint. Blocks
        # equity/ETF intents submitted outside regular trading hours BEFORE the
        # idempotency claim (no row for a blocked intent). Futures/crypto exempt.
        rth_blocked = self._evaluate_rth_gate(intent, idempotency_key, now)
        if rth_blocked is not None:
            return rth_blocked

        if self._idempotency:
            # GAP-036 (Phase-53): state-machine-aware claim. The ib_probe
            # callable is bound to the currently-connected IB instance (if
            # any). When the probe is unavailable (pure DRY_RUN paths) the
            # claim_or_reclaim() implementation fails closed on the stale
            # branch, preserving today's block-on-conflict semantics for
            # non-stale rows.
            def _probe_for_key(broker_order_id: Optional[int]) -> str:
                return self._ib_probe(self._ib, broker_order_id)

            claimed = self._idempotency.claim_or_reclaim(
                idempotency_key,
                self._intent_payload(intent),
                now,
                stale_threshold_s=float(self._config.stale_threshold_s),
                ib_probe=_probe_for_key,
                terminal_positive_ttl_s=self._resolve_terminal_positive_ttl_s(),
            )
            if not claimed:
                existing = self._idempotency.get(idempotency_key) or {}
                LOGGER.warning(
                    "ibkr.duplicate_submission_blocked",
                    extra={"symbol": intent.symbol, "idempotency_key": idempotency_key},
                )
                return SubmittedOrder(
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    strategy=list(intent.source_strategies),
                    dry_run=self._config.dry_run,
                    submitted_at=now,
                    ib_order_id=existing.get("broker_order_id"),
                    status="duplicate_blocked",
                    asset_class=intent.asset_class,
                    sec_type=intent.sec_type,
                    exchange=intent.exchange,
                    currency=intent.currency,
                    order_type=intent.order_type,
                    what_if=False,
                    idempotency_key=idempotency_key,
                    raw=existing,
                )

        needs_live_connection = not self._config.dry_run or self._config.simulate_what_if_in_dry_run
        needs_contract_validation = self._config.validate_contracts_in_dry_run or needs_live_connection
        ib: Optional[IBLike] = None
        if needs_live_connection or needs_contract_validation:
            self.ensure_connected(force=True)
            ib = self._ib

        resolved_contract = self._resolver.resolve(ib, intent)
        what_if = bool(self._config.dry_run and self._config.simulate_what_if_in_dry_run)

        disciplined_intent = _resolve_bag_lmt_discipline(intent)
        if disciplined_intent is None:
            # BAG intent has no positive limit_price after hydration attempts;
            # skip without building or previewing an order. The marker line
            # is emitted inside the helper.
            return None
        intent = disciplined_intent

        prepared = self._order_factory.build(intent, what_if=what_if)

        if self._config.dry_run and not self._config.simulate_what_if_in_dry_run:
            result = SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=prepared.quantity,
                strategy=list(intent.source_strategies),
                dry_run=True,
                submitted_at=now,
                ib_order_id=None,
                status="dry_run",
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                limit_price=intent.limit_price,
                what_if=False,
                idempotency_key=idempotency_key,
                contract_summary=resolved_contract.summary,
                raw={"intent": _jsonable(self._intent_payload(intent))},
            )
            LOGGER.info(
                "ibkr.dry_run_order",
                extra={
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "quantity": prepared.quantity,
                    "strategies": list(intent.source_strategies),
                    "sec_type": intent.sec_type,
                    "exchange": intent.exchange,
                },
            )
            LOGGER.info(
                "EXECUTION_RESULT",
                extra={
                    "symbol": intent.symbol,
                    "sec_type": intent.sec_type,
                    "exchange": intent.exchange,
                    "side": intent.side,
                    "quantity": prepared.quantity,
                    "status": "dry_run",
                    "classification": "DRY_RUN",
                    "error": None,
                    "strategy": intent.strategy,
                    "ts_utc": now.isoformat(),
                },
            )
            return result

        if ib is None:
            raise SubmissionError("Internal error: connected IB client is required for submission")

        result = self._submit_via_ib(
            ib=ib,
            intent=intent,
            resolved_contract=resolved_contract,
            prepared=prepared,
            submitted_at=now,
            idempotency_key=idempotency_key,
        )
        if self._idempotency:
            self._idempotency.mark(
                idempotency_key,
                status=result.status,
                broker_order_id=result.ib_order_id,
                result=_jsonable(asdict(result)),
                now=self._now_fn(),
            )
        return result

    def _submit_via_ib(
        self,
        *,
        ib: IBLike,
        intent: NormalizedIntent,
        resolved_contract: _ResolvedContract,
        prepared: _PreparedOrder,
        submitted_at: datetime,
        idempotency_key: str,
    ) -> SubmittedOrder:
        # --- Futures execution off-switch: HARD GATE at the broker chokepoint ---
        # When any futures-disable flag is set, hard-block EVERY futures order
        # (FUT/FOP, BOTH sides, ALL intent classes incl. exit/flip — this layer
        # has no notion of exit/flip, so there is no carve-out) before it can
        # reach the broker. Fail-closed: never call placeOrder/whatIfOrder for a
        # gated futures contract; return a clean rejected SubmittedOrder
        # mirroring the open-order-guard reject shape (never raise on the order
        # path). Equities/options/crypto/forex are unaffected. The predicate is
        # the single shared source of truth (chad.execution.futures_gate); the
        # live_loop early skip consults the SAME predicate. This is additive
        # defense-in-depth on top of the Bug B Fix A per-symbol position cap —
        # neither replaces the other.
        if futures_execution_disabled(os.environ) and is_futures_sec_type(intent.sec_type):
            _intent_class = ""
            try:
                _meta = intent.meta if isinstance(intent.meta, Mapping) else {}
                _intent_class = str(
                    _meta.get("intent")
                    or _meta.get("intent_class")
                    or _meta.get("reason")
                    or ""
                ).strip()
            except Exception:  # noqa: BLE001
                _intent_class = ""
            LOGGER.warning(
                "FUTURES_EXECUTION_GATE_BLOCKED symbol=%s side=%s sec_type=%s "
                "intent_class=%s strategy=%s reason=futures_execution_disabled",
                intent.symbol,
                intent.side,
                intent.sec_type,
                _intent_class or "n/a",
                intent.strategy,
            )
            return SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=prepared.quantity,
                strategy=list(intent.source_strategies),
                dry_run=False,
                submitted_at=submitted_at,
                ib_order_id=None,
                status="futures_execution_disabled",
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                limit_price=intent.limit_price,
                what_if=False,
                idempotency_key=idempotency_key,
                contract_summary=resolved_contract.summary,
                raw={
                    "guard": "futures_execution_disabled",
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "sec_type": intent.sec_type,
                    "intent_class": _intent_class or "n/a",
                },
            )

        last_exc: Optional[BaseException] = None

        for attempt in range(1, self._config.max_submit_retries + 1):
            try:
                qualified_contract = self._qualify_if_possible(ib, resolved_contract.contract)
                if prepared.what_if:
                    what_if_order = self._broker_call(ib, "whatIfOrder", qualified_contract, prepared.order, label="whatIfOrder")
                    raw = {"what_if_order": _jsonable(what_if_order)}
                    LOGGER.info(
                        "EXECUTION_RESULT",
                        extra={
                            "symbol": intent.symbol,
                            "sec_type": intent.sec_type,
                            "exchange": intent.exchange,
                            "side": intent.side,
                            "quantity": prepared.quantity,
                            "status": "what-if",
                            "classification": "WHAT_IF",
                            "error": None,
                            "strategy": intent.strategy,
                            "ts_utc": submitted_at.isoformat(),
                        },
                    )
                    return SubmittedOrder(
                        symbol=intent.symbol,
                        side=intent.side,
                        quantity=prepared.quantity,
                        strategy=list(intent.source_strategies),
                        dry_run=True,
                        submitted_at=submitted_at,
                        ib_order_id=int(getattr(what_if_order, "orderId", 0) or 0),
                        status="what-if",
                        asset_class=intent.asset_class,
                        sec_type=intent.sec_type,
                        exchange=intent.exchange,
                        currency=intent.currency,
                        order_type=intent.order_type,
                        limit_price=intent.limit_price,
                        what_if=True,
                        idempotency_key=idempotency_key,
                        contract_summary=resolved_contract.summary,
                        raw=raw,
                    )

                guard_result = self._enforce_open_order_guard(
                    ib=ib,
                    intent=intent,
                    resolved_contract=resolved_contract,
                    prepared=prepared,
                    submitted_at=submitted_at,
                    idempotency_key=idempotency_key,
                )
                if guard_result is not None:
                    return guard_result

                LOGGER.info(
                    "ibkr.place_order",
                    extra={
                        "symbol": intent.symbol,
                        "side": intent.side,
                        "quantity": prepared.quantity,
                        "strategies": list(intent.source_strategies),
                        "sec_type": intent.sec_type,
                        "exchange": intent.exchange,
                    },
                )
                # GAP-036 (Phase-53): bounded wait for terminal orderStatus,
                # with a per-Trade statusEvent handler that promotes the
                # idempotency row on every transition. The handler ONLY
                # writes the idempotency row — it never writes paper-fill
                # evidence (ISSUE-29 single-shot persist path is preserved
                # at chad/core/live_loop.py: _should_persist_paper_evidence).
                def _place_and_wait() -> Any:
                    t = ib.placeOrder(qualified_contract, prepared.order)
                    # Initial pump so the first status event is visible.
                    initial_wait = float(self._config.initial_status_wait_s)
                    if initial_wait > 0.0:
                        ib.sleep(initial_wait)
                    self._install_trade_status_handler(t, idempotency_key)
                    self._await_terminal_status(
                        ib,
                        t,
                        timeout_s=float(self._config.terminal_wait_s),
                    )
                    return t
                if self._owner_loop_active(ib):
                    # Owner-loop path: place + wait as a coroutine on the loop
                    # that owns the socket. Bounded liveness deadline > the
                    # internal initial+terminal waits so it never pre-empts them.
                    place_timeout = (
                        float(self._config.initial_status_wait_s)
                        + float(self._config.terminal_wait_s)
                        + _BROKER_TIMEOUT_S
                    )
                    try:
                        trade = self._broker_loop.submit_coro(  # type: ignore[union-attr]
                            self._place_and_wait_async(
                                ib, qualified_contract, prepared, idempotency_key
                            ),
                            timeout_s=place_timeout,
                            label="placeOrder",
                        )
                    except (BrokerLoopTimeout, BrokerLoopDown) as exc:
                        LOGGER.error(
                            "ibkr.broker_call_timeout",
                            extra={"label": "placeOrder", "failure_class": "TIMEOUT"},
                        )
                        raise BrokerTimeoutError(
                            "Broker call 'placeOrder' exceeded liveness deadline "
                            "— failure_class=TIMEOUT"
                        ) from exc
                else:
                    trade = _call_with_timeout(_place_and_wait, label="placeOrder")

                order = getattr(trade, "order", None)
                order_status = getattr(trade, "orderStatus", None)
                fills = getattr(trade, "fills", []) or []
                commissions = getattr(trade, "commissionReport", []) or []

                raw = {
                    "order": _jsonable(order),
                    "order_status": _jsonable(order_status),
                    "fills": _jsonable(fills),
                    "commissions": _jsonable(commissions),
                    "trade": _jsonable(trade),
                }

                _live_status = _safe_str(getattr(order_status, "status", ""), "submitted")
                LOGGER.info(
                    "EXECUTION_RESULT",
                    extra={
                        "symbol": intent.symbol,
                        "sec_type": intent.sec_type,
                        "exchange": intent.exchange,
                        "side": intent.side,
                        "quantity": prepared.quantity,
                        "status": _live_status,
                        "classification": "SUBMITTED",
                        "error": None,
                        "strategy": intent.strategy,
                        "ts_utc": submitted_at.isoformat(),
                    },
                )

                return SubmittedOrder(
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=prepared.quantity,
                    strategy=list(intent.source_strategies),
                    dry_run=False,
                    submitted_at=submitted_at,
                    ib_order_id=int(getattr(order, "orderId", 0) or 0),
                    status=_safe_str(getattr(order_status, "status", ""), "submitted"),
                    asset_class=intent.asset_class,
                    sec_type=intent.sec_type,
                    exchange=intent.exchange,
                    currency=intent.currency,
                    order_type=intent.order_type,
                    limit_price=intent.limit_price,
                    what_if=False,
                    perm_id=(int(getattr(order, "permId", 0)) if getattr(order, "permId", None) else None),
                    idempotency_key=idempotency_key,
                    contract_summary=resolved_contract.summary,
                    raw=raw,
                )
            except BaseException as exc:  # noqa: BLE001
                last_exc = exc
                LOGGER.warning(
                    "ibkr.submit_attempt_failed",
                    extra={
                        "attempt": attempt,
                        "symbol": intent.symbol,
                        "error": str(exc),
                    },
                )
                if attempt < self._config.max_submit_retries:
                    time.sleep(self._config.retry_backoff_s * attempt)

        _fail_class = "TIMEOUT" if isinstance(last_exc, BrokerTimeoutError) else "FAILED"
        LOGGER.info(
            "EXECUTION_RESULT",
            extra={
                "symbol": intent.symbol,
                "sec_type": intent.sec_type,
                "exchange": intent.exchange,
                "side": intent.side,
                "quantity": prepared.quantity,
                "status": "error",
                "classification": _fail_class,
                "error": str(last_exc),
                "strategy": intent.strategy,
                "ts_utc": self._now_fn().isoformat(),
            },
        )
        raise SubmissionError(f"Failed to submit order for {intent.symbol}: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    # GAP-036 (Phase-53): submit→confirm lifecycle helpers
    # ------------------------------------------------------------------

    def _install_trade_status_handler(self, trade: Any, idempotency_key: str) -> None:
        """Register a statusEvent handler on the per-Trade object that calls
        ``_idempotency.mark()`` on every orderStatus transition.

        The handler ONLY writes the idempotency row. It does NOT write
        paper-fill evidence — the ISSUE-29 single-shot persist path in
        ``chad/core/live_loop.py:_should_persist_paper_evidence`` remains the
        sole writer.
        """
        if self._idempotency is None:
            return
        event = getattr(trade, "statusEvent", None)
        if event is None:
            return

        def _on_status(t: Any) -> None:
            try:
                if self._idempotency is None:
                    return
                order_status = getattr(t, "orderStatus", None)
                status_str = _safe_str(getattr(order_status, "status", ""), "")
                order_obj = getattr(t, "order", None)
                bro_id: Optional[int] = None
                try:
                    bro_id = int(getattr(order_obj, "orderId", 0) or 0) or None
                except (TypeError, ValueError):
                    bro_id = None
                result_payload: Dict[str, Any] = {
                    "order_status": _jsonable(order_status),
                    "order": _jsonable(order_obj),
                    "fills": _jsonable(getattr(t, "fills", []) or []),
                }
                self._idempotency.mark(
                    idempotency_key,
                    status=status_str,
                    broker_order_id=bro_id,
                    result=result_payload,
                    now=self._now_fn(),
                )
            except BaseException as exc:  # noqa: BLE001
                # Handler must never raise into the IB event loop.
                LOGGER.warning(
                    "ibkr.trade_status_handler_error",
                    extra={"key": idempotency_key, "error": str(exc)},
                )

        try:
            # ib_async events expose ``+=`` for subscription.
            event += _on_status
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning(
                "ibkr.trade_status_handler_register_failed",
                extra={"key": idempotency_key, "error": str(exc)},
            )

    def _await_terminal_status(
        self,
        ib: IBLike,
        trade: Any,
        *,
        timeout_s: float,
    ) -> None:
        """Bounded wait for ``trade.orderStatus.status`` to reach a terminal
        bucket (positive or negative). Short-circuits as soon as a terminal
        status is observed; falls through after ``timeout_s`` even if still
        non-terminal — the in-flight statusEvent handler will continue to
        promote the row when the real transition arrives.
        """
        if timeout_s <= 0.0:
            return
        deadline = time.monotonic() + float(timeout_s)
        sleep_fn = getattr(ib, "sleep", None)
        step = 0.1
        while time.monotonic() < deadline:
            status_str = ""
            try:
                order_status = getattr(trade, "orderStatus", None)
                status_str = _safe_str(getattr(order_status, "status", ""), "")
            except BaseException:  # noqa: BLE001
                status_str = ""
            klass = _classify_idempotency_status(status_str)
            if klass in ("terminal_positive", "terminal_negative"):
                return
            try:
                if callable(sleep_fn):
                    sleep_fn(step)
                else:
                    time.sleep(step)
            except BaseException:  # noqa: BLE001
                time.sleep(step)

    def _ib_probe(self, ib: Optional[IBLike], broker_order_id: Optional[int]) -> str:
        """Determine the live status of an order at IBKR for the
        ``claim_or_reclaim`` stale-reclaim branch.

        Return one of:
          - ``_PROBE_STILL_ACTIVE`` — order appears in openTrades/trades with
            a non-terminal orderStatus.
          - ``_PROBE_TERMINAL_PREFIX + <ib_status>`` — order found at IB and
            its orderStatus is already terminal (e.g. ``Filled`` / ``Cancelled``).
          - ``_PROBE_ABSENT`` — order has NO record in openTrades/trades AND
            no execution evidence in ib.fills() for the same orderId/permId.

        ABSENT-EXECUTION HARDENING (Phase-54 mandate):
          ABSENT is only returned after we have additionally checked
          ``ib.fills()`` (execDetails / Fill objects) for an execution whose
          orderId or permId matches ``broker_order_id``. If one is found,
          the order was actually filled — return TERMINAL_AT_BROKER:Filled
          so claim_or_reclaim promotes the row to terminal-positive
          (UNCONDITIONAL block) and refuses to reclaim. This closes the
          restart-gap double-submit hole where openTrades alone would lie.
        """
        if ib is None:
            # Without a live IB handle we cannot prove the order is gone.
            # Refuse to claim ABSENT — fail closed.
            return _PROBE_STILL_ACTIVE
        if broker_order_id is None or broker_order_id <= 0:
            # OPS-OMEGA-01 Pattern A: a stale non-terminal row with no
            # broker_order_id means no broker order was ever placed (the
            # row was written by a pre-submission short-circuit such as
            # ``duplicate_open_order`` / ``suppressed_open_orders_cap``,
            # or an orphaned ``claimed`` row from a crash). Treat as
            # ABSENT so the row can be reclaimed; without this, the row
            # is locked forever because the previous fail-closed default
            # bumped updated_at_utc on every probe.
            return _PROBE_ABSENT

        # ----- Step 1: scan openTrades / trades for the order. -----
        found_status: Optional[str] = None

        def _orderids(t: Any) -> Tuple[Optional[int], Optional[int]]:
            o = getattr(t, "order", None)
            try:
                oid = int(getattr(o, "orderId", 0) or 0) or None
            except (TypeError, ValueError):
                oid = None
            try:
                pid = int(getattr(o, "permId", 0) or 0) or None
            except (TypeError, ValueError):
                pid = None
            return oid, pid

        def _scan(trades_iterable: Iterable[Any]) -> Optional[str]:
            for t in trades_iterable or []:
                oid, pid = _orderids(t)
                if oid == broker_order_id or (pid is not None and pid == broker_order_id):
                    os_obj = getattr(t, "orderStatus", None)
                    return _safe_str(getattr(os_obj, "status", ""), "")
            return None

        try:
            open_fn = getattr(ib, "openTrades", None)
            if callable(open_fn):
                found_status = _scan(open_fn() or [])
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning(
                "ibkr.probe_open_trades_failed",
                extra={"broker_order_id": broker_order_id, "error": str(exc)},
            )

        if found_status is None:
            try:
                trades_fn = getattr(ib, "trades", None)
                if callable(trades_fn):
                    found_status = _scan(trades_fn() or [])
            except BaseException as exc:  # noqa: BLE001
                LOGGER.warning(
                    "ibkr.probe_trades_failed",
                    extra={"broker_order_id": broker_order_id, "error": str(exc)},
                )

        if found_status is not None:
            klass = _classify_idempotency_status(found_status)
            if klass in ("terminal_positive", "terminal_negative"):
                return _PROBE_TERMINAL_PREFIX + found_status
            return _PROBE_STILL_ACTIVE

        # ----- Step 2: ABSENT-execution hardening (Phase-54 mandate).
        # Before returning ABSENT, scan ib.fills() / executions for an
        # execution attached to this orderId/permId. If one exists, the
        # order WAS filled — promote to terminal-positive to block reclaim.
        try:
            fills_fn = getattr(ib, "fills", None)
            fills_iter: Iterable[Any] = []
            if callable(fills_fn):
                fills_iter = fills_fn() or []
            for f in fills_iter:
                execution = getattr(f, "execution", None)
                if execution is None:
                    continue
                try:
                    e_oid = int(getattr(execution, "orderId", 0) or 0) or None
                except (TypeError, ValueError):
                    e_oid = None
                try:
                    e_pid = int(getattr(execution, "permId", 0) or 0) or None
                except (TypeError, ValueError):
                    e_pid = None
                if e_oid == broker_order_id or (
                    e_pid is not None and e_pid == broker_order_id
                ):
                    # Execution exists for this order → it filled.
                    return _PROBE_TERMINAL_PREFIX + "Filled"
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning(
                "ibkr.probe_fills_failed",
                extra={"broker_order_id": broker_order_id, "error": str(exc)},
            )
            # Fail closed: if we could not scan fills, do NOT claim ABSENT.
            return _PROBE_STILL_ACTIVE

        return _PROBE_ABSENT

    # ------------------------------------------------------------------
    # Broker-side open-order guard
    # ------------------------------------------------------------------

    def _snapshot_open_trades(self, ib: IBLike) -> List[Any]:
        """Read-only snapshot of ib.openTrades(). Never cancels or modifies.

        Falls back to an empty list when the underlying IB client does not
        expose openTrades() or the call fails — the guard fails open in that
        case and the submission proceeds as before.
        """
        fn = getattr(ib, "openTrades", None)
        if not callable(fn):
            return []
        try:
            result = self._broker_call(ib, "openTrades", label="openTrades")
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning(
                "ibkr.open_trades_snapshot_failed",
                extra={"error": str(exc)},
            )
            return []
        return list(result or [])

    def _enforce_open_order_guard(
        self,
        *,
        ib: IBLike,
        intent: NormalizedIntent,
        resolved_contract: _ResolvedContract,
        prepared: _PreparedOrder,
        submitted_at: datetime,
        idempotency_key: str,
    ) -> Optional[SubmittedOrder]:
        """Block submission when broker already has a working order on the
        same lane, or when the per-lane working-order count would breach
        IBKR's working-order cap.

        Returns a SubmittedOrder with status `duplicate_open_order` or
        `suppressed_open_orders_cap` to short-circuit submission, or None to
        let the caller proceed to placeOrder.
        """
        if not self._config.enable_open_order_guard:
            return None

        snapshot = self._snapshot_open_trades(ib)
        if not snapshot:
            return None

        intent_key = _intent_lane_key(intent, resolved_contract, prepared)
        # Lane bucket for the cap check is the broker-truth (sec_type, symbol, side).
        intent_bucket = (intent_key[0], intent_key[1], intent_key[2])

        existing_keys: List[Tuple[Any, ...]] = []
        existing_order_ids: List[int] = []
        bucket_count = 0
        bucket_order_ids: List[int] = []

        for trade in snapshot:
            key = _open_trade_lane_key(trade)
            if key is None:
                continue
            existing_keys.append(key)

            order_obj = getattr(trade, "order", None)
            try:
                order_id = int(getattr(order_obj, "orderId", 0) or 0)
            except (TypeError, ValueError):
                order_id = 0
            if order_id:
                existing_order_ids.append(order_id)

            if (key[0], key[1], key[2]) == intent_bucket:
                bucket_count += 1
                if order_id:
                    bucket_order_ids.append(order_id)

        if intent_key in existing_keys:
            LOGGER.warning(
                "IBKR_OPEN_ORDER_DUPLICATE_BLOCKED symbol=%s sec_type=%s side=%s "
                "exchange=%s currency=%s contract_month=%s strike=%s right=%s "
                "limit_price=%s existing_order_ids=%s",
                intent_key[1], intent_key[0], intent_key[2],
                intent_key[3], intent_key[4], intent_key[5],
                intent_key[6], intent_key[7], intent_key[8],
                bucket_order_ids,
            )
            return SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=prepared.quantity,
                strategy=list(intent.source_strategies),
                dry_run=False,
                submitted_at=submitted_at,
                ib_order_id=None,
                status="duplicate_open_order",
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                limit_price=intent.limit_price,
                what_if=False,
                idempotency_key=idempotency_key,
                contract_summary=resolved_contract.summary,
                raw={
                    "guard": "duplicate_open_order",
                    "lane": list(intent_key),
                    "existing_order_ids": list(bucket_order_ids),
                },
            )

        cap = max(0, int(self._config.open_order_cap_per_lane))
        if cap > 0 and bucket_count >= cap:
            LOGGER.warning(
                "IBKR_OPEN_ORDER_CAP_BLOCKED symbol=%s sec_type=%s side=%s "
                "count=%d cap=%d existing_order_ids=%s",
                intent_bucket[1], intent_bucket[0], intent_bucket[2],
                bucket_count, cap, bucket_order_ids,
            )
            return SubmittedOrder(
                symbol=intent.symbol,
                side=intent.side,
                quantity=prepared.quantity,
                strategy=list(intent.source_strategies),
                dry_run=False,
                submitted_at=submitted_at,
                ib_order_id=None,
                status="suppressed_open_orders_cap",
                asset_class=intent.asset_class,
                sec_type=intent.sec_type,
                exchange=intent.exchange,
                currency=intent.currency,
                order_type=intent.order_type,
                limit_price=intent.limit_price,
                what_if=False,
                idempotency_key=idempotency_key,
                contract_summary=resolved_contract.summary,
                raw={
                    "guard": "suppressed_open_orders_cap",
                    "lane": list(intent_bucket),
                    "count": bucket_count,
                    "cap": cap,
                    "existing_order_ids": list(bucket_order_ids),
                },
            )

        return None

    def _qualify_if_possible(self, ib: IBLike, contract: Any) -> Any:
        # IBKR does not support reqContractDetails / qualifyContracts on BAG
        # (combo) contracts — only on atomic contracts (STK, FUT, OPT, CASH).
        # Combo legs are qualified individually inside _resolve_combo before
        # the BAG is constructed, so the BAG itself must be passed through
        # unchanged. Calling qualifyContracts on a BAG triggers IBKR
        # Error 321 ("BAG isn't supported for contract data request").
        sec_type = _safe_upper(getattr(contract, "secType", ""))
        if sec_type == "BAG":
            LOGGER.info(
                "IBKR_BAG_QUALIFY_SKIPPED symbol=%s sec_type=BAG legs=%d",
                getattr(contract, "symbol", ""),
                len(getattr(contract, "comboLegs", []) or []),
            )
            return contract
        cached = self._qualify_cache.get(contract)
        if cached is not None:
            LOGGER.info(
                "IBKR_QUALIFY_CACHE_HIT symbol=%s sec_type=%s",
                getattr(contract, "symbol", ""),
                getattr(contract, "secType", ""),
            )
            return cached
        try:
            qualified = list(self._broker_call(ib, "qualifyContracts", contract, label="qualifyContracts") or [])
        except BaseException as exc:  # noqa: BLE001
            LOGGER.warning("ibkr.qualify_contract_failed", extra={"error": str(exc)})
            return contract
        if qualified:
            result = qualified[0]
            self._qualify_cache.store(contract, result)
            LOGGER.info(
                "IBKR_QUALIFY_CACHE_STORE symbol=%s sec_type=%s",
                getattr(contract, "symbol", ""),
                getattr(contract, "secType", ""),
            )
            return result
        return contract

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------

    def _compute_idempotency_key(self, intent: NormalizedIntent) -> str:
        payload = self._stable_idempotency_payload(intent)
        return _hash_payload(payload)

    def _resolve_terminal_positive_ttl_s(self) -> Optional[float]:
        """Return the effective terminal-positive idempotency TTL in seconds.

        Honours the ``CHAD_DUPLICATE_CACHE_TTL_SECONDS`` env var as a runtime
        override; otherwise falls back to the value baked into the
        ``IbkrAdapterConfig``. A non-positive value disables the TTL and
        preserves the original "Filled blocks forever" semantics.
        """
        raw = os.environ.get("CHAD_DUPLICATE_CACHE_TTL_SECONDS")
        if raw is not None and raw.strip() != "":
            try:
                return float(raw)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "ibkr.duplicate_cache_ttl_env_invalid",
                    extra={"raw": raw},
                )
        return float(self._config.terminal_positive_ttl_s)

    @staticmethod
    def _stable_idempotency_payload(intent: NormalizedIntent) -> Mapping[str, Any]:
        """
        Stable subset of intent fields used to derive the idempotency key.

        Excludes timestamps (created_at), price-derived estimates
        (notional_estimate), attribution noise (source_strategies), and the
        free-form meta dict — any of which can change cycle-to-cycle for the
        same logical order and would otherwise produce a fresh hash on every
        loop, defeating duplicate-submit suppression and triggering IBKR
        Error 201.
        """
        meta = intent.meta or {}
        sec_type = (intent.sec_type or "").upper()
        asset_class = (intent.asset_class or "").lower()

        limit_price = intent.limit_price
        if limit_price is not None:
            try:
                if math.isnan(float(limit_price)) or math.isinf(float(limit_price)):
                    limit_price = None
            except (TypeError, ValueError):
                limit_price = None

        payload: Dict[str, Any] = {
            "strategy": intent.strategy,
            "symbol": intent.symbol,
            "sec_type": sec_type,
            "exchange": (intent.exchange or "").upper(),
            "currency": (intent.currency or "").upper(),
            "side": (intent.side or "").upper(),
            "order_type": (intent.order_type or "").upper(),
            "quantity": intent.quantity,
            "asset_class": asset_class,
            "limit_price": limit_price,
        }

        if sec_type == "BAG" or asset_class == "options_spread":
            payload["bag"] = {
                "expiry": _safe_str(
                    meta.get("expiry") or meta.get("lastTradeDateOrContractMonth")
                ),
                "long_strike": meta.get("long_strike"),
                "short_strike": meta.get("short_strike"),
                "long_right": _safe_upper(_safe_str(meta.get("long_right"))),
                "short_right": _safe_upper(_safe_str(meta.get("short_right"))),
            }
        elif sec_type == "OPT" or asset_class == "options":
            payload["option"] = {
                "expiry": _safe_str(
                    meta.get("expiry") or meta.get("lastTradeDateOrContractMonth")
                ),
                "strike": meta.get("strike"),
                "right": _safe_upper(_safe_str(meta.get("right"))),
            }
        elif sec_type == "FUT" or asset_class == "futures":
            payload["futures"] = {
                "contract_month": _safe_str(
                    meta.get("contract_month")
                    or meta.get("lastTradeDateOrContractMonth")
                ),
            }

        return payload

    @staticmethod
    def _intent_payload(intent: NormalizedIntent) -> Mapping[str, Any]:
        return {
            "strategy": intent.strategy,
            "symbol": intent.symbol,
            "sec_type": intent.sec_type,
            "exchange": intent.exchange,
            "currency": intent.currency,
            "side": intent.side,
            "order_type": intent.order_type,
            "quantity": intent.quantity,
            "notional_estimate": intent.notional_estimate,
            "asset_class": intent.asset_class,
            "source_strategies": list(intent.source_strategies),
            "created_at": intent.created_at.isoformat(),
            "limit_price": intent.limit_price,
            "meta": dict(intent.meta),
        }

    def _build_error_result_from_routed_signal(self, routed: RoutedSignal, exc: BaseException) -> SubmittedOrder:
        now = self._now_fn()
        symbol = _safe_upper(getattr(routed, "symbol", ""))
        side = _side_to_ib(getattr(routed, "side", "BUY")) if getattr(routed, "side", None) else "BUY"
        qty = _safe_float(getattr(routed, "net_size", 0.0))
        strategies = [_strategy_name(s) for s in (getattr(routed, "source_strategies", ()) or ())]
        return SubmittedOrder(
            symbol=symbol,
            side=side,
            quantity=qty,
            strategy=strategies,
            dry_run=self._config.dry_run,
            submitted_at=now,
            status="error",
            asset_class=_asset_class_key(getattr(routed, "asset_class", "unknown")),
            error=str(exc),
            raw={"exception": str(exc)},
        )

    def _build_error_result_from_trade_intent(self, raw_intent: "StrategyTradeIntent", exc: BaseException) -> SubmittedOrder:
        now = self._now_fn()
        symbol = _safe_upper(getattr(raw_intent, "symbol", ""))
        sec_type = _safe_upper(getattr(raw_intent, "sec_type", ""))
        return SubmittedOrder(
            symbol=symbol,
            side=_safe_upper(getattr(raw_intent, "side", "")),
            quantity=_safe_float(getattr(raw_intent, "quantity", 0.0)),
            strategy=[_safe_str(getattr(raw_intent, "strategy", "unknown"))],
            dry_run=self._config.dry_run,
            submitted_at=now,
            status="error",
            asset_class=resolve_asset_class(symbol, sec_type),
            sec_type=sec_type,
            exchange=_safe_upper(getattr(raw_intent, "exchange", "")),
            currency=_safe_upper(getattr(raw_intent, "currency", "")),
            order_type=_safe_upper(getattr(raw_intent, "order_type", "MKT")),
            error=str(exc),
            raw={"exception": str(exc)},
        )


__all__ = [
    "FuturesContractSpec",
    "IbkrConfig",
    "SubmittedOrder",
    "IbkrAdapter",
    "IbkrAdapterError",
    "ConnectionError",
    "ValidationError",
    "ContractResolutionError",
    "SubmissionError",
]

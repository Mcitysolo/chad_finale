
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

import hashlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, TypeVar, TYPE_CHECKING

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


def _call_with_timeout(fn: Callable[..., Any], *args: Any, timeout_s: float = _BROKER_TIMEOUT_S, label: str = "broker_call") -> Any:
    """
    Run *fn(*args)* in a daemon thread with a hard wall-clock timeout.

    On timeout: logs TIMEOUT classification and raises BrokerTimeoutError.
    On inner exception: re-raises as-is so the caller can classify.
    """
    result_box: list = []
    error_box: list = []

    def _target() -> None:
        try:
            result_box.append(fn(*args))
        except BaseException as exc:
            error_box.append(exc)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        LOGGER.error(
            "ibkr.broker_call_timeout",
            extra={"label": label, "timeout_s": timeout_s, "failure_class": "TIMEOUT"},
        )
        raise BrokerTimeoutError(
            f"Broker call {label!r} exceeded {timeout_s}s liveness deadline — failure_class=TIMEOUT"
        )

    if error_box:
        raise error_box[0]

    return result_box[0] if result_box else None


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
        with self._connect() as conn:
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
        with self._lock, self._connect() as conn:
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
        with self._lock, self._connect() as conn:
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
        with self._lock, self._connect() as conn:
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
        from ib_insync import Contract, Future, Forex, Order, Stock  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ConnectionError(
            "ib_insync is not installed. Install it inside the CHAD venv."
        ) from exc
    return Contract, Future, Forex, Order, Stock


def _lazy_import_option_class() -> Any:
    try:
        from ib_insync import Option  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ConnectionError(
            "ib_insync is not installed. Install it inside the CHAD venv."
        ) from exc
    return Option


# ---------------------------------------------------------------------------
# Contract resolution
# ---------------------------------------------------------------------------


class _ContractResolver:
    def __init__(self, config: IbkrConfig, now_fn: NowFn) -> None:
        self._config = config
        self._now_fn = now_fn
        self._cache: MutableMapping[str, _ResolvedContract] = {}

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
                qualified = ib.qualifyContracts(contract)
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
        expiry = _safe_str(meta.get("expiry"))
        if not expiry:
            raise ContractResolutionError(
                f"BAG contract for {intent.symbol} requires 'expiry' in intent.meta"
            )

        long_strike = meta.get("long_strike")
        short_strike = meta.get("short_strike")
        if long_strike is None or short_strike is None:
            raise ContractResolutionError(
                f"BAG contract for {intent.symbol} requires 'long_strike' and 'short_strike' in intent.meta"
            )
        try:
            long_strike = float(long_strike)
            short_strike = float(short_strike)
        except (TypeError, ValueError):
            raise ContractResolutionError(
                f"BAG contract for {intent.symbol}: invalid strike values"
            )

        long_right = _safe_upper(_safe_str(meta.get("long_right")))
        short_right = _safe_upper(_safe_str(meta.get("short_right")))
        if long_right not in ("C", "P") or short_right not in ("C", "P"):
            raise ContractResolutionError(
                f"BAG contract for {intent.symbol} requires 'long_right' and 'short_right' (C or P)"
            )

        exchange = _safe_str(meta.get("exchange")) or intent.exchange or "SMART"
        currency = intent.currency or "USD"

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
                qualified = _call_with_timeout(
                    ib.qualifyContracts,
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
            from ib_insync import ComboLeg  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ConnectionError("ib_insync is not installed") from exc

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
            order.lmtPrice = float(limit_price)

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
    ) -> None:
        self._config = config or IbkrConfig()
        self._ib_factory = ib_factory or _lazy_import_ib_factory()
        self._now_fn = now_fn or _utc_now
        self._ib: Optional[IBLike] = None
        self._lock = threading.RLock()
        self._resolver = _ContractResolver(self._config, self._now_fn)
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

            if self._ib.isConnected():
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
                    self._ib.connect(
                        self._config.host,
                        self._config.port,
                        clientId=self._config.client_id,
                        timeout=float(self._config.connect_timeout_s),
                    )
                    if self._ib.isConnected():
                        LOGGER.info(
                            "ibkr.connected",
                            extra={"accounts": list(self._ib.managedAccounts() or [])},
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

    def shutdown(self) -> None:
        """Disconnect cleanly; safe to call multiple times."""
        with self._lock:
            if self._ib is not None and self._ib.isConnected():
                try:
                    self._ib.disconnect()
                except BaseException as exc:  # noqa: BLE001
                    LOGGER.warning("ibkr.disconnect_error", extra={"error": str(exc)})
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

    def _submit_intent(self, intent: NormalizedIntent) -> Optional[SubmittedOrder]:
        if intent.quantity <= 0.0:
            LOGGER.info(
                "ibkr.skip_non_positive_size",
                extra={"symbol": intent.symbol, "quantity": intent.quantity},
            )
            return None

        now = self._now_fn()
        idempotency_key = self._compute_idempotency_key(intent)
        if self._idempotency:
            claimed = self._idempotency.claim(idempotency_key, self._intent_payload(intent), now)
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
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self._config.max_submit_retries + 1):
            try:
                qualified_contract = self._qualify_if_possible(ib, resolved_contract.contract)
                if prepared.what_if:
                    what_if_order = _call_with_timeout(ib.whatIfOrder, qualified_contract, prepared.order, label="whatIfOrder")
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
                def _place_and_wait() -> Any:
                    t = ib.placeOrder(qualified_contract, prepared.order)
                    ib.sleep(float(self._config.initial_status_wait_s))
                    return t
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
            qualified = list(_call_with_timeout(ib.qualifyContracts, contract, label="qualifyContracts") or [])
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

#!/usr/bin/env python3
"""
chad/market_data/ibkr_bar_provider.py

IBKR Native Bar Provider for CHAD.

Polls reqHistoricalData on a fixed interval (default 60s) instead of
subscribing to reqRealTimeBars — the latter requires paid market-data
entitlements and fails with Error 420 on paper accounts, while
reqHistoricalData works without subscriptions.

- Persistent IB session (injected, not created internally)
- Historical bar fetches via reqHistoricalData
- Polling loop that refreshes 1-minute bars every cache_interval seconds
- Per-symbol writes to data/bars/1m/{symbol}.json (consumer path)
- Aggregate file-based cache export (runtime/ibkr_bars_cache.json)
- In-memory LRU cache (deque per symbol, maxlen=500)
- Graceful fallback: cached bars returned on IBKR failure
- Symbol normalization: STK vs FUT sec_type auto-detection
- Auto-reconnect on IBKR disconnect
- Thread-safe cache writes

Operational behavior (current):
- IBKR historical bars are AUTHORITATIVE for ALL active symbols
  (equities, ETFs, and futures). The daemon writes IBKR bars for the
  full universe; Polygon is NOT in the active code path.

Bar Provider Modes (CHAD_BAR_PROVIDER env var):
- "ibkr":    IBKR for all instruments (current operational mode)
- "hybrid":  IBKR for futures; equities path delegates to IBKR in this
             build (legacy "Polygon for equities" wiring is retired)
- "polygon": legacy/opt-in only — requires explicit operator action.
             Not used in production. Will not run unless
             CHAD_BAR_PROVIDER=polygon is set deliberately.

Daemon mode:
    python3 -m chad.market_data.ibkr_bar_provider

Service mode:
    systemd unit chad-ibkr-bar-provider.service
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger("chad.market_data.ibkr_bar_provider")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
CACHE_OUTPUT_PATH = RUNTIME_DIR / "ibkr_bars_cache.json"
BARS_1M_DIR = REPO_ROOT / "data" / "bars" / "1m"
UNIVERSE_CONFIG_PATH = REPO_ROOT / "config" / "universe.json"

# Maximum bars per symbol in memory
MAX_BARS_PER_SYMBOL = 500

# Polling interval (seconds) — reqHistoricalData every N seconds per symbol
POLLING_INTERVAL_S = 60.0

# Delay between per-symbol requests to respect IBKR rate limits
# (50 requests per 10 seconds max)
PER_SYMBOL_DELAY_S = 0.5

# Known futures symbols -> contract specs
FUTURES_SYMBOLS = {
    "MES", "MNQ", "MCL", "MGC",    # Alpha/Gamma futures
    "ZN", "ZB", "M6E", "SIL",       # Omega Macro
    "MYM", "M2K",                   # Micro Dow / Micro Russell — gamma_futures + alpha_futures
}

# Full universe for bar subscriptions (fallback when config/universe.json is missing)
DEFAULT_UNIVERSE = [
    # Equities/ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    # Futures
    "MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL", "MYM", "M2K",
]


def _load_universe_from_config() -> List[str]:
    """Load the full trading universe (equities + futures) for bar subscriptions.

    Equity/ETF symbols come from the central universe provider, which prefers
    the live-screened runtime/universe.json over config/universe.json. Futures
    symbols continue to come straight from config/universe.json — runtime
    universe carries equities only, and the ``futures`` block is the only
    source of contract metadata. Falls back to DEFAULT_UNIVERSE on any failure.
    """
    try:
        from chad.utils.universe_provider import load_active_universe
        equity_syms = list(load_active_universe().symbols)
    except Exception:
        equity_syms = []

    futures_syms: List[str] = []
    try:
        if UNIVERSE_CONFIG_PATH.is_file():
            obj = json.loads(UNIVERSE_CONFIG_PATH.read_text(encoding="utf-8"))
            for entry in obj.get("futures", []) or []:
                if isinstance(entry, dict):
                    sym = str(entry.get("symbol", "")).strip().upper()
                    if sym:
                        futures_syms.append(sym)
    except Exception:
        futures_syms = []

    # Deduplicate, preserve order: equities first, then futures.
    seen = set()
    out: List[str] = []
    for s in equity_syms + futures_syms:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out or list(DEFAULT_UNIVERSE)


# ---------------------------------------------------------------------------
# Bar dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bar:
    """Standard OHLCV bar with source attribution."""
    ts_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    source: str = "ibkr"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_utc": self.ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_futures_symbol(symbol: str) -> bool:
    return symbol.strip().upper() in FUTURES_SYMBOLS


def _make_ib_contract(symbol: str, ib: Any = None) -> Any:
    """
    Create an ib_insync contract for the given symbol.

    Auto-detects futures vs stock based on known futures set.
    For futures, sets tradingClass to disambiguate micro vs full-size
    (SI vs SIL share root symbol but different tradingClass/multiplier),
    then uses reqContractDetails to resolve the front-month expiry.
    SIL maps to IBKR symbol SI (Silver futures) with tradingClass 'SIL'.
    """
    from ib_async import Stock, Future

    sym = symbol.strip().upper()

    if _is_futures_symbol(sym):
        # IBKR root-symbol mapping (SIL trades as SI + tradingClass=SIL)
        ibkr_sym_map = {
            "SIL": "SI",
        }
        # tradingClass prevents "ambiguous contract" (Error 200) when the
        # root symbol covers multiple product sizes on the same exchange.
        # Default: tradingClass == sym.
        trading_class_map = {
            "SIL": "SIL",
        }
        exchange_map = {
            "MES": "CME",
            "MNQ": "CME",
            "MCL": "NYMEX",
            "MGC": "COMEX",
            "ZN": "CBOT",
            "ZB": "CBOT",
            "M6E": "CME",
            "SIL": "COMEX",
            "SI": "COMEX",
            "MYM": "CBOT",
            "M2K": "CME",
        }
        ibkr_sym = ibkr_sym_map.get(sym, sym)
        exchange = exchange_map.get(sym, "CME")
        trading_class = trading_class_map.get(sym, sym)
        contract = Future(
            symbol=ibkr_sym,
            exchange=exchange,
            currency="USD",
            tradingClass=trading_class,
        )

        # Resolve front-month expiry via reqContractDetails.
        # Even with tradingClass set, IBKR returns every listed expiry; we
        # pick the earliest contract whose lastTradeDateOrContractMonth is
        # today or later. Falls back to qualifyContracts on any error.
        if ib is not None:
            try:
                details = ib.reqContractDetails(contract)
                if details:
                    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
                    future_contracts: List[tuple] = []
                    for d in details:
                        c = getattr(d, "contract", None)
                        if c is None:
                            continue
                        ltd = str(getattr(c, "lastTradeDateOrContractMonth", "") or "")
                        # ltd can be YYYYMMDD or YYYYMM; compare first 8/6 chars
                        if not ltd:
                            continue
                        if len(ltd) >= 8 and ltd[:8] >= today_str:
                            future_contracts.append((ltd, c))
                        elif len(ltd) == 6 and ltd >= today_str[:6]:
                            future_contracts.append((ltd, c))
                    if future_contracts:
                        future_contracts.sort(key=lambda x: x[0])
                        return future_contracts[0][1]
                qualified = ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
            except Exception:
                pass

        return contract
    else:
        return Stock(sym, "SMART", "USD")


# ---------------------------------------------------------------------------
# IBKRBarProvider
# ---------------------------------------------------------------------------

class IBKRBarProvider:
    """
    Manages IBKR bar data fetching and caching.

    Owns a persistent IB session reference (injected, not created).
    Thread-safe: cache reads/writes protected by a lock.
    """

    def __init__(
        self,
        ib: Any,  # ib_insync.IB instance
        *,
        universe: Optional[Sequence[str]] = None,
        max_bars_per_symbol: int = MAX_BARS_PER_SYMBOL,
        cache_output_path: Optional[Path] = None,
    ) -> None:
        self._ib = ib
        self._universe = [s.strip().upper() for s in (universe or DEFAULT_UNIVERSE)]
        self._max_bars = max_bars_per_symbol
        self._cache_path = cache_output_path or CACHE_OUTPUT_PATH
        self._lock = threading.Lock()

        # In-memory bar cache: symbol -> deque of Bar
        self._cache: Dict[str, Deque[Bar]] = {}
        for sym in self._universe:
            self._cache[sym] = deque(maxlen=self._max_bars)

        # Resolved IBKR contract cache: symbol -> ib_insync Contract
        # Contract resolution (reqContractDetails for futures front-month)
        # is expensive; cache after first lookup so polling cycles don't
        # re-resolve every 60 seconds.
        self._contracts: Dict[str, Any] = {}

    @property
    def universe(self) -> List[str]:
        return list(self._universe)

    @property
    def connected(self) -> bool:
        try:
            return bool(self._ib.isConnected())
        except Exception:
            return False

    # --------------------------
    # Historical bar fetch
    # --------------------------

    def fetch_historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "1 min",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> List[Bar]:
        """
        Fetch historical bars from IBKR for a single symbol.

        Falls back to cached bars on failure.

        Parameters
        ----------
        symbol : str
            Instrument symbol (e.g., "SPY", "MES").
        duration : str
            IBKR duration string (e.g., "2 D", "5 D", "1 W").
        bar_size : str
            IBKR bar size (e.g., "1 min", "5 mins", "1 day").
        what_to_show : str
            Data type (TRADES, MIDPOINT, BID, ASK).
        use_rth : bool
            Regular trading hours only.

        Returns
        -------
        List[Bar]
            Bars sorted by timestamp ascending.
        """
        sym = symbol.strip().upper()
        contract = self._get_contract(sym)

        try:
            ib_bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

            bars: List[Bar] = []
            for b in ib_bars:
                ts = str(getattr(b, "date", ""))
                o = _safe_float(getattr(b, "open", 0))
                h = _safe_float(getattr(b, "high", 0))
                l = _safe_float(getattr(b, "low", 0))
                c = _safe_float(getattr(b, "close", 0))
                v = _safe_float(getattr(b, "volume", 0))

                if min(o, h, l, c) <= 0 or h < l:
                    continue

                bars.append(Bar(
                    ts_utc=ts,
                    open=o, high=h, low=l, close=c,
                    volume=max(v, 0.0),
                    symbol=sym,
                    source="ibkr",
                ))

            # Update cache
            with self._lock:
                if sym not in self._cache:
                    self._cache[sym] = deque(maxlen=self._max_bars)
                self._cache[sym].clear()
                self._cache[sym].extend(bars[-self._max_bars:])

            LOGGER.info(
                "ibkr_bar_provider.historical_fetched",
                extra={"symbol": sym, "bars": len(bars)},
            )
            return bars

        except Exception as exc:
            LOGGER.warning(
                "ibkr_bar_provider.historical_failed_using_cache",
                extra={"symbol": sym, "error": str(exc)},
            )
            return self.get_latest_bars(sym)

    # --------------------------
    # Contract resolution (cached)
    # --------------------------

    def _get_contract(self, symbol: str) -> Any:
        """Resolve and cache an IBKR contract for the given symbol.

        Contract lookups (reqContractDetails for futures front-month
        resolution) are expensive; cache per symbol so polling does not
        re-resolve every cycle.
        """
        sym = symbol.strip().upper()
        contract = self._contracts.get(sym)
        if contract is None:
            contract = _make_ib_contract(sym, ib=self._ib)
            self._contracts[sym] = contract
        return contract

    # --------------------------
    # 1-minute bar file writer
    # --------------------------

    def write_1m_bars_file(
        self,
        symbol: str,
        bars: List[Bar],
        bars_1m_dir: Optional[Path] = None,
    ) -> Path:
        """Write bars to data/bars/1m/{symbol}.json atomically.

        Uses the same payload shape as ibkr_historical_provider so
        context_builder.py can consume the files interchangeably.
        """
        sym = symbol.strip().upper()
        out_dir = bars_1m_dir or BARS_1M_DIR
        out_path = out_dir / f"{sym}.json"
        payload = {
            "bars": [b.to_dict() for b in bars],
            "symbol": sym,
            "source": "ibkr",
            "timeframe": "1m",
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": 3600,
        }
        _atomic_write_json(out_path, payload)
        return out_path

    # --------------------------
    # Polling loop
    # --------------------------

    def _filter_expired_futures(self) -> set:
        """FUTURES-ROLL-1 — return the set of futures symbols whose front-month
        contract is expired or expires today, per runtime/futures_roll_state.json.

        Emits one structured warning per (symbol, session-day) so the journal
        does not flood with one log per cycle (stops the SILK6-class
        Error 162 spam noted in the 2026-05-27 audit).
        """
        try:
            from chad.market_data.futures_expiry_gate import filter_universe
        except Exception:
            return set()
        futures_only = [s for s in self._universe if _is_futures_symbol(s)]
        if not futures_only:
            return set()

        def _warn(verdict):
            LOGGER.warning(
                "ibkr_bar_provider.futures_expiry_gate_skip",
                extra={
                    "symbol": verdict.symbol,
                    "reason": verdict.reason,
                    "current_expiry": verdict.current_expiry,
                    "next_expiry": verdict.next_expiry,
                    "roll_state_present": verdict.roll_state_present,
                },
            )

        _kept, skipped = filter_universe(futures_only, log_callback=_warn)
        return {v.symbol for v in skipped}

    def poll_once(
        self,
        duration: str = "3600 S",
        bar_size: str = "1 min",
        per_symbol_delay_s: float = PER_SYMBOL_DELAY_S,
    ) -> Dict[str, int]:
        """Fetch fresh bars for every symbol in the universe and write 1m files.

        Errors for individual symbols are logged and skipped — never crash
        the caller. Returns dict of symbol -> bar count written.

        FUTURES-ROLL-1 expiry gate: before issuing any IBKR request, skip
        futures symbols whose front-month contract is expired or expires
        today (per runtime/futures_roll_state.json). One warning is emitted
        per (symbol, session-day) — not per cycle — to stop the SILK6-class
        Error 162 spam.
        """
        results: Dict[str, int] = {}
        expired_skipped = self._filter_expired_futures()
        for sym in self._universe:
            if sym in expired_skipped:
                results[sym] = 0
                continue
            try:
                bars = self.fetch_historical_bars(
                    sym,
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show="TRADES",
                    use_rth=False,
                )
                if bars:
                    try:
                        self.write_1m_bars_file(sym, bars)
                    except Exception as werr:
                        LOGGER.warning(
                            "ibkr_bar_provider.write_1m_failed",
                            extra={"symbol": sym, "error": str(werr)},
                        )
                results[sym] = len(bars)
            except Exception as exc:
                LOGGER.warning(
                    "ibkr_bar_provider.poll_symbol_failed",
                    extra={"symbol": sym, "error": str(exc)},
                )
                results[sym] = 0
            # Stagger to respect IBKR pacing (50 req / 10s)
            if per_symbol_delay_s > 0:
                try:
                    self._ib.sleep(per_symbol_delay_s)
                except Exception:
                    time.sleep(per_symbol_delay_s)
        return results

    # --------------------------
    # Cache access
    # --------------------------

    def get_latest_bars(self, symbol: str, n: int = 100) -> List[Bar]:
        """
        Return the most recent n bars from the in-memory cache.

        Thread-safe. Returns empty list if no data available.
        """
        sym = symbol.strip().upper()
        with self._lock:
            buf = self._cache.get(sym)
            if buf is None or len(buf) == 0:
                return []
            # Return last n bars as a list
            bars = list(buf)
            return bars[-n:] if len(bars) > n else bars

    def get_all_cached_symbols(self) -> List[str]:
        """Return symbols that have cached data."""
        with self._lock:
            return [sym for sym, buf in self._cache.items() if len(buf) > 0]

    # --------------------------
    # File-based cache export
    # --------------------------

    def write_cache_file(self, path: Optional[Path] = None) -> None:
        """
        Write all cached bars to a JSON file for polling consumers.

        Output format matches build_bars_cache.py output:
        {
            "ts_utc": "...",
            "source": "ibkr",
            "symbols": { "SPY": [{bar}, ...], ... },
            "bar_count": 12345
        }
        """
        out_path = path or self._cache_path

        with self._lock:
            symbols_data: Dict[str, List[Dict[str, Any]]] = {}
            total_bars = 0
            for sym, buf in self._cache.items():
                if len(buf) > 0:
                    symbols_data[sym] = [b.to_dict() for b in buf]
                    total_bars += len(buf)

        payload = {
            "ts_utc": _utc_now_iso(),
            "source": "ibkr",
            "provider_mode": get_bar_provider_mode(),
            "symbols": symbols_data,
            "bar_count": total_bars,
        }

        _atomic_write_json(out_path, payload)
        LOGGER.debug(
            "ibkr_bar_provider.cache_written",
            extra={"path": str(out_path), "symbols": len(symbols_data), "bars": total_bars},
        )

    # --------------------------
    # Bulk operations
    # --------------------------

    def fetch_all_historical(
        self,
        duration: str = "2 D",
        bar_size: str = "1 min",
    ) -> Dict[str, int]:
        """
        Fetch historical bars for all universe symbols.

        Returns dict of symbol -> bar count. FUTURES-ROLL-1: expired-future
        symbols are skipped (see _filter_expired_futures).
        """
        results: Dict[str, int] = {}
        expired_skipped = self._filter_expired_futures()
        for sym in self._universe:
            if sym in expired_skipped:
                results[sym] = 0
                continue
            try:
                bars = self.fetch_historical_bars(sym, duration=duration, bar_size=bar_size)
                results[sym] = len(bars)
            except Exception as exc:
                LOGGER.warning("ibkr_bar_provider.fetch_all_error", extra={"symbol": sym, "error": str(exc)})
                results[sym] = 0
            # Small delay to avoid IBKR pacing violations
            time.sleep(0.5)
        return results



# ---------------------------------------------------------------------------
# Bar provider mode
# ---------------------------------------------------------------------------

def get_bar_provider_mode() -> str:
    """
    Get the configured bar provider mode.

    CHAD_BAR_PROVIDER env var:
    - "ibkr":    IBKR for all instruments (current operational mode)
    - "hybrid":  IBKR for futures; the equities branch returns through
                 should_use_ibkr() — in this build IBKR is authoritative
                 for the full universe and the legacy Polygon-for-equities
                 wiring has been retired.
    - "polygon": legacy/opt-in only. Not used in production. Will not run
                 unless CHAD_BAR_PROVIDER=polygon is set deliberately.
    """
    mode = os.environ.get("CHAD_BAR_PROVIDER", "hybrid").strip().lower()
    if mode in ("ibkr", "polygon", "hybrid"):
        return mode
    return "hybrid"


def should_use_ibkr(symbol: str) -> bool:
    """Check if a symbol should use IBKR bars based on provider mode."""
    mode = get_bar_provider_mode()
    if mode == "ibkr":
        return True
    if mode == "polygon":
        return False
    # hybrid: IBKR for futures; equities also routed through IBKR in current
    # operational state (legacy "Polygon for equities" path is retired and
    # only reachable via explicit CHAD_BAR_PROVIDER=polygon override).
    return _is_futures_symbol(symbol)


# ---------------------------------------------------------------------------
# Daemon mode
# ---------------------------------------------------------------------------

def _connect_ib(ib: Any, host: str, port: int, client_id: int) -> None:
    """Connect an ib_insync IB instance, logging the attempt."""
    LOGGER.info(
        "ibkr_bar_provider.connecting",
        extra={"host": host, "port": port, "client_id": client_id},
    )
    ib.connect(host, port, clientId=client_id, timeout=30)
    LOGGER.info("ibkr_bar_provider.connected")


def run_daemon(
    *,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 9021,
    universe: Optional[Sequence[str]] = None,
    cache_interval: float = POLLING_INTERVAL_S,
) -> None:
    """
    Run the IBKR bar provider as a persistent polling daemon.

    1. Connect to IBKR Gateway
    2. Poll reqHistoricalData for every symbol every ``cache_interval`` seconds
    3. Write per-symbol bars to data/bars/1m/{symbol}.json
    4. Write aggregate cache file runtime/ibkr_bars_cache.json
    5. Reconnect automatically on IBKR disconnect

    reqHistoricalData works on paper accounts without market-data
    subscriptions, unlike reqRealTimeBars which fails with Error 420.
    """
    from ib_async import IB

    resolved_universe = list(universe) if universe else _load_universe_from_config()

    ib = IB()
    provider: Optional[IBKRBarProvider] = None

    try:
        _connect_ib(ib, host, port, client_id)

        provider = IBKRBarProvider(ib, universe=resolved_universe)

        LOGGER.info(
            "ibkr_bar_provider.daemon_running",
            extra={
                "cache_interval": cache_interval,
                "universe_size": len(provider.universe),
            },
        )

        cycle = 0
        while True:
            try:
                if not ib.isConnected():
                    LOGGER.warning("ibkr_bar_provider.disconnected_reconnecting")
                    try:
                        ib.disconnect()
                    except Exception:
                        pass
                    time.sleep(2)
                    try:
                        _connect_ib(ib, host, port, client_id)
                    except Exception as cerr:
                        LOGGER.warning(
                            "ibkr_bar_provider.reconnect_failed",
                            extra={"error": str(cerr)},
                        )
                        ib.sleep(10)
                        continue

                cycle += 1
                t0 = time.time()
                results = provider.poll_once(
                    duration="3600 S",
                    bar_size="1 min",
                    per_symbol_delay_s=PER_SYMBOL_DELAY_S,
                )
                ok_count = sum(1 for n in results.values() if n > 0)
                fail_count = len(results) - ok_count
                try:
                    provider.write_cache_file()
                except Exception as werr:
                    LOGGER.warning(
                        "ibkr_bar_provider.cache_write_error",
                        extra={"error": str(werr)},
                    )
                elapsed = time.time() - t0
                LOGGER.info(
                    "ibkr_bar_provider.polling_cycle",
                    extra={
                        "cycle": cycle,
                        "elapsed_s": round(elapsed, 2),
                        "ok": ok_count,
                        "failed": fail_count,
                    },
                )

                remaining = cache_interval - elapsed
                if remaining > 0:
                    ib.sleep(remaining)

            except KeyboardInterrupt:
                break
            except Exception as exc:
                LOGGER.warning(
                    "ibkr_bar_provider.poll_cycle_error",
                    extra={"error": str(exc)},
                )
                time.sleep(5)

    except KeyboardInterrupt:
        LOGGER.info("ibkr_bar_provider.daemon_interrupted")
    except Exception as exc:
        LOGGER.error("ibkr_bar_provider.daemon_fatal", extra={"error": str(exc)})
        raise
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
        LOGGER.info("ibkr_bar_provider.daemon_stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="CHAD IBKR Bar Provider Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="IBKR Gateway host")
    parser.add_argument("--port", type=int, default=4002, help="IBKR Gateway port")
    parser.add_argument("--client-id", type=int, default=9021, help="IBKR client ID")
    parser.add_argument(
        "--cache-interval",
        type=float,
        default=POLLING_INTERVAL_S,
        help="Polling interval in seconds (reqHistoricalData per symbol). Default 60.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    run_daemon(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        cache_interval=args.cache_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

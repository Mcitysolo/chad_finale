#!/usr/bin/env python3
"""
chad/market_data/ibkr_bar_provider.py

IBKR Native Bar Provider for CHAD.

Replaces Polygon WebSocket → NDJSON → bars_cache pipeline with direct
IBKR reqHistoricalData + reqRealTimeBars. Provides:

- Persistent IB session (injected, not created internally)
- Historical bar fetches via reqHistoricalData
- Realtime 5-second bar subscriptions via reqRealTimeBars
- In-memory LRU cache (deque per symbol, maxlen=500)
- File-based cache export (runtime/ibkr_bars_cache.json) for polling consumers
- Graceful fallback: cached bars returned on IBKR failure
- Symbol normalization: STK vs FUT sec_type auto-detection
- Thread-safe cache writes

Bar Provider Modes (CHAD_BAR_PROVIDER env var):
- "ibkr":   IBKR for all instruments
- "polygon": Polygon for all (legacy)
- "hybrid":  IBKR for futures, Polygon for equities (default)

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

# Maximum bars per symbol in memory
MAX_BARS_PER_SYMBOL = 500

# Cache write interval (seconds)
CACHE_WRITE_INTERVAL = 30.0

# Known futures symbols -> contract specs
FUTURES_SYMBOLS = {
    "MES", "MNQ", "MCL", "MGC",    # Alpha/Gamma futures
    "ZN", "ZB", "M6E", "SIL",       # Omega Macro
}

# Full universe for bar subscriptions
DEFAULT_UNIVERSE = [
    # Equities/ETFs
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    # Futures
    "MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL",
]


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
    from ib_insync import Stock, Future

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

        # Realtime subscription handles
        self._rt_handles: Dict[str, Any] = {}

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
        contract = _make_ib_contract(sym, ib=self._ib)

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
    # Realtime bar subscription
    # --------------------------

    def subscribe_realtime_bars(
        self,
        symbol: str,
        callback: Optional[Callable[[Bar], None]] = None,
    ) -> Optional[Any]:
        """
        Subscribe to 5-second realtime bars for a symbol.

        Bars are automatically added to the in-memory cache.
        Optional callback is invoked with each new Bar.

        Returns the subscription handle (reqRealTimeBars result).
        """
        sym = symbol.strip().upper()

        if sym in self._rt_handles:
            LOGGER.debug("ibkr_bar_provider.already_subscribed", extra={"symbol": sym})
            return self._rt_handles[sym]

        contract = _make_ib_contract(sym, ib=self._ib)

        def _on_bar_update(bars: Any, has_new_bar: bool) -> None:
            if not has_new_bar:
                return
            try:
                b = bars[-1]
                ts = str(getattr(b, "date", getattr(b, "time", "")))
                bar = Bar(
                    ts_utc=ts,
                    open=_safe_float(getattr(b, "open", 0)),
                    high=_safe_float(getattr(b, "high", 0)),
                    low=_safe_float(getattr(b, "low", 0)),
                    close=_safe_float(getattr(b, "close", 0)),
                    volume=_safe_float(getattr(b, "volume", 0)),
                    symbol=sym,
                    source="ibkr_rt",
                )
                with self._lock:
                    if sym not in self._cache:
                        self._cache[sym] = deque(maxlen=self._max_bars)
                    self._cache[sym].append(bar)

                if callback:
                    callback(bar)
            except Exception as exc:
                LOGGER.debug("ibkr_bar_provider.rt_bar_error", extra={"symbol": sym, "error": str(exc)})

        try:
            handle = self._ib.reqRealTimeBars(
                contract,
                barSize=5,
                whatToShow="TRADES",
                useRTH=False,
            )
            handle.updateEvent += _on_bar_update
            self._rt_handles[sym] = handle

            LOGGER.info("ibkr_bar_provider.rt_subscribed", extra={"symbol": sym})
            return handle

        except Exception as exc:
            LOGGER.warning(
                "ibkr_bar_provider.rt_subscribe_failed",
                extra={"symbol": sym, "error": str(exc)},
            )
            return None

    def unsubscribe_realtime_bars(self, symbol: str) -> None:
        """Cancel realtime bar subscription for a symbol."""
        sym = symbol.strip().upper()
        handle = self._rt_handles.pop(sym, None)
        if handle is not None:
            try:
                self._ib.cancelRealTimeBars(handle)
            except Exception:
                pass

    def unsubscribe_all(self) -> None:
        """Cancel all realtime bar subscriptions."""
        for sym in list(self._rt_handles.keys()):
            self.unsubscribe_realtime_bars(sym)

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

        Returns dict of symbol -> bar count.
        """
        results: Dict[str, int] = {}
        for sym in self._universe:
            try:
                bars = self.fetch_historical_bars(sym, duration=duration, bar_size=bar_size)
                results[sym] = len(bars)
            except Exception as exc:
                LOGGER.warning("ibkr_bar_provider.fetch_all_error", extra={"symbol": sym, "error": str(exc)})
                results[sym] = 0
            # Small delay to avoid IBKR pacing violations
            time.sleep(0.5)
        return results

    def subscribe_all_realtime(self) -> Dict[str, bool]:
        """Subscribe to realtime bars for all universe symbols."""
        results: Dict[str, bool] = {}
        for sym in self._universe:
            handle = self.subscribe_realtime_bars(sym)
            results[sym] = handle is not None
            time.sleep(0.2)  # Avoid pacing violations
        return results


# ---------------------------------------------------------------------------
# Bar provider mode
# ---------------------------------------------------------------------------

def get_bar_provider_mode() -> str:
    """
    Get the configured bar provider mode.

    CHAD_BAR_PROVIDER env var:
    - "ibkr":   IBKR for all instruments
    - "polygon": Polygon for all (legacy)
    - "hybrid":  IBKR for futures, Polygon for equities (default)
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
    # hybrid: IBKR for futures, Polygon for equities
    return _is_futures_symbol(symbol)


# ---------------------------------------------------------------------------
# Daemon mode
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    host: str = "127.0.0.1",
    port: int = 4002,
    client_id: int = 9021,
    universe: Optional[Sequence[str]] = None,
    cache_interval: float = CACHE_WRITE_INTERVAL,
) -> None:
    """
    Run the IBKR bar provider as a persistent daemon.

    1. Connect to IBKR Gateway
    2. Fetch initial historical bars for all symbols
    3. Subscribe to realtime 5-second bars
    4. Write cache file every cache_interval seconds
    """
    from ib_insync import IB, util

    ib = IB()
    provider = None

    try:
        LOGGER.info(
            "ibkr_bar_provider.daemon_starting",
            extra={"host": host, "port": port, "client_id": client_id},
        )

        ib.connect(host, port, clientId=client_id, timeout=30)
        LOGGER.info("ibkr_bar_provider.connected")

        provider = IBKRBarProvider(ib, universe=universe)

        # Phase 1: Fetch historical bars
        LOGGER.info("ibkr_bar_provider.fetching_historical")
        hist_results = provider.fetch_all_historical(duration="2 D", bar_size="1 min")
        total_hist = sum(hist_results.values())
        LOGGER.info(
            "ibkr_bar_provider.historical_complete",
            extra={"symbols": len(hist_results), "total_bars": total_hist},
        )

        # Phase 2: Subscribe to realtime bars
        LOGGER.info("ibkr_bar_provider.subscribing_realtime")
        rt_results = provider.subscribe_all_realtime()
        rt_ok = sum(1 for v in rt_results.values() if v)
        LOGGER.info(
            "ibkr_bar_provider.realtime_subscribed",
            extra={"subscribed": rt_ok, "failed": len(rt_results) - rt_ok},
        )

        # Phase 3: Periodic cache write loop
        LOGGER.info("ibkr_bar_provider.daemon_running", extra={"cache_interval": cache_interval})

        while True:
            try:
                ib.sleep(cache_interval)
                provider.write_cache_file()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                LOGGER.warning("ibkr_bar_provider.cache_write_error", extra={"error": str(exc)})
                time.sleep(5)

    except KeyboardInterrupt:
        LOGGER.info("ibkr_bar_provider.daemon_interrupted")
    except Exception as exc:
        LOGGER.error("ibkr_bar_provider.daemon_fatal", extra={"error": str(exc)})
        raise
    finally:
        if provider:
            provider.unsubscribe_all()
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
    parser.add_argument("--cache-interval", type=float, default=CACHE_WRITE_INTERVAL)
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

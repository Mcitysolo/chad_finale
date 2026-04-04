#!/usr/bin/env python3
"""
chad/market_data/ibkr_price_provider.py

IBKR Native Price Provider for CHAD.

Replaces Polygon snapshot/FX calls with direct IBKR reqMktData.
Thread-safe, cache-backed, with delayed-data fallback.

Usage:
    from ib_insync import IB
    from chad.market_data.ibkr_price_provider import IBKRPriceProvider

    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=9030)
    provider = IBKRPriceProvider(ib)
    snap = provider.get_snapshot('SPY')
    print(snap.last, snap.close, snap.source)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("chad.market_data.ibkr_price_provider")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "runtime"
PRICE_CACHE_PATH = RUNTIME_DIR / "price_cache.json"


@dataclass(frozen=True)
class PriceSnapshot:
    """Single-symbol price snapshot from IBKR."""
    symbol: str
    last: float
    close: float
    bid: float
    ask: float
    ts_utc: str
    source: str = "ibkr"
    delayed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(x: Any) -> float:
    """Return x as float, or NaN if not convertible/finite."""
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _is_nan(x: float) -> bool:
    return math.isnan(x)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class IBKRPriceProvider:
    """
    Thread-safe IBKR price snapshot provider.

    Uses reqMktData with marketDataType=3 (delayed OK).
    Falls back to price_cache.json for symbols that return all NaN.
    """

    SNAPSHOT_TIMEOUT_S = 5.0

    def __init__(
        self,
        ib: Any,
        *,
        price_cache_path: Optional[Path] = None,
    ) -> None:
        self._ib = ib
        self._lock = threading.Lock()
        self._price_cache_path = price_cache_path or PRICE_CACHE_PATH
        self._last_known: Dict[str, float] = {}

        # Load existing price cache as fallback
        self._load_fallback_cache()

        # Request delayed data (free, no market data subscription needed)
        try:
            self._ib.reqMarketDataType(3)
        except Exception:
            pass

    def _load_fallback_cache(self) -> None:
        """Load last-known prices from price_cache.json."""
        try:
            if self._price_cache_path.is_file():
                obj = json.loads(self._price_cache_path.read_text(encoding="utf-8"))
                prices = obj.get("prices", {})
                if isinstance(prices, dict):
                    for sym, px in prices.items():
                        if isinstance(px, (int, float)) and math.isfinite(px) and px > 0:
                            self._last_known[str(sym).upper()] = float(px)
                    LOGGER.info("ibkr_price_provider.fallback_loaded symbols=%d", len(self._last_known))
        except Exception as exc:
            LOGGER.warning("ibkr_price_provider.fallback_load_failed: %s", exc)

    def _make_contract(self, symbol: str, sec_type: str = "STK") -> Any:
        """Create ib_insync contract for symbol."""
        from ib_insync import Stock, Forex, Future

        sym = symbol.strip().upper()
        if sec_type == "FX":
            return Forex(sym)
        if sec_type == "FUT":
            # Known futures exchanges
            exchange_map = {
                "MES": "CME", "MNQ": "CME", "MCL": "NYMEX",
                "MGC": "COMEX", "ZN": "CBOT", "ZB": "CBOT",
                "M6E": "CME", "SIL": "COMEX",
            }
            exchange = exchange_map.get(sym, "CME")
            contract = Future(symbol=sym, exchange=exchange, currency="USD")
            try:
                qualified = self._ib.qualifyContracts(contract)
                if qualified:
                    return qualified[0]
            except Exception:
                pass
            return contract
        return Stock(sym, "SMART", "USD")

    def get_snapshot(
        self,
        symbol: str,
        sec_type: str = "STK",
    ) -> PriceSnapshot:
        """
        Get a price snapshot for a single symbol.

        Falls back through: last -> close -> cached price.
        Thread-safe.
        """
        sym = symbol.strip().upper()
        contract = self._make_contract(sym, sec_type)

        try:
            self._ib.qualifyContracts(contract)
        except Exception:
            pass

        ticker = self._ib.reqMktData(contract, "", False, False)

        # Wait for data with timeout
        deadline = time.time() + self.SNAPSHOT_TIMEOUT_S
        while time.time() < deadline:
            self._ib.sleep(0.2)
            last = _safe_float(ticker.last)
            close = _safe_float(ticker.close)
            if not _is_nan(last) or not _is_nan(close):
                break

        last = _safe_float(ticker.last)
        close = _safe_float(ticker.close)
        bid = _safe_float(ticker.bid)
        ask = _safe_float(ticker.ask)

        # Cancel subscription
        try:
            self._ib.cancelMktData(contract)
        except Exception:
            pass

        # Determine best price: last -> close -> fallback
        resolved_last = last
        resolved_close = close
        delayed = False

        if _is_nan(resolved_last) and not _is_nan(resolved_close):
            resolved_last = resolved_close
            delayed = True

        if _is_nan(resolved_last):
            # All NaN — use fallback cache
            with self._lock:
                cached = self._last_known.get(sym, float("nan"))
            if not _is_nan(cached):
                resolved_last = cached
                resolved_close = cached if _is_nan(resolved_close) else resolved_close
                delayed = True
                LOGGER.info("ibkr_price_provider.using_fallback symbol=%s price=%.4f", sym, cached)
            else:
                LOGGER.warning("ibkr_price_provider.no_price symbol=%s", sym)

        # Update last-known cache
        if not _is_nan(resolved_last):
            with self._lock:
                self._last_known[sym] = resolved_last

        return PriceSnapshot(
            symbol=sym,
            last=resolved_last if not _is_nan(resolved_last) else 0.0,
            close=resolved_close if not _is_nan(resolved_close) else 0.0,
            bid=bid if not _is_nan(bid) else 0.0,
            ask=ask if not _is_nan(ask) else 0.0,
            ts_utc=_utc_now_iso(),
            source="ibkr",
            delayed=delayed,
        )

    def get_batch_snapshots(
        self,
        symbols: List[str],
        sec_type: str = "STK",
    ) -> Dict[str, PriceSnapshot]:
        """
        Get snapshots for multiple symbols.

        Fetches sequentially with small delay to avoid IBKR pacing.
        """
        results: Dict[str, PriceSnapshot] = {}
        for sym in symbols:
            try:
                results[sym.strip().upper()] = self.get_snapshot(sym, sec_type=sec_type)
            except Exception as exc:
                LOGGER.warning("ibkr_price_provider.batch_error symbol=%s: %s", sym, exc)
            time.sleep(0.1)
        return results

    def get_fx_rate(self, pair: str = "USDCAD") -> float:
        """
        Get FX rate for a currency pair.

        Uses IBKR Forex contract. Returns 0.0 on failure.
        """
        from ib_insync import Forex

        pair = pair.strip().upper()
        try:
            contract = Forex(pair)
            ticker = self._ib.reqMktData(contract, "", False, False)
            self._ib.sleep(3)

            close = _safe_float(ticker.close)
            last = _safe_float(ticker.last)
            bid = _safe_float(ticker.bid)
            ask = _safe_float(ticker.ask)

            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass

            # Best available: last -> close -> midpoint
            if not _is_nan(last):
                return last
            if not _is_nan(close):
                return close
            if not _is_nan(bid) and not _is_nan(ask):
                return (bid + ask) / 2.0

            LOGGER.warning("ibkr_price_provider.fx_no_data pair=%s", pair)
            return 0.0

        except Exception as exc:
            LOGGER.warning("ibkr_price_provider.fx_error pair=%s: %s", pair, exc)
            return 0.0

    def write_price_cache(
        self,
        symbols: List[str],
        sec_type: str = "STK",
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Fetch snapshots for all symbols and write price_cache.json.

        Returns the payload written.
        """
        out = output_path or self._price_cache_path
        snapshots = self.get_batch_snapshots(symbols, sec_type=sec_type)

        prices: Dict[str, float] = {}
        for sym, snap in snapshots.items():
            if snap.last > 0:
                prices[sym] = snap.last

        payload = {
            "prices": dict(sorted(prices.items())),
            "ts_utc": _utc_now_iso(),
            "ttl_seconds": 300,
        }

        _atomic_write_json(out, payload)
        LOGGER.info("ibkr_price_provider.cache_written symbols=%d path=%s", len(prices), out)
        return payload

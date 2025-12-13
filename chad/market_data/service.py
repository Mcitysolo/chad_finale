#!/usr/bin/env python3
"""
chad/market_data/service.py

Unified market data access layer for CHAD.

Goals
-----

- Provide a single, well-typed interface for fetching latest prices and
  simple daily changes across multiple providers (Polygon, IBKR, crypto
  exchanges like Coinbase/Kraken, etc.).
- Centralize error handling, environment loading, and provider selection
  so that frontends (Telegram, web, voice, strategies) do NOT talk to
  raw APIs directly.
- Be safe, predictable, and easy to extend as new providers are added.

Initial implementation:
    - Polygon only (for US stocks/ETFs).
    - Designed so IBKR, Coinbase, Kraken can be added later without
      changing external interfaces.

Usage
-----

    from chad.market_data.service import MarketDataService, MarketDataError

    service = MarketDataService()
    snap = service.get_price_snapshot("AAPL")
    print(snap.symbol, snap.price, snap.percent_change)

Configuration
-------------

Environment variables:

    POLYGON_API_KEY
        Required if using the Polygon provider.
        If not found in os.environ, this module will attempt to load
        /etc/chad/polygon.env with KEY=VALUE lines.

    MARKET_DATA_DEFAULT_PROVIDER
        Optional: "polygon" (default for now).

Provider priority is currently:
    1) Explicit provider arg passed to MarketDataService
    2) MARKET_DATA_DEFAULT_PROVIDER env var
    3) "polygon"

API Notes
---------

Polygon responses observed in this environment include shapes like:

    {
        "results": {
            "T": "AAPL",
            "c": [37],
            "f": 1765383564330546035,
            "i": "143134",
            "p": 277.41,
            "q": 3930614,
            "r": 202,
            "s": 1,
            "t": 1765383564330566878,
            "x": 4,
            "y": 1765383564298000000,
            "z": 3
        },
        "status": "OK",
        "request_id": "..."
    }

This module defensively supports both the documented "last" style and
the "results" style by looking for price in:

    - last["price"]
    - results["p"]
    - results["price"]

"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

LOGGER_NAME = "chad.market_data"
logger = logging.getLogger(LOGGER_NAME)


def _ensure_logging() -> None:
    """
    Ensure logging is configured for this module.

    Safe to call multiple times.
    """
    if logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class MarketDataError(RuntimeError):
    """Base error type for market data failures."""


@dataclass(frozen=True)
class PriceSnapshot:
    """
    Simple representation of a price snapshot for a symbol.

    Fields
    ------

    symbol: str
        Normalized symbol requested (e.g. "AAPL", "SPY").
    asset_class: str
        Broad asset class: "equity", "etf", "crypto", etc.
    price: float
        Latest trade/spot price.
    change: Optional[float]
        Absolute change vs previous close (if available).
    percent_change: Optional[float]
        Percent change vs previous close (if available).
    as_of: str
        ISO-8601 timestamp of the price (UTC), if available.
    source: str
        Provider name, e.g. "polygon".
    """

    symbol: str
    asset_class: str
    price: float
    change: Optional[float]
    percent_change: Optional[float]
    as_of: str
    source: str


# ---------------------------------------------------------------------------
# Env loading helpers
# ---------------------------------------------------------------------------


def _load_env_file(path: str) -> None:
    """
    Best-effort loader for KEY=VALUE env files.

    Will not overwrite existing os.environ entries.
    """
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:  # noqa: BLE001
        _ensure_logging()
        logger.exception("Failed to load env file %s: %s", path, exc)


def _get_polygon_api_key() -> str:
    """
    Retrieve the Polygon API key from env or /etc/chad/polygon.env.

    Raises:
        MarketDataError if no key can be found.
    """
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if key:
        return key

    _load_env_file("/etc/chad/polygon.env")
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not key:
        raise MarketDataError(
            "POLYGON_API_KEY is not set and /etc/chad/polygon.env could not provide one."
        )
    return key


# ---------------------------------------------------------------------------
# Polygon provider implementation
# ---------------------------------------------------------------------------


class _PolygonClient:
    """
    Minimal Polygon client used by MarketDataService.

    Only implements what is needed for price snapshots, in a safe and
    defensive manner. Designed so the rest of CHAD never talks to
    Polygon directly.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 5.0) -> None:
        self.api_key = api_key or _get_polygon_api_key()
        self.timeout = timeout
        self._session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _ensure_logging()
        url = f"{self.BASE_URL}{path}"
        params = dict(params or {})
        params["apiKey"] = self.api_key

        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise MarketDataError(f"Polygon network error: {exc}") from exc

        if resp.status_code != 200:
            raise MarketDataError(
                f"Polygon returned {resp.status_code}: {resp.text[:256]}"
            )

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise MarketDataError(f"Polygon JSON decode error: {exc}") from exc

        return data

    def get_last_trade(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch last trade for a symbol.

        Supports both "last" and "results" shapes.

        Returns:
            Parsed JSON dict with at least a nested object containing:
                - price (float) under one of:
                    * last["price"]
                    * results["p"]
                    * results["price"]
        """
        path = f"/v2/last/trade/{symbol.upper()}"
        data = self._get(path)
        return data

    def get_previous_close(self, symbol: str) -> Optional[float]:
        """
        Fetch previous close price for a symbol.

        Returns:
            float or None if not available.
        """
        path = f"/v2/aggs/ticker/{symbol.upper()}/prev"
        data = self._get(path)
        if not isinstance(data, dict):
            return None

        results = data.get("results")
        if not isinstance(results, list) or not results:
            return None

        first = results[0]
        close = first.get("c")
        if isinstance(close, (int, float)):
            return float(close)
        return None


# ---------------------------------------------------------------------------
# MarketDataService
# ---------------------------------------------------------------------------


class MarketDataService:
    """
    Unified market data access layer.

    Currently supports:
        - Provider: "polygon" (default)

    Future extensions:
        - "ibkr"    for account-linked data
        - "coinbase"/"kraken" for crypto
        - caching backends (Redis, in-memory LRU)

    This class is safe to instantiate frequently; provider clients are
    relatively lightweight and use internal sessions.
    """

    def __init__(self, provider: Optional[str] = None) -> None:
        _ensure_logging()
        self.provider = provider or os.environ.get(
            "MARKET_DATA_DEFAULT_PROVIDER", "polygon"
        ).lower()

        if self.provider not in ("polygon",):
            raise MarketDataError(
                f"Unsupported market data provider {self.provider!r}. "
                "Currently only 'polygon' is implemented."
            )

        self._polygon_client: Optional[_PolygonClient] = None

    def _get_polygon_client(self) -> _PolygonClient:
        if self._polygon_client is None:
            self._polygon_client = _PolygonClient()
        return self._polygon_client

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_price_snapshot(self, symbol: str) -> PriceSnapshot:
        """
        Fetch a PriceSnapshot for a given symbol using the configured provider.

        For now, this delegates to Polygon.

        Args:
            symbol: Ticker symbol, e.g. "AAPL", "SPY".

        Returns:
            PriceSnapshot with price and optional daily change fields.

        Raises:
            MarketDataError on network/API/parse failures.
        """
        symbol = symbol.upper().strip()
        if not symbol:
            raise MarketDataError("Symbol must be a non-empty string.")

        if self.provider == "polygon":
            return self._get_polygon_price_snapshot(symbol)

        # Safety net: provider validation is done in __init__.
        raise MarketDataError(f"Provider {self.provider!r} is not implemented.")

    # ------------------------------------------------------------------ #
    # Provider-specific implementations
    # ------------------------------------------------------------------ #

    def _get_polygon_price_snapshot(self, symbol: str) -> PriceSnapshot:
        client = self._get_polygon_client()

        last_data = client.get_last_trade(symbol)

        # Support both "last" and "results" shapes.
        record: Optional[Dict[str, Any]] = None

        if isinstance(last_data, dict):
            if "last" in last_data and isinstance(last_data["last"], dict):
                record = last_data["last"]
            elif "results" in last_data and isinstance(last_data["results"], dict):
                record = last_data["results"]

        if record is None:
            raise MarketDataError(
                f"Polygon last trade for {symbol!r} missing trade record: {json.dumps(last_data)[:256]}"
            )

        # Price can be "p" (observed) or "price" (documented).
        price_val = record.get("p")
        if price_val is None:
            price_val = record.get("price")

        if not isinstance(price_val, (int, float)):
            raise MarketDataError(
                f"Polygon last trade for {symbol!r} missing price: {json.dumps(last_data)[:256]}"
            )

        price = float(price_val)

        # Preferred timestamp fields: "t" (trade timestamp in ms) or fallback empty.
        ts = record.get("t")
        as_of_iso = ""
        if isinstance(ts, (int, float)):
            try:
                as_of_dt = dt.datetime.fromtimestamp(ts / 1000.0, tz=dt.timezone.utc)
                as_of_iso = as_of_dt.isoformat()
            except Exception:  # noqa: BLE001
                as_of_iso = ""

        prev_close = None
        change: Optional[float] = None
        percent_change: Optional[float] = None

        try:
            prev_close = client.get_previous_close(symbol)
        except MarketDataError as exc:
            # Log and continue; we can still return a snapshot with just price.
            logger.warning(
                "Failed to fetch previous close for %s via Polygon: %s", symbol, exc
            )

        if isinstance(prev_close, (int, float)):
            change = price - float(prev_close)
            if prev_close != 0:
                percent_change = (change / float(prev_close)) * 100.0

        snapshot = PriceSnapshot(
            symbol=symbol,
            asset_class="equity",  # TODO: refine via metadata if needed
            price=price,
            change=change,
            percent_change=percent_change,
            as_of=as_of_iso,
            source="polygon",
        )

        logger.info(
            "PriceSnapshot: symbol=%s price=%.4f change=%s pct=%s as_of=%s source=%s",
            snapshot.symbol,
            snapshot.price,
            snapshot.change,
            snapshot.percent_change,
            snapshot.as_of,
            snapshot.source,
        )

        return snapshot

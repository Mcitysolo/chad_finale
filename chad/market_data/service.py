#!/usr/bin/env python3
"""
chad/market_data/service.py

Unified market data access layer for CHAD.

Goals
-----

- Provide a single, well-typed interface for fetching latest prices and
  simple daily changes from IBKR.
- Centralize error handling, environment loading, and provider selection
  so that frontends (Telegram, web, voice, strategies) do NOT talk to
  raw APIs directly.
- Be safe, predictable, and easy to extend if new providers are added.

Current implementation:
    - IBKR only. Polygon support was removed when the subscription was
      cancelled; IBKR is the sole authoritative market data source.

Usage
-----

    from chad.market_data.service import MarketDataService, MarketDataError

    service = MarketDataService(ib=ib_connection)
    snap = service.get_price_snapshot("AAPL")
    print(snap.symbol, snap.price, snap.percent_change)

Configuration
-------------

Environment variables:

    CHAD_MARKET_DATA_PROVIDER
        Optional: must be "ibkr" if set (anything else raises).

Provider priority is:
    1) Explicit provider arg passed to MarketDataService
    2) CHAD_MARKET_DATA_PROVIDER env var
    3) "ibkr"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

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
        Provider name, e.g. "ibkr".
    """

    symbol: str
    asset_class: str
    price: float
    change: Optional[float]
    percent_change: Optional[float]
    as_of: str
    source: str


# ---------------------------------------------------------------------------
# MarketDataService
# ---------------------------------------------------------------------------


class IBKRMarketDataService:
    """
    IBKR-native market data service.

    Uses IBKRPriceProvider for snapshots and IBKRHistoricalProvider for bars.
    """

    def __init__(self, ib: Optional[Any] = None) -> None:
        _ensure_logging()
        self.provider = "ibkr"
        self._ib = ib
        self._price_provider = None

    def _get_price_provider(self) -> Any:
        if self._price_provider is None:
            from chad.market_data.ibkr_price_provider import IBKRPriceProvider
            if self._ib is None:
                raise MarketDataError("IBKRMarketDataService requires an IB connection")
            self._price_provider = IBKRPriceProvider(self._ib)
        return self._price_provider

    def get_price_snapshot(self, symbol: str) -> PriceSnapshot:
        symbol = symbol.upper().strip()
        if not symbol:
            raise MarketDataError("Symbol must be a non-empty string.")

        provider = self._get_price_provider()
        snap = provider.get_snapshot(symbol)

        price = snap.last if snap.last > 0 else snap.close
        return PriceSnapshot(
            symbol=symbol,
            asset_class="equity",
            price=price,
            change=None,
            percent_change=None,
            as_of=snap.ts_utc,
            source="ibkr",
        )

    def get_bars(self, symbol: str, days: int = 400) -> list:
        from chad.market_data.ibkr_historical_provider import IBKRHistoricalProvider
        if self._ib is None:
            raise MarketDataError("IBKRMarketDataService requires an IB connection")
        hist = IBKRHistoricalProvider(self._ib)
        return hist.fetch_daily_bars(symbol, days=days)


class MarketDataService:
    """
    Unified market data access layer.

    Supports:
        - Provider: "ibkr" (only supported provider)

    Provider selection:
        1) Explicit provider arg
        2) CHAD_MARKET_DATA_PROVIDER env var
        3) "ibkr" (default)
    """

    def __init__(self, provider: Optional[str] = None, ib: Optional[Any] = None) -> None:
        _ensure_logging()
        self.provider = provider or os.environ.get(
            "CHAD_MARKET_DATA_PROVIDER", "ibkr"
        ).lower()

        if self.provider != "ibkr":
            raise MarketDataError(
                f"Unsupported market data provider {self.provider!r}. "
                "Supported: 'ibkr'."
            )

        self._ibkr_service: Optional[IBKRMarketDataService] = None
        self._ib = ib

    def _get_ibkr_service(self) -> IBKRMarketDataService:
        if self._ibkr_service is None:
            self._ibkr_service = IBKRMarketDataService(ib=self._ib)
        return self._ibkr_service

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_price_snapshot(self, symbol: str) -> PriceSnapshot:
        """
        Fetch a PriceSnapshot for a given symbol using the configured provider.

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

        return self._get_ibkr_service().get_price_snapshot(symbol)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_market_data_service(
    provider: Optional[str] = None,
    ib: Optional[Any] = None,
) -> MarketDataService:
    """
    Factory: return a MarketDataService configured for the requested provider.

    Provider resolution:
        1) Explicit `provider` arg
        2) CHAD_MARKET_DATA_PROVIDER env var
        3) "ibkr" (default)
    """
    return MarketDataService(provider=provider, ib=ib)

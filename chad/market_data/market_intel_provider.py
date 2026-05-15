"""Market intelligence provider abstraction.

Phase B FMP Phase 1 — abstraction layer only. No publisher is migrated
in this phase. The goal is to give future migrations a single seam so
the existing publishers can swap to FMP one at a time without touching
the strategy code.

Provider routing:
- ``CHAD_INTEL_PROVIDER``  default ``fmp``  — controls equities/earnings
  fact lookups (quote, profile, earnings calendar, price target consensus,
  analyst estimates, SEC filings).
- ``CHAD_NEWS_PROVIDER``   default ``existing`` — keeps Polygon/Yahoo
  flow inside ``news_intel_publisher`` because FMP /stable/news/stock
  is restricted on the current plan.
- ``CHAD_VOLUME_PROVIDER`` default ``existing`` — keeps the
  ``volume_scan_publisher`` rolling_1m fallback in place; FMP volume
  migration is deferred.

Stubs for news/volume return empty containers and explicitly do NOT
issue any HTTP request.

This module has no side effects at import time. The first FMP client
instance is built lazily on first ``get_fmp_client()`` call.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from chad.market_data.fmp_client import (
    FMPAnalystEstimate,
    FMPClient,
    FMPEarningsEvent,
    FMPPriceTargetConsensus,
    FMPProfile,
    FMPQuote,
    FMPSecFiling,
)

DEFAULT_INTEL_PROVIDER = "fmp"
DEFAULT_NEWS_PROVIDER = "existing"
DEFAULT_VOLUME_PROVIDER = "existing"


_client_singleton: Optional[FMPClient] = None


def _intel_provider() -> str:
    return os.environ.get("CHAD_INTEL_PROVIDER", DEFAULT_INTEL_PROVIDER).strip().lower() or DEFAULT_INTEL_PROVIDER


def _news_provider() -> str:
    return os.environ.get("CHAD_NEWS_PROVIDER", DEFAULT_NEWS_PROVIDER).strip().lower() or DEFAULT_NEWS_PROVIDER


def _volume_provider() -> str:
    return os.environ.get("CHAD_VOLUME_PROVIDER", DEFAULT_VOLUME_PROVIDER).strip().lower() or DEFAULT_VOLUME_PROVIDER


def get_fmp_client() -> FMPClient:
    """Return a process-wide FMP client. Lazy-built on first call."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = FMPClient()
    return _client_singleton


def reset_fmp_client_for_tests(client: Optional[FMPClient] = None) -> None:
    """Replace the cached singleton. Test-only helper.

    Production code never imports this; it exists so the test suite can
    inject a fake client without touching the singleton's internals.
    """
    global _client_singleton
    _client_singleton = client


# ---------------------------------------------------------------------------
# Equities / fundamentals — routed to FMP when provider==fmp.
# ---------------------------------------------------------------------------


def fetch_quote(symbol: str) -> List[FMPQuote]:
    if _intel_provider() != "fmp":
        return []
    return get_fmp_client().get_quote(symbol)


def fetch_profile(symbol: str) -> List[FMPProfile]:
    if _intel_provider() != "fmp":
        return []
    return get_fmp_client().get_profile(symbol)


def fetch_earnings_calendar(
    date_from: str, date_to: str
) -> List[FMPEarningsEvent]:
    if _intel_provider() != "fmp":
        return []
    return get_fmp_client().get_earnings_calendar(date_from, date_to)


def fetch_price_target_consensus(
    symbol: str,
) -> List[FMPPriceTargetConsensus]:
    if _intel_provider() != "fmp":
        return []
    return get_fmp_client().get_price_target_consensus(symbol)


def fetch_analyst_estimates_annual(
    symbol: str,
) -> List[FMPAnalystEstimate]:
    if _intel_provider() != "fmp":
        return []
    return get_fmp_client().get_analyst_estimates_annual(symbol)


def fetch_sec_filings(
    symbol: str, date_from: str, date_to: str
) -> List[FMPSecFiling]:
    if _intel_provider() != "fmp":
        return []
    return get_fmp_client().get_sec_filings(symbol, date_from, date_to)


# ---------------------------------------------------------------------------
# Deferred surfaces — explicit no-ops so callers see a stable signature.
# ---------------------------------------------------------------------------


def fetch_news(symbol: str, limit: int = 10) -> list:
    # FMP /stable/news/stock is restricted on the current plan.
    # The existing news_intel_publisher (Polygon + Yahoo fallback) remains
    # the authoritative source until that plan changes.
    _ = (symbol, limit)
    return []


def fetch_volume_snapshot(symbols: list) -> dict:
    # Polygon snapshot is gone and FMP volume migration is deferred.
    # volume_scan_publisher continues to use its rolling_1m fallback.
    _ = symbols
    return {}


__all__ = [
    "DEFAULT_INTEL_PROVIDER",
    "DEFAULT_NEWS_PROVIDER",
    "DEFAULT_VOLUME_PROVIDER",
    "fetch_analyst_estimates_annual",
    "fetch_earnings_calendar",
    "fetch_news",
    "fetch_price_target_consensus",
    "fetch_profile",
    "fetch_quote",
    "fetch_sec_filings",
    "fetch_volume_snapshot",
    "get_fmp_client",
    "reset_fmp_client_for_tests",
]

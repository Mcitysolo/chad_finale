#!/usr/bin/env python3
"""
chad/market_data/yahoo_news_provider.py

Yahoo Finance News Provider for CHAD.

Two data sources with automatic fallback:
  1. Yahoo Finance Search API (structured JSON)
  2. Yahoo Finance RSS feed (XML headlines)

No API key required. Works from Canada. Replaces AlpacaNewsProvider.
"""

from __future__ import annotations

import logging
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

LOGGER = logging.getLogger("chad.market_data.yahoo_news_provider")

_USER_AGENT = "Mozilla/5.0 (compatible; CHAD/1.0)"
_TIMEOUT = 5


@dataclass(frozen=True)
class NewsItem:
    """Single news headline."""
    headline: str
    summary: str
    url: str
    published_utc: str
    symbols: List[str]
    source: str = "yahoo_finance"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _make_request(url: str) -> bytes:
    """HTTP GET with User-Agent header and timeout."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    resp = urlopen(req, timeout=_TIMEOUT)
    return resp.read()


def _parse_search_json(data: bytes, symbols: List[str]) -> List[NewsItem]:
    """Parse Yahoo Finance Search API JSON response."""
    payload = json.loads(data)
    news_list = payload.get("news", [])
    if not isinstance(news_list, list):
        return []

    items: List[NewsItem] = []
    for entry in news_list:
        if not isinstance(entry, dict):
            continue
        headline = str(entry.get("title", "")).strip()
        if not headline:
            continue

        pub_ts = entry.get("providerPublishTime", 0)
        try:
            pub_utc = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc).isoformat()
        except (ValueError, TypeError, OSError):
            pub_utc = ""

        items.append(NewsItem(
            headline=headline,
            summary=str(entry.get("publisher", "")).strip(),
            url=str(entry.get("link", "")).strip(),
            published_utc=pub_utc,
            symbols=list(symbols),
            source="yahoo_finance",
        ))
    return items


def _parse_rss_xml(data: bytes, symbols: List[str]) -> List[NewsItem]:
    """Parse Yahoo Finance RSS XML response."""
    text = data.decode("utf-8", errors="replace")
    items: List[NewsItem] = []

    # Extract <item> blocks
    for item_match in re.finditer(r"<item>(.*?)</item>", text, re.DOTALL):
        block = item_match.group(1)

        title_m = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
        link_m = re.search(r"<link>(.*?)</link>", block, re.DOTALL)
        pubdate_m = re.search(r"<pubDate>(.*?)</pubDate>", block, re.DOTALL)

        headline = (title_m.group(1).strip() if title_m else "").strip()
        if not headline:
            continue

        pub_utc = ""
        if pubdate_m:
            raw_date = pubdate_m.group(1).strip()
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(raw_date)
                pub_utc = dt.astimezone(timezone.utc).isoformat()
            except Exception:
                pub_utc = raw_date

        items.append(NewsItem(
            headline=headline,
            summary="",
            url=(link_m.group(1).strip() if link_m else ""),
            published_utc=pub_utc,
            symbols=list(symbols),
            source="yahoo_finance_rss",
        ))
    return items


class YahooNewsProvider:
    """
    Fetch news headlines from Yahoo Finance.

    Two sources with fallback: Search API -> RSS feed.
    No API key required. Never crashes: returns empty list on any error.
    """

    @property
    def configured(self) -> bool:
        """Always configured — no API key needed."""
        return True

    def get_headlines(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[NewsItem]:
        """
        Fetch recent news headlines.

        Parameters
        ----------
        symbols : list of str, optional
            Symbols to fetch news for. If None/empty, uses SPY.
        limit : int
            Max headlines to return.

        Returns
        -------
        List[NewsItem]
            Headlines. Empty list on any error.
        """
        syms = [s.strip().upper() for s in (symbols or []) if s.strip()]
        if not syms:
            syms = ["SPY"]

        limit = max(1, min(int(limit), 50))
        all_items: List[NewsItem] = []

        for symbol in syms:
            items = self._fetch_search_api(symbol, limit)
            if not items:
                items = self._fetch_rss(symbol, limit)
            all_items.extend(items)

        return all_items[:limit]

    def get_market_headlines(self, limit: int = 5) -> List[NewsItem]:
        """General market news using SPY as proxy."""
        return self.get_headlines(symbols=["SPY"], limit=limit)

    def _fetch_search_api(self, symbol: str, limit: int) -> List[NewsItem]:
        """Source 1: Yahoo Finance Search API."""
        try:
            url = (
                f"https://query1.finance.yahoo.com/v1/finance/search"
                f"?q={symbol}&newsCount={limit}"
            )
            data = _make_request(url)
            items = _parse_search_json(data, [symbol])
            if items:
                LOGGER.debug("yahoo_news.search_api: %d headlines for %s", len(items), symbol)
            return items[:limit]
        except Exception as exc:
            LOGGER.debug("yahoo_news.search_api_error(%s): %s", symbol, exc)
            return []

    def _fetch_rss(self, symbol: str, limit: int) -> List[NewsItem]:
        """Source 2: Yahoo Finance RSS feed (fallback)."""
        try:
            url = (
                f"https://feeds.finance.yahoo.com/rss/2.0/headline"
                f"?s={symbol}&region=US&lang=en-US"
            )
            data = _make_request(url)
            items = _parse_rss_xml(data, [symbol])
            if items:
                LOGGER.debug("yahoo_news.rss: %d headlines for %s", len(items), symbol)
            return items[:limit]
        except Exception as exc:
            LOGGER.debug("yahoo_news.rss_error(%s): %s", symbol, exc)
            return []

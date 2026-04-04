#!/usr/bin/env python3
"""
chad/market_data/alpaca_news_provider.py

Alpaca News Provider for CHAD.

Replaces Polygon news headlines with Alpaca's free news API.
Graceful fallback: returns empty list on any error (never crashes advisory).

Usage:
    from chad.market_data.alpaca_news_provider import AlpacaNewsProvider

    provider = AlpacaNewsProvider()
    headlines = provider.get_headlines(symbols=["SPY", "AAPL"], limit=5)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

LOGGER = logging.getLogger("chad.market_data.alpaca_news_provider")

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


@dataclass(frozen=True)
class NewsItem:
    """Single news headline."""
    headline: str
    summary: str
    url: str
    published_utc: str
    symbols: List[str]
    source: str = "alpaca"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_alpaca_keys() -> tuple:
    """
    Load Alpaca API credentials from env or /etc/chad/alpaca.env.

    Returns (api_key, api_secret) or ("", "").
    """
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    api_secret = os.environ.get("ALPACA_API_SECRET", "").strip()

    if api_key and api_secret:
        return api_key, api_secret

    env_path = Path("/etc/chad/alpaca.env")
    if env_path.is_file():
        try:
            for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "ALPACA_API_KEY" and v:
                    api_key = v
                elif k == "ALPACA_API_SECRET" and v:
                    api_secret = v
        except Exception:
            pass

    return api_key, api_secret


class AlpacaNewsProvider:
    """
    Fetch news headlines from Alpaca's free news API.

    Never crashes: returns empty list on any error.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        if api_key and api_secret:
            self._api_key = api_key
            self._api_secret = api_secret
        else:
            self._api_key, self._api_secret = _load_alpaca_keys()
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._api_key and self._api_secret)

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
            Filter news to these symbols. If None, returns general market news.
        limit : int
            Max headlines to return (1-50).

        Returns
        -------
        List[NewsItem]
            Headlines sorted by published_utc desc. Empty list on any error.
        """
        if not self.configured:
            LOGGER.debug("alpaca_news.no_api_key — returning empty")
            return []

        try:
            params: Dict[str, Any] = {
                "limit": max(1, min(int(limit), 50)),
                "sort": "desc",
            }
            if symbols:
                params["symbols"] = ",".join(
                    s.strip().upper() for s in symbols if s.strip()
                )

            resp = requests.get(
                ALPACA_NEWS_URL,
                headers={
                    "APCA-API-KEY-ID": self._api_key,
                    "APCA-API-SECRET-KEY": self._api_secret,
                },
                params=params,
                timeout=self._timeout,
            )

            if resp.status_code != 200:
                LOGGER.warning("alpaca_news.http_%d", resp.status_code)
                return []

            payload = resp.json()
            news_list = payload.get("news", [])
            if not isinstance(news_list, list):
                return []

            items: List[NewsItem] = []
            for item in news_list:
                if not isinstance(item, dict):
                    continue

                headline = str(item.get("headline", "")).strip()
                if not headline:
                    continue

                raw_symbols = item.get("symbols", [])
                syms = (
                    [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
                    if isinstance(raw_symbols, list)
                    else []
                )

                items.append(NewsItem(
                    headline=headline,
                    summary=str(item.get("summary", "")).strip()[:500],
                    url=str(item.get("url", "")).strip(),
                    published_utc=str(item.get("created_at", "")).strip(),
                    symbols=syms,
                    source="alpaca",
                ))

            return items

        except Exception as exc:
            LOGGER.warning("alpaca_news.fetch_error: %s", exc)
            return []

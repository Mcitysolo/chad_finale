"""Phase B Item 1 — Catalyst news provider.

Fetches recent headlines (Polygon primary, Yahoo fallback) and classifies
each per-symbol into a CatalystIntel record consumed by the entry-only
catalyst gate.

Design contract:
- Strict typing, stdlib only on the hot path.
- ``get_catalyst_intel`` never raises — provider failures fall through to
  Yahoo and ultimately to a "none" record so the entry gate fails open.
- ``polygon_fetcher`` and ``yahoo_fetcher`` are dependency-injection seams
  so tests never touch the network.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"
POLYGON_TIMEOUT_SECONDS = 6.0
POLYGON_DEFAULT_LIMIT = 10

_HIGH_KEYWORDS = (
    "earnings", "eps", "revenue", "beat", "miss",
    "fda", "approval", "approved", "rejected",
    "merger", "acquisition", "acquires", "buyout",
    "guidance", "raised", "lowered", "cut",
    "bankruptcy", "default", "halt", "suspended", "delisted",
)

_MEDIUM_KEYWORDS = (
    "upgrade", "downgrade", "initiated", "reiterate", "price target",
    "partnership", "contract", "deal",
    "executive", "ceo", "cfo", "appointed", "resigned",
    "investigation", "recall", "lawsuit", "settlement",
)

_BULLISH_KEYWORDS = (
    "beat", "beats", "exceeded", "surpassed",
    "approved", "approval", "upgrade", "upgraded",
    "acquires", "merger", "raises guidance", "raised",
    "above estimates", "record",
    "partnership", "new contract", "appointed",
    "positive", "upbeat",
)

_BEARISH_KEYWORDS = (
    "miss", "misses", "missed", "below estimates",
    "rejected", "rejection",
    "downgrade", "downgraded", "cut", "cuts",
    "warning", "warn", "warns",
    "bankruptcy", "default", "halt", "halted", "delisted",
    "suspend", "suspended",
    "lawsuit", "probe", "investigation",
    "lowered guidance", "lowered",
    "negative", "concern",
)


@dataclass(frozen=True)
class NewsArticle:
    headline: str
    published_utc: str
    source: str = ""
    url: str = ""


@dataclass(frozen=True)
class CatalystIntel:
    symbol: str
    has_catalyst: bool
    catalyst_strength: str
    catalyst_direction: str
    news_count: int
    catalyst_count: int
    latest_headline: str
    latest_ts_utc: str
    catalyst_categories: List[str] = field(default_factory=list)
    source_provider: str = "unknown"


def _read_polygon_key() -> Optional[str]:
    env_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if env_key:
        return env_key
    env_path = Path("/etc/chad/polygon.env")
    if not env_path.is_file():
        return None
    try:
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "POLYGON_API_KEY":
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    except Exception:
        return None
    return None


def _polygon_news(
    symbol: str,
    api_key: str,
    lookback_hours: int = 24,
    limit: int = POLYGON_DEFAULT_LIMIT,
) -> List[NewsArticle]:
    if not api_key or not symbol:
        return []
    params = {
        "ticker": symbol,
        "limit": str(int(limit)),
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }
    url = f"{POLYGON_NEWS_URL}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CHAD/1.0"})
        with urllib.request.urlopen(req, timeout=POLYGON_TIMEOUT_SECONDS) as resp:
            raw = resp.read()
    except Exception:
        return []
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    out: List[NewsArticle] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        headline = str(entry.get("title") or "").strip()
        if not headline:
            continue
        out.append(
            NewsArticle(
                headline=headline,
                published_utc=str(entry.get("published_utc") or "").strip(),
                source=str(entry.get("publisher", {}).get("name") or "polygon").strip()
                if isinstance(entry.get("publisher"), dict)
                else "polygon",
                url=str(entry.get("article_url") or "").strip(),
            )
        )
    return out


def _yahoo_news(symbol: str, limit: int = POLYGON_DEFAULT_LIMIT) -> List[NewsArticle]:
    try:
        from chad.market_data.yahoo_news_provider import YahooNewsProvider
    except Exception:
        return []
    try:
        provider = YahooNewsProvider()
        items = provider.get_headlines(symbols=[symbol], limit=int(limit))
    except Exception:
        return []
    out: List[NewsArticle] = []
    for item in items or []:
        try:
            headline = str(getattr(item, "headline", "") or "").strip()
            if not headline:
                continue
            out.append(
                NewsArticle(
                    headline=headline,
                    published_utc=str(getattr(item, "published_utc", "") or "").strip(),
                    source=str(getattr(item, "source", "") or "yahoo").strip(),
                    url=str(getattr(item, "url", "") or "").strip(),
                )
            )
        except Exception:
            continue
    return out


def _normalize_yahoo_record(rec: Any) -> Optional[NewsArticle]:
    """Coerce miscellaneous Yahoo-shaped records into NewsArticle.

    Accepts NewsItem dataclasses, plain dicts, or duck-typed objects with the
    common attribute names.
    """
    if rec is None:
        return None
    if isinstance(rec, NewsArticle):
        return rec
    if isinstance(rec, dict):
        headline = str(
            rec.get("headline") or rec.get("title") or ""
        ).strip()
        if not headline:
            return None
        return NewsArticle(
            headline=headline,
            published_utc=str(
                rec.get("published_utc")
                or rec.get("published_at")
                or rec.get("ts_utc")
                or ""
            ).strip(),
            source=str(rec.get("source") or rec.get("provider") or "").strip(),
            url=str(rec.get("url") or rec.get("link") or "").strip(),
        )
    headline = str(getattr(rec, "headline", "") or getattr(rec, "title", "") or "").strip()
    if not headline:
        return None
    return NewsArticle(
        headline=headline,
        published_utc=str(
            getattr(rec, "published_utc", "")
            or getattr(rec, "published_at", "")
            or getattr(rec, "ts_utc", "")
            or ""
        ).strip(),
        source=str(getattr(rec, "source", "") or getattr(rec, "provider", "") or "").strip(),
        url=str(getattr(rec, "url", "") or getattr(rec, "link", "") or "").strip(),
    )


def _classify_article(article: NewsArticle) -> Tuple[str, str]:
    """Return (strength, direction) for a single article.

    strength ∈ {"high","medium","none"}
    direction ∈ {"bullish","bearish","neutral"}
    """
    text = (article.headline or "").lower()
    if not text:
        return "none", "neutral"
    strength = "none"
    if any(k in text for k in _HIGH_KEYWORDS):
        strength = "high"
    elif any(k in text for k in _MEDIUM_KEYWORDS):
        strength = "medium"
    bullish_hit = any(k in text for k in _BULLISH_KEYWORDS)
    bearish_hit = any(k in text for k in _BEARISH_KEYWORDS)
    if bullish_hit and not bearish_hit:
        direction = "bullish"
    elif bearish_hit and not bullish_hit:
        direction = "bearish"
    else:
        direction = "neutral"
    return strength, direction


def _categorize(article: NewsArticle) -> List[str]:
    text = (article.headline or "").lower()
    cats: List[str] = []
    if any(k in text for k in ("earnings", "eps", "revenue", "guidance")):
        cats.append("earnings")
    if any(k in text for k in ("fda", "approval", "approved", "rejected")):
        cats.append("regulatory")
    if any(k in text for k in ("merger", "acquisition", "acquires", "buyout")):
        cats.append("m&a")
    if any(k in text for k in ("upgrade", "downgrade", "initiated", "reiterate", "price target")):
        cats.append("ratings")
    if any(k in text for k in ("halt", "halted", "suspended", "delisted", "bankruptcy", "default")):
        cats.append("distress")
    if any(k in text for k in ("lawsuit", "investigation", "probe", "settlement", "recall")):
        cats.append("legal")
    return cats


_STRENGTH_RANK = {"high": 3, "medium": 2, "low": 1, "none": 0}


def build_catalyst_intel(
    symbol: str,
    articles: List[NewsArticle],
    source_provider: str = "unknown",
) -> CatalystIntel:
    sym = (symbol or "").strip().upper()
    if not articles:
        return CatalystIntel(
            symbol=sym,
            has_catalyst=False,
            catalyst_strength="none",
            catalyst_direction="none",
            news_count=0,
            catalyst_count=0,
            latest_headline="",
            latest_ts_utc="",
            catalyst_categories=[],
            source_provider=source_provider if source_provider else "none",
        )

    best_strength = "none"
    best_direction = "neutral"
    catalyst_count = 0
    categories: List[str] = []
    seen_cats: set[str] = set()

    for art in articles:
        strength, direction = _classify_article(art)
        if strength in ("high", "medium"):
            catalyst_count += 1
        for cat in _categorize(art):
            if cat not in seen_cats:
                seen_cats.add(cat)
                categories.append(cat)
        if _STRENGTH_RANK.get(strength, 0) > _STRENGTH_RANK.get(best_strength, 0):
            best_strength = strength
            best_direction = direction
        elif _STRENGTH_RANK.get(strength, 0) == _STRENGTH_RANK.get(best_strength, 0):
            if best_direction == "neutral" and direction != "neutral":
                best_direction = direction

    has_catalyst = best_strength in ("high", "medium")
    if not has_catalyst:
        best_direction = "none"

    latest = articles[0]
    return CatalystIntel(
        symbol=sym,
        has_catalyst=has_catalyst,
        catalyst_strength=best_strength,
        catalyst_direction=best_direction,
        news_count=len(articles),
        catalyst_count=catalyst_count,
        latest_headline=latest.headline,
        latest_ts_utc=latest.published_utc,
        catalyst_categories=categories,
        source_provider=source_provider if source_provider else "unknown",
    )


def get_catalyst_intel(
    symbols: List[str],
    lookback_hours: int = 24,
    *,
    polygon_fetcher: Optional[Callable[[str, str, int, int], List[NewsArticle]]] = None,
    yahoo_fetcher: Optional[Callable[[str, int], List[NewsArticle]]] = None,
) -> Dict[str, CatalystIntel]:
    """Fetch and classify per-symbol catalyst intel.

    Polygon is tried first when an API key is available; on empty/error the
    Yahoo fallback is used. Failures never raise — symbols with no data
    receive an empty (``has_catalyst=False``) record so the entry gate fails
    open downstream.
    """
    out: Dict[str, CatalystIntel] = {}
    if not symbols:
        return out

    poly_fetch = polygon_fetcher if polygon_fetcher is not None else _polygon_news
    yahoo_fetch = yahoo_fetcher if yahoo_fetcher is not None else _yahoo_news

    api_key = "" if polygon_fetcher is not None else (_read_polygon_key() or "")

    for raw_sym in symbols:
        sym = (raw_sym or "").strip().upper()
        if not sym:
            continue

        articles: List[NewsArticle] = []
        provider_used = "none"

        if polygon_fetcher is not None or api_key:
            try:
                fetched = poly_fetch(sym, api_key, int(lookback_hours), POLYGON_DEFAULT_LIMIT)
                articles = list(fetched or [])
                if articles:
                    provider_used = "polygon"
            except Exception:
                articles = []

        if not articles:
            try:
                fetched = yahoo_fetch(sym, POLYGON_DEFAULT_LIMIT)
                normalized = [_normalize_yahoo_record(r) for r in (fetched or [])]
                articles = [a for a in normalized if a is not None]
                if articles:
                    provider_used = "yahoo"
            except Exception:
                articles = []

        out[sym] = build_catalyst_intel(sym, articles, source_provider=provider_used)

    return out


__all__ = [
    "NewsArticle",
    "CatalystIntel",
    "build_catalyst_intel",
    "get_catalyst_intel",
    "_classify_article",
    "_polygon_news",
    "_yahoo_news",
    "_read_polygon_key",
]

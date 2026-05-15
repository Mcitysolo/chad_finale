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
import re
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
    symbols: List[str] = field(default_factory=list)


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
    symbol_relevance: str = "unknown"
    relevant_news_count: int = 0
    # True only when headline ticker token/alias confirms the symbol — safe
    # for the catalyst gate to use as a blocking input. Provider ticker tags
    # alone (e.g., Polygon's article.tickers) never set this to True.
    confirmed_gate_relevant: bool = False


_SYMBOL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "AAPL": ("Apple",),
    "AMZN": ("Amazon",),
    "AVGO": ("Broadcom",),
    "BAC": ("Bank of America",),
    "GOOGL": ("Alphabet", "Google"),
    "GOOG": ("Alphabet", "Google"),
    "LLY": ("Eli Lilly", "Lilly"),
    "MSFT": ("Microsoft",),
    "NVDA": ("Nvidia", "NVIDIA"),
    "SPY": ("S&P 500", "SPDR S&P 500"),
    "QQQ": ("Nasdaq 100", "Invesco QQQ"),
    "GLD": ("SPDR Gold", "Gold ETF"),
}

_ANALYST_BANKS = frozenset(
    {"BAC", "JPM", "MS", "GS", "C", "WFC", "BCS", "DB", "UBS", "HSBC", "RBC"}
)

_ANALYST_VERBS = (
    "raises", "raise", "lifts", "lifted",
    "cuts", "cut", "lowers", "lowered",
    "resets", "reset",
    "downgrade", "downgrades", "downgrading",
    "upgrade", "upgrades", "upgrading",
    "initiates", "initiated", "initiating",
    "reiterates", "reiterated",
    "boosts", "boosted",
    "trims", "trimmed",
)

_ANALYST_TARGET_KEYWORDS = (
    "price target",
    "target price",
    "stock price target",
    "rating",
    "outlook",
    "to buy",
    "to sell",
    "to hold",
)

_BROAD_MARKET_KEYWORDS = (
    "dow jones",
    "russell 2000",
    "broad market",
    "stocks rise", "stocks fall", "stocks gain", "stocks drop", "stocks soar", "stocks tumble",
    "stocks mixed", "stocks slip", "stocks edge",
    "market rally", "market sell", "market drops", "market plunges",
    "best stocks", "stocks to buy", "stocks to watch", "top stocks",
    "unstoppable stock", "unstoppable stocks",
    "ipo",
    "futures rise", "futures fall", "futures rally", "futures drop", "futures gain",
    "futures slip", "futures soar", "futures tumble", "futures mixed",
    "dow futures", "nasdaq futures", "s&p futures",
)


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _article_symbols(article: NewsArticle) -> set[str]:
    raw = getattr(article, "symbols", None) or []
    return {_normalize_symbol(str(s)) for s in raw if str(s).strip()}


def _headline_has_symbol(headline: str, symbol: str) -> bool:
    sym = _normalize_symbol(symbol)
    if not sym or not headline:
        return False
    if not re.fullmatch(r"[A-Z0-9.\-]+", sym):
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(sym) + r"(?![A-Za-z0-9])"
    return re.search(pattern, headline) is not None


def _headline_has_alias(headline: str, symbol: str) -> bool:
    sym = _normalize_symbol(symbol)
    aliases = _SYMBOL_ALIASES.get(sym, ())
    if not aliases or not headline:
        return False
    text = headline.lower()
    return any(alias.lower() in text for alias in aliases)


def _is_broad_market_headline(headline: str) -> bool:
    text = (headline or "").lower()
    if not text:
        return False
    return any(k in text for k in _BROAD_MARKET_KEYWORDS)


def _is_analyst_source_headline_for_bank(headline: str, symbol: str) -> bool:
    sym = _normalize_symbol(symbol)
    if sym not in _ANALYST_BANKS:
        return False
    text = (headline or "").lower()
    if not text:
        return False
    has_verb = any(re.search(r"\b" + re.escape(v) + r"\b", text) for v in _ANALYST_VERBS)
    has_target = any(k in text for k in _ANALYST_TARGET_KEYWORDS)
    return has_verb and has_target


def classify_symbol_relevance(symbol: str, article: NewsArticle) -> str:
    """Return one of {"direct", "weak", "broad_market", "unknown"}."""
    sym = _normalize_symbol(symbol)
    if not sym:
        return "unknown"

    if sym in _article_symbols(article):
        return "direct"

    headline = article.headline or ""
    if not headline.strip():
        return "unknown"

    analyst_source = _is_analyst_source_headline_for_bank(headline, sym)

    if _headline_has_symbol(headline, sym):
        return "weak" if analyst_source else "direct"

    if _headline_has_alias(headline, sym):
        return "weak" if analyst_source else "direct"

    if _is_broad_market_headline(headline):
        return "broad_market"

    return "unknown"


def _headline_confirms_symbol(symbol: str, article: NewsArticle) -> bool:
    """Return True only when the headline itself confirms the symbol.

    Provider ticker tags (Polygon's article.tickers, Yahoo's query echo) are
    not sufficient. The catalyst gate uses this signal to decide whether a
    news catalyst is precise enough to block a trade.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        return False
    headline = article.headline or ""
    if not headline.strip():
        return False
    if _is_analyst_source_headline_for_bank(headline, sym):
        return False
    if _headline_has_symbol(headline, sym):
        return True
    if _headline_has_alias(headline, sym):
        return True
    return False


_RELEVANCE_RANK = {"direct": 3, "weak": 2, "broad_market": 1, "unknown": 0}


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
        tickers_raw = entry.get("tickers")
        syms: List[str] = []
        if isinstance(tickers_raw, list):
            syms = [_normalize_symbol(str(t)) for t in tickers_raw if str(t).strip()]
        out.append(
            NewsArticle(
                headline=headline,
                published_utc=str(entry.get("published_utc") or "").strip(),
                source=str(entry.get("publisher", {}).get("name") or "polygon").strip()
                if isinstance(entry.get("publisher"), dict)
                else "polygon",
                url=str(entry.get("article_url") or "").strip(),
                symbols=syms,
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
            # NewsItem.symbols is the query echo, not per-article attribution —
            # do not propagate it; downstream relevance must use headline matching.
            out.append(
                NewsArticle(
                    headline=headline,
                    published_utc=str(getattr(item, "published_utc", "") or "").strip(),
                    source=str(getattr(item, "source", "") or "yahoo").strip(),
                    url=str(getattr(item, "url", "") or "").strip(),
                    symbols=[],
                )
            )
        except Exception:
            continue
    return out


def _extract_symbols_from_record(rec: Any) -> List[str]:
    candidates: List[Any] = []
    if isinstance(rec, dict):
        for key in ("symbols", "tickers", "ticker"):
            if key in rec:
                candidates.append(rec.get(key))
    else:
        for attr in ("symbols", "tickers", "ticker"):
            val = getattr(rec, attr, None)
            if val:
                candidates.append(val)
    for val in candidates:
        if isinstance(val, list):
            return [_normalize_symbol(str(s)) for s in val if str(s).strip()]
        if isinstance(val, tuple):
            return [_normalize_symbol(str(s)) for s in val if str(s).strip()]
        if isinstance(val, str) and val.strip():
            return [_normalize_symbol(val)]
    return []


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
            symbols=_extract_symbols_from_record(rec),
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
        symbols=_extract_symbols_from_record(rec),
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
    sym = _normalize_symbol(symbol)
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
            symbol_relevance="unknown",
            relevant_news_count=0,
            confirmed_gate_relevant=False,
        )

    direct_articles: List[NewsArticle] = []
    confirmed_articles: List[NewsArticle] = []
    best_observed_relevance = "unknown"
    for art in articles:
        rel = classify_symbol_relevance(sym, art)
        if _RELEVANCE_RANK.get(rel, 0) > _RELEVANCE_RANK.get(best_observed_relevance, 0):
            best_observed_relevance = rel
        if rel == "direct":
            direct_articles.append(art)
            if _headline_confirms_symbol(sym, art):
                confirmed_articles.append(art)

    symbol_relevance = "direct" if direct_articles else best_observed_relevance

    if not confirmed_articles:
        # No headline-confirmed relevance — provider ticker tags alone are
        # not enough to block trades. Keep news_count / symbol_relevance /
        # relevant_news_count informational; suppress catalyst eligibility.
        latest = direct_articles[0] if direct_articles else articles[0]
        return CatalystIntel(
            symbol=sym,
            has_catalyst=False,
            catalyst_strength="none",
            catalyst_direction="none",
            news_count=len(articles),
            catalyst_count=0,
            latest_headline=latest.headline,
            latest_ts_utc=latest.published_utc,
            catalyst_categories=[],
            source_provider=source_provider if source_provider else "unknown",
            symbol_relevance=symbol_relevance,
            relevant_news_count=len(direct_articles),
            confirmed_gate_relevant=False,
        )

    best_strength = "none"
    best_direction = "neutral"
    catalyst_count = 0
    categories: List[str] = []
    seen_cats: set[str] = set()

    for art in confirmed_articles:
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

    latest = confirmed_articles[0]
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
        symbol_relevance="direct",
        relevant_news_count=len(direct_articles),
        confirmed_gate_relevant=True,
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
    "classify_symbol_relevance",
    "_classify_article",
    "_headline_confirms_symbol",
    "_polygon_news",
    "_yahoo_news",
    "_read_polygon_key",
]

#!/usr/bin/env python3
"""
CHAD — Reddit Sentiment Provider

Monitors public Reddit posts for symbol mentions and sentiment signals.
Uses Reddit public JSON API (no API key required for read-only access).

Design:
- Caches results for 2 hours (Reddit rate-limits aggressively)
- Rate limit: 1 request per 2 seconds between subreddits
- Fails silently — never crashes advisory or orchestrator
- Writes state to runtime/reddit_sentiment.json for audit
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
import urllib.error
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("chad.intel.reddit_sentiment")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
STATE_PATH = RUNTIME_DIR / "reddit_sentiment.json"
CACHE_TTL_SEC = 2 * 3600  # 2 hours
REQUEST_DELAY_SEC = 2.0  # Rate limit between subreddit requests
REQUEST_TIMEOUT_SEC = 10

SUBREDDITS = ["wallstreetbets", "investing", "stocks", "options", "StockMarket"]

USER_AGENT = "CHAD-TradingSystem/1.0 (market-sentiment-research)"

# Simple word-based sentiment lexicon
POSITIVE_WORDS = frozenset({
    "moon", "buy", "calls", "bullish", "rocket", "long", "up", "green",
    "gains", "pump", "rally", "breakout", "yolo", "diamond", "tendies",
    "soar", "surge", "rip", "bull",
})

NEGATIVE_WORDS = frozenset({
    "puts", "short", "bearish", "crash", "dump", "red", "loss", "down",
    "sell", "drop", "tank", "plunge", "bear", "fade", "bag", "rekt",
    "collapse", "drill", "recession",
})

HYPE_MENTION_THRESHOLD = 50


@dataclass(frozen=True)
class SentimentSignal:
    symbol: str
    mention_count: int
    positive_count: int
    negative_count: int
    sentiment_score: float  # -1.0 to +1.0
    signal: str  # BULLISH, BEARISH, NEUTRAL, HYPE
    ts_utc: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        LOG.warning("Failed to write %s: %s", path, exc)


def _score_text(text: str) -> tuple[int, int]:
    """Count positive and negative sentiment words in text.

    Returns (positive_count, negative_count).
    """
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return pos, neg


def _classify(mention_count: int, sentiment_score: float) -> str:
    if mention_count >= HYPE_MENTION_THRESHOLD and sentiment_score > 0.1:
        return "HYPE"
    if sentiment_score > 0.2:
        return "BULLISH"
    if sentiment_score < -0.2:
        return "BEARISH"
    return "NEUTRAL"


def _compute_score(positive: int, negative: int) -> float:
    total = positive + negative
    if total == 0:
        return 0.0
    return round((positive - negative) / total, 4)


class RedditSentimentProvider:
    """
    Reddit sentiment provider using public JSON API.

    All methods fail silently — returns empty dicts on any error.
    """

    def __init__(self, runtime_dir: Optional[Path] = None) -> None:
        self._runtime_dir = runtime_dir or RUNTIME_DIR
        self._state_path = self._runtime_dir / "reddit_sentiment.json"
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._load_cache()

    def _load_cache(self) -> None:
        state = _read_json(self._state_path)
        cached = state.get("cache", {})
        ts = state.get("cache_ts", 0.0)
        if isinstance(cached, dict) and isinstance(ts, (int, float)):
            self._cache = cached
            self._cache_ts = float(ts)

    def _save_state(self, signals: Dict[str, SentimentSignal]) -> None:
        state = {
            "signals": {sym: asdict(sig) for sym, sig in signals.items()},
            "cache": self._cache,
            "cache_ts": self._cache_ts,
            "last_updated_utc": _utc_now_iso(),
        }
        _write_json(self._state_path, state)

    def _is_cache_fresh(self) -> bool:
        if not self._cache:
            return False
        return (time.time() - self._cache_ts) < CACHE_TTL_SEC

    def get_sentiment(self, symbol: str) -> Optional[SentimentSignal]:
        """Get Reddit sentiment for a single symbol. Fails silently."""
        result = self.get_batch_sentiment([symbol])
        return result.get(symbol)

    def get_batch_sentiment(
        self, symbols: List[str],
    ) -> Dict[str, SentimentSignal]:
        """Get Reddit sentiment for multiple symbols. Fails silently."""
        if not symbols:
            return {}

        # Check cache
        if self._is_cache_fresh():
            cached_signals = {}
            all_cached = True
            for sym in symbols:
                cached = self._cache.get(sym)
                if cached and isinstance(cached, dict):
                    try:
                        cached_signals[sym] = SentimentSignal(**cached)
                    except Exception:
                        all_cached = False
                        break
                else:
                    all_cached = False
                    break
            if all_cached:
                LOG.debug("Returning cached reddit sentiment for %s", symbols)
                return cached_signals

        try:
            return self._fetch_sentiment(symbols)
        except Exception as exc:
            LOG.warning("Reddit sentiment fetch failed: %s", exc)
            return {}

    def _fetch_reddit_posts(self, symbol: str, subreddit: str) -> List[Dict[str, Any]]:
        """Fetch posts mentioning symbol from a subreddit. Returns raw post data."""
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={symbol}&sort=new&limit=25&restrict_sr=on&t=week"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        posts = []
        children = data.get("data", {}).get("children", [])
        for child in children:
            post_data = child.get("data", {})
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            posts.append({
                "title": title,
                "selftext": selftext,
                "score": post_data.get("score", 0),
                "num_comments": post_data.get("num_comments", 0),
            })
        return posts

    def _fetch_sentiment(
        self, symbols: List[str],
    ) -> Dict[str, SentimentSignal]:
        """Fetch sentiment from Reddit for all symbols."""
        signals: Dict[str, SentimentSignal] = {}

        for sym in symbols:
            total_mentions = 0
            total_pos = 0
            total_neg = 0

            for sub in SUBREDDITS:
                try:
                    posts = self._fetch_reddit_posts(sym, sub)
                    for post in posts:
                        text = f"{post.get('title', '')} {post.get('selftext', '')}"
                        # Only count if symbol actually appears in text
                        if sym.upper() in text.upper() or f"${sym.upper()}" in text.upper():
                            total_mentions += 1
                            pos, neg = _score_text(text)
                            total_pos += pos
                            total_neg += neg
                except Exception as exc:
                    LOG.debug("Reddit fetch failed for %s/%s: %s", sym, sub, exc)
                    continue

                # Rate limit between subreddit requests
                time.sleep(REQUEST_DELAY_SEC)

            score = _compute_score(total_pos, total_neg)
            signal_label = _classify(total_mentions, score)

            sig = SentimentSignal(
                symbol=sym,
                mention_count=total_mentions,
                positive_count=total_pos,
                negative_count=total_neg,
                sentiment_score=score,
                signal=signal_label,
                ts_utc=_utc_now_iso(),
            )
            signals[sym] = sig
            self._cache[sym] = asdict(sig)

        self._cache_ts = time.time()
        self._save_state(signals)
        return signals

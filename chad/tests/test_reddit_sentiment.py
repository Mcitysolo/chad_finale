"""Tests for RedditSentimentProvider."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chad.intel.reddit_sentiment import (
    HYPE_MENTION_THRESHOLD,
    RedditSentimentProvider,
    SentimentSignal,
    _classify,
    _compute_score,
    _score_text,
)


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------


class TestScoreText:
    def test_positive_words(self):
        pos, neg = _score_text("Buy calls, to the moon! Bullish gains!")
        assert pos >= 4
        assert neg == 0

    def test_negative_words(self):
        pos, neg = _score_text("Puts, short, crash, dump, bearish!")
        assert neg >= 4
        assert pos == 0

    def test_mixed_words(self):
        pos, neg = _score_text("I'm bullish on the long term but short term crash possible")
        assert pos > 0
        assert neg > 0

    def test_empty_text(self):
        pos, neg = _score_text("")
        assert pos == 0
        assert neg == 0

    def test_no_sentiment_words(self):
        pos, neg = _score_text("The weather is nice today")
        assert pos == 0
        assert neg == 0


class TestComputeScore:
    def test_all_positive(self):
        assert _compute_score(5, 0) == 1.0

    def test_all_negative(self):
        assert _compute_score(0, 5) == -1.0

    def test_balanced(self):
        assert _compute_score(3, 3) == 0.0

    def test_no_words(self):
        assert _compute_score(0, 0) == 0.0

    def test_mostly_positive(self):
        score = _compute_score(7, 3)
        assert 0.3 < score < 0.5


class TestClassify:
    def test_hype_high_mentions_positive(self):
        assert _classify(60, 0.3) == "HYPE"

    def test_hype_threshold_exact(self):
        assert _classify(HYPE_MENTION_THRESHOLD, 0.2) == "HYPE"

    def test_bullish(self):
        assert _classify(10, 0.5) == "BULLISH"

    def test_bearish(self):
        assert _classify(10, -0.5) == "BEARISH"

    def test_neutral_low_score(self):
        assert _classify(10, 0.0) == "NEUTRAL"

    def test_neutral_mixed(self):
        assert _classify(10, 0.1) == "NEUTRAL"

    def test_high_mentions_negative_not_hype(self):
        # HYPE only triggers on positive sentiment
        assert _classify(100, -0.5) == "BEARISH"


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCacheBehavior:
    def test_cache_fresh(self, tmp_path: Path):
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        sig = SentimentSignal(
            symbol="SPY", mention_count=10, positive_count=5, negative_count=2,
            sentiment_score=0.4286, signal="BULLISH", ts_utc="2026-04-05T10:00:00Z",
        )
        from dataclasses import asdict
        provider._cache = {"SPY": asdict(sig)}
        provider._cache_ts = time.time()  # Fresh

        result = provider.get_batch_sentiment(["SPY"])
        assert "SPY" in result
        assert result["SPY"].signal == "BULLISH"

    def test_cache_stale_triggers_fetch(self, tmp_path: Path):
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        provider._cache = {"SPY": {"symbol": "SPY", "mention_count": 1,
                                    "positive_count": 0, "negative_count": 0,
                                    "sentiment_score": 0.0, "signal": "NEUTRAL",
                                    "ts_utc": "2026-04-01T00:00:00Z"}}
        provider._cache_ts = time.time() - 3 * 3600  # 3 hours ago (> 2 hour TTL)

        # Will try to fetch and fail (no network in tests) — returns empty
        with patch.object(provider, "_fetch_sentiment", side_effect=Exception("no network")):
            result = provider.get_batch_sentiment(["SPY"])
        assert result == {}


# ---------------------------------------------------------------------------
# Fail-silent behavior
# ---------------------------------------------------------------------------


class TestFailSilent:
    def test_network_error_returns_neutral(self, tmp_path: Path):
        """When all HTTP calls fail, we still get a signal with 0 mentions (NEUTRAL)."""
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        with patch("chad.intel.reddit_sentiment.urllib.request.urlopen",
                   side_effect=Exception("Connection refused")):
            with patch("chad.intel.reddit_sentiment.time.sleep"):
                result = provider.get_batch_sentiment(["SPY"])
        # Should return a signal with 0 mentions rather than crashing
        assert "SPY" in result
        assert result["SPY"].mention_count == 0
        assert result["SPY"].signal == "NEUTRAL"

    def test_fetch_raises_returns_empty(self, tmp_path: Path):
        """When _fetch_sentiment itself raises, get_batch_sentiment returns empty."""
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        with patch.object(provider, "_fetch_sentiment", side_effect=Exception("fatal")):
            result = provider.get_batch_sentiment(["SPY"])
        assert result == {}

    def test_empty_symbols(self, tmp_path: Path):
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        result = provider.get_batch_sentiment([])
        assert result == {}

    def test_single_symbol_fail_silent(self, tmp_path: Path):
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        with patch.object(provider, "_fetch_sentiment", side_effect=Exception("fail")):
            result = provider.get_sentiment("SPY")
        assert result is None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_save_and_load(self, tmp_path: Path):
        provider = RedditSentimentProvider(runtime_dir=tmp_path)
        sig = SentimentSignal(
            symbol="NVDA", mention_count=25, positive_count=10, negative_count=3,
            sentiment_score=0.5385, signal="BULLISH", ts_utc="2026-04-05T10:00:00Z",
        )
        # Populate cache as _fetch_sentiment would
        from dataclasses import asdict
        provider._cache = {"NVDA": asdict(sig)}
        provider._cache_ts = time.time()
        provider._save_state({"NVDA": sig})

        # Load in a new instance
        provider2 = RedditSentimentProvider(runtime_dir=tmp_path)
        assert "NVDA" in provider2._cache


# ---------------------------------------------------------------------------
# Mock HTTP integration
# ---------------------------------------------------------------------------


class TestMockFetch:
    def _mock_reddit_response(self, symbol: str):
        """Build a fake Reddit search JSON response."""
        return json.dumps({
            "data": {
                "children": [
                    {"data": {"title": f"{symbol} to the moon! Buy calls!",
                              "selftext": "Bullish gains ahead", "score": 100,
                              "num_comments": 50}},
                    {"data": {"title": f"Why {symbol} is going up",
                              "selftext": "Long term bull case", "score": 50,
                              "num_comments": 20}},
                    {"data": {"title": f"Selling my {symbol} puts",
                              "selftext": "This is going down, bearish crash",
                              "score": 30, "num_comments": 10}},
                ]
            }
        }).encode("utf-8")

    def test_fetch_parses_posts(self, tmp_path: Path):
        provider = RedditSentimentProvider(runtime_dir=tmp_path)

        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_reddit_response("SPY")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("chad.intel.reddit_sentiment.urllib.request.urlopen", return_value=mock_resp):
            with patch("chad.intel.reddit_sentiment.time.sleep"):
                posts = provider._fetch_reddit_posts("SPY", "wallstreetbets")

        assert len(posts) == 3
        assert "moon" in posts[0]["title"].lower()

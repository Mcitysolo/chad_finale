#!/usr/bin/env python3
"""Tests for chad.market_data.yahoo_news_provider."""

import json
from unittest.mock import patch, MagicMock

import pytest

from chad.market_data.yahoo_news_provider import (
    YahooNewsProvider,
    NewsItem,
    _parse_search_json,
    _parse_rss_xml,
    _USER_AGENT,
)


# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_SEARCH_JSON = json.dumps({
    "news": [
        {
            "title": "S&P 500 rallies on strong jobs data",
            "publisher": "Reuters",
            "link": "https://example.com/1",
            "providerPublishTime": 1712188800,
        },
        {
            "title": "Fed signals patience on rate cuts",
            "publisher": "Bloomberg",
            "link": "https://example.com/2",
            "providerPublishTime": 1712185200,
        },
    ]
}).encode()

SAMPLE_RSS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
<title>Yahoo Finance SPY</title>
<item>
  <title>Oil prices surge on supply concerns</title>
  <link>https://example.com/rss1</link>
  <pubDate>Thu, 04 Apr 2026 12:00:00 +0000</pubDate>
</item>
<item>
  <title>Tech stocks lead market higher</title>
  <link>https://example.com/rss2</link>
  <pubDate>Thu, 04 Apr 2026 11:00:00 +0000</pubDate>
</item>
</channel>
</rss>"""


# ── Unit tests: JSON parsing ─────────────────────────────────────────

def test_parse_search_json_extracts_headlines():
    items = _parse_search_json(SAMPLE_SEARCH_JSON, ["SPY"])
    assert len(items) == 2
    assert items[0].headline == "S&P 500 rallies on strong jobs data"
    assert items[0].source == "yahoo_finance"
    assert items[0].symbols == ["SPY"]
    assert items[0].summary == "Reuters"
    assert items[0].url == "https://example.com/1"
    assert items[0].published_utc  # non-empty


def test_parse_search_json_empty_news():
    data = json.dumps({"news": []}).encode()
    assert _parse_search_json(data, ["SPY"]) == []


def test_parse_search_json_missing_title_skipped():
    data = json.dumps({"news": [{"title": "", "link": "x"}]}).encode()
    assert _parse_search_json(data, ["AAPL"]) == []


# ── Unit tests: RSS parsing ──────────────────────────────────────────

def test_parse_rss_xml_extracts_headlines():
    items = _parse_rss_xml(SAMPLE_RSS_XML, ["SPY"])
    assert len(items) == 2
    assert items[0].headline == "Oil prices surge on supply concerns"
    assert items[0].source == "yahoo_finance_rss"
    assert items[0].symbols == ["SPY"]
    assert items[0].url == "https://example.com/rss1"


def test_parse_rss_xml_empty():
    data = b"<rss><channel></channel></rss>"
    assert _parse_rss_xml(data, ["SPY"]) == []


# ── Integration tests: YahooNewsProvider with mocked HTTP ─────────────

@patch("chad.market_data.yahoo_news_provider._make_request")
def test_get_headlines_uses_search_api(mock_req):
    mock_req.return_value = SAMPLE_SEARCH_JSON
    provider = YahooNewsProvider()
    items = provider.get_headlines(["SPY"], limit=2)
    assert len(items) == 2
    assert items[0].headline == "S&P 500 rallies on strong jobs data"
    # Should call search API, not RSS
    mock_req.assert_called_once()
    assert "finance/search" in mock_req.call_args[0][0]


@patch("chad.market_data.yahoo_news_provider._make_request")
def test_rss_fallback_on_search_failure(mock_req):
    """When search API fails, falls back to RSS."""
    mock_req.side_effect = [Exception("search down"), SAMPLE_RSS_XML]
    provider = YahooNewsProvider()
    items = provider.get_headlines(["SPY"], limit=2)
    assert len(items) == 2
    assert items[0].headline == "Oil prices surge on supply concerns"
    assert mock_req.call_count == 2


@patch("chad.market_data.yahoo_news_provider._make_request")
def test_empty_list_on_total_failure(mock_req):
    """Both sources fail -> empty list, no crash."""
    mock_req.side_effect = Exception("network down")
    provider = YahooNewsProvider()
    items = provider.get_headlines(["SPY"], limit=5)
    assert items == []


@patch("chad.market_data.yahoo_news_provider._make_request")
def test_get_market_headlines(mock_req):
    mock_req.return_value = SAMPLE_SEARCH_JSON
    provider = YahooNewsProvider()
    items = provider.get_market_headlines(limit=2)
    assert len(items) == 2
    assert "finance/search" in mock_req.call_args[0][0]
    assert "SPY" in mock_req.call_args[0][0]


def test_configured_always_true():
    provider = YahooNewsProvider()
    assert provider.configured is True


def test_newsitem_to_dict():
    item = NewsItem(
        headline="Test", summary="Sum", url="http://x",
        published_utc="2026-01-01", symbols=["SPY"],
    )
    d = item.to_dict()
    assert d["headline"] == "Test"
    assert d["source"] == "yahoo_finance"


def test_user_agent_header_is_set():
    """Verify the User-Agent constant exists and is non-empty."""
    assert _USER_AGENT
    assert "Mozilla" in _USER_AGENT or "CHAD" in _USER_AGENT


@patch("chad.market_data.yahoo_news_provider._make_request")
def test_defaults_to_spy_when_no_symbols(mock_req):
    mock_req.return_value = SAMPLE_SEARCH_JSON
    provider = YahooNewsProvider()
    provider.get_headlines(symbols=None, limit=2)
    assert "SPY" in mock_req.call_args[0][0]

"""Tests for chad.market_data.fmp_client.

Phase B FMP Phase 1 — scaffolding-only coverage. No live FMP call is
made; every transport hop is intercepted with an injected opener so the
suite is hermetic.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

from chad.market_data import fmp_client as fmp_mod
from chad.market_data.fmp_client import (
    FMP_BASE_URL,
    FMP_PLACEHOLDER_KEY,
    FMPAnalystEstimate,
    FMPClient,
    FMPEarningsEvent,
    FMPPriceTargetConsensus,
    FMPProfile,
    FMPQuote,
    FMPSecFiling,
    _read_fmp_key,
)


def _make_recording_opener(payload: Any) -> Tuple[Callable[[str], bytes], List[str]]:
    """Return (opener, captured_urls).

    The opener serializes ``payload`` once and records the requested URL
    for later assertions on query params.
    """
    captured: List[str] = []
    encoded = json.dumps(payload).encode("utf-8")

    def _opener(url: str) -> bytes:
        captured.append(url)
        return encoded

    return _opener, captured


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------


def test_read_fmp_key_env_before_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FMP_API_KEY", "env-key-xyz")
    fake_env = tmp_path / "fmp.env"
    fake_env.write_text("FMP_API_KEY=file-key-abc\n", encoding="utf-8")
    monkeypatch.setattr(fmp_mod, "FMP_ENV_PATH", fake_env)

    assert _read_fmp_key() == "env-key-xyz"


def test_read_fmp_key_placeholder_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FMP_API_KEY", FMP_PLACEHOLDER_KEY)
    placeholder_file = tmp_path / "fmp.env"
    placeholder_file.write_text(
        f"FMP_API_KEY={FMP_PLACEHOLDER_KEY}\n", encoding="utf-8"
    )
    monkeypatch.setattr(fmp_mod, "FMP_ENV_PATH", placeholder_file)

    assert _read_fmp_key() is None


# ---------------------------------------------------------------------------
# Endpoint parsing
# ---------------------------------------------------------------------------


def test_get_quote_parses_stable_response() -> None:
    payload = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "price": 187.45,
            "changePercentage": 1.23,
            "volume": 50_000_000,
            "averageVolume": 60_000_000,
            "marketCap": 2_900_000_000_000,
            "timestamp": 1700000000,
        }
    ]
    opener, urls = _make_recording_opener(payload)
    client = FMPClient(api_key="k", opener=opener)

    out = client.get_quote("aapl")
    assert len(out) == 1
    quote = out[0]
    assert isinstance(quote, FMPQuote)
    assert quote.symbol == "AAPL"
    assert quote.name == "Apple Inc."
    assert quote.price == 187.45
    assert quote.change_percentage == 1.23
    assert quote.volume == 50_000_000
    assert quote.average_volume == 60_000_000
    assert quote.market_cap == 2_900_000_000_000
    assert quote.timestamp == 1700000000

    assert urls and urls[0].startswith(f"{FMP_BASE_URL}/quote?")
    assert "symbol=AAPL" in urls[0]
    assert "apikey=k" in urls[0]


def test_get_profile_parses_stable_response() -> None:
    payload = [
        {
            "symbol": "AAPL",
            "companyName": "Apple Inc.",
            "price": 187.45,
            "marketCap": 2_900_000_000_000,
            "beta": 1.25,
            "averageVolume": 60_000_000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NASDAQ",
            "currency": "USD",
        }
    ]
    opener, urls = _make_recording_opener(payload)
    client = FMPClient(api_key="k", opener=opener)

    out = client.get_profile("AAPL")
    assert len(out) == 1
    profile = out[0]
    assert isinstance(profile, FMPProfile)
    assert profile.symbol == "AAPL"
    assert profile.company_name == "Apple Inc."
    assert profile.market_cap == 2_900_000_000_000
    assert profile.beta == 1.25
    assert profile.average_volume == 60_000_000
    assert profile.sector == "Technology"
    assert profile.industry == "Consumer Electronics"
    assert profile.exchange == "NASDAQ"
    assert profile.currency == "USD"
    assert urls and "/profile?" in urls[0]


def test_get_earnings_calendar_parses_stable_response() -> None:
    payload = [
        {
            "symbol": "AAPL",
            "date": "2026-05-01",
            "epsActual": 1.40,
            "epsEstimated": 1.35,
            "revenueActual": 95_000_000_000,
            "revenueEstimated": 94_500_000_000,
            "lastUpdated": "2026-05-01",
        }
    ]
    opener, urls = _make_recording_opener(payload)
    client = FMPClient(api_key="k", opener=opener)

    out = client.get_earnings_calendar("2026-05-01", "2026-05-31")
    assert len(out) == 1
    event = out[0]
    assert isinstance(event, FMPEarningsEvent)
    assert event.symbol == "AAPL"
    assert event.eps_actual == 1.40
    assert event.eps_estimated == 1.35
    assert event.revenue_actual == 95_000_000_000
    assert event.revenue_estimated == 94_500_000_000
    assert event.last_updated == "2026-05-01"

    assert urls and "/earnings-calendar?" in urls[0]
    assert "from=2026-05-01" in urls[0]
    assert "to=2026-05-31" in urls[0]


def test_get_price_target_consensus_parses_stable_response() -> None:
    payload = [
        {
            "symbol": "AAPL",
            "targetHigh": 250.0,
            "targetLow": 150.0,
            "targetConsensus": 210.0,
            "targetMedian": 205.0,
        }
    ]
    opener, urls = _make_recording_opener(payload)
    client = FMPClient(api_key="k", opener=opener)

    out = client.get_price_target_consensus("AAPL")
    assert len(out) == 1
    ptc = out[0]
    assert isinstance(ptc, FMPPriceTargetConsensus)
    assert ptc.target_high == 250.0
    assert ptc.target_low == 150.0
    assert ptc.target_consensus == 210.0
    assert ptc.target_median == 205.0
    assert urls and "/price-target-consensus?" in urls[0]


def test_get_analyst_estimates_annual_sends_period_and_parses() -> None:
    payload = [
        {
            "symbol": "AAPL",
            "date": "2027",
            "revenueLow": 380_000_000_000,
            "revenueHigh": 420_000_000_000,
            "revenueAvg": 400_000_000_000,
            "epsLow": 6.5,
            "epsHigh": 7.5,
            "epsAvg": 7.0,
        }
    ]
    opener, urls = _make_recording_opener(payload)
    client = FMPClient(api_key="k", opener=opener)

    out = client.get_analyst_estimates_annual("AAPL")
    assert len(out) == 1
    est = out[0]
    assert isinstance(est, FMPAnalystEstimate)
    assert est.date == "2027"
    assert est.revenue_avg == 400_000_000_000
    assert est.eps_avg == 7.0

    assert urls
    assert "/analyst-estimates?" in urls[0]
    assert "period=annual" in urls[0]


def test_get_sec_filings_sends_from_to_and_parses() -> None:
    payload = [
        {
            "symbol": "AAPL",
            "cik": "0000320193",
            "filingDate": "2026-02-01",
            "acceptedDate": "2026-02-01 16:30:00",
            "formType": "10-Q",
            "link": "https://www.sec.gov/example/index",
            "finalLink": "https://www.sec.gov/example/filing.htm",
        }
    ]
    opener, urls = _make_recording_opener(payload)
    client = FMPClient(api_key="k", opener=opener)

    out = client.get_sec_filings("AAPL", "2026-01-01", "2026-05-15")
    assert len(out) == 1
    filing = out[0]
    assert isinstance(filing, FMPSecFiling)
    assert filing.cik == "0000320193"
    assert filing.filing_date == "2026-02-01"
    assert filing.form_type == "10-Q"
    assert filing.link.startswith("https://")
    assert filing.final_link.endswith(".htm")

    assert urls
    assert "/sec-filings-search/symbol?" in urls[0]
    assert "from=2026-01-01" in urls[0]
    assert "to=2026-05-15" in urls[0]


# ---------------------------------------------------------------------------
# Fail-open paths
# ---------------------------------------------------------------------------


def test_http_403_returns_empty_list_fail_open() -> None:
    def _opener(url: str) -> bytes:
        raise urllib.error.HTTPError(
            url, 403, "Forbidden", hdrs=None, fp=None  # type: ignore[arg-type]
        )

    client = FMPClient(api_key="k", opener=_opener)
    assert client.get_quote("AAPL") == []
    assert client.get_profile("AAPL") == []
    assert client.get_earnings_calendar("2026-05-01", "2026-05-31") == []
    assert client.get_price_target_consensus("AAPL") == []
    assert client.get_analyst_estimates_annual("AAPL") == []
    assert client.get_sec_filings("AAPL", "2026-01-01", "2026-05-15") == []


def test_malformed_json_returns_empty_list_fail_open() -> None:
    def _opener(url: str) -> bytes:
        return b"{this-is-not-json"

    client = FMPClient(api_key="k", opener=_opener)
    assert client.get_quote("AAPL") == []
    assert client.get_profile("AAPL") == []


def test_missing_api_key_returns_empty_list_fail_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No env key, no env file ⇒ key is None ⇒ all public methods empty.
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    monkeypatch.setattr(fmp_mod, "FMP_ENV_PATH", tmp_path / "missing.env")

    sentinel: Dict[str, int] = {"calls": 0}

    def _opener(url: str) -> bytes:
        sentinel["calls"] += 1
        return b"[]"

    client = FMPClient(opener=_opener)
    assert client.get_quote("AAPL") == []
    assert client.get_profile("AAPL") == []
    assert client.get_earnings_calendar("2026-05-01", "2026-05-31") == []
    assert client.get_price_target_consensus("AAPL") == []
    assert client.get_analyst_estimates_annual("AAPL") == []
    assert client.get_sec_filings("AAPL", "2026-01-01", "2026-05-15") == []
    assert sentinel["calls"] == 0  # opener never invoked when key missing


# ---------------------------------------------------------------------------
# Legacy /api/v3 must not appear in the new module.
# ---------------------------------------------------------------------------


def test_no_legacy_api_v3_in_fmp_client_source() -> None:
    source_path = Path(fmp_mod.__file__)
    text = source_path.read_text(encoding="utf-8")
    assert "/api/v3" not in text, (
        "fmp_client.py must not reference legacy /api/v3 endpoints"
    )

"""Tests for chad.market_data.market_intel_provider.

Phase B FMP Phase 1 — abstraction-only coverage. Every test injects a
fake FMP client through ``reset_fmp_client_for_tests`` so no live HTTP
is issued.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from chad.market_data import market_intel_provider as mip
from chad.market_data.fmp_client import (
    FMPAnalystEstimate,
    FMPEarningsEvent,
    FMPPriceTargetConsensus,
    FMPProfile,
    FMPQuote,
    FMPSecFiling,
)


class _FakeFMPClient:
    """Records each call so the delegation tests can assert on arguments."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def get_quote(self, symbol: str) -> List[FMPQuote]:
        self.calls.append(("get_quote", (symbol,)))
        return [
            FMPQuote(
                symbol=symbol.upper(),
                name="Fake",
                price=1.0,
                change_percentage=0.0,
                volume=0.0,
                average_volume=0.0,
                market_cap=0.0,
                timestamp=0,
            )
        ]

    def get_profile(self, symbol: str) -> List[FMPProfile]:
        self.calls.append(("get_profile", (symbol,)))
        return [
            FMPProfile(
                symbol=symbol.upper(),
                company_name="Fake",
                price=None,
                market_cap=None,
                beta=None,
                average_volume=None,
                sector="",
                industry="",
                exchange="",
                currency="",
            )
        ]

    def get_earnings_calendar(
        self, date_from: str, date_to: str
    ) -> List[FMPEarningsEvent]:
        self.calls.append(("get_earnings_calendar", (date_from, date_to)))
        return [
            FMPEarningsEvent(
                symbol="AAPL",
                date=date_from,
                eps_actual=None,
                eps_estimated=None,
                revenue_actual=None,
                revenue_estimated=None,
                last_updated="",
            )
        ]

    def get_price_target_consensus(
        self, symbol: str
    ) -> List[FMPPriceTargetConsensus]:
        self.calls.append(("get_price_target_consensus", (symbol,)))
        return [
            FMPPriceTargetConsensus(
                symbol=symbol.upper(),
                target_high=None,
                target_low=None,
                target_consensus=None,
                target_median=None,
            )
        ]

    def get_analyst_estimates_annual(
        self, symbol: str
    ) -> List[FMPAnalystEstimate]:
        self.calls.append(("get_analyst_estimates_annual", (symbol,)))
        return [
            FMPAnalystEstimate(
                symbol=symbol.upper(),
                date="2027",
                revenue_low=None,
                revenue_high=None,
                revenue_avg=None,
                eps_low=None,
                eps_high=None,
                eps_avg=None,
            )
        ]

    def get_sec_filings(
        self, symbol: str, date_from: str, date_to: str
    ) -> List[FMPSecFiling]:
        self.calls.append(("get_sec_filings", (symbol, date_from, date_to)))
        return [
            FMPSecFiling(
                symbol=symbol.upper(),
                cik="",
                filing_date=date_from,
                accepted_date="",
                form_type="10-Q",
                link="",
                final_link="",
            )
        ]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeFMPClient:
    monkeypatch.delenv("CHAD_INTEL_PROVIDER", raising=False)
    monkeypatch.delenv("CHAD_NEWS_PROVIDER", raising=False)
    monkeypatch.delenv("CHAD_VOLUME_PROVIDER", raising=False)
    fake = _FakeFMPClient()
    mip.reset_fmp_client_for_tests(fake)  # type: ignore[arg-type]
    yield fake
    mip.reset_fmp_client_for_tests(None)


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


def test_fetch_quote_delegates_to_fmp_client(fake_client: _FakeFMPClient) -> None:
    out = mip.fetch_quote("aapl")
    assert len(out) == 1
    assert out[0].symbol == "AAPL"
    assert fake_client.calls == [("get_quote", ("aapl",))]


def test_fetch_profile_delegates_to_fmp_client(
    fake_client: _FakeFMPClient,
) -> None:
    out = mip.fetch_profile("AAPL")
    assert len(out) == 1
    assert fake_client.calls == [("get_profile", ("AAPL",))]


def test_fetch_earnings_calendar_delegates(
    fake_client: _FakeFMPClient,
) -> None:
    out = mip.fetch_earnings_calendar("2026-05-01", "2026-05-31")
    assert len(out) == 1
    assert fake_client.calls == [
        ("get_earnings_calendar", ("2026-05-01", "2026-05-31"))
    ]


def test_fetch_price_target_consensus_delegates(
    fake_client: _FakeFMPClient,
) -> None:
    out = mip.fetch_price_target_consensus("AAPL")
    assert len(out) == 1
    assert fake_client.calls == [("get_price_target_consensus", ("AAPL",))]


def test_fetch_analyst_estimates_annual_delegates(
    fake_client: _FakeFMPClient,
) -> None:
    out = mip.fetch_analyst_estimates_annual("AAPL")
    assert len(out) == 1
    assert fake_client.calls == [("get_analyst_estimates_annual", ("AAPL",))]


def test_fetch_sec_filings_delegates(fake_client: _FakeFMPClient) -> None:
    out = mip.fetch_sec_filings("AAPL", "2026-01-01", "2026-05-15")
    assert len(out) == 1
    assert fake_client.calls == [
        ("get_sec_filings", ("AAPL", "2026-01-01", "2026-05-15"))
    ]


# ---------------------------------------------------------------------------
# Deferred surfaces
# ---------------------------------------------------------------------------


def test_fetch_news_returns_empty_and_does_not_call_fmp(
    fake_client: _FakeFMPClient,
) -> None:
    out = mip.fetch_news("AAPL", limit=5)
    assert out == []
    assert fake_client.calls == []


def test_fetch_volume_snapshot_returns_empty_and_skips_providers(
    fake_client: _FakeFMPClient,
) -> None:
    out = mip.fetch_volume_snapshot(["AAPL", "MSFT"])
    assert out == {}
    assert fake_client.calls == []


# ---------------------------------------------------------------------------
# Provider defaults & import safety
# ---------------------------------------------------------------------------


def test_default_intel_provider_is_fmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHAD_INTEL_PROVIDER", raising=False)
    assert mip.DEFAULT_INTEL_PROVIDER == "fmp"
    assert mip._intel_provider() == "fmp"


def test_no_import_time_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the module must not touch the network or build a client."""
    import importlib
    import urllib.request as _req

    def _explode(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("import-time network call detected")

    monkeypatch.setattr(_req, "urlopen", _explode)

    # Force a fresh import path.
    import sys

    sys.modules.pop("chad.market_data.market_intel_provider", None)
    reloaded = importlib.import_module(
        "chad.market_data.market_intel_provider"
    )
    # The singleton should not have been constructed yet.
    assert reloaded._client_singleton is None
    # Restore the cached singleton state for any tests that follow.
    reloaded.reset_fmp_client_for_tests(None)

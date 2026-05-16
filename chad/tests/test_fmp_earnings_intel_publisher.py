"""Tests for the FMP earnings / analyst intelligence publisher.

Hermetic: no live FMP call is made. The market_intel_provider fetchers
are monkeypatched to deterministic fake payloads on every test.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.market_data import fmp_earnings_intel_publisher as pub
from chad.market_data.fmp_client import (
    FMPAnalystEstimate,
    FMPEarningsEvent,
    FMPPriceTargetConsensus,
    FMPSecFiling,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PUB_SOURCE_PATH = REPO_ROOT / "chad" / "market_data" / "fmp_earnings_intel_publisher.py"
DEPLOY_DIR = REPO_ROOT / "deploy"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


REFERENCE_TODAY = date(2026, 5, 16)


def _today_monkeypatch(monkeypatch: pytest.MonkeyPatch, when: date = REFERENCE_TODAY) -> None:
    monkeypatch.setattr(pub, "_today_utc", lambda: when)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    earnings: List[FMPEarningsEvent] | None = None,
    price_targets: Dict[str, List[FMPPriceTargetConsensus]] | None = None,
    estimates: Dict[str, List[FMPAnalystEstimate]] | None = None,
    filings: Dict[str, List[FMPSecFiling]] | None = None,
    raise_on: set[str] | None = None,
) -> None:
    raise_on = raise_on or set()

    def _earnings(date_from: str, date_to: str) -> List[FMPEarningsEvent]:
        if "earnings" in raise_on:
            raise RuntimeError("earnings boom")
        return list(earnings or [])

    def _pt(symbol: str) -> List[FMPPriceTargetConsensus]:
        if "pt" in raise_on:
            raise RuntimeError("pt boom")
        return list((price_targets or {}).get(symbol, []))

    def _est(symbol: str) -> List[FMPAnalystEstimate]:
        if "est" in raise_on:
            raise RuntimeError("est boom")
        return list((estimates or {}).get(symbol, []))

    def _sec(symbol: str, date_from: str, date_to: str) -> List[FMPSecFiling]:
        if "sec" in raise_on:
            raise RuntimeError("sec boom")
        return list((filings or {}).get(symbol, []))

    monkeypatch.setattr(
        pub.market_intel_provider, "fetch_earnings_calendar", _earnings,
    )
    monkeypatch.setattr(
        pub.market_intel_provider, "fetch_price_target_consensus", _pt,
    )
    monkeypatch.setattr(
        pub.market_intel_provider, "fetch_analyst_estimates_annual", _est,
    )
    monkeypatch.setattr(
        pub.market_intel_provider, "fetch_sec_filings", _sec,
    )


def _earnings_event(
    sym: str, d: date,
    *, eps_est: float | None = None, eps_act: float | None = None,
    rev_est: float | None = None, rev_act: float | None = None,
) -> FMPEarningsEvent:
    return FMPEarningsEvent(
        symbol=sym,
        date=d.strftime("%Y-%m-%d"),
        eps_actual=eps_act,
        eps_estimated=eps_est,
        revenue_actual=rev_act,
        revenue_estimated=rev_est,
        last_updated="",
    )


# ---------------------------------------------------------------------------
# 1. Symbol filter removes crypto / futures
# ---------------------------------------------------------------------------


def test_filter_universe_removes_crypto_and_futures() -> None:
    out = pub.filter_universe(
        ["AAPL", "btc-usd", "MES", "msft", "MNQ", "QQQ", "ETH-USD", "ZN", "  "]
    )
    assert out == ["AAPL", "MSFT", "QQQ"]


# ---------------------------------------------------------------------------
# 2. Payload schema_version is earnings_intel.v1
# ---------------------------------------------------------------------------


def test_build_payload_has_schema_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    _install_fakes(monkeypatch)
    payload = pub.build_payload(
        ["AAPL"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    assert payload["schema_version"] == "earnings_intel.v1"
    assert payload["source"]["provider"] == "fmp_stable"
    assert payload["window"]["lookback_days"] == 14
    assert payload["window"]["forward_days"] == 45
    assert payload["window"]["date_from"] == "2026-05-02"
    assert payload["window"]["date_to"] == "2026-06-30"
    assert payload["ttl_seconds"] == 21600


# ---------------------------------------------------------------------------
# 3. Next earnings date picks the soonest future event
# ---------------------------------------------------------------------------


def test_next_earnings_picks_future_event(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    events = [
        _earnings_event("AAPL", REFERENCE_TODAY + timedelta(days=10), eps_est=1.1),
        _earnings_event("AAPL", REFERENCE_TODAY + timedelta(days=30), eps_est=1.2),
        _earnings_event("AAPL", REFERENCE_TODAY - timedelta(days=80), eps_act=0.9),
    ]
    _install_fakes(monkeypatch, earnings=events)
    payload = pub.build_payload(
        ["AAPL"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["AAPL"]
    assert rec["next_earnings_date"] == (REFERENCE_TODAY + timedelta(days=10)).strftime("%Y-%m-%d")
    assert rec["days_to_next_earnings"] == 10
    assert rec["eps_estimated"] == 1.1


# ---------------------------------------------------------------------------
# 4. Last earnings date picks the most-recent past event
# ---------------------------------------------------------------------------


def test_last_earnings_picks_past_event(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    events = [
        _earnings_event("AAPL", REFERENCE_TODAY - timedelta(days=10), eps_act=2.0),
        _earnings_event("AAPL", REFERENCE_TODAY - timedelta(days=120), eps_act=1.4),
    ]
    _install_fakes(monkeypatch, earnings=events)
    payload = pub.build_payload(
        ["AAPL"], lookback_days=180, forward_days=45, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["AAPL"]
    assert rec["last_earnings_date"] == (REFERENCE_TODAY - timedelta(days=10)).strftime("%Y-%m-%d")
    assert rec["eps_actual"] == 2.0


# ---------------------------------------------------------------------------
# 5. Price target consensus populates correctly
# ---------------------------------------------------------------------------


def test_price_target_consensus_populates(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    pts = {
        "NVDA": [FMPPriceTargetConsensus(
            symbol="NVDA",
            target_high=1500.0, target_low=900.0,
            target_consensus=1200.0, target_median=1180.0,
        )]
    }
    _install_fakes(monkeypatch, price_targets=pts)
    payload = pub.build_payload(
        ["NVDA"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["NVDA"]
    assert rec["price_target_high"] == 1500.0
    assert rec["price_target_low"] == 900.0
    assert rec["price_target_consensus"] == 1200.0
    assert rec["price_target_median"] == 1180.0
    assert "price-target-consensus" not in rec["provider_errors"]


# ---------------------------------------------------------------------------
# 6. Analyst annual estimate populates
# ---------------------------------------------------------------------------


def test_analyst_estimate_populates(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    estimates = {
        "MSFT": [
            FMPAnalystEstimate(
                symbol="MSFT", date="2025-12-31",
                revenue_low=200.0, revenue_high=260.0, revenue_avg=230.0,
                eps_low=10.0, eps_high=12.0, eps_avg=11.0,
            ),
            FMPAnalystEstimate(
                symbol="MSFT", date="2027-06-30",
                revenue_low=240.0, revenue_high=300.0, revenue_avg=270.0,
                eps_low=11.0, eps_high=14.0, eps_avg=12.5,
            ),
        ]
    }
    _install_fakes(monkeypatch, estimates=estimates)
    payload = pub.build_payload(
        ["MSFT"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["MSFT"]
    # Today is 2026-05-16; nearest future annual estimate is 2027-06-30.
    assert rec["annual_revenue_avg_estimate"] == 270.0
    assert rec["annual_eps_avg_estimate"] == 12.5


# ---------------------------------------------------------------------------
# 7. SEC filing latest selected by accepted_date
# ---------------------------------------------------------------------------


def test_latest_sec_filing_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    filings = {
        "AAPL": [
            FMPSecFiling(
                symbol="AAPL", cik="0000320193",
                filing_date="2026-04-01",
                accepted_date="2026-04-01T20:30:00",
                form_type="10-Q", link="", final_link="",
            ),
            FMPSecFiling(
                symbol="AAPL", cik="0000320193",
                filing_date="2026-05-10",
                accepted_date="2026-05-10T16:00:00",
                form_type="8-K", link="", final_link="",
            ),
        ]
    }
    _install_fakes(monkeypatch, filings=filings)
    payload = pub.build_payload(
        ["AAPL"], lookback_days=60, forward_days=10, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["AAPL"]
    assert rec["sec_filings_count"] == 2
    assert rec["latest_filing_type"] == "8-K"
    assert rec["latest_filing_date"] == "2026-05-10T16:00:00"


# ---------------------------------------------------------------------------
# 8. Partial data still flags data_available=True
# ---------------------------------------------------------------------------


def test_partial_data_sets_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    _install_fakes(
        monkeypatch,
        price_targets={
            "GOOGL": [FMPPriceTargetConsensus(
                symbol="GOOGL", target_high=200.0, target_low=140.0,
                target_consensus=170.0, target_median=168.0,
            )]
        },
    )
    payload = pub.build_payload(
        ["GOOGL"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["GOOGL"]
    assert rec["data_available"] is True
    # Other endpoints returned empty — recorded as provider_errors but not fatal.
    assert "analyst-estimates" in rec["provider_errors"]
    assert "sec-filings" in rec["provider_errors"]


# ---------------------------------------------------------------------------
# 9. Empty provider data yields empty/partial status, no crash
# ---------------------------------------------------------------------------


def test_empty_provider_returns_empty_status(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    _install_fakes(monkeypatch)
    payload = pub.build_payload(
        ["AAPL", "MSFT"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    assert payload["status"] in {"empty", "partial"}
    for sym in ("AAPL", "MSFT"):
        rec = payload["symbols"][sym]
        assert rec["data_available"] is False
        assert rec["sec_filings_count"] == 0


# ---------------------------------------------------------------------------
# 10. Dry-run does not write a file
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _today_monkeypatch(monkeypatch)
    _install_fakes(monkeypatch)
    pub.publish(
        runtime_dir=tmp_path,
        symbols=["AAPL"],
        lookback_days=14, forward_days=45,
        dry_run=True,
    )
    assert not (tmp_path / "earnings_intel.json").exists()


# ---------------------------------------------------------------------------
# 11. publish() writes earnings_intel.json when dry_run=False
# ---------------------------------------------------------------------------


def test_publish_writes_runtime_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _today_monkeypatch(monkeypatch)
    _install_fakes(monkeypatch)
    pub.publish(
        runtime_dir=tmp_path,
        symbols=["AAPL", "MSFT"],
        lookback_days=14, forward_days=45,
        dry_run=False,
    )
    out = tmp_path / "earnings_intel.json"
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "earnings_intel.v1"
    assert set(payload["symbols"].keys()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# 12. Provider exceptions are recorded per symbol
# ---------------------------------------------------------------------------


def test_provider_errors_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _today_monkeypatch(monkeypatch)
    _install_fakes(monkeypatch, raise_on={"pt", "sec"})
    payload = pub.build_payload(
        ["AAPL"], lookback_days=14, forward_days=45, today=REFERENCE_TODAY,
    )
    rec = payload["symbols"]["AAPL"]
    assert "price-target-consensus" in rec["provider_errors"]
    assert "sec-filings" in rec["provider_errors"]


# ---------------------------------------------------------------------------
# 13. Deploy unit files exist with expected directives
# ---------------------------------------------------------------------------


def test_deploy_service_and_timer_contents() -> None:
    svc_path = DEPLOY_DIR / "chad-fmp-earnings-intel-refresh.service"
    timer_path = DEPLOY_DIR / "chad-fmp-earnings-intel-refresh.timer"
    assert svc_path.exists()
    assert timer_path.exists()

    svc = svc_path.read_text(encoding="utf-8")
    assert "EnvironmentFile=-/etc/chad/fmp.env" in svc
    assert "chad.market_data.fmp_earnings_intel_publisher" in svc

    timer = timer_path.read_text(encoding="utf-8")
    assert "OnUnitActiveSec=21600" in timer
    assert "Persistent=true" in timer


# ---------------------------------------------------------------------------
# 14. Publisher does not import strategies / execution modules
# ---------------------------------------------------------------------------


def test_publisher_no_forbidden_imports() -> None:
    src = PUB_SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "chad.strategies",
        "chad.execution",
        "chad.core.orchestrator",
        "chad.core.live_loop",
        "chad.core.live_gate",
        "ib_async",
        "ib_insync",
    )
    for needle in forbidden:
        assert needle not in src, f"forbidden import present: {needle}"


# ---------------------------------------------------------------------------
# 15. Publisher does not reference the FMP news endpoint
# ---------------------------------------------------------------------------


def test_publisher_no_news_endpoint() -> None:
    src = PUB_SOURCE_PATH.read_text(encoding="utf-8")
    for needle in ("news/stock", "stock_news", "stable/news", "fetch_news"):
        assert needle not in src, f"news endpoint reference present: {needle}"

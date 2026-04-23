"""Tests for chad.market_data.sec_13f_fetcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

import pytest

from chad.market_data.sec_13f_fetcher import (
    Holding,
    SEC13FFetcher,
    TARGET_FUNDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_INFOTABLE_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>123456789</value>
    <shrsOrPrnAmt>
      <sshPrnamt>500000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>98765432</value>
    <shrsOrPrnAmt>
      <sshPrnamt>300000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
</informationTable>
"""


def _fake_submissions(accession: str = "0001234567-24-000100") -> Dict[str, Any]:
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "13F-HR", "8-K"],
                "accessionNumber": ["0000000000-00-000000", accession, "0000000000-00-999999"],
                "primaryDocument": ["cover.htm", "primary_doc.xml", "other.htm"],
                "reportDate": ["2026-03-31", "2026-03-31", "2026-04-01"],
            }
        }
    }


def _fake_archive_index() -> Dict[str, Any]:
    return {
        "directory": {
            "item": [
                {"name": "primary_doc.xml"},
                {"name": "form13fInfoTable.xml"},
                {"name": "index.json"},
            ]
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_target_funds_are_padded_ciks() -> None:
    for name, cik in TARGET_FUNDS.items():
        assert isinstance(cik, str)
        assert len(cik) == 10
        assert cik.isdigit(), f"non-digit CIK for {name}: {cik}"


def test_parse_info_table_extracts_symbol_shares_value(tmp_path: Path) -> None:
    f = SEC13FFetcher(cache_dir=tmp_path)
    holdings = f._parse_info_table(SAMPLE_INFOTABLE_XML)
    assert len(holdings) == 2
    by_cusip = {h.cusip: h for h in holdings}
    aapl = by_cusip["037833100"]
    assert aapl.name_of_issuer == "APPLE INC"
    assert aapl.shares == 500_000
    assert aapl.value_usd == 123_456_789.0
    msft = by_cusip["594918104"]
    assert msft.shares == 300_000


def test_parse_info_table_skips_malformed(tmp_path: Path) -> None:
    bad_xml = b"not <xml at all"
    f = SEC13FFetcher(cache_dir=tmp_path)
    assert f._parse_info_table(bad_xml) == []


def test_parse_info_table_skips_zero_or_negative(tmp_path: Path) -> None:
    xml = b"""<?xml version='1.0'?>
    <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable>
        <nameOfIssuer>ZERO CORP</nameOfIssuer>
        <cusip>000000000</cusip>
        <value>0</value>
        <shrsOrPrnAmt><sshPrnamt>0</sshPrnamt></shrsOrPrnAmt>
      </infoTable>
    </informationTable>"""
    f = SEC13FFetcher(cache_dir=tmp_path)
    assert f._parse_info_table(xml) == []


def test_fetch_latest_13f_happy_path(tmp_path: Path) -> None:
    f = SEC13FFetcher(cache_dir=tmp_path)

    responses = [
        json.dumps(_fake_submissions("0001234567-24-000100")).encode("utf-8"),
        json.dumps(_fake_archive_index()).encode("utf-8"),
        SAMPLE_INFOTABLE_XML,
    ]

    with mock.patch.object(f, "_http_get", side_effect=responses):
        result = f.fetch_latest_13f("0001067983")

    assert result["cik"] == "0001067983"
    assert result["accession"] == "0001234567-24-000100"
    assert result["report_date"] == "2026-03-31"
    assert result["source"] == "sec_edgar"
    assert len(result["holdings"]) == 2

    # And a second fetch should hit the cache (only submissions call)
    responses2 = [json.dumps(_fake_submissions("0001234567-24-000100")).encode("utf-8")]
    with mock.patch.object(f, "_http_get", side_effect=responses2) as patched:
        result2 = f.fetch_latest_13f("0001067983")
    assert result2["source"] == "cache"
    assert len(result2["holdings"]) == 2
    # Only the submissions URL was called — info table came from cache
    assert patched.call_count == 1


def test_fetch_latest_13f_handles_network_error_gracefully(tmp_path: Path) -> None:
    f = SEC13FFetcher(cache_dir=tmp_path)
    with mock.patch.object(f, "_http_get", return_value=None):
        result = f.fetch_latest_13f("0001067983")
    assert result["holdings"] == []
    assert result["accession"] is None


def test_fetch_latest_13f_handles_no_13f_in_submissions(tmp_path: Path) -> None:
    no_13f = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K"],
                "accessionNumber": ["a", "b"],
                "primaryDocument": ["x.htm", "y.htm"],
                "reportDate": ["2026-01-01", "2026-02-01"],
            }
        }
    }
    f = SEC13FFetcher(cache_dir=tmp_path)
    with mock.patch.object(
        f, "_http_get", return_value=json.dumps(no_13f).encode("utf-8")
    ):
        result = f.fetch_latest_13f("0001067983")
    assert result["holdings"] == []
    assert result["accession"] is None


def test_caches_results(tmp_path: Path) -> None:
    f = SEC13FFetcher(cache_dir=tmp_path)
    responses = [
        json.dumps(_fake_submissions("0009999999-24-000001")).encode("utf-8"),
        json.dumps(_fake_archive_index()).encode("utf-8"),
        SAMPLE_INFOTABLE_XML,
    ]
    with mock.patch.object(f, "_http_get", side_effect=responses):
        f.fetch_latest_13f("0001067983")

    # Cache file on disk, readable via the public API path too
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text())
    assert payload["accession"] == "0009999999-24-000001"
    assert len(payload["holdings"]) == 2


def test_get_all_fund_holdings_partial_failure_still_returns_results(tmp_path: Path) -> None:
    f = SEC13FFetcher(cache_dir=tmp_path)

    def fake_fetch(cik_padded: str) -> Dict[str, Any]:
        if cik_padded == "0001067983":
            return {
                "cik": cik_padded,
                "accession": "x",
                "report_date": "2026-03-31",
                "holdings": [Holding("APPLE INC", "037833100", 1_000_000.0, 100)],
                "source": "cache",
            }
        return {"cik": cik_padded, "accession": None, "report_date": None, "holdings": []}

    with mock.patch.object(f, "fetch_latest_13f", side_effect=fake_fetch):
        result = f.get_all_fund_holdings(
            funds={"berkshire": "0001067983", "other": "0000000001"}
        )

    assert set(result.keys()) == {"berkshire", "other"}
    assert len(result["berkshire"]["holdings"]) == 1
    assert result["other"]["holdings"] == []


def test_holding_to_dict_round_trip() -> None:
    h = Holding("APPLE INC", "037833100", 12345.67, 100)
    d = h.to_dict()
    assert d == {
        "name_of_issuer": "APPLE INC",
        "cusip": "037833100",
        "value_usd": 12345.67,
        "shares": 100,
    }

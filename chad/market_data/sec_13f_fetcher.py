"""
chad/market_data/sec_13f_fetcher.py

SEC EDGAR 13F filing fetcher.

13F-HR filings are quarterly institutional holdings reports (filing deadlines
Feb 14, May 15, Aug 14, Nov 14). Data is 45 days delayed — this is a feature
for a long-term strategy, not a bug.

- Uses SEC EDGAR free public API. No API key required.
- Required: User-Agent header with real contact info.
- Rate limit: 10 req/sec — enforced locally via time.sleep.
- Fails gracefully: network / parse errors return empty lists, never raise.

Outputs a list of Holding records per fund. A sibling module
(institutional_consensus) aggregates across funds.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "runtime" / "sec_13f_cache"

# SEC requires a descriptive User-Agent with contact info. See
# https://www.sec.gov/os/accessing-edgar-data
USER_AGENT = "CHAD-Trading-System contact@chad.trading"

# 10 req/sec limit -> ~0.11s between requests is safe
RATE_LIMIT_DELAY = 0.12

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

# Target institutional filers keyed by 10-digit zero-padded CIK
TARGET_FUNDS: Dict[str, str] = {
    "berkshire_hathaway": "0001067983",
    "bridgewater":        "0001350694",
    "renaissance":        "0001037389",
    "citadel":            "0001423053",
    "blackrock":          "0001364742",
    "vanguard":           "0000102909",
    "appaloosa":          "0001006438",
    "pershing_square":    "0001336528",
}

HTTP_TIMEOUT_SEC = 15


@dataclass(frozen=True)
class Holding:
    """A single position reported in a 13F-HR information table."""
    name_of_issuer: str   # e.g. "APPLE INC"
    cusip: str            # 9-char CUSIP
    value_usd: float      # in raw dollars (13F reports in $ thousands until Q1 2023, dollars after)
    shares: int           # sshPrnamt (typically shares, see shrsOrPrnAmt.sshPrnamtType)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SEC13FFetcher:
    """
    Fetches and parses 13F-HR filings from SEC EDGAR.

    Fail-soft semantics: any HTTP, parse, or filesystem failure logs a
    warning and returns an empty list. This module is consumed by Beta,
    which further fails-closed (returns [] signals) when the consensus
    file is stale or missing — so data unavailability cannot crash CHAD.
    """

    def __init__(self, cache_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
        self.log = logger or LOG
        self._last_request_at: float = 0.0

    # ---- HTTP ---------------------------------------------------------

    def _throttle(self) -> None:
        now = time.monotonic()
        gap = now - self._last_request_at
        if gap < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - gap)
        self._last_request_at = time.monotonic()

    def _http_get(self, url: str) -> Optional[bytes]:
        """GET url with SEC-compliant headers. Returns bytes or None."""
        self._throttle()
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/xml, */*",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            self.log.warning("sec_13f_fetcher: GET failed url=%s err=%s", url, exc)
            return None

    # ---- Submissions list --------------------------------------------

    def _fetch_submissions(self, cik_padded: str) -> Optional[Dict[str, Any]]:
        """Fetch the submissions.json index for a CIK."""
        body = self._http_get(SUBMISSIONS_URL.format(cik_padded=cik_padded))
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self.log.warning("sec_13f_fetcher: submissions JSON parse failed cik=%s err=%s", cik_padded, exc)
            return None

    def _latest_13f_accession(self, submissions: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Return {accession_no, primary_document, report_date} for the most recent 13F-HR."""
        recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        accns = recent.get("accessionNumber") or []
        docs = recent.get("primaryDocument") or []
        dates = recent.get("reportDate") or recent.get("filingDate") or []
        n = min(len(forms), len(accns), len(docs), len(dates))
        for i in range(n):
            if str(forms[i]).strip().upper() == "13F-HR":
                return {
                    "accession": str(accns[i]),
                    "primary_document": str(docs[i]),
                    "report_date": str(dates[i]),
                }
        return None

    # ---- Information table -------------------------------------------

    def _info_table_url(self, cik_padded: str, accession: str) -> Optional[str]:
        """
        13F info tables are XML files in the filing's archive folder. The
        primary document is the 13F form itself (also XML). We locate the
        INFORMATION TABLE by fetching the filing index and picking the file
        whose name ends with 'informationtable.xml' or contains 'infotable'.
        """
        accession_nodash = accession.replace("-", "")
        cik_nopad = str(int(cik_padded))
        idx_url = f"{ARCHIVES_BASE}/{cik_nopad}/{accession_nodash}/"
        # SEC exposes an index.json for every archive folder
        body = self._http_get(idx_url + "index.json")
        if not body:
            return None
        try:
            idx = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

        items = ((idx or {}).get("directory") or {}).get("item") or []
        # Prefer files explicitly named as the info table
        candidates: List[str] = []
        for item in items:
            name = str(item.get("name") or "").lower()
            if name.endswith(".xml") and ("informationtable" in name or "infotable" in name):
                candidates.append(str(item["name"]))
        if not candidates:
            # Fallback: take any XML that isn't the primary_document (the form cover)
            for item in items:
                name = str(item.get("name") or "")
                if name.lower().endswith(".xml"):
                    candidates.append(name)
        if not candidates:
            return None
        return idx_url + candidates[0]

    def _parse_info_table(self, xml_bytes: bytes) -> List[Holding]:
        """Parse 13F information table XML into Holding records."""
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            self.log.warning("sec_13f_fetcher: XML parse failed err=%s", exc)
            return []

        holdings: List[Holding] = []
        # 13F XML uses a namespace; match localname regardless
        for info in root.iter():
            tag = info.tag.split("}", 1)[-1] if "}" in info.tag else info.tag
            if tag != "infoTable":
                continue

            def _child_text(parent: ET.Element, local: str) -> Optional[str]:
                for child in parent.iter():
                    ct = child.tag.split("}", 1)[-1] if "}" in child.tag else child.tag
                    if ct == local:
                        return (child.text or "").strip() or None
                return None

            name = _child_text(info, "nameOfIssuer") or ""
            cusip = (_child_text(info, "cusip") or "").strip().upper()
            value_raw = _child_text(info, "value") or "0"
            shares_raw = _child_text(info, "sshPrnamt") or "0"
            try:
                value = float(value_raw)
                shares = int(float(shares_raw))
            except (ValueError, TypeError):
                continue
            if not cusip or value <= 0 or shares <= 0:
                continue
            holdings.append(Holding(
                name_of_issuer=name,
                cusip=cusip,
                value_usd=value,
                shares=shares,
            ))
        return holdings

    # ---- Cache --------------------------------------------------------

    def _cache_path(self, cik_padded: str, accession: str) -> Path:
        return self.cache_dir / f"{cik_padded}_{accession.replace('-', '')}.json"

    def _read_cache(self, cik_padded: str, accession: str) -> Optional[List[Holding]]:
        path = self._cache_path(cik_padded, accession)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            holdings = payload.get("holdings") or []
            return [
                Holding(
                    name_of_issuer=str(h.get("name_of_issuer") or ""),
                    cusip=str(h.get("cusip") or ""),
                    value_usd=float(h.get("value_usd") or 0.0),
                    shares=int(h.get("shares") or 0),
                )
                for h in holdings
            ]
        except (OSError, ValueError, TypeError) as exc:
            self.log.warning("sec_13f_fetcher: cache read failed path=%s err=%s", path, exc)
            return None

    def _write_cache(
        self,
        cik_padded: str,
        accession: str,
        report_date: str,
        holdings: List[Holding],
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(cik_padded, accession)
        payload = {
            "cik": cik_padded,
            "accession": accession,
            "report_date": report_date,
            "holdings": [h.to_dict() for h in holdings],
        }
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            self.log.warning("sec_13f_fetcher: cache write failed path=%s err=%s", path, exc)

    # ---- Public API ---------------------------------------------------

    def fetch_latest_13f(self, cik_padded: str) -> Dict[str, Any]:
        """
        Fetch the most recent 13F-HR filing for a fund.

        Returns a dict with {cik, accession, report_date, holdings:[Holding,...]}.
        On any failure, returns a dict with an empty holdings list so callers
        can aggregate across partial fetches.
        """
        result: Dict[str, Any] = {
            "cik": cik_padded,
            "accession": None,
            "report_date": None,
            "holdings": [],
            "source": "sec_edgar",
        }

        submissions = self._fetch_submissions(cik_padded)
        if not submissions:
            return result

        latest = self._latest_13f_accession(submissions)
        if not latest:
            self.log.warning("sec_13f_fetcher: no 13F-HR filings found cik=%s", cik_padded)
            return result

        accession = latest["accession"]
        result["accession"] = accession
        result["report_date"] = latest["report_date"]

        # Cache check
        cached = self._read_cache(cik_padded, accession)
        if cached is not None:
            result["holdings"] = cached
            result["source"] = "cache"
            return result

        table_url = self._info_table_url(cik_padded, accession)
        if not table_url:
            self.log.warning(
                "sec_13f_fetcher: no info table found cik=%s accession=%s",
                cik_padded, accession,
            )
            return result

        body = self._http_get(table_url)
        if not body:
            return result

        holdings = self._parse_info_table(body)
        result["holdings"] = holdings
        if holdings:
            self._write_cache(cik_padded, accession, latest["report_date"], holdings)
        return result

    def get_all_fund_holdings(
        self,
        funds: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch holdings for every fund in TARGET_FUNDS (or override via `funds`).

        Returns {fund_name: {cik, accession, report_date, holdings, source}}.
        Funds that fail to resolve still appear in the result with an empty
        holdings list, so the consensus aggregator can still process partial
        data.
        """
        funds = funds if funds is not None else TARGET_FUNDS
        results: Dict[str, Dict[str, Any]] = {}
        for name, cik in funds.items():
            res = self.fetch_latest_13f(cik)
            res["fund_name"] = name
            results[name] = res
            self.log.info(
                "sec_13f_fetcher: fund=%s cik=%s holdings=%d source=%s",
                name, cik, len(res.get("holdings") or []), res.get("source"),
            )
        return results

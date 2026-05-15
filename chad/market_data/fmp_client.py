"""Financial Modeling Prep (FMP) stable-API client.

Phase B FMP Phase 1 — provider scaffolding only. No publisher is migrated
in this phase. The client wraps the stable endpoints confirmed working on
the current plan:

- /stable/quote
- /stable/profile
- /stable/earnings-calendar
- /stable/price-target-consensus
- /stable/analyst-estimates  (period=annual)
- /stable/sec-filings-search/symbol  (from/to)

Legacy v3 endpoints are blocked for this account and intentionally
not referenced anywhere in this module. /stable/news/stock is restricted
and is exposed via the higher-level provider as a no-op stub.

Design contract:
- Strict typing, stdlib only.
- Public methods never raise. Missing key, HTTP errors, network errors,
  decode errors, and unexpected payload shapes all return an empty list.
- The HTTP transport is injectable via ``opener`` for tests so no live
  call is ever made by the unit suite.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_DEFAULT_TIMEOUT_SECONDS = 8.0
FMP_USER_AGENT = "CHAD/1.0"
FMP_ENV_PATH = Path("/etc/chad/fmp.env")
FMP_PLACEHOLDER_KEY = "YOUR_REAL_FMP_KEY_HERE"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FMPQuote:
    symbol: str
    name: str
    price: Optional[float]
    change_percentage: Optional[float]
    volume: Optional[float]
    average_volume: Optional[float]
    market_cap: Optional[float]
    timestamp: Optional[int]


@dataclass(frozen=True)
class FMPProfile:
    symbol: str
    company_name: str
    price: Optional[float]
    market_cap: Optional[float]
    beta: Optional[float]
    average_volume: Optional[float]
    sector: str
    industry: str
    exchange: str
    currency: str


@dataclass(frozen=True)
class FMPEarningsEvent:
    symbol: str
    date: str
    eps_actual: Optional[float]
    eps_estimated: Optional[float]
    revenue_actual: Optional[float]
    revenue_estimated: Optional[float]
    last_updated: str


@dataclass(frozen=True)
class FMPPriceTargetConsensus:
    symbol: str
    target_high: Optional[float]
    target_low: Optional[float]
    target_consensus: Optional[float]
    target_median: Optional[float]


@dataclass(frozen=True)
class FMPAnalystEstimate:
    symbol: str
    date: str
    revenue_low: Optional[float]
    revenue_high: Optional[float]
    revenue_avg: Optional[float]
    eps_low: Optional[float]
    eps_high: Optional[float]
    eps_avg: Optional[float]


@dataclass(frozen=True)
class FMPSecFiling:
    symbol: str
    cik: str
    filing_date: str
    accepted_date: str
    form_type: str
    link: str
    final_link: str


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _as_int(value: Any) -> Optional[int]:
    f = _as_float(value)
    if f is None:
        return None
    try:
        return int(f)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------


def _read_fmp_key() -> Optional[str]:
    """Resolve the FMP API key.

    Order: env var ``FMP_API_KEY`` first, then ``/etc/chad/fmp.env``.
    A placeholder value (``YOUR_REAL_FMP_KEY_HERE``) is treated as
    missing so misconfigured environments fail closed.
    """
    env_key = os.environ.get("FMP_API_KEY", "").strip().strip('"').strip("'")
    if env_key and env_key != FMP_PLACEHOLDER_KEY:
        return env_key

    if not FMP_ENV_PATH.is_file():
        return None
    try:
        text = FMP_ENV_PATH.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() != "FMP_API_KEY":
            continue
        v = v.strip().strip('"').strip("'")
        if not v or v == FMP_PLACEHOLDER_KEY:
            return None
        return v
    return None


# ---------------------------------------------------------------------------
# FMPClient
# ---------------------------------------------------------------------------


class FMPClient:
    """Thin wrapper around the FMP /stable HTTP API.

    All public methods are fail-open: any exception, restricted endpoint,
    or malformed payload yields an empty list. The transport is injected
    through ``opener`` so unit tests never touch the network.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = FMP_BASE_URL,
        timeout: float = FMP_DEFAULT_TIMEOUT_SECONDS,
        opener: Optional[Callable[[str], Any]] = None,
    ) -> None:
        resolved_key: Optional[str]
        if api_key is None:
            resolved_key = _read_fmp_key()
        else:
            stripped = api_key.strip()
            resolved_key = stripped if stripped and stripped != FMP_PLACEHOLDER_KEY else None
        self._api_key: Optional[str] = resolved_key
        self._base_url: str = base_url.rstrip("/")
        self._timeout: float = float(timeout)
        self._opener: Optional[Callable[[str], Any]] = opener

    # -- internal HTTP / decode -----------------------------------------

    def _request_json(self, path: str, params: Dict[str, str]) -> Any:
        """Issue a GET and decode JSON. Returns ``None`` on any failure."""
        if not self._api_key:
            return None
        merged: Dict[str, str] = {k: v for k, v in params.items() if v is not None}
        merged["apikey"] = self._api_key
        query = urllib.parse.urlencode(merged)
        url = f"{self._base_url}/{path.lstrip('/')}?{query}"
        try:
            if self._opener is not None:
                raw = self._opener(url)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": FMP_USER_AGENT})
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
            return None
        except Exception:
            return None
        if raw is None:
            return None
        try:
            if isinstance(raw, (bytes, bytearray)):
                return json.loads(raw.decode("utf-8", errors="replace"))
            if isinstance(raw, str):
                return json.loads(raw)
            # Already-decoded payload (test seam).
            return raw
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _ensure_list(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        return []

    # -- public endpoints -----------------------------------------------

    def get_quote(self, symbol: str) -> List[FMPQuote]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return []
        payload = self._request_json("quote", {"symbol": sym})
        out: List[FMPQuote] = []
        for item in self._ensure_list(payload):
            try:
                out.append(
                    FMPQuote(
                        symbol=_as_str(item.get("symbol")).upper() or sym,
                        name=_as_str(item.get("name")),
                        price=_as_float(item.get("price")),
                        change_percentage=_as_float(
                            item.get("changePercentage")
                            if "changePercentage" in item
                            else item.get("changesPercentage")
                        ),
                        volume=_as_float(item.get("volume")),
                        average_volume=_as_float(
                            item.get("averageVolume")
                            if "averageVolume" in item
                            else item.get("avgVolume")
                        ),
                        market_cap=_as_float(item.get("marketCap")),
                        timestamp=_as_int(item.get("timestamp")),
                    )
                )
            except Exception:
                continue
        return out

    def get_profile(self, symbol: str) -> List[FMPProfile]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return []
        payload = self._request_json("profile", {"symbol": sym})
        out: List[FMPProfile] = []
        for item in self._ensure_list(payload):
            try:
                out.append(
                    FMPProfile(
                        symbol=_as_str(item.get("symbol")).upper() or sym,
                        company_name=_as_str(
                            item.get("companyName") or item.get("company_name")
                        ),
                        price=_as_float(item.get("price")),
                        market_cap=_as_float(
                            item.get("marketCap") or item.get("mktCap")
                        ),
                        beta=_as_float(item.get("beta")),
                        average_volume=_as_float(
                            item.get("averageVolume")
                            if "averageVolume" in item
                            else item.get("volAvg")
                        ),
                        sector=_as_str(item.get("sector")),
                        industry=_as_str(item.get("industry")),
                        exchange=_as_str(
                            item.get("exchange")
                            or item.get("exchangeFullName")
                            or item.get("exchangeShortName")
                        ),
                        currency=_as_str(item.get("currency")),
                    )
                )
            except Exception:
                continue
        return out

    def get_earnings_calendar(
        self, date_from: str, date_to: str
    ) -> List[FMPEarningsEvent]:
        d_from = (date_from or "").strip()
        d_to = (date_to or "").strip()
        if not d_from or not d_to:
            return []
        payload = self._request_json(
            "earnings-calendar", {"from": d_from, "to": d_to}
        )
        out: List[FMPEarningsEvent] = []
        for item in self._ensure_list(payload):
            try:
                out.append(
                    FMPEarningsEvent(
                        symbol=_as_str(item.get("symbol")).upper(),
                        date=_as_str(item.get("date")),
                        eps_actual=_as_float(
                            item.get("epsActual") or item.get("eps")
                        ),
                        eps_estimated=_as_float(
                            item.get("epsEstimated")
                            or item.get("epsEstimate")
                        ),
                        revenue_actual=_as_float(
                            item.get("revenueActual") or item.get("revenue")
                        ),
                        revenue_estimated=_as_float(
                            item.get("revenueEstimated")
                            or item.get("revenueEstimate")
                        ),
                        last_updated=_as_str(
                            item.get("lastUpdated")
                            or item.get("updatedFromDate")
                        ),
                    )
                )
            except Exception:
                continue
        return out

    def get_price_target_consensus(
        self, symbol: str
    ) -> List[FMPPriceTargetConsensus]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return []
        payload = self._request_json(
            "price-target-consensus", {"symbol": sym}
        )
        out: List[FMPPriceTargetConsensus] = []
        for item in self._ensure_list(payload):
            try:
                out.append(
                    FMPPriceTargetConsensus(
                        symbol=_as_str(item.get("symbol")).upper() or sym,
                        target_high=_as_float(item.get("targetHigh")),
                        target_low=_as_float(item.get("targetLow")),
                        target_consensus=_as_float(item.get("targetConsensus")),
                        target_median=_as_float(item.get("targetMedian")),
                    )
                )
            except Exception:
                continue
        return out

    def get_analyst_estimates_annual(
        self, symbol: str
    ) -> List[FMPAnalystEstimate]:
        sym = (symbol or "").strip().upper()
        if not sym:
            return []
        payload = self._request_json(
            "analyst-estimates",
            {"symbol": sym, "period": "annual"},
        )
        out: List[FMPAnalystEstimate] = []
        for item in self._ensure_list(payload):
            try:
                out.append(
                    FMPAnalystEstimate(
                        symbol=_as_str(item.get("symbol")).upper() or sym,
                        date=_as_str(item.get("date")),
                        revenue_low=_as_float(item.get("revenueLow")),
                        revenue_high=_as_float(item.get("revenueHigh")),
                        revenue_avg=_as_float(
                            item.get("revenueAvg")
                            or item.get("estimatedRevenueAvg")
                        ),
                        eps_low=_as_float(item.get("epsLow")),
                        eps_high=_as_float(item.get("epsHigh")),
                        eps_avg=_as_float(
                            item.get("epsAvg")
                            or item.get("estimatedEpsAvg")
                        ),
                    )
                )
            except Exception:
                continue
        return out

    def get_sec_filings(
        self, symbol: str, date_from: str, date_to: str
    ) -> List[FMPSecFiling]:
        sym = (symbol or "").strip().upper()
        d_from = (date_from or "").strip()
        d_to = (date_to or "").strip()
        if not sym or not d_from or not d_to:
            return []
        payload = self._request_json(
            "sec-filings-search/symbol",
            {"symbol": sym, "from": d_from, "to": d_to},
        )
        out: List[FMPSecFiling] = []
        for item in self._ensure_list(payload):
            try:
                out.append(
                    FMPSecFiling(
                        symbol=_as_str(item.get("symbol")).upper() or sym,
                        cik=_as_str(item.get("cik")),
                        filing_date=_as_str(
                            item.get("filingDate") or item.get("fillingDate")
                        ),
                        accepted_date=_as_str(item.get("acceptedDate")),
                        form_type=_as_str(
                            item.get("formType") or item.get("type")
                        ),
                        link=_as_str(item.get("link")),
                        final_link=_as_str(item.get("finalLink")),
                    )
                )
            except Exception:
                continue
        return out


__all__ = [
    "FMPClient",
    "FMPQuote",
    "FMPProfile",
    "FMPEarningsEvent",
    "FMPPriceTargetConsensus",
    "FMPAnalystEstimate",
    "FMPSecFiling",
    "FMP_BASE_URL",
    "FMP_DEFAULT_TIMEOUT_SECONDS",
]

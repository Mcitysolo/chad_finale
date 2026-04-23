"""
chad/analytics/institutional_consensus.py

Aggregates 13F holdings across a set of tracked funds to surface the stocks
that multiple top institutions own with high conviction. The output feeds
the Beta strategy, which builds long-term positions weighted by this
aggregated consensus.

The aggregation is deliberately simple:
  - fund_count     : how many tracked funds hold the stock
  - total_value    : aggregate USD value of the position across funds
  - avg_pct        : average share of each holder's portfolio the stock is
  - conviction     : normalized score = 0.5*(fund_count/N) + 0.5*(share of total value)

Fail-soft: given 13F holdings that vary in how they label issuers (name +
CUSIP), we aggregate by CUSIP first and carry the issuer name through for
display. Symbol resolution is a separate, best-effort step driven by a
name-to-ticker map that can be expanded over time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSENSUS_PATH = REPO_ROOT / "runtime" / "institutional_consensus.json"

# Name-to-ticker map for resolving SEC-reported issuer names to trading symbols.
# 13F reports by issuer name + CUSIP, never by ticker. This map is starter
# coverage for the most frequently held large-caps; anything not here falls
# through and is labelled "UNRESOLVED" so operators can extend the map.
#
# Keys are uppercased + collapsed-whitespace issuer names. Values are U.S.
# exchange tickers. Kept small on purpose — only add names once you've
# confirmed the ticker mapping.
NAME_TO_TICKER: Dict[str, str] = {
    # Mega-cap tech
    "APPLE INC": "AAPL",
    "MICROSOFT CORP": "MSFT",
    "ALPHABET INC": "GOOGL",          # class A (13F filers vary between GOOG/GOOGL)
    "ALPHABET INC CL A": "GOOGL",
    "ALPHABET INC CL C": "GOOG",
    "ALPHABET INC COM CL A": "GOOGL",
    "ALPHABET INC COM CL C": "GOOG",
    "GOOG": "GOOGL",                  # defensive: ticker-as-name folds into GOOGL
    "AMAZON COM INC": "AMZN",
    "AMAZON.COM INC": "AMZN",
    "META PLATFORMS INC": "META",
    "META PLATFORMS INC CL A": "META",
    "NVIDIA CORP": "NVDA",
    "NVIDIA CORPORATION": "NVDA",
    "TESLA INC": "TSLA",
    "NETFLIX INC": "NFLX",
    # Mega-cap non-tech
    "BERKSHIRE HATHAWAY INC DEL": "BRK.B",
    "BERKSHIRE HATHAWAY INC DEL CL B": "BRK.B",
    "BERKSHIRE HATHAWAY INC": "BRK.B",
    "JPMORGAN CHASE & CO": "JPM",
    "JOHNSON & JOHNSON": "JNJ",
    "BANK OF AMERICA CORP": "BAC",
    "UNITEDHEALTH GROUP INC": "UNH",
    "VISA INC": "V",
    "VISA INC COM CL A": "V",
    "MASTERCARD INC": "MA",
    "MASTERCARD INCORPORATED": "MA",
    "PROCTER & GAMBLE CO": "PG",
    "PROCTER & GAMBLE CO THE": "PG",
    "HOME DEPOT INC": "HD",
    "EXXON MOBIL CORP": "XOM",
    "CHEVRON CORP NEW": "CVX",
    "COCA COLA CO": "KO",
    "COCA-COLA CO": "KO",
    "PEPSICO INC": "PEP",
    "WALT DISNEY CO": "DIS",
    "WALT DISNEY CO/THE": "DIS",
    "COSTCO WHOLESALE CORP NEW": "COST",
    "COSTCO WHOLESALE CORP": "COST",
    "WALMART INC": "WMT",
    "MCDONALDS CORP": "MCD",
    "MC DONALDS CORP": "MCD",
    "ELI LILLY & CO": "LLY",
    "LILLY ELI & CO": "LLY",
    "BROADCOM INC": "AVGO",
    "ADOBE INC": "ADBE",
    "SALESFORCE INC": "CRM",
    "ORACLE CORP": "ORCL",
    "QUALCOMM INC": "QCOM",
    "INTEL CORP": "INTC",
    "ADVANCED MICRO DEVICES INC": "AMD",
    "CISCO SYSTEMS INC": "CSCO",
    "IBM": "IBM",
    "INTERNATIONAL BUSINESS MACHS CORP": "IBM",
    # Index/ETF exposures (sometimes appear in 13Fs)
    "SPDR S&P 500 ETF TR": "SPY",
    "INVESCO QQQ TR UNIT SER 1": "QQQ",
}


@dataclass
class ConsensusEntry:
    """A single aggregated consensus record."""
    symbol: str                      # resolved ticker (or "UNRESOLVED:<cusip>")
    name_of_issuer: str              # human-readable issuer name
    cusip: str
    fund_count: int                  # number of tracked funds holding this stock
    fund_holders: List[str]          # names of holding funds
    total_value_usd: float           # summed across funds
    total_shares: int                # summed across funds
    avg_pct_portfolio: float         # average % of each holder's portfolio
    conviction_score: float          # 0..1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name_of_issuer": self.name_of_issuer,
            "cusip": self.cusip,
            "fund_count": self.fund_count,
            "fund_holders": list(self.fund_holders),
            "total_value_usd": float(self.total_value_usd),
            "total_shares": int(self.total_shares),
            "avg_pct_portfolio": float(self.avg_pct_portfolio),
            "conviction_score": float(self.conviction_score),
        }


def _normalize_name(name: str) -> str:
    """Uppercase + collapse whitespace for NAME_TO_TICKER lookups."""
    return " ".join((name or "").upper().split())


def _resolve_symbol(name_of_issuer: str, cusip: str) -> Optional[str]:
    """Resolve issuer name to a ticker via NAME_TO_TICKER. None if unknown."""
    key = _normalize_name(name_of_issuer)
    if not key:
        return None
    return NAME_TO_TICKER.get(key)


class InstitutionalConsensus:
    """
    Aggregates 13F holdings across funds into a ranked consensus table.

    Inputs match the shape produced by SEC13FFetcher.get_all_fund_holdings:
        {
            "fund_a": {"holdings": [Holding, ...], "report_date": "...",
                       "source": "...", "accession": "..."},
            ...
        }
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.log = logger or LOG

    def compute_consensus(
        self,
        all_holdings: Dict[str, Dict[str, Any]],
        top_n: int = 25,
        *,
        include_unresolved: bool = False,
    ) -> List[ConsensusEntry]:
        """
        Aggregate by CUSIP across all fund holdings.

        Parameters
        ----------
        all_holdings : dict
            Mapping from fund_name -> {"holdings": [Holding, ...], ...}
        top_n : int
            Cap on the number of entries returned (ranked by conviction).
        include_unresolved : bool
            If False (default), drop rows where we can't resolve the issuer
            name to a ticker. Useful for production Beta wiring.
            If True, keep them with symbol="UNRESOLVED:<cusip>" for ops triage.
        """
        # Per-fund portfolio totals (used to weight "% of portfolio")
        fund_totals: Dict[str, float] = {}
        for fund, payload in all_holdings.items():
            holdings = (payload or {}).get("holdings") or []
            fund_totals[fund] = float(sum(
                (h.to_dict()["value_usd"] if hasattr(h, "to_dict") else float(h.get("value_usd", 0.0)))
                for h in holdings
            ))

        # Aggregate by CUSIP
        by_cusip: Dict[str, Dict[str, Any]] = {}
        for fund, payload in all_holdings.items():
            holdings = (payload or {}).get("holdings") or []
            fund_total = fund_totals.get(fund) or 0.0
            for raw in holdings:
                h = raw.to_dict() if hasattr(raw, "to_dict") else raw
                cusip = str(h.get("cusip") or "").strip().upper()
                if not cusip:
                    continue
                value = float(h.get("value_usd") or 0.0)
                shares = int(h.get("shares") or 0)
                if value <= 0 or shares <= 0:
                    continue
                pct = (value / fund_total) if fund_total > 0 else 0.0

                bucket = by_cusip.setdefault(cusip, {
                    "name_of_issuer": h.get("name_of_issuer") or "",
                    "cusip": cusip,
                    "fund_holders": [],
                    "total_value_usd": 0.0,
                    "total_shares": 0,
                    "pct_by_fund": [],  # for averaging
                })
                if fund not in bucket["fund_holders"]:
                    bucket["fund_holders"].append(fund)
                bucket["total_value_usd"] += value
                bucket["total_shares"] += shares
                bucket["pct_by_fund"].append(pct)
                # Prefer longer / more-specific issuer labels
                if len(h.get("name_of_issuer") or "") > len(bucket["name_of_issuer"]):
                    bucket["name_of_issuer"] = h["name_of_issuer"]

        if not by_cusip:
            return []

        # Scoring: blend of breadth (fund_count / tracked_funds) and weight
        # (this position's share of total value across all aggregated).
        total_value_all = sum(b["total_value_usd"] for b in by_cusip.values()) or 1.0
        tracked_funds = max(1, len(all_holdings))

        entries: List[ConsensusEntry] = []
        for cusip, b in by_cusip.items():
            fund_count = len(b["fund_holders"])
            breadth = fund_count / tracked_funds
            depth = b["total_value_usd"] / total_value_all
            conviction = 0.5 * breadth + 0.5 * depth
            avg_pct = (sum(b["pct_by_fund"]) / len(b["pct_by_fund"])) if b["pct_by_fund"] else 0.0
            symbol = _resolve_symbol(b["name_of_issuer"], cusip)
            resolved = symbol is not None
            if not resolved:
                if not include_unresolved:
                    continue
                symbol = f"UNRESOLVED:{cusip}"

            entries.append(ConsensusEntry(
                symbol=symbol,
                name_of_issuer=b["name_of_issuer"],
                cusip=cusip,
                fund_count=fund_count,
                fund_holders=list(b["fund_holders"]),
                total_value_usd=b["total_value_usd"],
                total_shares=b["total_shares"],
                avg_pct_portfolio=avg_pct,
                conviction_score=conviction,
            ))

        # Dedup by resolved symbol: different CUSIPs (e.g. GOOG/GOOGL share
        # classes of ALPHABET INC) can resolve to the same ticker. Merge
        # fund_holders (union), sum totals, keep the max conviction score
        # and the labels from the higher-conviction entry.
        by_symbol: Dict[str, ConsensusEntry] = {}
        for e in entries:
            existing = by_symbol.get(e.symbol)
            if existing is None:
                by_symbol[e.symbol] = e
                continue
            merged_holders = list(dict.fromkeys(existing.fund_holders + e.fund_holders))
            keep = existing if existing.conviction_score >= e.conviction_score else e
            by_symbol[e.symbol] = ConsensusEntry(
                symbol=keep.symbol,
                name_of_issuer=keep.name_of_issuer,
                cusip=keep.cusip,
                fund_count=len(merged_holders),
                fund_holders=merged_holders,
                total_value_usd=existing.total_value_usd + e.total_value_usd,
                total_shares=existing.total_shares + e.total_shares,
                avg_pct_portfolio=keep.avg_pct_portfolio,
                conviction_score=max(existing.conviction_score, e.conviction_score),
            )
        entries = list(by_symbol.values())

        entries.sort(key=lambda e: e.conviction_score, reverse=True)
        return entries[:top_n]

    def get_consensus_weights(
        self,
        consensus: List[ConsensusEntry],
    ) -> Dict[str, float]:
        """
        Convert conviction scores into portfolio weights summing to 1.0.

        Empty or degenerate inputs yield an empty dict so callers can detect
        "no consensus available" cleanly.
        """
        if not consensus:
            return {}
        total = sum(max(0.0, e.conviction_score) for e in consensus)
        if total <= 0:
            return {}
        weights: Dict[str, float] = {}
        for e in consensus:
            if e.symbol.startswith("UNRESOLVED:"):
                continue
            w = max(0.0, e.conviction_score) / total
            if w > 0:
                weights[e.symbol] = w
        # Re-normalize after filtering unresolved
        w_sum = sum(weights.values())
        if w_sum > 0:
            for k in list(weights):
                weights[k] /= w_sum
        return weights

    # ---- Persistence --------------------------------------------------

    def write_cache(
        self,
        consensus: List[ConsensusEntry],
        weights: Dict[str, float],
        funds_included: List[str],
        path: Optional[Path] = None,
    ) -> Path:
        """Write the consensus snapshot to runtime/institutional_consensus.json."""
        out_path = Path(path) if path is not None else CONSENSUS_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "institutional_consensus.v1",
            "updated_ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "funds_included": sorted(funds_included),
            "top_holdings": [e.to_dict() for e in consensus],
            "weights": dict(weights),
        }
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        return out_path

    def load_cache(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Read the consensus JSON. Returns the full dict or None if missing/bad.

        Beta consumes this via the dict directly (weights key).
        """
        p = Path(path) if path is not None else CONSENSUS_PATH
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.log.warning("institutional_consensus: load failed path=%s err=%s", p, exc)
            return None

#!/usr/bin/env python3
"""
chad/market_data/futures_contract_resolver.py

Deterministic futures front-month resolver.

The hot execution path is forbidden from calling reqContractDetails (P0-1 rule).
This module provides a static schedule + resolver shared by:
  - chad.execution.execution_pipeline.build_ibkr_intents_from_plan (intent enrichment)
  - chad.market_data.ibkr_historical_provider (bar backfill)

The schedule is a flat YYYYMM list per symbol, ordered ascending. Resolver
selects the first expiry at least 5 days from today; if the first listed
expiry is the same calendar month as the cutoff, it is accepted only when
the cutoff day-of-month is <= 20 (avoids picking an expiry that is about
to roll).

Update annually or when a roll schedule changes. Keep in sync with the
mirror copy in IBKRHistoricalProvider._EXPIRY_SCHEDULE until that path
is migrated to import from here.

Source: confirmed against live IBKR contract data 2026-04-10.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


EXPIRY_SCHEDULE: Dict[str, List[str]] = {
    "MES":  ["202606", "202609", "202612", "202703", "202706"],
    "MNQ":  ["202606", "202609", "202612", "202703", "202706"],
    # NYMEX crude-oil contracts expire in the calendar month BEFORE their
    # delivery month: MCLM6 (June delivery) last traded 2026-05-18 per IBKR
    # truth. Drop "202606" from the schedule so the resolver advances to
    # "202607" once "202606" has expired. (Equity-index / metal micro-futures
    # listed below keep the standard "expires in delivery month" calendar.)
    "MCL":  ["202607", "202608", "202609", "202610", "202611", "202612", "202701"],
    "MGC":  ["202604", "202606", "202608", "202610", "202612", "202702", "202704"],
    "ZN":   ["202606", "202609", "202612", "202703"],
    "ZB":   ["202606", "202609", "202612", "202703"],
    "M6E":  ["202606", "202609", "202612", "202703"],
    "SIL":  ["202605", "202606", "202607", "202608", "202609", "202610", "202611", "202612"],
    "MYM":  ["202606", "202609", "202612", "202703", "202706"],
    "M2K":  ["202606", "202609", "202612", "202703", "202706"],
}


SOURCE_NAME = "chad.market_data.futures_contract_resolver"


def resolve_contract_month(symbol: str, *, now: Optional[datetime] = None) -> Optional[str]:
    """
    Return YYYYMM front-month expiry for `symbol`, or None if unsupported.

    Deterministic. Performs no network I/O. Safe for hot path.
    """
    sym = (symbol or "").strip().upper()
    schedule = EXPIRY_SCHEDULE.get(sym)
    if not schedule:
        return None
    today = now or datetime.now(timezone.utc)
    cutoff = today + timedelta(days=5)
    cutoff_str = cutoff.strftime("%Y%m")
    cutoff_day = cutoff.day
    for expiry in schedule:
        if expiry > cutoff_str:
            return expiry
        if expiry == cutoff_str and cutoff_day <= 20:
            return expiry
    return schedule[-1]


__all__ = ["EXPIRY_SCHEDULE", "SOURCE_NAME", "resolve_contract_month"]

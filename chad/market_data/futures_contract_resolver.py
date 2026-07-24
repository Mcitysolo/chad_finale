#!/usr/bin/env python3
"""
chad/market_data/futures_contract_resolver.py

Deterministic, roll-aware futures front-month resolver.

The hot execution path is forbidden from calling reqContractDetails (P0-1 rule).
This module provides the single canonical contract calendar shared by:
  - chad.execution.execution_pipeline.build_ibkr_intents_from_plan (intent enrichment)
  - chad.core.position_reconciler (close-intent enrichment)
  - chad.market_data.ibkr_historical_provider (bar backfill)

W6A-1 — what changed and why
----------------------------
The previous implementation compared hand-maintained ``YYYYMM`` strings at
*month* granularity with a ``cutoff_day <= 20`` heuristic, and fell back to
``schedule[-1]`` when the hand-written list ran out. That produced two defects:

  1. It could not express "expires in the month BEFORE the delivery month".
     NYMEX crude (MCL) delivery-month entries were therefore treated as if
     they expired in the delivery month. On 2026-07-23 the resolver still
     returned ``202608`` even though that contract last traded 2026-07-21 —
     evidenced by MCL being the only symbol in ``data/bars/1d`` whose last
     bar was 2026-07-21 while all nine others were 2026-07-22.
  2. Exhaustion was silent. ``return schedule[-1]`` handed back an expired
     month with no exception and no log once the list ran out (first cliff:
     SIL, ~Dec 2026).

Both are fixed here by computing an actual **last-trade DATE** per contract
family and generating contract months from a per-symbol cycle rather than
maintaining a literal list. Exhaustion now raises
:class:`ExpiryScheduleExhausted` — it is a backstop, not a code path.

Business-day approximation
--------------------------
Business days are Mon-Fri. Exchange holidays are NOT modelled (the repo has
no holiday calendar; ``chad/utils/market_hours.py`` documents the same
limitation), so a computed last-trade date can be *optimistic* — later than
the exchange's true date. Worked example: MCL Jun-2026 computes to
2026-05-20, while IBKR truth is 2026-05-18 (Memorial Day on 05-25 shifts the
anchor); the computed date is 2 days LATE.

``ROLL_BUFFER_DAYS`` (5) is what makes that safe, and is why it must stay
comfortably larger than the worst-case holiday error: a contract is selected
only if its computed last-trade date is at least 5 days out, so a 2-day
optimistic error still leaves ~3 days of real life. The net effect is that we
roll *early*, never late onto an expired contract. Erring early costs a few
days of a slightly-less-liquid back month; erring late is defect (1) above.
Do not lower ROLL_BUFFER_DAYS without adding a holiday calendar.

Calendar primitives (``third_friday`` and friends) are canonical here.
``chad.market_data.futures_roll_publisher`` (the "B4" roll calendar) grew the
original third-Friday implementation; the resolver is the lower-level module,
so the primitives live here and the publisher delegates to them (W6A-4)
rather than the hot path importing a publisher.

Source: rules confirmed against live IBKR contract data 2026-04-10; MCL rule
re-validated 2026-07-23 against observed bar data (see above).
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Tuple


class ExpiryScheduleExhausted(RuntimeError):
    """No contract month with a future last-trade date within the horizon.

    Raised instead of silently returning a stale/expired month. Reaching this
    means the generation horizon was exceeded or a family rule is broken —
    either way the caller must fail loudly, never trade a guessed contract.
    """


SOURCE_NAME = "chad.market_data.futures_contract_resolver"

# A contract must have at least this many days of remaining life to be picked.
# Also absorbs the holiday approximation described in the module docstring.
ROLL_BUFFER_DAYS: int = 5

# How far ahead month generation runs before declaring exhaustion.
HORIZON_YEARS: int = 5

# --- contract cycles -------------------------------------------------------
_QUARTERLY: Tuple[int, ...] = (3, 6, 9, 12)
_EVEN_MONTHS: Tuple[int, ...] = (2, 4, 6, 8, 10, 12)
_ALL_MONTHS: Tuple[int, ...] = tuple(range(1, 13))
# COMEX silver trades a Mar/May/Jul/Sep/Dec liquid cycle. The serial months
# (Jan/Feb/Apr/Jun/Aug/Oct/Nov) are listed but near-dead: on 2026-07-23 the
# Aug-2026 contract that the old month-granularity resolver selected had a
# median 20-day volume of 179 contracts, against 107,654 for MGC on the same
# day — roughly 600x thinner. Serial months are excluded so resolution can
# only ever land on a liquid contract. See also W6A-3: SIL is data-only and
# is deliberately NOT part of the tradable/gated universe.
_SILVER_LIQUID: Tuple[int, ...] = (3, 5, 7, 9, 12)

# --- last-trade-date rule families ----------------------------------------
# equity_index_3rd_friday : 3rd Friday of the contract month (CME equity micros)
# rates_7bd_before_eom    : 7th business day preceding the last business day
#                           of the delivery month (CBOT treasuries)
# fx_2bd_before_3rd_wed   : 2 business days before the 3rd Wednesday (CME FX)
# energy_3bd_before_25th  : 3 business days before the 25th calendar day of
#                           the month PRIOR to the delivery month; if the 25th
#                           is not a business day, count back from the
#                           business day on-or-before it (NYMEX WTI / MCL)
# metals_3rd_last_bd      : 3rd-last business day of the delivery month (COMEX)
FAMILY_EQUITY_INDEX = "equity_index_3rd_friday"
FAMILY_RATES = "rates_7bd_before_eom"
FAMILY_FX = "fx_2bd_before_3rd_wed"
FAMILY_ENERGY = "energy_3bd_before_25th"
FAMILY_METALS = "metals_3rd_last_bd"

# symbol -> (family, cycle months)
CONTRACT_SPECS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "MES": (FAMILY_EQUITY_INDEX, _QUARTERLY),
    "MNQ": (FAMILY_EQUITY_INDEX, _QUARTERLY),
    "MYM": (FAMILY_EQUITY_INDEX, _QUARTERLY),
    "M2K": (FAMILY_EQUITY_INDEX, _QUARTERLY),
    "ZN":  (FAMILY_RATES, _QUARTERLY),
    "ZB":  (FAMILY_RATES, _QUARTERLY),
    "M6E": (FAMILY_FX, _QUARTERLY),
    "MCL": (FAMILY_ENERGY, _ALL_MONTHS),
    "MGC": (FAMILY_METALS, _EVEN_MONTHS),
    "SIL": (FAMILY_METALS, _SILVER_LIQUID),
}


# ---------------------------------------------------------------------------
# Calendar primitives (canonical; Mon-Fri business days, no holiday calendar)
# ---------------------------------------------------------------------------

def third_friday(year: int, month: int) -> date:
    """Return the third-Friday date of (year, month)."""
    first = date(year, month, 1)
    days_to_friday = (4 - first.weekday()) % 7
    return first + timedelta(days=days_to_friday) + timedelta(weeks=2)


def third_wednesday(year: int, month: int) -> date:
    """Return the third-Wednesday date of (year, month)."""
    first = date(year, month, 1)
    days_to_wed = (2 - first.weekday()) % 7
    return first + timedelta(days=days_to_wed) + timedelta(weeks=2)


def is_business_day(d: date) -> bool:
    """Mon-Fri. Exchange holidays are not modelled (see module docstring)."""
    return d.weekday() < 5


def business_day_on_or_before(d: date) -> date:
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def subtract_business_days(d: date, n: int) -> date:
    """Step back ``n`` business days from ``d`` (``d`` itself not counted)."""
    out = d
    for _ in range(n):
        out -= timedelta(days=1)
        while not is_business_day(out):
            out -= timedelta(days=1)
    return out


def nth_last_business_day(year: int, month: int, n: int) -> date:
    """Return the ``n``-th last business day of (year, month). n=1 is last."""
    last = date(year, month, calendar.monthrange(year, month)[1])
    return subtract_business_days(business_day_on_or_before(last), n - 1)


def _prev_month(year: int, month: int) -> Tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


# ---------------------------------------------------------------------------
# Last-trade date
# ---------------------------------------------------------------------------

def last_trade_date(symbol: str, year: int, month: int) -> Optional[date]:
    """Last trade date for ``symbol``'s (year, month) contract.

    ``month`` is the contract/delivery month. Returns None for unknown symbols.
    """
    spec = CONTRACT_SPECS.get((symbol or "").strip().upper())
    if spec is None:
        return None
    family, _cycle = spec

    if family == FAMILY_EQUITY_INDEX:
        return third_friday(year, month)
    if family == FAMILY_RATES:
        # 7th business day preceding the last business day of the month.
        return subtract_business_days(nth_last_business_day(year, month, 1), 7)
    if family == FAMILY_FX:
        return subtract_business_days(third_wednesday(year, month), 2)
    if family == FAMILY_ENERGY:
        # Terminates in the month PRIOR to delivery: 3 business days before
        # the 25th (or the business day on-or-before the 25th).
        py, pm = _prev_month(year, month)
        anchor = business_day_on_or_before(date(py, pm, 25))
        return subtract_business_days(anchor, 3)
    if family == FAMILY_METALS:
        return nth_last_business_day(year, month, 3)
    return None


def iter_contract_months(
    symbol: str,
    *,
    start: Optional[date] = None,
    horizon_years: int = HORIZON_YEARS,
) -> Iterator[Tuple[int, int]]:
    """Yield (year, month) contract months on ``symbol``'s cycle, ascending."""
    spec = CONTRACT_SPECS.get((symbol or "").strip().upper())
    if spec is None:
        return
    _family, cycle = spec
    begin = start or datetime.now(timezone.utc).date()
    for year in range(begin.year, begin.year + horizon_years + 1):
        for month in cycle:
            if (year, month) < (begin.year, begin.month):
                continue
            yield (year, month)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_contract_month(
    symbol: str,
    *,
    now: Optional[datetime] = None,
    roll_buffer_days: int = ROLL_BUFFER_DAYS,
) -> Optional[str]:
    """Return the ``YYYYMM`` front-month for ``symbol``.

    Selects the first contract on the symbol's cycle whose computed last-trade
    date is at least ``roll_buffer_days`` in the future. Deterministic; no
    network I/O; safe for the hot path.

    Returns None for symbols this module does not model (unchanged contract —
    callers already treat None as "cannot enrich").

    Raises :class:`ExpiryScheduleExhausted` for a *known* symbol when no
    contract inside the horizon is still live. This replaces the previous
    silent ``schedule[-1]`` fallback, which returned an expired month.
    """
    sym = (symbol or "").strip().upper()
    if sym not in CONTRACT_SPECS:
        return None

    today = (now or datetime.now(timezone.utc)).date()
    cutoff = today + timedelta(days=roll_buffer_days)

    for year, month in iter_contract_months(sym, start=today):
        ltd = last_trade_date(sym, year, month)
        if ltd is None:
            continue
        if ltd >= cutoff:
            return f"{year:04d}{month:02d}"

    raise ExpiryScheduleExhausted(
        f"no live contract for {sym} within {HORIZON_YEARS}y of {today.isoformat()} "
        f"(cutoff={cutoff.isoformat()}) — refusing to return an expired month"
    )


def resolve_contract_expiry_date(
    symbol: str,
    *,
    now: Optional[datetime] = None,
    roll_buffer_days: int = ROLL_BUFFER_DAYS,
) -> Optional[date]:
    """Last-trade date of the contract :func:`resolve_contract_month` returns."""
    month = resolve_contract_month(symbol, now=now, roll_buffer_days=roll_buffer_days)
    if month is None:
        return None
    return last_trade_date(symbol, int(month[:4]), int(month[4:]))


def _build_compat_schedule(horizon_years: int = 3) -> Dict[str, List[str]]:
    """Derived ``YYYYMM`` lists, kept only for backwards compatibility.

    DEPRECATED. The resolver no longer reads this — months are generated from
    ``CONTRACT_SPECS``. Retained so any external reader of ``EXPIRY_SCHEDULE``
    keeps working; do not add symbols here.
    """
    today = datetime.now(timezone.utc).date()
    out: Dict[str, List[str]] = {}
    for sym in CONTRACT_SPECS:
        out[sym] = [
            f"{year:04d}{month:02d}"
            for year, month in iter_contract_months(sym, start=today, horizon_years=horizon_years)
        ]
    return out


EXPIRY_SCHEDULE: Dict[str, List[str]] = _build_compat_schedule()


__all__ = [
    "CONTRACT_SPECS",
    "EXPIRY_SCHEDULE",
    "ExpiryScheduleExhausted",
    "HORIZON_YEARS",
    "ROLL_BUFFER_DAYS",
    "SOURCE_NAME",
    "business_day_on_or_before",
    "is_business_day",
    "iter_contract_months",
    "last_trade_date",
    "nth_last_business_day",
    "resolve_contract_expiry_date",
    "resolve_contract_month",
    "subtract_business_days",
    "third_friday",
    "third_wednesday",
]

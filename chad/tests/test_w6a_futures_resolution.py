#!/usr/bin/env python3
"""
chad/tests/test_w6a_futures_resolution.py

W6A Lane A — roll-aware futures contract resolution.

Covers:
  1. The MCL live defect: the resolver must not return a contract that has
     already stopped trading. Anchored to real evidence — MCL Aug-2026 last
     traded 2026-07-21, which is why data/bars/1d/MCL.json stopped at 07-21
     while all nine other futures reached 07-22.
  2. Exhaustion is LOUD. The old `return schedule[-1]` silently handed back an
     expired month; it must now raise.
  3. The private mirror schedule stays deleted.
  4. Family last-trade-date rules, each pinned to a hand-computed date.
  5. Every symbol resolves to a contract with real remaining life, on any day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from chad.market_data.futures_contract_resolver import (
    CONTRACT_SPECS,
    ROLL_BUFFER_DAYS,
    ExpiryScheduleExhausted,
    last_trade_date,
    resolve_contract_expiry_date,
    resolve_contract_month,
    third_friday,
    third_wednesday,
)
from chad.market_data.ibkr_historical_provider import IBKRHistoricalProvider

ALL_SYMBOLS = ("MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL", "MYM", "M2K")


# ---------------------------------------------------------------------------
# 1) The MCL defect this lane exists to fix
# ---------------------------------------------------------------------------


def test_mcl_aug_2026_last_trade_matches_observed_bar_data() -> None:
    """The energy rule must reproduce reality.

    data/bars/1d/MCL.json stopped at 2026-07-21 while the other nine futures
    reached 2026-07-22 — that gap IS the Aug-2026 contract expiring.
    """
    assert last_trade_date("MCL", 2026, 8) == date(2026, 7, 21)


def test_mcl_does_not_resolve_to_expired_august_contract() -> None:
    """The live defect: on 2026-07-23 the old resolver still returned 202608,
    two days after that contract stopped trading."""
    now = datetime(2026, 7, 23, tzinfo=timezone.utc)
    assert resolve_contract_month("MCL", now=now) == "202609"


def test_mcl_rolls_before_expiry_not_after() -> None:
    """Walk each day across the Aug->Sep roll. The resolver must never name a
    contract whose last-trade date has already passed."""
    for offset in range(0, 40):
        now = datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=offset)
        month = resolve_contract_month("MCL", now=now)
        ltd = last_trade_date("MCL", int(month[:4]), int(month[4:]))
        assert ltd >= now.date(), f"{now.date()} resolved to expired {month} (ltd={ltd})"


# ---------------------------------------------------------------------------
# 2) Exhaustion is loud, never a stale month
# ---------------------------------------------------------------------------


def test_exhaustion_raises_instead_of_returning_expired_month() -> None:
    """The old code did `return schedule[-1]` — an expired month, silently.

    Demand more remaining life (6y) than the 5y generation horizon can offer,
    so every candidate is rejected and the resolver runs off the end.
    """
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ExpiryScheduleExhausted):
        resolve_contract_month("MES", now=now, roll_buffer_days=365 * 6)


def test_unknown_symbol_still_returns_none_not_raises() -> None:
    """Callers (position_reconciler, execution_pipeline) treat None as
    'cannot enrich'. That contract must not change."""
    assert resolve_contract_month("ZZZ") is None
    assert resolve_contract_month("") is None


# ---------------------------------------------------------------------------
# 3) The mirror stays dead
# ---------------------------------------------------------------------------


def test_w6a_no_private_expiry_schedule() -> None:
    """IBKRHistoricalProvider carried a byte-identical hand-maintained copy of
    the resolver's schedule. Two calendars, nothing keeping them in sync."""
    assert not hasattr(IBKRHistoricalProvider, "_EXPIRY_SCHEDULE"), (
        "private expiry mirror reintroduced — resolution must delegate to "
        "chad.market_data.futures_contract_resolver"
    )


def test_provider_delegates_to_canonical_resolver() -> None:
    now = datetime(2026, 7, 23, tzinfo=timezone.utc)
    for sym in ALL_SYMBOLS:
        assert IBKRHistoricalProvider._resolve_front_month(sym, now=now) == \
            resolve_contract_month(sym, now=now)


def test_provider_returns_none_on_exhaustion_never_a_stale_month(monkeypatch) -> None:
    """Exhaustion must degrade to None (symbol skipped, isolated + reported),
    never to an expired contract month."""
    def _boom(*_a, **_kw):
        raise ExpiryScheduleExhausted("forced")

    monkeypatch.setattr(
        "chad.market_data.futures_contract_resolver.resolve_contract_month", _boom
    )
    assert IBKRHistoricalProvider._resolve_front_month("MES") is None


# ---------------------------------------------------------------------------
# 4) Family rules, each pinned to a hand-computed date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,year,month,expected",
    [
        # equity index — 3rd Friday of the contract month
        ("MES", 2026, 9, date(2026, 9, 18)),
        ("M2K", 2026, 12, date(2026, 12, 18)),
        # rates — 7th business day preceding the last business day
        ("ZN", 2026, 9, date(2026, 9, 21)),
        ("ZB", 2026, 9, date(2026, 9, 21)),
        # FX — 2 business days before the 3rd Wednesday (2026-09-16)
        ("M6E", 2026, 9, date(2026, 9, 14)),
        # energy — 3 business days before the 25th of the PRIOR month
        ("MCL", 2026, 8, date(2026, 7, 21)),
        ("MCL", 2026, 9, date(2026, 8, 20)),
        # metals — 3rd-last business day of the delivery month
        ("MGC", 2026, 8, date(2026, 8, 27)),
        ("SIL", 2026, 9, date(2026, 9, 28)),
    ],
)
def test_family_last_trade_dates(symbol: str, year: int, month: int, expected: date) -> None:
    assert last_trade_date(symbol, year, month) == expected


def test_calendar_primitives() -> None:
    assert third_friday(2026, 9) == date(2026, 9, 18)
    assert third_wednesday(2026, 9) == date(2026, 9, 16)


def test_sil_only_resolves_to_liquid_months() -> None:
    """SIL's Aug-2026 serial had median 20d volume 179 vs MGC's 107,654.
    Resolution must stay on the Mar/May/Jul/Sep/Dec liquid cycle."""
    liquid = {3, 5, 7, 9, 12}
    for offset in range(0, 400, 7):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset)
        month = resolve_contract_month("SIL", now=now)
        assert int(month[4:]) in liquid, f"{now.date()} resolved SIL to serial month {month}"


# ---------------------------------------------------------------------------
# 5) No symbol, on any day, resolves to something already dead
# ---------------------------------------------------------------------------


def test_no_symbol_ever_resolves_to_an_expired_contract() -> None:
    for offset in range(0, 730, 5):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=offset)
        for sym in ALL_SYMBOLS:
            ltd = resolve_contract_expiry_date(sym, now=now)
            assert ltd is not None
            assert ltd >= now.date() + timedelta(days=ROLL_BUFFER_DAYS), (
                f"{sym} on {now.date()} resolved to {ltd}, inside the roll buffer"
            )


def test_every_known_symbol_is_modelled() -> None:
    assert set(CONTRACT_SPECS) == set(ALL_SYMBOLS)

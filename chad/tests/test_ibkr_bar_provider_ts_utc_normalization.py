"""PA-EP6 — bar-writer ts_utc UTC-normalization guards.

Covers chad/market_data/ibkr_bar_provider._bar_ts_to_utc_iso():
- tz-aware intraday datetimes (formatDate=1 exchange-local) are converted to
  true UTC, instant-preserving, for both EDT (-04:00) and EST (-05:00)
- naive datetimes are assumed UTC (matches the repo's other ts parsers)
- daily `date` objects pass through via str() unchanged
- missing/empty values degrade to "" without crashing

No IBKR connection or network required — pure function under test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from chad.market_data import ibkr_bar_provider as bp


def test_aware_edt_datetime_converts_to_utc() -> None:
    """19:00 EDT (-04:00) → 23:00 UTC, same instant."""
    edt = timezone(timedelta(hours=-4))
    raw = datetime(2026, 6, 12, 19, 0, 0, tzinfo=edt)
    assert bp._bar_ts_to_utc_iso(raw) == "2026-06-12 23:00:00+00:00"


def test_aware_est_datetime_rolls_to_next_utc_day() -> None:
    """19:00 EST (-05:00) → 00:00 UTC next day, same instant."""
    est = timezone(timedelta(hours=-5))
    raw = datetime(2026, 1, 5, 19, 0, 0, tzinfo=est)
    assert bp._bar_ts_to_utc_iso(raw) == "2026-01-06 00:00:00+00:00"


def test_naive_datetime_assumed_utc() -> None:
    """Naive datetime is stamped UTC with no wall-clock shift."""
    raw = datetime(2026, 6, 12, 19, 0, 0)
    assert bp._bar_ts_to_utc_iso(raw) == "2026-06-12 19:00:00+00:00"


def test_date_object_passes_through_via_str() -> None:
    """Daily-bar `date` objects are not datetimes → str() passthrough."""
    raw = date(2026, 6, 12)
    assert bp._bar_ts_to_utc_iso(raw) == str(raw) == "2026-06-12"


def test_empty_string_degrades_safely() -> None:
    """Missing date attribute ('' fallback) must not crash."""
    assert bp._bar_ts_to_utc_iso("") == ""

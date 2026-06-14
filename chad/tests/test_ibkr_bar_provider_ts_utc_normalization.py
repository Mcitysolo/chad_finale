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
from chad.market_data.ibkr_historical_provider import IBKRHistoricalProvider


class _FakeBar:
    """Minimal stand-in for an ib_async historical bar row."""

    def __init__(self, raw_date: object) -> None:
        self.date = raw_date
        self.open = 100.0
        self.high = 101.0
        self.low = 99.0
        self.close = 100.5
        self.volume = 1000.0


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


# --- PA-EP6b: co-writer durability — the historical provider's _parse_bars
# path must emit true UTC too, so a backfill cannot re-introduce local-labeled
# bars into data/bars/1m and data/bars/1d.


def test_historical_provider_aware_edt_bar_emits_true_utc() -> None:
    """Backfill path: 19:00 EDT (-04:00) bar → 23:00 UTC ts_utc, same instant."""
    edt = timezone(timedelta(hours=-4))
    provider = IBKRHistoricalProvider(ib=None)
    bars = provider._parse_bars(
        [_FakeBar(datetime(2026, 6, 12, 19, 0, 0, tzinfo=edt))], "SPY"
    )
    assert len(bars) == 1
    assert bars[0].ts_utc == "2026-06-12 23:00:00+00:00"


def test_historical_provider_daily_date_passes_through() -> None:
    """Backfill path: daily `date` rows pass through str()-unchanged."""
    provider = IBKRHistoricalProvider(ib=None)
    bars = provider._parse_bars([_FakeBar(date(2026, 6, 12))], "SPY")
    assert len(bars) == 1
    assert bars[0].ts_utc == "2026-06-12"

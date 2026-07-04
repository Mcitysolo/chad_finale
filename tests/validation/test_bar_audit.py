"""Unit tests for chad.validation.bar_audit (Phase 0 data-quality audit).

Fixtures are constructed in memory — these tests never depend on the real 52-file
corpus, so they stay deterministic and fast. Each check has a test proving it
fires on a crafted violation and stays silent on a clean series.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from chad.validation.bar_audit import (
    AuditConfig,
    CorpusAudit,
    Finding,
    Status,
    SymbolAudit,
    audit_bar_file,
    audit_corpus,
    audit_symbol,
    classify_currency,
    render_corpus_summary,
    render_symbol_audit,
)


# --------------------------------------------------------------------------- #
# Fixture builders — clean weekday series; violations derived by mutation.
# --------------------------------------------------------------------------- #
def _next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def make_clean_bars(
    n: int = 100,
    start: date = date(2025, 1, 6),  # a Monday
    base: float = 100.0,
    step: float = 0.5,
    volume: float = 1000.0,
) -> list[dict]:
    """n consecutive weekday bars, strictly increasing distinct closes, valid OHLC."""
    bars: list[dict] = []
    d = start
    while d.weekday() >= 5:
        d = _next_weekday(d)
    price = base
    for _ in range(n):
        close = round(price, 2)
        open_ = round(price - 0.1, 2)
        high = round(max(open_, close) + 0.2, 2)
        low = round(min(open_, close) - 0.2, 2)
        bars.append(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "ts_utc": d.isoformat()}
        )
        price += step
        d = _next_weekday(d)
    return bars


def _codes(audit: SymbolAudit) -> set[str]:
    return {f.code for f in audit.findings}


# --------------------------------------------------------------------------- #
# 1. Clean series → CLEAN.
# --------------------------------------------------------------------------- #
def test_clean_series_is_clean():
    audit = audit_symbol("AAPL", make_clean_bars(), source="ibkr", timeframe="1d")
    assert audit.status is Status.CLEAN, [f.to_dict() for f in audit.findings]
    assert audit.findings == ()
    assert audit.bar_count == 100
    assert audit.first_date == "2025-01-06"
    assert audit.quote_currency == "USD"
    assert audit.currency_provenance == "assumed_usd_undeclared"


# --------------------------------------------------------------------------- #
# 2. OHLC sanity violation → FAIL.
# --------------------------------------------------------------------------- #
def test_ohlc_violation_fails():
    bars = make_clean_bars()
    # high below the close is a hard OHLC violation.
    bars[30] = dict(bars[30], high=1.0, low=0.5, open=50.0, close=60.0)
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.FAIL
    assert "OHLC_VIOLATION" in _codes(audit)


def test_non_positive_price_fails():
    bars = make_clean_bars()
    bars[10] = dict(bars[10], open=-5.0, high=-1.0, low=-9.0, close=-5.0)
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.FAIL
    ohlc = [f for f in audit.findings if f.code == "OHLC_VIOLATION"]
    assert ohlc and "non_positive_price" in ohlc[0].detail["examples"][0]["reasons"]


# --------------------------------------------------------------------------- #
# 3. Injected gap → WARN.
# --------------------------------------------------------------------------- #
def test_injected_gap_warns():
    bars = make_clean_bars()
    gapped = bars[:40] + bars[45:]  # remove 5 consecutive weekday sessions
    audit = audit_symbol("AAPL", gapped, source="ibkr")
    assert audit.status is Status.WARN
    assert "UNUSUAL_GAP" in _codes(audit)
    assert audit.metrics["largest_gap_sessions"] >= 4


def test_weekend_is_not_a_gap():
    # A plain consecutive-weekday series has no gap findings at all.
    audit = audit_symbol("AAPL", make_clean_bars(), source="ibkr")
    assert "UNUSUAL_GAP" not in _codes(audit)
    assert audit.metrics["gap_count"] == 0


def test_crypto_uses_calendar_days_for_gaps():
    # Continuous market: 7-day cadence; skipping 4 calendar days is a gap.
    bars = []
    d = date(2025, 1, 1)
    for i in range(30):
        step = 5 if i == 15 else 1  # a 4-session hole mid-series
        bars.append(
            {"open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i, "close": 100.5 + i, "volume": 10.0,
             "ts_utc": d.isoformat()}
        )
        d = d + timedelta(days=step)
    audit = audit_symbol("BTC-USD", bars, source="kraken")
    assert audit.metrics["continuous_market"] is True
    assert "UNUSUAL_GAP" in _codes(audit)


# --------------------------------------------------------------------------- #
# 4. Stale prints → WARN.
# --------------------------------------------------------------------------- #
def test_stale_close_run_warns():
    bars = make_clean_bars()
    for i in range(10, 16):  # 6 identical closes
        bars[i] = dict(bars[i], open=120.0, high=120.0, low=120.0, close=120.0)
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.WARN
    assert "STALE_CLOSE_RUN" in _codes(audit)
    assert audit.metrics["max_stale_close_run"] >= 6


def test_zero_volume_run_warns():
    bars = make_clean_bars()
    for i in range(0, 6):  # 6 consecutive zero-volume bars (MES-style synthetic prints)
        bars[i] = dict(bars[i], volume=0.0)
    audit = audit_symbol("MES", bars, source="ibkr")
    assert "ZERO_VOLUME_RUN" in _codes(audit)
    assert audit.metrics["max_zero_volume_run"] >= 6
    assert audit.metrics["zero_volume_total"] >= 6


# --------------------------------------------------------------------------- #
# 5. Duplicate timestamp → FAIL; non-monotonic order → WARN.
# --------------------------------------------------------------------------- #
def test_duplicate_timestamp_fails():
    bars = make_clean_bars()
    bars[20] = dict(bars[20], ts_utc=bars[19]["ts_utc"])
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.FAIL
    assert "DUPLICATE_TIMESTAMP" in _codes(audit)
    assert audit.metrics["duplicate_count"] == 1


def test_non_monotonic_order_warns():
    bars = make_clean_bars(n=40)
    bars[10], bars[11] = bars[11], bars[10]  # swap dates out of order
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert "NON_MONOTONIC_TIMESTAMPS" in _codes(audit)
    assert audit.metrics["non_monotonic_count"] >= 1


# --------------------------------------------------------------------------- #
# 6. Suspected unadjusted split → WARN (report only).
# --------------------------------------------------------------------------- #
def test_suspected_split_warns():
    bars = make_clean_bars()
    prev_close = bars[49]["close"]
    doubled = round(prev_close * 2.0, 2)
    bars[50] = dict(bars[50], open=doubled, high=doubled + 1, low=doubled - 1, close=doubled)
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.WARN
    assert "SUSPECTED_UNADJUSTED" in _codes(audit)
    assert audit.metrics["suspected_split_count"] >= 1


# --------------------------------------------------------------------------- #
# 7. Malformed row → FAIL.
# --------------------------------------------------------------------------- #
def test_malformed_missing_field_fails():
    bars = make_clean_bars()
    bars[5] = {"open": 1.0, "high": 2.0, "low": 0.5, "ts_utc": bars[5]["ts_utc"]}  # no close/volume
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.FAIL
    assert "MALFORMED_ROWS" in _codes(audit)
    assert audit.metrics["malformed_count"] == 1


def test_malformed_non_numeric_and_bool_fail():
    bars = make_clean_bars()
    bars[5] = dict(bars[5], close="oops")
    bars[6] = dict(bars[6], volume=True)  # bool must be rejected as non-numeric
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert audit.status is Status.FAIL
    assert audit.metrics["malformed_count"] == 2


def test_bad_timestamp_is_malformed():
    bars = make_clean_bars()
    bars[7] = dict(bars[7], ts_utc="not-a-date")
    audit = audit_symbol("AAPL", bars, source="ibkr")
    assert "MALFORMED_ROWS" in _codes(audit)


# --------------------------------------------------------------------------- #
# 8. Coverage / freshness.
# --------------------------------------------------------------------------- #
def test_thin_history_warns():
    audit = audit_symbol("M6E", make_clean_bars(n=30), source="ibkr")
    assert "THIN_HISTORY" in _codes(audit)


def test_empty_series_fails():
    audit = audit_symbol("AAPL", [], source="ibkr")
    assert audit.status is Status.FAIL
    assert "NO_VALID_BARS" in _codes(audit)
    assert audit.bar_count == 0


def test_stale_feed_warns_against_corpus_last_date():
    bars = make_clean_bars()
    last = date.fromisoformat(bars[-1]["ts_utc"])
    audit = audit_symbol(
        "AAPL", bars, source="ibkr", corpus_last_date=last + timedelta(days=30)
    )
    assert "STALE_FEED" in _codes(audit)
    assert audit.metrics["coverage_lag_days"] >= 10


# --------------------------------------------------------------------------- #
# FX provenance (SSOT §5).
# --------------------------------------------------------------------------- #
def test_fx_cross_rate_symbol_warns():
    audit = audit_symbol("M6E", make_clean_bars(), source="ibkr")
    assert audit.status is Status.WARN
    assert "FX_CROSS_RATE" in _codes(audit)
    assert audit.currency_provenance == "fx_cross_rate"


def test_cad_symbol_flagged():
    audit = audit_symbol("SHOP.TO", make_clean_bars(), source="ibkr")
    assert "CAD_QUOTED" in _codes(audit)
    assert audit.quote_currency == "CAD"


def test_classify_currency_matrix():
    assert classify_currency("BTC-USD", "kraken") == ("USD", "explicit_symbol_suffix")
    assert classify_currency("VIX", "cboe") == ("USD", "index_points_no_currency")
    assert classify_currency("M6E", "ibkr") == ("USD", "fx_cross_rate")
    assert classify_currency("AAPL", "ibkr") == ("USD", "assumed_usd_undeclared")


# --------------------------------------------------------------------------- #
# Config knobs are honoured.
# --------------------------------------------------------------------------- #
def test_config_threshold_changes_verdict():
    bars = make_clean_bars()
    for i in range(10, 13):  # 3 identical closes
        bars[i] = dict(bars[i], open=120.0, high=120.0, low=120.0, close=120.0)
    strict = AuditConfig(stale_close_run=3)
    lenient = AuditConfig(stale_close_run=10)
    assert "STALE_CLOSE_RUN" in _codes(audit_symbol("AAPL", bars, config=strict))
    assert "STALE_CLOSE_RUN" not in _codes(audit_symbol("AAPL", bars, config=lenient))


# --------------------------------------------------------------------------- #
# Corpus aggregation + file I/O.
# --------------------------------------------------------------------------- #
def _write_bar_file(path, symbol, bars, source="ibkr"):
    path.write_text(
        json.dumps({"bars": bars, "symbol": symbol, "source": source, "timeframe": "1d", "ts_utc": "x"}),
        encoding="utf-8",
    )


def test_audit_corpus_aggregates(tmp_path):
    clean = make_clean_bars()
    fail_bars = make_clean_bars()
    fail_bars[3] = dict(fail_bars[3], high=1.0, low=0.5, open=50.0, close=60.0)  # OHLC violation
    warn_bars = make_clean_bars()[:40] + make_clean_bars()[45:]  # gap

    _write_bar_file(tmp_path / "AAA.json", "AAA", clean)
    _write_bar_file(tmp_path / "BBB.json", "BBB", fail_bars)
    _write_bar_file(tmp_path / "CCC.json", "CCC", warn_bars)

    corpus = audit_corpus(tmp_path)
    assert isinstance(corpus, CorpusAudit)
    assert corpus.symbol_count == 3
    assert corpus.clean == 1
    assert corpus.warn == 1
    assert corpus.fail == 1
    assert corpus.status_by_symbol["AAA"] == "CLEAN"
    assert corpus.status_by_symbol["BBB"] == "FAIL"
    assert corpus.status_by_symbol["CCC"] == "WARN"
    # corpus-level undeclared-currency note fires (all three are assumed-USD).
    assert any(f.code == "UNDECLARED_CURRENCY_CORPUS" for f in corpus.corpus_findings)


def test_audit_corpus_stale_feed_cross_check(tmp_path):
    fresh = make_clean_bars(start=date(2025, 3, 3))  # ends later
    stale = make_clean_bars(start=date(2024, 6, 3))  # ends ~months earlier
    _write_bar_file(tmp_path / "FRESH.json", "FRESH", fresh)
    _write_bar_file(tmp_path / "STALE.json", "STALE", stale)
    corpus = audit_corpus(tmp_path)
    stale_audit = next(s for s in corpus.symbols if s.symbol == "STALE")
    assert "STALE_FEED" in {f.code for f in stale_audit.findings}


def test_audit_bar_file_invalid_json_fails(tmp_path):
    bad = tmp_path / "BAD.json"
    bad.write_text("{not valid json", encoding="utf-8")
    audit = audit_bar_file(bad)
    assert audit.status is Status.FAIL
    assert audit.findings[0].code == "INVALID_JSON"


def test_audit_bar_file_missing_bars_array_fails(tmp_path):
    f = tmp_path / "NB.json"
    f.write_text(json.dumps({"symbol": "NB", "source": "ibkr"}), encoding="utf-8")
    audit = audit_bar_file(f)
    assert audit.status is Status.FAIL
    assert audit.findings[0].code == "MISSING_BARS_ARRAY"


def test_symbol_filename_mismatch_warns(tmp_path):
    f = tmp_path / "AAPL.json"
    _write_bar_file(f, "MSFT", make_clean_bars())  # declared != filename
    audit = audit_bar_file(f)
    assert "SYMBOL_FILENAME_MISMATCH" in {fd.code for fd in audit.findings}


# --------------------------------------------------------------------------- #
# Serialisation, determinism, rendering.
# --------------------------------------------------------------------------- #
def test_to_dict_is_json_serialisable():
    audit = audit_symbol("AAPL", make_clean_bars(), source="ibkr")
    blob = json.dumps(audit.to_dict())
    assert '"status": "CLEAN"' in blob


def test_corpus_to_dict_is_json_serialisable(tmp_path):
    _write_bar_file(tmp_path / "AAA.json", "AAA", make_clean_bars())
    corpus = audit_corpus(tmp_path)
    blob = json.dumps(corpus.to_dict())
    assert "usdcad_conversion_constant" in blob


def test_determinism():
    bars = make_clean_bars()
    a = audit_symbol("AAPL", bars, source="ibkr").to_dict()
    b = audit_symbol("AAPL", bars, source="ibkr").to_dict()
    assert a == b


def test_renderers_return_text():
    audit = audit_symbol("AAPL", make_clean_bars(), source="ibkr")
    line = render_symbol_audit(audit)
    assert "AAPL" in line and "CLEAN" in line


def test_render_corpus_summary(tmp_path):
    _write_bar_file(tmp_path / "AAA.json", "AAA", make_clean_bars())
    corpus = audit_corpus(tmp_path)
    text = render_corpus_summary(corpus)
    assert "BAR CORPUS DATA-QUALITY AUDIT" in text
    assert "CLEAN=1" in text
    assert "thresholds:" in text  # active config echoed into the rendered report


def test_corpus_to_dict_config_via_asdict(tmp_path):
    # to_dict must serialise every AuditConfig field (asdict, not a hand-list).
    _write_bar_file(tmp_path / "AAA.json", "AAA", make_clean_bars())
    corpus = audit_corpus(tmp_path, config=AuditConfig(stale_close_run=7))
    cfg = corpus.to_dict()["config"]
    assert set(cfg) == {
        "stale_close_run", "zero_volume_run", "gap_session_threshold",
        "split_ratio_high", "split_ratio_low", "thin_history_bars",
        "coverage_lag_days", "max_examples",
    }
    assert cfg["stale_close_run"] == 7


# --------------------------------------------------------------------------- #
# Read-only guarantee: auditing does not mutate the input list/dicts.
# --------------------------------------------------------------------------- #
def test_audit_does_not_mutate_input():
    bars = make_clean_bars()
    snapshot = json.dumps(bars)
    audit_symbol("AAPL", bars, source="ibkr")
    assert json.dumps(bars) == snapshot

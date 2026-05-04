"""Regression tests for the 1m-preferring bar_timestamp source.

Covers the 2026-05-04 stale data_freshness_gate fix where
build_ibkr_intents_from_plan / build_kraken_intents read intent
bar_timestamp exclusively from data/bars/1d/, causing every Monday
intent to age out (~89h > 48h threshold) even though fresh 1m bars
existed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.execution import execution_pipeline as ep
from chad.execution.routing_gates import data_freshness_gate


def _write_bar_file(tmp_root: Path, timeframe: str, symbol: str, ts_utc: str) -> Path:
    d = tmp_root / "data" / "bars" / timeframe
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{symbol.upper()}.json"
    p.write_text(
        json.dumps(
            {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "ts_utc": ts_utc,
                "bars": [
                    {
                        "open": 100.0,
                        "high": 100.5,
                        "low": 99.5,
                        "close": 100.25,
                        "volume": 1000,
                        "ts_utc": ts_utc,
                    }
                ],
            }
        )
    )
    return p


def _patch_bar_root(monkeypatch, tmp_root: Path) -> None:
    """Redirect the loader's filesystem reads at tmp_root.

    The production helper resolves data/bars/{tf}/{SYMBOL}.json relative
    to ``__file__.parents[2]``. Tests substitute a stub that reads from
    ``tmp_root`` instead so they never touch the real data tree.
    """

    def _stub(symbol: str, timeframe: str):
        if not symbol or not timeframe:
            return None
        path = tmp_root / "data" / "bars" / str(timeframe) / f"{symbol.upper()}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        bars = data.get("bars")
        if not isinstance(bars, list) or not bars:
            return None
        last = bars[-1]
        return last if isinstance(last, dict) else None

    monkeypatch.setattr(ep, "_load_latest_bar_for_symbol_with_timeframe", _stub)


def test_execution_pipeline_prefers_1m_bar_timestamp_when_available(monkeypatch, tmp_path):
    _write_bar_file(tmp_path, "1d", "SPY", "2026-05-01")
    _write_bar_file(tmp_path, "1m", "SPY", "2026-05-04 12:41:00-04:00")
    _patch_bar_root(monkeypatch, tmp_path)

    ts = ep._load_latest_bar_ts_for_symbol("SPY")
    assert ts == "2026-05-04 12:41:00-04:00"


def test_execution_pipeline_falls_back_to_1d_when_1m_missing(monkeypatch, tmp_path):
    _write_bar_file(tmp_path, "1d", "SPY", "2026-05-01")
    # No 1m file written.
    _patch_bar_root(monkeypatch, tmp_path)

    ts = ep._load_latest_bar_ts_for_symbol("SPY")
    assert ts == "2026-05-01"


def test_execution_pipeline_returns_empty_when_neither_source_available(monkeypatch, tmp_path):
    _patch_bar_root(monkeypatch, tmp_path)
    assert ep._load_latest_bar_ts_for_symbol("ZZZZ") == ""


def test_data_freshness_gate_passes_with_fresh_1m_timestamp():
    """A fresh 1m timestamp (~2 min old) must pass the non-intraday gate."""

    class _Intent:
        strategy = "delta"
        symbol = "SPY"
        signal_family = "trend"
        asset_class = "ETF"

    fresh_dt = datetime.now(timezone.utc) - timedelta(seconds=120)
    passed, reason = data_freshness_gate(
        _Intent(), bar_timestamp=fresh_dt, max_bar_age_seconds=172800
    )
    assert passed is True
    assert reason == "ok"


def test_date_only_daily_bar_monday_age_regression(monkeypatch):
    """Pin the failure mode the fix targets.

    A daily-bar ts_utc of '2026-05-01' (Friday's date string) parses to
    midnight UTC. From a Mon-afternoon ``now``, age exceeds the 172800s
    non-intraday threshold; a fresh 1m bar from the same Mon afternoon
    is comfortably under it.
    """

    parsed_daily = ep._parse_bar_timestamp("2026-05-01")
    assert parsed_daily is not None and parsed_daily.tzinfo is not None
    parsed_1m = ep._parse_bar_timestamp("2026-05-04 12:41:00-04:00")
    assert parsed_1m is not None and parsed_1m.tzinfo is not None

    frozen_now = datetime(2026, 5, 4, 16, 54, tzinfo=timezone.utc)
    age_daily = (frozen_now - parsed_daily).total_seconds()
    age_1m = (frozen_now - parsed_1m).total_seconds()

    assert age_daily > 172800  # confirms the original failure
    assert age_1m < 172800  # confirms the fix path is viable

    class _Intent:
        strategy = "delta"
        symbol = "SPY"
        signal_family = "trend"
        asset_class = "ETF"

    from chad.execution import routing_gates as rg

    monkeypatch.setattr(rg, "_utcnow", lambda: frozen_now)

    daily_passed, daily_reason = rg.data_freshness_gate(
        _Intent(), bar_timestamp=parsed_daily, max_bar_age_seconds=172800
    )
    one_min_passed, one_min_reason = rg.data_freshness_gate(
        _Intent(), bar_timestamp=parsed_1m, max_bar_age_seconds=172800
    )
    # The 1m path is viable; the daily-only path under the same now is the
    # exact bar_stale signature observed in production.
    assert daily_passed is False
    assert "bar_stale" in daily_reason
    # The 1m bar at 2026-05-04 12:41-04:00 (≈16:41 UTC) is ~13 min old at
    # the frozen now=16:54 UTC, well inside the 172800s budget.
    assert one_min_passed is True
    assert one_min_reason == "ok"

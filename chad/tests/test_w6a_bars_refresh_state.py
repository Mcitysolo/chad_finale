#!/usr/bin/env python3
"""
chad/tests/test_w6a_bars_refresh_state.py

W6A-3 (D1) — per-symbol exit-status isolation + bars_refresh_state.v1.

Two things must hold together:
  * the unit stops going red because ONE symbol failed, and
  * a partial failure is never SILENT (Lane B's exit-0-with-nothing lesson).

The cohort-lag assertions are anchored to the real 2026-07-23 shape: MCL's
last bar was 2026-07-21 while all nine other futures reached 2026-07-22,
because MCL was pinned to a contract that had stopped trading. Absolute
staleness would not have flagged it — the +1 day versus its peers is the tell.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

import chad.market_data.nightly_bars_refresh as nbr


# ---------------------------------------------------------------------------
# Age + cohort-lag maths
# ---------------------------------------------------------------------------


def test_age_days_from_bar_date() -> None:
    assert nbr._age_days("2026-07-21", date(2026, 7, 23)) == 2
    assert nbr._age_days("2026-07-22", date(2026, 7, 23)) == 1
    assert nbr._age_days(None, date(2026, 7, 23)) is None
    assert nbr._age_days("not-a-date", date(2026, 7, 23)) is None


@pytest.fixture
def bars_dir(tmp_path, monkeypatch):
    d = tmp_path / "bars"
    d.mkdir()
    monkeypatch.setattr(nbr, "BARS_1D_DIR", d)
    monkeypatch.setattr(nbr, "REPO_ROOT", tmp_path)
    (tmp_path / "runtime").mkdir()
    return d


def _write_bars(bars_dir, symbol: str, last_date: str) -> None:
    (bars_dir / f"{symbol}.json").write_text(
        json.dumps({"symbol": symbol, "bars": [{"ts_utc": last_date, "close": 1.0}]})
    )


def _state(tmp_path) -> dict:
    return json.loads((tmp_path / "runtime" / "bars_refresh_state.json").read_text())


def test_lagging_symbol_is_visible_the_mcl_signature(bars_dir, tmp_path) -> None:
    """The real 2026-07-23 shape: MCL one day behind an otherwise-healthy cohort."""
    futures = ["MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL", "MYM", "M2K"]
    for sym in futures:
        _write_bars(bars_dir, sym, "2026-07-21" if sym == "MCL" else "2026-07-22")

    results = {s: True for s in futures}  # every symbol "succeeded"
    nbr._write_bars_refresh_state(results, futures, today=date(2026, 7, 23))
    state = _state(tmp_path)

    assert state["schema_version"] == "bars_refresh_state.v1"
    assert state["fail"] == 0, "every symbol reported success..."
    assert state["lagging_symbols"] == ["MCL"], "...yet MCL must still surface as lagging"
    assert state["symbols"]["MCL"]["age_vs_cohort_days"] == 1
    assert state["symbols"]["MES"]["age_vs_cohort_days"] == 0


def test_failed_symbol_reports_the_age_of_its_stale_file(bars_dir, tmp_path) -> None:
    """A symbol that failed today still has to say how dark it has gone."""
    _write_bars(bars_dir, "MES", "2026-07-22")
    _write_bars(bars_dir, "MCL", "2026-06-01")

    nbr._write_bars_refresh_state({"MES": True, "MCL": False}, ["MES", "MCL"], today=date(2026, 7, 23))
    state = _state(tmp_path)

    assert state["symbols"]["MCL"]["status"] == "fail"
    assert state["symbols"]["MCL"]["last_bar_utc"] == "2026-06-01"
    assert state["symbols"]["MCL"]["age_vs_cohort_days"] > 0
    assert "MCL" in state["lagging_symbols"]


def test_crypto_cadence_does_not_mark_every_ibkr_symbol_lagging(bars_dir, tmp_path) -> None:
    """Kraken bars land same-day, IBKR bars a day behind. Comparing across the
    two sources would flag every equity and future as lagging."""
    for sym in ("BTC-USD", "ETH-USD"):
        _write_bars(bars_dir, sym, "2026-07-23")   # crypto: same-day
    for sym in ("SPY", "AAPL"):
        _write_bars(bars_dir, sym, "2026-07-22")   # equities: one day behind
    for sym in ("MES", "MCL"):
        _write_bars(bars_dir, sym, "2026-07-22")

    results = {s: True for s in ("BTC-USD", "ETH-USD", "SPY", "AAPL", "MES", "MCL")}
    nbr._write_bars_refresh_state(results, ["MES", "MCL"], today=date(2026, 7, 23))
    state = _state(tmp_path)

    assert state["lagging_symbols"] == [], "cross-source cadence must not read as lag"
    assert state["group_min_age_days"]["crypto"] == 0
    assert state["group_min_age_days"]["equity"] == 1
    assert state["group_min_age_days"]["future"] == 1


def test_stale_majority_cannot_hide_a_laggard(bars_dir, tmp_path) -> None:
    """A median reference is dragged along by the stale symbols themselves.
    The group minimum is not: one fresh symbol still exposes the rest."""
    _write_bars(bars_dir, "MES", "2026-07-22")                     # the only fresh one
    for sym in ("MCL", "MGC", "ZN", "ZB", "M6E"):
        _write_bars(bars_dir, sym, "2026-07-10")                   # stale majority

    futures = ["MES", "MCL", "MGC", "ZN", "ZB", "M6E"]
    nbr._write_bars_refresh_state({s: True for s in futures}, futures, today=date(2026, 7, 23))
    state = _state(tmp_path)

    assert state["lagging_symbols"] == ["M6E", "MCL", "MGC", "ZB", "ZN"]
    assert state["symbols"]["MES"]["age_vs_cohort_days"] == 0


def test_symbol_with_no_bar_file_is_called_out(bars_dir, tmp_path) -> None:
    _write_bars(bars_dir, "MES", "2026-07-22")
    nbr._write_bars_refresh_state({"MES": True, "GHOST": False}, ["MES", "GHOST"], today=date(2026, 7, 23))
    state = _state(tmp_path)

    assert state["no_data_symbols"] == ["GHOST"]
    assert state["symbols"]["GHOST"]["age_days"] is None


def test_state_write_failure_never_propagates(bars_dir, tmp_path, monkeypatch) -> None:
    """Observability must never be able to fail the refresh."""
    def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(nbr, "_atomic_write_json", _boom)
    nbr._write_bars_refresh_state({"MES": True}, ["MES"])  # must not raise


# ---------------------------------------------------------------------------
# Exit-status policy (D1)
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, results: dict, ib_connected: bool = True) -> int:
    monkeypatch.setattr(nbr, "_load_universe", lambda: (["SPY"], [{"symbol": "MES"}]))

    def _fake_ibkr(_eq, _fut, res):
        res.update(results)
        return ib_connected

    monkeypatch.setattr(nbr, "_run_ibkr", _fake_ibkr)
    monkeypatch.setattr(nbr, "_run_kraken", lambda _r: None)
    monkeypatch.setattr(nbr, "_run_vix", lambda _r: None)
    monkeypatch.setattr(nbr, "_write_bars_refresh_state", lambda *_a, **_kw: None)
    return nbr.main()


def test_one_bad_symbol_no_longer_reds_the_unit(monkeypatch) -> None:
    """The EXS5 mechanism: `failure > 0 -> return 1` reddened the unit whenever
    a single contract out of ~52 failed."""
    results = {f"SYM{i}": True for i in range(51)}
    results["MCL"] = False
    assert _run_main(monkeypatch, results) == 0


def test_all_symbols_failing_still_fails_the_unit(monkeypatch) -> None:
    assert _run_main(monkeypatch, {"MES": False, "SPY": False}) == 1


def test_no_symbols_attempted_is_systemic(monkeypatch) -> None:
    assert _run_main(monkeypatch, {}) == 1


def test_dead_ibkr_connection_still_exits_2(monkeypatch) -> None:
    assert _run_main(monkeypatch, {"MES": False, "SPY": True}, ib_connected=False) == 2


def test_full_success_exits_zero(monkeypatch) -> None:
    assert _run_main(monkeypatch, {"MES": True, "SPY": True}) == 0

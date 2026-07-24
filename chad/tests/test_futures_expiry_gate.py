"""Unit tests for chad/market_data/futures_expiry_gate.py (FUTURES-ROLL-1)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chad.market_data import futures_expiry_gate as gate

NOW = datetime(2026, 5, 27, 15, 0, 0, tzinfo=timezone.utc)  # 2026-05-27, SILK6 expiry day


@pytest.fixture(autouse=True)
def _reset_dedupe():
    gate.reset_warning_dedupe()
    yield
    gate.reset_warning_dedupe()


def _state(symbols: dict) -> dict:
    return {
        "schema_version": "futures_roll_state.v1",
        "symbols": symbols,
        "status": "ok",
    }


def test_unexpired_contract_is_polled_normally():
    state = _state({
        "MES": {
            "current_expiry": "2026-06-19",
            "next_expiry": "2026-09-18",
            "block_new_entries": False,
            "roll_supported": True,
        }
    })
    v = gate.evaluate_symbol("MES", roll_state=state, now=NOW)
    assert v.skip is False
    assert v.reason == "not_expired"


def test_expired_contract_skipped_with_reason_expired():
    state = _state({
        "SIL": {
            "current_expiry": "2026-05-26",  # yesterday relative to NOW
            "next_expiry": "2026-07-29",
            "block_new_entries": True,
            "roll_supported": True,
        }
    })
    v = gate.evaluate_symbol("SIL", roll_state=state, now=NOW)
    assert v.skip is True
    assert v.reason == "expired"
    assert v.next_expiry == "2026-07-29"


def test_contract_expiring_today_skipped():
    state = _state({
        "SIL": {
            "current_expiry": "2026-05-27",  # today
            "next_expiry": "2026-07-29",
            "block_new_entries": True,
            "roll_supported": True,
        }
    })
    v = gate.evaluate_symbol("SIL", roll_state=state, now=NOW)
    assert v.skip is True
    assert v.reason == "expires_today"


def test_no_roll_mapping_does_not_skip_but_signals_reason():
    state = _state({})  # empty
    v = gate.evaluate_symbol("UNKNOWN", roll_state=state, now=NOW)
    assert v.skip is False
    assert v.reason == "no_roll_mapping"


def test_expiry_unknown_for_unsupported_roll_pattern():
    state = _state({
        "M6E": {
            "current_expiry": None,
            "next_expiry": None,
            "roll_pattern": "unsupported_v1",
            "roll_supported": False,
        }
    })
    v = gate.evaluate_symbol("M6E", roll_state=state, now=NOW)
    assert v.skip is False
    assert v.reason == "expiry_unknown"


def test_filter_universe_partitions_expired_and_fresh():
    state = _state({
        "MES": {"current_expiry": "2026-06-19", "next_expiry": "2026-09-18"},
        "SIL": {"current_expiry": "2026-05-26", "next_expiry": "2026-07-29"},  # expired
        "M2K": {"current_expiry": "2026-06-19", "next_expiry": "2026-09-18"},
    })
    kept, skipped = gate.filter_universe(
        ["MES", "SIL", "M2K"], roll_state=state, now=NOW
    )
    assert set(kept) == {"MES", "M2K"}
    assert len(skipped) == 1
    assert skipped[0].symbol == "SIL"
    assert skipped[0].next_expiry == "2026-07-29"


def test_warning_dedupe_fires_once_per_symbol_per_day():
    state = _state({
        "SIL": {"current_expiry": "2026-05-26", "next_expiry": "2026-07-29"},
    })
    captured: list = []

    def _log(verdict):
        captured.append(verdict.symbol)

    # First call: should log once
    gate.filter_universe(["SIL"], roll_state=state, now=NOW, log_callback=_log)
    assert captured == ["SIL"]

    # Second call same day: should NOT re-log
    gate.filter_universe(["SIL"], roll_state=state, now=NOW, log_callback=_log)
    assert captured == ["SIL"], f"expected dedupe; got {captured}"


def test_silk6_pattern_produces_zero_polls_in_filtered_universe():
    """Audit-specific: SIL with expiry 2026-05-27 (SILK6 contract) must be
    skipped by the filter, so the bar provider issues zero IBKR requests for
    it on or after that date."""
    state = _state({
        "SIL": {
            "current_expiry": "2026-05-27",  # SILK6 last trade
            "next_expiry": "2026-07-29",
            "block_new_entries": True,
            "roll_supported": True,
        },
    })
    kept, skipped = gate.filter_universe(["SIL"], roll_state=state, now=NOW)
    assert kept == []
    assert len(skipped) == 1
    assert skipped[0].symbol == "SIL"
    assert skipped[0].reason in ("expires_today", "expired")


def test_bar_provider_skips_expired_in_polling_loop(monkeypatch):
    """Integration: IBKRBarProvider._filter_expired_futures returns the
    expired symbols, and poll_once short-circuits without invoking
    fetch_historical_bars on them.

    W6A-5: this test used to read the real wall clock. It stubs MES with
    current_expiry=2026-06-19 and asserts MES is still polled — true when the
    test was written, false from 2026-06-20 onward, at which point the gate
    correctly skipped MES too and the assertion failed. The defect was in the
    test, not the product: poll_once threaded no ``now`` down to the gate,
    even though the gate has always accepted one. The clock is now pinned to
    NOW, the same instant the rest of this module already uses.
    """
    from chad.market_data import ibkr_bar_provider as ibp

    # Stub roll state. Dates are relative to NOW (2026-05-27):
    #   SIL expired the day before -> must be skipped
    #   MES expires three weeks out -> must still be polled
    fake_state = _state({
        "SIL": {"current_expiry": "2026-05-26", "next_expiry": "2026-07-29"},
        "MES": {"current_expiry": "2026-06-19", "next_expiry": "2026-09-18"},
    })
    monkeypatch.setattr(
        "chad.market_data.futures_expiry_gate._load_roll_state",
        lambda *a, **kw: fake_state,
    )

    class _StubIB:
        def isConnected(self):
            return True
        def sleep(self, _s):
            pass

    fetched: list = []

    class _Provider(ibp.IBKRBarProvider):
        def fetch_historical_bars(self, sym, **kw):
            fetched.append(sym)
            return []
        def write_1m_bars_file(self, sym, bars):
            pass

    provider = _Provider(_StubIB(), universe=["MES", "SIL", "M2K"])
    # The future-roll state knows SIL but not M2K; M2K should still be polled
    # (no_roll_mapping = no skip).
    results = provider.poll_once(per_symbol_delay_s=0.0, now=NOW)

    assert "SIL" not in fetched, f"SIL must be skipped; fetched={fetched}"
    assert "MES" in fetched
    assert "M2K" in fetched, "no_roll_mapping must not cause a skip"
    # results dict still contains SIL with bar_count=0 (skipped)
    assert results.get("SIL") == 0


def test_poll_once_gate_is_not_wall_clock_dependent(monkeypatch):
    """Regression guard for the time-bomb class itself.

    The same stubbed state must produce opposite verdicts at two pinned
    instants. If ``now`` ever stops being threaded through poll_once, the gate
    silently falls back to the real clock and this test fails.
    """
    from datetime import datetime, timezone

    from chad.market_data import ibkr_bar_provider as ibp

    fake_state = _state({"MES": {"current_expiry": "2026-06-19", "next_expiry": "2026-09-18"}})
    monkeypatch.setattr(
        "chad.market_data.futures_expiry_gate._load_roll_state",
        lambda *a, **kw: fake_state,
    )
    monkeypatch.setattr("chad.market_data.futures_expiry_gate.reset_warning_dedupe", lambda: None)

    class _StubIB:
        def isConnected(self):
            return True
        def sleep(self, _s):
            pass

    def _poll(now):
        fetched: list = []

        class _Provider(ibp.IBKRBarProvider):
            def fetch_historical_bars(self, sym, **kw):
                fetched.append(sym)
                return []
            def write_1m_bars_file(self, sym, bars):
                pass

        _Provider(_StubIB(), universe=["MES"]).poll_once(per_symbol_delay_s=0.0, now=now)
        return fetched

    before = datetime(2026, 6, 1, tzinfo=timezone.utc)
    after = datetime(2026, 7, 1, tzinfo=timezone.utc)

    assert "MES" in _poll(before), "live contract must be polled"
    assert "MES" not in _poll(after), "expired contract must be skipped"

"""
FU1-B6 (FLIP-UNBLOCK 2026-07-17): the overlay is exclusion-aware in its own right.

ULTRA_CLOSE_AUDIT §C / §F.2 / §H.7: the PA §3 CLAIMED the overlay "honors
``_EFFECTIVE_NON_CHAD_SYMBOLS``", but no exclusion term existed anywhere in the module.
Operator-owned symbols were skipped only accidentally, one layer downstream at the
``apply_close_intents`` chokepoint, and — worse — the phantom guard confirmed CHAD's position
against the operator's OWN broker shares (AAPL: guard says gamma owns 14, broker shows 7, all the
operator's → ``broker_confirmed_qty=7``, "confirmed"). Defense was single-layer and accidental:
refactor or bypass ``apply_close_intents`` and CHAD sells the operator's stock.

These tests pin the second, module-level layer:

  * an excluded symbol yields SKIP_EXCLUDED — no WOULD_CLOSE, no close intent, EVER;
  * the skip happens BEFORE the phantom confirmation, so the operator's shares are never read
    as confirmation of a CHAD position;
  * a non-excluded position in the same batch still evaluates normally;
  * the runner applies an explicit exclusion set;
  * the production resolver returns the reconciler's single-source set;
  * under pytest the runner defaults to NO exclusions (so fixtures may use real tickers).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from chad.risk import position_exit_overlay as pxo

UTC = timezone.utc
NOW = datetime(2026, 7, 17, 15, 0, 0, tzinfo=UTC)


def _cfg(mode="shadow"):
    return pxo.load_overlay_config({
        "mode": mode, "atr_period": 14, "atr_trail_mult": 2.5,
        "hard_stop_loss_pct": 0.08, "min_bars_for_atr": 16,
        "max_hold_days": {"equity": 20.0, "etf": 30.0, "default": 20.0},
    })


def _guard(strategy_entries, broker_entries):
    state = {"_version": 1}
    for key, (sym, side, qty, days) in strategy_entries.items():
        state[key] = {"open": True, "symbol": sym, "side": side, "quantity": qty,
                      "strategy": key.split("|")[0],
                      "opened_at": (NOW - timedelta(days=days)).isoformat()}
    for sym, (side, qty) in broker_entries.items():
        state[f"broker_sync|{sym}"] = {"open": True, "symbol": sym, "side": side,
                                       "quantity": qty, "strategy": "broker_sync"}
    return state


def _open(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("_")}


def _evaluate(state, prices, anchors, excluded=None):
    return pxo.evaluate_positions(
        open_positions=_open(state), guard_state=state,
        bars_by_symbol={}, price_by_symbol=prices, anchors=anchors,
        config=_cfg(), now_utc=NOW, excluded_symbols=excluded,
    )


def _by_key(result):
    return {v.position_key: v for v in result.verdicts}


# --------------------------------------------------------------------------- #
# core evaluate_positions exclusion
# --------------------------------------------------------------------------- #
def test_excluded_symbol_is_skipped_not_closed():
    # AAPL is at its hard stop (90 < 100*0.92) AND broker-confirmed — pre-B-6 this was a
    # WOULD_CLOSE that confirmed against the operator's shares. Now: SKIP_EXCLUDED.
    state = _guard({"gamma|AAPL": ("AAPL", "BUY", 14, 1)}, {"AAPL": ("BUY", 7)})
    anchors = {"gamma|AAPL": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {"AAPL": 90.0}, anchors, excluded={"AAPL"})
    v = _by_key(res)["gamma|AAPL"]
    assert v.verdict == "SKIP_EXCLUDED"
    assert v.reason == "operator_excluded_symbol"
    assert res.close_intents == []          # never proposes a close
    assert not res.would_close


def test_exclusion_precedes_phantom_confirmation():
    # Even with broker truth that WOULD confirm the position, the excluded symbol is skipped
    # before the phantom guard reads it → broker_confirmed_qty stays 0 (operator shares never
    # counted as confirmation).
    state = _guard({"gamma|MSFT": ("MSFT", "BUY", 100, 1)}, {"MSFT": ("BUY", 100)})
    anchors = {"gamma|MSFT": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {"MSFT": 90.0}, anchors, excluded={"MSFT"})
    v = _by_key(res)["gamma|MSFT"]
    assert v.verdict == "SKIP_EXCLUDED"
    assert v.broker_confirmed_qty == pytest.approx(0.0)


def test_non_excluded_symbol_still_evaluates_in_same_batch():
    state = _guard(
        {"gamma|AAPL": ("AAPL", "BUY", 14, 1), "gamma|UNH": ("UNH", "BUY", 100, 1)},
        {"AAPL": ("BUY", 7), "UNH": ("BUY", 100)},
    )
    anchors = {
        "gamma|AAPL": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0, "first_seen_utc": NOW.isoformat()},
        "gamma|UNH": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0, "first_seen_utc": NOW.isoformat()},
    }
    res = _evaluate(state, {"AAPL": 90.0, "UNH": 90.0}, anchors, excluded={"AAPL"})
    by = _by_key(res)
    assert by["gamma|AAPL"].verdict == "SKIP_EXCLUDED"
    assert by["gamma|UNH"].verdict == "WOULD_CLOSE"        # UNH is NOT excluded → still closes
    assert [c["symbol"] for c in res.close_intents] == ["UNH"]


def test_no_exclusion_set_means_no_skips():
    # excluded_symbols=None → pre-B-6 behaviour (AAPL closes). Proves it is strictly additive.
    state = _guard({"gamma|AAPL": ("AAPL", "BUY", 14, 1)}, {"AAPL": ("BUY", 7)})
    anchors = {"gamma|AAPL": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {"AAPL": 90.0}, anchors, excluded=None)
    assert _by_key(res)["gamma|AAPL"].verdict == "WOULD_CLOSE"


# --------------------------------------------------------------------------- #
# runner integration + resolver
# --------------------------------------------------------------------------- #
def test_runner_applies_explicit_exclusion_set(tmp_path):
    state = _guard({"gamma|AAPL": ("AAPL", "BUY", 14, 1)}, {"AAPL": ("BUY", 7)})
    (tmp_path / "state.json").write_text(
        '{"anchors": {"gamma|AAPL": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,'
        ' "first_seen_utc": "%s"}}}' % NOW.isoformat()
    )
    runner = pxo.PositionExitOverlay(
        _cfg("shadow"),
        evidence_path=tmp_path / "evi", state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        guard_loader=lambda: state, open_positions_loader=lambda: _open(state),
        bars_loader=lambda s: {}, price_loader=lambda s: {"AAPL": 90.0},
        env={}, excluded_symbols={"AAPL"},
    )
    res = runner.run_cycle(object(), now_utc=NOW)
    assert _by_key(res)["gamma|AAPL"].verdict == "SKIP_EXCLUDED"


def test_default_resolver_returns_reconciler_single_source_set():
    # The production resolver pulls the reconciler's already-computed set (operator-owned tickers).
    got = pxo._default_excluded_symbols()
    assert isinstance(got, frozenset)
    assert {"AAPL", "SPY", "MSFT"} <= got   # the known operator-owned names


def test_runner_under_pytest_defaults_to_no_exclusions(tmp_path):
    # No explicit set + under pytest → empty exclusions, so a fixture using an operator-owned
    # ticker (BAC) still evaluates. This is what keeps the rest of the overlay suite valid.
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    (tmp_path / "state.json").write_text(
        '{"anchors": {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,'
        ' "first_seen_utc": "%s"}}}' % NOW.isoformat()
    )
    runner = pxo.PositionExitOverlay(
        _cfg("shadow"),
        evidence_path=tmp_path / "evi", state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        guard_loader=lambda: state, open_positions_loader=lambda: _open(state),
        bars_loader=lambda s: {}, price_loader=lambda s: {"BAC": 90.0},
        env={},  # no excluded_symbols → pytest default = empty
    )
    res = runner.run_cycle(object(), now_utc=NOW)
    assert _by_key(res)["gamma|BAC"].verdict == "WOULD_CLOSE"

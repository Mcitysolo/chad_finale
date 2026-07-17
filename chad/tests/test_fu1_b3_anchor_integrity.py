"""FU1/B3 — exit-overlay anchor integrity (FLIP-UNBLOCK 2026-07-17).

ULTRA_CLOSE_AUDIT B-3 proved the exit-overlay anchor is fiction: ``_save_anchors`` replaced the
state file wholesale with only the fully-evaluated keys, so any skip cycle ERASED a still-open
position's anchor and the next cycle re-seeded ``entry = peak = trough = spot``. Two live wipes
were measured, both right after a 23:45 broker false-flat. Compounding it: the guard carries no
cost basis (entry_price structurally unknowable → fell back to spot), ``opened_at`` is rewritten
by paper_ledger_rebuild every 1–2 days (max_hold could never fire), and the ATR trailing stop
LOOSENED as ATR was revised up (un-firing standing closes).

These tests pin the fix on all four axes:
  * merge-never-replace (skip + whole-book false-flat both preserve anchors; real close prunes),
  * FIFO-truth cost basis + opened_at,
  * max_hold survives the guard-rebuild rewrite (x3),
  * monotonic (ratcheting) trailing stop.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from chad.risk import position_exit_overlay as pxo

UTC = timezone.utc
NOW = datetime(2026, 7, 17, 15, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# helpers (mirror test_position_exit_overlay conventions)
# --------------------------------------------------------------------------- #
def _cfg(mode="shadow", **kw):
    payload = {
        "mode": mode, "atr_period": 14, "atr_trail_mult": 2.5, "hard_stop_loss_pct": 0.08,
        "min_bars_for_atr": 16, "max_hold_days": {"equity": 20.0, "etf": 30.0, "default": 20.0},
    }
    payload.update(kw)
    return pxo.load_overlay_config(payload)


def _guard(strategy_entries, broker_entries, opened_days=1):
    state = {"_version": 1}
    for key, (sym, side, qty) in strategy_entries.items():
        state[key] = {
            "open": True, "symbol": sym, "side": side, "quantity": qty,
            "strategy": key.split("|")[0],
            "opened_at": (NOW - timedelta(days=opened_days)).isoformat(),
        }
    for sym, (side, qty) in broker_entries.items():
        state[f"broker_sync|{sym}"] = {
            "open": False, "symbol": sym, "side": side, "quantity": qty, "strategy": "broker_sync",
        }
    return state


def _open_positions(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("_")}


def _flat_bars(n=20, close=100.0, tr=1.0):
    half = tr / 2.0
    return [{"open": close, "high": close + half, "low": close - half, "close": close}
            for _ in range(n)]


def _evaluate(state, bars, prices, anchors, config, fifo=None, now=NOW):
    return pxo.evaluate_positions(
        open_positions=_open_positions(state), guard_state=state, bars_by_symbol=bars,
        price_by_symbol=prices, anchors=anchors, config=config, now_utc=now,
        fifo_truth_by_key=fifo,
    )


def _by_key(result):
    return {v.position_key: v for v in result.verdicts}


def _runner(config, tmp_path, guard_state, bars, prices, fifo_loader=None):
    return pxo.PositionExitOverlay(
        config, evidence_path=tmp_path / "evi", state_path=tmp_path / "state.json",
        guard_loader=lambda: guard_state,
        open_positions_loader=lambda: _open_positions(guard_state),
        bars_loader=lambda syms: bars, price_loader=lambda syms: prices,
        fifo_truth_loader=fifo_loader, env={},
    )


class _FakeAdapter:
    def submit_strategy_trade_intents(self, intents):
        return []


def _saved_anchors(tmp_path):
    return json.loads((tmp_path / "state.json").read_text())["anchors"]


# --------------------------------------------------------------------------- #
# (a) merge-never-replace — the anchor wipe cannot recur
# --------------------------------------------------------------------------- #
def test_skip_cycle_preserves_a_still_open_anchor(tmp_path):
    # Cycle 1: BAC broker-confirmed → anchor persisted with a real peak.
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100)}, {"BAC": ("BUY", 100)})
    runner = _runner(_cfg(), tmp_path, state, {"BAC": _flat_bars()}, {"BAC": 130.0})
    runner.run_cycle(_FakeAdapter(), now_utc=NOW)
    assert _saved_anchors(tmp_path)["gamma|BAC"]["peak"] == 130.0

    # Cycle 2: broker truth vanishes → SKIP_UNCONFIRMED. Pre-B-3 this wiped the anchor.
    state2 = _guard({"gamma|BAC": ("BAC", "BUY", 100)}, {})  # no broker_sync
    runner2 = _runner(_cfg(), tmp_path, state2, {"BAC": _flat_bars()}, {"BAC": 90.0})
    runner2.run_cycle(_FakeAdapter(), now_utc=NOW + timedelta(minutes=1))
    saved = _saved_anchors(tmp_path)
    assert "gamma|BAC" in saved                      # NOT wiped
    assert saved["gamma|BAC"]["peak"] == 130.0        # history intact, not re-seeded at spot 90


def test_whole_book_false_flat_prunes_nothing(tmp_path):
    # Cycle 1: anchor persisted.
    state = _guard({"gamma|UNH": ("UNH", "BUY", 273)}, {"UNH": ("BUY", 273)})
    runner = _runner(_cfg(), tmp_path, state, {"UNH": _flat_bars()}, {"UNH": 430.0})
    runner.run_cycle(_FakeAdapter(), now_utc=NOW)
    assert "gamma|UNH" in _saved_anchors(tmp_path)

    # Cycle 2: the guard read returns an EMPTY book — the exact 23:45 false-flat signature.
    runner2 = _runner(_cfg(), tmp_path, {"_version": 1}, {}, {})
    res = runner2.run_cycle(_FakeAdapter(), now_utc=NOW + timedelta(minutes=1))
    assert res.verdicts == []                         # nothing evaluated (empty book)
    assert "gamma|UNH" in _saved_anchors(tmp_path)     # anchor PRESERVED, not wiped


def test_confirmed_close_prunes_the_anchor(tmp_path):
    # Two positions anchored, then one genuinely leaves the (non-empty) book → pruned.
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100), "gamma|IWM": ("IWM", "BUY", 50)},
                   {"BAC": ("BUY", 100), "IWM": ("BUY", 50)})
    _runner(_cfg(), tmp_path, state, {}, {"BAC": 100.0, "IWM": 100.0}).run_cycle(
        _FakeAdapter(), now_utc=NOW)
    assert {"gamma|BAC", "gamma|IWM"} <= set(_saved_anchors(tmp_path))

    state2 = _guard({"gamma|IWM": ("IWM", "BUY", 50)}, {"IWM": ("BUY", 50)})  # BAC closed
    _runner(_cfg(), tmp_path, state2, {}, {"IWM": 100.0}).run_cycle(
        _FakeAdapter(), now_utc=NOW + timedelta(minutes=1))
    saved = _saved_anchors(tmp_path)
    assert "gamma|BAC" not in saved                   # pruned on confirmed close
    assert "gamma|IWM" in saved


# --------------------------------------------------------------------------- #
# (b) FIFO-truth cost basis + opened_at
# --------------------------------------------------------------------------- #
def test_entry_price_comes_from_fifo_cost_basis_not_spot():
    state = _guard({"gamma|UNH": ("UNH", "BUY", 273)}, {"UNH": ("BUY", 273)})
    fifo = {"gamma|UNH": {"entry_price": 420.71, "opened_at": (NOW - timedelta(days=2)).isoformat()}}
    res = _evaluate(state, {}, {"UNH": 430.0}, {}, _cfg(), fifo=fifo)   # spot 430 != basis 420.71
    v = _by_key(res)["gamma|UNH"]
    assert v.entry_price == 420.71                     # real cost basis, not spot
    assert abs(v.hard_stop_price - 420.71 * 0.92) < 1e-6
    assert res.updated_anchors["gamma|UNH"]["entry_price_source"] == "fifo_cost_basis"


def test_entry_price_falls_back_to_spot_without_fifo():
    state = _guard({"gamma|XYZ": ("XYZ", "BUY", 10)}, {"XYZ": ("BUY", 10)})
    res = _evaluate(state, {}, {"XYZ": 55.0}, {}, _cfg(), fifo=None)
    v = _by_key(res)["gamma|XYZ"]
    assert v.entry_price == 55.0
    assert res.updated_anchors["gamma|XYZ"]["entry_price_source"] == "spot_fallback"


def test_persisted_anchor_entry_wins_over_fifo():
    # A previously-persisted basis is stable and must not be overwritten by a later FIFO read.
    state = _guard({"gamma|UNH": ("UNH", "BUY", 273)}, {"UNH": ("BUY", 273)})
    anchors = {"gamma|UNH": {"entry_price": 425.01, "peak": 460.0, "trough": 425.01,
                             "first_seen_utc": NOW.isoformat()}}
    fifo = {"gamma|UNH": {"entry_price": 420.71}}
    res = _evaluate(state, {}, {"UNH": 430.0}, anchors, _cfg(), fifo=fifo)
    v = _by_key(res)["gamma|UNH"]
    assert v.entry_price == 425.01
    assert res.updated_anchors["gamma|UNH"]["entry_price_source"] == "anchor"


def test_fifo_truth_from_state_weights_and_earliest_ts():
    data = {"queues": [{
        "strategy": "gamma", "symbol": "unh", "lots": [
            {"fill_price": 420.0, "quantity": 100, "ts_utc": "2026-07-15T10:00:00+00:00"},
            {"fill_price": 430.0, "quantity": 200, "ts_utc": "2026-07-15T09:00:00+00:00"},
        ],
    }]}
    out = pxo._fifo_truth_from_state(data)
    assert "gamma|UNH" in out
    assert abs(out["gamma|UNH"]["entry_price"] - (420 * 100 + 430 * 200) / 300.0) < 1e-6
    assert out["gamma|UNH"]["opened_at"] == "2026-07-15T09:00:00+00:00"   # earliest lot


def test_fifo_truth_from_state_skips_lots_without_usable_price():
    data = {"queues": [{"strategy": "gamma", "symbol": "V", "lots": [
        {"fill_price": 0.0, "quantity": 100, "ts_utc": "x"}]}]}
    assert pxo._fifo_truth_from_state(data) == {}


# --------------------------------------------------------------------------- #
# (c) max_hold survives paper_ledger_rebuild rewriting opened_at (x3)
# --------------------------------------------------------------------------- #
def test_max_hold_survives_guard_opened_at_rewrite_x3():
    # FIFO says the lot opened 40d ago; the guard's opened_at is rewritten to "now" every cycle
    # (the rebuild cadence). Age must stay ~40 and max_hold must fire on all three cycles.
    fifo = {"gamma|NVDA": {"entry_price": 100.0,
                           "opened_at": (NOW - timedelta(days=40)).isoformat()}}
    anchors = {}
    for i in range(3):
        # guard opened_at freshly rewritten to now — deliberately hostile.
        state = _guard({"gamma|NVDA": ("NVDA", "BUY", 5)}, {"NVDA": ("BUY", 5)}, opened_days=0)
        res = _evaluate(state, {}, {"NVDA": 100.0}, anchors, _cfg(), fifo=fifo,
                        now=NOW + timedelta(minutes=i))
        v = _by_key(res)["gamma|NVDA"]
        assert v.age_days is not None and v.age_days >= 20, f"cycle {i} age={v.age_days}"
        assert v.verdict == "WOULD_CLOSE" and v.reason == "max_hold", f"cycle {i}"
        anchors = res.updated_anchors   # carry persisted state forward like the runner does


def test_opened_at_persists_even_after_fifo_disappears():
    # Once the anchor persists opened_at, a later cycle with NO fifo still ages from it.
    fifo = {"gamma|NVDA": {"entry_price": 100.0,
                           "opened_at": (NOW - timedelta(days=40)).isoformat()}}
    state = _guard({"gamma|NVDA": ("NVDA", "BUY", 5)}, {"NVDA": ("BUY", 5)}, opened_days=0)
    res1 = _evaluate(state, {}, {"NVDA": 100.0}, {}, _cfg(), fifo=fifo)
    res2 = _evaluate(state, {}, {"NVDA": 100.0}, res1.updated_anchors, _cfg(), fifo=None)
    v = _by_key(res2)["gamma|NVDA"]
    assert v.age_days is not None and v.age_days >= 20
    assert v.reason == "max_hold"


# --------------------------------------------------------------------------- #
# (d) monotonic (ratcheting) trailing stop
# --------------------------------------------------------------------------- #
def test_atr_trailing_stop_ratchets_and_never_loosens():
    state = _guard({"gamma|MSFT": ("MSFT", "BUY", 10)}, {"MSFT": ("BUY", 10)})
    # Cycle 1: ATR=1 → raw stop = 105 - 2.5*1 = 102.5. Price 103 holds; ratchet persists 102.5.
    a1 = {"gamma|MSFT": {"entry_price": 100.0, "peak": 105.0, "trough": 100.0,
                         "first_seen_utc": NOW.isoformat()}}
    res1 = _evaluate(state, {"MSFT": _flat_bars(tr=1.0)}, {"MSFT": 103.0}, a1, _cfg())
    v1 = _by_key(res1)["gamma|MSFT"]
    assert v1.verdict == "HOLD"
    assert abs(res1.updated_anchors["gamma|MSFT"]["atr_stop_ratchet"] - 102.5) < 0.05

    # Cycle 2: ATR widens to 2 → raw stop drops to 105 - 2.5*2 = 100 (LOOSER). The ratchet holds
    # at 102.5, so price 101 — which the loosened raw stop would HOLD — fires the trail.
    res2 = _evaluate(state, {"MSFT": _flat_bars(tr=2.0)}, {"MSFT": 101.0},
                     res1.updated_anchors, _cfg())
    v2 = _by_key(res2)["gamma|MSFT"]
    assert v2.verdict == "WOULD_CLOSE" and v2.reason == "atr_trailing_stop"
    assert abs(v2.atr_stop - 102.5) < 0.05             # the stop did NOT loosen to 100


def test_short_atr_trailing_stop_ratchets_down():
    state = _guard({"gr|TSLA": ("TSLA", "SELL", 20)}, {"TSLA": ("SELL", 20)})
    # short: raw stop = trough + 2.5*atr. Cycle 1 trough=95, atr=1 → stop 97.5, price 96 holds.
    a1 = {"gr|TSLA": {"entry_price": 100.0, "peak": 100.0, "trough": 95.0,
                      "first_seen_utc": NOW.isoformat()}}
    res1 = _evaluate(state, {"TSLA": _flat_bars(tr=1.0)}, {"TSLA": 96.0}, a1, _cfg())
    assert _by_key(res1)["gr|TSLA"].verdict == "HOLD"
    assert abs(res1.updated_anchors["gr|TSLA"]["atr_stop_ratchet"] - 97.5) < 0.05
    # Cycle 2 atr widens to 2 → raw 95 + 5 = 100 (LOOSER for a short). Ratchet holds 97.5;
    # price 98 fires against the tighter ratchet.
    res2 = _evaluate(state, {"TSLA": _flat_bars(tr=2.0)}, {"TSLA": 98.0}, res1.updated_anchors, _cfg())
    v2 = _by_key(res2)["gr|TSLA"]
    assert v2.verdict == "WOULD_CLOSE" and v2.reason == "atr_trailing_stop"
    assert abs(v2.atr_stop - 97.5) < 0.05


def test_ratchet_holds_through_transient_atr_gap():
    # A momentary ATR outage (too few bars) must not drop the persisted ratchet to None.
    state = _guard({"gamma|MSFT": ("MSFT", "BUY", 10)}, {"MSFT": ("BUY", 10)})
    a = {"gamma|MSFT": {"entry_price": 100.0, "peak": 105.0, "trough": 100.0,
                        "first_seen_utc": NOW.isoformat(), "atr_stop_ratchet": 102.5}}
    # min_bars_for_atr=16 but only 3 bars → atr is None this cycle.
    res = _evaluate(state, {"MSFT": _flat_bars(n=3, tr=1.0)}, {"MSFT": 101.0}, a, _cfg())
    v = _by_key(res)["gamma|MSFT"]
    assert v.atr is None
    assert v.verdict == "WOULD_CLOSE" and v.reason == "atr_trailing_stop"   # ratchet still fired
    assert abs(v.atr_stop - 102.5) < 0.05

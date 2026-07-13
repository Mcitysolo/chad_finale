"""U1 tests — chad/risk/position_exit_overlay.py core + config + shadow evidence.

Covers: each exit condition fires on synthetic positions/bars; HOLD when none met; phantom
(unconfirmed) skip; non-equity skip; reduce-only clamp; short mirror; no-data skip; anchor
seed/update/prune; config validation + fail-open; env kill-switch; shadow evidence written +
nothing submitted; OFF inert; error fail-open-in-shadow.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.risk import position_exit_overlay as pxo

UTC = timezone.utc
NOW = datetime(2026, 7, 13, 15, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg(mode="shadow", **kw):
    payload = {
        "mode": mode,
        "atr_period": 14,
        "atr_trail_mult": 2.5,
        "hard_stop_loss_pct": 0.08,
        "min_bars_for_atr": 16,
        "max_hold_days": {"equity": 20.0, "etf": 30.0, "default": 20.0},
    }
    payload.update(kw)
    return pxo.load_overlay_config(payload)


def _guard(strategy_entries, broker_entries):
    """strategy_entries: {key: (symbol, side, qty, opened_days_ago)}
    broker_entries: {symbol: (side, qty)}  -> broker_sync|symbol"""
    state = {"_version": 1, "_written_by": 1}
    for key, (sym, side, qty, days) in strategy_entries.items():
        state[key] = {
            "open": True, "symbol": sym, "side": side, "quantity": qty,
            "strategy": key.split("|")[0],
            "opened_at": (NOW - timedelta(days=days)).isoformat(),
        }
    for sym, (side, qty) in broker_entries.items():
        state[f"broker_sync|{sym}"] = {
            "open": False, "symbol": sym, "side": side, "quantity": qty,
            "strategy": "broker_sync",
        }
    return state


def _open_positions(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("_")}


def _flat_bars(n=20, close=100.0, tr=1.0):
    """n bars with a fixed ~tr true range and flat closes -> ATR ~= tr."""
    half = tr / 2.0
    return [{"open": close, "high": close + half, "low": close - half, "close": close}
            for _ in range(n)]


def _evaluate(state, bars, prices, anchors, config):
    return pxo.evaluate_positions(
        open_positions=_open_positions(state),
        guard_state=state,
        bars_by_symbol=bars,
        price_by_symbol=prices,
        anchors=anchors,
        config=config,
        now_utc=NOW,
    )


def _by_key(result):
    return {v.position_key: v for v in result.verdicts}


# --------------------------------------------------------------------------- #
# exit conditions
# --------------------------------------------------------------------------- #
def test_hard_stop_fires_long():
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    anchors = {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                             "first_seen_utc": NOW.isoformat()}}
    # price 90 < entry*(1-0.08)=92 -> hard stop
    res = _evaluate(state, {}, {"BAC": 90.0}, anchors, _cfg())
    v = _by_key(res)["gamma|BAC"]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "hard_stop_loss"
    assert v.close_qty == 100
    assert res.close_intents[0]["close_side"] == "SELL"
    assert res.close_intents[0]["reason"] == "exit_overlay_hard_stop_loss"


def test_atr_trailing_fires_long():
    state = _guard({"gamma|MSFT": ("MSFT", "BUY", 10, 1)}, {"MSFT": ("BUY", 10)})
    # ATR ~= 1.0; peak 105 -> atr_stop = 105 - 2.5*1 = 102.5; price 100 <= 102.5 -> fire.
    # entry 100 -> hard_stop 92, price 100 > 92 so hard does NOT fire first.
    anchors = {"gamma|MSFT": {"entry_price": 100.0, "peak": 105.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {"MSFT": _flat_bars()}, {"MSFT": 100.0}, anchors, _cfg())
    v = _by_key(res)["gamma|MSFT"]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "atr_trailing_stop"
    assert v.atr is not None and abs(v.atr - 1.0) < 0.05
    assert v.atr_stop is not None and abs(v.atr_stop - 102.5) < 0.2


def test_max_hold_fires_long():
    state = _guard({"alpha|NVDA": ("NVDA", "BUY", 5, 40)}, {"NVDA": ("BUY", 5)})
    # opened 40d ago, equity max_hold=20; price at entry so hard/atr don't fire; no bars -> atr None
    anchors = {"alpha|NVDA": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": (NOW - timedelta(days=40)).isoformat()}}
    res = _evaluate(state, {}, {"NVDA": 100.0}, anchors, _cfg())
    v = _by_key(res)["alpha|NVDA"]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "max_hold"
    assert v.age_days is not None and v.age_days >= 20


def test_hold_when_no_condition_met():
    state = _guard({"gamma|AAPL": ("AAPL", "BUY", 10, 1)}, {"AAPL": ("BUY", 10)})
    anchors = {"gamma|AAPL": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                              "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {"AAPL": _flat_bars()}, {"AAPL": 100.0}, anchors, _cfg())
    v = _by_key(res)["gamma|AAPL"]
    assert v.verdict == "HOLD" and v.reason == "no_condition_met"
    assert res.close_intents == []


def test_short_position_hard_stop_mirror():
    state = _guard({"gamma_reversion|TSLA": ("TSLA", "SELL", 20, 1)}, {"TSLA": ("SELL", 20)})
    # short: hard_stop = entry*(1.08)=108; price 110 >= 108 -> fire; close_side BUY
    anchors = {"gamma_reversion|TSLA": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                                        "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {}, {"TSLA": 110.0}, anchors, _cfg())
    v = _by_key(res)["gamma_reversion|TSLA"]
    assert v.verdict == "WOULD_CLOSE" and v.reason == "hard_stop_loss"
    assert res.close_intents[0]["close_side"] == "BUY"


# --------------------------------------------------------------------------- #
# phantom / scope / data guards
# --------------------------------------------------------------------------- #
def test_phantom_skip_no_broker_confirmation():
    # guard says gamma holds BAC but broker_sync holds nothing -> Fault-C phantom.
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {})
    res = _evaluate(state, {}, {"BAC": 90.0}, {}, _cfg())
    v = _by_key(res)["gamma|BAC"]
    assert v.verdict == "SKIP_UNCONFIRMED"
    assert res.close_intents == []


def test_phantom_skip_opposite_side_broker():
    # broker holds SELL but guard says BUY -> not same-side confirmed -> skip.
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("SELL", 100)})
    res = _evaluate(state, {}, {"BAC": 90.0}, {}, _cfg())
    assert _by_key(res)["gamma|BAC"].verdict == "SKIP_UNCONFIRMED"


def test_non_equity_futures_skipped():
    state = _guard({"omega_macro|M6E": ("M6E", "SELL", 3, 1)}, {"M6E": ("SELL", 3)})
    res = _evaluate(state, {}, {"M6E": 1.05}, {}, _cfg())
    v = _by_key(res)["omega_macro|M6E"]
    assert v.verdict == "SKIP_NON_EQUITY"
    assert res.close_intents == []


def test_no_price_no_bars_skips_no_data():
    state = _guard({"gamma|XYZ": ("XYZ", "BUY", 10, 1)}, {"XYZ": ("BUY", 10)})
    res = _evaluate(state, {}, {}, {}, _cfg())
    assert _by_key(res)["gamma|XYZ"].verdict == "SKIP_NO_DATA"


def test_price_falls_back_to_last_bar_close():
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    anchors = {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                             "first_seen_utc": NOW.isoformat()}}
    bars = _flat_bars(close=90.0)  # last close 90 < hard_stop 92
    res = _evaluate(state, {"BAC": bars}, {}, anchors, _cfg())
    v = _by_key(res)["gamma|BAC"]
    assert v.price == 90.0 and v.verdict == "WOULD_CLOSE" and v.reason == "hard_stop_loss"


# --------------------------------------------------------------------------- #
# reduce-only invariant
# --------------------------------------------------------------------------- #
def test_reduce_only_clamps_to_broker_held():
    # guard says 100 but broker only confirms 30 -> close_qty clamps to 30.
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 30)})
    anchors = {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                             "first_seen_utc": NOW.isoformat()}}
    res = _evaluate(state, {}, {"BAC": 90.0}, anchors, _cfg())
    v = _by_key(res)["gamma|BAC"]
    assert v.verdict == "WOULD_CLOSE"
    assert v.close_qty == 30.0
    assert res.close_intents[0]["quantity"] == 30.0


def test_reduce_only_reclamp_drops_phantom_at_submit():
    intents = [{"symbol": "BAC", "open_side": "BUY", "close_side": "SELL",
                "quantity": 100.0, "position_key": "gamma|BAC", "strategy": "gamma"}]
    # fresh guard now shows no broker truth -> dropped.
    fresh = _guard({}, {})
    assert pxo._reduce_only_reclamp(intents, fresh) == []
    # broker confirms 40 -> clamp to 40.
    fresh2 = _guard({}, {"BAC": ("BUY", 40)})
    out = pxo._reduce_only_reclamp(intents, fresh2)
    assert len(out) == 1 and out[0]["quantity"] == 40.0


# --------------------------------------------------------------------------- #
# anchors: seed / update / prune
# --------------------------------------------------------------------------- #
def test_anchor_seed_update_and_prune():
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    # prior anchors carry a stale key no longer open + a BAC anchor with lower peak
    anchors = {
        "gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                      "first_seen_utc": NOW.isoformat()},
        "gamma|GONE": {"entry_price": 50.0, "peak": 55.0, "trough": 50.0,
                       "first_seen_utc": NOW.isoformat()},
    }
    res = _evaluate(state, {"BAC": _flat_bars()}, {"BAC": 130.0}, anchors, _cfg())
    ua = res.updated_anchors
    assert "gamma|GONE" not in ua                 # stale key pruned
    assert ua["gamma|BAC"]["peak"] == 130.0        # peak trailed up to new high
    assert ua["gamma|BAC"]["entry_price"] == 100.0  # entry preserved


# --------------------------------------------------------------------------- #
# config validation + kill switch
# --------------------------------------------------------------------------- #
def test_config_rejects_bad_mode():
    with pytest.raises(pxo.ExitOverlayConfigError):
        _cfg(mode="enforce")


def test_config_rejects_hard_stop_ge_one():
    with pytest.raises(pxo.ExitOverlayConfigError):
        _cfg(hard_stop_loss_pct=1.5)


def test_config_rejects_missing_max_hold():
    with pytest.raises(pxo.ExitOverlayConfigError):
        pxo.load_overlay_config({
            "mode": "shadow", "atr_period": 14, "atr_trail_mult": 2.5,
            "hard_stop_loss_pct": 0.08, "min_bars_for_atr": 16,
        })


def test_resolve_mode_env_killswitch():
    assert pxo.resolve_mode("shadow", {"CHAD_POSITION_EXIT_OVERLAY": "off"}) == "off"
    assert pxo.resolve_mode("shadow", {"CHAD_POSITION_EXIT_OVERLAY": "active"}) == "active"
    assert pxo.resolve_mode("shadow", {"CHAD_POSITION_EXIT_OVERLAY": "garbage"}) == "shadow"
    assert pxo.resolve_mode("shadow", {}) == "shadow"


def test_shipped_config_parses_and_is_shadow():
    repo_root = Path(__file__).resolve().parents[2]
    payload = json.loads((repo_root / "config" / "position_exit_overlay.json").read_text())
    cfg = pxo.load_overlay_config(payload)
    assert cfg.mode == "shadow"  # ships default-safe
    assert cfg.atr_period == 14 and cfg.max_hold_for("etf") == 30.0


# --------------------------------------------------------------------------- #
# runner: shadow evidence + OFF inert + error fail-open
# --------------------------------------------------------------------------- #
class _FakeAdapter:
    def __init__(self):
        self.calls = []

    def submit_strategy_trade_intents(self, intents):
        self.calls.append(list(intents))
        return []


def _runner(config, tmp_path, guard_state, bars, prices, anchors_seed=None):
    state_path = tmp_path / "state.json"
    if anchors_seed is not None:
        state_path.write_text(json.dumps({"anchors": anchors_seed}))
    return pxo.PositionExitOverlay(
        config,
        evidence_path=tmp_path / "evi",
        state_path=state_path,
        guard_loader=lambda: guard_state,
        open_positions_loader=lambda: _open_positions(guard_state),
        bars_loader=lambda syms: bars,
        price_loader=lambda syms: prices,
        env={},
    )


def test_runner_shadow_writes_evidence_and_submits_nothing(tmp_path):
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    anchors = {"gamma|BAC": {"entry_price": 100.0, "peak": 100.0, "trough": 100.0,
                             "first_seen_utc": NOW.isoformat()}}
    runner = _runner(_cfg("shadow"), tmp_path, state, {}, {"BAC": 90.0}, anchors_seed=anchors)
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW)
    assert res.would_close and res.would_close[0].reason == "hard_stop_loss"
    assert adapter.calls == []  # SHADOW closes nothing
    ev = list((tmp_path / "evi").glob("exit_overlay_*.ndjson"))
    assert ev, "shadow evidence ndjson written"
    row = json.loads(ev[0].read_text().splitlines()[0])
    assert row["schema_version"] == "exit_overlay.v1" and row["verdict"] == "WOULD_CLOSE"
    # anchors persisted
    saved = json.loads((tmp_path / "state.json").read_text())
    assert "gamma|BAC" in saved["anchors"]


def test_runner_off_is_inert(tmp_path):
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    runner = _runner(_cfg("off"), tmp_path, state, {}, {"BAC": 90.0})
    adapter = _FakeAdapter()
    res = runner.run_cycle(adapter, now_utc=NOW)
    assert res.evaluated is False
    assert adapter.calls == []
    assert not list((tmp_path / "evi").glob("*.ndjson"))


def test_runner_error_fail_open_in_shadow(tmp_path, caplog):
    def _boom():
        raise RuntimeError("guard read exploded")

    runner = pxo.PositionExitOverlay(
        _cfg("shadow"),
        evidence_path=tmp_path / "evi",
        state_path=tmp_path / "state.json",
        guard_loader=_boom,
        open_positions_loader=lambda: {},
        bars_loader=lambda syms: {},
        price_loader=lambda syms: {},
        env={},
    )
    adapter = _FakeAdapter()
    with caplog.at_level(logging.ERROR):
        res = runner.run_cycle(adapter, now_utc=NOW)
    assert res.evaluated is False
    assert adapter.calls == []  # never submits on error
    assert any(pxo.MARKER_ERROR in r.message or pxo.MARKER_ERROR in str(r.args) for r in caplog.records)


def test_build_default_overlay_fail_open_on_bad_config(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"mode": "shadow"}')  # missing required numeric keys
    ov = pxo.build_default_overlay(
        repo_root=tmp_path, config_path=bad,
        evidence_dir=tmp_path / "evi", state_path=tmp_path / "s.json",
    )
    assert ov is None  # fail-open: overlay not wired


def test_build_default_overlay_pytest_leak_guard(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "mode": "shadow", "atr_period": 14, "atr_trail_mult": 2.5,
        "hard_stop_loss_pct": 0.08, "min_bars_for_atr": 16,
        "max_hold_days": {"default": 20.0},
    }))
    # under pytest, omitting evidence_dir must raise the leak guard (never write the real tree)
    with pytest.raises(RuntimeError):
        pxo.build_default_overlay(repo_root=tmp_path, config_path=good, state_path=tmp_path / "s.json")

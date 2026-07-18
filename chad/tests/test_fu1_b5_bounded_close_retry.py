"""
FU1-B5 (FLIP-UNBLOCK 2026-07-17): bounded close-retry governor for the exit overlay.

ULTRA_CLOSE_AUDIT §B-5: a standing WOULD_CLOSE re-proposes the SAME close every ~72s cycle, and
the adapter idempotency store has two holes (a rejected row re-INSERTs → retry every cycle
forever; a filled key reclaims after 900s → another close every 15 min). The overlay had no
feedback channel and no cooldown, so a stuck close storms the broker unbounded.

These tests pin the governor:

  * the pure ``_submit_gate`` / backoff math (submit → exponential backoff → stand-down);
  * a rapid 72s-cycle storm collapses to a handful of submits, not one per cycle;
  * after ``SUBMIT_MAX_ATTEMPTS`` the key STANDS DOWN — no further submits ever;
  * the coach alert (loud marker + ``exit_overlay_stand_down.json``) fires exactly once;
  * a position that closes prunes its ledger record (a later close starts fresh);
  * a whole-book false-flat prunes NOTHING (parity with B-3's anchor merge).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from chad.risk import position_exit_overlay as pxo

UTC = timezone.utc
NOW = datetime(2026, 7, 17, 15, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# helpers (mirror test_position_exit_overlay)
# --------------------------------------------------------------------------- #
def _cfg(mode="active", **kw):
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
    state = {"_version": 1, "_written_by": 1}
    for key, (sym, side, qty, days) in strategy_entries.items():
        state[key] = {
            "open": True, "symbol": sym, "side": side, "quantity": qty,
            "strategy": key.split("|")[0],
            "opened_at": (NOW - timedelta(days=days)).isoformat(),
        }
    for sym, (side, qty) in broker_entries.items():
        state[f"broker_sync|{sym}"] = {
            "open": True, "symbol": sym, "side": side, "quantity": qty,
            "strategy": "broker_sync",
        }
    return state


def _open_positions(state):
    return {k: v for k, v in state.items()
            if isinstance(v, dict) and v.get("open") and not k.startswith("_")}


class _Recorder:
    """Stand-in for position_reconciler.apply_close_intents — records every submitted batch."""
    def __init__(self):
        self.batches = []

    def __call__(self, intents, adapter):
        self.batches.append([dict(i) for i in intents])

    @property
    def total_intents(self):
        return sum(len(b) for b in self.batches)


def _active_runner(tmp_path, guard_state):
    return pxo.PositionExitOverlay(
        _cfg("active"),
        evidence_path=tmp_path / "evi",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        guard_loader=lambda: guard_state,
        open_positions_loader=lambda: _open_positions(guard_state),
        bars_loader=lambda syms: {},
        price_loader=lambda syms: {"BAC": 90.0},  # 90 < 100*(1-0.08)=92 → hard-stop fires
        env={},
    )


def _seed_anchor(tmp_path, key="gamma|BAC", entry=100.0):
    (tmp_path / "state.json").write_text(json.dumps({
        "anchors": {key: {"entry_price": entry, "peak": entry, "trough": entry,
                          "first_seen_utc": NOW.isoformat()}}
    }))


def _ledger(tmp_path):
    p = tmp_path / "state_submit_ledger.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text()).get("ledger", {})


# --------------------------------------------------------------------------- #
# pure governor math
# --------------------------------------------------------------------------- #
def test_backoff_schedule_is_exponential_and_capped():
    assert pxo._submit_backoff_required_seconds(0) == 0.0          # first submit never gated
    assert pxo._submit_backoff_required_seconds(1) == pytest.approx(pxo.SUBMIT_BACKOFF_BASE_SECONDS)
    assert pxo._submit_backoff_required_seconds(2) == pytest.approx(2 * pxo.SUBMIT_BACKOFF_BASE_SECONDS)
    assert pxo._submit_backoff_required_seconds(3) == pytest.approx(4 * pxo.SUBMIT_BACKOFF_BASE_SECONDS)
    # capped
    assert pxo._submit_backoff_required_seconds(99) == pytest.approx(pxo.SUBMIT_BACKOFF_MAX_SECONDS)


def test_gate_first_sight_submits():
    decision, rec = pxo._submit_gate(None, NOW)
    assert decision == "submit"
    assert rec["attempts"] == 1
    assert rec["last_utc"] == pxo._iso(NOW)
    assert rec["stood_down"] is False


def test_gate_backoff_blocks_until_interval_elapses():
    rec0 = {"attempts": 1, "first_utc": pxo._iso(NOW), "last_utc": pxo._iso(NOW),
            "stood_down": False, "alerted": False}
    # 72s later — inside the 300s cooldown → backoff, no new attempt.
    d, rec = pxo._submit_gate(rec0, NOW + timedelta(seconds=72))
    assert d == "backoff"
    assert rec["attempts"] == 1
    # 301s later — cooldown elapsed → submit, attempt increments.
    d, rec = pxo._submit_gate(rec0, NOW + timedelta(seconds=301))
    assert d == "submit"
    assert rec["attempts"] == 2


def test_gate_stands_down_at_ceiling():
    rec = {"attempts": pxo.SUBMIT_MAX_ATTEMPTS, "first_utc": pxo._iso(NOW),
           "last_utc": pxo._iso(NOW), "stood_down": False, "alerted": False}
    d, out = pxo._submit_gate(rec, NOW + timedelta(hours=5))
    assert d == "stand_down"
    assert out["stood_down"] is True
    assert out["stood_down_utc"]


def test_gate_stood_down_is_terminal():
    rec = {"attempts": 2, "stood_down": True, "alerted": True,
           "first_utc": pxo._iso(NOW), "last_utc": pxo._iso(NOW)}
    d, out = pxo._submit_gate(rec, NOW + timedelta(days=1))
    assert d == "stand_down"
    assert out["stood_down"] is True


# --------------------------------------------------------------------------- #
# integrated run_cycle — the storm is bounded
# --------------------------------------------------------------------------- #
def test_rapid_72s_storm_collapses_to_a_few_submits(tmp_path, monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("chad.core.position_reconciler.apply_close_intents", rec)
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    _seed_anchor(tmp_path)
    runner = _active_runner(tmp_path, state)

    # 20 cycles, 72s apart (~24 minutes of wall-clock).
    for i in range(20):
        runner.run_cycle(object(), now_utc=NOW + timedelta(seconds=72 * i))

    # Without a governor this is 20 submits. With 300s base backoff over ~24 min it is a
    # small handful — strictly far below the cycle count, and never zero (the first fires).
    assert 1 <= len(rec.batches) < 20
    assert len(rec.batches) <= 6


def test_stands_down_after_max_attempts_and_alerts_once(tmp_path, monkeypatch, caplog):
    rec = _Recorder()
    monkeypatch.setattr("chad.core.position_reconciler.apply_close_intents", rec)
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    _seed_anchor(tmp_path)
    runner = _active_runner(tmp_path, state)

    # Advance 2h/cycle so every backoff gate is satisfied → one submit per cycle until ceiling.
    with caplog.at_level(logging.ERROR):
        for i in range(8):
            runner.run_cycle(object(), now_utc=NOW + timedelta(hours=2 * i))

    # Exactly SUBMIT_MAX_ATTEMPTS real submits, then permanent stand-down.
    assert len(rec.batches) == pxo.SUBMIT_MAX_ATTEMPTS

    led = _ledger(tmp_path)["gamma|BAC"]
    assert led["stood_down"] is True and led["alerted"] is True

    # The coach alert marker fires exactly once (deduped by the `alerted` flag).
    stand_down_logs = [r for r in caplog.records if pxo.MARKER_STAND_DOWN in r.getMessage()]
    assert len(stand_down_logs) == 1

    # ...and the stand-down alert file is published for the alert spine.
    alert = json.loads((tmp_path / "exit_overlay_stand_down.json").read_text())
    assert alert["schema_version"] == pxo.STAND_DOWN_SCHEMA_VERSION
    assert alert["stood_down"][0]["position_key"] == "gamma|BAC"


def test_ledger_record_pruned_when_position_closes(tmp_path, monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("chad.core.position_reconciler.apply_close_intents", rec)
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    _seed_anchor(tmp_path)
    runner = _active_runner(tmp_path, state)

    runner.run_cycle(object(), now_utc=NOW)
    assert "gamma|BAC" in _ledger(tmp_path)

    # Position closes: guard entry gone, but broker mirror + another live position remain so the
    # book is NON-empty (a real close, not a false-flat).
    state.pop("gamma|BAC")
    state["gamma|XLF"] = {"open": True, "symbol": "XLF", "side": "BUY", "quantity": 10,
                          "strategy": "gamma", "opened_at": (NOW - timedelta(days=1)).isoformat()}
    runner.run_cycle(object(), now_utc=NOW + timedelta(seconds=72))
    assert "gamma|BAC" not in _ledger(tmp_path)  # stale record pruned


def test_false_flat_does_not_prune_ledger(tmp_path, monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr("chad.core.position_reconciler.apply_close_intents", rec)
    state = _guard({"gamma|BAC": ("BAC", "BUY", 100, 1)}, {"BAC": ("BUY", 100)})
    _seed_anchor(tmp_path)
    runner = _active_runner(tmp_path, state)

    runner.run_cycle(object(), now_utc=NOW)
    assert "gamma|BAC" in _ledger(tmp_path)

    # Whole-book false-flat: the guard read returns an empty book (no verdicts at all). The
    # governor must NOT prune — the exact empty-book signature B-3 refuses to act on.
    empty = {"_version": 1}
    flat_runner = pxo.PositionExitOverlay(
        _cfg("active"),
        evidence_path=tmp_path / "evi",
        state_path=tmp_path / "state.json",
        heartbeat_path=tmp_path / "hb.json",
        guard_loader=lambda: empty,
        open_positions_loader=lambda: {},
        bars_loader=lambda syms: {},
        price_loader=lambda syms: {},
        env={},
    )
    flat_runner.run_cycle(object(), now_utc=NOW + timedelta(seconds=72))
    assert "gamma|BAC" in _ledger(tmp_path)  # survived the false-flat

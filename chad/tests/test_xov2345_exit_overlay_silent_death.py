"""XOV-2345 — the exit overlay must never die silently.

Forensic summary of the outage these tests pin:

  ib_async's ``ib.positions()`` is a pure read of ``wrapper.positions``, a LOCAL
  cache emptied by ``wrapper.reset()`` on any socket drop and repopulated ONLY by
  the ``reqPositions`` subscription issued inside ``connectAsync``. live_loop's
  shared ``ib`` is connected exactly once at module import and has no reconnect,
  so when the IB API connection dropped at 23:45 UTC the cache went empty and
  stayed empty. ``fetch_positions()`` returned {} — byte-identical to a flat
  broker — and ``_rebuild_guard_from_broker`` read that as "broker holds nothing",
  false-flatting EVERY guard entry to open=False. ``load_open_positions()`` then
  returned {}, so the overlay evaluated zero positions: no verdicts, no evidence,
  no EXIT_OVERLAY_ERROR. It watched nothing for 16h and only a process restart
  (which re-runs the module-level connect) ever revived it.

Three defences, one per failure link:
  1. fetch_positions fails CLOSED — unreadable broker is UNKNOWN, never flat.
  2. the guard sweep is skipped (state preserved) + alerts when truth is missing.
  3. the overlay heartbeats every cycle, and R14 alerts on stale OR blind.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from chad.core import live_loop
from chad.core.broker_position_sync import (
    BrokerPositionSync,
    BrokerTruthUnavailable,
)
from chad.ops import health_monitor_rules as hmr
from chad.risk import position_exit_overlay as pxo

NOW = datetime(2026, 7, 14, 23, 44, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# 1. fetch_positions fails closed
# --------------------------------------------------------------------------- #
class _Contract:
    def __init__(self, symbol):
        self.symbol = symbol
        self.localSymbol = symbol
        self.secType = "STK"


class _IbPos:
    def __init__(self, symbol, qty):
        self.contract = _Contract(symbol)
        self.position = qty
        self.avgCost = 10.0


class _FakeIb:
    """Mirrors the ib_async contract: positions() reads a cache; reset empties it."""

    def __init__(self, connected=True, positions=()):
        self._connected = connected
        self._positions = list(positions)

    def isConnected(self):
        return self._connected

    def positions(self):
        return list(self._positions)

    def drop_connection(self):
        # exactly what wrapper.reset() does on a socket drop
        self._connected = False
        self._positions = []


def test_fetch_positions_returns_truth_when_connected():
    sync = BrokerPositionSync(_FakeIb(True, [_IbPos("UNH", 261)]))
    out = sync.fetch_positions()
    assert out["UNH"].quantity == 261


def test_fetch_positions_raises_when_connection_down_not_empty_dict():
    """The regression: a dead connection must NOT look like a flat broker."""
    ib = _FakeIb(True, [_IbPos("UNH", 261)])
    sync = BrokerPositionSync(ib)
    assert sync.fetch_positions()["UNH"].quantity == 261

    ib.drop_connection()  # 23:45 UTC
    with pytest.raises(BrokerTruthUnavailable):
        sync.fetch_positions()


def test_fetch_positions_empty_while_connected_is_a_real_flat_broker():
    """Fail-closed must not over-trigger: connected + empty is genuinely flat."""
    assert BrokerPositionSync(_FakeIb(True, [])).fetch_positions() == {}


# --------------------------------------------------------------------------- #
# 2. the guard sweep must not false-flat on unavailable truth
# --------------------------------------------------------------------------- #
@pytest.fixture
def guard_env(monkeypatch):
    state = {
        "gamma|UNH": {
            "open": True, "symbol": "UNH", "side": "BUY", "quantity": 261.0,
            "strategy": "gamma",
        },
    }
    monkeypatch.setattr(live_loop, "_load_state", lambda: dict(state), raising=False)
    from chad.core import position_guard
    monkeypatch.setattr(position_guard, "_load_state", lambda: json.loads(json.dumps(state)))

    def _save(s):
        state.clear()
        state.update(json.loads(json.dumps(s)))

    monkeypatch.setattr(position_guard, "save_state", _save)
    monkeypatch.setattr(live_loop, "_skip_ib_connect", lambda: False)
    return state


def test_guard_sweep_skipped_when_broker_truth_unavailable(guard_env, monkeypatch, caplog):
    """THE fix: an unreadable broker must leave the position book untouched."""
    sent = []
    monkeypatch.setattr(
        live_loop.position_sync, "fetch_positions",
        lambda: (_ for _ in ()).throw(BrokerTruthUnavailable("api down")),
    )
    monkeypatch.setattr(
        live_loop, "_fire_alert_safe",
        lambda kind, fn, *a, **k: sent.append(kind) or True,
    )

    with caplog.at_level(logging.ERROR):
        live_loop._rebuild_guard_from_broker(logging.getLogger("test.xov"))

    assert guard_env["gamma|UNH"]["open"] is True, "false-flatted on a dead connection"
    assert "BROKER_TRUTH_UNAVAILABLE" in caplog.text
    assert sent == ["broker_truth_unavailable"], "silent degradation"


def test_guard_sweep_still_closes_on_a_genuinely_flat_broker(guard_env, monkeypatch):
    """The fail-closed guard must not disable real broker-truth reconciliation."""
    monkeypatch.setattr(live_loop.position_sync, "fetch_positions", lambda: {})
    live_loop._rebuild_guard_from_broker(logging.getLogger("test.xov"))
    assert guard_env["gamma|UNH"]["open"] is False
    assert guard_env["gamma|UNH"]["closed_by"] == "broker_truth_rebuild"


# --------------------------------------------------------------------------- #
# 3. heartbeat — a silent watcher must be impossible
# --------------------------------------------------------------------------- #
def _cfg(mode="shadow"):
    return pxo.load_overlay_config({
        "schema_version": pxo.CONFIG_SCHEMA_VERSION,
        "mode": mode,
        "atr_period": 14,
        "atr_trail_mult": 3.0,
        "min_bars_for_atr": 5,
        "hard_stop_loss_pct": 0.08,
        "max_hold_days": {"equity": 20, "etf": 30},
    })


def _overlay(tmp_path, *, mode="shadow", open_positions=None, hb=True):
    return pxo.PositionExitOverlay(
        _cfg(mode),
        evidence_path=tmp_path / "evi",
        heartbeat_path=(tmp_path / "hb.json") if hb else None,
        state_path=tmp_path / "state.json",
        guard_loader=lambda: {},
        open_positions_loader=lambda: (open_positions or {}),
        bars_loader=lambda syms: {},
        price_loader=lambda syms: {},
        env={},
    )


def _hb(tmp_path):
    return json.loads((tmp_path / "hb.json").read_text())


def test_heartbeat_fires_when_evaluating_nothing(tmp_path, caplog):
    """The XOV-2345 state itself: alive, zero positions — must be LOUD."""
    ov = _overlay(tmp_path, open_positions={})
    with caplog.at_level(logging.INFO):
        ov.run_cycle(None, now_utc=NOW)

    assert pxo.MARKER_HEARTBEAT in caplog.text
    assert "evaluated=0" in caplog.text
    payload = _hb(tmp_path)
    assert payload["evaluated"] == 0
    assert payload["schema_version"] == pxo.HEARTBEAT_SCHEMA_VERSION
    assert payload["ts_utc"] == "2026-07-14T23:44:00Z"


def test_heartbeat_fires_when_mode_off(tmp_path):
    """'disabled' must stay distinguishable from 'dead'."""
    _overlay(tmp_path, mode="off").run_cycle(None, now_utc=NOW)
    assert _hb(tmp_path)["mode"] == "off"


def test_heartbeat_fires_on_the_error_path(tmp_path):
    ov = pxo.PositionExitOverlay(
        _cfg(), evidence_path=tmp_path / "evi", heartbeat_path=tmp_path / "hb.json",
        state_path=tmp_path / "state.json",
        guard_loader=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        open_positions_loader=lambda: {}, bars_loader=lambda s: {},
        price_loader=lambda s: {}, env={},
    )
    ov.run_cycle(None, now_utc=NOW)
    assert _hb(tmp_path)["healthy"] is False


def test_heartbeat_failure_never_breaks_the_cycle(tmp_path):
    """Heartbeat is observability: it must never become a new way to die.

    Note `evaluated=True` here despite zero verdicts — the flag means "the core
    ran", not "positions were seen". That ambiguity is exactly why the outage was
    invisible, and why the heartbeat reports the verdict COUNT instead.
    """
    ov = _overlay(tmp_path, open_positions={})
    ov._heartbeat_path = tmp_path / "nope" / "x.json"
    (tmp_path / "nope").write_text("i am a file, not a dir")
    res = ov.run_cycle(None, now_utc=NOW)  # unwritable heartbeat must not raise
    assert res.verdicts == []


def test_build_default_overlay_refuses_real_heartbeat_path_under_pytest(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "schema_version": pxo.CONFIG_SCHEMA_VERSION, "mode": "shadow",
        "atr_period": 14, "atr_trail_mult": 3.0, "min_bars_for_atr": 5,
        "hard_stop_loss_pct": 0.08, "max_hold_days": {"equity": 20, "etf": 30},
    }))
    with pytest.raises(RuntimeError, match="heartbeat_path is REQUIRED-explicit"):
        pxo.build_default_overlay(
            repo_root=tmp_path, config_path=cfg,
            evidence_dir=tmp_path / "e", state_path=tmp_path / "s.json",
        )


# --------------------------------------------------------------------------- #
# 4. date rollover — the ruled-out hypothesis, pinned so it stays ruled out
# --------------------------------------------------------------------------- #
def test_date_rollover_does_not_stop_evaluation_or_heartbeat(tmp_path):
    """Simulate crossing 23:45 -> 00:15 UTC: evidence rolls to a NEW dated file,
    evaluation and the heartbeat continue across the boundary.

    The real overlay resolves its evidence path per-record from verdict.ts_utc,
    so rollover was never the cause — this pins that, and proves rollover cannot
    silently stop the watcher.
    """
    positions = {
        "gamma|UNH": {
            "open": True, "symbol": "UNH", "side": "BUY", "quantity": 261.0,
        },
    }
    ov = pxo.PositionExitOverlay(
        _cfg(), evidence_path=tmp_path / "evi", heartbeat_path=tmp_path / "hb.json",
        state_path=tmp_path / "state.json",
        guard_loader=lambda: {"broker_sync|UNH": {
            "open": True, "symbol": "UNH", "side": "BUY", "quantity": 261.0,
        }},
        open_positions_loader=lambda: positions,
        bars_loader=lambda syms: {},
        price_loader=lambda syms: {"UNH": 425.0},
        env={},
    )

    start = datetime(2026, 7, 14, 23, 45, 0, tzinfo=timezone.utc)
    stamps = [start + timedelta(minutes=10 * i) for i in range(4)]  # 23:45 -> 00:15
    for ts in stamps:
        res = ov.run_cycle(None, now_utc=ts)
        assert res.evaluated is True, f"evaluation stopped at {ts}"
        assert _hb(tmp_path)["evaluated"] == 1, f"heartbeat stalled at {ts}"

    day1 = tmp_path / "evi" / "exit_overlay_20260714.ndjson"
    day2 = tmp_path / "evi" / "exit_overlay_20260715.ndjson"
    assert day1.is_file() and day2.is_file(), "evidence did not roll over"
    assert len(day1.read_text().strip().splitlines()) == 2   # 23:45, 23:55
    assert len(day2.read_text().strip().splitlines()) == 2   # 00:05, 00:15
    assert _hb(tmp_path)["ts_utc"] == "2026-07-15T00:15:00Z"


# --------------------------------------------------------------------------- #
# 5. R14 — stale OR blind must reach the operator
# --------------------------------------------------------------------------- #
def _seed(tmp_path, monkeypatch, hb_payload, snap_positions=None, hb_age=0.0):
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    (tmp_path / "exit_overlay_heartbeat.json").write_text(json.dumps(hb_payload))
    if snap_positions is not None:
        (tmp_path / "positions_snapshot.json").write_text(
            json.dumps({"positions": snap_positions})
        )
    if hb_age:
        import os
        import time
        t = time.time() - hb_age
        os.utime(tmp_path / "exit_overlay_heartbeat.json", (t, t))


_FRESH = {"schema_version": "exit_overlay_heartbeat.v1", "ttl_seconds": 900,
          "mode": "shadow", "evaluated": 0, "would_close": 0, "healthy": True}
_HELD = [{"symbol": "UNH", "secType": "STK", "position": 261.0}]


def test_r14_missing_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert f[0].title == "Exit overlay heartbeat MISSING"


def test_r14_stale_heartbeat(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, dict(_FRESH, evaluated=3), hb_age=4000)
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert f[0].title == "Exit overlay heartbeat STALE"
    assert f[0].severity == "CRITICAL"
    assert f[0].remedy_type == "NOTIFY_ONLY"  # SS01: never restart a trading engine


def test_r14_alive_but_blind_is_the_xov2345_signature(tmp_path, monkeypatch):
    """Fresh heartbeat + evaluated=0 + broker holding real stock = the outage."""
    _seed(tmp_path, monkeypatch, _FRESH, snap_positions=_HELD)
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert len(f) == 1
    assert f[0].title == "Exit overlay is watching nothing"
    assert f[0].severity == "CRITICAL"


def test_r14_quiet_when_broker_is_genuinely_flat(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, _FRESH, snap_positions=[])
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert f == []


def test_r14_quiet_when_evaluating_normally(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, dict(_FRESH, evaluated=8), snap_positions=_HELD)
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert f == []


def test_r14_does_not_cry_wolf_on_stale_broker_truth(tmp_path, monkeypatch):
    """evaluated=0 proves nothing if the broker snapshot itself is stale."""
    _seed(tmp_path, monkeypatch, _FRESH, snap_positions=_HELD)
    import os
    import time
    t = time.time() - 5000
    os.utime(tmp_path / "positions_snapshot.json", (t, t))
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert f == []


def test_r14_titles_are_stable_identities(tmp_path, monkeypatch):
    """CTF-T2: titles must not embed fluctuating values or dedupe breaks."""
    _seed(tmp_path, monkeypatch, dict(_FRESH, evaluated=3), hb_age=4000)
    f = []
    hmr.rule_exit_overlay_heartbeat(f)
    assert not any(ch.isdigit() for ch in f[0].title)

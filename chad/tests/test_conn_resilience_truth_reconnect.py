"""CONN-RESILIENCE — the shared truth-reading `ib` must self-heal after a
gateway reset instead of staying dead until a manual restart.

Forensic summary of the outage this heals:

  The shared ``ib`` (clientId LIVE_LOOP=99) connects exactly ONCE at module
  import and never reconnects. IBKR's ~23:45 UTC daily session reset AND our own
  03:15 gateway nightly restart both drop it. XOV-2345 made that drop
  safe-and-loud (fail-closed: no false-flat, BROKER_TRUTH_UNAVAILABLE marker +
  alert) but the connection stayed DEAD — observed 13:30->19:11 UTC, 272
  BROKER_TRUTH_UNAVAILABLE cycles — until an operator restarted the process.

CONN-RESILIENCE adds a bounded, backed-off reconnect of the shared truth `ib`
inside the guarded ``fetch_positions`` path. The L1-CLD lessons are honoured:
the reconnect runs ONLY on MainThread (the loop the shared reader is bound to),
it never touches the dedicated execution owner-loop connection (clientId 9007),
attempts are bounded (no reconnect storm), and every ``ib.connect`` is bounded
by a timeout. Failure stays XOV2-1 loud.

These tests never perform real IB I/O — the fake ``ib`` reconnects in memory.
"""

from __future__ import annotations

import logging
import threading

import pytest

from chad.core import live_loop
from chad.core.broker_position_sync import (
    BrokerPositionSync,
    BrokerTruthUnavailable,
)
from chad.execution.ibkr_client_ids import EXECUTION as _EXECUTION_CLIENT_ID
from chad.execution.ibkr_client_ids import LIVE_LOOP as _LIVE_LOOP_CLIENT_ID


# --------------------------------------------------------------------------- #
# fakes — model ib_async: positions() reads a cache reset on drop; connect()
# re-establishes and repopulates from reqPositions (as connectAsync does).
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


class _HealingFakeIb:
    def __init__(self, live_positions=()):
        self._live = [_IbPos(s, q) for s, q in live_positions]  # broker truth
        self._connected = True
        self._cache = list(self._live)
        self.connect_calls = []      # list of (clientId, timeout)
        self.disconnect_calls = 0
        self.connect_should_fail = False

    def isConnected(self):
        return self._connected

    def positions(self):
        return list(self._cache)

    def drop_connection(self):
        # gateway reset: socket drop → wrapper.reset() empties the cache
        self._connected = False
        self._cache = []

    def disconnect(self):
        self.disconnect_calls += 1
        self._connected = False
        self._cache = []

    def connect(self, host, port, clientId=None, timeout=None):
        self.connect_calls.append((clientId, timeout))
        if self.connect_should_fail:
            raise ConnectionRefusedError("no gateway on 4002")
        self._connected = True
        self._cache = list(self._live)  # connectAsync re-issues reqPositions


@pytest.fixture
def heal_env(monkeypatch):
    """Wire a healing fake `ib` in, allow real (in-memory) reconnects, and make
    backoff instant."""
    fake = _HealingFakeIb([("UNH", 261)])
    monkeypatch.setattr(live_loop, "position_sync", BrokerPositionSync(fake))
    # the hard I/O-safety gate reads the env directly — clear it so the fake's
    # in-memory connect() is allowed to run (no real socket is ever opened).
    monkeypatch.delenv("CHAD_SKIP_IB_CONNECT", raising=False)
    monkeypatch.delenv("CHAD_TRUTH_RECONNECT", raising=False)
    monkeypatch.delenv("CHAD_TRUTH_RECONNECT_MAX_ATTEMPTS", raising=False)
    # never really sleep during backoff
    slept: list = []
    monkeypatch.setattr(live_loop.time, "sleep", lambda s: slept.append(s))
    return fake, slept


# --------------------------------------------------------------------------- #
# 1. disconnect -> reconnect -> truth resumes
# --------------------------------------------------------------------------- #
def test_reconnect_heals_and_probe_recovers(heal_env, caplog):
    fake, _ = heal_env
    fake.drop_connection()  # 23:45 UTC session reset
    assert not fake.isConnected()

    with caplog.at_level(logging.WARNING):
        ok = live_loop._attempt_shared_ib_reconnect(logging.getLogger("test.conn"))

    assert ok is True
    assert fake.isConnected()
    assert "BROKER_TRUTH_RECONNECT_OK" in caplog.text
    # reconnected with the shared LIVE_LOOP id, never the execution id
    assert fake.connect_calls == [(_LIVE_LOOP_CLIENT_ID, 30.0)]


def test_handle_returns_fresh_truth_and_emits_restored(heal_env, monkeypatch, caplog):
    fake, _ = heal_env
    fake.drop_connection()
    monkeypatch.setattr(live_loop, "_skip_ib_connect", lambda: False)
    alerts: list = []
    monkeypatch.setattr(
        live_loop, "_fire_alert_safe",
        lambda kind, fn, *a, **k: alerts.append(kind) or True,
    )

    with caplog.at_level(logging.WARNING):
        out = live_loop._handle_broker_truth_unavailable(
            logging.getLogger("test.conn"), BrokerTruthUnavailable("api down")
        )

    assert out is not None and out["UNH"].quantity == 261, "truth did not resume"
    assert "BROKER_TRUTH_RESTORED" in caplog.text
    assert alerts == ["broker_truth_restored"], "no coach-voiced recovery note"


def test_full_rebuild_resumes_from_truth_after_heal(monkeypatch, heal_env):
    """End-to-end: fetch first raises (dead), reconnect heals, the guard rebuild
    reconciles against real broker truth — the open entry the broker still holds
    stays OPEN (not false-flatted, and not skipped)."""
    fake, _ = heal_env
    import json

    from chad.core import position_guard

    state = {
        "gamma|UNH": {
            "open": True, "symbol": "UNH", "side": "BUY", "quantity": 261.0,
            "strategy": "gamma",
        },
    }
    monkeypatch.setattr(live_loop, "_load_state", lambda: dict(state), raising=False)
    monkeypatch.setattr(position_guard, "_load_state", lambda: json.loads(json.dumps(state)))
    monkeypatch.setattr(position_guard, "save_state",
                        lambda s: state.update(json.loads(json.dumps(s))))
    monkeypatch.setattr(live_loop, "_skip_ib_connect", lambda: False)
    monkeypatch.setattr(live_loop, "_fire_alert_safe", lambda *a, **k: True)

    # First fetch raises (connection down); after the heal the real fetch works.
    calls = {"n": 0}
    real_fetch = live_loop.position_sync.fetch_positions

    def flaky_fetch():
        calls["n"] += 1
        if calls["n"] == 1:
            raise BrokerTruthUnavailable("api down at 23:45")
        return real_fetch()

    monkeypatch.setattr(live_loop.position_sync, "fetch_positions", flaky_fetch)

    live_loop._rebuild_guard_from_broker(logging.getLogger("test.conn"))

    assert state["gamma|UNH"]["open"] is True, "healed truth was not applied"
    # broker_sync mirror of the still-held UNH was created from live truth
    assert any(k.startswith("broker_sync|") for k in state), "did not reconcile from truth"


# --------------------------------------------------------------------------- #
# 2. reconnect storm is bounded
# --------------------------------------------------------------------------- #
def test_reconnect_storm_is_bounded_and_backed_off(heal_env, monkeypatch, caplog):
    fake, slept = heal_env
    fake.drop_connection()
    fake.connect_should_fail = True  # every attempt fails
    monkeypatch.setenv("CHAD_TRUTH_RECONNECT_MAX_ATTEMPTS", "3")

    with caplog.at_level(logging.ERROR):
        ok = live_loop._attempt_shared_ib_reconnect(logging.getLogger("test.conn"))

    assert ok is False
    assert len(fake.connect_calls) == 3, "unbounded reconnect storm"
    assert slept == [2.0, 4.0], "backoff not applied between attempts (no sleep after last)"
    assert "BROKER_TRUTH_RECONNECT_EXHAUSTED" in caplog.text


def test_exhausted_reconnect_stays_xov2_loud(heal_env, monkeypatch):
    fake, _ = heal_env
    fake.drop_connection()
    fake.connect_should_fail = True
    monkeypatch.setattr(live_loop, "_skip_ib_connect", lambda: False)
    alerts: list = []
    monkeypatch.setattr(
        live_loop, "_fire_alert_safe",
        lambda kind, fn, *a, **k: alerts.append(kind) or True,
    )

    out = live_loop._handle_broker_truth_unavailable(
        logging.getLogger("test.conn"), BrokerTruthUnavailable("api down")
    )

    assert out is None, "must abort the rebuild when it cannot heal"
    assert alerts == ["broker_truth_unavailable"], "silent degradation on exhaustion"


# --------------------------------------------------------------------------- #
# 3. owner loop / foreign-loop safety (L1-CLD)
# --------------------------------------------------------------------------- #
def test_reconnect_refused_off_mainthread(heal_env):
    """L1-CLD: the shared reader must NEVER be rebound to a foreign loop — a
    reconnect attempted off MainThread refuses and performs no I/O."""
    fake, _ = heal_env
    fake.drop_connection()
    result: dict = {}

    def worker():
        result["ok"] = live_loop._attempt_shared_ib_reconnect(
            logging.getLogger("test.conn")
        )

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert result.get("ok") is False
    assert fake.connect_calls == [], "reconnected the shared `ib` off MainThread"


def test_reconnect_never_uses_execution_clientid(heal_env):
    """The dedicated execution connection (9007) is separate and healthy — the
    truth heal must only ever reconnect the shared LIVE_LOOP id (99)."""
    fake, _ = heal_env
    fake.drop_connection()
    live_loop._attempt_shared_ib_reconnect(logging.getLogger("test.conn"))
    assert _LIVE_LOOP_CLIENT_ID != _EXECUTION_CLIENT_ID
    assert all(cid == _LIVE_LOOP_CLIENT_ID for cid, _ in fake.connect_calls)
    assert all(cid != _EXECUTION_CLIENT_ID for cid, _ in fake.connect_calls)


def test_heal_never_touches_execution_owner_loop(heal_env, monkeypatch):
    """Behaviour guard: a truth heal must not invoke the execution owner-loop
    homing (_home_execution_connection) nor the execution adapter's connect."""
    fake, _ = heal_env
    fake.drop_connection()
    touched: list = []
    monkeypatch.setattr(live_loop, "_home_execution_connection",
                        lambda *a, **k: touched.append("home") or False)
    if hasattr(live_loop, "_paper_adapter"):
        monkeypatch.setattr(
            live_loop._paper_adapter, "ensure_connected",
            lambda *a, **k: touched.append("exec_connect"), raising=False,
        )

    ok = live_loop._attempt_shared_ib_reconnect(logging.getLogger("test.conn"))
    assert ok is True
    assert touched == [], "truth heal reached into the execution owner-loop path"


# --------------------------------------------------------------------------- #
# 4. kill-switch + hard I/O-safety gate
# --------------------------------------------------------------------------- #
def test_kill_switch_disables_heal(heal_env, monkeypatch, caplog):
    fake, _ = heal_env
    fake.drop_connection()
    monkeypatch.setenv("CHAD_TRUTH_RECONNECT", "0")
    with caplog.at_level(logging.WARNING):
        ok = live_loop._attempt_shared_ib_reconnect(logging.getLogger("test.conn"))
    assert ok is False
    assert fake.connect_calls == [], "healed despite kill-switch off"
    assert "BROKER_TRUTH_RECONNECT_DISABLED" in caplog.text


def test_default_on_when_unset(heal_env, monkeypatch):
    monkeypatch.delenv("CHAD_TRUTH_RECONNECT", raising=False)
    assert live_loop._truth_reconnect_enabled() is True


def test_hard_env_gate_blocks_real_io_under_pytest(heal_env, monkeypatch):
    """Even if a test forces _skip_ib_connect() False, CHAD_SKIP_IB_CONNECT=1
    must hard-block any real reconnect I/O so a suite run never touches the live
    gateway / clientId 99."""
    fake, _ = heal_env
    fake.drop_connection()
    monkeypatch.setenv("CHAD_SKIP_IB_CONNECT", "1")
    monkeypatch.setattr(live_loop, "_skip_ib_connect", lambda: False)  # forced
    ok = live_loop._attempt_shared_ib_reconnect(logging.getLogger("test.conn"))
    assert ok is False
    assert fake.connect_calls == [], "performed real IB I/O during a suite run"


# --------------------------------------------------------------------------- #
# 5. tunables degrade gracefully
# --------------------------------------------------------------------------- #
def test_invalid_tunables_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("CHAD_TRUTH_RECONNECT_MAX_ATTEMPTS", "not-an-int")
    monkeypatch.setenv("CHAD_TRUTH_RECONNECT_TIMEOUT_S", "nope")
    assert live_loop._truth_reconnect_max_attempts() == 3
    assert live_loop._truth_reconnect_timeout_s() == 30.0
    monkeypatch.setenv("CHAD_TRUTH_RECONNECT_MAX_ATTEMPTS", "0")  # clamped to >=1
    assert live_loop._truth_reconnect_max_attempts() == 1

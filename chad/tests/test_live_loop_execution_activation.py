"""L1-CLD U7 — activation: live_loop hands execution-connection ownership to
the broker owner loop.

These tests pin the U7 wiring that makes the cross-loop-deadlock fix
LIVE-effective (PA ops/pending_actions/L1_CLD_cross_loop_deadlock_fix_2026-07-08):

  * params-mode is the production default (the execution adapter owns a
    DEDICATED IB connection instead of adopting live_loop's MainThread `ib`);
  * the dedicated execution clientId comes from config — the canonical
    registry default ``ibkr_client_ids.EXECUTION`` (9007) or the
    ``CHAD_EXECUTION_CLIENT_ID`` env override — and never collides with
    ``LIVE_LOOP`` or any registered service id;
  * boot homing establishes the connection ON the owner-loop thread
    (``chad-broker-loop``) via ``connectAsync`` — NOT the MainThread sync
    ``connect``;
  * boot-homing FAILURE puts live_loop in a NO-EXECUTION state and emits the
    ``BROKER_LOOP_DOWN`` marker, with NO silent MainThread fallback.
"""

from __future__ import annotations

import os

# Import parity with the pytest environment: never claim a live clientId at
# import time (the running live_loop process holds LIVE_LOOP=99).
os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")

import logging
import threading

import pytest

from chad.execution import ibkr_client_ids
from chad.execution.ibkr_adapter import IbkrAdapter, IbkrConfig

OWNER_THREAD = "chad-broker-loop"


def _live_loop():
    import chad.core.live_loop as ll

    return ll


# ---------------------------------------------------------------------------
# P2 — dedicated clientId is registered and collision-free
# ---------------------------------------------------------------------------

def test_execution_clientid_registered_and_collision_free() -> None:
    # Registered in the canonical registry with a matching map entry.
    assert hasattr(ibkr_client_ids, "EXECUTION")
    mapping = ibkr_client_ids.client_id_map()
    assert mapping["EXECUTION"] == ibkr_client_ids.EXECUTION == 9007
    # Registry self-consistency: no two named services share an id (raises if
    # a collision is ever introduced).
    ibkr_client_ids.assert_no_collisions()
    # Dedicated: distinct from the shared MainThread live-loop connection.
    assert ibkr_client_ids.EXECUTION != ibkr_client_ids.LIVE_LOOP
    # Absent from the 2026-07-08 IB Gateway client census.
    gateway_observed = {80, 99, 9001, 9003, 9021, 9035}
    assert ibkr_client_ids.EXECUTION not in gateway_observed
    # Unique across the whole registered id set.
    ids = ibkr_client_ids.all_client_ids()
    assert ids.count(ibkr_client_ids.EXECUTION) == 1


# ---------------------------------------------------------------------------
# live_loop wiring helpers — params-mode default + config-driven clientId
# ---------------------------------------------------------------------------

def test_params_mode_is_production_default(monkeypatch) -> None:
    ll = _live_loop()
    monkeypatch.delenv("CHAD_EXECUTION_OWN_CONNECTION", raising=False)
    assert ll._execution_owns_connection() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF"])
def test_own_connection_kill_switch_forces_legacy(monkeypatch, val) -> None:
    ll = _live_loop()
    monkeypatch.setenv("CHAD_EXECUTION_OWN_CONNECTION", val)
    assert ll._execution_owns_connection() is False


def test_execution_clientid_default_env_override_and_fallback(monkeypatch) -> None:
    ll = _live_loop()
    monkeypatch.delenv("CHAD_EXECUTION_CLIENT_ID", raising=False)
    assert ll._execution_client_id() == ibkr_client_ids.EXECUTION
    monkeypatch.setenv("CHAD_EXECUTION_CLIENT_ID", "12345")
    assert ll._execution_client_id() == 12345
    # A non-int override never crashes boot — it falls back to the safe default.
    monkeypatch.setenv("CHAD_EXECUTION_CLIENT_ID", "not-an-int")
    assert ll._execution_client_id() == ibkr_client_ids.EXECUTION


def test_module_paper_adapter_uses_dedicated_execution_clientid() -> None:
    ll = _live_loop()
    # Imported under params-mode default -> the module adapter carries the
    # dedicated execution clientId, not LIVE_LOOP's.
    assert ll._execution_owns_connection() is True
    assert ll._paper_adapter._config.client_id == ibkr_client_ids.EXECUTION
    assert ll._paper_adapter._config.client_id != ibkr_client_ids.LIVE_LOOP


# ---------------------------------------------------------------------------
# U7(c) — params-mode boots -> connection created on the OWNER loop, not
# MainThread. Async-capable fake IB records the thread each call runs on.
# ---------------------------------------------------------------------------

class _Ev:
    def __init__(self) -> None:
        self._h: list = []

    def __iadd__(self, h):
        self._h.append(h)
        return self

    def emit(self, *a, **k):
        for h in list(self._h):
            h(*a, **k)

    def __len__(self):
        return len(self._h)


class _Conn:
    def __init__(self) -> None:
        self.hasData = _Ev()


class _Client:
    def __init__(self) -> None:
        self.conn = _Conn()


class _ParamsFakeIB:
    """Constructed UNCONNECTED, exactly as live_loop's params-mode factory
    (``lambda: IB()``) does. Exposes the ``*Async`` twins so the adapter routes
    it through the owner loop, and records the thread ``connectAsync`` runs on.
    Its sync ``connect`` raises — a MainThread sync connect must NEVER happen on
    the params-mode path."""

    def __init__(self) -> None:
        self._connected = False
        self.client = _Client()
        self.updateEvent = _Ev()
        self.connect_thread = None
        self.connect_count = 0
        self.client_id_seen = None

    def isConnected(self) -> bool:
        return self._connected

    async def connectAsync(self, host, port, clientId, timeout=None, **kw):  # noqa: N803
        self.connect_thread = threading.current_thread().name
        self.client_id_seen = clientId
        self.connect_count += 1
        self._connected = True
        return self

    def connect(self, *a, **k):  # legacy sync — must not run in params-mode
        raise AssertionError(
            "MainThread sync connect() must never run on the params-mode path"
        )

    def disconnect(self):
        self._connected = False

    def managedAccounts(self):
        return []

    async def qualifyContractsAsync(self, *contracts):
        return list(contracts)

    def qualifyContracts(self, *contracts):
        return list(contracts)


def test_params_mode_connects_on_owner_loop_not_mainthread() -> None:
    fake = _ParamsFakeIB()
    adapter = IbkrAdapter(
        config=IbkrConfig(
            dry_run=False,
            enable_idempotency=False,
            client_id=ibkr_client_ids.EXECUTION,
        ),
        ib_factory=lambda: fake,
    )
    try:
        adapter.ensure_connected(force=True)
        # Homed on the owner loop, not adopted.
        assert adapter._owner_loop_homed is True
        assert adapter._broker_loop is not None
        assert adapter._broker_loop.is_alive()
        # Connection was established via connectAsync ON the owner-loop thread.
        assert fake.connect_count == 1
        assert fake.connect_thread == OWNER_THREAD
        assert fake.connect_thread != threading.main_thread().name
        # ...with the SAME dedicated execution clientId (constraint 4).
        assert fake.client_id_seen == ibkr_client_ids.EXECUTION
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# U7(d) — boot-homing failure -> NO-EXECUTION state + BROKER_LOOP_DOWN, no
# silent fallback. Also: success path re-enables; legacy mode never homes.
# ---------------------------------------------------------------------------

class _BoomAdapter:
    def ensure_connected(self, *, force: bool = False):
        raise ConnectionError("simulated gateway unreachable at boot")


class _OkAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def ensure_connected(self, *, force: bool = False):
        self.calls += 1


def test_boot_homing_failure_sets_no_execution_and_marker(monkeypatch, caplog) -> None:
    ll = _live_loop()
    monkeypatch.delenv("CHAD_SKIP_IB_CONNECT", raising=False)
    monkeypatch.setenv("CHAD_EXECUTION_OWN_CONNECTION", "1")
    monkeypatch.setattr(ll, "_paper_adapter", _BoomAdapter())
    monkeypatch.setattr(ll, "_EXECUTION_DISABLED", False, raising=False)

    logger = logging.getLogger("chad.live_loop.test_u7")
    with caplog.at_level(logging.ERROR):
        homed = ll._home_execution_connection(logger)

    assert homed is False
    # Fail-closed: execution disabled for the rest of the process.
    assert ll._EXECUTION_DISABLED is True
    # BROKER_LOOP_DOWN marker was emitted (message text or the structured tag).
    assert any(
        "BROKER_LOOP_DOWN" in rec.getMessage()
        or rec.__dict__.get("marker") == "BROKER_LOOP_DOWN"
        for rec in caplog.records
    )


def test_boot_homing_success_engages_and_enables(monkeypatch) -> None:
    ll = _live_loop()
    monkeypatch.delenv("CHAD_SKIP_IB_CONNECT", raising=False)
    monkeypatch.setenv("CHAD_EXECUTION_OWN_CONNECTION", "1")
    ok = _OkAdapter()
    monkeypatch.setattr(ll, "_paper_adapter", ok)
    # Pretend a previous failure left it disabled; a successful homing clears it.
    monkeypatch.setattr(ll, "_EXECUTION_DISABLED", True, raising=False)

    homed = ll._home_execution_connection(logging.getLogger("chad.live_loop.test_u7"))

    assert homed is True
    assert ok.calls == 1
    assert ll._EXECUTION_DISABLED is False


def test_legacy_mode_never_homes(monkeypatch) -> None:
    ll = _live_loop()
    monkeypatch.setenv("CHAD_EXECUTION_OWN_CONNECTION", "0")

    class _MustNotHome:
        def ensure_connected(self, *, force: bool = False):
            raise AssertionError("must not attempt homing in legacy-adoption mode")

    monkeypatch.setattr(ll, "_paper_adapter", _MustNotHome())
    assert ll._home_execution_connection(logging.getLogger("chad.live_loop.test_u7")) is False


def test_skip_ib_connect_short_circuits_homing(monkeypatch) -> None:
    ll = _live_loop()
    monkeypatch.setenv("CHAD_EXECUTION_OWN_CONNECTION", "1")
    monkeypatch.setenv("CHAD_SKIP_IB_CONNECT", "1")

    class _MustNotHome:
        def ensure_connected(self, *, force: bool = False):
            raise AssertionError("must not connect when CHAD_SKIP_IB_CONNECT is set")

    monkeypatch.setattr(ll, "_paper_adapter", _MustNotHome())
    assert ll._home_execution_connection(logging.getLogger("chad.live_loop.test_u7")) is False

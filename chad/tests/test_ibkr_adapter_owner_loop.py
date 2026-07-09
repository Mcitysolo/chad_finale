"""L1-CLD U3 — IbkrAdapter connection-owner-loop migration.

Exercises the owner-loop routing with an async-capable fake IB (exposes the
ib_async ``*Async`` twins). Proves: the adapter homes the connection on the
owner loop, marshals qualify / whatIf / openTrades / placeOrder onto it, wires
the reader-progress hook, translates owner-loop timeouts to BrokerTimeoutError,
adopts an externally pre-connected IB WITHOUT re-homing (legacy behavior), and
tears the loop down on shutdown.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from chad.execution.ibkr_adapter import (
    BrokerTimeoutError,
    IbkrAdapter,
    IbkrConfig,
    _PreparedOrder,
)

OWNER_THREAD = "chad-broker-loop"


# ---------------------------------------------------------------------------
# Async-capable fake IB — mirrors ib_async's dual sync+async surface.
# ---------------------------------------------------------------------------

class _Ev:
    def __init__(self) -> None:
        self._handlers = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def emit(self, *a, **k):
        for h in list(self._handlers):
            h(*a, **k)

    def __len__(self):
        return len(self._handlers)


class _Conn:
    def __init__(self) -> None:
        self.hasData = _Ev()


class _Client:
    def __init__(self) -> None:
        self.conn = _Conn()


class _OrderStatus:
    def __init__(self, status: str) -> None:
        self.status = status


class _Order:
    def __init__(self, oid: int = 555) -> None:
        self.orderId = oid


class _Trade:
    def __init__(self, status: str = "Filled") -> None:
        self.orderStatus = _OrderStatus(status)
        self.order = _Order()
        self.fills = []
        self.commissionReport = []
        self.statusEvent = _Ev()


class _AsyncFakeIB:
    """Has BOTH sync and async twins, like a real ib_async.IB, so the owner-loop
    path (async twins) and the legacy fallback (sync twins) are both realistic."""

    def __init__(self, *, connected: bool = False, hang_qualify: bool = False) -> None:
        self._connected = connected
        self.client = _Client()
        self.updateEvent = _Ev()
        self._hang_qualify = hang_qualify
        self.calls = []  # (method, thread_name)
        self.connect_count = 0

    def isConnected(self) -> bool:
        return self._connected

    async def connectAsync(self, host, port, clientId, timeout=None, **kw):  # noqa: N803
        self.connect_count += 1
        self._connected = True
        return self

    def connect(self, host, port, clientId, timeout):  # noqa: N803 - legacy sync
        self.connect_count += 1
        self._connected = True

    def disconnect(self):
        self._connected = False

    def managedAccounts(self):
        return []

    async def qualifyContractsAsync(self, *contracts):
        self.calls.append(("qualifyContractsAsync", threading.current_thread().name))
        if self._hang_qualify:
            await asyncio.sleep(100)
        return list(contracts)

    def qualifyContracts(self, *contracts):
        self.calls.append(("qualifyContracts", threading.current_thread().name))
        return list(contracts)

    async def whatIfOrderAsync(self, contract, order):
        self.calls.append(("whatIfOrderAsync", threading.current_thread().name))
        return {"initMargin": "1.0"}

    def whatIfOrder(self, contract, order):
        self.calls.append(("whatIfOrder", threading.current_thread().name))
        return {"initMargin": "1.0"}

    def placeOrder(self, contract, order):
        self.calls.append(("placeOrder", threading.current_thread().name))
        return _Trade("Filled")

    def openTrades(self):
        self.calls.append(("openTrades", threading.current_thread().name))
        return []


def _cfg() -> IbkrConfig:
    return IbkrConfig(dry_run=True, enable_idempotency=False)


def _adapter(fake: _AsyncFakeIB) -> IbkrAdapter:
    return IbkrAdapter(config=_cfg(), ib_factory=lambda: fake)


# ---------------------------------------------------------------------------
# Homing + reader-hook wiring
# ---------------------------------------------------------------------------

def test_homes_connection_on_owner_loop() -> None:
    fake = _AsyncFakeIB(connected=False)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        assert adapter._owner_loop_homed is True
        assert adapter._broker_loop is not None
        assert adapter._broker_loop.is_alive()
        assert fake.connect_count == 1
        assert fake.isConnected()
        # Reader-progress hook subscribed to the persistent conn.hasData event.
        assert len(fake.client.conn.hasData) >= 1
    finally:
        adapter.shutdown()


def test_shutdown_tears_down_owner_loop() -> None:
    fake = _AsyncFakeIB(connected=False)
    adapter = _adapter(fake)
    adapter.ensure_connected(force=True)
    bl = adapter._broker_loop
    assert bl is not None and bl.is_alive()
    adapter.shutdown()
    assert not bl.is_alive()
    assert adapter._broker_loop is None
    assert adapter._owner_loop_homed is False


# ---------------------------------------------------------------------------
# Broker calls marshalled onto the owner loop
# ---------------------------------------------------------------------------

def test_qualify_runs_on_owner_loop() -> None:
    fake = _AsyncFakeIB(connected=False)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        result = adapter._broker_call(fake, "qualifyContracts", "C1", label="q")
        assert result == ["C1"]
        assert fake.calls[-1] == ("qualifyContractsAsync", OWNER_THREAD)
    finally:
        adapter.shutdown()


def test_whatif_runs_on_owner_loop() -> None:
    fake = _AsyncFakeIB(connected=False)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        result = adapter._broker_call(fake, "whatIfOrder", "C", "O", label="w")
        assert result == {"initMargin": "1.0"}
        assert fake.calls[-1] == ("whatIfOrderAsync", OWNER_THREAD)
    finally:
        adapter.shutdown()


def test_open_trades_sync_only_runs_on_owner_loop() -> None:
    """openTrades has no async twin -> must run ON the owner loop via submit_call
    (thread-safe read of the IB's wrapper state)."""
    fake = _AsyncFakeIB(connected=False)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        result = adapter._broker_call(fake, "openTrades", label="ot")
        assert result == []
        assert fake.calls[-1] == ("openTrades", OWNER_THREAD)
    finally:
        adapter.shutdown()


def test_place_and_wait_async_places_and_returns_terminal() -> None:
    fake = _AsyncFakeIB(connected=False)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        prepared = _PreparedOrder(order=_Order(), quantity=1.0, what_if=False)
        trade = adapter._broker_loop.submit_coro(
            adapter._place_and_wait_async(fake, "C", prepared, "idem-key"),
            timeout_s=5.0,
            label="place",
        )
        assert trade.orderStatus.status == "Filled"
        assert ("placeOrder", OWNER_THREAD) in fake.calls
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Owner-loop timeout translation
# ---------------------------------------------------------------------------

def test_owner_loop_timeout_translates_to_broker_timeout() -> None:
    fake = _AsyncFakeIB(connected=False, hang_qualify=True)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        with pytest.raises(BrokerTimeoutError):
            adapter._broker_call(fake, "qualifyContracts", "C", timeout_s=0.2, label="q")
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# Legacy: externally pre-connected IB is adopted, NOT re-homed
# ---------------------------------------------------------------------------

def test_preconnected_ib_adopted_without_rehoming() -> None:
    fake = _AsyncFakeIB(connected=True)  # already connected (e.g. live_loop)
    adapter = _adapter(fake)
    try:
        adapter.ensure_connected(force=True)
        # Not re-homed: owner loop never started; no reconnect issued.
        assert adapter._owner_loop_homed is False
        assert adapter._broker_loop is None
        assert fake.connect_count == 0
        # Broker calls take the legacy bounded-sync fallback (sync twin).
        result = adapter._broker_call(fake, "qualifyContracts", "C1", label="q")
        assert result == ["C1"]
        assert fake.calls[-1][0] == "qualifyContracts"  # sync, not the async twin
        assert fake.calls[-1][1] != OWNER_THREAD
    finally:
        adapter.shutdown()

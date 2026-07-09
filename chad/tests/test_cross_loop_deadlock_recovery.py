"""L1-CLD U5 — cross-loop deadlock recovery integration test (no network).

The regression capstone. A scripted fake IB injects the exact killer from the
2026-07-08 autopsy: a contract-details / qualify round-trip whose response never
arrives (the socket reader is "starved"). This test proves the connection-owner
loop architecture makes that call:

  1. TIME OUT under a bounded deadline (interruptible — cancelled on the owner
     loop) instead of hanging forever;
  2. FREE its slot (the owner loop keeps serving);
  3. trip the reader-progress watchdog -> forced reconnect (BROKER_READER_STALLED);
  4. and a SUBSEQUENT call SUCCEEDS after recovery.

If this test hangs or fails, the old cross-loop deadlock is back.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from chad.execution.broker_loop import BROKER_READER_STALLED
from chad.execution.ibkr_adapter import (
    BrokerTimeoutError,
    IbkrAdapter,
    IbkrConfig,
)


# ---------------------------------------------------------------------------
# Scripted fake IB (dual sync+async surface, hang injection, reader modelling)
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


class _Conn:
    def __init__(self) -> None:
        self.hasData = _Ev()


class _Client:
    def __init__(self) -> None:
        self.conn = _Conn()


class _ScriptedIB:
    """A healthy qualify emits conn.hasData (models the response arriving ->
    reader progress). A hung qualify emits nothing and never returns (models the
    starved reader / cross-loop deadlock). A fresh connectAsync clears the hang
    (models the reconnect restoring a working connection)."""

    def __init__(self) -> None:
        self._connected = False
        self.client = _Client()
        self.updateEvent = _Ev()
        self.hang = False
        self.connect_count = 0
        self.qualify_ok_count = 0

    def isConnected(self) -> bool:
        return self._connected

    async def connectAsync(self, host, port, clientId, timeout=None, **kw):  # noqa: N803
        self.connect_count += 1
        self._connected = True
        self.hang = False  # a fresh connection is healthy again
        return self

    def disconnect(self):
        self._connected = False

    def managedAccounts(self):
        return []

    async def qualifyContractsAsync(self, *contracts):
        if self.hang:
            # The killer: reqContractDetails whose reply never comes. Only
            # cancellation (bounded-timeout) unwinds this — it must never wedge
            # the owner loop.
            await asyncio.Event().wait()
        # Healthy path: simulate the response arriving -> reader progress.
        self.client.conn.hasData.emit(b"contract-details-response")
        self.qualify_ok_count += 1
        return list(contracts)

    def qualifyContracts(self, *contracts):
        return list(contracts)


def _adapter(fake: _ScriptedIB, *, stall_timeout_s: float, interval_s: float) -> IbkrAdapter:
    adapter = IbkrAdapter(
        config=IbkrConfig(dry_run=True, enable_idempotency=False),
        ib_factory=lambda: fake,
    )
    adapter._reader_stall_timeout_s = stall_timeout_s
    adapter._reader_watchdog_interval_s = interval_s
    return adapter


# ---------------------------------------------------------------------------
# 1) Bounded timeout + slot recovery (watchdog disabled to isolate this axis)
# ---------------------------------------------------------------------------

def test_contract_details_hang_times_out_and_slot_recovers() -> None:
    fake = _ScriptedIB()
    adapter = _adapter(fake, stall_timeout_s=1000.0, interval_s=0.05)  # watchdog inert
    try:
        adapter.ensure_connected(force=True)
        assert adapter._owner_loop_homed

        # Healthy qualify works.
        assert adapter._broker_call(fake, "qualifyContracts", "OK1", label="q") == ["OK1"]

        # Inject the hang. A bounded call MUST time out (interruptible), not hang.
        fake.hang = True
        t0 = time.monotonic()
        with pytest.raises(BrokerTimeoutError):
            adapter._broker_call(fake, "qualifyContracts", "HANG", timeout_s=0.3, label="hang")
        assert time.monotonic() - t0 < 3.0, "hang was not bounded"

        # The owner loop is not wedged — the slot recovered.
        assert adapter._broker_loop.pending_calls() == 0

        # After recovery a subsequent call succeeds (owner loop still serving).
        fake.hang = False
        assert adapter._broker_call(fake, "qualifyContracts", "OK2", label="q") == ["OK2"]
    finally:
        adapter.shutdown()


# ---------------------------------------------------------------------------
# 2) Watchdog detects the stall, forces a reconnect, subsequent call SUCCEEDS
# ---------------------------------------------------------------------------

def test_reader_stall_forces_reconnect_then_call_succeeds(caplog) -> None:
    fake = _ScriptedIB()
    adapter = _adapter(fake, stall_timeout_s=0.4, interval_s=0.05)
    try:
        with caplog.at_level(logging.WARNING):  # capture broker_loop marker
            adapter.ensure_connected(force=True)
            connects_before = fake.connect_count  # == 1

            # Inject the hang, then keep a call PENDING on the owner loop so the
            # reader-progress watchdog engages (pending call + no reader
            # progress -> stall).
            fake.hang = True
            bg_result: list = []

            def _bg():
                try:
                    adapter._broker_call(fake, "qualifyContracts", "HANG", timeout_s=2.0, label="bg")
                except BaseException as exc:  # noqa: BLE001
                    bg_result.append(exc)

            bg = threading.Thread(target=_bg, daemon=True)
            bg.start()

            # The watchdog must force a reconnect within a few stall windows.
            deadline = time.monotonic() + 4.0
            while time.monotonic() < deadline and fake.connect_count == connects_before:
                time.sleep(0.05)
            assert fake.connect_count > connects_before, "watchdog did not force a reconnect"

            bg.join(timeout=5)
            # The reconnect restored a healthy connection -> a NEW call succeeds.
            assert adapter._broker_call(fake, "qualifyContracts", "AFTER", label="q") == ["AFTER"]

        assert any(BROKER_READER_STALLED in r.getMessage() for r in caplog.records), (
            "expected BROKER_READER_STALLED marker"
        )
    finally:
        adapter.shutdown()

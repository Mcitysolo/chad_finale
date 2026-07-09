"""Unit tests for chad/execution/broker_loop.py (L1-CLD U1).

Proves the connection-owner loop: normal submission, inner-exception
propagation, cancel-on-timeout of a hanging coroutine, that a timed-out call
frees the owner loop for subsequent work, and fail-closed BROKER_LOOP_DOWN
semantics before start / after stop.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from chad.execution.broker_loop import (
    BROKER_READER_STALLED,
    BrokerLoop,
    BrokerLoopDown,
    BrokerLoopTimeout,
)


class _Clock:
    """Manually-advanced monotonic clock injected as the watchdog's now_fn so
    reader-idle time is controlled deterministically (independent of the real
    check cadence)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def loop():
    bl = BrokerLoop(name="test-broker-loop")
    bl.start()
    try:
        yield bl
    finally:
        bl.stop(timeout_s=2.0)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_submit_coro_returns_result(loop: BrokerLoop) -> None:
    async def _ok():
        await asyncio.sleep(0)
        return 41 + 1

    assert loop.submit_coro(_ok(), timeout_s=2.0, label="ok") == 42


def test_submit_coro_reraises_inner_exception(loop: BrokerLoop) -> None:
    async def _boom():
        raise ValueError("inner")

    with pytest.raises(ValueError, match="inner"):
        loop.submit_coro(_boom(), timeout_s=2.0, label="boom")


def test_submit_call_runs_sync_on_owner_loop(loop: BrokerLoop) -> None:
    seen = {}

    def _sync(a, b):
        # Executes ON the owner loop thread.
        seen["thread"] = threading.current_thread().name
        return a * b

    assert loop.submit_call(_sync, 6, 7, timeout_s=2.0, label="mul") == 42
    assert seen["thread"] == "test-broker-loop"


def test_pending_calls_returns_to_zero(loop: BrokerLoop) -> None:
    async def _ok():
        return 1

    loop.submit_coro(_ok(), timeout_s=2.0)
    assert loop.pending_calls() == 0


# ---------------------------------------------------------------------------
# Cancel-on-timeout — THE core U1 guarantee
# ---------------------------------------------------------------------------

def test_timeout_cancels_hanging_coroutine(loop: BrokerLoop) -> None:
    cancelled = threading.Event()

    async def _hang():
        try:
            await asyncio.sleep(100)  # never resolves within the test
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(BrokerLoopTimeout):
        loop.submit_coro(_hang(), timeout_s=0.2, label="hang")

    # The coroutine must actually be cancelled on the owner loop (not merely
    # abandoned) — this is what makes the call interruptible.
    assert cancelled.wait(timeout=3.0), "hanging coroutine was never cancelled"
    assert loop.pending_calls() == 0


def test_owner_loop_free_after_timeout(loop: BrokerLoop) -> None:
    """A timed-out/cancelled call must not wedge the owner loop: a subsequent
    submission still completes (regression guard for run_until_complete-style
    wedging)."""

    async def _hang():
        await asyncio.sleep(100)

    with pytest.raises(BrokerLoopTimeout):
        loop.submit_coro(_hang(), timeout_s=0.2, label="hang")

    async def _ok():
        return "alive"

    assert loop.submit_coro(_ok(), timeout_s=2.0, label="after") == "alive"


# ---------------------------------------------------------------------------
# Fail-closed BROKER_LOOP_DOWN
# ---------------------------------------------------------------------------

def test_submit_before_start_fails_closed() -> None:
    bl = BrokerLoop(name="never-started")

    async def _ok():
        return 1

    coro = _ok()
    with pytest.raises(BrokerLoopDown):
        bl.submit_coro(coro, timeout_s=1.0, label="pre_start")
    # The un-submitted coroutine must be closed (no 'never awaited' warning).
    assert coro.cr_running is False


def test_submit_after_stop_fails_closed() -> None:
    bl = BrokerLoop(name="stopped")
    bl.start()
    assert bl.is_alive()
    bl.stop(timeout_s=2.0)
    assert not bl.is_alive()

    async def _ok():
        return 1

    with pytest.raises(BrokerLoopDown):
        bl.submit_coro(_ok(), timeout_s=1.0, label="post_stop")


def test_fail_closed_marker_logged(caplog) -> None:
    bl = BrokerLoop(name="marker")

    async def _ok():
        return 1

    import logging

    with caplog.at_level(logging.ERROR, logger="chad.execution.broker_loop"):
        with pytest.raises(BrokerLoopDown):
            bl.submit_coro(_ok(), timeout_s=1.0)
    assert any("BROKER_LOOP_DOWN" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_is_idempotent() -> None:
    bl = BrokerLoop(name="idem")
    try:
        bl.start()
        t1 = bl._thread
        bl.start()
        assert bl._thread is t1  # same thread, not re-created
        assert bl.is_alive()
    finally:
        bl.stop(timeout_s=2.0)


def test_stop_is_idempotent() -> None:
    bl = BrokerLoop(name="idem-stop")
    bl.start()
    bl.stop(timeout_s=2.0)
    bl.stop(timeout_s=2.0)  # must not raise
    assert not bl.is_alive()


# ---------------------------------------------------------------------------
# U2 — reader-progress watchdog + forced reconnect + BROKER_READER_STALLED
# ---------------------------------------------------------------------------

def test_watchdog_reconnects_on_reader_stall(loop: BrokerLoop) -> None:
    clock = _Clock()
    reconnected = threading.Event()

    async def _reconnect():
        reconnected.set()

    loop.attach_watchdog(
        reconnect=_reconnect,
        stall_timeout_s=0.5,
        interval_s=0.02,
        pending_provider=lambda: 1,  # a call is in flight
        now_fn=clock,
    )
    # A call is pending and the reader makes no progress -> stall.
    clock.t = 5.0
    assert reconnected.wait(3.0), "watchdog did not force a reconnect on stall"


def test_watchdog_no_reconnect_when_reader_progresses(loop: BrokerLoop) -> None:
    clock = _Clock()
    reconnected = threading.Event()

    async def _reconnect():
        reconnected.set()

    loop.attach_watchdog(
        reconnect=_reconnect,
        stall_timeout_s=0.5,
        interval_s=0.02,
        pending_provider=lambda: 1,
        now_fn=clock,
    )
    # A call is pending, but the reader keeps advancing (mark_reader_progress
    # in lockstep with the clock) -> never a stall.
    for _ in range(30):
        clock.t += 0.1
        loop.mark_reader_progress()
        time.sleep(0.02)
    assert not reconnected.is_set(), "healthy reader must not trigger reconnect"


def test_watchdog_no_reconnect_when_no_calls_pending(loop: BrokerLoop) -> None:
    clock = _Clock()
    reconnected = threading.Event()

    async def _reconnect():
        reconnected.set()

    loop.attach_watchdog(
        reconnect=_reconnect,
        stall_timeout_s=0.5,
        interval_s=0.02,
        pending_provider=lambda: 0,  # nothing in flight
        now_fn=clock,
    )
    clock.t = 100.0  # reader long-idle, but no pending call
    time.sleep(0.3)
    assert not reconnected.is_set(), "idle reader with no pending call is normal"


def test_watchdog_marker_logged(loop: BrokerLoop, caplog) -> None:
    import logging

    clock = _Clock()
    reconnected = threading.Event()

    async def _reconnect():
        reconnected.set()

    with caplog.at_level(logging.ERROR, logger="chad.execution.broker_loop"):
        loop.attach_watchdog(
            reconnect=_reconnect,
            stall_timeout_s=0.3,
            interval_s=0.02,
            pending_provider=lambda: 1,
            now_fn=clock,
        )
        clock.t = 9.0
        assert reconnected.wait(3.0)
    assert any(BROKER_READER_STALLED in r.getMessage() for r in caplog.records)


def test_watchdog_does_not_storm_reconnects(loop: BrokerLoop) -> None:
    clock = _Clock()
    count = {"n": 0}
    lock = threading.Lock()

    async def _reconnect():
        with lock:
            count["n"] += 1

    loop.attach_watchdog(
        reconnect=_reconnect,
        stall_timeout_s=0.5,
        interval_s=0.02,
        pending_provider=lambda: 1,
        now_fn=clock,
    )
    # One stall episode -> exactly one reconnect; the clock reset after the
    # reconnect keeps it from firing every tick.
    clock.t = 5.0
    time.sleep(0.5)
    with lock:
        assert count["n"] == 1
    # A fresh stall window elapses -> a second, separate reconnect.
    clock.t = 10.0
    time.sleep(0.3)
    with lock:
        assert count["n"] == 2

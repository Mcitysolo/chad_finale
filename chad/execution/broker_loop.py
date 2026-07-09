from __future__ import annotations

"""
chad/execution/broker_loop.py

Connection-owner event-loop thread — root fix for the cross-loop deadlock
(autopsy 2026-07-08; PA ops/pending_actions/L1_CLD_cross_loop_deadlock_fix_2026-07-08.md).

The bug
-------
ib_async binds a connection's socket transport and its incoming-data reader to
whichever event loop is running when ``connectAsync`` executes. CHAD connected
the shared IB on MainThread (``chad/core/live_loop.py`` sync ``ib.connect``) and
then dispatched broker calls onto *other* loops — the per-worker persistent
loops in ``chad/execution/broker_executor.py``. ib_async's sync surface resolves
the loop per call via ``util.getLoop()`` (installed 2.1.0:
``site-packages/ib_async/util.py:484``), which returns the *calling thread's*
loop and is not cached. A request/response call (``qualifyContracts`` ->
``reqContractDetailsAsync``) therefore issues its request on a worker loop while
the reply is delivered by the reader on MainThread's *un-pumped* loop — the
await never completes, the worker is pinned uninterruptibly, and the pool
saturates. ``isConnected()`` stays True the whole time.

The fix (Option A — connection-owner loop)
------------------------------------------
ONE dedicated daemon thread runs a single asyncio loop via ``run_forever()``.
The IB object is created and ``connectAsync``'d ON that loop, so the reader runs
continuously and every response is processed. All broker work is submitted as a
coroutine via :func:`BrokerLoop.submit_coro`, which wraps
``asyncio.run_coroutine_threadsafe`` and blocks the caller on the returned
``concurrent.futures.Future`` with a hard wall-clock timeout. On timeout the
coroutine is CANCELLED on the owner loop (``fut.cancel()``), so no broker call
can ever be uninterruptible again — the cancellation is actually delivered
because the owner loop keeps running (unlike a stopped ``run_until_complete``
loop).

Fail-closed: if the owner thread/loop is not alive, every submission fails fast
with :class:`BrokerLoopDown` and the ``BROKER_LOOP_DOWN`` marker — no silent
hang, no auto trade-through.

U2 extends this module with a reader-progress watchdog (``BROKER_READER_STALLED``).
"""

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Awaitable, Callable, Optional

LOGGER = logging.getLogger("chad.execution.broker_loop")

# journald-observable markers.
BROKER_LOOP_DOWN = "BROKER_LOOP_DOWN"

# Default wall-clock cap for a single owner-loop submission.
_DEFAULT_SUBMIT_TIMEOUT_S = 10.0
# How long start() waits for the loop thread to actually begin running.
_DEFAULT_START_TIMEOUT_S = 5.0


class BrokerLoopError(Exception):
    """Base for owner-loop failures. Intentionally NOT a subclass of
    :class:`TimeoutError` so callers can catch it unambiguously and it never
    collides with a builtin/asyncio ``TimeoutError`` raised by the inner coro."""


class BrokerLoopDown(BrokerLoopError):
    """The connection-owner loop/thread is not alive; the submission failed fast
    (marker ``BROKER_LOOP_DOWN``) instead of hanging."""


class BrokerLoopTimeout(BrokerLoopError):
    """A submitted coroutine did not complete within its wall-clock deadline;
    the coroutine was cancelled on the owner loop."""


def _close_coro(coro: Any) -> None:
    """Close an un-submitted coroutine so Python does not warn 'coroutine was
    never awaited'. No-op for anything without a ``close`` method."""
    try:
        close = getattr(coro, "close", None)
        if callable(close):
            close()
    except BaseException:  # noqa: BLE001 - best-effort cleanup only
        pass


def _log_marker(marker: str, label: str, detail: str) -> None:
    LOGGER.error(
        marker,
        extra={"marker": marker, "label": label, "detail": detail},
    )


class BrokerLoop:
    """A single daemon thread owning one asyncio event loop (``run_forever``).

    Lifecycle: :meth:`start` (idempotent) -> :meth:`submit_coro` /
    :meth:`submit_call` -> :meth:`stop`. :meth:`is_alive` reports health.

    Thread-safe: :meth:`submit_coro` may be called concurrently from any thread
    (``run_coroutine_threadsafe`` marshals onto the owner loop).
    """

    def __init__(
        self,
        *,
        name: str = "chad-broker-loop",
        start_timeout_s: float = _DEFAULT_START_TIMEOUT_S,
    ) -> None:
        self._name = name
        self._start_timeout_s = float(start_timeout_s)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lifecycle_lock = threading.RLock()
        # Pending-call accounting (used by the U2 reader-progress watchdog:
        # a stall only matters while at least one call is in flight).
        self._pending_lock = threading.Lock()
        self._pending_calls = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "BrokerLoop":
        """Start the owner thread + loop. Idempotent; blocks until the loop is
        actually running (or raises :class:`BrokerLoopDown` on start timeout)."""
        with self._lifecycle_lock:
            if self._started and self._thread is not None and self._thread.is_alive():
                return self
            loop = asyncio.new_event_loop()
            ready = threading.Event()
            thread = threading.Thread(
                target=self._run_forever,
                args=(loop, ready),
                name=self._name,
                daemon=True,
            )
            self._loop = loop
            self._thread = thread
            thread.start()
            if not ready.wait(timeout=self._start_timeout_s):
                self._started = False
                _log_marker(BROKER_LOOP_DOWN, self._name, "start_timeout")
                raise BrokerLoopDown(
                    f"broker owner loop {self._name!r} failed to start within "
                    f"{self._start_timeout_s}s"
                )
            self._started = True
            return self

    def _run_forever(
        self, loop: asyncio.AbstractEventLoop, ready: threading.Event
    ) -> None:
        asyncio.set_event_loop(loop)
        # Signal readiness only once the loop is actually turning.
        loop.call_soon(ready.set)
        try:
            loop.run_forever()
        finally:
            # Cancel and drain any tasks still pending, then close the loop so
            # file descriptors are released (never leaks a loop).
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except BaseException:  # noqa: BLE001 - best-effort drain
                pass
            finally:
                try:
                    loop.close()
                except BaseException:  # noqa: BLE001
                    pass

    def stop(self, *, timeout_s: float = 5.0) -> None:
        """Stop the loop and join the thread. Idempotent."""
        with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            loop = self._loop
            thread = self._thread
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        if thread is not None:
            thread.join(timeout=timeout_s)

    def is_alive(self) -> bool:
        """True iff the owner thread is alive and its loop is running."""
        thread = self._thread
        loop = self._loop
        try:
            return bool(
                self._started
                and thread is not None
                and thread.is_alive()
                and loop is not None
                and not loop.is_closed()
                and loop.is_running()
            )
        except BaseException:  # noqa: BLE001
            return False

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._loop

    def pending_calls(self) -> int:
        with self._pending_lock:
            return self._pending_calls

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_coro(
        self,
        coro: Awaitable[Any],
        *,
        timeout_s: Optional[float] = _DEFAULT_SUBMIT_TIMEOUT_S,
        label: str = "broker_coro",
    ) -> Any:
        """Run ``coro`` on the owner loop and block until it completes.

        Returns the coroutine's result. Re-raises any exception it raises
        unchanged. Raises :class:`BrokerLoopTimeout` (after CANCELLING the coro
        on the owner loop) if it does not finish within ``timeout_s``. Raises
        :class:`BrokerLoopDown` (marker ``BROKER_LOOP_DOWN``) — WITHOUT hanging —
        if the owner loop is not alive.
        """
        if not self.is_alive():
            _close_coro(coro)
            _log_marker(BROKER_LOOP_DOWN, label, "loop_not_alive")
            raise BrokerLoopDown(
                f"broker owner loop is down; refusing {label!r} (fail-closed)"
            )

        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]
        except RuntimeError as exc:
            # Loop was closed/stopped between the is_alive() check and here.
            _close_coro(coro)
            _log_marker(BROKER_LOOP_DOWN, label, f"submit_failed:{exc}")
            raise BrokerLoopDown(
                f"broker owner loop unavailable for {label!r}: {exc}"
            ) from exc

        self._incr_pending()
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            # Bounded deadline hit. Cancel the coroutine ON the owner loop so it
            # is never left uninterruptible — the owner loop is run_forever, so
            # the cancellation is actually delivered and the task unwinds.
            future.cancel()
            raise BrokerLoopTimeout(
                f"broker coro {label!r} exceeded {timeout_s}s deadline "
                f"(cancelled on owner loop)"
            ) from exc
        except concurrent.futures.CancelledError as exc:
            raise BrokerLoopTimeout(
                f"broker coro {label!r} was cancelled on the owner loop"
            ) from exc
        finally:
            self._decr_pending()

    def submit_call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        timeout_s: Optional[float] = _DEFAULT_SUBMIT_TIMEOUT_S,
        label: str = "broker_call",
    ) -> Any:
        """Run a *synchronous* ``fn(*args)`` ON the owner loop (e.g. ib_async's
        sync ``placeOrder`` / ``openTrades``, which must touch the IB object on
        its owning loop). Same timeout / fail-closed semantics as
        :meth:`submit_coro`."""

        async def _wrap() -> Any:
            return fn(*args)

        return self.submit_coro(_wrap(), timeout_s=timeout_s, label=label)

    # ------------------------------------------------------------------
    # Pending-call accounting
    # ------------------------------------------------------------------

    def _incr_pending(self) -> None:
        with self._pending_lock:
            self._pending_calls += 1

    def _decr_pending(self) -> None:
        with self._pending_lock:
            if self._pending_calls > 0:
                self._pending_calls -= 1

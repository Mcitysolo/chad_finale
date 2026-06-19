from __future__ import annotations

"""
chad/execution/broker_executor.py

Shared bounded broker-call executor — root fix for Bug A (L-01 / L-02): the
per-call event-loop / file-descriptor leak behind the 2026-05-30 fd-exhaustion
(Errno 24) freeze.

The bug
-------
The previous ``_call_with_timeout`` (in both ibkr_adapter.py and
ibkr_trade_router.py) minted a brand-new daemon thread per broker call and ran
the ib_async *sync* API inside it. ib_async's sync surface routes through
``util.run()`` -> ``util.getLoop()``; in a thread with no installed event loop
that helper calls ``asyncio.new_event_loop() + set_event_loop()`` and never
closes the loop. So:

  * every broker call minted one fresh, never-closed event loop, reclaimed only
    by nondeterministic GC after the worker thread died; and
  * a TIMED-OUT call left the worker thread *alive forever* (abandoned on
    ``Thread.join(timeout)``), so its event loop (and ~3 fds) leaked
    permanently — the dangerous mode under a submission/timeout storm.

The fix (PA: ops/pending_actions/L1_bug_a_event_loop_leak_2026-06-04.md §4)
--------------------------------------------------------------------------
ONE module-level ``ThreadPoolExecutor(max_workers=4)``. A worker-initializer
installs exactly ONE persistent event loop per worker thread, once, for the
process lifetime (loops are never closed; daemon workers are reaped at process
exit). Because each worker thread has a loop installed via ``set_event_loop``,
``util.getLoop()`` returns that persistent loop (via
``get_event_loop_policy().get_event_loop()``) and ``util.run()`` reuses it
instead of minting a fresh one. Live-loop count becomes constant
(~max_workers + 1) regardless of broker-call volume; the 05-30 incident class
becomes structurally impossible.

Deliberate semantic change (PA §4): if all workers are busy, a new call cannot
be serviced within its timeout and fails fast — correct fail-closed behaviour
against a dead/slow gateway, but a change from "always mint a fresh thread".
This saturation case emits the ``BUG_A_POOL_SATURATED`` log marker so the
fail-fast mode is journald-observable rather than silent.

Call-site contract is preserved by the thin per-file wrappers
(``ibkr_adapter._call_with_timeout`` raising ``BrokerTimeoutError``,
``ibkr_trade_router._call_with_timeout`` raising ``TimeoutError``): this module
raises the neutral :class:`BrokerCallTimeout`, which each wrapper translates to
its existing exception type — ZERO call-site signature changes.
"""

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Callable

LOGGER = logging.getLogger("chad.execution.broker_executor")

# Pinned per PA §4. Bounds the live event-loop / fd count to max_workers + 1.
_MAX_WORKERS = 4

# journald-observable marker for the fail-closed pool-saturation path (PA §4).
BUG_A_POOL_SATURATED = "BUG_A_POOL_SATURATED"


class BrokerCallTimeout(Exception):
    """Raised by :func:`call_with_timeout` when a broker call does not complete
    within its wall-clock deadline.

    Covers BOTH a genuinely slow call (a worker started the call but it ran past
    the deadline) and pool saturation (no worker was free to start it within the
    deadline — see ``started`` in the message and the ``BUG_A_POOL_SATURATED``
    marker). Intentionally NOT a subclass of :class:`TimeoutError`, so it can be
    caught unambiguously by the per-file wrappers and never collides with a
    builtin/asyncio ``TimeoutError`` raised by the inner call itself.
    """


def _init_broker_worker() -> None:
    """Worker-thread initializer: install ONE persistent event loop per worker.

    Runs once per worker thread at pool creation. The loop is intentionally
    never closed — it lives for the process lifetime and is reaped when the
    daemon worker is torn down at interpreter exit. ib_async's sync
    ``util.getLoop()`` returns this installed loop (via the event-loop policy),
    so every ``util.run()`` executed on this worker reuses it instead of minting
    and leaking a fresh loop per broker call.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Module-level singleton: created at import, daemon workers, never shut down /
# never recreated (process exit reaps the workers). Do NOT call .shutdown().
_BROKER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="chad-broker",
    initializer=_init_broker_worker,
)


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_s: float,
    label: str = "broker_call",
) -> Any:
    """Run ``fn(*args)`` on the shared bounded broker executor with a hard
    wall-clock timeout.

    Returns ``fn``'s result. Re-raises any exception ``fn`` raises unchanged
    (including a builtin/asyncio ``TimeoutError`` raised by the call itself).
    Raises :class:`BrokerCallTimeout` only when the submitted call does not
    complete within ``timeout_s`` — i.e. the genuine broker-deadline path. On
    pool saturation (no worker ever picked up the call) it additionally logs the
    ``BUG_A_POOL_SATURATED`` marker before raising.
    """
    started = threading.Event()

    def _run() -> Any:
        started.set()
        return fn(*args)

    future = _BROKER_EXECUTOR.submit(_run)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        # NOTE: in Python 3.11+ concurrent.futures.TimeoutError IS builtin
        # TimeoutError, which is also what asyncio raises — so an inner call
        # raising its own TimeoutError lands here too. Disambiguate via
        # future.done(): if it is done, the timeout came from fn (or fn just
        # finished in the race window) -> propagate the real result/exception
        # unchanged. Only a NOT-done future is the true broker-deadline path.
        if future.done():
            return future.result()
        # True deadline timeout. The worker keeps running the call to completion
        # (uninterruptible) but on its PERSISTENT loop — no new loop, no leak;
        # the worker returns to the pool when the call finishes.
        future.cancel()  # no-op if running; removes it if still queued
        if not started.is_set():
            # The call never began executing => every worker was busy. Fail
            # closed and make the saturation observable (PA §4).
            LOGGER.error(
                BUG_A_POOL_SATURATED,
                extra={
                    "marker": BUG_A_POOL_SATURATED,
                    "label": label,
                    "timeout_s": timeout_s,
                    "max_workers": _MAX_WORKERS,
                    "failure_class": "TIMEOUT",
                },
            )
        raise BrokerCallTimeout(
            f"Broker call {label!r} exceeded {timeout_s}s liveness deadline "
            f"(started={started.is_set()}) — failure_class=TIMEOUT"
        )

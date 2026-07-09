from __future__ import annotations

"""
chad/execution/broker_executor.py

Bounded broker-call admission gate.

History
-------
This module was introduced as the Bug-A (L-01 / L-02) fix: a shared
``ThreadPoolExecutor(max_workers=4)`` whose worker-initializer installed ONE
persistent event loop per worker so ib_async's sync ``util.getLoop()`` reused it
instead of minting (and leaking on timeout) a fresh loop per call.

L1-CLD change (this revision)
-----------------------------
With the connection-owner-loop fix (chad/execution/broker_loop.py), real
ib_async coroutine calls no longer run their sync surface on these worker
threads — they are marshalled onto the single connection-owner loop, where a
timeout CANCELS the coroutine. This module therefore RETIRES the per-worker
persistent event-loop initializer (the old lines 88-89) and becomes a plain
bounded-concurrency admission gate:

  * A ``Semaphore(_MAX_INFLIGHT)`` bounds concurrent in-flight calls. The
    ``BUG_A_POOL_SATURATED`` marker now means "this many calls already in
    flight" (not "this many dead workers"); ``started=`` telemetry still
    distinguishes an admission-wait/saturation (started=False) from an admitted
    call that ran slow (started=True).
  * A timed-out call FREES its admission slot immediately (``future.cancel()`` +
    semaphore release) even though the abandoned worker may run on — THE fix for
    the "4 hung workers wedge the pool forever" failure mode.

Bug-A non-regression
--------------------
Threads are still REUSED (a pooled ``ThreadPoolExecutor``), so any residual
synchronous ib_async caller (e.g. ibkr_trade_router — not migrated in L1-CLD)
still gets ONE lazily-created, reused event loop per thread via
``util.getLoop()`` (util 2.1.0: creates + ``set_event_loop`` on first miss,
reused thereafter) — never a fresh loop per call. Removing the eager
initializer makes loop creation lazy, not per-call. The pool is sized with
headroom over ``_MAX_INFLIGHT`` so a freed-but-abandoned timed-out call lingers
on a spare thread rather than starving admission.

Public contract (unchanged)
----------------------------
``call_with_timeout(fn, *args, timeout_s, label)`` returns ``fn``'s result,
re-raises ``fn``'s exceptions unchanged, and raises the neutral
:class:`BrokerCallTimeout` on a deadline/saturation. The per-file wrappers
translate that to their existing exception types (adapter ->
``BrokerTimeoutError``; router -> ``TimeoutError``).
"""

import concurrent.futures
import logging
import threading
import time
from typing import Any, Callable

LOGGER = logging.getLogger("chad.execution.broker_executor")

# Max concurrent in-flight broker calls (admission bound).
_MAX_INFLIGHT = 4
# Backward-compat alias — some tooling still references the old name; it now
# denotes the in-flight bound, not a fixed worker count.
_MAX_WORKERS = _MAX_INFLIGHT

# Headroom over the in-flight bound so a timed-out/abandoned call (whose worker
# keeps running to completion) lingers on a spare pool thread instead of holding
# an admission slot.
_EXECUTOR_MAX_WORKERS = _MAX_INFLIGHT * 8

# journald-observable marker for the fail-closed saturation path.
BUG_A_POOL_SATURATED = "BUG_A_POOL_SATURATED"


class BrokerCallTimeout(Exception):
    """Raised by :func:`call_with_timeout` when a broker call does not complete
    within its wall-clock deadline.

    Covers BOTH a genuinely slow admitted call (a worker started it but it ran
    past the deadline; ``started=True``) and pool saturation (no admission slot
    was free within the deadline; ``started=False`` + the
    ``BUG_A_POOL_SATURATED`` marker). Intentionally NOT a subclass of
    :class:`TimeoutError`, so it can be caught unambiguously by the per-file
    wrappers and never collides with a builtin/asyncio ``TimeoutError`` raised
    by the inner call itself.
    """


# Admission gate bounding concurrent in-flight calls.
_INFLIGHT = threading.Semaphore(_MAX_INFLIGHT)

# Reused thread pool WITHOUT a per-worker loop initializer (see module docstring
# for the Bug-A non-regression argument). Module-level singleton; never
# .shutdown() (daemon workers are reaped at interpreter exit).
_BROKER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_EXECUTOR_MAX_WORKERS,
    thread_name_prefix="chad-broker",
)


def _log_saturation(label: str, timeout_s: float, failure_class: str) -> None:
    LOGGER.error(
        BUG_A_POOL_SATURATED,
        extra={
            "marker": BUG_A_POOL_SATURATED,
            "label": label,
            "timeout_s": timeout_s,
            "max_inflight": _MAX_INFLIGHT,
            "failure_class": failure_class,
            "started": False,
        },
    )


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout_s: float,
    label: str = "broker_call",
) -> Any:
    """Run ``fn(*args)`` under the bounded admission gate with a hard wall-clock
    timeout.

    Returns ``fn``'s result. Re-raises any exception ``fn`` raises unchanged.
    Raises :class:`BrokerCallTimeout` when no admission slot frees within
    ``timeout_s`` (saturation; logs ``BUG_A_POOL_SATURATED``, ``started=False``)
    or when an admitted call exceeds ``timeout_s`` (``started=True``). A
    timed-out call frees its admission slot immediately.
    """
    deadline = time.monotonic() + float(timeout_s)

    # --- Admission: bound concurrent in-flight calls (fail-closed) -----------
    if not _INFLIGHT.acquire(timeout=max(0.0, float(timeout_s))):
        _log_saturation(label, timeout_s, failure_class="SATURATED")
        raise BrokerCallTimeout(
            f"Broker call {label!r} could not be admitted within {timeout_s}s — "
            f"pool saturated ({_MAX_INFLIGHT} in-flight) (started=False) — "
            f"failure_class=SATURATED"
        )

    released = threading.Event()

    def _release_slot() -> None:
        # Release the admission slot exactly once, regardless of path.
        if not released.is_set():
            released.set()
            _INFLIGHT.release()

    started = threading.Event()

    def _run() -> Any:
        started.set()
        return fn(*args)

    future = _BROKER_EXECUTOR.submit(_run)
    try:
        remaining = max(0.0, deadline - time.monotonic())
        return future.result(timeout=remaining)
    except concurrent.futures.TimeoutError:
        # NOTE: in Python 3.11+ concurrent.futures.TimeoutError IS builtin
        # TimeoutError (also what asyncio raises), so an inner call raising its
        # own TimeoutError lands here too. Disambiguate via future.done(): if it
        # is done, the timeout came from fn (or fn just finished in the race
        # window) -> propagate the real result/exception unchanged.
        if future.done():
            return future.result()
        # True deadline timeout. The worker keeps running the call to completion
        # (abandoned) on its REUSED pool thread (persistent loop, no leak) and
        # returns to the pool when it finishes. Its admission slot is freed NOW
        # (in the finally) so it never wedges the pool.
        future.cancel()  # no-op if running; removes it if still queued
        if not started.is_set():
            # Admitted but never began (executor backlog) — rare given the
            # headroom pool; surface it as saturation for observability.
            _log_saturation(label, timeout_s, failure_class="TIMEOUT")
        raise BrokerCallTimeout(
            f"Broker call {label!r} exceeded {timeout_s}s liveness deadline "
            f"(started={started.is_set()}) — failure_class=TIMEOUT"
        )
    finally:
        _release_slot()

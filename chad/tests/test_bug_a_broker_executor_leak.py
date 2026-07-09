"""
Broker executor — Bug-A non-regression + L1-CLD U4 admission-model proof tests.

broker_executor.py was reworked for L1-CLD (connection-owner loop): the
per-worker persistent event-loop initializer is retired and the module becomes a
bounded-concurrency admission gate (Semaphore + reused thread pool). These tests
prove:

  (a) BUG-A NON-REGRESSION: ~200 sequential calls that each exercise the
      ib_async sync loop path do NOT grow the live event-loop count per call
      (loops stay bounded by the small pool, not ~N) and the fd count is stable.
      Removing the eager initializer made loop creation lazy, not per-call
      (threads are reused; util.getLoop reuses each thread's loop).
  (b) OLD per-call impl leaks a loop per timeout (the original bug), shown for
      contrast.
  (c) THE REGRESSION TEST: a timed-out call FREES its admission slot even though
      its worker is abandoned — filling every slot with timing-out calls still
      leaves the gate open for a subsequent call.
  (d) SATURATION: with all _MAX_INFLIGHT slots held, the next concurrent call
      fails fast with BrokerCallTimeout(started=False) + BUG_A_POOL_SATURATED.
  (e) inner exceptions propagate unchanged; timed-out slots recover.

All tests release held slots in `finally` so the shared gate returns to idle.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import threading

import pytest

import ib_async.util as ib_util

import chad.execution.broker_executor as bx
from chad.execution.broker_executor import BrokerCallTimeout, call_with_timeout


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _open_loop_count() -> int:
    gc.collect()
    n = 0
    for obj in gc.get_objects():
        if isinstance(obj, asyncio.AbstractEventLoop):
            try:
                if not obj.is_closed():
                    n += 1
            except Exception:
                pass
    return n


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _broker_worker_thread_count() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("chad-broker"))


def _make_slow_fn(stop: threading.Event):
    async def _slow_coro():
        while not stop.is_set():
            await asyncio.sleep(0.02)
        return "released"

    def _slow_fn():
        return ib_util.run(_slow_coro())

    return _slow_fn


def _quick_fn():
    async def _coro():
        await asyncio.sleep(0)
        return 7

    return ib_util.run(_coro())


def _call_in_thread(fn, *, timeout_s, label, sink: list) -> threading.Thread:
    """Run call_with_timeout on a background thread, recording ('ok', result) or
    ('err', exc) into ``sink``."""

    def _target():
        try:
            sink.append(("ok", call_with_timeout(fn, timeout_s=timeout_s, label=label)))
        except BaseException as exc:  # noqa: BLE001
            sink.append(("err", exc))

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# (a) Bug-A non-regression: no per-call loop/fd growth
# ---------------------------------------------------------------------------

def test_a_sequential_calls_do_not_grow_loops_per_call():
    # Warm the pool so first-touch lazy loop creation is not counted as growth.
    for _ in range(5):
        assert call_with_timeout(_quick_fn, timeout_s=5.0, label="warmup") == 7

    base_loops = _open_loop_count()
    base_fds = _open_fd_count()

    N = 200
    for _ in range(N):
        assert call_with_timeout(_quick_fn, timeout_s=5.0, label="seq") == 7

    loop_growth = _open_loop_count() - base_loops
    fd_growth = _open_fd_count() - base_fds
    print(f"\n(a) sequential N={N}: loop delta={loop_growth}, fd delta={fd_growth}")

    # The Bug-A signal: NOT ~N loops. Bounded by the small reused pool, not per
    # call. (Sequential submissions reuse threads, so this is typically ~0.)
    assert loop_growth <= bx._MAX_INFLIGHT, (
        f"event-loop count grew by {loop_growth} over {N} calls (per-call leak?)"
    )
    assert abs(fd_growth) <= 4, f"open-fd count moved by {fd_growth} over {N} calls"


# ---------------------------------------------------------------------------
# (b) OLD per-call impl leaks a loop per timeout (contrast)
# ---------------------------------------------------------------------------

def _old_call_with_timeout(fn, *args, timeout_s, label="broker_call"):
    result_box: list = []
    error_box: list = []

    def _target() -> None:
        try:
            result_box.append(fn(*args))
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"{label} timed out")
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def test_b_old_impl_leaks_loop_per_timeout():
    K = 3
    stop = threading.Event()
    slow_fn = _make_slow_fn(stop)
    try:
        base = _open_loop_count()
        for i in range(K):
            with pytest.raises(TimeoutError):
                _old_call_with_timeout(slow_fn, timeout_s=0.3, label=f"old_{i}")
        growth = _open_loop_count() - base
        print(f"\n(b) OLD per-call: loop delta={growth} over K={K} timeouts [leaks]")
        assert growth >= K, f"expected OLD code to leak >= {K} loops, saw {growth}"
    finally:
        stop.set()


# ---------------------------------------------------------------------------
# (c) THE REGRESSION TEST: a timed-out call frees its slot
# ---------------------------------------------------------------------------

def test_c_timed_out_call_frees_its_slot():
    """Fill EVERY admission slot with calls that time out. Each abandoned worker
    keeps running, but its slot is freed on timeout — so a subsequent call is
    still admitted (the old 4-hung-workers-wedge-the-pool bug is gone)."""
    stop = threading.Event()
    slow_fn = _make_slow_fn(stop)
    try:
        # _MAX_INFLIGHT concurrent calls, all of which will time out.
        sinks: list = [[] for _ in range(bx._MAX_INFLIGHT)]
        threads = [
            _call_in_thread(slow_fn, timeout_s=0.4, label=f"hang_{i}", sink=sinks[i])
            for i in range(bx._MAX_INFLIGHT)
        ]
        for t in threads:
            t.join(timeout=5)
        # All timed out (workers still abandoned, but slots released).
        outcomes = [s[0][0] for s in sinks if s]
        assert outcomes == ["err"] * bx._MAX_INFLIGHT
        assert all(isinstance(s[0][1], BrokerCallTimeout) for s in sinks)

        # THE assertion: the gate is open again despite 4 abandoned workers.
        assert call_with_timeout(lambda: 99, timeout_s=2.0, label="after") == 99
    finally:
        stop.set()
        # Let the abandoned workers drain their slow calls.
        assert call_with_timeout(_quick_fn, timeout_s=5.0, label="drain") == 7


# ---------------------------------------------------------------------------
# (d) Saturation: the (N+1)th concurrent call fails fast with the marker
# ---------------------------------------------------------------------------

def test_d_fifth_concurrent_call_saturates_with_marker(caplog):
    release = threading.Event()
    started = threading.Semaphore(0)

    def _block():
        started.release()
        release.wait(timeout=10)
        return "unblocked"

    sinks: list = [[] for _ in range(bx._MAX_INFLIGHT)]
    threads = [
        _call_in_thread(_block, timeout_s=30.0, label=f"hold_{i}", sink=sinks[i])
        for i in range(bx._MAX_INFLIGHT)
    ]
    try:
        # Every slot genuinely in flight (admitted + running).
        for _ in range(bx._MAX_INFLIGHT):
            assert started.acquire(timeout=5), "a holding call failed to start"

        with caplog.at_level(logging.ERROR, logger="chad.execution.broker_executor"):
            with pytest.raises(BrokerCallTimeout) as ei:
                call_with_timeout(lambda: 42, timeout_s=0.3, label="fifth")

        print(f"\n(d) saturation raised: {ei.value}")
        assert "started=False" in str(ei.value)
        assert bx.BUG_A_POOL_SATURATED in caplog.text
    finally:
        release.set()
        for t in threads:
            t.join(timeout=5)
        assert all(s and s[0][0] == "ok" for s in sinks)


# ---------------------------------------------------------------------------
# (e) contract: inner exceptions propagate unchanged; slots recover
# ---------------------------------------------------------------------------

def test_e_inner_exception_propagates_unchanged():
    class _Boom(Exception):
        pass

    def _fn():
        raise _Boom("inner")

    with pytest.raises(_Boom, match="inner"):
        call_with_timeout(_fn, timeout_s=1.0, label="boom")
    # Slot recovered after the exception.
    assert call_with_timeout(lambda: 5, timeout_s=1.0, label="after") == 5

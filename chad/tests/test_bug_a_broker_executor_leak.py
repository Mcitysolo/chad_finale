"""
Bug A (L-01 / L-02) — event-loop / fd leak root-fix proof tests.

PA: ops/pending_actions/L1_bug_a_event_loop_leak_2026-06-04.md (§7 unit/test plan).

These tests prove the shared bounded broker executor
(chad/execution/broker_executor.py) eliminates the per-call event-loop / fd
leak that the previous per-call daemon-thread `_call_with_timeout` exhibited:

  (a) ~200 sequential calls that each exercise the ib_async sync loop path
      (util.run) do NOT grow the live event-loop count or the open-fd count.
  (b) TIMEOUT path (the actual bug): the OLD per-call implementation grows the
      live-loop count by one per timeout (FAIL); the NEW shared-executor
      implementation stays flat (PASS) and raises the timeout exception while
      leaving no abandoned worker thread. Both numbers are reported.
  (c) POOL SATURATION: with every worker blocked, an extra call fails fast with
      BrokerCallTimeout and emits the BUG_A_POOL_SATURATED log marker.

All tests use the SHARED module-level executor and therefore release their held
workers in `finally` blocks so the pool is returned to idle for the rest of the
suite.
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
    """Count live (non-closed) asyncio event loops reachable by the GC.

    A leaked loop (minted in an abandoned thread and never closed) stays
    reachable via that still-running thread's frame, so it is counted here.
    """
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
    return sum(
        1 for t in threading.enumerate() if t.name.startswith("chad-broker")
    )


# ---------------------------------------------------------------------------
# A slow inner call that mints/uses an asyncio loop via the ib_async sync path
# and stays alive until explicitly released — the realistic timeout scenario.
# ---------------------------------------------------------------------------


def _make_slow_fn(stop: threading.Event):
    async def _slow_coro():
        # Yield repeatedly so the loop is actually running (and, in the OLD
        # path, the minted loop stays referenced by the live thread frame).
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


def _warm_all_workers():
    """Force every pool worker to exist (initializer ran -> loop installed), so
    a subsequent timeout test does not create NEW worker loops and can assert a
    flat (zero) delta."""
    release = threading.Event()
    started = threading.Semaphore(0)

    def _occupy():
        started.release()
        release.wait(timeout=10)

    held = [bx._BROKER_EXECUTOR.submit(_occupy) for _ in range(bx._MAX_WORKERS)]
    try:
        for _ in range(bx._MAX_WORKERS):
            assert started.acquire(timeout=5), "worker failed to start"
    finally:
        release.set()
        for h in held:
            h.result(timeout=5)


# ---------------------------------------------------------------------------
# (a) Sequential-volume: no loop / fd growth
# ---------------------------------------------------------------------------


def test_a_sequential_calls_do_not_grow_loops_or_fds():
    # Warm the pool so worker-loop creation is not counted as growth.
    for _ in range(5):
        assert call_with_timeout(_quick_fn, timeout_s=5.0, label="warmup") == 7

    base_loops = _open_loop_count()
    base_fds = _open_fd_count()

    N = 200
    for _ in range(N):
        assert call_with_timeout(_quick_fn, timeout_s=5.0, label="seq") == 7

    end_loops = _open_loop_count()
    end_fds = _open_fd_count()

    loop_growth = end_loops - base_loops
    fd_growth = end_fds - base_fds
    print(
        f"\n(a) sequential N={N}: loops {base_loops}->{end_loops} "
        f"(delta={loop_growth}), fds {base_fds}->{end_fds} (delta={fd_growth})"
    )

    # The real signal: zero live-loop accumulation across 200 calls.
    assert loop_growth == 0, f"event-loop count grew by {loop_growth} over {N} calls"
    # fd band (PA §8: small tolerance for pytest plugin fd churn).
    assert abs(fd_growth) <= 4, f"open-fd count moved by {fd_growth} over {N} calls"


# ---------------------------------------------------------------------------
# (b) Timeout path: OLD leaks a loop per timeout (FAIL) vs NEW flat (PASS)
# ---------------------------------------------------------------------------


def _old_call_with_timeout(fn, *args, timeout_s, label="broker_call"):
    """Verbatim re-creation of the PRE-FIX per-call daemon-thread implementation
    (the leak), used only to demonstrate the bug the fix removes."""
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


def test_b_timeout_path_old_leaks_new_flat():
    K = 3  # < max_workers so each NEW timeout ties up a distinct (already-warm) worker
    stop = threading.Event()
    slow_fn = _make_slow_fn(stop)

    old_threads_before = threading.active_count()
    try:
        # --- OLD implementation: one leaked loop (+ one leaked live thread) per timeout ---
        old_base_loops = _open_loop_count()
        for i in range(K):
            with pytest.raises(TimeoutError):
                _old_call_with_timeout(slow_fn, timeout_s=0.3, label=f"old_{i}")
        old_end_loops = _open_loop_count()
        old_growth = old_end_loops - old_base_loops
        old_threads_after = threading.active_count()

        # --- NEW implementation: flat loop count; no abandoned threads ---
        _warm_all_workers()
        new_base_loops = _open_loop_count()
        new_base_workers = _broker_worker_thread_count()
        for i in range(K):
            with pytest.raises(BrokerCallTimeout):
                call_with_timeout(slow_fn, timeout_s=0.3, label=f"new_{i}")
        new_end_loops = _open_loop_count()
        new_growth = new_end_loops - new_base_loops
        new_end_workers = _broker_worker_thread_count()

        print(
            f"\n(b) timeout path over K={K} timeouts:"
            f"\n    OLD per-call:  loops {old_base_loops}->{old_end_loops} (delta={old_growth})  [FAIL: leaks]"
            f"\n    OLD threads:   {old_threads_before}->{old_threads_after} (abandoned daemon threads linger)"
            f"\n    NEW executor:  loops {new_base_loops}->{new_end_loops} (delta={new_growth})  [PASS: flat]"
            f"\n    NEW workers:   {new_base_workers}->{new_end_workers} (bounded by max_workers={bx._MAX_WORKERS})"
        )

        # OLD: demonstrably leaks at least one live loop per timeout.
        assert old_growth >= K, (
            f"expected OLD code to leak >= {K} loops on timeout, saw {old_growth}"
        )
        # NEW: zero loop growth — the busy workers run on their persistent loops.
        assert new_growth == 0, (
            f"expected NEW code to leak 0 loops on timeout, saw {new_growth}"
        )
        # NEW: no new worker threads beyond the bounded pool; nothing abandoned.
        assert new_end_workers <= bx._MAX_WORKERS, (
            f"NEW worker-thread count {new_end_workers} exceeded max_workers"
        )
    finally:
        # Release every hung call (OLD leaked threads + NEW busy workers) so the
        # shared pool returns to idle for the rest of the suite.
        stop.set()
        # Give the freed NEW workers a moment to drain their slow calls.
        drain = call_with_timeout(_quick_fn, timeout_s=5.0, label="drain")
        assert drain == 7


# ---------------------------------------------------------------------------
# (c) Pool saturation: fail fast + BUG_A_POOL_SATURATED marker
# ---------------------------------------------------------------------------


def test_c_pool_saturation_fails_fast_and_logs_marker(caplog):
    release = threading.Event()
    started = threading.Semaphore(0)

    def _block():
        started.release()
        release.wait(timeout=10)

    held = [bx._BROKER_EXECUTOR.submit(_block) for _ in range(bx._MAX_WORKERS)]
    try:
        # Confirm all workers are actually running (not merely queued).
        for _ in range(bx._MAX_WORKERS):
            assert started.acquire(timeout=5), "worker failed to start"

        with caplog.at_level(logging.ERROR, logger="chad.execution.broker_executor"):
            with pytest.raises(BrokerCallTimeout) as ei:
                call_with_timeout(lambda: 42, timeout_s=0.3, label="saturation_probe")

        print(
            "\n(c) saturation: raised "
            f"{type(ei.value).__name__}: {ei.value}"
        )
        assert "started=False" in str(ei.value)  # the probe never began executing
        assert bx.BUG_A_POOL_SATURATED in caplog.text, (
            "expected BUG_A_POOL_SATURATED marker in logs"
        )
    finally:
        release.set()
        for h in held:
            h.result(timeout=5)

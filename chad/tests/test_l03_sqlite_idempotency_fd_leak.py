"""L-03 regression — _SQLiteIdempotencyStore must not leak fds per call.

Bug A / L-03: `_SQLiteIdempotencyStore` opened a fresh WAL sqlite connection
per call via `with self._connect() as conn:`. `with` on a sqlite connection
manages the *transaction*, not the *lifetime* — the connection (and its
db/-wal/-shm fds) was never closed, so the open-fd count grew under the live
order-submission hot path (audit BUG_A_LEAK_SURFACE_2026-06-09 Appendix A.1:
+102 fds / 1000 calls). The fix wraps each use-site in `contextlib.closing(...)`.

This test exercises ~500 sequential claim/get/mark cycles and asserts the
process open-fd count does not grow beyond a small tolerance band. It is a
genuine leak-detector: against the unfixed code it FAILS (fd count climbs by
tens), against the fixed code it PASSES (flat).

Linux-only (reads /proc/self/fd); CHAD runs on Linux exclusively.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chad.execution.ibkr_adapter import _SQLiteIdempotencyStore

_PROC_SELF_FD = "/proc/self/fd"


def _open_fd_count() -> int:
    return len(os.listdir(_PROC_SELF_FD))


@pytest.mark.skipif(
    not os.path.isdir(_PROC_SELF_FD),
    reason="fd-count assertion requires /proc/self/fd (Linux)",
)
def test_sqlite_idempotency_store_no_fd_growth(tmp_path):
    db = tmp_path / "exec_state_l03.sqlite3"
    store = _SQLiteIdempotencyStore(Path(db))
    now = datetime.now(timezone.utc)

    # Warm-up: the first calls create the WAL/-shm sidecar files and any
    # one-time fds. Exclude that setup from the measured window so we isolate
    # the *per-call* connection leak, not first-touch allocation.
    store.claim("warmup", {"k": "v"}, now)
    store.get("warmup")
    store.mark("warmup", status="filled", broker_order_id=1, result={"ok": True}, now=now)

    baseline = _open_fd_count()

    n = 500
    for i in range(n):
        key = f"key-{i}"
        store.claim(key, {"i": i}, now)
        store.get(key)
        store.mark(key, status="filled", broker_order_id=i, result={"i": i}, now=now)

    growth = _open_fd_count() - baseline
    assert growth <= 2, (
        f"open-fd count grew by {growth} over {n} claim/get/mark cycles "
        f"(tolerance +2); per-call SQLite connection is leaking (L-03)"
    )

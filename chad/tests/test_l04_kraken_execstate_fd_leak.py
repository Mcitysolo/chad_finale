"""L-04 regression — ExecStateStore must not leak fds per call.

Bug A / L-04: `chad.execution.kraken_executor.ExecStateStore` opened a fresh
WAL sqlite connection per call via `with self._connect() as con:`. `with` on a
sqlite connection manages the *transaction*, not the *lifetime* — the
connection (and its db/-wal/-shm fds) was never closed, so the open-fd count
grew under the live order-submission hot path. The store is
`isolation_level=None` (autocommit) and every method does its own explicit
`BEGIN IMMEDIATE; ... COMMIT;`, so the `with con:` commit-on-exit was already a
no-op — the fix wraps each of the 4 use-sites in `contextlib.closing(...)`,
which is purely additive (no commit-semantics change; close() still discards
any uncommitted transaction on the exception path).

This test exercises ~500 sequential claim/bump/mark_submitted/mark_error cycles
and asserts the process open-fd count does not grow beyond a small tolerance
band. It is a genuine leak-detector: against the unfixed code it FAILS (fd
count climbs by tens), against the fixed code it PASSES (flat).

Linux-only (reads /proc/self/fd); CHAD runs on Linux exclusively.
"""

import contextlib
import os
import sqlite3
from pathlib import Path

import pytest

from chad.execution.kraken_executor import ExecStateStore

_PROC_SELF_FD = "/proc/self/fd"

# Superset of the live `exec_state` schema that also carries the
# intent_canonical_json / extra_json columns the executor's claim() INSERT
# references. ExecStateStore assumes the table already exists (it does not
# self-provision), so the test creates it.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS exec_state (
    idempotency_key TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    broker TEXT,
    strategy TEXT,
    symbol TEXT,
    side TEXT,
    quantity REAL,
    asset_class TEXT,
    broker_order_id TEXT,
    intent_canonical_json TEXT,
    extra_json TEXT,
    claim_attempts INTEGER NOT NULL DEFAULT 0,
    submit_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
"""


def _open_fd_count() -> int:
    return len(os.listdir(_PROC_SELF_FD))


def _provision_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(sqlite3.connect(str(db_path))) as con:
        con.executescript(_SCHEMA)
        con.commit()


def _one_cycle(store: ExecStateStore, key: str) -> None:
    store.claim(
        idempotency_key=key,
        broker="kraken",
        strategy="leaktest",
        symbol="XBTUSD",
        side="buy",
        quantity=1.0,
        asset_class="CRYPTO",
        intent_canonical_json="{}",
        extra_json="{}",
    )
    store.bump_submit_attempt(idempotency_key=key)
    store.mark_submitted(idempotency_key=key, broker_order_id=f"TX-{key}")
    store.mark_error(idempotency_key=key, err="synthetic")


@pytest.mark.skipif(
    not os.path.isdir(_PROC_SELF_FD),
    reason="fd-count assertion requires /proc/self/fd (Linux)",
)
def test_execstate_store_no_fd_growth(tmp_path):
    db = tmp_path / "exec_state_l04.sqlite3"
    _provision_schema(db)
    store = ExecStateStore(Path(db))

    # Warm-up: the first calls create the WAL/-shm sidecar files and any
    # one-time fds. Exclude that setup from the measured window so we isolate
    # the *per-call* connection leak, not first-touch allocation.
    _one_cycle(store, "warmup")

    baseline = _open_fd_count()

    n = 500
    for i in range(n):
        _one_cycle(store, f"key-{i}")

    growth = _open_fd_count() - baseline
    assert growth <= 2, (
        f"open-fd count grew by {growth} over {n} "
        f"claim/bump/mark_submitted/mark_error cycles (tolerance +2); "
        f"per-call SQLite connection is leaking (L-04)"
    )

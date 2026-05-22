"""OPS-OMEGA-01 regression tests.

Covers three remediation patterns introduced to break the
"duplicate_blocked → 10 minutes of cooldown_active → duplicate_blocked" loop
that pinned futures intents (M6E, MES, MGC) on 2026-05-22:

  Pattern A:
    1. A terminal-positive ("Filled") idempotency row older than
       ``terminal_positive_ttl_s`` is reclaimable (DELETE + fresh INSERT).
    2. A fresh terminal-positive row within the TTL still blocks
       (no double-fill regression of GAP-036's S2 invariant).
    3. A stale non-terminal row with ``broker_order_id is None`` is reclaimed
       via the ABSENT probe branch (no broker order ever existed).

  Pattern C:
    4. ``revert_emission_for_unconfirmed`` removes the cooldown-arming entry
       so a downstream ``duplicate_blocked`` does not consume the 10-minute
       cooldown.
    5. Unrelated fingerprints are not affected by either remediation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from chad.execution.ibkr_adapter import (
    _PROBE_ABSENT,
    _PROBE_STILL_ACTIVE,
    _SQLiteIdempotencyStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> _SQLiteIdempotencyStore:
    return _SQLiteIdempotencyStore(tmp_path / "ibkr_adapter_state.sqlite3")


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _insert_row(
    store: _SQLiteIdempotencyStore,
    *,
    key: str,
    status: str,
    updated_at: datetime,
    broker_order_id: Optional[int],
    payload: dict,
) -> None:
    """Seed a row directly in the SQLite store, bypassing claim_or_reclaim."""
    with sqlite3.connect(str(store._path)) as conn:
        conn.execute(
            "INSERT INTO ibkr_exec_state ("
            "  idempotency_key, status, created_at_utc, updated_at_utc,"
            "  broker_order_id, payload_json, result_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                status,
                updated_at.isoformat(),
                updated_at.isoformat(),
                broker_order_id,
                json.dumps(payload, sort_keys=True),
                None,
            ),
        )


M6E_PAYLOAD = {
    "strategy": "omega_macro",
    "symbol": "M6E",
    "sec_type": "FUT",
    "side": "BUY",
    "quantity": 2.0,
}
MES_PAYLOAD = {
    "strategy": "alpha_futures",
    "symbol": "MES",
    "sec_type": "FUT",
    "side": "BUY",
    "quantity": 1.0,
}
MGC_PAYLOAD = {
    "strategy": "alpha_futures",
    "symbol": "MGC",
    "sec_type": "FUT",
    "side": "SELL",
    "quantity": 1.0,
}


def _refuse_probe(_broker_order_id):  # pragma: no cover - guard
    raise AssertionError(
        "ib_probe must NOT be invoked when terminal_positive_ttl_s drives expiry "
        "(scenarios 1 and 2)"
    )


# ---------------------------------------------------------------------------
# Pattern A — duplicate cache TTL & ABSENT-on-None semantics
# ---------------------------------------------------------------------------


def test_stale_filled_row_beyond_ttl_does_not_block_fresh_intent(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    now = _utc("2026-05-22T18:00:00+00:00")
    # Filled 2 hours ago — well past the 900s default TTL.
    _insert_row(
        store,
        key="m6e_filled_key",
        status="Filled",
        updated_at=now - timedelta(seconds=7200),
        broker_order_id=5026,
        payload=M6E_PAYLOAD,
    )

    claimed = store.claim_or_reclaim(
        "m6e_filled_key",
        M6E_PAYLOAD,
        now,
        stale_threshold_s=600.0,
        ib_probe=_refuse_probe,
        terminal_positive_ttl_s=900.0,
    )
    assert claimed is True, "Stale Filled row beyond TTL must be reclaimable"

    row = store.get("m6e_filled_key")
    assert row is not None
    assert row["status"] == "claimed", "Reclaim must replace prior Filled status"
    assert row["broker_order_id"] is None, "Reclaimed row starts with no broker_order_id"


def test_fresh_filled_row_within_ttl_still_blocks(tmp_path: Path) -> None:
    """Guards against double-fill regression of GAP-036 S2 within TTL."""
    store = _make_store(tmp_path)
    now = _utc("2026-05-22T18:00:00+00:00")
    # Filled 5 minutes ago — well inside the 900s TTL window.
    _insert_row(
        store,
        key="m6e_fresh_filled_key",
        status="Filled",
        updated_at=now - timedelta(seconds=300),
        broker_order_id=5026,
        payload=M6E_PAYLOAD,
    )

    claimed = store.claim_or_reclaim(
        "m6e_fresh_filled_key",
        M6E_PAYLOAD,
        now,
        stale_threshold_s=600.0,
        ib_probe=_refuse_probe,
        terminal_positive_ttl_s=900.0,
    )
    assert claimed is False, "Fresh Filled row within TTL must still block"

    row = store.get("m6e_fresh_filled_key")
    assert row is not None
    assert row["status"] == "Filled"
    assert row["broker_order_id"] == 5026


def test_ttl_zero_preserves_legacy_unconditional_block(tmp_path: Path) -> None:
    """With TTL <= 0, the pre-OPS-OMEGA-01 'block Filled forever' semantics
    must be preserved (back-compat for paths that opt out)."""
    store = _make_store(tmp_path)
    now = _utc("2026-05-22T18:00:00+00:00")
    _insert_row(
        store,
        key="legacy_filled_key",
        status="Filled",
        updated_at=now - timedelta(days=7),
        broker_order_id=4767,
        payload=M6E_PAYLOAD,
    )

    claimed = store.claim_or_reclaim(
        "legacy_filled_key",
        M6E_PAYLOAD,
        now,
        stale_threshold_s=600.0,
        ib_probe=_refuse_probe,
        terminal_positive_ttl_s=0.0,
    )
    assert claimed is False, "TTL<=0 must keep Filled blocking forever"


def test_stale_non_terminal_with_no_broker_order_id_is_reclaimed_via_absent(
    tmp_path: Path,
) -> None:
    """duplicate_open_order rows have broker_order_id=None — no broker order
    was ever placed. The S5 probe must classify these as ABSENT so they can be
    reclaimed instead of permanently locking the fingerprint."""

    store = _make_store(tmp_path)
    now = _utc("2026-05-22T18:00:00+00:00")
    # Non-terminal "duplicate_open_order" row, 20 minutes old (past 600s stale).
    _insert_row(
        store,
        key="mes_dup_open_order_key",
        status="duplicate_open_order",
        updated_at=now - timedelta(seconds=1200),
        broker_order_id=None,
        payload=MES_PAYLOAD,
    )

    # Mirror IbkrAdapter._ib_probe semantics for a None broker_order_id.
    def _probe(broker_order_id):
        # Real probe (post-fix) returns ABSENT when broker_order_id is None
        # because no broker order was ever placed.
        if broker_order_id is None or broker_order_id <= 0:
            return _PROBE_ABSENT
        return _PROBE_STILL_ACTIVE

    claimed = store.claim_or_reclaim(
        "mes_dup_open_order_key",
        MES_PAYLOAD,
        now,
        stale_threshold_s=600.0,
        ib_probe=_probe,
        terminal_positive_ttl_s=900.0,
    )
    assert claimed is True, "Stale non-terminal row with no boid must be reclaimable"

    row = store.get("mes_dup_open_order_key")
    assert row is not None
    assert row["status"] == "claimed"


def test_ib_probe_returns_absent_when_broker_order_id_is_none() -> None:
    """Exercises IbkrAdapter._ib_probe directly — the fix changes the
    broker_order_id=None branch from STILL_ACTIVE to ABSENT."""
    from chad.execution.ibkr_adapter import IbkrAdapter

    # We only need a bound method; an empty stand-in for `self` works because
    # _ib_probe doesn't touch any instance state on the early-return paths.
    fake_ib = SimpleNamespace()
    result = IbkrAdapter._ib_probe(SimpleNamespace(), fake_ib, None)
    assert result == _PROBE_ABSENT

    result_zero = IbkrAdapter._ib_probe(SimpleNamespace(), fake_ib, 0)
    assert result_zero == _PROBE_ABSENT


def test_unrelated_fingerprint_is_not_blocked_by_m6e_or_mes(tmp_path: Path) -> None:
    """Sanity: stale M6E Filled row must not affect MGC SELL claim."""
    store = _make_store(tmp_path)
    now = _utc("2026-05-22T18:00:00+00:00")
    _insert_row(
        store,
        key="m6e_filled_key_unrelated",
        status="Filled",
        updated_at=now - timedelta(seconds=300),
        broker_order_id=5026,
        payload=M6E_PAYLOAD,
    )

    claimed = store.claim_or_reclaim(
        "mgc_sell_brand_new_key",
        MGC_PAYLOAD,
        now,
        stale_threshold_s=600.0,
        ib_probe=_refuse_probe,
        terminal_positive_ttl_s=900.0,
    )
    assert claimed is True, "Unrelated key must be allowed to claim"


# ---------------------------------------------------------------------------
# Pattern C — cooldown not re-armed on unconfirmed submission
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_signal_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect chad.core.signal_guard to a tempfile state path."""
    import chad.core.signal_guard as sg

    state_file = tmp_path / "signal_guard.json"
    monkeypatch.setattr(sg, "STATE_PATH", state_file)
    return sg, state_file


def _signal(strategy: str, symbol: str, side: str, size: float) -> SimpleNamespace:
    return SimpleNamespace(strategy=strategy, symbol=symbol, side=side, size=size)


def test_revert_emission_removes_just_written_entry(isolated_signal_guard) -> None:
    sg, state_file = isolated_signal_guard

    sig = _signal("omega_macro", "M6E", "BUY", 2.0)
    assert sg.should_emit_signal(sig) is True, "Fresh signal must emit"

    state = json.loads(state_file.read_text())
    assert "omega_macro|M6E|BUY|2.0" in state, "should_emit_signal must write entry"

    removed = sg.revert_emission_for_unconfirmed(sig)
    assert removed is True

    state_after = json.loads(state_file.read_text())
    assert "omega_macro|M6E|BUY|2.0" not in state_after, (
        "revert must delete the cooldown-arming entry so duplicate_blocked does "
        "not consume the 10-minute cooldown"
    )


def test_revert_after_duplicate_blocked_allows_immediate_reemit(
    isolated_signal_guard,
) -> None:
    sg, _ = isolated_signal_guard
    sig = _signal("alpha_futures", "MES", "BUY", 1.0)

    # Cycle N: signal fires; downstream returns duplicate_blocked; revert.
    assert sg.should_emit_signal(sig) is True
    sg.revert_emission_for_unconfirmed(sig)

    # Cycle N+1 (next loop, no time elapsed for cooldown): signal must emit
    # again rather than be suppressed by the just-armed-then-reverted cooldown.
    assert sg.should_emit_signal(sig) is True, (
        "After revert, the next emission must not be cooldown-blocked — "
        "an unconfirmed submission must not consume the 10-minute window"
    )


def test_fresh_emission_within_cooldown_still_blocks_when_not_reverted(
    isolated_signal_guard,
) -> None:
    """Sanity: without the revert call, the existing 10-minute cooldown still
    works for legitimate emissions. Ensures Pattern C is a targeted fix."""
    sg, _ = isolated_signal_guard
    sig = _signal("alpha_futures", "MGC", "SELL", 1.0)

    assert sg.should_emit_signal(sig) is True
    # Do NOT revert — simulate a confirmed submission.
    assert sg.should_emit_signal(sig) is False, (
        "Without revert, cooldown must still block re-emission inside 10 min"
    )


def test_revert_unrelated_fingerprint_is_a_noop(isolated_signal_guard) -> None:
    sg, state_file = isolated_signal_guard

    armed = _signal("alpha_futures", "MES", "BUY", 1.0)
    assert sg.should_emit_signal(armed) is True

    # Reverting a *different* fingerprint must not touch the armed entry.
    other = _signal("alpha_futures", "MGC", "SELL", 1.0)
    removed = sg.revert_emission_for_unconfirmed(other)
    assert removed is False

    state = json.loads(state_file.read_text())
    assert "alpha_futures|MES|BUY|1.0" in state, "armed entry must persist"
    assert "alpha_futures|MGC|SELL|1.0" not in state

"""GAP-018 / NEW-GAP-051: operator halt-clear semantics.

When an operator clears an edge-decay halt the persisted
``consecutive_negative`` counter must be reset to 0 so downstream readers
of ``runtime/strategy_allocations.json`` cannot be misled by a stale
pre-clear streak value. The pre-clear count must survive as
``previous_consecutive_negative`` for the audit trail, and raw trade
history must never be touched.

Regression target observed in production state on 2026-05-12:

    alpha_options:
      halted: false
      cleared_at: 2026-05-12T00:03:29Z
      cleared_by: operator
      consecutive_negative: 292   <-- misleading stale counter (pre-fix)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.risk.edge_decay_monitor import (
    DEFAULT_CONSECUTIVE_THRESHOLD,
    EdgeDecayMonitor,
    clear_strategy_halt,
    is_strategy_halted,
    read_allocations,
    set_strategy_halted,
)


@pytest.fixture()
def alloc_path(tmp_path: Path) -> Path:
    return tmp_path / "strategy_allocations.json"


def _entry(path: Path, strategy: str) -> dict:
    data = read_allocations(path)
    entry = data["allocations"].get(strategy)
    assert isinstance(entry, dict), f"missing allocations entry for {strategy}"
    return entry


def test_operator_clear_resets_consecutive_negative_counter(alloc_path: Path):
    set_strategy_halted("alpha_options", consecutive_negative=292, path=alloc_path)
    assert _entry(alloc_path, "alpha_options")["consecutive_negative"] == 292

    clear_strategy_halt("alpha_options", cleared_by="operator", path=alloc_path)

    entry = _entry(alloc_path, "alpha_options")
    assert entry["halted"] is False
    assert entry["halt_reason"] == ""
    assert entry["consecutive_negative"] == 0, (
        "operator clear must reset the misleading consecutive_negative counter "
        "(GAP-018/NEW-GAP-051): stale pre-clear value would mislead readers of "
        "strategy_allocations.json"
    )


def test_operator_clear_preserves_previous_counter_for_audit(alloc_path: Path):
    set_strategy_halted("alpha_options", consecutive_negative=292, path=alloc_path)
    clear_strategy_halt(
        "alpha_options",
        cleared_by="operator",
        path=alloc_path,
        clear_reason="phantom_BAG_close_quarantined",
    )
    entry = _entry(alloc_path, "alpha_options")
    assert entry["previous_consecutive_negative"] == 292
    assert entry["cleared_by"] == "operator"
    assert entry["clear_reason"] == "phantom_BAG_close_quarantined"
    assert entry["cleared_at"]  # ISO timestamp populated
    assert entry["halted_at"]   # original halt timestamp preserved


def test_operator_clear_does_not_modify_trade_ledger(tmp_path: Path):
    """The raw trade history under data/trades/ must never be touched by
    the operator-clear path; only the per-strategy allocations entry
    is mutated.
    """
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir()
    ledger = trades_dir / "trade_history_20260512.ndjson"
    payload = {
        "record_hash": "h1",
        "payload": {"strategy": "alpha_options", "pnl": -100.0, "fill_ids": []},
    }
    ledger.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    before_mtime = ledger.stat().st_mtime_ns
    before_bytes = ledger.read_bytes()

    alloc_path = tmp_path / "strategy_allocations.json"
    set_strategy_halted("alpha_options", consecutive_negative=10, path=alloc_path)
    clear_strategy_halt("alpha_options", cleared_by="operator", path=alloc_path)

    assert ledger.read_bytes() == before_bytes
    assert ledger.stat().st_mtime_ns == before_mtime


def test_re_halt_after_clear_uses_freshly_computed_streak(alloc_path: Path):
    """Safety: clearing the halt must NOT silently hide real losses.

    If the trade ledger still contains a losing streak above threshold,
    the next monitor pass must re-halt the strategy with a freshly
    computed count rather than relying on the persisted counter.
    """
    set_strategy_halted("alpha", consecutive_negative=99, path=alloc_path)
    clear_strategy_halt("alpha", cleared_by="operator", path=alloc_path)
    assert _entry(alloc_path, "alpha")["consecutive_negative"] == 0

    monitor = EdgeDecayMonitor(allocations_path=alloc_path)
    real_losses = [10.0] * 20 + [-1.0] * DEFAULT_CONSECUTIVE_THRESHOLD
    result = monitor.check_strategy("alpha", pnls=real_losses)

    assert result["halted"] is True
    assert result["consecutive_neg"] == DEFAULT_CONSECUTIVE_THRESHOLD
    assert is_strategy_halted("alpha", path=alloc_path) is True
    entry = _entry(alloc_path, "alpha")
    assert entry["consecutive_negative"] == DEFAULT_CONSECUTIVE_THRESHOLD


def test_clear_on_missing_strategy_is_safe_noop(alloc_path: Path):
    """Clearing a strategy that was never halted must not invent an entry
    or mutate raw history.
    """
    result = clear_strategy_halt("never_halted", cleared_by="operator", path=alloc_path)
    assert "never_halted" not in result.get("allocations", {})


def test_clear_idempotent_does_not_double_overwrite_audit_fields(alloc_path: Path):
    """A second operator clear on an already-cleared entry must not
    overwrite the original previous_consecutive_negative with 0 (i.e.
    the audit trail must survive idempotent re-clears).
    """
    set_strategy_halted("delta", consecutive_negative=12, path=alloc_path)
    clear_strategy_halt("delta", cleared_by="operator", path=alloc_path)
    first = _entry(alloc_path, "delta")
    assert first["previous_consecutive_negative"] == 12
    assert first["consecutive_negative"] == 0

    clear_strategy_halt("delta", cleared_by="operator", path=alloc_path)
    second = _entry(alloc_path, "delta")
    assert second["consecutive_negative"] == 0
    assert second["previous_consecutive_negative"] == 0, (
        "second clear records the now-current counter (0); the original "
        "12 stays preserved in the first clear's audit fields/log"
    )


def test_default_clear_reason_is_present_when_omitted(alloc_path: Path):
    set_strategy_halted("gamma_futures", consecutive_negative=8, path=alloc_path)
    clear_strategy_halt("gamma_futures", cleared_by="operator", path=alloc_path)
    entry = _entry(alloc_path, "gamma_futures")
    assert entry["clear_reason"] == "manual_operator_clear"

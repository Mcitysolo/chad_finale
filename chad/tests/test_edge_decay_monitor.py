"""Tests for Phase-8 Session 5 EdgeDecayMonitor (F4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.risk.edge_decay_monitor import (
    DEFAULT_CONSECUTIVE_THRESHOLD,
    DEFAULT_MIN_TRADES,
    EdgeDecayMonitor,
    clear_strategy_halt,
    count_consecutive_negative,
    is_strategy_halted,
    read_allocations,
    set_strategy_halted,
)


@pytest.fixture()
def alloc_path(tmp_path: Path) -> Path:
    return tmp_path / "strategy_allocations.json"


# ---------------------------------------------------------------------------
# count_consecutive_negative (pure)
# ---------------------------------------------------------------------------


def test_count_consecutive_negative_empty_returns_zero():
    assert count_consecutive_negative([]) == 0


def test_count_consecutive_negative_all_losses():
    assert count_consecutive_negative([-1, -2, -3, -4]) == 4


def test_count_consecutive_negative_tail_only():
    """Only the trailing losing streak counts; earlier losses don't."""
    assert count_consecutive_negative([-5, 10, -1, -1, -1]) == 3


def test_count_consecutive_negative_last_win_resets():
    assert count_consecutive_negative([-1, -1, -1, 5]) == 0


# ---------------------------------------------------------------------------
# monitor streak logic
# ---------------------------------------------------------------------------


def test_below_min_trades_no_halt(alloc_path: Path):
    monitor = EdgeDecayMonitor(allocations_path=alloc_path)
    # All negative, but fewer than DEFAULT_MIN_TRADES.
    pnls = [-1.0] * (DEFAULT_MIN_TRADES - 1)
    result = monitor.check_strategy("alpha", pnls=pnls)
    assert result["halted"] is False
    assert result["reason"] == "insufficient_trades"
    assert is_strategy_halted("alpha", path=alloc_path) is False


def test_consecutive_negative_triggers_halt(alloc_path: Path):
    monitor = EdgeDecayMonitor(allocations_path=alloc_path)
    # Enough total trades, with the trailing tail meeting the threshold.
    pnls = [10.0] * 15 + [-1.0] * DEFAULT_CONSECUTIVE_THRESHOLD
    result = monitor.check_strategy("alpha", pnls=pnls)
    assert result["halted"] is True
    assert result["consecutive_neg"] == DEFAULT_CONSECUTIVE_THRESHOLD
    assert is_strategy_halted("alpha", path=alloc_path) is True


def test_positive_resets_streak(alloc_path: Path):
    monitor = EdgeDecayMonitor(allocations_path=alloc_path)
    # 20+ trades, a losing streak, then one positive that resets.
    pnls = [10.0] * 10 + [-1.0] * 9 + [5.0]
    result = monitor.check_strategy("alpha", pnls=pnls)
    assert result["halted"] is False
    assert result["consecutive_neg"] == 0


def test_halt_is_persisted_to_allocations_file(alloc_path: Path):
    monitor = EdgeDecayMonitor(allocations_path=alloc_path)
    pnls = [10.0] * 15 + [-1.0] * DEFAULT_CONSECUTIVE_THRESHOLD
    monitor.check_strategy("alpha", pnls=pnls)
    data = json.loads(alloc_path.read_text(encoding="utf-8"))
    entry = data["allocations"]["alpha"]
    assert entry["halted"] is True
    assert entry["consecutive_negative"] == DEFAULT_CONSECUTIVE_THRESHOLD


def test_custom_threshold(alloc_path: Path):
    monitor = EdgeDecayMonitor(
        consecutive_threshold=3,
        min_trades=5,
        allocations_path=alloc_path,
    )
    pnls = [10.0] * 10 + [-1.0, -1.0, -1.0]
    result = monitor.check_strategy("alpha", pnls=pnls)
    assert result["halted"] is True
    assert result["consecutive_neg"] == 3


# ---------------------------------------------------------------------------
# allocations state (read/write/clear)
# ---------------------------------------------------------------------------


def test_read_allocations_missing_returns_empty(alloc_path: Path):
    data = read_allocations(alloc_path)
    assert data["allocations"] == {}


def test_set_strategy_halted_persists(alloc_path: Path):
    set_strategy_halted("alpha", consecutive_negative=12, path=alloc_path)
    assert is_strategy_halted("alpha", path=alloc_path) is True
    data = read_allocations(alloc_path)
    assert data["allocations"]["alpha"]["halted"] is True
    assert data["allocations"]["alpha"]["consecutive_negative"] == 12


def test_clear_script_restores_allocation(alloc_path: Path):
    set_strategy_halted("alpha", consecutive_negative=12, path=alloc_path)
    assert is_strategy_halted("alpha", path=alloc_path) is True
    clear_strategy_halt("alpha", cleared_by="operator", path=alloc_path)
    assert is_strategy_halted("alpha", path=alloc_path) is False
    data = read_allocations(alloc_path)
    assert data["allocations"]["alpha"]["cleared_by"] == "operator"


def test_read_allocations_tolerates_malformed_file(tmp_path: Path):
    path = tmp_path / "strategy_allocations.json"
    path.write_text("{not valid json", encoding="utf-8")
    data = read_allocations(path)
    assert data["allocations"] == {}
    assert is_strategy_halted("alpha", path=path) is False

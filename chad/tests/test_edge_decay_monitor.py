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
    collect_recent_trades_by_strategy,
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


# ---------------------------------------------------------------------------
# Quarantine awareness (regression: 2026-05-11 alpha_options re-halt loop)
# ---------------------------------------------------------------------------


def _write_trade_row(
    fh,
    *,
    strategy: str,
    pnl: float,
    record_hash: str,
    fill_ids: list,
) -> None:
    """Write a single NDJSON trade row with the minimum fields the
    edge_decay monitor inspects (strategy, pnl, record_hash, fill_ids).
    """
    rec = {
        "record_hash": record_hash,
        "payload": {
            "strategy": strategy,
            "pnl": pnl,
            "broker": "ibkr_paper",
            "is_live": False,
            "fill_ids": list(fill_ids),
        },
    }
    fh.write(json.dumps(rec) + "\n")


def test_edge_decay_monitor_skips_quarantined_trades(tmp_path: Path):
    """5 quarantined consecutive losses + 1 real win -> streak = 0."""
    trades_path = tmp_path / "trade_history_20260511.ndjson"
    bad_hashes = {f"badhash{i}" for i in range(5)}
    with trades_path.open("w", encoding="utf-8") as fh:
        for i in range(5):
            _write_trade_row(
                fh,
                strategy="alpha_options",
                pnl=-100.0,
                record_hash=f"badhash{i}",
                fill_ids=[],
            )
        _write_trade_row(
            fh,
            strategy="alpha_options",
            pnl=50.0,
            record_hash="goodhash_win",
            fill_ids=[],
        )

    ledger = collect_recent_trades_by_strategy(
        glob_pattern=str(trades_path),
        invalid_fill_ids=set(),
        invalid_trade_hashes=bad_hashes,
    )
    pnls = ledger.get("alpha_options", [])
    assert pnls == [50.0], f"expected only the unquarantined win, got {pnls}"
    assert count_consecutive_negative(pnls) == 0


def test_edge_decay_monitor_counts_real_losses_after_quarantine(tmp_path: Path):
    """3 quarantined losses (oldest) + 2 real losses -> streak = 2."""
    trades_path = tmp_path / "trade_history_20260511.ndjson"
    bad_hashes = {f"badhash{i}" for i in range(3)}
    with trades_path.open("w", encoding="utf-8") as fh:
        for i in range(3):
            _write_trade_row(
                fh,
                strategy="alpha_options",
                pnl=-100.0,
                record_hash=f"badhash{i}",
                fill_ids=[],
            )
        for i in range(2):
            _write_trade_row(
                fh,
                strategy="alpha_options",
                pnl=-25.0,
                record_hash=f"realhash{i}",
                fill_ids=[],
            )

    ledger = collect_recent_trades_by_strategy(
        glob_pattern=str(trades_path),
        invalid_fill_ids=set(),
        invalid_trade_hashes=bad_hashes,
    )
    pnls = ledger.get("alpha_options", [])
    assert pnls == [-25.0, -25.0]
    assert count_consecutive_negative(pnls) == 2


def test_alpha_options_not_re_halted_after_quarantine_20260511():
    """Regression: 2026-05-11 alpha_options halt was driven by 292 phantom
    BAG close trades that were quarantined in
    runtime/quarantine_manifest_20260511.json. The edge_decay monitor must
    honour the manifest so it does not immediately re-halt the strategy
    after every manual clear.
    """
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "runtime" / "quarantine_manifest_20260511.json"
    trades_path = repo_root / "data" / "trades" / "trade_history_20260511.ndjson"
    if not manifest_path.is_file() or not trades_path.is_file():
        pytest.skip("manifest/trade fixture not present in this checkout")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    invalid_trade_hashes = {
        e["record_hash"]
        for e in manifest.get("invalid_trades", [])
        if isinstance(e, dict) and isinstance(e.get("record_hash"), str)
    }
    invalid_fill_ids = {
        e["fill_id"]
        for e in manifest.get("invalid_fills", [])
        if isinstance(e, dict) and isinstance(e.get("fill_id"), str)
    }
    assert invalid_trade_hashes, "manifest has no invalid_trades record_hashes"

    ledger = collect_recent_trades_by_strategy(
        glob_pattern=str(trades_path),
        invalid_fill_ids=invalid_fill_ids,
        invalid_trade_hashes=invalid_trade_hashes,
    )
    pnls = ledger.get("alpha_options", [])
    streak = count_consecutive_negative(pnls)
    assert streak < DEFAULT_CONSECUTIVE_THRESHOLD, (
        f"alpha_options would re-halt: streak={streak} >= "
        f"threshold={DEFAULT_CONSECUTIVE_THRESHOLD} after applying "
        f"{len(invalid_trade_hashes)} quarantined hashes / "
        f"{len(invalid_fill_ids)} quarantined fills"
    )

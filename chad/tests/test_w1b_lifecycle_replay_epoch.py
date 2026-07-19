"""W1B-4 — lifecycle-replay honours the Epoch-3 reset.

chad/ops/lifecycle_replay_engine.py reconstructed positions by summing signed
fill qty across ALL fill history with no epoch filter, so pre-reset residuals
(futures + QQQ that netted flat before the 2026-06-30T12:17:42Z Epoch-3
boundary) surfaced as phantom positions and drove the census "REPLAY_MISMATCH".

The fix mirrors the one epoch-aware engine (trade_stats_engine): resolve the
cutoff from runtime/epoch_state.json (via the passed repo_root) and skip fills
and fees whose realised time is strictly before it. build_replay_state already
takes a repo_root, so this test is fully hermetic against a tmp_path repo.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.ops import lifecycle_replay_engine as engine

_EPOCH = "2026-06-30T12:17:42Z"


def _write_fills(repo_root: Path, rows: list[dict]) -> None:
    d = repo_root / "data" / "fills"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "FILLS_test.ndjson").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _fill(symbol: str, side: str, fill_time_utc: str, *, qty: float = 1.0, price: float = 100.0) -> dict:
    # Title-case "Filled" on purpose — the engine uppercases status at read.
    return {
        "payload": {
            "symbol": symbol,
            "side": side,
            "status": "Filled",
            "quantity": qty,
            "fill_price": price,
            "fill_time_utc": fill_time_utc,
        }
    }


def _write_epoch_state(repo_root: Path, started_at: str) -> None:
    rd = repo_root / "runtime"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "epoch_state.json").write_text(
        json.dumps({"schema_version": "epoch_state.v1", "epoch_started_at_utc": started_at}),
        encoding="utf-8",
    )


def test_pre_epoch_fill_is_excluded(tmp_path: Path) -> None:
    _write_fills(tmp_path, [
        _fill("MES", "BUY", "2026-06-01T00:00:00Z"),   # pre-epoch residual
        _fill("SPY", "BUY", "2026-07-01T00:00:00Z"),   # current epoch
    ])
    _write_epoch_state(tmp_path, _EPOCH)

    state = engine.build_replay_state(tmp_path)

    assert "MES" not in state["positions"], "pre-epoch fill must not surface"
    assert "SPY" in state["positions"]
    assert state["positions_count"] == 1
    assert state["epoch_cutoff_utc"] == _EPOCH
    assert state["inputs"]["fills_pre_epoch_skipped"] == 1


def test_missing_epoch_state_preserves_legacy_behavior(tmp_path: Path) -> None:
    # No runtime/epoch_state.json -> no cutoff -> all-history legacy behaviour.
    _write_fills(tmp_path, [
        _fill("MES", "BUY", "2026-06-01T00:00:00Z"),
        _fill("SPY", "BUY", "2026-07-01T00:00:00Z"),
    ])

    state = engine.build_replay_state(tmp_path)

    assert "MES" in state["positions"]
    assert "SPY" in state["positions"]
    assert state["positions_count"] == 2
    assert state["epoch_cutoff_utc"] is None
    assert state["inputs"]["fills_pre_epoch_skipped"] == 0


def test_corrupt_epoch_state_fails_safe_to_legacy(tmp_path: Path) -> None:
    _write_fills(tmp_path, [_fill("MES", "BUY", "2026-06-01T00:00:00Z")])
    rd = tmp_path / "runtime"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "epoch_state.json").write_text("{ this is not json", encoding="utf-8")

    state = engine.build_replay_state(tmp_path)

    # Corrupt state -> load_epoch_state returns None -> legacy (keep the fill).
    assert "MES" in state["positions"]
    assert state["epoch_cutoff_utc"] is None


def test_fill_without_timestamp_is_kept(tmp_path: Path) -> None:
    # A fill with no usable timestamp is the safe/legacy direction: kept.
    _write_fills(tmp_path, [
        {"payload": {"symbol": "NOTS", "side": "BUY", "status": "Filled",
                     "quantity": 1, "fill_price": 100}},  # no fill_time_utc
        _fill("SPY", "BUY", "2026-07-01T00:00:00Z"),
    ])
    _write_epoch_state(tmp_path, _EPOCH)

    state = engine.build_replay_state(tmp_path)

    assert "NOTS" in state["positions"], "no-timestamp fill must be kept (is_pre_epoch->False)"
    assert "SPY" in state["positions"]


def test_replay_positions_epoch_cutoff_none_is_byte_for_byte_legacy(tmp_path: Path) -> None:
    # Direct unit on replay_positions: epoch_cutoff=None == the pre-fix path.
    evidence = {
        "fills": [
            _fill("MES", "BUY", "2026-06-01T00:00:00Z"),
            _fill("SPY", "BUY", "2026-07-01T00:00:00Z"),
        ],
        "fees": [],
        "broker_events": [],
    }
    unfiltered = engine.replay_positions(evidence, epoch_cutoff=None)
    assert set(unfiltered) == {"MES", "SPY"}

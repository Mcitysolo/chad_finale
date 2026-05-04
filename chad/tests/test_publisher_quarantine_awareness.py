"""
Tests for publisher quarantine awareness.

These tests verify that the runtime publishers that derive state from
data/fills/ + data/trades/ honour
runtime/quarantine_manifest_*.json by excluding listed records:

  * profit_lock.NdjsonPnlProvider     -> pnl_state.json
  * trade_stats_engine.load_and_compute -> SCR/shadow stats
  * expectancy_tracker.compute        -> expectancy_state.json
                                          (feeds strategy_health + winner_scaler)

Plus two helper edge-case tests:
  * missing manifest -> empty sets, no crash
  * corrupt manifest -> empty sets, no crash, warning logged
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _trade_record(*, record_hash: str, payload: dict) -> dict:
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": record_hash,
    }


def _fill_record(*, record_hash: str, fill_id: str, extras: dict | None = None) -> dict:
    payload = {
        "broker": "ibkr",
        "is_live": False,
        "strategy": "delta",
        "symbol": "SPY",
        "side": "SELL",
        "quantity": 10.0,
        "fill_price": 100.0,
        "notional": 1000.0,
        "fill_id": fill_id,
        "tags": ["paper", "filled"],
        "extra": dict(extras or {}),
    }
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": record_hash,
    }


def _write_manifest(runtime_dir: Path, *, fill_ids: list[str], trade_hashes: list[str]) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "quarantine_manifest_20260503.json"
    doc = {
        "quarantined_at_utc": "2026-05-03T20:58:00Z",
        "reason": "test",
        "invalid_fills": [{"fill_id": fid, "symbol": "SPY"} for fid in fill_ids],
        "invalid_trades": [{"record_hash": rh, "symbol": "SPY"} for rh in trade_hashes],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helper module: missing + corrupt manifest fail-safe
# ---------------------------------------------------------------------------


def test_quarantine_helper_missing_manifest_is_empty(tmp_path: Path) -> None:
    """No manifest in runtime/ -> empty sets, no crash."""
    from chad.utils.quarantine import get_quarantine_sets

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    fill_ids, trade_hashes = get_quarantine_sets(runtime_dir=runtime_dir)
    assert fill_ids == set()
    assert trade_hashes == set()

    # Also: nonexistent directory should not crash either.
    fill_ids2, trade_hashes2 = get_quarantine_sets(runtime_dir=tmp_path / "no_such_dir")
    assert fill_ids2 == set()
    assert trade_hashes2 == set()


def test_quarantine_helper_corrupt_manifest_does_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt JSON / wrong shape -> warning logged, empty sets, no raise."""
    from chad.utils.quarantine import get_quarantine_sets

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    bad = runtime_dir / "quarantine_manifest_corrupt.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")

    bad_shape = runtime_dir / "quarantine_manifest_badshape.json"
    bad_shape.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="chad.utils.quarantine"):
        fill_ids, trade_hashes = get_quarantine_sets(runtime_dir=runtime_dir)
    assert fill_ids == set()
    assert trade_hashes == set()
    # At least one warning emitted for the corrupt files.
    assert any("quarantine_manifest" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Publisher: pnl_state.json (profit_lock.NdjsonPnlProvider)
# ---------------------------------------------------------------------------


def test_pnl_state_excludes_quarantined_trades(tmp_path: Path) -> None:
    """NdjsonPnlProvider must skip trades whose record_hash is quarantined."""
    from chad.risk.profit_lock import NdjsonPnlProvider

    repo_root = tmp_path
    trades_dir = repo_root / "data" / "trades"
    trades_dir.mkdir(parents=True)
    fills_dir = repo_root / "data" / "fills"
    fills_dir.mkdir(parents=True)

    today = _today_ymd()

    # Two trades; one quarantined by record_hash, one trusted.
    trades = [
        _trade_record(
            record_hash="QUAR_TRADE_1",
            payload={
                "broker": "ibkr",
                "is_live": False,
                "strategy": "delta",
                "symbol": "SPY",
                "side": "BUY",
                "quantity": 10.0,
                "fill_price": 100.0,
                "notional": 1000.0,
                "pnl": -6200.0,
            },
        ),
        _trade_record(
            record_hash="GOOD_TRADE_1",
            payload={
                "broker": "ibkr",
                "is_live": False,
                "strategy": "delta",
                "symbol": "MES",
                "side": "BUY",
                "quantity": 1.0,
                "fill_price": 5500.0,
                "notional": 5500.0,
                "pnl": 25.0,
            },
        ),
    ]
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in trades) + "\n", encoding="utf-8"
    )

    # One fill; quarantined by fill_id.
    fills = [
        _fill_record(record_hash="FILL_REC_1", fill_id="QUAR_FILL_1"),
        _fill_record(record_hash="FILL_REC_2", fill_id="GOOD_FILL_1"),
    ]
    (fills_dir / f"FILLS_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in fills) + "\n", encoding="utf-8"
    )

    _write_manifest(
        repo_root / "runtime",
        fill_ids=["QUAR_FILL_1"],
        trade_hashes=["QUAR_TRADE_1"],
    )

    provider = NdjsonPnlProvider()
    pnl, count, _sources = asyncio.run(provider.get_realized_pnl(repo_root, days=0))

    # Quarantined -$6200 trade is excluded; only the +25 trade should sum.
    # Fill records contribute pnl=0.0 (entry-only); the quarantined fill
    # is dropped before that bookkeeping increment fires.
    assert pnl == 25.0, f"expected 25.0, got {pnl}"
    # 1 trusted trade + 1 trusted fill = 2 effective records counted.
    assert count == 2, f"expected 2 effective rows, got {count}"


# ---------------------------------------------------------------------------
# Publisher: SCR/shadow stats (trade_stats_engine.load_and_compute)
# ---------------------------------------------------------------------------


def test_scr_stats_exclude_quarantined_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """trade_stats_engine must skip quarantined record_hashes from SCR pool."""
    from chad.analytics import trade_stats_engine

    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)
    today = _today_ymd()

    trades = [
        _trade_record(
            record_hash="QUAR_SPY_1",
            payload={
                "broker": "ibkr",
                "is_live": False,
                "strategy": "delta",
                "symbol": "SPY",
                "side": "BUY",
                "quantity": 10.0,
                "fill_price": 100.0,
                "notional": 1000.0,
                "pnl": -6200.0,
                "tags": ["paper", "filled"],
            },
        ),
        _trade_record(
            record_hash="GOOD_TRADE_X",
            payload={
                "broker": "ibkr",
                "is_live": False,
                "strategy": "delta",
                "symbol": "MES",
                "side": "BUY",
                "quantity": 1.0,
                "fill_price": 5500.0,
                "notional": 5500.0,
                "pnl": 25.0,
                "tags": ["paper", "filled"],
            },
        ),
    ]
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in trades) + "\n", encoding="utf-8"
    )

    _write_manifest(
        tmp_path / "runtime",
        fill_ids=[],
        trade_hashes=["QUAR_SPY_1"],
    )

    # Point trade_stats_engine + the helper at our temp tree.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Quarantined trade dropped before parse; total + total_pnl reflect
    # only the trusted +25 row.
    assert stats["total_trades"] == 1, stats
    assert stats["total_pnl"] == 25.0, stats
    assert stats["excluded_quarantined"] == 1, stats
    # Surfaced via excluded_untrusted bucket too (SCR contract).
    assert stats["excluded_untrusted"] >= 1, stats


# ---------------------------------------------------------------------------
# Publisher: strategy_health (via expectancy_tracker.compute)
# ---------------------------------------------------------------------------


def test_strategy_health_excludes_quarantined_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """expectancy_tracker (feeds strategy_health + winner_scaler) must skip
    trades whose record_hash is quarantined."""
    # Re-target the module-level paths the tracker resolves at import.
    from chad.analytics import expectancy_tracker

    trades_dir = tmp_path / "data" / "trades"
    runtime_dir = tmp_path / "runtime"
    trades_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    today = _today_ymd()

    trades = [
        _trade_record(
            record_hash="QUAR_DELTA_1",
            payload={
                "strategy": "delta",
                "symbol": "SPY",
                "pnl": -6200.0,
            },
        ),
        _trade_record(
            record_hash="GOOD_DELTA_1",
            payload={
                "strategy": "delta",
                "symbol": "MES",
                "pnl": 25.0,
            },
        ),
    ]
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in trades) + "\n", encoding="utf-8"
    )

    _write_manifest(
        runtime_dir,
        fill_ids=[],
        trade_hashes=["QUAR_DELTA_1"],
    )

    monkeypatch.setattr(expectancy_tracker, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(
        expectancy_tracker,
        "TRADES_GLOB",
        str(trades_dir / "trade_history_*.ndjson"),
    )
    monkeypatch.setattr(
        expectancy_tracker, "PNL_STATE", str(runtime_dir / "pnl_state.json")
    )
    monkeypatch.setattr(
        expectancy_tracker, "DYNAMIC_CAPS", str(runtime_dir / "dynamic_caps.json")
    )

    state = expectancy_tracker.compute()
    delta = state["strategies"].get("delta") or {}
    # Only the +25 trusted trade should remain in expectancy.
    assert delta.get("total_trades") == 1, state
    assert delta.get("total_pnl") == 25.0, state
    assert state["total_clean_trades"] == 1, state

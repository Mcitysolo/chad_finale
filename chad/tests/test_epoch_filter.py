"""
Tests for the SCR epoch-boundary mechanism.

The Paper-Epoch-2 reset (2026-05-04) introduced ``runtime/epoch_state.json``
as the operator-controlled boundary used by the runtime publishers
(SCR/shadow stats, strategy_health, winner_scaler). Records realised
strictly before ``epoch_started_at_utc`` must be excluded from the
performance pool, while pre-existing hygiene (quarantine, untrusted,
nonfinite, manual) continues to apply.

These tests exercise:
  * ``test_trade_stats_respects_epoch_started_at``
  * ``test_scr_shadow_excludes_pre_epoch_records``
  * ``test_epoch_filter_combines_with_quarantine_manifest``
  * ``test_epoch_missing_state_preserves_legacy_behavior``
  * ``test_epoch_state_corrupt_fails_safe_or_legacy_with_warning``
"""
from __future__ import annotations

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


def _trade_record(
    *,
    record_hash: str,
    pnl: float,
    exit_time_utc: str,
    strategy: str = "delta",
    symbol: str = "AAPL",
    extra_payload: dict | None = None,
) -> dict:
    payload = {
        "broker": "ibkr",
        "is_live": False,
        "strategy": strategy,
        "symbol": symbol,
        "side": "BUY",
        "quantity": 1.0,
        "fill_price": 100.0,
        "notional": 100.0,
        "pnl": pnl,
        "tags": ["paper", "filled", strategy],
        "exit_time_utc": exit_time_utc,
        "entry_time_utc": exit_time_utc,
    }
    if extra_payload:
        payload.update(extra_payload)
    return {
        "timestamp_utc": exit_time_utc,
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": record_hash,
    }


def _write_epoch_state(runtime_dir: Path, *, started_at: str) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "epoch_state.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "epoch_state.v1",
                "active_epoch": "TEST_EPOCH",
                "epoch_started_at_utc": started_at,
                "paper_only": True,
                "ready_for_live": False,
                "previous_epoch_archive": str(runtime_dir / "archive" / "test"),
                "quarantine_manifest": str(runtime_dir / "quarantine_manifest_test.json"),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_quarantine_manifest(
    runtime_dir: Path, *, trade_hashes: list[str]
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "quarantine_manifest_test.json"
    path.write_text(
        json.dumps(
            {
                "quarantined_at_utc": "2026-05-03T20:58:00Z",
                "reason": "test",
                "invalid_fills": [],
                "invalid_trades": [{"record_hash": rh} for rh in trade_hashes],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_trades_file(
    repo_root: Path, *, ymd: str, records: list[dict]
) -> Path:
    trades_dir = repo_root / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    path = trades_dir / f"trade_history_{ymd}.ndjson"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# 1. trade_stats_engine respects epoch_started_at
# ---------------------------------------------------------------------------


def test_trade_stats_respects_epoch_started_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records realised strictly before epoch_started_at_utc are excluded."""
    from chad.analytics import trade_stats_engine

    today = _today_ymd()
    pre = _trade_record(
        record_hash="PRE_EPOCH_1",
        pnl=-100.0,
        exit_time_utc="2026-04-01T12:00:00+00:00",
    )
    post1 = _trade_record(
        record_hash="POST_EPOCH_1",
        pnl=10.0,
        exit_time_utc="2026-05-04T01:00:00+00:00",
    )
    post2 = _trade_record(
        record_hash="POST_EPOCH_2",
        pnl=-5.0,
        exit_time_utc="2026-05-04T02:00:00+00:00",
        symbol="ES",
    )
    _write_trades_file(tmp_path, ymd=today, records=[pre, post1, post2])

    _write_epoch_state(
        tmp_path / "runtime", started_at="2026-05-04T00:54:30Z"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Pre-epoch row dropped before parse — total/total_pnl reflect
    # only post-epoch rows.
    assert stats["total_trades"] == 2, stats
    assert stats["total_pnl"] == 5.0, stats
    assert stats["excluded_pre_epoch"] == 1, stats
    assert stats["epoch"]["filter_applied"] is True, stats
    assert stats["epoch"]["active_epoch"] == "TEST_EPOCH", stats


# ---------------------------------------------------------------------------
# 2. SCR/shadow effective_trades excludes pre-epoch records
# ---------------------------------------------------------------------------


def test_scr_shadow_excludes_pre_epoch_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Effective sample (used by SCR/shadow scoring) excludes pre-epoch rows.

    Pre-epoch trades must not contribute to effective_trades, win_rate,
    sharpe_like, or max_drawdown — those are the fields the shadow
    confidence router consumes.
    """
    from chad.analytics import trade_stats_engine

    today = _today_ymd()
    pre_loss_a = _trade_record(
        record_hash="PRE_LOSS_A", pnl=-50.0,
        exit_time_utc="2026-03-15T10:00:00+00:00",
    )
    pre_loss_b = _trade_record(
        record_hash="PRE_LOSS_B", pnl=-50.0,
        exit_time_utc="2026-03-16T10:00:00+00:00",
    )
    post_win = _trade_record(
        record_hash="POST_WIN_1", pnl=20.0,
        exit_time_utc="2026-05-04T05:00:00+00:00",
    )
    _write_trades_file(
        tmp_path, ymd=today, records=[pre_loss_a, pre_loss_b, post_win]
    )

    _write_epoch_state(
        tmp_path / "runtime", started_at="2026-05-04T00:54:30Z"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Only the post-epoch winner is in the effective pool.
    assert stats["effective_trades"] == 1, stats
    assert stats["win_rate"] == 1.0, stats
    assert stats["max_drawdown"] == 0.0, stats
    assert stats["excluded_pre_epoch"] == 2, stats


# ---------------------------------------------------------------------------
# 3. Epoch + quarantine manifest combine
# ---------------------------------------------------------------------------


def test_epoch_filter_combines_with_quarantine_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-epoch + quarantined exclusions stack without double-counting."""
    from chad.analytics import trade_stats_engine

    today = _today_ymd()
    pre = _trade_record(
        record_hash="PRE_EPOCH_X", pnl=-100.0,
        exit_time_utc="2026-04-10T10:00:00+00:00",
    )
    quar = _trade_record(
        record_hash="QUAR_TRADE_X", pnl=-200.0,
        exit_time_utc="2026-05-05T10:00:00+00:00",
    )
    good = _trade_record(
        record_hash="GOOD_TRADE_X", pnl=15.0,
        exit_time_utc="2026-05-05T11:00:00+00:00",
    )
    _write_trades_file(tmp_path, ymd=today, records=[pre, quar, good])

    _write_epoch_state(
        tmp_path / "runtime", started_at="2026-05-04T00:54:30Z"
    )
    _write_quarantine_manifest(
        tmp_path / "runtime", trade_hashes=["QUAR_TRADE_X"]
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Pre-epoch row -> excluded_pre_epoch.
    # Quarantined post-epoch row -> excluded_quarantined.
    # Trusted post-epoch row -> only one in trades/effective.
    assert stats["total_trades"] == 1, stats
    assert stats["total_pnl"] == 15.0, stats
    assert stats["effective_trades"] == 1, stats
    assert stats["excluded_pre_epoch"] == 1, stats
    assert stats["excluded_quarantined"] == 1, stats
    # Quarantine still surfaces via excluded_untrusted (legacy SCR contract);
    # epoch is its own bucket.
    assert stats["excluded_untrusted"] >= 1, stats


# ---------------------------------------------------------------------------
# 4. Missing epoch_state preserves legacy behavior
# ---------------------------------------------------------------------------


def test_epoch_missing_state_preserves_legacy_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No runtime/epoch_state.json -> legacy (no epoch filter) behavior."""
    from chad.analytics import trade_stats_engine

    today = _today_ymd()
    old = _trade_record(
        record_hash="OLD_TRADE", pnl=-30.0,
        exit_time_utc="2026-01-01T00:00:00+00:00",
    )
    new = _trade_record(
        record_hash="NEW_TRADE", pnl=10.0,
        exit_time_utc="2026-05-04T05:00:00+00:00",
    )
    _write_trades_file(tmp_path, ymd=today, records=[old, new])

    # No epoch_state.json written.
    (tmp_path / "runtime").mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Both trades counted — no boundary applied.
    assert stats["total_trades"] == 2, stats
    assert stats["total_pnl"] == -20.0, stats
    assert stats["excluded_pre_epoch"] == 0, stats
    assert stats["epoch"]["filter_applied"] is False, stats
    assert stats["epoch"]["active_epoch"] == "", stats


# ---------------------------------------------------------------------------
# 5. Corrupt epoch_state -> legacy with warning, no crash
# ---------------------------------------------------------------------------


def test_epoch_state_corrupt_fails_safe_or_legacy_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt JSON / missing timestamp -> warning + legacy behavior."""
    from chad.analytics import trade_stats_engine
    from chad.utils.epoch import load_epoch_state

    today = _today_ymd()
    record = _trade_record(
        record_hash="ANY_TRADE", pnl=5.0,
        exit_time_utc="2026-05-04T05:00:00+00:00",
    )
    _write_trades_file(tmp_path, ymd=today, records=[record])

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    # Corrupt JSON
    (runtime_dir / "epoch_state.json").write_text(
        "{ this is not valid json", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="chad.utils.epoch"):
        es = load_epoch_state(runtime_dir=runtime_dir)
    assert es is None
    assert any("epoch_state" in rec.message for rec in caplog.records)

    # Stats compute legacy (no filter) and do not crash.
    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )
    assert stats["total_trades"] == 1, stats
    assert stats["epoch"]["filter_applied"] is False, stats

    # Bad-shape (list instead of dict) and missing-timestamp also yield None.
    (runtime_dir / "epoch_state.json").write_text(
        json.dumps([1, 2, 3]), encoding="utf-8"
    )
    assert load_epoch_state(runtime_dir=runtime_dir) is None

    (runtime_dir / "epoch_state.json").write_text(
        json.dumps({"schema_version": "epoch_state.v1", "active_epoch": "X"}),
        encoding="utf-8",
    )
    assert load_epoch_state(runtime_dir=runtime_dir) is None

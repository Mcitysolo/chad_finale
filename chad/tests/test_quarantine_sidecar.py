"""Tests for the placeholder-fill quarantine surface.

Covers:
  1. Upstream guard: ``normalize_paper_fill_evidence`` marks placeholder
     fill_price=100 untrusted when the price cache cannot supply a
     reference price (the 2026-05-03..05-08 SPY/delta incident pattern).
  2. ``chad.analytics.quarantine.is_quarantined_fill`` matches sidecar
     entries for poisoned fills.
  3. ``chad.analytics.quarantine.is_quarantined_trade`` matches sidecar
     entries for phantom closed trades.
  4. ``per_strategy_loss_guard.compute_today_realized_pnl`` excludes
     quarantined delta closed trades.
  5. ``trade_stats_engine.load_and_compute`` excludes quarantined
     records via ``chad.utils.quarantine.get_exclusion_sets``.
  6. Non-quarantined normal trades still count.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from chad.execution.paper_exec_evidence_writer import (
    PaperExecEvidence,
    normalize_paper_fill_evidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_price_cache(tmp_path, monkeypatch):
    """Patch PRICE_CACHE_PATH to a tmp file with NO prices — so the
    50%-deviation guard cannot fire and the placeholder-without-cache
    branch is the only thing that can catch a $100 placeholder fill."""
    cache_path = tmp_path / "price_cache.json"
    cache_path.write_text(json.dumps({"prices": {}, "ts_utc": "2026-05-03T00:00:00Z"}))
    monkeypatch.setattr(
        "chad.execution.paper_exec_evidence_writer.PRICE_CACHE_PATH",
        cache_path,
    )
    return cache_path


def _write_sidecar(folder: Path, name: str, body: Dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. Upstream guard
# ---------------------------------------------------------------------------


def test_placeholder_100_without_price_cache_marked_untrusted(empty_price_cache):
    """SPY placeholder fill_price=100, expected=100, no cache → pnl_untrusted=True."""
    ev = PaperExecEvidence(
        symbol="SPY",
        strategy="delta",
        source_strategies=["delta"],
        side="SELL",
        quantity=10.0,
        fill_price=100.0,
        expected_price=100.0,
        status="filled",
        is_live=False,
        asset_class="equity",
        fill_time_utc="2026-05-03T11:48:55+00:00",
    )
    normalize_paper_fill_evidence(ev)
    assert isinstance(ev.extra, dict)
    assert ev.extra.get("pnl_untrusted") is True
    assert (
        ev.extra.get("pnl_untrusted_reason")
        == "placeholder_price_without_price_cache"
    )
    tags_lower = {str(t).strip().lower() for t in (ev.tags or ())}
    assert "pnl_untrusted" in tags_lower
    assert "placeholder_price" in tags_lower


def test_placeholder_guard_skips_non_liquid_symbol(empty_price_cache):
    """Symbol not on the liquid-priced equity allowlist must NOT be flagged
    by this guard — there are real low-priced ETFs that legitimately trade
    near $100. The 50% deviation guard remains the safety net for those."""
    ev = PaperExecEvidence(
        symbol="ZZZX",  # not on the allowlist
        strategy="delta",
        side="SELL",
        quantity=10.0,
        fill_price=100.0,
        expected_price=100.0,
        status="filled",
        is_live=False,
        asset_class="equity",
        fill_time_utc="2026-05-03T11:48:55+00:00",
    )
    normalize_paper_fill_evidence(ev)
    extra = ev.extra if isinstance(ev.extra, dict) else {}
    assert not extra.get("pnl_untrusted")


def test_placeholder_guard_skips_options_asset_class(empty_price_cache):
    """An options fill at $1.00 net debit shaped as fill_price=100 (per-contract
    in cents) should not be caught — the guard only fires on equity/etf/stk."""
    ev = PaperExecEvidence(
        symbol="SPY",
        strategy="alpha",
        side="BUY",
        quantity=1.0,
        fill_price=100.0,
        expected_price=100.0,
        status="filled",
        is_live=False,
        asset_class="options",
        fill_time_utc="2026-05-03T11:48:55+00:00",
    )
    normalize_paper_fill_evidence(ev)
    extra = ev.extra if isinstance(ev.extra, dict) else {}
    # options branch has its own consistency rules; this guard must not fire.
    # The pnl_untrusted_reason (if any) must NOT be the placeholder reason.
    assert (
        extra.get("pnl_untrusted_reason") != "placeholder_price_without_price_cache"
    )


# ---------------------------------------------------------------------------
# 2. fill quarantine loader
# ---------------------------------------------------------------------------


def test_is_quarantined_fill_matches_sidecar_entry(tmp_path):
    """Sidecar with a known fill_id matches via every supported identifier."""
    from chad.analytics import quarantine as qmod

    sidecar = {
        "schema_version": "fills_quarantine.v1",
        "invalid_fills": [
            {
                "source_file": "FILLS_20260503.ndjson",
                "sequence_id": 1,
                "record_hash": "abc123hash",
                "fill_id": "fill_xyz",
                "strategy": "delta",
                "symbol": "SPY",
                "side": "SELL",
                "quantity": 10,
                "fill_price": 100.0,
                "fill_time_utc": "2026-05-03T11:48:55+00:00",
                "reason": "placeholder_price_without_price_cache",
            }
        ],
    }
    _write_sidecar(tmp_path / "fills", "quarantine_test.json", sidecar)

    # by fill_id
    assert qmod.is_quarantined_fill(
        fill_id="fill_xyz", fills_dir=tmp_path / "fills"
    )
    # by record_hash
    assert qmod.is_quarantined_fill(
        record_hash="abc123hash", fills_dir=tmp_path / "fills"
    )
    # by (source_file, sequence_id) pair
    assert qmod.is_quarantined_fill(
        source_file="FILLS_20260503.ndjson",
        sequence_id=1,
        fills_dir=tmp_path / "fills",
    )
    # negative case
    assert not qmod.is_quarantined_fill(
        fill_id="other", fills_dir=tmp_path / "fills"
    )


def test_is_quarantined_trade_matches_sidecar_entry(tmp_path):
    from chad.analytics import quarantine as qmod

    sidecar = {
        "schema_version": "trades_quarantine.v1",
        "invalid_trades": [
            {
                "source_file": "trade_history_20260508.ndjson",
                "sequence_id": 5,
                "record_hash": "trade_hash_a",
                "pnl": -6314.4,
                "fill_ids": ["fill_open", "fill_close"],
                "reason": "closed_trade_from_placeholder_open_fill",
            }
        ],
    }
    _write_sidecar(tmp_path / "trades", "quarantine_test.json", sidecar)

    assert qmod.is_quarantined_trade(
        record_hash="trade_hash_a", trades_dir=tmp_path / "trades"
    )
    assert qmod.is_quarantined_trade(
        source_file="trade_history_20260508.ndjson",
        sequence_id=5,
        trades_dir=tmp_path / "trades",
    )
    assert not qmod.is_quarantined_trade(
        record_hash="other", trades_dir=tmp_path / "trades"
    )


def test_quarantine_loader_fails_open_on_malformed_sidecar(tmp_path, caplog):
    """Corrupt JSON / wrong shape must log a warning and return empty sets."""
    from chad.analytics import quarantine as qmod

    folder = tmp_path / "fills"
    folder.mkdir()
    (folder / "quarantine_corrupt.json").write_text("not json {")
    (folder / "quarantine_badshape.json").write_text(json.dumps([1, 2, 3]))

    with caplog.at_level("WARNING", logger="chad.analytics.quarantine"):
        out = qmod.load_fills_sidecar(fills_dir=folder)
    assert out == []
    assert any("quarantine_sidecar" in rec.message for rec in caplog.records)


def test_get_sidecar_exclusion_sets_unions_fill_ids_from_trades(tmp_path):
    from chad.analytics import quarantine as qmod

    fills_sidecar = {
        "invalid_fills": [
            {"fill_id": "F1", "record_hash": "RHF1"},
        ]
    }
    trades_sidecar = {
        "invalid_trades": [
            {"record_hash": "T_HASH_1", "fill_ids": ["F2", "F3"]},
        ]
    }
    _write_sidecar(tmp_path / "fills", "quarantine_a.json", fills_sidecar)
    _write_sidecar(tmp_path / "trades", "quarantine_b.json", trades_sidecar)

    fills, trades = qmod.get_sidecar_exclusion_sets(
        fills_dir=tmp_path / "fills",
        trades_dir=tmp_path / "trades",
    )
    assert fills == {"F1", "F2", "F3"}
    assert trades == {"T_HASH_1"}


# ---------------------------------------------------------------------------
# 3. per_strategy_loss_guard excludes quarantined delta trades
# ---------------------------------------------------------------------------


def _write_trade(
    folder: Path,
    fname: str,
    seq: int,
    record_hash: str,
    payload: Dict[str, Any],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / fname
    rec = {
        "payload": payload,
        "prev_hash": "GENESIS",
        "sequence_id": seq,
        "timestamp_utc": payload.get("exit_time_utc", ""),
        "record_hash": record_hash,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def test_per_strategy_loss_guard_excludes_quarantined_delta_trades(
    tmp_path, monkeypatch
):
    """compute_today_realized_pnl must drop trades pinned in the trades
    sidecar — a poisoned -$5000 delta record must not count toward today's
    delta PnL while a clean +$10 delta record still does."""
    from datetime import datetime, timezone

    from chad.risk import per_strategy_loss_guard as guard

    trades_dir = tmp_path / "trades"
    fills_dir = tmp_path / "fills"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    now = datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc)
    today_iso = now.isoformat()

    # Poisoned trade (entry=100, exit=731, pnl=-5000)
    _write_trade(
        trades_dir,
        "trade_history_20260508.ndjson",
        seq=1,
        record_hash="POISON_HASH_1",
        payload={
            "schema_version": "closed_trade.v1",
            "strategy": "delta",
            "symbol": "SPY",
            "side": "SELL",
            "pnl": -5000.0,
            "entry_price": 100.0,
            "exit_price": 731.44,
            "quantity": 10.0,
            "exit_time_utc": today_iso,
            "fill_ids": ["fill_open_a", "fill_close_a"],
            "tags": ["paper", "closed", "delta"],
        },
    )
    # Clean trade (real prices, +$10)
    _write_trade(
        trades_dir,
        "trade_history_20260508.ndjson",
        seq=2,
        record_hash="GOOD_HASH_1",
        payload={
            "schema_version": "closed_trade.v1",
            "strategy": "delta",
            "symbol": "SPY",
            "side": "BUY",
            "pnl": 10.0,
            "entry_price": 730.0,
            "exit_price": 731.0,
            "quantity": 10.0,
            "exit_time_utc": today_iso,
            "fill_ids": ["fill_open_b", "fill_close_b"],
            "tags": ["paper", "closed", "delta"],
        },
    )

    # Sidecar quarantines the poisoned trade by record_hash.
    _write_sidecar(
        trades_dir,
        "quarantine_test.json",
        {
            "invalid_trades": [
                {
                    "source_file": "trade_history_20260508.ndjson",
                    "sequence_id": 1,
                    "record_hash": "POISON_HASH_1",
                    "pnl": -5000.0,
                    "fill_ids": ["fill_open_a", "fill_close_a"],
                    "reason": "closed_trade_from_placeholder_open_fill",
                }
            ]
        },
    )

    # Patch module-level paths the guard's inline imports rely on.
    monkeypatch.setenv("CHAD_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("CHAD_FILLS_DIR", str(fills_dir))
    monkeypatch.setenv("CHAD_TRADES_DIR", str(trades_dir))
    monkeypatch.setattr(guard, "TRADES_DIR", trades_dir)
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)

    out = guard.compute_today_realized_pnl(now=now, trades_dir=trades_dir, fills_dir=fills_dir)
    # Only the clean +$10 trade should count.
    assert out.get("delta") == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 4. trade_stats_engine excludes quarantined trades but keeps clean ones
# ---------------------------------------------------------------------------


def test_trade_stats_engine_excludes_quarantined_trades(tmp_path, monkeypatch):
    """load_and_compute must skip records whose record_hash is in a sidecar."""
    from chad.analytics import trade_stats_engine as tse

    trades_dir = tmp_path / "data" / "trades"
    fills_dir = tmp_path / "data" / "fills"
    runtime_dir = tmp_path / "runtime"
    trades_dir.mkdir(parents=True, exist_ok=True)
    fills_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # 1 quarantined poisoned trade + 1 clean trade.
    _write_trade(
        trades_dir,
        "trade_history_20260508.ndjson",
        seq=1,
        record_hash="POISON_HASH",
        payload={
            "schema_version": "closed_trade.v1",
            "strategy": "delta",
            "symbol": "SPY",
            "side": "SELL",
            "pnl": -5000.0,
            "notional": 1000.0,
            "entry_price": 100.0,
            "exit_price": 731.44,
            "quantity": 10.0,
            "exit_time_utc": "2026-05-08T18:00:00+00:00",
            "fill_ids": ["P1", "P2"],
            "tags": ["paper", "closed", "delta"],
            "broker": "ibkr_paper",
            "is_live": False,
        },
    )
    _write_trade(
        trades_dir,
        "trade_history_20260508.ndjson",
        seq=2,
        record_hash="CLEAN_HASH",
        payload={
            "schema_version": "closed_trade.v1",
            "strategy": "delta",
            "symbol": "SPY",
            "side": "BUY",
            "pnl": 25.0,
            "notional": 7300.0,
            "entry_price": 730.0,
            "exit_price": 732.5,
            "quantity": 10.0,
            "exit_time_utc": "2026-05-08T18:30:00+00:00",
            "fill_ids": ["C1", "C2"],
            "tags": ["paper", "closed", "delta"],
            "broker": "ibkr_paper",
            "is_live": False,
        },
    )
    _write_sidecar(
        trades_dir,
        "quarantine_test.json",
        {
            "invalid_trades": [
                {"record_hash": "POISON_HASH", "fill_ids": ["P1", "P2"]}
            ]
        },
    )

    # CHAD_REPO_ROOT controls where trade_stats_engine looks for ledgers,
    # and CHAD_RUNTIME_DIR / CHAD_FILLS_DIR control the quarantine helper.
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("CHAD_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("CHAD_FILLS_DIR", str(fills_dir))
    monkeypatch.setenv("CHAD_TRADES_DIR", str(trades_dir))

    stats = tse.load_and_compute(
        max_trades=100,
        days_back=60,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    # The poisoned record must be excluded.
    assert stats["excluded_quarantined"] >= 1
    # total_pnl should only see the clean +$25.
    assert stats["total_pnl"] == pytest.approx(25.0)


def test_trade_stats_engine_keeps_non_quarantined_records(tmp_path, monkeypatch):
    """A clean trade with no sidecar entry must still be counted."""
    from chad.analytics import trade_stats_engine as tse

    trades_dir = tmp_path / "data" / "trades"
    runtime_dir = tmp_path / "runtime"
    trades_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    _write_trade(
        trades_dir,
        "trade_history_20260508.ndjson",
        seq=1,
        record_hash="OK_HASH",
        payload={
            "schema_version": "closed_trade.v1",
            "strategy": "alpha",
            "symbol": "MES",
            "side": "BUY",
            "pnl": 200.0,
            "notional": 7100.0,
            "entry_price": 7100.0,
            "exit_price": 7140.0,
            "quantity": 1.0,
            "exit_time_utc": "2026-05-08T18:30:00+00:00",
            "fill_ids": ["A1", "A2"],
            "tags": ["paper", "closed", "alpha"],
            "broker": "ibkr_paper",
            "is_live": False,
        },
    )

    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("CHAD_RUNTIME_DIR", str(runtime_dir))

    stats = tse.load_and_compute(
        max_trades=100,
        days_back=60,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    assert stats["total_trades"] >= 1
    assert stats["total_pnl"] == pytest.approx(200.0)
    assert stats.get("excluded_quarantined", 0) == 0

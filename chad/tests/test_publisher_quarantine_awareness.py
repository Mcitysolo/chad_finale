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


# ---------------------------------------------------------------------------
# Derived closed-trade contamination — exclusion via untrusted fill IDs
# ---------------------------------------------------------------------------


def _untrusted_fill_record(*, fill_id: str, marker: str = "extra") -> dict:
    """Construct a FILLS_*.ndjson record carrying a pnl_untrusted marker.

    ``marker`` selects which form of the marker to apply:
      * ``"extra"``    -> payload.extra.pnl_untrusted = True
      * ``"payload"``  -> payload.pnl_untrusted = True
      * ``"tag"``      -> "pnl_untrusted" in payload.tags
    """
    payload = {
        "broker": "ibkr_paper",
        "is_live": False,
        "strategy": "delta",
        "symbol": "SPY",
        "side": "SELL",
        "quantity": 10.0,
        "fill_price": 100.0,
        "notional": 1000.0,
        "fill_id": fill_id,
        "status": "rejected",
        "tags": ["paper", "filled", "delta"],
        "extra": {},
    }
    if marker == "extra":
        payload["extra"]["pnl_untrusted"] = True
    elif marker == "payload":
        payload["pnl_untrusted"] = True
    elif marker == "tag":
        payload["tags"] = list(payload["tags"]) + ["pnl_untrusted"]
    else:
        raise ValueError(f"unknown marker: {marker}")
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": f"FILL_REC_{fill_id}",
    }


def _derived_closed_trade_record(
    *,
    record_hash: str,
    fill_ids: list[str],
    pnl: float,
    entry_price: float = 100.0,
    exit_price: float = 720.04,
) -> dict:
    """Closed-trade record matching trade_closer.ClosedTrade.to_payload().

    The row itself has *no* pnl_untrusted marker — the only signal of
    contamination is that its fill_ids reference untrusted fills.
    """
    payload = {
        "schema_version": "closed_trade.v1",
        "strategy": "delta",
        "symbol": "SPY",
        "side": "SELL",
        "broker": "paper_exec",
        "is_live": False,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "fill_price": exit_price,
        "quantity": 10.0,
        "contract_multiplier": 1.0,
        "notional": entry_price * 10.0,
        "pnl": pnl,
        "net_pnl": pnl,
        "fill_ids": list(fill_ids),
        "tags": ["paper", "closed", "delta"],
    }
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": record_hash,
    }


def test_untrusted_fill_id_set_built_from_fills(tmp_path: Path) -> None:
    """get_untrusted_fill_ids_from_fills must collect fill_ids carrying
    any of the three pnl_untrusted markers (payload, payload.extra,
    payload.tags) and ignore trusted fills."""
    from chad.utils.quarantine import get_untrusted_fill_ids_from_fills

    fills_dir = tmp_path / "data" / "fills"
    fills_dir.mkdir(parents=True)

    today = _today_ymd()
    fills = [
        _untrusted_fill_record(fill_id="UNT_EXTRA", marker="extra"),
        _untrusted_fill_record(fill_id="UNT_PAYLOAD", marker="payload"),
        _untrusted_fill_record(fill_id="UNT_TAG", marker="tag"),
        _fill_record(record_hash="FILL_OK", fill_id="GOOD_FILL"),
    ]
    (fills_dir / f"FILLS_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in fills) + "\n", encoding="utf-8"
    )

    out = get_untrusted_fill_ids_from_fills(fills_dir=fills_dir)
    assert "UNT_EXTRA" in out
    assert "UNT_PAYLOAD" in out
    assert "UNT_TAG" in out
    assert "GOOD_FILL" not in out


def test_trade_stats_excludes_trade_referencing_untrusted_fill_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A derived closed trade with no pnl_untrusted marker must still
    be excluded from SCR/trade_stats when any of its fill_ids points
    at an untrusted FILLS_*.ndjson row."""
    from chad.analytics import trade_stats_engine

    trades_dir = tmp_path / "data" / "trades"
    fills_dir = tmp_path / "data" / "fills"
    runtime_dir = tmp_path / "runtime"
    trades_dir.mkdir(parents=True)
    fills_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

    today = _today_ymd()
    # Untrusted opening fill — placeholder $100 SPY.
    fills = [_untrusted_fill_record(fill_id="UNT_ENTRY", marker="extra")]
    (fills_dir / f"FILLS_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in fills) + "\n", encoding="utf-8"
    )

    trades = [
        _derived_closed_trade_record(
            record_hash="DERIVED_CONTAMINATED",
            fill_ids=["UNT_ENTRY", "TRUSTED_EXIT"],
            pnl=-3720.24,
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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Contaminated derived trade dropped; only the +25 trusted row remains.
    assert stats["total_trades"] == 1, stats
    assert stats["total_pnl"] == 25.0, stats
    assert stats["excluded_quarantined"] >= 1, stats
    assert stats["excluded_untrusted"] >= 1, stats


def test_profit_lock_excludes_trade_referencing_untrusted_fill_id(
    tmp_path: Path,
) -> None:
    """NdjsonPnlProvider must skip a derived closed trade whose
    fill_ids reference an untrusted FILLS_*.ndjson row, even when
    the closed-trade record itself carries no pnl_untrusted marker."""
    from chad.risk.profit_lock import NdjsonPnlProvider

    repo_root = tmp_path
    trades_dir = repo_root / "data" / "trades"
    fills_dir = repo_root / "data" / "fills"
    trades_dir.mkdir(parents=True)
    fills_dir.mkdir(parents=True)

    today = _today_ymd()

    fills = [_untrusted_fill_record(fill_id="UNT_ENTRY", marker="extra")]
    (fills_dir / f"FILLS_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in fills) + "\n", encoding="utf-8"
    )

    trades = [
        _derived_closed_trade_record(
            record_hash="DERIVED_CONTAMINATED",
            fill_ids=["UNT_ENTRY", "TRUSTED_EXIT"],
            pnl=-3720.24,
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

    provider = NdjsonPnlProvider()
    pnl, _count, _sources = asyncio.run(
        provider.get_realized_pnl(repo_root, days=0)
    )

    # The -3720.24 derived contaminated trade must NOT enter pnl_state.
    # Untrusted fill rows are also dropped. Only the +25 trusted trade
    # contributes; remaining files (e.g. the FILLS file's untrusted row)
    # are excluded entirely.
    assert pnl == 25.0, f"expected 25.0 (contaminated trade dropped), got {pnl}"


def test_expectancy_excludes_trade_referencing_untrusted_fill_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """expectancy_tracker must skip a derived closed trade whose
    fill_ids reference an untrusted FILLS_*.ndjson row."""
    from chad.analytics import expectancy_tracker

    trades_dir = tmp_path / "data" / "trades"
    fills_dir = tmp_path / "data" / "fills"
    runtime_dir = tmp_path / "runtime"
    trades_dir.mkdir(parents=True)
    fills_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    today = _today_ymd()

    fills = [_untrusted_fill_record(fill_id="UNT_ENTRY", marker="tag")]
    (fills_dir / f"FILLS_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in fills) + "\n", encoding="utf-8"
    )

    trades = [
        _derived_closed_trade_record(
            record_hash="DERIVED_CONTAMINATED",
            fill_ids=["UNT_ENTRY", "TRUSTED_EXIT"],
            pnl=-3720.24,
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

    # Derived contaminated trade dropped; only +25 remains.
    assert delta.get("total_trades") == 1, state
    assert delta.get("total_pnl") == 25.0, state
    assert state["total_clean_trades"] == 1, state

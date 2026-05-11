"""
Tests for the 2026-05-11 alpha_options synthetic BAG-close quarantine.

These tests pin the four behaviours that runtime/quarantine_manifest_20260511.json
must guarantee for SCR / pnl_state / strategy_health publishers:

  1. A closed_trade row whose record_hash is listed in invalid_trades is
     excluded from SCR aggregation.
  2. A closed_trade row whose payload.fill_ids contains an invalid_fill_id
     (the synthetic qty=304 SELL on 2026-05-11) is excluded even when its
     own record_hash is NOT in the manifest.
  3. An unrelated alpha_options closed_trade row is preserved (no
     over-blocking).
  4. With the manifest present, SCR stats over a contaminated fixture
     recover: total_pnl flips from negative back to positive and
     effective_trades drops by the count of phantom rows.

The tests use the established `monkeypatch.chdir(tmp_path)` + `CHAD_REPO_ROOT`
pattern (see chad/tests/test_publisher_quarantine_awareness.py) so they
never touch real runtime/ or data/ files.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


BAD_FILL_ID = "cc2cd31ee72a81de2541b2e6791db458d511b26522e874b57806f3ccbae02e29"


def _today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _closed_trade(
    *,
    record_hash: str,
    pnl: float,
    fill_ids: list[str],
    strategy: str = "alpha_options",
    symbol: str = "SPY",
    entry_price: float = 400.0,
    exit_price: float = 120.0,
    quantity: float = 1.0,
) -> dict:
    payload = {
        "schema_version": "closed_trade.v1",
        "broker": "paper_exec",
        "is_live": False,
        "strategy": strategy,
        "symbol": symbol,
        "side": "BUY",
        "quantity": quantity,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "fill_price": exit_price,
        "pnl": pnl,
        "notional": entry_price * quantity,
        "contract_multiplier": 1.0,
        "fill_ids": fill_ids,
        "tags": ["paper", "closed", strategy],
    }
    return {
        "timestamp_utc": "2026-05-11T15:44:38.932584+00:00",
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": record_hash,
    }


def _write_manifest_20260511(
    runtime_dir: Path,
    *,
    extra_trade_hashes: list[str] | None = None,
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "quarantine_manifest_20260511.json"
    doc = {
        "schema_version": "quarantine_manifest.v1",
        "quarantined_at_utc": "2026-05-11T19:30:00Z",
        "reason": "synthetic_bag_close_credit_30pct_not_market_quote",
        "incident_date": "2026-05-11",
        "detected_by": "SCR_PAUSED_ROOT_CAUSE_AUDIT",
        "invalid_fills": [
            {
                "fill_id": BAD_FILL_ID,
                "strategy": "alpha_options",
                "symbol": "SPY",
                "side": "SELL",
                "quantity": 304.0,
                "fill_price": 120.0,
                "source": "paper_trade_executor",
                "reason": "synthetic_bag_close_credit_30pct_not_market_quote",
            }
        ],
        "invalid_trades": [
            {"record_hash": rh, "reason": "synthetic_bag_close_credit_30pct_not_market_quote"}
            for rh in (extra_trade_hashes or [])
        ],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_manifest_excludes_closed_trade_by_record_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record_hash listed under invalid_trades is dropped from SCR pool."""
    from chad.analytics import trade_stats_engine

    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)
    today = _today_ymd()

    bad_hash = "PHANTOM_BAG_CLOSE_HASH_BY_RECORD_HASH_MATCH"
    rows = [
        _closed_trade(
            record_hash=bad_hash,
            pnl=-280.0,
            # Use an unrelated fill_id so only the record_hash gate exercises
            fill_ids=["unrelated_entry_fid", "unrelated_exit_fid"],
        ),
        _closed_trade(
            record_hash="GOOD_ALPHA_OPTIONS_ROW",
            pnl=42.0,
            fill_ids=["other_entry_fid", "other_exit_fid"],
            entry_price=100.0,
            exit_price=142.0,
        ),
    ]
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    _write_manifest_20260511(tmp_path / "runtime", extra_trade_hashes=[bad_hash])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    assert stats["excluded_quarantined"] >= 1, stats
    assert stats["total_trades"] == 1, stats
    assert stats["total_pnl"] == 42.0, stats


def test_manifest_excludes_closed_trade_referencing_bad_fill_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed_trade row whose payload.fill_ids contains the bad fill_id
    is excluded even when its own record_hash is not pinned in the manifest.

    This is the load-bearing gate for SCR recovery: the manifest lists ONE
    fill_id and the engine must drop every derived closed_trade that
    references it.
    """
    from chad.analytics import trade_stats_engine

    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)
    today = _today_ymd()

    rows = [
        _closed_trade(
            record_hash="DERIVED_FROM_BAD_FILL_NO_HASH_PIN",
            pnl=-280.0,
            fill_ids=["entry_fid_xyz", BAD_FILL_ID],  # bad exit fill
        ),
        _closed_trade(
            record_hash="ANOTHER_DERIVED_FROM_BAD_FILL",
            pnl=-4627.575,
            fill_ids=["entry_fid_abc", BAD_FILL_ID],
            quantity=7.5,
            entry_price=737.01,
        ),
        _closed_trade(
            record_hash="UNRELATED_ALPHA_OPTIONS_TRADE",
            pnl=15.50,
            fill_ids=["clean_entry", "clean_exit"],
            entry_price=10.0,
            exit_price=25.5,
        ),
    ]
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    # NO record_hash pins — exclusion must rely on the fill_id alone.
    _write_manifest_20260511(tmp_path / "runtime", extra_trade_hashes=[])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Two contaminated rows must be dropped via the fill_id path.
    assert stats["excluded_quarantined"] >= 2, stats
    # Only the clean +15.50 row survives.
    assert stats["total_trades"] == 1, stats
    assert stats["total_pnl"] == pytest.approx(15.50), stats


def test_manifest_does_not_over_block_unrelated_alpha_options_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated alpha_options closed trades must remain in the SCR pool.

    Regression guard: a coarse strategy/symbol filter would over-block.
    The manifest must only exclude rows referencing the bad fill_id or
    a pinned record_hash.
    """
    from chad.analytics import trade_stats_engine

    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)
    today = _today_ymd()

    rows = [
        _closed_trade(
            record_hash="CLEAN_ALPHA_OPT_GAIN",
            pnl=120.5,
            fill_ids=["clean_open_1", "clean_close_1"],
            entry_price=10.0,
            exit_price=130.5,
        ),
        _closed_trade(
            record_hash="CLEAN_ALPHA_OPT_LOSS",
            pnl=-25.0,
            fill_ids=["clean_open_2", "clean_close_2"],
            entry_price=100.0,
            exit_price=75.0,
        ),
        # Row that DOES reference the bad fill_id — must be the only drop.
        _closed_trade(
            record_hash="CONTAMINATED",
            pnl=-280.0,
            fill_ids=["entry_x", BAD_FILL_ID],
        ),
    ]
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    _write_manifest_20260511(tmp_path / "runtime", extra_trade_hashes=[])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    # Two clean alpha_options rows survive; one contaminated row dropped.
    assert stats["total_trades"] == 2, stats
    assert stats["excluded_quarantined"] >= 1, stats
    assert stats["total_pnl"] == pytest.approx(120.5 - 25.0), stats


def test_scr_stats_recover_when_manifest_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a contaminated fixture flips back to a healthy SCR
    profile once the 2026-05-11 manifest is dropped in.

    Fixture shape mirrors the production incident: many small phantom
    losses (entry=400, exit=120, qty=1, pnl=-280) attached to a single
    bad exit fill_id, plus a sprinkling of legitimate winning trades
    from other strategies.
    """
    from chad.analytics import trade_stats_engine

    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)
    runtime_dir = tmp_path / "runtime"
    today = _today_ymd()

    phantom_count = 20
    phantom_rows = [
        _closed_trade(
            record_hash=f"PHANTOM_{i:03d}",
            pnl=-280.0,
            fill_ids=[f"entry_{i:03d}", BAD_FILL_ID],
        )
        for i in range(phantom_count)
    ]
    clean_rows = [
        _closed_trade(
            record_hash="CLEAN_ALPHA_FUT_WIN",
            pnl=993.75,
            fill_ids=["af_open", "af_close"],
            strategy="alpha_futures",
            symbol="MES",
            entry_price=5500.0,
            exit_price=5520.0,
            quantity=1.0,
        ),
        _closed_trade(
            record_hash="CLEAN_BROKER_SYNC",
            pnl=366.72,
            fill_ids=["bs_open", "bs_close"],
            strategy="broker_sync",
            symbol="SPY",
            entry_price=720.0,
            exit_price=735.0,
            quantity=1.0,
        ),
        _closed_trade(
            record_hash="CLEAN_ALPHA_WIN",
            pnl=155.28,
            fill_ids=["a_open", "a_close"],
            strategy="alpha",
            symbol="SPY",
            entry_price=720.0,
            exit_price=735.0,
            quantity=1.0,
        ),
    ]
    rows = phantom_rows + clean_rows
    (trades_dir / f"trade_history_{today}.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    # ---- WITHOUT manifest ----
    before = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )
    expected_phantom_sum = -280.0 * phantom_count
    expected_clean_sum = 993.75 + 366.72 + 155.28
    assert before["total_pnl"] == pytest.approx(
        expected_phantom_sum + expected_clean_sum
    ), before
    assert before["total_pnl"] < 0.0, before  # contaminated -> negative

    # ---- WITH manifest ----
    _write_manifest_20260511(runtime_dir, extra_trade_hashes=[])

    after = trade_stats_engine.load_and_compute(
        max_trades=100, days_back=2, include_paper=True, include_live=True
    )

    assert after["total_pnl"] == pytest.approx(expected_clean_sum), after
    assert after["total_pnl"] > 0.0, after
    # All phantom_count rows dropped via the bad fill_id reference.
    assert after["excluded_quarantined"] >= phantom_count, after
    assert after["total_trades"] == len(clean_rows), after
    # Recovery delta matches the phantom contribution exactly.
    assert after["total_pnl"] - before["total_pnl"] == pytest.approx(
        -expected_phantom_sum
    )

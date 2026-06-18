from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chad.analytics.trade_stats_engine import load_and_compute


def _env(payload, seq=1):
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": seq,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": f"rh-{seq}",
    }


def _validate_only_kraken_paper_payload(seq: int = 1, pnl: float = 0.0):
    # Mirrors the row written by kraken_execution.py / kraken_trade_result_logger.py
    # for kraken_paper validate-only audit fills.
    return {
        "strategy": "alpha_crypto",
        "symbol": "SOL-USD",
        "side": "BUY",
        "quantity": 0.5,
        "fill_price": 92.0,
        "notional": 46.0,
        "pnl": pnl,
        "entry_time_utc": "2099-01-01T00:00:00+00:00",
        "exit_time_utc": "2099-01-01T00:00:00+00:00",
        "is_live": False,
        "broker": "kraken_paper",
        "tags": [
            "kraken_paper",
            "pnl_untrusted",
            "validate_only",
            "alpha_crypto",
            "buy",
            "limit",
        ],
        "extra": {
            "source": "kraken_paper_evidence",
            "txid": f"PAPER-KRAKEN-{seq:016x}",
            "validate_only": True,
            "pnl_untrusted": True,
            "pnl_untrusted_reason": "kraken_paper_validate_only_no_realized_fill",
        },
    }


def _untrusted_non_validate_only_payload(seq: int = 1):
    # IBKR paper row marked pnl_untrusted but NOT validate_only — should
    # still be counted under excluded_untrusted, not excluded_validate_only.
    return {
        "strategy": "alpha_futures",
        "symbol": "MES",
        "side": "SELL",
        "quantity": 1.0,
        "fill_price": 5000.0,
        "notional": 25000.0,
        "pnl": 0.0,
        "entry_time_utc": "2099-01-01T00:00:00+00:00",
        "exit_time_utc": "2099-01-01T00:00:00+00:00",
        "is_live": False,
        "broker": "ibkr",
        "tags": ["ibkr_paper", "filled", "pnl_untrusted"],
        "extra": {
            "source": "ibkr_paper_ledger_watcher",
            "pnl_untrusted": True,
            "pnl_untrusted_reason": "symbol_close_detected_without_fill_matcher",
        },
    }


def _trusted_closed_payload(seq: int = 1, pnl: float = 12.5):
    # Equity control row (a generic trusted closed trade). Uses an equity symbol
    # so it survives the item-5b futures exclusion — the symbol is incidental to
    # these validate-only tests.
    return {
        "strategy": "alpha",
        "symbol": "AAPL",
        "side": "SELL",
        "quantity": 1.0,
        "fill_price": 250.0,
        "notional": 250.0,
        "pnl": pnl,
        "entry_time_utc": "2099-01-01T00:00:00+00:00",
        "exit_time_utc": "2099-01-01T00:00:00+00:00",
        "is_live": False,
        "broker": "ibkr",
        "tags": ["ibkr_paper", "filled"],
        "extra": {"source": "ibkr_paper_ledger_watcher"},
    }


def _write_today(tmp_path: Path, rows):
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    f = trades_dir / f"trade_history_{today}.ndjson"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return f


def test_validate_only_does_not_affect_total_pnl(tmp_path, monkeypatch):
    rows = [
        _env(_validate_only_kraken_paper_payload(seq=1), seq=1),
        _env(_validate_only_kraken_paper_payload(seq=2), seq=2),
        _env(_trusted_closed_payload(seq=3, pnl=7.0), seq=3),
    ]
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    # validate_only rows have pnl=0; total_pnl reflects only the trusted close.
    assert stats["total_pnl"] == 7.0


def test_validate_only_does_not_count_as_effective_trade(tmp_path, monkeypatch):
    rows = [
        _env(_validate_only_kraken_paper_payload(seq=i), seq=i)
        for i in range(1, 6)
    ]
    rows.append(_env(_trusted_closed_payload(seq=99, pnl=4.0), seq=99))
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    assert stats["effective_trades"] == 1
    assert stats["total_trades"] == 6


def test_validate_only_increments_excluded_validate_only_not_untrusted(tmp_path, monkeypatch):
    rows = [
        _env(_validate_only_kraken_paper_payload(seq=i), seq=i)
        for i in range(1, 4)
    ]
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    assert stats["excluded_validate_only"] == 3
    # Without quarantined rows present, excluded_untrusted should be 0:
    # validate_only must not pollute the truly-untrusted bucket.
    assert stats["excluded_untrusted"] == 0


def test_non_validate_only_untrusted_still_counts_as_untrusted(tmp_path, monkeypatch):
    rows = [
        _env(_untrusted_non_validate_only_payload(seq=1), seq=1),
        _env(_untrusted_non_validate_only_payload(seq=2), seq=2),
        _env(_validate_only_kraken_paper_payload(seq=3), seq=3),
    ]
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    assert stats["excluded_untrusted"] == 2
    assert stats["excluded_validate_only"] == 1
    assert stats["effective_trades"] == 0


def test_trusted_closed_trade_still_counts_normally(tmp_path, monkeypatch):
    rows = [
        _env(_trusted_closed_payload(seq=1, pnl=10.0), seq=1),
        _env(_trusted_closed_payload(seq=2, pnl=-3.0), seq=2),
        _env(_validate_only_kraken_paper_payload(seq=3), seq=3),
    ]
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
        epoch_filter=False,
    )
    assert stats["effective_trades"] == 2
    assert stats["total_pnl"] == 7.0
    assert stats["win_rate"] == 0.5
    assert stats["excluded_validate_only"] == 1
    assert stats["excluded_untrusted"] == 0

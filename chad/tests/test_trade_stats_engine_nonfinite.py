from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chad.analytics.trade_stats_engine import load_and_compute


def _env(payload):
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": 1,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": "x",
    }


def test_load_and_compute_skips_nonfinite_pnl_and_notional(tmp_path: Path, monkeypatch) -> None:
    # Create a synthetic trade history file with bad rows, but name it for TODAY
    # so trade_stats_engine's date-based iterator will include it.
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    f = trades_dir / f"trade_history_{today}.ndjson"

    rows = [
        # NaN pnl -> must be skipped
        _env(
            {
                "strategy": "manual",
                "symbol": "EUR",
                "side": "BUY",
                "quantity": 1.0,
                "fill_price": 1.0,
                "notional": 100.0,
                "pnl": float("nan"),
                "entry_time_utc": "2099-01-01T00:00:00+00:00",
                "exit_time_utc": "2099-01-01T00:00:01+00:00",
                "is_live": False,
                "broker": "ibkr",
                "tags": ["ibkr_paper"],
                "extra": {"source": "ibkr_paper_ledger_watcher"},
            }
        ),
        # Inf notional -> must be skipped
        _env(
            {
                "strategy": "manual",
                "symbol": "EUR",
                "side": "BUY",
                "quantity": 1.0,
                "fill_price": 1.0,
                "notional": float("inf"),
                "pnl": -1.0,
                "entry_time_utc": "2099-01-01T00:00:00+00:00",
                "exit_time_utc": "2099-01-01T00:00:02+00:00",
                "is_live": False,
                "broker": "ibkr",
                "tags": ["ibkr_paper"],
                "extra": {"source": "ibkr_paper_ledger_watcher"},
            }
        ),
        # Valid trade -> must be counted
        _env(
            {
                "strategy": "manual",
                "symbol": "EUR",
                "side": "BUY",
                "quantity": 1.0,
                "fill_price": 1.0,
                "notional": 100.0,
                "pnl": -0.5,
                "entry_time_utc": "2099-01-01T00:00:00+00:00",
                "exit_time_utc": "2099-01-01T00:00:03+00:00",
                "is_live": False,
                "broker": "ibkr",
                "tags": ["ibkr_paper"],
                "extra": {"source": "ibkr_paper_ledger_watcher"},
            }
        ),
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Point trade_stats_engine to our temp trade dir by changing CWD.
    monkeypatch.chdir(tmp_path)

    stats = load_and_compute(max_trades=500, days_back=30, include_paper=True, include_live=True)
    assert stats["total_trades"] == 1
    assert stats["paper_trades"] == 1
    assert stats["live_trades"] == 0
    assert stats["total_pnl"] == -0.5

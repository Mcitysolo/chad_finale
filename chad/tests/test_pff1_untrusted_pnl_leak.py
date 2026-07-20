"""PFF1 (2026-07-20) — total_pnl must exclude pnl_untrusted rows exactly as
effective_trades does.

On 2026-07-20 an Epoch-3 seed-lot close carrying a fabricated cost basis booked
+625.17 tagged pnl_untrusted / scoring_excluded. effective_trades correctly
excluded it (excluded_untrusted=1), but total_pnl summed pnl over ALL finite
trades, so the fabricated gain leaked into the headline and flipped it from
-375.60 to +103.78. This pins the headline to trusted-only PnL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.trade_stats_engine import load_and_compute


def _env(payload, seq):
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": seq,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": f"rh-{seq}",
    }


def _trusted_close(seq, pnl):
    return _env({
        "schema_version": "closed_trade.v1",
        "strategy": "gamma",
        "symbol": "UNH",
        "side": "SELL",
        "quantity": abs(pnl) + 1.0,   # any finite non-zero qty
        "fill_price": 425.57,
        "entry_price": 424.87,
        "exit_price": 425.57,
        "notional": 10000.0,
        "pnl": pnl,
        "is_live": False,
        "broker": "paper_exec",
        "account_id": "PAPER_EXEC",
        "tags": ["paper", "closed", "gamma"],
    }, seq)


def _untrusted_seed_close(seq, pnl):
    # Faithful shape of the 2026-07-20 seq1 seed-lot close: markers on tags,
    # extra AND meta (the trade_closer B2 mirror writes all three).
    return _env({
        "schema_version": "closed_trade.v1",
        "strategy": "gamma",
        "symbol": "UNH",
        "side": "BUY",
        "quantity": 273.0,
        "fill_price": 423.0,
        "entry_price": 420.71,
        "exit_price": 423.0,
        "notional": 114853.83,
        "pnl": pnl,
        "is_live": False,
        "broker": "paper_exec",
        "account_id": "PAPER_EXEC",
        "tags": ["paper", "closed", "gamma", "pnl_untrusted", "scoring_excluded"],
        "extra": {
            "pnl_untrusted": True,
            "scoring_excluded": True,
            "provenance": "UNATTRIBUTED_EPOCH3_ACCUMULATION",
        },
        "meta": {
            "pnl_untrusted": True,
            "scoring_excluded": True,
            "provenance": "UNATTRIBUTED_EPOCH3_ACCUMULATION",
            "seeded_from": "broker_truth",
        },
    }, seq)


def test_untrusted_seed_close_excluded_from_total_pnl(tmp_path, monkeypatch):
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True)

    # 6 trusted gamma closes summing to -145.79 (the 2026-07-20 real values),
    # plus the untrusted seed close of +625.17.
    trusted_pnls = [12.05, -3.55, -22.72, -28.0, -56.0, -47.57]
    rows = [_trusted_close(i + 1, p) for i, p in enumerate(trusted_pnls)]
    rows.append(_untrusted_seed_close(len(rows) + 1, 625.17))

    (trades_dir / "trade_history_20260720.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    stats = load_and_compute(max_trades=5000, days_back=60,
                             include_paper=True, include_live=True)

    trusted_only = sum(trusted_pnls)  # -145.79

    # The untrusted seed close is excluded from effective_trades ...
    assert stats["effective_trades"] == 6
    assert stats["excluded_untrusted"] == 1
    # ... and MUST be excluded from total_pnl too. Pre-fix this was
    # -145.79 + 625.17 = +479.38 (the leak).
    assert stats["total_pnl"] == pytest.approx(trusted_only)
    assert stats["total_pnl"] == pytest.approx(-145.79)
    # Sanity: the leaked value must NOT be what we report.
    assert stats["total_pnl"] != pytest.approx(479.38)

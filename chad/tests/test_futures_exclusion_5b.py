"""Item 5b — futures exclusion from the SCR confidence sample.

The effective performance sample (trade_stats_engine -> SCR band) and the
per-strategy expectancy table must drop futures rows so Bug-B futures
contamination cannot inflate effective_trades or degrade win_rate / sharpe /
drawdown. Both consumers share one canonical classifier
(chad.analytics.futures_classifier.is_futures_row).

Covers:
  1. Classifier unit behaviour (roots, dated contracts, secType, false-positives).
  2. load_and_compute excludes futures (root + secType=FUT), keeps equities.
  3. pnl_untrusted precedence: an untrusted futures row counts as untrusted,
     not as excluded_futures (drop order is documented behaviour).
  4. expectancy_tracker._iter_trades mirrors the exclusion.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chad.analytics.futures_classifier import is_futures_row
from chad.analytics.trade_stats_engine import load_and_compute


# ---------------------------------------------------------------------------
# 1. Classifier unit behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    ["MES", "MNQ", "MYM", "M2K", "M6E", "MGC", "MCL", "mes", " M6E "],
)
def test_classifier_matches_every_micro_root(symbol):
    assert is_futures_row(symbol) is True


@pytest.mark.parametrize("symbol", ["MESZ5", "M6EM26", "MCLN5", "M2KH6"])
def test_classifier_matches_dated_contracts(symbol):
    assert is_futures_row(symbol) is True


@pytest.mark.parametrize(
    "symbol",
    ["AAPL", "SPY", "QQQ", "MA", "MESA", "ES", "NQ", "SOL-USD", "", None],
)
def test_classifier_rejects_non_micro_futures(symbol):
    # "MA"/"MESA" guard against over-matching on the M-prefix; "ES"/"NQ" are
    # full-size futures not in the micro set and carry no secType here.
    assert is_futures_row(symbol) is False


def test_classifier_sectype_fut_is_authoritative_for_unlisted_root():
    # secType=="FUT" short-circuits even when the root is not in the micro set.
    assert is_futures_row("XYZ", "FUT") is True
    assert is_futures_row("XYZ", "fut") is True
    assert is_futures_row("XYZ", "STK") is False
    assert is_futures_row("XYZ", None) is False


# ---------------------------------------------------------------------------
# Fixtures (mirror test_trade_stats_engine_validate_only.py)
# ---------------------------------------------------------------------------


def _env(payload, seq=1):
    return {
        "timestamp_utc": "2099-01-01T00:00:00+00:00",
        "sequence_id": seq,
        "payload": payload,
        "prev_hash": "GENESIS",
        "record_hash": f"rh-{seq}",
    }


def _trusted(symbol, strategy, pnl, *, extra=None, tags=None, broker="ibkr"):
    return {
        "strategy": strategy,
        "symbol": symbol,
        "side": "SELL",
        "quantity": 1.0,
        "fill_price": 100.0,
        "notional": 100.0,
        "pnl": pnl,
        "entry_time_utc": "2099-01-01T00:00:00+00:00",
        "exit_time_utc": "2099-01-01T00:00:00+00:00",
        "is_live": False,
        "broker": broker,
        "tags": tags if tags is not None else ["ibkr_paper", "filled"],
        "extra": extra or {"source": "ibkr_paper_ledger_watcher"},
    }


def _write_today(tmp_path: Path, rows):
    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    f = trades_dir / f"trade_history_{today}.ndjson"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# 2. load_and_compute excludes futures, keeps equities
# ---------------------------------------------------------------------------


def test_load_and_compute_excludes_futures_keeps_equities(tmp_path, monkeypatch):
    rows = [
        _env(_trusted("AAPL", "alpha", 10.0), seq=1),          # equity win  (kept)
        _env(_trusted("NVDA", "alpha", -4.0), seq=2),          # equity loss (kept)
        _env(_trusted("ZZ", "alpha", 5.0), seq=3),             # non-futures (kept)
        _env(_trusted("MES", "gamma_futures", 20.0), seq=4),   # micro root  (excluded)
        _env(_trusted("M6E", "omega_macro", 15.0), seq=5),     # micro root  (excluded)
        _env(_trusted("XYZ", "alpha_futures", 7.0,             # unlisted root + secType FUT
                      extra={"source": "x", "secType": "FUT"}), seq=6),
    ]
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = load_and_compute(
        max_trades=500, days_back=30, include_paper=True, include_live=True,
        epoch_filter=False,
    )

    # 3 equities survive (AAPL, NVDA, ZZ); 3 futures dropped (MES, M6E, XYZ/FUT).
    assert stats["effective_trades"] == 3, stats
    assert stats["excluded_futures"] == 3, stats
    assert stats["win_rate"] == pytest.approx(2.0 / 3.0), stats   # AAPL+ZZ win, NVDA loss
    # total_pnl is over ALL finite trades (incl futures), unchanged by 5b.
    assert stats["total_pnl"] == pytest.approx(53.0), stats


def test_untrusted_futures_counts_as_untrusted_not_futures(tmp_path, monkeypatch):
    # An untrusted MES row is caught by the pnl_untrusted gate BEFORE the futures
    # gate, so it lands in excluded_untrusted (drop-order is intentional).
    rows = [
        _env(_trusted("MES", "gamma_futures", 0.0,
                      tags=["ibkr_paper", "filled", "pnl_untrusted"],
                      extra={"pnl_untrusted": True}), seq=1),
        _env(_trusted("M6E", "omega_macro", 12.0), seq=2),   # trusted futures -> excluded_futures
        _env(_trusted("AAPL", "alpha", 9.0), seq=3),         # equity -> effective
    ]
    _write_today(tmp_path, rows)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAD_REPO_ROOT", str(tmp_path))

    stats = load_and_compute(
        max_trades=500, days_back=30, include_paper=True, include_live=True,
        epoch_filter=False,
    )
    assert stats["excluded_untrusted"] == 1, stats   # the untrusted MES
    assert stats["excluded_futures"] == 1, stats      # the trusted M6E
    assert stats["effective_trades"] == 1, stats      # only AAPL


# ---------------------------------------------------------------------------
# 4. expectancy_tracker mirrors the exclusion
# ---------------------------------------------------------------------------


def test_expectancy_tracker_excludes_futures(tmp_path, monkeypatch):
    from chad.analytics import expectancy_tracker as et
    import chad.utils.quarantine as quar

    trades_dir = tmp_path / "data" / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    f = trades_dir / f"trade_history_{today}.ndjson"
    rows = [
        _env(_trusted("AAPL", "alpha", 10.0), seq=1),
        _env(_trusted("MES", "gamma_futures", 20.0), seq=2),
        _env(_trusted("XYZ", "alpha_futures", 7.0,
                      extra={"source": "x", "secType": "FUT"}), seq=3),
    ]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(et, "TRADES_GLOB", str(trades_dir / "trade_history_*.ndjson"))
    monkeypatch.setattr(quar, "get_exclusion_sets", lambda **k: (set(), set()))

    yielded = list(et._iter_trades())
    symbols = {p.get("symbol") for p in yielded}
    assert symbols == {"AAPL"}, symbols

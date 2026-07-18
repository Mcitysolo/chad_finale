"""CBAR S2 — Kraken bars freshness marker + state file for the EXS1 sentinel table.

S2 emits, inside the EXISTING ``chad/market_data/nightly_bars_refresh.py::_run_kraken`` (no new
publisher, no new timer):
  * a ``KRAKEN_BARS_REFRESH ok=N fail=M`` marker line over the CRYPTO symbols only, and
  * ``runtime/kraken_bars_state.json`` (schema kraken_bars_state.v1, ttl 90000s) carrying its
    own ``ts_utc`` + ``ttl_seconds`` so the Exterminator Sentinel's ``check_stale_feeds`` can
    judge crypto-bar freshness once an operator adds the feed row to config/exterminator.json.

Observability only: a state-write failure is swallowed so it can never fail the refresh run.
"""

from __future__ import annotations

import json
from typing import Dict


def _import_nbr():
    import chad.market_data.nightly_bars_refresh as nbr
    return nbr


def test_s2_marker_and_state_schema(monkeypatch, tmp_path, capsys):
    nbr = _import_nbr()
    (tmp_path / "runtime").mkdir()
    monkeypatch.setattr(nbr, "REPO_ROOT", tmp_path)
    nbr._write_kraken_bars_state({"BTC-USD": True, "ETH-USD": True, "SOL-USD": False})
    assert "KRAKEN_BARS_REFRESH ok=2 fail=1" in capsys.readouterr().out
    st = json.loads((tmp_path / "runtime" / "kraken_bars_state.json").read_text())
    assert st["schema_version"] == "kraken_bars_state.v1"
    assert st["ttl_seconds"] == 90000            # 25h ttl, 1h grace over the 24h cadence
    assert st["ok"] == 2 and st["fail"] == 1
    assert st["symbols"] == {"BTC-USD": "ok", "ETH-USD": "ok", "SOL-USD": "fail"}
    # sentinel check_stale_feeds reads ts_field(default 'ts_utc') + artifact ttl_seconds
    assert "ts_utc" in st and "ttl_seconds" in st


def test_s2_run_kraken_counts_crypto_only_not_shared_results(monkeypatch, tmp_path, capsys):
    nbr = _import_nbr()
    fake = [{"ts_utc": "2026-07-18", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
             "volume": 1.0}]
    monkeypatch.setattr(nbr, "fetch_kraken_bars", lambda s, p: fake)
    monkeypatch.setattr(nbr, "_write_bar_file", lambda s, b, source: None)
    (tmp_path / "runtime").mkdir()
    monkeypatch.setattr(nbr, "REPO_ROOT", tmp_path)
    # main() seeds `results` with IBKR/VIX rows BEFORE _run_kraken; the marker/state must count
    # only the crypto symbols, never the shared dict.
    results: Dict[str, bool] = {"AAPL": False, "MES": True, "VIX": False}
    nbr._run_kraken(results)
    st = json.loads((tmp_path / "runtime" / "kraken_bars_state.json").read_text())
    assert st["ok"] == 3 and st["fail"] == 0
    assert set(st["symbols"]) == {"BTC-USD", "ETH-USD", "SOL-USD"}
    assert "KRAKEN_BARS_REFRESH ok=3 fail=0" in capsys.readouterr().out
    # shared results still updated so main()'s SUMMARY line stays correct
    assert results["BTC-USD"] is True and results["AAPL"] is False


def test_s2_state_write_failure_is_swallowed(monkeypatch, tmp_path, capsys):
    nbr = _import_nbr()
    monkeypatch.setattr(nbr, "REPO_ROOT", tmp_path)

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(nbr, "_atomic_write_json", _boom)
    # observability must never fail the refresh: no exception, marker still printed
    nbr._write_kraken_bars_state({"BTC-USD": True, "ETH-USD": True, "SOL-USD": True})
    assert "KRAKEN_BARS_REFRESH ok=3 fail=0" in capsys.readouterr().out

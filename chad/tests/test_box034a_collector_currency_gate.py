"""BOX-034A Increment 2 — collector NetLiquidation currency gate (fail-closed).

Spec: BOX-034A §3. Base currency = CAD (overridable via CHAD_BASE_CURRENCY).
Scope: the IBKRPortfolioCollector.update_snapshot currency gate ONLY. These
tests mock accountSummary rows (AccountValue-shaped objects) so no live IBKR
connection is required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.portfolio.ibkr_portfolio_collector_v2 import (
    IBKRConnectionConfig,
    IBKRPortfolioCollector,
)


class _Row:
    """Minimal AccountValue stand-in: account, tag, value, currency."""

    def __init__(self, account: str, tag: str, value, currency: str) -> None:
        self.account = account
        self.tag = tag
        self.value = value
        self.currency = currency


def _install_fake_ib(monkeypatch, rows):
    """Patch ib_async.IB so accountSummary() returns the supplied rows."""

    class _FakeIB:
        def connect(self, *a, **k):  # noqa: D401
            return None

        def accountSummary(self):
            return list(rows)

        def positions(self):
            return []

        def disconnect(self):
            return None

    import ib_async

    monkeypatch.setattr(ib_async, "IB", _FakeIB)


def _collector():
    cfg = IBKRConnectionConfig(host="127.0.0.1", port=4002, client_id=99, account_id="DU123")
    return IBKRPortfolioCollector(cfg)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_currency_match_cad_writes_value_and_flags_ok(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)  # default -> CAD
    _install_fake_ib(monkeypatch, [_Row("DU123", "NetLiquidation", "250000.0", "CAD")])

    snap = tmp_path / "portfolio_snapshot.json"
    out = _collector().update_snapshot(snapshot_path=snap)

    data = _read(out)
    assert data["ibkr_equity"] == 250000.0
    assert data["ibkr_equity_currency"] == "CAD"
    assert data["ibkr_equity_currency_ok"] is True


def test_currency_mismatch_usd_fail_closed_preserves_prior(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)  # default -> CAD
    # Prior canonical CAD value already on disk.
    snap = tmp_path / "portfolio_snapshot.json"
    snap.write_text(json.dumps({"ibkr_equity": 199999.0, "coinbase_equity": 10.0}), encoding="utf-8")

    _install_fake_ib(monkeypatch, [_Row("DU123", "NetLiquidation", "250000.0", "USD")])

    out = _collector().update_snapshot(snapshot_path=snap)
    data = _read(out)

    # Wrong-currency value (250000 USD) must NOT be written; prior CAD preserved.
    assert data["ibkr_equity"] == 199999.0
    assert data["ibkr_equity_currency_ok"] is False
    assert data["ibkr_equity_currency"] == "CAD"
    # Other venues still preserved.
    assert data["coinbase_equity"] == 10.0
    # Loud mismatch log emitted to stderr.
    err = capsys.readouterr().err
    assert "IBKR_NETLIQ_CURRENCY_MISMATCH" in err
    assert "expected=CAD" in err
    assert "USD" in err


def test_currency_mismatch_eur_fail_closed(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    snap = tmp_path / "portfolio_snapshot.json"
    snap.write_text(json.dumps({"ibkr_equity": 12345.0}), encoding="utf-8")

    _install_fake_ib(monkeypatch, [_Row("DU123", "NetLiquidation", "9999.0", "EUR")])

    out = _collector().update_snapshot(snapshot_path=snap)
    data = _read(out)

    assert data["ibkr_equity"] == 12345.0  # preserved, not 9999
    assert data["ibkr_equity_currency_ok"] is False
    assert "IBKR_NETLIQ_CURRENCY_MISMATCH" in capsys.readouterr().err


def test_base_currency_override_respected(monkeypatch, tmp_path):
    # Operator declares USD as base; a USD row should now MATCH and write.
    monkeypatch.setenv("CHAD_BASE_CURRENCY", "usd")  # lower-case to prove normalization
    _install_fake_ib(monkeypatch, [_Row("DU123", "NetLiquidation", "183264.0", "USD")])

    snap = tmp_path / "portfolio_snapshot.json"
    out = _collector().update_snapshot(snapshot_path=snap)
    data = _read(out)

    assert data["ibkr_equity"] == 183264.0
    assert data["ibkr_equity_currency"] == "USD"
    assert data["ibkr_equity_currency_ok"] is True


def test_mismatch_with_no_prior_writes_zero_fail_closed(monkeypatch, tmp_path, capsys):
    # Degenerate first-run case: mismatch + no prior canonical -> 0.0, ok=false.
    monkeypatch.delenv("CHAD_BASE_CURRENCY", raising=False)
    _install_fake_ib(monkeypatch, [_Row("DU123", "NetLiquidation", "250000.0", "USD")])

    snap = tmp_path / "portfolio_snapshot.json"
    out = _collector().update_snapshot(snapshot_path=snap)
    data = _read(out)

    assert data["ibkr_equity"] == 0.0
    assert data["ibkr_equity_currency_ok"] is False
    assert "IBKR_NETLIQ_CURRENCY_MISMATCH" in capsys.readouterr().err

"""W5A-4 — E3 closed_trade mae_mfe stamp (best-effort, flag-gated).

sidecar/off ⇒ no stamp on the closed_trade; stamp ⇒ best-effort read of the
excursion sidecar joined temporally on (strategy, symbol). Absent on a miss
(the sidecar stays authoritative). Additive, schema unchanged.
"""

from __future__ import annotations

import json

import chad.analytics.excursion_recorder as ex
from chad.execution.trade_closer import ClosedTrade


def _closed(**kw):
    base = dict(
        strategy="gamma", symbol="BAC", side="BUY", entry_price=100.0,
        exit_price=101.0, quantity=100.0,
        entry_time_utc="2026-07-23T10:00:00+00:00",
        exit_time_utc="2026-07-23T16:00:00+00:00", pnl=100.0,
        contract_multiplier=1.0, fill_ids=["open", "close"], meta={},
    )
    base.update(kw)
    return ClosedTrade(**base)


def _seed_sidecar(tmp_path, monkeypatch, *, opened="2026-07-23T10:00:00Z"):
    d = tmp_path / "exit_overlay"
    d.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "mae_mfe.v1", "ts_utc": "2026-07-23T15:59:00Z",
        "lane": "equity", "position_key": "gamma|BAC", "strategy": "gamma",
        "symbol": "BAC", "side": "BUY", "entry_price": 100.0, "hwm": 110.0,
        "lwm": 95.0, "quantity": 100.0, "opened_at_utc": opened,
        "closed_detect_utc": "2026-07-23T15:59:00Z",
        "excursion_source": "watermark_bar_hilo",
        "mae_pct": -0.05, "mae_usd": -500.0, "mfe_pct": 0.10, "mfe_usd": 1000.0,
        "mae_reason": "resolved", "mfe_reason": "resolved",
    }
    (d / "excursion_20260723.ndjson").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(ex, "DEFAULT_EVIDENCE_DIR", d)
    ex._EXCURSION_CACHE.clear()
    return d


def test_off_no_stamp(monkeypatch, tmp_path):
    _seed_sidecar(tmp_path, monkeypatch)
    monkeypatch.delenv("CHAD_E3_EXCURSION", raising=False)
    payload = _closed().to_payload()
    assert "mae_mfe" not in payload


def test_sidecar_mode_no_stamp(monkeypatch, tmp_path):
    """sidecar mode writes the sidecar but does NOT stamp the closed_trade."""
    _seed_sidecar(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAD_E3_EXCURSION", "sidecar")
    assert "mae_mfe" not in _closed().to_payload()


def test_stamp_mode_stamps_block(monkeypatch, tmp_path):
    _seed_sidecar(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAD_E3_EXCURSION", "stamp")
    payload = _closed().to_payload()
    assert payload["schema_version"] == "closed_trade.v1"  # unchanged
    block = payload["mae_mfe"]
    assert block["schema_version"] == "mae_mfe.v1"
    assert block["mae_usd"] == -500.0 and block["mfe_usd"] == 1000.0
    assert block["join"]["on"] == "strategy_symbol_temporal"


def test_stamp_absent_on_miss(monkeypatch, tmp_path):
    """stamp mode but the sidecar has no matching row (the D8 race) ⇒ no key,
    not a fabricated block."""
    d = tmp_path / "exit_overlay"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ex, "DEFAULT_EVIDENCE_DIR", d)
    ex._EXCURSION_CACHE.clear()
    monkeypatch.setenv("CHAD_E3_EXCURSION", "stamp")
    assert "mae_mfe" not in _closed().to_payload()


def test_stamp_does_not_change_pnl(monkeypatch, tmp_path):
    _seed_sidecar(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAD_E3_EXCURSION", "stamp")
    p_on = _closed().to_payload()
    monkeypatch.setenv("CHAD_E3_EXCURSION", "off")
    p_off = _closed().to_payload()
    assert p_on["net_pnl"] == p_off["net_pnl"]
    assert p_on["pnl_breakdown"] == p_off["pnl_breakdown"]


def test_reader_temporal_join_picks_closest(monkeypatch, tmp_path):
    d = tmp_path / "exit_overlay"
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {"schema_version": "mae_mfe.v1", "strategy": "gamma", "symbol": "BAC",
         "opened_at_utc": "2026-07-20T10:00:00Z", "closed_detect_utc": "2026-07-20T16:00:00Z",
         "mfe_usd": 1.0, "ts_utc": "2026-07-23T00:00:00Z"},
        {"schema_version": "mae_mfe.v1", "strategy": "gamma", "symbol": "BAC",
         "opened_at_utc": "2026-07-23T10:00:00Z", "closed_detect_utc": "2026-07-23T15:59:00Z",
         "mfe_usd": 2.0, "ts_utc": "2026-07-23T00:00:00Z"},
    ]
    (d / "excursion_20260723.ndjson").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ex._EXCURSION_CACHE.clear()
    block = ex.read_lap_excursion(
        strategy="gamma", symbol="BAC",
        entry_time_utc="2026-07-23T10:00:00Z", exit_time_utc="2026-07-23T16:00:00Z",
        evidence_dir=d,
    )
    assert block["mfe_usd"] == 2.0  # the 07-23 lap, not the 07-20 one

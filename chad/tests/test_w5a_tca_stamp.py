"""W5A-2 — TCA stamp at mint (trade_closer.to_payload).

Default OFF ⇒ byte-identical closed_trade.v1 (no implementation_shortfall key,
schema unchanged). ON ⇒ additive implementation_shortfall.v1 block from the
sibling fill evidence, both lanes. Never a top-level schema bump.
"""

from __future__ import annotations

import json

import chad.analytics.implementation_shortfall as is_mod
from chad.execution.trade_closer import ClosedTrade


def _closed(**kw):
    base = dict(
        strategy="gamma", symbol="PSQ", side="BUY", entry_price=26.0,
        exit_price=27.0, quantity=5.0, entry_time_utc="2026-07-23T16:00:00+00:00",
        exit_time_utc="2026-07-23T17:00:00+00:00", pnl=5.0,
        contract_multiplier=1.0, fill_ids=["open", "close"], meta={},
    )
    base.update(kw)
    return ClosedTrade(**base)


def _fill(fid, side, ts, *, price=26.0, slippage_bps=10.0, expected=26.0):
    return {"payload": {
        "schema_version": "paper_exec_fill.v4", "fill_id": fid, "symbol": "PSQ",
        "side": side, "strategy": "gamma", "quantity": 5.0, "fill_price": price,
        "fill_time_utc": ts, "status": "paper_fill",
        "extra": {"slippage_bps": slippage_bps, "fee_model": "ibkr_fixed_v1",
                  "submit_quote": {"ref_price": expected, "quote_ts": ts,
                                   "source": "price_cache_mid_or_last"}},
    }}


def _fee(side, ts, amt=1.0):
    return {"payload": {"schema_version": "paper_exec_fee.v4", "symbol": "PSQ",
                        "side": side, "strategy": "gamma", "fill_time_utc": ts,
                        "fee_amount": amt}}


def _seed_evidence(tmp_path, monkeypatch):
    d = tmp_path / "fills"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "FILLS_20260723.ndjson").open("w") as f:
        f.write(json.dumps(_fill("open", "BUY", "2026-07-23T16:00:00+00:00")) + "\n")
        f.write(json.dumps(_fill("close", "SELL", "2026-07-23T17:00:00+00:00", price=27.0)) + "\n")
    with (d / "FEES_20260723.ndjson").open("w") as f:
        f.write(json.dumps(_fee("BUY", "2026-07-23T16:00:00+00:00")) + "\n")
        f.write(json.dumps(_fee("SELL", "2026-07-23T17:00:00+00:00")) + "\n")
    monkeypatch.setattr(is_mod, "DEFAULT_FILLS_DIR", d)
    is_mod._INDEX_CACHE.clear()


# --------------------------------------------------------------------------- #
# OFF = byte-identical
# --------------------------------------------------------------------------- #

def test_off_no_block(monkeypatch):
    monkeypatch.delenv("CHAD_TCA_STAMP", raising=False)
    payload = _closed().to_payload()
    assert "implementation_shortfall" not in payload
    assert payload["schema_version"] == "closed_trade.v1"


def test_off_byte_identical_to_pre_w5a(monkeypatch, tmp_path):
    """Even with evidence present, OFF adds nothing."""
    _seed_evidence(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAD_TCA_STAMP", "off")
    payload = _closed().to_payload()
    assert "implementation_shortfall" not in payload


def test_garbage_flag_is_off(monkeypatch):
    monkeypatch.setenv("CHAD_TCA_STAMP", "yes")  # not "on"
    assert "implementation_shortfall" not in _closed().to_payload()


# --------------------------------------------------------------------------- #
# ON = additive block, schema unchanged
# --------------------------------------------------------------------------- #

def test_on_stamps_block(monkeypatch, tmp_path):
    _seed_evidence(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAD_TCA_STAMP", "on")
    payload = _closed(meta={"stop_width_usd": 100.0}).to_payload()
    assert payload["schema_version"] == "closed_trade.v1"  # NO top-level bump
    block = payload["implementation_shortfall"]
    assert block["schema_version"] == "implementation_shortfall.v1"
    assert block["fees_usd"] == 2.0
    assert block["cost_basis_status"] == "real"
    assert block["is_r"] is not None
    assert block["decision_price"] == 26.0
    # accounting numbers untouched (observer-class)
    assert payload["net_pnl"] == _closed().to_payload()["net_pnl"]


def test_on_honest_null_when_no_evidence(monkeypatch, tmp_path):
    """ON but no matching evidence ⇒ a well-formed block with honest nulls,
    NOT a missing block and NOT zeros (R1)."""
    d = tmp_path / "fills"
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(is_mod, "DEFAULT_FILLS_DIR", d)
    is_mod._INDEX_CACHE.clear()
    monkeypatch.setenv("CHAD_TCA_STAMP", "on")
    block = _closed().to_payload()["implementation_shortfall"]
    assert block["cost_basis_status"] == "unavailable"
    assert block["is_usd"] is None
    assert block["fees_usd"] is None
    assert block["legs"][0]["found"] is False


def test_on_stamp_does_not_change_pnl_or_breakdown(monkeypatch, tmp_path):
    _seed_evidence(tmp_path, monkeypatch)
    monkeypatch.setenv("CHAD_TCA_STAMP", "on")
    p_on = _closed().to_payload()
    monkeypatch.setenv("CHAD_TCA_STAMP", "off")
    p_off = _closed().to_payload()
    for k in ("pnl", "gross_pnl", "net_pnl", "pnl_breakdown", "commission",
              "slippage", "fees"):
        assert p_on[k] == p_off[k], k
    # pnl_breakdown still reports unavailable — real costs live in the new block
    assert p_on["pnl_breakdown"]["cost_basis_status"] == "unavailable"

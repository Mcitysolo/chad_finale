"""W5A-1 — Implementation Shortfall joiner core.

R1 honest-nulls throughout: unresolved legs are None + a reason code, never
zero-filled; a zero-cost lap is distinguishable from an unknown-cost lap.
"""

from __future__ import annotations

import json

from chad.analytics.implementation_shortfall import (
    IS_SCHEMA_VERSION,
    build_fee_index,
    build_fill_cost_index,
    compute_lap_is,
)

DATE = "20260723"


def _fill_row(fid, *, symbol="PSQ", side="BUY", strategy="gamma", qty=5.0,
              price=26.0, ts="2026-07-23T16:00:00+00:00", status="paper_fill",
              ref_price=26.0, expected_price=None, slippage_bps=0.0,
              submit_quote=True):
    extra = {"slippage_bps": slippage_bps, "fee_model": "ibkr_fixed_v1"}
    if submit_quote:
        extra["submit_quote"] = {"ref_price": ref_price, "quote_ts": ts,
                                 "source": "price_cache_mid_or_last"}
    if expected_price is not None:
        extra["expected_price"] = expected_price
    return {"payload": {
        "schema_version": "paper_exec_fill.v4", "fill_id": fid, "symbol": symbol,
        "side": side, "strategy": strategy, "quantity": qty, "fill_price": price,
        "fill_time_utc": ts, "status": status, "extra": extra,
    }}


def _fee_row(*, symbol="PSQ", side="BUY", strategy="gamma",
             ts="2026-07-23T16:00:00+00:00", fee_amount=1.0):
    return {"payload": {
        "schema_version": "paper_exec_fee.v4", "symbol": symbol, "side": side,
        "strategy": strategy, "fill_time_utc": ts, "fee_amount": fee_amount,
    }}


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _dirs(tmp_path, fills, fees=None):
    _write(tmp_path / "fills" / f"FILLS_{DATE}.ndjson", fills)
    if fees:
        _write(tmp_path / "fills" / f"FEES_{DATE}.ndjson", fees)
    return build_fill_cost_index(DATE, fills_dir=tmp_path / "fills")


# --------------------------------------------------------------------------- #
# Fee index join
# --------------------------------------------------------------------------- #

def test_fee_index_join_on_tuple(tmp_path):
    _write(tmp_path / "fills" / f"FEES_{DATE}.ndjson", [
        _fee_row(fee_amount=1.0),
        _fee_row(fee_amount=0.5, side="SELL", ts="2026-07-23T17:00:00+00:00"),
    ])
    idx = build_fee_index(DATE, fills_dir=tmp_path / "fills")
    assert idx[("PSQ", "BUY", "2026-07-23T16:00:00+00:00", "gamma")] == 1.0
    assert idx[("PSQ", "SELL", "2026-07-23T17:00:00+00:00", "gamma")] == 0.5


def test_fee_index_sums_collisions(tmp_path):
    _write(tmp_path / "fills" / f"FEES_{DATE}.ndjson",
           [_fee_row(fee_amount=1.0), _fee_row(fee_amount=0.65)])
    idx = build_fee_index(DATE, fills_dir=tmp_path / "fills")
    assert idx[("PSQ", "BUY", "2026-07-23T16:00:00+00:00", "gamma")] == 1.65


# --------------------------------------------------------------------------- #
# Per-fill decision price + slippage (from the fill's own extra)
# --------------------------------------------------------------------------- #

def test_decision_price_from_submit_quote(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("f1", ref_price=26.2)])
    fc = idx["f1"]
    assert fc.decision_price == 26.2
    assert fc.decision_price_source == "submit_quote_ref_price"
    assert fc.decision_price_reason == "resolved"


def test_decision_price_fallback_to_expected(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("f1", submit_quote=False, expected_price=25.9)])
    fc = idx["f1"]
    assert fc.decision_price == 25.9
    assert fc.decision_price_source == "expected_price"
    assert fc.decision_price_reason == "resolved_fallback"


def test_decision_price_honest_null(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("f1", submit_quote=False)])
    fc = idx["f1"]
    assert fc.decision_price is None
    assert fc.decision_price_reason == "no_decision_stamp"


def test_slippage_usd_from_bps(tmp_path):
    # 20 bps adverse on 5 @ 26.0 => 20/1e4 * 26 * 5 = 0.26
    idx = _dirs(tmp_path, [_fill_row("f1", slippage_bps=20.0, qty=5.0, price=26.0)])
    assert abs(idx["f1"].slippage_usd - 0.26) < 1e-9
    assert idx["f1"].slippage_reason == "resolved_from_fill_bps"


def test_non_genuine_status_nulls_costs(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("f1", status="dry_run", slippage_bps=20.0)],
                fees=[_fee_row()])
    fc = idx["f1"]
    assert fc.slippage_usd is None and fc.slippage_reason == "non_genuine_fill_status"
    assert fc.fee_usd is None and fc.fee_reason == "non_genuine_fill_status"


def test_fee_resolved_when_joined(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("f1")], fees=[_fee_row(fee_amount=1.0)])
    assert idx["f1"].fee_usd == 1.0
    assert idx["f1"].fee_reason == "resolved_from_fees_ledger"


def test_fee_null_when_no_row(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("f1")])
    assert idx["f1"].fee_usd is None
    assert idx["f1"].fee_reason == "no_fee_row"


# --------------------------------------------------------------------------- #
# Lap aggregation + honest nulls (R1)
# --------------------------------------------------------------------------- #

def _lap_is(idx, fill_ids=("open", "close"), qty=5.0, mult=1.0,
            broker="paper_exec", stop_width=None, kraken_native=None):
    return compute_lap_is(
        fill_ids=fill_ids, quantity=qty, contract_multiplier=mult,
        broker=broker, stop_width_usd=stop_width, index=idx,
        kraken_native=kraken_native,
    )


def test_full_real_lap(tmp_path):
    idx = _dirs(tmp_path, [
        _fill_row("open", side="BUY", slippage_bps=10.0, price=26.0, qty=5.0),
        _fill_row("close", side="SELL", slippage_bps=10.0, price=27.0, qty=5.0,
                  ts="2026-07-23T17:00:00+00:00"),
    ], fees=[
        _fee_row(side="BUY", fee_amount=1.0),
        _fee_row(side="SELL", ts="2026-07-23T17:00:00+00:00", fee_amount=1.0),
    ])
    block = _lap_is(idx, stop_width=100.0)
    assert block["schema_version"] == IS_SCHEMA_VERSION
    # slippage: 10bps*26*5/1e4 + 10bps*27*5/1e4 = 0.13 + 0.135 = 0.265
    assert abs(block["slippage_usd"] - 0.265) < 1e-6
    assert block["fees_usd"] == 2.0
    assert block["cost_basis_status"] == "real"
    # IS = slippage + fees (funding/opp null) = 2.265
    assert abs(block["is_usd"] - 2.265) < 1e-6
    # bps vs decision price 26.0 * 5 = 130 notional
    assert abs(block["is_bps"] - (2.265 / 130.0 * 10000)) < 1e-2
    assert abs(block["is_r"] - (2.265 / 100.0)) < 1e-3  # is_r rounded to 4 dp


def test_honest_nulls_funding_and_opp_always_null(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("open"), _fill_row("close", ts="2026-07-23T17:00:00+00:00")])
    block = _lap_is(idx)
    assert block["funding_usd"] is None
    assert block["funding_reason"] == "not_modeled_paper_lane"
    assert block["opportunity_cost_usd"] is None
    assert block["opportunity_cost_reason"] == "no_unfilled_qty_evidence"


def test_zero_cost_distinct_from_unknown(tmp_path):
    """R1 core: a lap with slippage_bps=0 (genuine ZERO) is distinguishable
    from a lap whose slippage is UNKNOWN (no bps)."""
    idx_zero = _dirs(tmp_path, [_fill_row("z", slippage_bps=0.0)],
                     fees=[_fee_row(fee_amount=1.0)])
    tmp2 = tmp_path / "b"
    idx_unk = build_fill_cost_index(DATE, fills_dir=(tmp2 / "fills")) if False else None
    _write(tmp2 / "fills" / f"FILLS_{DATE}.ndjson", [
        {"payload": {"schema_version": "paper_exec_fill.v4", "fill_id": "u",
                     "symbol": "PSQ", "side": "BUY", "strategy": "gamma",
                     "quantity": 5.0, "fill_price": 26.0,
                     "fill_time_utc": "2026-07-23T16:00:00+00:00",
                     "status": "paper_fill", "extra": {"fee_model": "x"}}}])
    idx_unk = build_fill_cost_index(DATE, fills_dir=(tmp2 / "fills"))
    assert idx_zero["z"].slippage_usd == 0.0            # genuine zero
    assert idx_zero["z"].slippage_reason == "resolved_from_fill_bps"
    assert idx_unk["u"].slippage_usd is None            # unknown
    assert idx_unk["u"].slippage_reason == "no_slippage_bps"


def test_missing_fill_is_null_not_zero(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("open")])
    block = _lap_is(idx, fill_ids=("open", "RECON_ADOPT_UNH"))
    close_leg = block["legs"][1]
    assert close_leg["found"] is False
    assert close_leg["slippage_usd"] is None and close_leg["fee_usd"] is None
    # partial: open resolved, close absent
    assert block["cost_basis_status"] in {"partial", "unavailable"}


def test_is_r_null_without_stop_width(tmp_path):
    idx = _dirs(tmp_path, [_fill_row("open", slippage_bps=10.0)],
                fees=[_fee_row(fee_amount=1.0)])
    block = _lap_is(idx, fill_ids=("open",), stop_width=None)
    assert block["is_r"] is None
    assert block["is_r_reason"] == "no_stop_width_usd"


def test_unavailable_when_no_costs(tmp_path):
    idx = _dirs(tmp_path, [{"payload": {
        "schema_version": "paper_exec_fill.v4", "fill_id": "open", "symbol": "PSQ",
        "side": "BUY", "strategy": "gamma", "quantity": 5.0, "fill_price": 26.0,
        "fill_time_utc": "2026-07-23T16:00:00+00:00", "status": "paper_fill",
        "extra": {"fee_model": "x"}}}])
    block = _lap_is(idx, fill_ids=("open",))
    assert block["is_usd"] is None
    assert block["cost_basis_status"] == "unavailable"
    assert block["is_bps"] is None and block["is_bps_reason"] == "no_is_usd"


def test_kraken_native_fees_preferred(tmp_path):
    idx = _dirs(tmp_path, [])
    block = _lap_is(
        idx, fill_ids=("kopen", "kclose"), broker="kraken",
        kraken_native={"entry_fee": 0.26, "exit_fee": 0.27, "expected_price": 60000.0},
    )
    assert abs(block["fees_usd"] - 0.53) < 1e-9
    assert block["decision_price"] == 60000.0
    assert block["decision_price_source"] == "kraken_expected_price"

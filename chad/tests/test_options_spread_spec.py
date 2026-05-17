"""
Tier 1 Phase D Item 2 — typed OptionsSpreadSpec contract tests.
"""

from __future__ import annotations

import pytest

from chad.options.spread_spec import OptionsSpreadSpec


def _valid_kwargs(**overrides):
    base = dict(
        symbol="SPY",
        expiry="20260618",
        long_strike=737.0,
        short_strike=744.0,
        long_right="C",
        short_right="C",
        ratio_long=1,
        ratio_short=1,
        exchange="SMART",
        currency="USD",
        spread_type="BULL_CALL",
        max_loss_per_contract=700.0,
        net_debit_estimate=350.0,
        spread_id="abc-123",
        dte=32,
    )
    base.update(overrides)
    return base


# ---------- 1. construction normalization ----------

def test_valid_spread_spec_constructs_and_normalizes_rights() -> None:
    spec = OptionsSpreadSpec(**_valid_kwargs(long_right="c", short_right="p"))
    assert spec.symbol == "SPY"
    assert spec.expiry == "20260618"
    assert spec.long_right == "C"
    assert spec.short_right == "P"
    assert spec.exchange == "SMART"
    assert spec.currency == "USD"
    assert spec.spread_type == "BULL_CALL"
    assert spec.ratio_long == 1
    assert spec.ratio_short == 1


def test_symbol_normalizes_to_upper_and_strips_whitespace() -> None:
    spec = OptionsSpreadSpec(**_valid_kwargs(symbol="  spy  "))
    assert spec.symbol == "SPY"


# ---------- 2. invalid expiry ----------

@pytest.mark.parametrize("bad_expiry", ["2026-06-18", "20260", "", "abc12345", "202606189"])
def test_invalid_expiry_rejected(bad_expiry) -> None:
    with pytest.raises(ValueError, match="expiry"):
        OptionsSpreadSpec(**_valid_kwargs(expiry=bad_expiry))


# ---------- 3. invalid right ----------

@pytest.mark.parametrize("bad_right", ["X", "", "CALL", "PUT", "1"])
def test_invalid_long_right_rejected(bad_right) -> None:
    with pytest.raises(ValueError, match="long_right"):
        OptionsSpreadSpec(**_valid_kwargs(long_right=bad_right))


@pytest.mark.parametrize("bad_right", ["Z", "call", " "])
def test_invalid_short_right_rejected(bad_right) -> None:
    with pytest.raises(ValueError, match="short_right"):
        OptionsSpreadSpec(**_valid_kwargs(short_right=bad_right))


# ---------- 4. same strike ----------

def test_same_strike_rejected() -> None:
    with pytest.raises(ValueError, match="differ"):
        OptionsSpreadSpec(**_valid_kwargs(long_strike=740.0, short_strike=740.0))


# ---------- 5. negative / zero strike ----------

@pytest.mark.parametrize("bad", [0.0, -1.0, -100.0])
def test_negative_long_strike_rejected(bad) -> None:
    with pytest.raises(ValueError, match="long_strike"):
        OptionsSpreadSpec(**_valid_kwargs(long_strike=bad))


@pytest.mark.parametrize("bad", [0.0, -50.0])
def test_negative_short_strike_rejected(bad) -> None:
    with pytest.raises(ValueError, match="short_strike"):
        OptionsSpreadSpec(**_valid_kwargs(short_strike=bad))


# ---------- ratios + exchange + currency + symbol guards ----------

def test_ratio_long_must_be_positive_int() -> None:
    with pytest.raises(ValueError, match="ratio_long"):
        OptionsSpreadSpec(**_valid_kwargs(ratio_long=0))


def test_ratio_short_must_be_positive_int() -> None:
    with pytest.raises(ValueError, match="ratio_short"):
        OptionsSpreadSpec(**_valid_kwargs(ratio_short=-1))


def test_empty_symbol_rejected() -> None:
    with pytest.raises(ValueError, match="symbol"):
        OptionsSpreadSpec(**_valid_kwargs(symbol="   "))


# ---------- 6. to_legacy_meta ----------

def test_to_legacy_meta_emits_expected_keys() -> None:
    spec = OptionsSpreadSpec(**_valid_kwargs())
    meta = spec.to_legacy_meta()
    assert meta["sec_type"] == "BAG"
    assert meta["required_asset_class"] == "options"
    assert meta["spread_type"] == "BULL_CALL"
    assert meta["expiry"] == "20260618"
    assert meta["long_strike"] == 737.0
    assert meta["short_strike"] == 744.0
    assert meta["long_right"] == "C"
    assert meta["short_right"] == "C"
    assert meta["exchange"] == "SMART"
    assert meta["currency"] == "USD"
    assert meta["ratio_long"] == 1
    assert meta["ratio_short"] == 1
    assert meta["max_loss_per_contract"] == 700.0
    assert meta["net_debit_estimate"] == 350.0
    assert meta["spread_id"] == "abc-123"
    assert meta["dte"] == 32


def test_to_legacy_meta_omits_none_optional_fields() -> None:
    spec = OptionsSpreadSpec(
        **_valid_kwargs(
            max_loss_per_contract=None,
            net_debit_estimate=None,
            spread_id=None,
            dte=None,
        )
    )
    meta = spec.to_legacy_meta()
    assert "max_loss_per_contract" not in meta
    assert "net_debit_estimate" not in meta
    assert "spread_id" not in meta
    assert "dte" not in meta


# ---------- 7. from_legacy_meta ----------

def test_from_legacy_meta_reconstructs_spec() -> None:
    src_meta = {
        "sec_type": "BAG",
        "spread_type": "BEAR_PUT",
        "expiry": "20260620",
        "long_strike": 720,
        "short_strike": 715,
        "long_right": "p",
        "short_right": "p",
        "exchange": "SMART",
        "currency": "USD",
        "max_loss_per_contract": 500,
        "net_debit_estimate": 250,
        "spread_id": "xyz-7",
        "dte": 21,
    }
    spec = OptionsSpreadSpec.from_legacy_meta("spy", src_meta)
    assert spec.symbol == "SPY"
    assert spec.spread_type == "BEAR_PUT"
    assert spec.long_strike == 720.0
    assert spec.short_strike == 715.0
    assert spec.long_right == "P"
    assert spec.short_right == "P"
    assert spec.spread_id == "xyz-7"
    assert spec.dte == 21


def test_from_legacy_meta_accepts_lastTradeDateOrContractMonth_alias() -> None:
    spec = OptionsSpreadSpec.from_legacy_meta(
        "SPY",
        {
            "lastTradeDateOrContractMonth": "20260618",
            "long_strike": 737.0,
            "short_strike": 744.0,
            "long_right": "C",
            "short_right": "C",
        },
    )
    assert spec.expiry == "20260618"


def test_from_legacy_meta_missing_strikes_raises() -> None:
    with pytest.raises(ValueError, match="long_strike"):
        OptionsSpreadSpec.from_legacy_meta(
            "SPY",
            {
                "expiry": "20260618",
                "long_right": "C",
                "short_right": "C",
            },
        )


# ---------- 8. bag_leg_dicts ----------

def test_bag_leg_dicts_returns_buy_and_sell_legs() -> None:
    spec = OptionsSpreadSpec(**_valid_kwargs())
    legs = spec.bag_leg_dicts()
    assert len(legs) == 2
    assert legs[0]["action"] == "BUY"
    assert legs[0]["strike"] == 737.0
    assert legs[0]["right"] == "C"
    assert legs[0]["ratio"] == 1
    assert legs[0]["expiry"] == "20260618"
    assert legs[1]["action"] == "SELL"
    assert legs[1]["strike"] == 744.0
    assert legs[1]["right"] == "C"
    assert legs[1]["ratio"] == 1
    assert legs[1]["expiry"] == "20260618"


# ---------- frozen / slots guard ----------

def test_spec_is_frozen() -> None:
    spec = OptionsSpreadSpec(**_valid_kwargs())
    with pytest.raises(Exception):
        spec.symbol = "QQQ"  # type: ignore[misc]


def test_round_trip_to_legacy_and_back_preserves_fields() -> None:
    spec = OptionsSpreadSpec(**_valid_kwargs())
    meta = spec.to_legacy_meta()
    rebuilt = OptionsSpreadSpec.from_legacy_meta(spec.symbol, meta)
    assert rebuilt.symbol == spec.symbol
    assert rebuilt.expiry == spec.expiry
    assert rebuilt.long_strike == spec.long_strike
    assert rebuilt.short_strike == spec.short_strike
    assert rebuilt.long_right == spec.long_right
    assert rebuilt.short_right == spec.short_right
    assert rebuilt.spread_type == spec.spread_type
    assert rebuilt.spread_id == spec.spread_id
    assert rebuilt.dte == spec.dte

"""CRYPTO-TRUST — strict/fail-closed loader for config/kraken_trading.json."""

from __future__ import annotations

import json

import pytest

from chad.execution.kraken_trading_config import (
    KrakenTradingConfigError,
    load_kraken_trading_config,
)

_GOOD = {
    "schema_version": "kraken_trading.v1",
    "frozen_utc": "2026-07-12",
    "taker_fee": {"operator_verify": True, "default_taker_bps": 26.0,
                  "taker_bps_by_pair": {"SOLUSD": 26.0}},
    "slippage_model": {"operator_verify": False, "slippage_impact_floor_bps": 5.0,
                       "max_tick_age_seconds": 30.0},
    "min_order_size_by_pair": {"operator_verify": True, "default_min_volume": 0.0001,
                               "min_volume_by_pair": {"SOLUSD": 0.05}},
}


def _write(tmp_path, obj):
    p = tmp_path / "kraken_trading.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_repo_config_loads():
    cfg = load_kraken_trading_config()  # default path = config/kraken_trading.json
    assert cfg.taker_bps("SOLUSD") == 26.0
    assert cfg.min_volume("SOLUSD") == 0.05
    assert cfg.taker_fee(100.0, "SOLUSD") == pytest.approx(0.26)
    assert cfg.slippage_impact_floor_bps == 5.0


def test_good_config_roundtrip(tmp_path):
    cfg = load_kraken_trading_config(_write(tmp_path, _GOOD))
    assert cfg.taker_bps("SOLUSD") == 26.0
    assert cfg.taker_bps("UNKNOWN") == 26.0  # falls back to default


def test_unknown_key_rejected(tmp_path):
    bad = json.loads(json.dumps(_GOOD))
    bad["taker_fee"]["surprise"] = 1
    with pytest.raises(KrakenTradingConfigError):
        load_kraken_trading_config(_write(tmp_path, bad))


def test_missing_key_rejected(tmp_path):
    bad = json.loads(json.dumps(_GOOD))
    del bad["slippage_model"]["slippage_impact_floor_bps"]
    with pytest.raises(KrakenTradingConfigError):
        load_kraken_trading_config(_write(tmp_path, bad))


def test_wrong_type_rejected(tmp_path):
    bad = json.loads(json.dumps(_GOOD))
    bad["taker_fee"]["default_taker_bps"] = "twenty-six"
    with pytest.raises(KrakenTradingConfigError):
        load_kraken_trading_config(_write(tmp_path, bad))


def test_bool_not_accepted_as_number(tmp_path):
    bad = json.loads(json.dumps(_GOOD))
    bad["slippage_model"]["max_tick_age_seconds"] = True
    with pytest.raises(KrakenTradingConfigError):
        load_kraken_trading_config(_write(tmp_path, bad))


def test_schema_version_mismatch_rejected(tmp_path):
    bad = json.loads(json.dumps(_GOOD))
    bad["schema_version"] = "kraken_trading.v2"
    with pytest.raises(KrakenTradingConfigError):
        load_kraken_trading_config(_write(tmp_path, bad))

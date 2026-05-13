"""Tests for chad/execution/tier_instrument_gate.py — Gap-1 of the v9.1
post-implementation audit.

Verifies the TierInstrumentGate blocks raw TradeSignals whose symbol is
not in the active tier's allowed_instruments list, with fail-open
semantics on missing / malformed / null tier state.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import pytest

from chad.execution.tier_instrument_gate import TierInstrumentGate
from chad.types import AssetClass, SignalSide, StrategyName, TradeSignal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_tier_state(
    runtime_dir: Path,
    *,
    tier_name: str = "STARTER",
    allowed_instruments=None,
) -> None:
    payload = {
        "schema_version": "tier_state.v2",
        "tier_name": tier_name,
        "allowed_instruments": allowed_instruments,
    }
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "tier_state.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _mk_signal(symbol: str, strategy: StrategyName = StrategyName.ALPHA_CRYPTO) -> TradeSignal:
    # alpha_crypto's natural shape — BUY, CRYPTO asset class. The tier
    # gate only consults `symbol` so the rest is incidental.
    asset_class = AssetClass.CRYPTO if "-USD" in symbol else AssetClass.FUTURES
    return TradeSignal(
        strategy=strategy,
        symbol=symbol,
        side=SignalSide.BUY,
        size=1.0,
        confidence=0.6,
        asset_class=asset_class,
        meta={},
    )


def _read_state(runtime_dir: Path) -> dict:
    return json.loads((runtime_dir / "tier_instrument_gate_state.json").read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_scale_tier_wildcard_passes_all(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path, tier_name="SCALE", allowed_instruments=["*"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [
        _mk_signal("BTC-USD"),
        _mk_signal("ETH-USD"),
        _mk_signal("SOL-USD"),
        _mk_signal("MES", strategy=StrategyName.ALPHA_INTRADAY_MICRO),
    ]

    allowed, blocked = gate.filter_signals(sigs)

    assert len(allowed) == 4
    assert blocked == []
    state = _read_state(tmp_path)
    assert state["tier"] == "SCALE"
    assert state["signals_blocked"] == 0


def test_2_starter_btc_allowed(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [_mk_signal("BTC-USD")]

    allowed, blocked = gate.filter_signals(sigs)

    assert len(allowed) == 1
    assert blocked == []


def test_3_starter_eth_blocked(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    with caplog.at_level(logging.WARNING, logger="chad.execution.tier_instrument_gate"):
        allowed, blocked = gate.filter_signals([_mk_signal("ETH-USD")])

    assert allowed == []
    assert len(blocked) == 1
    assert any(
        "INSTRUMENT_NOT_IN_TIER_ALLOWLIST" in rec.message
        and "ETH-USD" in rec.message
        for rec in caplog.records
    )
    state = _read_state(tmp_path)
    assert state["reason"] == "INSTRUMENT_NOT_IN_TIER_ALLOWLIST"
    assert "ETH-USD" in state["blocked_symbols"]


def test_4_starter_sol_blocked(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    allowed, blocked = gate.filter_signals([_mk_signal("SOL-USD")])

    assert allowed == []
    assert len(blocked) == 1
    state = _read_state(tmp_path)
    assert "SOL-USD" in state["blocked_symbols"]


def test_5_micro_mes_allowed(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path, tier_name="MICRO", allowed_instruments=["MES"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    allowed, blocked = gate.filter_signals(
        [_mk_signal("MES", strategy=StrategyName.ALPHA_INTRADAY_MICRO)]
    )

    assert len(allowed) == 1
    assert blocked == []


def test_6_micro_mnq_blocked(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path, tier_name="MICRO", allowed_instruments=["MES"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    allowed, blocked = gate.filter_signals(
        [_mk_signal("MNQ", strategy=StrategyName.ALPHA_INTRADAY_MICRO)]
    )

    assert allowed == []
    assert len(blocked) == 1


def test_7_tier_state_missing_fail_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # No tier_state.json written.
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [_mk_signal("ETH-USD"), _mk_signal("SOL-USD")]

    with caplog.at_level(logging.WARNING, logger="chad.execution.tier_instrument_gate"):
        allowed, blocked = gate.filter_signals(sigs)

    assert len(allowed) == 2
    assert blocked == []
    state = _read_state(tmp_path)
    assert state["fail_open"] is True
    assert state["fail_open_reason"] == "tier_state_missing"
    assert any("fail_open" in rec.message for rec in caplog.records)


def test_8_tier_state_malformed_fail_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "tier_state.json").write_text("{not valid json", encoding="utf-8")
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [_mk_signal("ETH-USD")]

    with caplog.at_level(logging.WARNING, logger="chad.execution.tier_instrument_gate"):
        allowed, blocked = gate.filter_signals(sigs)

    assert len(allowed) == 1
    assert blocked == []
    state = _read_state(tmp_path)
    assert state["fail_open"] is True
    assert state["fail_open_reason"] is not None
    assert "malformed" in state["fail_open_reason"]
    assert any("fail_open" in rec.message for rec in caplog.records)


def test_9_allowed_instruments_null_passes_all(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path, tier_name="STARTER", allowed_instruments=None,
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    allowed, blocked = gate.filter_signals(
        [_mk_signal("ETH-USD"), _mk_signal("SOL-USD")]
    )

    assert len(allowed) == 2
    assert blocked == []


def test_10_allowed_instruments_empty_list_passes_all(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path, tier_name="STARTER", allowed_instruments=[],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    allowed, blocked = gate.filter_signals(
        [_mk_signal("ETH-USD"), _mk_signal("SOL-USD")]
    )

    assert len(allowed) == 2
    assert blocked == []


def test_11_mixed_batch_starter(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [
        _mk_signal("BTC-USD"),
        _mk_signal("ETH-USD"),
        _mk_signal("SOL-USD"),
    ]

    allowed, blocked = gate.filter_signals(sigs)

    assert len(allowed) == 1
    assert allowed[0].symbol == "BTC-USD"
    assert len(blocked) == 2
    blocked_syms = {s.symbol for s in blocked}
    assert blocked_syms == {"ETH-USD", "SOL-USD"}
    state = _read_state(tmp_path)
    assert set(state["blocked_symbols"]) == {"ETH-USD", "SOL-USD"}


def test_12_state_file_written_atomically(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [
        _mk_signal("BTC-USD"),
        _mk_signal("ETH-USD"),
        _mk_signal("SOL-USD"),
    ]

    gate.filter_signals(sigs)

    state_path = tmp_path / "tier_instrument_gate_state.json"
    assert state_path.is_file()
    # No leftover .tmp files in the directory.
    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover == []
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "tier_instrument_gate.v1"
    assert set(payload["blocked_symbols"]) == {"ETH-USD", "SOL-USD"}
    assert payload["signals_evaluated"] == 3
    assert payload["signals_allowed"] == 1
    assert payload["signals_blocked"] == 2
    # ts_utc is a non-empty string.
    assert isinstance(payload["ts_utc"], str) and payload["ts_utc"]


def test_13_blocked_signals_do_not_reach_downstream_routing(
    tmp_path: Path,
) -> None:
    """Mock downstream router: only the allowed signals should be passed
    to it; blocked signals must be excluded."""
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)
    sigs = [_mk_signal("ETH-USD"), _mk_signal("BTC-USD")]

    allowed, blocked = gate.filter_signals(sigs)

    routed_through_downstream: List[TradeSignal] = []

    def fake_router(input_signals):
        # Capture exactly what would reach signal_router.route() in prod.
        routed_through_downstream.extend(input_signals)
        return input_signals

    fake_router(allowed)

    downstream_symbols = {s.symbol for s in routed_through_downstream}
    assert "ETH-USD" not in downstream_symbols
    assert downstream_symbols == {"BTC-USD"}
    assert {s.symbol for s in blocked} == {"ETH-USD"}


def test_14_empty_signal_list(tmp_path: Path) -> None:
    _write_tier_state(
        tmp_path,
        tier_name="STARTER",
        allowed_instruments=["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
    )
    gate = TierInstrumentGate(runtime_dir=tmp_path)

    allowed, blocked = gate.filter_signals([])

    assert allowed == []
    assert blocked == []
    state = _read_state(tmp_path)
    assert state["signals_evaluated"] == 0
    assert state["signals_allowed"] == 0
    assert state["signals_blocked"] == 0
    assert state["blocked_symbols"] == []
    assert state["blocked_strategies"] == []
    assert state["reason"] is None

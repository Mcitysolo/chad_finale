from __future__ import annotations

import pytest

from chad.execution.exit_only_executor import (
    AssetClass,
    ExitOnlyError,
    LiveGateDecision,
    Position,
    Side,
    build_exit_only_plan,
)


def _gate(*, exits_only: bool) -> LiveGateDecision:
    return LiveGateDecision(
        allow_exits_only=exits_only,
        allow_ibkr_paper=False,
        allow_ibkr_live=False,
        operator_mode="ALLOW_LIVE",
        reasons=(),
    )


def test_denies_when_not_exit_only() -> None:
    with pytest.raises(ExitOnlyError):
        build_exit_only_plan(live_gate=_gate(exits_only=False), positions=[])


def test_denies_if_live_flag_true_even_when_exit_only() -> None:
    gate = LiveGateDecision(
        allow_exits_only=True,
        allow_ibkr_paper=False,
        allow_ibkr_live=True,  # forbidden in Phase 9.1
        operator_mode="ALLOW_LIVE",
        reasons=("bad",),
    )
    with pytest.raises(ExitOnlyError):
        build_exit_only_plan(live_gate=gate, positions=[])


def test_ignores_flat_positions() -> None:
    plan = build_exit_only_plan(
        live_gate=_gate(exits_only=True),
        positions=[
            Position(symbol="AAPL", asset_class=AssetClass.EQUITY, qty=0.0),
            Position(symbol="SPY", asset_class=AssetClass.ETF, qty=0.0),
        ],
    )
    assert plan.exits == ()
    assert any("no_action" in n for n in plan.notes)


def test_builds_sell_exit_for_long_position() -> None:
    plan = build_exit_only_plan(
        live_gate=_gate(exits_only=True),
        positions=[Position(symbol="AAPL", asset_class=AssetClass.EQUITY, qty=5.0)],
    )
    assert len(plan.exits) == 1
    e = plan.exits[0]
    assert e.symbol == "AAPL"
    assert e.side == Side.SELL
    assert e.qty == 5.0


def test_builds_buy_exit_for_short_position() -> None:
    plan = build_exit_only_plan(
        live_gate=_gate(exits_only=True),
        positions=[Position(symbol="SPY", asset_class=AssetClass.ETF, qty=-2.0)],
    )
    assert len(plan.exits) == 1
    e = plan.exits[0]
    assert e.symbol == "SPY"
    assert e.side == Side.BUY
    assert e.qty == 2.0


def test_filters_derivatives_in_phase91() -> None:
    plan = build_exit_only_plan(
        live_gate=_gate(exits_only=True),
        positions=[
            Position(symbol="AAPL", asset_class=AssetClass.EQUITY, qty=1.0),
            Position(symbol="ES", asset_class=AssetClass.FUTURES, qty=1.0),
            Position(symbol="SPY_2026C", asset_class=AssetClass.OPTIONS, qty=1.0),
        ],
    )
    assert len(plan.exits) == 1
    assert plan.exits[0].symbol == "AAPL"


def test_lane_id_is_passthrough_only() -> None:
    plan = build_exit_only_plan(
        live_gate=_gate(exits_only=True),
        positions=[Position(symbol="AAPL", asset_class=AssetClass.EQUITY, qty=1.0)],
        lane_id="CORE",
    )
    assert plan.lane_id == "CORE"
    assert len(plan.exits) == 1
    assert plan.exits[0].lane_id == "CORE"

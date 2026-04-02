from __future__ import annotations

import json
from pathlib import Path
from datetime import date

from chad.risk.daily_throttle import throttle_signals_from_dynamic_caps
from chad.types import RoutedSignal, SignalSide, StrategyName, AssetClass


def _write_dynamic_caps(tmp_path: Path, *, global_cap: float, caps: dict[str, float]) -> Path:
    p = tmp_path / "dynamic_caps.json"
    payload = {"portfolio_risk_cap": global_cap, "strategy_caps": caps}
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_dynamic_caps_global_cap_blocks(tmp_path: Path) -> None:
    caps_path = _write_dynamic_caps(tmp_path, global_cap=100.0, caps={"alpha": 1000.0})

    s = RoutedSignal(
        symbol="AAPL",
        side=SignalSide.BUY,
        net_size=10.0,
        source_strategies=(StrategyName.ALPHA,),
        confidence=0.9,
        asset_class=AssetClass.EQUITY,
        created_at=None,
        meta={},
    )

    price_map = {"AAPL": 20.0}  # notional=200 > global_cap=100

    res = throttle_signals_from_dynamic_caps(
        [s],
        price_map,
        dynamic_caps_path=caps_path,
        today=date(2026, 1, 1),
        log_path=None,
    )

    assert res.accepted == []
    assert len(res.rejected) == 1
    assert "global cap exceeded" in res.rejected[0]


def test_dynamic_caps_strategy_cap_blocks(tmp_path: Path) -> None:
    caps_path = _write_dynamic_caps(tmp_path, global_cap=1000.0, caps={"alpha": 50.0})

    s = RoutedSignal(
        symbol="AAPL",
        side=SignalSide.BUY,
        net_size=10.0,
        source_strategies=(StrategyName.ALPHA,),
        confidence=0.9,
        asset_class=AssetClass.EQUITY,
        created_at=None,
        meta={},
    )

    price_map = {"AAPL": 10.0}  # notional=100 > alpha cap=50

    res = throttle_signals_from_dynamic_caps(
        [s],
        price_map,
        dynamic_caps_path=caps_path,
        today=date(2026, 1, 1),
        log_path=None,
    )

    assert res.accepted == []
    assert len(res.rejected) == 1
    assert "strategy cap exceeded" in res.rejected[0]
    assert "alpha" in res.rejected[0]


def test_dynamic_caps_missing_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    s = RoutedSignal(
        symbol="AAPL",
        side=SignalSide.BUY,
        net_size=1.0,
        source_strategies=(StrategyName.ALPHA,),
        confidence=0.9,
        asset_class=AssetClass.EQUITY,
        created_at=None,
        meta={},
    )

    price_map = {"AAPL": 10.0}

    res = throttle_signals_from_dynamic_caps(
        [s],
        price_map,
        dynamic_caps_path=missing,
        today=date(2026, 1, 1),
        log_path=None,
        fail_closed=True,
    )

    assert res.accepted == []
    assert len(res.rejected) == 1
    assert "dynamic_caps_unavailable" in res.rejected[0]

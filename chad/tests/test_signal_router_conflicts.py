from __future__ import annotations

from datetime import datetime, timezone

from chad.types import TradeSignal, StrategyName, SignalSide, AssetClass
from chad.utils.signal_router import SignalRouter


def _ts(symbol: str, side: SignalSide, size: float, strat: StrategyName, conf: float) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        side=side,
        size=size,
        strategy=strat,
        confidence=conf,
        asset_class=AssetClass.EQUITY,
        created_at=datetime.now(timezone.utc),
        meta={},
    )


def test_router_merges_same_symbol_same_side_deterministically() -> None:
    r = SignalRouter()

    s1 = _ts("AAPL", SignalSide.BUY, 10.0, StrategyName.ALPHA, 0.9)
    s2 = _ts("AAPL", SignalSide.BUY, 5.0, StrategyName.GAMMA, 0.5)

    out1 = r.route([s1, s2])
    out2 = r.route([s2, s1])  # reversed input order must not change result

    assert len(out1) == 1
    assert len(out2) == 1
    assert out1[0].symbol == out2[0].symbol == "AAPL"
    assert out1[0].side == out2[0].side == SignalSide.BUY
    assert out1[0].net_size == out2[0].net_size == 15.0
    assert out1[0].source_strategies == out2[0].source_strategies  # stable set->tuple order
    assert set(out1[0].source_strategies) == {StrategyName.ALPHA, StrategyName.GAMMA}


def test_router_does_not_net_opposite_sides() -> None:
    r = SignalRouter()

    buy = _ts("AAPL", SignalSide.BUY, 10.0, StrategyName.ALPHA, 0.9)
    sell = _ts("AAPL", SignalSide.SELL, 7.0, StrategyName.GAMMA, 0.8)

    out = r.route([buy, sell])

    # Must produce two routed signals (BUY bucket + SELL bucket), no netting here.
    assert len(out) == 2
    sides = {o.side for o in out}
    assert sides == {SignalSide.BUY, SignalSide.SELL}
    sizes = {(o.side, o.net_size) for o in out}
    assert (SignalSide.BUY, 10.0) in sizes
    assert (SignalSide.SELL, 7.0) in sizes

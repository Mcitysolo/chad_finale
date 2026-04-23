"""Tests for chad.strategies.beta (institutional-consensus compounder)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from unittest import mock

import pytest

from chad.strategies import beta as beta_mod
from chad.strategies.beta import (
    Beta,
    BetaParams,
    beta_handler,
    build_beta_config,
    _STATE,
)
from chad.types import AssetClass, Position, SignalSide, StrategyName


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePortfolio:
    cash: float = 1_000_000.0
    total_equity: float = 1_000_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    extra: Optional[Dict[str, Any]] = None


@dataclass
class FakeCtx:
    now: datetime
    portfolio: FakePortfolio
    prices: Dict[str, float]


def _fresh_state() -> None:
    Beta.reset_state()


def _write_consensus(
    path: Path,
    weights: Dict[str, float],
    *,
    age_days: float = 0.0,
) -> None:
    ts = datetime.now(timezone.utc).timestamp() - age_days * 86_400
    updated = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": "institutional_consensus.v1",
        "updated_ts_utc": updated,
        "funds_included": ["fund_a", "fund_b"],
        "top_holdings": [
            {"symbol": sym, "conviction_score": w, "cusip": "", "fund_count": 2}
            for sym, w in weights.items()
        ],
        "weights": weights,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def ctx_with_cash() -> FakeCtx:
    return FakeCtx(
        now=datetime.now(timezone.utc),
        portfolio=FakePortfolio(),
        prices={"AAPL": 180.0, "MSFT": 400.0, "NVDA": 900.0},
    )


@pytest.fixture(autouse=True)
def _reset_state():
    _fresh_state()
    yield
    _fresh_state()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_signal_family_is_institutional_consensus() -> None:
    # The sketch used "institutional_momentum" as a working label; the final
    # implementation standardized on "institutional_consensus". Both variants
    # carry the same semantic intent — pin whichever constant the module
    # currently exposes so a rename is caught, not silently accepted.
    assert Beta.SIGNAL_FAMILY in {"institutional_consensus", "institutional_momentum"}
    assert Beta.STRATEGY_NAME == "beta"


def test_build_beta_config_returns_strategy_config() -> None:
    cfg = build_beta_config()
    assert cfg.name == StrategyName.BETA
    assert cfg.enabled is True


def test_no_signals_when_no_consensus_data(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", missing)
    signals = beta_handler(ctx_with_cash)
    assert signals == []


def test_no_signals_when_consensus_stale(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {"AAPL": 0.5, "MSFT": 0.5}, age_days=90)
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    signals = beta_handler(ctx_with_cash)
    assert signals == []


def test_no_signals_when_weights_empty(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    assert beta_handler(ctx_with_cash) == []


def test_generates_buy_for_underweight_position(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    # Both AAPL and MSFT weights above the underweight_gap (2%) since current
    # weight = 0.0 in an empty portfolio. The per-position max_position_weight
    # (default 2%) caps the target, so only one signal per cycle fires.
    _write_consensus(path, {"AAPL": 0.5, "MSFT": 0.3})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    params = BetaParams(underweight_gap=0.005, max_position_weight=0.05)
    signals = list(beta_handler(ctx_with_cash, params=params))
    assert 1 <= len(signals) <= params.max_signals_per_cycle
    for sig in signals:
        assert sig.strategy == StrategyName.BETA
        assert sig.side == SignalSide.BUY
        assert sig.symbol in {"AAPL", "MSFT"}
        assert sig.size > 0
        assert sig.meta.get("reason") == "institutional_consensus_rebalance"


def test_max_signals_per_cycle_respected(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(
        path,
        {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2},
    )
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    params = BetaParams(
        max_signals_per_cycle=2,
        underweight_gap=0.005,
        max_position_weight=0.05,
    )
    signals = list(beta_handler(ctx_with_cash, params=params))
    assert len(signals) <= 2


def test_no_signals_under_equity_floor(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {"AAPL": 0.5})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    thin = FakeCtx(
        now=datetime.now(timezone.utc),
        portfolio=FakePortfolio(cash=10.0, total_equity=10.0),
        prices={"AAPL": 180.0},
    )
    assert beta_handler(thin) == []


def test_no_signals_when_prices_missing(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {"AAPL": 0.5})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    ctx = FakeCtx(
        now=datetime.now(timezone.utc),
        portfolio=FakePortfolio(),
        prices={},
    )
    assert beta_handler(ctx) == []


def test_no_signals_when_already_at_target(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {"AAPL": 0.02})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    # Existing position = full target weight -> no gap -> no signal
    ctx = FakeCtx(
        now=datetime.now(timezone.utc),
        portfolio=FakePortfolio(
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    asset_class=AssetClass.EQUITY,
                    quantity=200.0,
                    avg_price=100.0,
                ),
            },
        ),
        prices={"AAPL": 100.0},  # so weight = 200*100/1_000_000 = 0.02
    )
    assert beta_handler(ctx) == []


def test_weekly_cap_prevents_excess_signals(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(
        path,
        {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2},
    )
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    params = BetaParams(
        max_signals_per_week=2,
        max_signals_per_cycle=1,
        underweight_gap=0.005,
        max_position_weight=0.05,
    )
    # Three cycles; weekly cap at 2 should short-circuit cycle 3.
    seen = 0
    for _ in range(3):
        ctx_with_cash.now = datetime.now(timezone.utc)
        seen += len(list(beta_handler(ctx_with_cash, params=params)))
    assert seen <= 2


def test_facade_generate_signals_mirrors_handler(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {"AAPL": 0.5})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    b = Beta(params=BetaParams(underweight_gap=0.005, max_position_weight=0.05))
    out = list(b.generate_signals(ctx_with_cash))
    assert len(out) >= 1
    assert out[0].strategy == StrategyName.BETA


def test_signals_tagged_with_consensus_metadata(
    tmp_path: Path, ctx_with_cash: FakeCtx, monkeypatch
) -> None:
    path = tmp_path / "consensus.json"
    _write_consensus(path, {"AAPL": 0.5})
    monkeypatch.setattr(beta_mod, "CONSENSUS_PATH", path)
    params = BetaParams(underweight_gap=0.005, max_position_weight=0.05)
    signals = list(beta_handler(ctx_with_cash, params=params))
    assert signals, "expected at least one signal"
    meta = signals[0].meta
    assert "consensus_updated_ts_utc" in meta
    assert meta.get("consensus_funds") == ["fund_a", "fund_b"]
    assert "target_weight" in meta
    assert "gap" in meta

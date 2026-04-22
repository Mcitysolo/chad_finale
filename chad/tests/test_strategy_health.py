"""Tests for Phase-8 Session 6 (F3) strategy_health scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.strategy_health import (
    DEFAULT_NEUTRAL_SCORE,
    HEALTH_PATH,
    SCHEMA_VERSION,
    StrategyHealthScorer,
    read_health,
)


class _FakeSlippageTracker:
    """Minimal stand-in exposing get_rolling_stats(strategy=...)."""

    def __init__(self, per_strategy=None) -> None:
        self._per = per_strategy or {}

    def get_rolling_stats(self, symbol=None, strategy=None, last_n=None):
        return self._per.get(strategy, {"n": 0, "mean": None, "std": None})


def _expectancy_state(entries):
    return {"strategies": entries}


def test_high_sharpe_high_score(tmp_path: Path):
    path = tmp_path / "strategy_health.json"
    scorer = StrategyHealthScorer(output_path=path)
    state = _expectancy_state(
        {
            "alpha": {
                "total_trades": 100,
                "win_rate": 0.7,
                "expectancy": 1.5,
                "avg_win": 100.0,
                "rolling_sharpe": 2.0,
            }
        }
    )
    health = scorer.compute(
        "alpha",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker({"alpha": {"mean": 0.0, "n": 10}}),
        regime_state="trending_bull",
    )
    assert health["health_score"] > 0.8


def test_high_slippage_reduces_score(tmp_path: Path):
    path = tmp_path / "strategy_health.json"
    scorer = StrategyHealthScorer(output_path=path)
    state = _expectancy_state(
        {
            "alpha": {
                "total_trades": 50,
                "win_rate": 0.5,
                "expectancy": 0.2,
                "avg_win": 10.0,
                "rolling_sharpe": 1.0,
            }
        }
    )
    good = scorer.compute(
        "alpha",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker({"alpha": {"mean": 0.0, "n": 20}}),
        regime_state="ranging",
    )
    bad = scorer.compute(
        "alpha",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker(
            {"alpha": {"mean": 100.0, "n": 20}}  # all edge consumed
        ),
        regime_state="ranging",
    )
    assert bad["health_score"] < good["health_score"]
    assert bad["slippage_ratio"] > good["slippage_ratio"]


def test_score_clamped_0_to_1(tmp_path: Path):
    path = tmp_path / "strategy_health.json"
    scorer = StrategyHealthScorer(output_path=path)
    # Stuff values that would naively over/undershoot [0,1].
    state = _expectancy_state(
        {
            "alpha": {
                "total_trades": 100,
                "win_rate": 999.0,       # clamp to 1.0 via _clamp01
                "expectancy": -9999.0,
                "avg_win": 1.0,
                "rolling_sharpe": 99.0,  # clamps to normalized 1.0
            }
        }
    )
    health = scorer.compute(
        "alpha",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker({"alpha": {"mean": -100.0, "n": 10}}),
        regime_state="trending_bull",
    )
    assert 0.0 <= health["health_score"] <= 1.0


def test_no_expectancy_data_is_neutral(tmp_path: Path):
    path = tmp_path / "strategy_health.json"
    scorer = StrategyHealthScorer(output_path=path)
    health = scorer.compute(
        "alpha",
        expectancy_tracker=_expectancy_state({}),
        slippage_tracker=_FakeSlippageTracker(),
        regime_state="unknown",
    )
    assert health["health_score"] == DEFAULT_NEUTRAL_SCORE
    assert health["reason"] == "no_expectancy_data"


def test_all_strategies_written_to_json(tmp_path: Path):
    path = tmp_path / "strategy_health.json"
    scorer = StrategyHealthScorer(output_path=path)
    state = _expectancy_state(
        {
            "alpha": {"total_trades": 25, "win_rate": 0.55, "expectancy": 0.2,
                      "avg_win": 10.0, "rolling_sharpe": 1.0},
            "delta": {"total_trades": 25, "win_rate": 0.45, "expectancy": 0.1,
                      "avg_win": 10.0, "rolling_sharpe": 0.5},
        }
    )
    results = scorer.compute_all(
        strategy_names=["alpha", "delta"],
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker(),
        regime_state="trending_bull",
    )
    assert set(results.keys()) == {"alpha", "delta"}
    # Persisted file has the right schema.
    data = json.loads(path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert "alpha" in data["strategies"]
    assert "delta" in data["strategies"]
    # Weights present.
    assert data["weights"]["sharpe"] == pytest.approx(0.4)
    assert data["weights"]["win_rate"] == pytest.approx(0.3)


def test_read_health_returns_default_when_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    state = read_health(path=missing)
    assert state["strategies"] == {}
    assert state["schema_version"] == SCHEMA_VERSION


def test_regime_alignment_favourable_and_unfavourable(tmp_path: Path):
    path = tmp_path / "strategy_health.json"
    scorer = StrategyHealthScorer(output_path=path)
    entry = {"total_trades": 25, "win_rate": 0.5, "expectancy": 0.1,
             "avg_win": 10.0, "rolling_sharpe": 1.0}
    state = _expectancy_state({"delta_pairs": entry})
    # delta_pairs favours ranging.
    favourable = scorer.compute(
        "delta_pairs",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker(),
        regime_state="ranging",
    )
    unfavourable = scorer.compute(
        "delta_pairs",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippageTracker(),
        regime_state="trending_bull",
    )
    assert favourable["regime_alignment"] == 1.0
    assert unfavourable["regime_alignment"] == 0.0
    assert favourable["health_score"] > unfavourable["health_score"]

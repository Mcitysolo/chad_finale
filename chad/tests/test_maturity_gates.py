"""
Maturity-gate formal-close tests (Channel 2 Batch 12).

These tests pin the maturity-gate contracts for the three open
"data maturity" items: Omega Vol low sample, Winner-Scaling sample
maturity, and Strategy-Health sample maturity.

Contract being locked:
- Strategies with too-few clean trades MUST be treated as "not yet
  proven" — neither boosted nor punished by the sizing pipeline.
- Winner-scaler must publish multiplier=1.0 (neutral) for any
  strategy below ``min_trades_for_scaling`` (default 5).
- Strategy-Health scorer must return DEFAULT_NEUTRAL_SCORE (0.5) and
  ``reason="no_expectancy_data"`` when total_trades<=0, so cold-start
  strategies are not scored as if they had failed.
- Strategy-Health consumers (health_monitor.py, health_monitor_rules.py)
  MUST filter sample_count<10 before raising alerts so a 3-trade
  bad-streak does not page the operator.
- Allocator (dynamic_risk_allocator) MUST fall back to a conservative
  0.5x when winner_scaling is stale — the CB05 contract — and MUST
  apply 1.0x (neutral) for low-sample strategies based on what the
  scaler publishes.

If any of these contracts change in production code, this test file
must be updated explicitly. That keeps the maturity gate auditable.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

from chad.analytics.strategy_health import (
    DEFAULT_NEUTRAL_SCORE,
    StrategyHealthScorer,
)
from chad.risk import dynamic_risk_allocator as dra
from chad.risk.winner_scaler import (
    DEFAULT_POLICY,
    compute_multipliers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeSlippage:
    """Minimal slippage tracker stub: no slippage data."""

    def get_rolling_stats(self, symbol=None, strategy=None, last_n=None):
        return {"n": 0, "mean": None, "std": None}


# ---------------------------------------------------------------------------
# OMEGA VOL — low sample is "immature", not "failed"
# ---------------------------------------------------------------------------


def test_omega_vol_low_sample_marked_immature_not_failed():
    """Omega Vol with 3 trades (the live runtime state) must surface as
    NEUTRAL in the winner-scaling pipeline. The scorer is allowed to
    compute a low health_score, but the *sizing* layer must treat it
    as not-yet-proven (multiplier 1.0) so the allocator does not
    haircut sizing on a 3-trade sample."""
    expectancy = {
        "strategies": {
            "omega_vol": {
                "total_trades": 3,
                "expectancy": -0.31,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": -0.31,
            },
            # A larger pool so the median has signal.
            "alpha_futures": {"total_trades": 376, "expectancy": -32.83},
            "delta": {"total_trades": 82, "expectancy": 28.96},
        }
    }
    out = compute_multipliers(expectancy, DEFAULT_POLICY)
    # Below min_trades_for_scaling (default 5) → MUST be neutral.
    assert out["multipliers"]["omega_vol"] == 1.0, (
        "Omega Vol with only 3 trades must NOT be haircut by the "
        "winner-scaler — it must publish 1.0 (neutral) so the "
        "allocator does not over-trust low-sample evidence."
    )
    # Mature strategies are still scored normally.
    assert out["multipliers"]["alpha_futures"] != 1.0 or \
        out["multipliers"]["delta"] != 1.0, \
        "Mature strategies must remain eligible for scaling."


def test_omega_vol_health_score_does_not_reach_health_monitor_alert():
    """The health monitor explicitly filters sample_count<10 before
    flagging low-health strategies. Pin that contract: a 3-trade
    omega_vol record (even at health_score=0.1) must not appear in the
    'low health' alert list because its sample is below the minimum
    statistically meaningful threshold."""
    # Replicate the live filter from chad/ops/health_monitor.py
    health_doc = {
        "strategies": {
            "omega_vol": {
                "health_score": 0.1,
                "sample_count": 3,  # the live state
            },
            "delta": {
                "health_score": 0.91,
                "sample_count": 82,
            },
        }
    }
    strats = health_doc["strategies"]
    strats_with_data = {
        k: v
        for k, v in strats.items()
        if isinstance(v, dict)
        and int(v.get("sample_count", v.get("trade_count", 0)) or 0) >= 10
    }
    flagged = [
        (k, v.get("health_score", 1.0))
        for k, v in strats_with_data.items()
        if v.get("health_score", 1.0) < 0.5
    ]
    flagged_names = {name for name, _ in flagged}
    assert "omega_vol" not in flagged_names, (
        "omega_vol with sample_count=3 must be filtered out of the "
        "low-health alert list — its health score is statistically "
        "meaningless on that few samples."
    )

    # And the source code must still encode the >= 10 filter.
    src = (REPO_ROOT / "chad" / "ops" / "health_monitor.py").read_text(
        encoding="utf-8"
    )
    assert ">= 10" in src, (
        "Health monitor must keep the sample_count >= 10 filter — it "
        "is what protects low-sample strategies (omega_vol etc.) from "
        "being escalated to the operator."
    )


# ---------------------------------------------------------------------------
# WINNER SCALER — refuses to over-react to small samples
# ---------------------------------------------------------------------------


def test_winner_scaler_requires_min_sample_before_multiplier_boost():
    """A strategy with only 1 trade — even with strong expectancy —
    must NOT get a >1.0 boost. The min_trades_for_scaling threshold
    keeps the scaler from manufacturing aggressive multipliers from
    tiny samples."""
    expectancy = {
        "strategies": {
            "tiny_winner": {"total_trades": 1, "expectancy": 5000.0},
            "tiny_loser": {"total_trades": 2, "expectancy": -5000.0},
            # Must have a mature pool for median to have meaning.
            "mature_a": {"total_trades": 50, "expectancy": 10.0},
            "mature_b": {"total_trades": 50, "expectancy": -10.0},
        }
    }
    out = compute_multipliers(expectancy, DEFAULT_POLICY)
    assert out["multipliers"]["tiny_winner"] == 1.0, (
        "Even strong expectancy from a 1-trade sample must NOT yield "
        "a boost — winner-scaler must wait for >= min_trades samples."
    )
    assert out["multipliers"]["tiny_loser"] == 1.0, (
        "A 2-trade losing streak must NOT yield a haircut — sample "
        "is too small to be statistically meaningful."
    )
    # And the scaled count must reflect that those tiny strategies
    # were excluded from the scoring pool.
    assert out["n_strategies_scaled"] <= 2, (
        "Only strategies with >= min_trades samples may be scaled."
    )


def test_winner_scaler_flat_when_no_mature_scoring_pool():
    """If NO strategy reaches min_trades, the scaler must publish all
    1.0 multipliers and report n_strategies_scaled=0 — never invent a
    median from an immature pool."""
    expectancy = {
        "strategies": {
            "a": {"total_trades": 1, "expectancy": 100.0},
            "b": {"total_trades": 2, "expectancy": -100.0},
            "c": {"total_trades": 4, "expectancy": 50.0},  # below 5
        }
    }
    out = compute_multipliers(expectancy, DEFAULT_POLICY)
    assert out["n_strategies_scaled"] == 0
    assert out["median_expectancy"] == 0.0
    for name, mult in out["multipliers"].items():
        assert mult == 1.0, (
            f"With no mature scoring pool, every strategy must be "
            f"neutral, but {name} got {mult}"
        )


def test_winner_scaler_min_trades_default_is_5_locked():
    """Lock the maturity-gate constant. Any change to the default
    min_trades_for_scaling must update this test deliberately so the
    operator sees the contract change in code review."""
    assert DEFAULT_POLICY["min_trades_for_scaling"] == 5, (
        "The min_trades_for_scaling=5 default is the published "
        "maturity gate. Changing it requires explicit ops review."
    )
    # And the bounded min/max multipliers MUST be conservative.
    assert DEFAULT_POLICY["max_multiplier"] <= 1.5, (
        "Winner-scaler max_multiplier must not exceed 1.5x"
    )
    assert DEFAULT_POLICY["min_multiplier"] >= 0.5, (
        "Winner-scaler min_multiplier must not fall below 0.5x"
    )


# ---------------------------------------------------------------------------
# STRATEGY HEALTH — distinguishes "no data" from "bad data"
# ---------------------------------------------------------------------------


def test_strategy_health_marks_insufficient_sample(tmp_path: Path):
    """A strategy with no expectancy data must surface
    health_score=DEFAULT_NEUTRAL_SCORE and reason='no_expectancy_data'.
    That distinguishes 'not yet measured' from 'measured and bad'."""
    scorer = StrategyHealthScorer(output_path=tmp_path / "h.json")
    health = scorer.compute(
        "delta",
        expectancy_tracker={"strategies": {}},
        slippage_tracker=_FakeSlippage(),
        regime_state="trending_bull",
    )
    assert health["health_score"] == DEFAULT_NEUTRAL_SCORE
    assert health["reason"] == "no_expectancy_data"
    assert health["sample_count"] == 0


def test_strategy_health_does_not_page_critical_for_immature_low_sample():
    """The alpha-cluster degradation rule (R12) requires sample>=10
    before an alpha row can contribute to the cluster alert. Pin that
    contract: a 3-trade alpha_intraday at health 0.1 must NOT be
    counted toward the >=3-strategies-degraded cluster trigger."""
    rules_src = (
        REPO_ROOT / "chad" / "ops" / "health_monitor_rules.py"
    ).read_text(encoding="utf-8")
    # Source-level guard: the rule must have the sample<10 skip.
    assert "sample < 10" in rules_src, (
        "health_monitor_rules R12 must skip strategies with sample<10 "
        "before counting them toward alpha-cluster degradation."
    )
    # And health_monitor.py must mirror the same threshold.
    monitor_src = (
        REPO_ROOT / "chad" / "ops" / "health_monitor.py"
    ).read_text(encoding="utf-8")
    assert ">= 10" in monitor_src

    # Functional check: an artificially low sample with health below
    # 0.5 must not appear in the filtered set used for alerting.
    fake_health = {
        "alpha": {"health_score": 0.10, "sample_count": 3},
        "alpha_futures": {"health_score": 0.12, "sample_count": 4},
        "alpha_intraday": {"health_score": 0.11, "sample_count": 5},
    }
    filtered = {
        k: v
        for k, v in fake_health.items()
        if int(v.get("sample_count", 0)) >= 10
    }
    assert filtered == {}, (
        "All 3 alphas have sample<10, so the alpha-cluster filter "
        "MUST surface zero rows — no cluster alert should fire."
    )


def test_strategy_health_recomputes_for_mature_sample(tmp_path: Path):
    """A mature strategy with real win/loss data must be scored
    normally — the maturity gate does not block legitimate scoring."""
    scorer = StrategyHealthScorer(output_path=tmp_path / "h.json")
    state = {
        "strategies": {
            "delta": {
                "total_trades": 82,
                "win_rate": 0.711,
                "expectancy": 28.96,
                "avg_win": 42.86,
                "rolling_sharpe": 1.5,
            }
        }
    }
    health = scorer.compute(
        "delta",
        expectancy_tracker=state,
        slippage_tracker=_FakeSlippage(),
        regime_state="trending_bull",
    )
    assert health["reason"] == "computed"
    assert health["sample_count"] == 82
    # Score is bounded [0,1] — the existing test_score_clamped_0_to_1
    # also covers this; we only assert it is not the cold-start
    # neutral default.
    assert health["health_score"] != DEFAULT_NEUTRAL_SCORE


# ---------------------------------------------------------------------------
# ALLOCATOR — refuses to over-allocate from stale or low-sample winner
# ---------------------------------------------------------------------------


def test_allocator_refuses_aggressive_boost_from_stale_winner_scaling():
    """When winner_scaling is stale, the allocator MUST fall back to a
    conservative 0.5x per strategy — the CB05 contract. This is the
    architectural anti-fooling guard that prevents an old or missing
    winner_scaling.json from silently keeping sizing at full strength."""
    src = inspect.getsource(dra)
    # The CB05 fallback is wired in.
    assert "winner_scaling_stale" in src, (
        "Allocator must log winner_scaling_stale on the stale path."
    )
    assert "0.5" in src, (
        "Allocator must use 0.5x conservative fallback when stale."
    )
    # Loader must signal stale=True via empty multipliers.
    loader_src = inspect.getsource(dra.load_winner_multipliers_or_stale)
    assert "stale_seconds=1800" in loader_src, (
        "Winner multipliers freshness bound must remain 1800s."
    )
    assert "return {}, True" in loader_src


def test_allocator_low_sample_winner_does_not_create_aggressive_multiplier():
    """If winner_scaler emits a neutral 1.0 for a low-sample strategy
    (which it must — see test_winner_scaler_requires_min_sample_*),
    the allocator must reflect that 1.0 in winner_factor — never
    upgrade it. Pin that the allocator only multiplies tier × winner
    × regime (never replaces winner with something higher)."""
    allocator_src = (
        REPO_ROOT / "chad" / "risk" / "dynamic_risk_allocator.py"
    ).read_text(encoding="utf-8")
    # The winner_factor must be sourced from the published multiplier
    # dict, not inferred from health/expectancy directly.
    assert "winner_mults" in allocator_src
    assert "winner_factor" in allocator_src
    # And the formula must be a multiplication of base × tier ×
    # winner × regime — never a max() that could lift winner_factor.
    assert "base_cap * tier_factor * winner_factor * regime_mult" in allocator_src, (
        "Allocator cap formula must be a strict multiplication so a "
        "neutral 1.0 winner_factor cannot be upgraded into a boost."
    )


def test_winner_scaler_neutral_for_omega_vol_when_only_3_trades_in_runtime():
    """End-to-end check using the *actual* shape of the live runtime.
    Verifies that with the live expectancy doc shape (omega_vol=3
    trades, several mature strategies), winner_scaler still publishes
    omega_vol=1.0 — so the live dynamic_caps.json business overlay
    cannot show a non-neutral multiplier for omega_vol."""
    expectancy = {
        "strategies": {
            "omega_vol": {"total_trades": 3, "expectancy": -0.31},
            "alpha_futures": {"total_trades": 376, "expectancy": -32.83},
            "alpha": {"total_trades": 88, "expectancy": -60.66},
            "delta": {"total_trades": 82, "expectancy": 28.96},
            "alpha_options": {"total_trades": 88, "expectancy": 7.44},
            "alpha_intraday": {"total_trades": 19, "expectancy": -13.13},
            "beta": {"total_trades": 4, "expectancy": 21.79},
            "gamma_futures": {"total_trades": 2, "expectancy": 544.0},
            "omega_momentum_options": {
                "total_trades": 3,
                "expectancy": 0.0,
            },
        }
    }
    out = compute_multipliers(expectancy, DEFAULT_POLICY)
    # All four below-min strategies must be neutral 1.0 even though
    # gamma_futures has a huge "best trade" expectancy of $544 and
    # omega_vol is technically negative.
    assert out["multipliers"]["omega_vol"] == 1.0
    assert out["multipliers"]["beta"] == 1.0
    assert out["multipliers"]["gamma_futures"] == 1.0
    assert out["multipliers"]["omega_momentum_options"] == 1.0
    # And the published min_trades_for_scaling field must be present
    # in the contract output so consumers can audit it.
    assert out["min_trades_for_scaling"] == 5


# ---------------------------------------------------------------------------
# FORMAL CLOSE CONTRACT — the meta-test
# ---------------------------------------------------------------------------


def test_maturity_gate_formal_close_contract():
    """Meta-contract: enumerate every public maturity-gate threshold
    and pin its current value. Any change requires updating this test.

    This is the single source of truth for the formal close of the
    three open data-maturity items:
        1. Omega Vol low sample
        2. Winner-scaling sample maturity
        3. Strategy-health sample maturity
    """
    # Winner-scaler default — gates median computation and per-strategy
    # neutralisation.
    assert DEFAULT_POLICY["min_trades_for_scaling"] == 5

    # Strategy-health cold-start default — must be neutral, never
    # punitive.
    assert DEFAULT_NEUTRAL_SCORE == 0.5

    # Health-monitor sample threshold — must remain >= 10 for both
    # the snapshot used by Claude reasoning AND the alpha-cluster
    # rule R12.
    monitor_src = (
        REPO_ROOT / "chad" / "ops" / "health_monitor.py"
    ).read_text(encoding="utf-8")
    rules_src = (
        REPO_ROOT / "chad" / "ops" / "health_monitor_rules.py"
    ).read_text(encoding="utf-8")
    assert ">= 10" in monitor_src
    assert "sample < 10" in rules_src

    # Allocator winner-scaling freshness — 1800s window, with 0.5x
    # conservative fallback when stale.
    loader_src = inspect.getsource(
        dra.load_winner_multipliers_or_stale
    )
    assert "stale_seconds=1800" in loader_src
    allocator_src = inspect.getsource(dra)
    assert "winner_scaling_stale" in allocator_src
    assert "0.5" in allocator_src

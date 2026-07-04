"""Known-answer + property tests for chad.validation.significance (Phase 3).

The anti-off-by-one anchors are two PUBLISHED worked examples, hand-checked in the
test docstrings (no expected value is produced by re-running the module under test):

  * The Deflated Sharpe Ratio pipeline reproduces the worked example widely cited
    from López de Prado's framework (see marti.ai's reproduction, "How to detect
    false strategies? The Deflated Sharpe Ratio", 2018, and Bailey & López de Prado
    2014, *The Deflated Sharpe Ratio*): SR_hat = 2.5/√252, V = 0.5/252, N = 100,
    n = 1250, skew = −3, kurt = 10 → DSR ≈ 0.8997.
  * The Probabilistic Sharpe Ratio reproduces Bailey & López de Prado's 24-month
    example (2012, *The Sharpe Ratio Efficient Frontier*): a fund with monthly
    SR = 0.458, skew = −2.448, kurt = 10.164 over n = 24 months. Under a Normal fit
    PSR(0) ≈ 0.982; admitting the fat tails drops it to ≈ 0.913 — the published
    point that non-Normality is a real hit to confidence.

The block-bootstrap-preserves-loss-clustering guarantee is anchored by an
explicitly constructed clustered-loss series on which block-bootstrap ruin must
exceed IID-bootstrap ruin (the two resamplers, same threshold, same paths).

Fixtures are tiny in-memory sequences; nothing here touches the bar corpus, the
network, or any runtime state. ``statistics.NormalDist`` is used in a few tests as
an INDEPENDENT re-derivation of the closed-form composition (it is a different code
path from the module's arithmetic assembly — it guards the weights/1-1/N/(Ne)/γ
composition, which is where the deflation off-by-ones live, not Φ itself).
"""

from __future__ import annotations

import json
import math
import statistics

import pytest

from chad.validation.significance import (
    DEFAULT_SEED_SET,
    DEFAULT_TRIAL_MULTIPLE,
    EULER_MASCHERONI,
    DeflatedSharpeResult,
    RuinConfig,
    RuinResult,
    SeedSweepResult,
    deflated_sharpe_ratio,
    estimate_ruin,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    punitive_trial_count,
    sample_kurtosis,
    sample_skewness,
    seed_sweep,
    time_average_growth_rate,
    worst_quantile,
)
from chad.validation.significance import (
    _estimate_block_length,
    _lag1_autocorrelation,
)

_NORMAL = statistics.NormalDist(0.0, 1.0)


# =========================================================================== #
# 1. DEFLATED SHARPE — the published worked-example anti-off-by-one anchor.
# =========================================================================== #
def test_dsr_published_worked_example_known_answer():
    """PUBLISHED DSR worked example → DSR ≈ 0.8997 (the anti-off-by-one anchor).

    Inputs (López de Prado framework; marti.ai 2018 reproduction):
      estimated (per-period) Sharpe = 2.5/√252 = 0.15748773
      cross-trial Sharpe variance V = 0.5/252 = 0.00198413
      N trials = 100, backtest horizon n = 1250, skew = −3, raw kurt = 10.

    Hand derivation:
      SR*_0 = √V·[(1−γ)·Φ⁻¹(0.99) + γ·Φ⁻¹(1−1/(100e))]
            = 0.04454357·[0.42278434·2.32634787 + 0.57721566·2.68021]
            = 0.04454357·2.530479 = 0.112722
      denom = √(1 − (−3)(0.15748773) + (10−1)/4·0.15748773²)
            = √(1 + 0.47246319 + 0.05580536) = √1.52826855 = 1.23623160
      z = (0.15748773 − 0.112722)·√1249 / 1.23623160 = 1.582005/1.23623160 = 1.279693
      DSR = Φ(1.279693) = 0.89967  →  published value 0.8997.
    """
    sr = 2.5 / math.sqrt(252)
    res = deflated_sharpe_ratio(sr, 1250, -3.0, 10.0, 100, trials_sharpe_variance=0.5 / 252)
    assert isinstance(res, DeflatedSharpeResult)
    assert res.deflated_sharpe_ratio == pytest.approx(0.8997, abs=2e-3)
    # SR*_0 pinned to the hand-derived published sub-result (guards the deflation half).
    assert res.expected_max_sharpe == pytest.approx(0.112722, abs=1e-4)
    # Provenance is logged (SSOT §3.3: N printed in every report), V not defaulted.
    assert res.n_effective_trials == 100
    assert res.trials_variance_defaulted is False
    assert res.trials_sharpe_variance == pytest.approx(0.5 / 252, rel=1e-12)


def test_psr_published_24_month_example_known_answer():
    """PUBLISHED PSR example (Bailey & López de Prado 2012, 24-month fund).

    SR = 0.458 (monthly), n = 24. Under a Normal fit (skew 0, kurt 3):
      denom = √(1 + 0.5·0.458²) = √1.104882 = 1.05113368
      z = 0.458·√23 / 1.05113368 = 2.19649084/1.05113368 = 2.089618
      PSR(0) = Φ(2.089618) = 0.98168.
    Admitting the true fat tails (skew −2.448, kurt 10.164):
      denom = √(1 + 2.448·0.458 + (10.164−1)/4·0.458²)
            = √(1 + 1.121184 + 0.480569) = √2.601753 = 1.61300437
      z = 0.458·√23 / 1.61300437 = 1.361844
      PSR(0) = Φ(1.361844) = 0.91336.
    The published finding: non-Normality is a real hit — confidence falls ~0.982→0.913.
    """
    psr_normal = probabilistic_sharpe_ratio(0.458, 24, 0.0, 3.0)
    psr_fat = probabilistic_sharpe_ratio(0.458, 24, -2.448, 10.164)
    assert psr_normal == pytest.approx(0.98168, abs=1e-3)
    assert psr_fat == pytest.approx(0.91336, abs=1e-3)
    # The published qualitative point: fat tails LOWER confidence.
    assert psr_fat < psr_normal


def test_psr_raw_kurtosis_convention_normal_recovers_lo_denominator():
    """The (γ4−1)/4 term must use RAW kurtosis (Normal=3), i.e. Lo (2002) SE.

    With skew 0 and kurt 3 the variance term is exactly 1 + SR²/2 (Lo's normal-returns
    Sharpe standard error, ×n). If the code wrongly used EXCESS kurtosis (Normal=0)
    the term would be 1 + (0−1)/4·SR² = 1 − SR²/4 and PSR would differ sharply. Here
    SR = √2 makes 1 + SR²/2 = 2 exactly, so z = √2·√(n−1)/√2 = √(n−1); pick n = 5 →
    z = 2 → PSR(0) = Φ(2).
    """
    psr = probabilistic_sharpe_ratio(math.sqrt(2.0), 5, 0.0, 3.0)
    assert psr == pytest.approx(_NORMAL.cdf(2.0), abs=1e-12)  # 0.9772498...


def test_psr_at_own_sharpe_is_one_half():
    """PSR evaluated at benchmark == observed Sharpe → z = 0 → Φ(0) = 0.5 exactly."""
    psr = probabilistic_sharpe_ratio(0.3, 100, -0.5, 4.0, benchmark_sharpe=0.3)
    assert psr == pytest.approx(0.5, abs=1e-12)


def test_dsr_monotonically_decreases_as_trials_rise():
    """More trials ⇒ higher SR*_0 ⇒ lower DSR (the whole point of the deflation)."""
    sr = 0.15
    dsrs = []
    srstars = []
    for n_trials in (10, 100, 1000, 10000):
        res = deflated_sharpe_ratio(sr, 1250, -1.0, 5.0, n_trials, trials_sharpe_variance=0.5 / 252)
        dsrs.append(res.deflated_sharpe_ratio)
        srstars.append(res.expected_max_sharpe)
    assert all(dsrs[i] > dsrs[i + 1] for i in range(len(dsrs) - 1)), dsrs
    assert all(srstars[i] < srstars[i + 1] for i in range(len(srstars) - 1)), srstars


# =========================================================================== #
# 2. Expected-max Sharpe (False Strategy Theorem) — composition + properties.
# =========================================================================== #
def test_expected_max_sharpe_single_trial_is_zero():
    """N = 1: no selection bias; E[max of one standard-normal draw] = 0 exactly."""
    assert expected_max_sharpe(1, trials_sharpe_variance=0.25) == 0.0


def test_expected_max_sharpe_matches_independent_formula_rederivation():
    """Re-derive SR*_0 independently via NormalDist (guards the weight composition).

    For N = 50, V = 4 (√V = 2):
      SR*_0 = 2·[(1−γ)·Φ⁻¹(1−1/50) + γ·Φ⁻¹(1−1/(50e))].
    This checks the (1−γ)/γ split, the 1−1/N and 1−1/(Ne) arguments, and √V — the
    exact places a deflation off-by-one would hide — against a hand-written copy of
    the formula (Φ⁻¹ itself is the same trusted stdlib routine, deliberately).
    """
    n, v = 50, 4.0
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / n)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (n * math.e))
    expected = math.sqrt(v) * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    assert expected_max_sharpe(n, trials_sharpe_variance=v) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_tracks_sqrt_2lnN_and_is_monotone():
    """Sanity vs the leading asymptotic √(2·ln N): the López de Prado estimator sits
    just BELOW it (it carries the sub-leading Gumbel correction the leading term
    drops) and rises monotonically in N. Bracket 0.8·√(2lnN) < E[max] < √(2lnN)."""
    prev = -1.0
    for n in (1_000, 100_000, 10_000_000):
        val = expected_max_sharpe(n, trials_sharpe_variance=1.0)
        lead = math.sqrt(2.0 * math.log(n))
        assert 0.8 * lead < val < lead, (n, val, lead)
        assert val > prev  # monotone increasing in N
        prev = val


def test_expected_max_sharpe_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        expected_max_sharpe(100, trials_sharpe_variance=0.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(100, trials_sharpe_variance=-1.0)


def test_expected_max_sharpe_rejects_bad_trial_count():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, trials_sharpe_variance=1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(True, trials_sharpe_variance=1.0)  # bool rejected as int


# =========================================================================== #
# 3. Punitive trial count (S1).
# =========================================================================== #
def test_punitive_trial_count_applies_multiple():
    """17 survivors × default 7 = 119 (SSOT §3.3 punitive N)."""
    assert punitive_trial_count(17) == 119
    assert DEFAULT_TRIAL_MULTIPLE == 7.0
    assert punitive_trial_count(17, multiple=10) == 170


def test_punitive_trial_count_never_below_survivor_floor():
    """A misconfigured multiple < 1 can never weaken below the survivor count."""
    assert punitive_trial_count(17, multiple=0.5) == 17
    assert punitive_trial_count(17, multiple=1.0) == 17


def test_punitive_trial_count_invalid_config_raises():
    with pytest.raises(ValueError):
        punitive_trial_count(0)
    with pytest.raises(ValueError):
        punitive_trial_count(17, multiple=0.0)
    with pytest.raises(ValueError):
        punitive_trial_count(True)  # bool rejected


# =========================================================================== #
# 4. DSR defaults, provenance, degenerate sentinels.
# =========================================================================== #
def test_dsr_defaults_variance_to_null_floor():
    """No V supplied → V = 1/(n−1) floor, flagged as defaulted (logged provenance)."""
    res = deflated_sharpe_ratio(0.1, 101, 0.0, 3.0, 100)
    assert res.trials_variance_defaulted is True
    assert res.trials_sharpe_variance == pytest.approx(1.0 / 100, rel=1e-12)
    assert any("defaulted" in w for w in res.warnings)


def test_dsr_degenerate_too_few_returns_is_sentinel():
    """n_returns < 2 → None DSR/PSR/SR*_0 with a reason, never a raise."""
    res = deflated_sharpe_ratio(0.5, 1, 0.0, 3.0, 100)
    assert res.deflated_sharpe_ratio is None
    assert res.psr_vs_zero is None
    assert res.expected_max_sharpe is None
    assert res.n_returns == 1
    assert res.n_effective_trials == 100  # still echoed
    # Undefined V is the None sentinel (not NaN) so to_dict() is standard JSON.
    assert res.trials_sharpe_variance is None
    json.dumps(res.to_dict())  # must not emit a bare NaN token
    assert res.warnings


def test_dsr_negative_variance_term_is_sentinel():
    """Extreme skew×Sharpe drives the PSR variance term ≤ 0 → None (documented)."""
    # skew * SR huge and positive makes 1 - skew*SR negative.
    psr = probabilistic_sharpe_ratio(5.0, 100, 10.0, 3.0)
    assert psr is None
    res = deflated_sharpe_ratio(5.0, 100, 10.0, 3.0, 100, trials_sharpe_variance=0.5)
    assert res.deflated_sharpe_ratio is None
    assert res.warnings


def test_dsr_invalid_config_raises():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, 100, 0.0, 3.0, 0)  # n_trials < 1
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, 100, 0.0, 3.0, 100, trials_sharpe_variance=0.0)
    with pytest.raises(TypeError):
        deflated_sharpe_ratio("x", 100, 0.0, 3.0, 100)  # non-number Sharpe


def test_dsr_result_is_json_serialisable():
    res = deflated_sharpe_ratio(0.15, 1250, -1.0, 5.0, 100, trials_sharpe_variance=0.5 / 252)
    json.dumps(res.to_dict())  # must not raise


# =========================================================================== #
# 5. Sample moments — raw-kurtosis convention known-answers.
# =========================================================================== #
def test_sample_skewness_and_kurtosis_known_answers():
    """[-2,-1,0,1,2]: mean 0, m2 2, m3 0, m4 6.8 → skew 0, RAW kurt 6.8/4 = 1.7.

    A raw-kurtosis Normal is 3.0; this symmetric 5-point set is platykurtic (1.7),
    confirming the estimator is raw (not excess, which would be 1.7 − 3 = −1.3).
    """
    assert sample_skewness([-2, -1, 0, 1, 2]) == pytest.approx(0.0, abs=1e-12)
    assert sample_kurtosis([-2, -1, 0, 1, 2]) == pytest.approx(1.7, rel=1e-12)


def test_sample_skewness_kurtosis_skewed_known_answer():
    """[0,0,0,0,10]: mean 2 → m2 16, m3 96, m4 832 → skew 96/64 = 1.5, kurt 832/256 = 3.25."""
    assert sample_skewness([0, 0, 0, 0, 10]) == pytest.approx(1.5, rel=1e-12)
    assert sample_kurtosis([0, 0, 0, 0, 10]) == pytest.approx(3.25, rel=1e-12)


def test_sample_moments_degenerate_sentinels():
    assert sample_skewness([1.0]) is None
    assert sample_kurtosis([]) is None
    assert sample_skewness([2.0, 2.0, 2.0]) is None  # zero variance


def test_per_period_sharpe_known_answer_and_sentinels():
    """[0.01,0.02,0.03]: mean 0.02, sample var 0.0001, stdev 0.01 → Sharpe 2.0."""
    assert per_period_sharpe([0.01, 0.02, 0.03]) == pytest.approx(2.0, rel=1e-12)
    assert per_period_sharpe([0.01]) is None
    assert per_period_sharpe([0.05, 0.05, 0.05]) is None  # zero vol


# =========================================================================== #
# 6. Ergodicity — time-average growth.
# =========================================================================== #
def test_time_average_growth_is_geometric_not_arithmetic():
    """[+1.0, −0.5]: arithmetic mean +0.25 but geometric growth = 0 exactly.

    gross = (2.0)(0.5) = 1.0 → per-period geo growth = 1.0^(1/2) − 1 = 0. The
    textbook ergodicity gap: a positive ensemble mean with zero time-average growth.
    """
    assert time_average_growth_rate([1.0, -0.5]) == pytest.approx(0.0, abs=1e-15)


def test_time_average_growth_sentinels():
    assert time_average_growth_rate([]) is None
    assert time_average_growth_rate([-1.0]) is None  # total wipeout → log undefined
    assert time_average_growth_rate([0.1, -1.2]) is None  # 1+r <= 0


# =========================================================================== #
# 7. Block bootstrap PRESERVES loss-clustering — the reviewer-critical guarantee.
# =========================================================================== #
# Clustered-loss series: 84 quiet +0.4% days then a 4-day −8% crash cluster, ×3.
# Near-zero time-average drift, so IID paths (which scatter the crash days) rarely
# reach a 20% drawdown, while block paths (which keep the crash intact) frequently
# do. A single crash cluster compounds to 1−0.92⁴ = 28.4% > 20% threshold.
_CLUSTERED = ([0.004] * 84 + [-0.08] * 4) * 3


def test_block_bootstrap_ruin_exceeds_iid_on_clustered_losses():
    """Block-bootstrap ruin > IID-bootstrap ruin on a clustered-loss series (S2).

    This is the physical proof that the stationary block bootstrap preserves
    loss-clustering: keeping the −8% crash days consecutive produces deep drawdowns
    the IID resampler dilutes away. Asserted across several seeds so it is not a
    single-seed fluke; the empirical gap is ≈ +0.27.
    """
    cfg = RuinConfig(bootstrap_paths=600, ruin_drawdown_threshold=0.20, path_length=88)
    for seed in range(6):
        rr = estimate_ruin(_CLUSTERED, seed=seed, config=cfg)
        assert rr.ruin_prob_block > rr.ruin_prob_iid, (seed, rr.ruin_prob_block, rr.ruin_prob_iid)
        assert rr.ruin_prob_worse == max(rr.ruin_prob_block, rr.ruin_prob_iid)
        # neither saturated (a saturated pair could not demonstrate the effect)
        assert 0.0 < rr.ruin_prob_iid < 1.0
        assert 0.0 < rr.ruin_prob_block <= 1.0


def test_block_length_tracks_autocorrelation():
    """Positive lag-1 autocorrelation lengthens blocks; IID / mean-reverting → 1."""
    assert _estimate_block_length(_CLUSTERED) > 1  # clustered → block > 1
    assert _lag1_autocorrelation(_CLUSTERED) > 0.0
    assert _estimate_block_length([0.01, -0.01] * 40) == 1  # perfectly alternating
    assert _estimate_block_length([0.02] * 10) == 1  # zero dispersion → 1


def test_block_length_one_reduces_to_iid_equivalence():
    """With mean_block_length forced to 1 the block resampler == the IID resampler,
    so their ruin estimates are close (both destroy clustering)."""
    cfg = RuinConfig(
        bootstrap_paths=800, ruin_drawdown_threshold=0.20, path_length=88, mean_block_length=1
    )
    rr = estimate_ruin(_CLUSTERED, seed=3, config=cfg)
    assert rr.mean_block_length == 1
    assert rr.ruin_prob_block == pytest.approx(rr.ruin_prob_iid, abs=0.06)


# =========================================================================== #
# 8. Ruin determinism + degenerate sentinels.
# =========================================================================== #
def test_estimate_ruin_is_deterministic_given_seed():
    cfg = RuinConfig(bootstrap_paths=300, ruin_drawdown_threshold=0.2, path_length=88)
    a = estimate_ruin(_CLUSTERED, seed=7, config=cfg)
    b = estimate_ruin(_CLUSTERED, seed=7, config=cfg)
    assert a.to_dict() == b.to_dict()
    # A different seed generally differs (guards against an ignored seed).
    c = estimate_ruin(_CLUSTERED, seed=8, config=cfg)
    assert (a.ruin_prob_block, a.ruin_prob_iid) != (c.ruin_prob_block, c.ruin_prob_iid) or \
        a.max_drawdown_block != c.max_drawdown_block


def test_ruin_drawdown_magnitude_capped_at_full_ruin():
    """A bankrupt path (a return ≤ −1) caps max drawdown at 1.0, never > 100%.

    [-1.5] alone is degenerate (n<2), so embed a wipeout inside a ≥2-distinct series
    and force block length 1 so paths mix the −1.5 bar in; every reported drawdown
    quantile must stay ≤ 1.0.
    """
    series = [0.01, -1.5, 0.02, 0.03, 0.01]
    cfg = RuinConfig(bootstrap_paths=200, ruin_drawdown_threshold=0.5, mean_block_length=1)
    rr = estimate_ruin(series, seed=0, config=cfg)
    for dd in list(rr.max_drawdown_block.values()) + list(rr.max_drawdown_iid.values()):
        assert 0.0 <= dd <= 1.0, dd


def test_estimate_ruin_degenerate_sentinels():
    cfg = RuinConfig(bootstrap_paths=100)
    too_few = estimate_ruin([0.01], seed=0, config=cfg)
    assert too_few.ruin_prob_worse is None and too_few.warnings
    flat = estimate_ruin([0.01] * 40, seed=0, config=cfg)  # zero dispersion
    assert flat.ruin_prob_block is None and flat.max_drawdown_block == {}
    empty = estimate_ruin([], seed=0, config=cfg)
    assert empty.ruin_prob_worse is None and empty.time_average_growth is None


def test_estimate_ruin_result_json_serialisable():
    cfg = RuinConfig(bootstrap_paths=100, path_length=88)
    json.dumps(estimate_ruin(_CLUSTERED, seed=0, config=cfg).to_dict())


def test_ruin_config_invalid_raises():
    with pytest.raises(ValueError):
        RuinConfig(bootstrap_paths=0)
    with pytest.raises(ValueError):
        RuinConfig(ruin_drawdown_threshold=0.0)
    with pytest.raises(ValueError):
        RuinConfig(ruin_drawdown_threshold=1.5)
    with pytest.raises(ValueError):
        RuinConfig(path_length=0)
    with pytest.raises(ValueError):
        RuinConfig(mean_block_length=0)


def test_estimate_ruin_rejects_negative_seed():
    with pytest.raises(ValueError):
        estimate_ruin(_CLUSTERED, seed=-1)


# =========================================================================== #
# 9. Seed sweep (S3) — distributions, determinism, worst-quantile.
# =========================================================================== #
def test_seed_sweep_is_deterministic():
    cfg = RuinConfig(bootstrap_paths=200, ruin_drawdown_threshold=0.2, path_length=88)
    seeds = tuple(range(6))
    a = seed_sweep(_CLUSTERED, n_effective_trials=100, seeds=seeds, config=cfg,
                   trials_sharpe_variance=0.5 / 252)
    b = seed_sweep(_CLUSTERED, n_effective_trials=100, seeds=seeds, config=cfg,
                   trials_sharpe_variance=0.5 / 252)
    assert a.to_dict() == b.to_dict()
    assert isinstance(a, SeedSweepResult)
    assert len(a.ruin_results) == len(seeds)


def test_seed_sweep_worst_quantile_ruin_and_dsr():
    """ruin worst-quantile = an UPPER quantile (higher ruin is worse); dsr worst =
    a LOWER quantile (lower DSR is worse). Both sit inside the realized range."""
    cfg = RuinConfig(bootstrap_paths=200, ruin_drawdown_threshold=0.2, path_length=88)
    sw = seed_sweep(_CLUSTERED, n_effective_trials=100, seeds=tuple(range(12)), config=cfg,
                    trials_sharpe_variance=0.5 / 252)
    # tail=0.05 → the 95th-percentile ruin (the adverse upper tail).
    rwq = sw.ruin_worst_quantile(0.05)
    assert rwq is not None
    assert rwq >= sw.mean_ruin_worse()  # upper-tail quantile >= mean
    assert min(sw.ruin_worse_distribution) <= rwq <= max(sw.ruin_worse_distribution)
    if sw.dsr_distribution:
        dwq = sw.dsr_worst_quantile(0.05)  # 5th-percentile DSR (adverse lower tail)
        assert dwq <= sw.mean_dsr()  # lower-tail quantile <= mean


def test_seed_sweep_default_seed_set_is_25():
    assert len(DEFAULT_SEED_SET) == 25
    assert DEFAULT_SEED_SET == tuple(range(25))


def test_seed_sweep_invalid_config_raises():
    with pytest.raises(ValueError):
        seed_sweep(_CLUSTERED, n_effective_trials=0)
    with pytest.raises(ValueError):
        seed_sweep(_CLUSTERED, n_effective_trials=100, seeds=())
    with pytest.raises(ValueError):
        seed_sweep(_CLUSTERED, n_effective_trials=100, seeds=(1, 1, 2))  # duplicate


# =========================================================================== #
# 10. worst_quantile helper — worst-quantile < mean on a skewed distribution.
# =========================================================================== #
def test_worst_quantile_below_mean_for_skewed_distribution():
    """Right-skewed sample: the low (worst, lower_is_worse) quantile sits below mean.

    [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.9,2.0] has mean 0.465; the long right tail
    pulls the mean above the bulk, so the 10th-percentile worst quantile ≈ 0.09 < mean.
    """
    skewed = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.9, 2.0]
    mean = statistics.fmean(skewed)
    wq = worst_quantile(skewed, 0.1, lower_is_worse=True)
    assert wq < mean


def test_worst_quantile_direction_and_sentinels():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    # lower_is_worse → lower quantile; not-lower → upper quantile; they straddle the median.
    low = worst_quantile(vals, 0.1, lower_is_worse=True)
    high = worst_quantile(vals, 0.1, lower_is_worse=False)
    assert low < 3.0 < high
    assert worst_quantile([], 0.5, lower_is_worse=True) is None
    with pytest.raises(ValueError):
        worst_quantile(vals, 1.5, lower_is_worse=True)


# =========================================================================== #
# 11. Determinism of the DSR path (pure function).
# =========================================================================== #
def test_dsr_is_pure_and_reproducible():
    args = (0.12, 500, -0.7, 6.0, 84)
    a = deflated_sharpe_ratio(*args, trials_sharpe_variance=0.003)
    b = deflated_sharpe_ratio(*args, trials_sharpe_variance=0.003)
    assert a.to_dict() == b.to_dict()

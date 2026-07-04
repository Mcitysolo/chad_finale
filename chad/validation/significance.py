"""chad/validation/significance.py — Phase 3 significance + ruin/ergodicity.

The edge-validation harness's "teeth": the multiple-testing-corrected Sharpe test
(Deflated Sharpe Ratio) and the ruin/ergodicity block-bootstrap machinery (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §3.3 / §3.6). This module
computes **metrics only** — it renders NO pass/fail verdict (thresholds and the
worst-quantile decision live in Phase 5 ``verdict.py``). It exposes the raw
numbers a verdict will later threshold, plus a worst-quantile accessor so Phase 5
can apply "trust the worse number" without re-deriving the distribution.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic. Imports only the
standard library — ``math``, ``random``, ``statistics``, ``dataclasses``,
``typing``. numpy is present in the environment (2.3.x) but is deliberately
avoided (the isolation test asserts it stays out of the closure): the bootstrap
is seeded ``random.Random`` so a fixed seed set yields byte-identical results,
which numpy's global RNG state would put at risk. ``statistics.NormalDist`` is the
trusted stdlib standard-normal CDF / inverse-CDF — hand-rolling ``erf``/``erfinv``
is exactly the kind of precision bug this module is built to forbid, so we use the
library implementation rather than re-derive it.

================================================================================
1. DEFLATED SHARPE RATIO (SSOT §3.3, S1) — Bailey & López de Prado
================================================================================
The Deflated Sharpe Ratio (DSR) is the Probabilistic Sharpe Ratio (PSR) evaluated
at a benchmark equal to the *expected maximum* Sharpe ratio that ``N`` zero-skill
trials would produce by chance (the "False Strategy Theorem"). It corrects the
Sharpe ratio for (a) non-Normal returns and (b) selection under multiple testing.

**Probabilistic Sharpe Ratio** (Bailey & López de Prado 2012, "The Sharpe Ratio
Efficient Frontier", J. Risk):

        PSR(SR*) = Φ( (SR_hat − SR*) · sqrt(n − 1)
                      / sqrt( 1 − γ3·SR_hat + (γ4 − 1)/4 · SR_hat² ) )

  where SR_hat is the **per-period, non-annualized** observed Sharpe ratio, n is
  the number of return observations, γ3 is the (non-excess) skewness, γ4 is the
  **raw / non-excess kurtosis (Normal = 3, NOT excess where Normal = 0)**, Φ is the
  standard-normal CDF, and SR* is a benchmark Sharpe. The denominator is exactly
  sqrt(n) × Lo's (2002, "The Statistics of Sharpe Ratios", FAJ) standard error of
  the Sharpe estimator; under Normality (γ3=0, γ4=3) the (γ4−1)/4 term becomes
  (3−1)/4 = 1/2, recovering Lo's sqrt(1 + SR_hat²/2). Using **excess** kurtosis
  here would be the classic off-by-one that silently defeats the whole machine —
  see the docstring convention above and the published known-answer tests.

**Expected maximum Sharpe of N trials** (Bailey & López de Prado 2014, "The
Deflated Sharpe Ratio", J. Portfolio Management — the False Strategy Theorem):

        SR*_0 = sqrt(V) · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

  where V is the **variance of the Sharpe ratios across the N trials**, γ is the
  Euler–Mascheroni constant (≈ 0.5772156649), e is Euler's number, N is the trial
  count, and Φ⁻¹ is the inverse standard-normal CDF. This is the Gumbel /
  extreme-value approximation to E[max of N i.i.d. standard normals]; for large N
  it grows like sqrt(2·ln N).

        DSR = PSR(SR*_0)   ∈ [0, 1]   (a probability the edge is real net of luck)

**Punitive N (S1).** Only surviving heads are visible; every abandoned head,
parameter sweep, and classifier iteration was also a trial. So N is *punitive*: a
documented multiple (default ``DEFAULT_TRIAL_MULTIPLE`` = 7×, in the SSOT's 5–10×
band) of the surviving-head count, never below it. ``n_effective_trials`` is a
**required, logged** parameter of :func:`deflated_sharpe_ratio`; :func:`punitive_trial_count`
computes it transparently from the survivor count and the multiple, and the chosen
N + multiple are echoed in every result for the report.

**Cross-trial variance V.** ``trials_sharpe_variance`` is the empirical V above.
When it is not supplied it defaults to the null sampling-variance floor
``1 / (n − 1)`` — the variance of a single zero-skill strategy's per-period Sharpe
estimator under Normality (SR=0 in the PSR denominator). This floor is honest but
usually *smaller* than a real search's cross-trial dispersion; a larger V yields a
larger SR*_0 and thus a lower (more punitive) DSR, so the operator SHOULD pass the
measured cross-trial variance when a real trial set exists. The value used and
whether it was defaulted are both echoed in the result.

Known-answer anchors (see tests): the DSR pipeline reproduces the published worked
example (SR_hat = 2.5/√252, V = 0.5/252, N = 100, n = 1250, skew = −3, kurt = 10 →
DSR ≈ 0.8997) and the PSR reproduces Bailey & López de Prado's 24-month example
(SR = 0.458, skew = −2.448, kurt = 10.164, n = 24 → PSR falls from ≈ 0.982 under a
Normal fit to ≈ 0.913 once the fat tails are admitted).

================================================================================
2. RUIN / ERGODICITY (SSOT §3.6, S2/S3) — block bootstrap + seed sweep
================================================================================
- **Time-average growth** (ergodicity): the per-period compound growth rate
  ``exp(mean(ln(1+r))) − 1`` — what a single bankroll compounds at through time,
  NOT the ensemble arithmetic mean. A positive arithmetic mean with a negative
  time-average growth is a ruin signature.
- **Ruin probability** is estimated by resampling the empirical return series into
  many synthetic equity paths and measuring how often a path's peak-to-trough
  drawdown reaches ``ruin_drawdown_threshold``. Two resamplers are run and BOTH are
  reported (SSOT: *trust the worse number*):
    * **stationary block bootstrap** (Politis & Romano 1994): geometric-length
      circular blocks with mean length tied to the observed lag-1 autocorrelation,
      so runs of clustered losses survive resampling → deeper drawdowns.
    * **IID bootstrap**: single returns drawn with replacement — destroys
      loss-clustering and therefore *understates* ruin for a dependent series.
  ``ruin_prob_worse = max(block, iid)`` is surfaced for the Phase-5 verdict.
- **Path-dependent max-drawdown distribution**: quantiles of each path's max
  drawdown, per resampler.
- **Seed sweep (S3)**: :func:`seed_sweep` runs the whole estimate across a fixed
  seed set (default 25 seeds) so ruin and a bootstrapped-DSR are reported as
  *distributions*; :meth:`SeedSweepResult.ruin_worst_quantile` /
  :meth:`SeedSweepResult.dsr_worst_quantile` expose the worst quantile the Phase-5
  verdict thresholds against (the mean is reported but never the gate).

Sentinel / raise convention (mirrors the rest of the harness): degenerate-but-valid
DATA never raises — it yields a documented sentinel (``None`` DSR / ``None`` ruin /
empty distribution) so a report prints ``null`` instead of crashing. Only invalid
*configuration* raises ``ValueError`` (non-positive trial count / variance /
bootstrap size, a threshold outside ``(0, 1]``, a non-int where an int is required).
Every result echoes its frozen config.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Final, Optional, Sequence, Union

__all__ = [
    "EULER_MASCHERONI",
    "DEFAULT_TRIAL_MULTIPLE",
    "DEFAULT_SEED_SET",
    "DEFAULT_RUIN_CONFIG",
    "RuinConfig",
    "DeflatedSharpeResult",
    "RuinResult",
    "SeedSweepResult",
    "per_period_sharpe",
    "sample_skewness",
    "sample_kurtosis",
    "punitive_trial_count",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "time_average_growth_rate",
    "estimate_ruin",
    "seed_sweep",
    "worst_quantile",
]

Number = Union[int, float]

# Euler–Mascheroni constant γ (False Strategy Theorem), to full double precision
# (0.5772156649015329 — the standard textbook value; neither ``math`` nor
# ``statistics`` exposes it as a constant, so it is stated literally here).
EULER_MASCHERONI: Final[float] = 0.5772156649015329

# Punitive multiple applied to the surviving-head count to get N_effective_trials
# (SSOT §3.3 / S1: "starting 5–10×"). 7 is the mid-band default; the caller may
# raise it. Whatever is used is logged in the result.
DEFAULT_TRIAL_MULTIPLE: Final[float] = 7.0

# The fixed seed set the sealed run sweeps (SSOT §3.6 / S3: "e.g. 25"). A plain
# ascending range so it is trivially reproducible and documented.
DEFAULT_SEED_SET: Final[tuple[int, ...]] = tuple(range(25))

# The single trusted standard-normal used everywhere (stdlib; deterministic).
_STD_NORMAL: Final[statistics.NormalDist] = statistics.NormalDist(0.0, 1.0)


# --------------------------------------------------------------------------- #
# Config / bool-rejecting coercion helpers (house style — reject bool as number).
# --------------------------------------------------------------------------- #
def _as_float(x: Number, name: str) -> float:
    """Coerce ``x`` to ``float``; reject ``bool`` and non-numbers (``TypeError``)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(x).__name__}: {x!r}")
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v


def _require_int(name: str, value: Any, *, minimum: int) -> int:
    """Require a plain ``int`` (reject ``bool``) ``>= minimum`` (``ValueError``)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _std_cdf(x: float) -> float:
    """Standard-normal CDF Φ(x) (stdlib NormalDist; total on the real line)."""
    return _STD_NORMAL.cdf(x)


def _std_ppf(p: float) -> float:
    """Standard-normal inverse CDF Φ⁻¹(p). Caller guarantees ``0 < p < 1``."""
    return _STD_NORMAL.inv_cdf(p)


# --------------------------------------------------------------------------- #
# Sample-moment helpers — method-of-moments (biased) estimators, matching the
# Bailey/López de Prado g1 skew and g2 raw (non-excess) kurtosis. Hand-rolled for
# transparent re-derivation and fixed summation order (same discipline as the
# scoring spine / regime labeler).
# --------------------------------------------------------------------------- #
def _central_moments(xs: Sequence[float]) -> tuple[float, float, float]:
    """Return ``(m2, m3, m4)`` — the 2nd/3rd/4th central moments dividing by n.

    ``m_k = (1/n) Σ (x − mean)**k``. Caller guarantees ``len(xs) >= 1``.
    """
    n = len(xs)
    mean = math.fsum(xs) / n
    dev = [x - mean for x in xs]
    m2 = math.fsum(d * d for d in dev) / n
    m3 = math.fsum(d * d * d for d in dev) / n
    m4 = math.fsum(d * d * d * d for d in dev) / n
    return (m2, m3, m4)


def per_period_sharpe(returns: Sequence[Number]) -> Optional[float]:
    """Per-period (non-annualized) Sharpe ratio = mean / sample stdev (ddof=1).

    Returns ``None`` (documented sentinel) for fewer than two returns or zero
    volatility — the inputs the PSR/DSR numerator cannot use.
    """
    rs = [_as_float(r, "return") for r in returns]
    n = len(rs)
    if n < 2 or max(rs) == min(rs):
        # max == min is exact zero dispersion (guards fsum rounding dust for a
        # constant series that would otherwise leave a ~1e-34 spurious variance).
        return None
    mean = math.fsum(rs) / n
    var = math.fsum((r - mean) ** 2 for r in rs) / (n - 1)
    if var <= 0.0:
        return None
    return mean / math.sqrt(var)


def sample_skewness(xs: Sequence[Number]) -> Optional[float]:
    """Method-of-moments skewness γ3 = m3 / m2**1.5 (``None`` if n<2 or zero var)."""
    vals = [_as_float(x, "value") for x in xs]
    if len(vals) < 2 or max(vals) == min(vals):
        return None
    m2, m3, _ = _central_moments(vals)
    if m2 <= 0.0:
        return None
    return m3 / (m2 ** 1.5)


def sample_kurtosis(xs: Sequence[Number]) -> Optional[float]:
    """Method-of-moments **raw** kurtosis γ4 = m4 / m2**2 (Normal → 3.0).

    Non-excess by design: the PSR/DSR denominator's ``(γ4 − 1)/4`` term expects the
    raw convention (Normal = 3). ``None`` if fewer than two values or zero variance.
    """
    vals = [_as_float(x, "value") for x in xs]
    if len(vals) < 2 or max(vals) == min(vals):
        return None
    m2, _, m4 = _central_moments(vals)
    if m2 <= 0.0:
        return None
    return m4 / (m2 * m2)


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DeflatedSharpeResult:
    """All Deflated-Sharpe outputs in one JSON-serialisable record (metrics only).

    ``deflated_sharpe_ratio`` / ``psr_vs_zero`` are ``None`` when the statistic is
    undefined (too few returns, or a non-positive PSR denominator under extreme
    skew×Sharpe). ``expected_max_sharpe`` is the SR*_0 deflation benchmark. Every
    input and the punitive-N provenance are echoed for the report. No verdict is
    rendered here — Phase 5 thresholds ``deflated_sharpe_ratio``.
    """

    observed_sharpe: float
    n_returns: int
    skew: float
    kurtosis: float
    n_effective_trials: int
    trials_sharpe_variance: Optional[float]  # None when undefined (n_returns < 2, defaulted)
    trials_variance_defaulted: bool
    expected_max_sharpe: Optional[float]
    psr_vs_zero: Optional[float]
    deflated_sharpe_ratio: Optional[float]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_sharpe": self.observed_sharpe,
            "n_returns": self.n_returns,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "n_effective_trials": self.n_effective_trials,
            "trials_sharpe_variance": self.trials_sharpe_variance,
            "trials_variance_defaulted": self.trials_variance_defaulted,
            "expected_max_sharpe": self.expected_max_sharpe,
            "psr_vs_zero": self.psr_vs_zero,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "warnings": list(self.warnings),
        }


def punitive_trial_count(
    surviving_heads: int, *, multiple: Number = DEFAULT_TRIAL_MULTIPLE
) -> int:
    """The punitive effective-trial count N (SSOT §3.3 / S1).

    ``N = max(surviving_heads, round(surviving_heads * multiple))`` — a documented
    multiple of the visible survivor count that is *never below* it (so a
    misconfigured ``multiple < 1`` cannot make the correction weaker than the
    survivor floor). ``multiple`` defaults to ``DEFAULT_TRIAL_MULTIPLE`` (7×, mid
    of the SSOT 5–10× band). Raises ``ValueError`` on a non-positive survivor count
    or a non-positive multiple (invalid config).
    """
    heads = _require_int("surviving_heads", surviving_heads, minimum=1)
    m = _as_float(multiple, "multiple")
    if m <= 0.0:
        raise ValueError(f"multiple must be > 0, got {m}")
    scaled = int(round(heads * m))
    return max(heads, scaled)


def expected_max_sharpe(
    n_trials: int, *, trials_sharpe_variance: Number
) -> float:
    """Expected maximum Sharpe of ``n_trials`` zero-skill trials (SR*_0).

    The False Strategy Theorem benchmark (see module docstring §1). ``n_trials``
    must be ``>= 1`` and ``trials_sharpe_variance`` (V) must be ``> 0`` — both are
    config, so invalid values raise ``ValueError``. ``n_trials == 1`` is the
    single-trial case with no selection bias, so E[max of one draw] = 0.0 exactly
    (the closed form's Φ⁻¹(1 − 1/N) → Φ⁻¹(0) = −∞ is avoided by this special case).
    """
    n = _require_int("n_trials", n_trials, minimum=1)
    v = _as_float(trials_sharpe_variance, "trials_sharpe_variance")
    if v <= 0.0:
        raise ValueError(f"trials_sharpe_variance must be > 0, got {v}")
    if n == 1:
        return 0.0
    # Both quantile arguments lie strictly in (0, 1) for n >= 2:
    #   1 - 1/n      ∈ [0.5, 1)      and   1 - 1/(n·e) ∈ (0.5, 1).
    z1 = _std_ppf(1.0 - 1.0 / n)
    z2 = _std_ppf(1.0 - 1.0 / (n * math.e))
    bracket = (1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2
    return math.sqrt(v) * bracket


def probabilistic_sharpe_ratio(
    observed_sharpe: Number,
    n_returns: int,
    skew: Number,
    kurtosis: Number,
    *,
    benchmark_sharpe: Number = 0.0,
) -> Optional[float]:
    """Probabilistic Sharpe Ratio PSR(benchmark) (see module docstring §1).

    ``observed_sharpe`` is the **per-period, non-annualized** Sharpe; ``kurtosis``
    is **raw** (Normal = 3). Returns ``None`` (sentinel) when the statistic is
    undefined: fewer than two returns, or a non-positive variance term
    ``1 − skew·SR + (kurt − 1)/4·SR²`` (possible only under extreme skew×Sharpe).
    ``n_returns`` below 2 is degenerate DATA, not invalid config, so it does not
    raise.
    """
    sr = _as_float(observed_sharpe, "observed_sharpe")
    sk = _as_float(skew, "skew")
    ku = _as_float(kurtosis, "kurtosis")
    srb = _as_float(benchmark_sharpe, "benchmark_sharpe")
    n = _require_int("n_returns", n_returns, minimum=0)
    if n < 2:
        return None
    var_term = 1.0 - sk * sr + (ku - 1.0) / 4.0 * sr * sr
    if var_term <= 0.0:
        return None
    z = (sr - srb) * math.sqrt(n - 1) / math.sqrt(var_term)
    return _std_cdf(z)


def deflated_sharpe_ratio(
    observed_sharpe: Number,
    n_returns: int,
    skew: Number,
    kurtosis: Number,
    n_effective_trials: int,
    *,
    trials_sharpe_variance: Optional[Number] = None,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio = PSR(SR*_0) with a punitive N (SSOT §3.3 / S1).

    Parameters
    ----------
    observed_sharpe:
        **Per-period, non-annualized** Sharpe ratio (``SR_annual / sqrt(ppy)``).
    n_returns:
        Number of return observations behind the Sharpe (< 2 → sentinel result).
    skew, kurtosis:
        Non-excess skewness γ3 and **raw** kurtosis γ4 (Normal = 3).
    n_effective_trials:
        The punitive trial count N (see :func:`punitive_trial_count`); **required**
        and echoed in the result. Must be ``>= 1``.
    trials_sharpe_variance:
        Cross-trial Sharpe variance V. ``None`` (default) uses the null
        sampling-variance floor ``1/(n_returns − 1)`` and flags
        ``trials_variance_defaulted = True`` (see module docstring §1); an explicit
        positive value overrides it.

    Returns a :class:`DeflatedSharpeResult` (metrics only; no verdict). Degenerate
    DATA (``n_returns < 2``) yields ``None`` DSR/PSR with the reason in ``warnings``;
    invalid CONFIG (``n_effective_trials < 1``, explicit ``V <= 0``) raises.
    """
    sr = _as_float(observed_sharpe, "observed_sharpe")
    sk = _as_float(skew, "skew")
    ku = _as_float(kurtosis, "kurtosis")
    n = _require_int("n_returns", n_returns, minimum=0)
    n_trials = _require_int("n_effective_trials", n_effective_trials, minimum=1)

    warnings: list[str] = []

    # --- resolve V (explicit override vs documented null-floor default) ------ #
    v: Optional[float]
    if trials_sharpe_variance is None:
        if n < 2:
            # No floor is definable without >= 2 returns; keep V as the None
            # sentinel (honouring the "undefined → null" contract — not NaN, which
            # is non-standard JSON) and take the degenerate-data return below.
            v = None
            defaulted = True
        else:
            v = 1.0 / (n - 1)
            defaulted = True
    else:
        v = _as_float(trials_sharpe_variance, "trials_sharpe_variance")
        if v <= 0.0:
            raise ValueError(f"trials_sharpe_variance must be > 0, got {v}")
        defaulted = False

    if n < 2:
        warnings.append(
            f"n_returns={n} < 2: Sharpe statistic undefined; DSR/PSR are None sentinels"
        )
        return DeflatedSharpeResult(
            observed_sharpe=sr,
            n_returns=n,
            skew=sk,
            kurtosis=ku,
            n_effective_trials=n_trials,
            trials_sharpe_variance=v,
            trials_variance_defaulted=defaulted,
            expected_max_sharpe=None,
            psr_vs_zero=None,
            deflated_sharpe_ratio=None,
            warnings=tuple(warnings),
        )

    srstar = expected_max_sharpe(n_trials, trials_sharpe_variance=v)
    psr0 = probabilistic_sharpe_ratio(sr, n, sk, ku, benchmark_sharpe=0.0)
    dsr = probabilistic_sharpe_ratio(sr, n, sk, ku, benchmark_sharpe=srstar)
    if dsr is None:
        warnings.append(
            "PSR variance term <= 0 (extreme skew×Sharpe): DSR is a None sentinel"
        )
    if defaulted:
        warnings.append(
            "trials_sharpe_variance defaulted to null floor 1/(n-1); pass the "
            "measured cross-trial variance for a stricter (larger-V) deflation"
        )

    return DeflatedSharpeResult(
        observed_sharpe=sr,
        n_returns=n,
        skew=sk,
        kurtosis=ku,
        n_effective_trials=n_trials,
        trials_sharpe_variance=v,
        trials_variance_defaulted=defaulted,
        expected_max_sharpe=srstar,
        psr_vs_zero=psr0,
        deflated_sharpe_ratio=dsr,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Ergodicity / ruin — time-average growth, drawdown, block-bootstrap ruin.
# --------------------------------------------------------------------------- #
def time_average_growth_rate(returns: Sequence[Number]) -> Optional[float]:
    """Per-period time-average (ergodic) compound growth rate.

    ``exp( mean( ln(1 + r_t) ) ) − 1`` — the rate a single bankroll compounds at
    through time (SSOT §3.6). Returns ``None`` (sentinel) for an empty series or if
    any ``1 + r_t <= 0`` (a period that wipes the bankroll makes the log-growth
    ``−inf``; reported as undefined rather than a fabricated number).
    """
    rs = [_as_float(r, "return") for r in returns]
    if not rs:
        return None
    log_sum = 0.0
    for r in rs:
        gross = 1.0 + r
        if gross <= 0.0:
            return None
        log_sum += math.log(gross)
    return math.exp(log_sum / len(rs)) - 1.0


def _max_drawdown_from_returns(returns: Sequence[float]) -> float:
    """Largest peak-to-trough decline as a positive magnitude in ``[0, 1]``.

    Equity starts at 1.0 and compounds; drawdown is ``1 − E_t / running_peak``. The
    running peak starts at (and never drops below) 1.0, so a bankrupt path — any
    period return ``<= -1`` drives equity ``<= 0`` — would give ``1 − E_t/peak > 1``;
    that is clamped to ``1.0`` (a single bankroll cannot lose more than 100%: full
    ruin). Matches the scoring-spine convention (re-implemented locally to keep this
    module self-contained and free of intra-package coupling, per house style),
    with the extra full-ruin clamp so the reported magnitude stays a valid fraction.
    """
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for r in returns:
        equity *= 1.0 + r
        if equity > peak:
            peak = equity
        dd = 1.0 - equity / peak  # peak >= 1.0 > 0 always
        if dd > 1.0:
            dd = 1.0  # equity crossed <= 0 → full ruin, capped at 100%
        if dd > worst:
            worst = dd
    return worst


def _lag1_autocorrelation(returns: Sequence[float]) -> Optional[float]:
    """Lag-1 sample autocorrelation of ``returns`` (``None`` if n<2 or zero var)."""
    n = len(returns)
    if n < 2:
        return None
    mean = math.fsum(returns) / n
    dev = [r - mean for r in returns]
    denom = math.fsum(d * d for d in dev)
    if denom <= 0.0:
        return None
    numer = math.fsum(dev[i] * dev[i + 1] for i in range(n - 1))
    return numer / denom


def _estimate_block_length(returns: Sequence[float]) -> int:
    """Mean stationary-bootstrap block length, tied to observed autocorrelation.

    Only *positive* lag-1 autocorrelation ρ⁺ = max(0, ρ1) lengthens blocks (that is
    the loss-clustering / persistence we must preserve). The rule uses the
    integrated-autocorrelation time of an AR(1) with coefficient ρ⁺:

        L = round( (1 + ρ⁺) / (1 − ρ⁺) ),   clamped to [1, max(1, n // 2)].

    ρ1 ≤ 0 (IID or mean-reverting) → L = 1, i.e. near-IID resampling. A short/flat
    series → L = 1. Documented and unit-tested directly.
    """
    n = len(returns)
    rho = _lag1_autocorrelation(returns)
    if rho is None or rho <= 0.0:
        return 1
    rho = min(rho, 0.999)  # guard the division as ρ → 1
    raw = (1.0 + rho) / (1.0 - rho)
    length = int(round(raw))
    upper = max(1, n // 2)
    return max(1, min(length, upper))


def _stationary_bootstrap_path(
    rng: random.Random,
    returns: Sequence[float],
    length: int,
    mean_block_length: int,
) -> list[float]:
    """One Politis–Romano stationary-bootstrap path of ``length`` returns.

    Circular blocks of geometric length (mean ``mean_block_length``): each step
    either continues the current block (prob ``1 − 1/L``) or jumps to a fresh
    uniform random start (prob ``1/L``), wrapping around the series. ``L = 1``
    reduces exactly to the IID bootstrap. Deterministic given ``rng``.
    """
    n = len(returns)
    p_new = 1.0 / mean_block_length
    out: list[float] = []
    idx = rng.randrange(n)
    for _ in range(length):
        if rng.random() < p_new:
            idx = rng.randrange(n)
        out.append(returns[idx])
        idx = (idx + 1) % n
    return out


def _iid_bootstrap_path(
    rng: random.Random, returns: Sequence[float], length: int
) -> list[float]:
    """One IID-bootstrap path: ``length`` returns drawn uniformly with replacement."""
    n = len(returns)
    return [returns[rng.randrange(n)] for _ in range(length)]


def _quantiles(sorted_vals: Sequence[float], qs: Sequence[float]) -> dict[str, float]:
    """Linear-interpolated (type-7) quantiles of an ascending ``sorted_vals``.

    Keyed by the quantile as a stable ``"q{percent}"`` string. Caller guarantees a
    non-empty, ascending sequence.
    """
    m = len(sorted_vals)
    out: dict[str, float] = {}
    for q in qs:
        if m == 1:
            out[_q_key(q)] = sorted_vals[0]
            continue
        pos = q * (m - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            out[_q_key(q)] = sorted_vals[lo]
        else:
            frac = pos - lo
            out[_q_key(q)] = sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
    return out


def _q_key(q: float) -> str:
    """Stable dict key for a quantile, e.g. 0.95 → ``'q95'``, 0.999 → ``'q99.9'``."""
    pct = q * 100.0
    if pct == int(pct):
        return f"q{int(pct)}"
    return f"q{pct:g}"


@dataclass(frozen=True)
class RuinConfig:
    """Frozen, echoed configuration for the ruin/ergodicity bootstrap (SSOT §3.6).

    ``ruin_drawdown_threshold`` is the peak-to-trough drawdown (a fraction in
    ``(0, 1]``) that counts a path as "ruined"; it is a metric input, NOT the
    Phase-5 verdict bound. ``path_length`` is the horizon of each synthetic path
    (``None`` → the length of the empirical series). ``mean_block_length`` overrides
    the autocorrelation-derived block length (``None`` → estimated). ``bootstrap_paths``
    is the number of synthetic paths per resampler per seed.
    """

    bootstrap_paths: int = 2000
    ruin_drawdown_threshold: float = 0.5
    path_length: Optional[int] = None
    mean_block_length: Optional[int] = None
    drawdown_quantiles: tuple[float, ...] = (0.5, 0.95, 0.99)

    def __post_init__(self) -> None:
        _require_int("bootstrap_paths", self.bootstrap_paths, minimum=1)
        if isinstance(self.ruin_drawdown_threshold, bool) or not isinstance(
            self.ruin_drawdown_threshold, (int, float)
        ):
            raise ValueError(
                f"ruin_drawdown_threshold must be a real number, got "
                f"{self.ruin_drawdown_threshold!r}"
            )
        if not (0.0 < float(self.ruin_drawdown_threshold) <= 1.0):
            raise ValueError(
                f"ruin_drawdown_threshold must be in (0, 1], got "
                f"{self.ruin_drawdown_threshold}"
            )
        if self.path_length is not None:
            _require_int("path_length", self.path_length, minimum=1)
        if self.mean_block_length is not None:
            _require_int("mean_block_length", self.mean_block_length, minimum=1)
        for q in self.drawdown_quantiles:
            if isinstance(q, bool) or not isinstance(q, (int, float)):
                raise ValueError(f"drawdown_quantiles must be reals, got {q!r}")
            if not (0.0 <= float(q) <= 1.0):
                raise ValueError(f"each drawdown quantile must be in [0, 1], got {q}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bootstrap_paths": self.bootstrap_paths,
            "ruin_drawdown_threshold": self.ruin_drawdown_threshold,
            "path_length": self.path_length,
            "mean_block_length": self.mean_block_length,
            "drawdown_quantiles": list(self.drawdown_quantiles),
        }


DEFAULT_RUIN_CONFIG: Final[RuinConfig] = RuinConfig()


@dataclass(frozen=True)
class RuinResult:
    """Ruin / drawdown / growth metrics for ONE seed (metrics only; no verdict).

    ``ruin_prob_block`` uses the loss-clustering-preserving stationary bootstrap;
    ``ruin_prob_iid`` the clustering-destroying IID bootstrap; ``ruin_prob_worse``
    is ``max`` of the two (SSOT: trust the worse number). Drawdown-quantile dicts
    are keyed ``"q50"``/``"q95"``/… . All are ``None``/empty for degenerate data
    (fewer than two returns or zero dispersion). ``time_average_growth`` is a
    deterministic property of the empirical series (not bootstrapped).
    """

    seed: int
    n_returns: int
    time_average_growth: Optional[float]
    lag1_autocorrelation: Optional[float]
    mean_block_length: Optional[int]
    ruin_prob_block: Optional[float]
    ruin_prob_iid: Optional[float]
    ruin_prob_worse: Optional[float]
    max_drawdown_block: dict[str, float]
    max_drawdown_iid: dict[str, float]
    config_echo: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_returns": self.n_returns,
            "time_average_growth": self.time_average_growth,
            "lag1_autocorrelation": self.lag1_autocorrelation,
            "mean_block_length": self.mean_block_length,
            "ruin_prob_block": self.ruin_prob_block,
            "ruin_prob_iid": self.ruin_prob_iid,
            "ruin_prob_worse": self.ruin_prob_worse,
            "max_drawdown_block": dict(self.max_drawdown_block),
            "max_drawdown_iid": dict(self.max_drawdown_iid),
            "config_echo": self.config_echo,
            "warnings": list(self.warnings),
        }


def estimate_ruin(
    returns: Sequence[Number],
    *,
    seed: int,
    config: RuinConfig = DEFAULT_RUIN_CONFIG,
) -> RuinResult:
    """Estimate ruin/drawdown/growth for one seed via block + IID bootstrap.

    Runs ``config.bootstrap_paths`` synthetic paths under BOTH the stationary block
    bootstrap (preserves loss-clustering) and the IID bootstrap (destroys it), and
    reports the ruin probability and max-drawdown quantiles for each plus their
    worse (SSOT §3.6). Deterministic given ``seed`` + ``config``.

    Degenerate DATA (fewer than two returns, or zero dispersion so every bar is the
    same value) never raises: it returns ``None`` ruin probabilities and empty
    drawdown dicts with the reason in ``warnings``. Invalid CONFIG raised earlier by
    :class:`RuinConfig`. ``seed`` must be a non-negative int.
    """
    seed_i = _require_int("seed", seed, minimum=0)
    if not isinstance(config, RuinConfig):
        raise ValueError(f"config must be a RuinConfig, got {type(config).__name__}")

    rs = [_as_float(r, "return") for r in returns]
    n = len(rs)
    warnings: list[str] = []

    growth = time_average_growth_rate(rs) if n >= 1 else None
    rho = _lag1_autocorrelation(rs)

    # Degenerate data → honest sentinels (no bootstrap is meaningful).
    distinct = len(set(rs))
    if n < 2 or distinct < 2:
        if n < 2:
            warnings.append(f"n_returns={n} < 2: ruin bootstrap undefined (sentinel)")
        else:
            warnings.append(
                "returns have zero dispersion (all identical): ruin bootstrap undefined"
            )
        return RuinResult(
            seed=seed_i,
            n_returns=n,
            time_average_growth=growth,
            lag1_autocorrelation=rho,
            mean_block_length=None,
            ruin_prob_block=None,
            ruin_prob_iid=None,
            ruin_prob_worse=None,
            max_drawdown_block={},
            max_drawdown_iid={},
            config_echo=config.to_dict(),
            warnings=tuple(warnings),
        )

    length = config.path_length if config.path_length is not None else n
    block_len = (
        config.mean_block_length
        if config.mean_block_length is not None
        else _estimate_block_length(rs)
    )
    threshold = float(config.ruin_drawdown_threshold)
    paths = config.bootstrap_paths

    # Independent RNG streams for the two resamplers so neither perturbs the other's
    # draw sequence; both are derived deterministically from the single seed.
    rng_block = random.Random(f"{seed_i}:block")
    rng_iid = random.Random(f"{seed_i}:iid")

    block_dds: list[float] = []
    iid_dds: list[float] = []
    ruin_block = 0
    ruin_iid = 0
    for _ in range(paths):
        bpath = _stationary_bootstrap_path(rng_block, rs, length, block_len)
        dd_b = _max_drawdown_from_returns(bpath)
        block_dds.append(dd_b)
        if dd_b >= threshold:
            ruin_block += 1

        ipath = _iid_bootstrap_path(rng_iid, rs, length)
        dd_i = _max_drawdown_from_returns(ipath)
        iid_dds.append(dd_i)
        if dd_i >= threshold:
            ruin_iid += 1

    p_block = ruin_block / paths
    p_iid = ruin_iid / paths
    block_dds.sort()
    iid_dds.sort()

    return RuinResult(
        seed=seed_i,
        n_returns=n,
        time_average_growth=growth,
        lag1_autocorrelation=rho,
        mean_block_length=block_len,
        ruin_prob_block=p_block,
        ruin_prob_iid=p_iid,
        ruin_prob_worse=max(p_block, p_iid),
        max_drawdown_block=_quantiles(block_dds, config.drawdown_quantiles),
        max_drawdown_iid=_quantiles(iid_dds, config.drawdown_quantiles),
        config_echo=config.to_dict(),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Seed sweep — ruin + bootstrapped-DSR as distributions, worst-quantile access.
# --------------------------------------------------------------------------- #
def worst_quantile(
    values: Sequence[float], q: float, *, lower_is_worse: bool
) -> Optional[float]:
    """The worst-``q``-tail quantile of ``values`` (metrics helper; no verdict).

    ``q`` is the ADVERSE-tail probability. ``lower_is_worse=True`` (e.g. DSR — a
    small value is the adverse case) returns the lower ``q`` quantile;
    ``lower_is_worse=False`` (e.g. ruin probability — a large value is adverse)
    returns the upper ``(1 − q)`` quantile. So ``q=0.05`` is the 5%-worst outcome in
    either direction. ``q`` in ``[0, 1]`` (raises otherwise — invalid config).
    Returns ``None`` for an empty sequence (documented sentinel). Uses the same
    type-7 interpolation as the drawdown quantiles for consistency.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise ValueError(f"q must be a real number, got {q!r}")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError(f"q must be in [0, 1], got {q}")
    finite = sorted(float(v) for v in values)
    if not finite:
        return None
    target = float(q) if lower_is_worse else 1.0 - float(q)
    return _quantiles(finite, (target,))[_q_key(target)]


@dataclass(frozen=True)
class SeedSweepResult:
    """Seed-sweep distributions of ruin and bootstrapped DSR (SSOT §3.6 / S3).

    ``ruin_results`` holds one :class:`RuinResult` per seed. ``dsr_distribution`` is
    the DSR recomputed on each seed's block-bootstrap resample of the returns (same
    punitive N), so the DSR is reported as a distribution rather than a point; it is
    empty when the DSR was undefined on every resample. The worst-quantile accessors
    are what a Phase-5 verdict thresholds against (mean is reported, never the gate).
    """

    seeds: tuple[int, ...]
    ruin_results: tuple[RuinResult, ...]
    ruin_worse_distribution: tuple[float, ...]
    dsr_distribution: tuple[float, ...]
    n_effective_trials: int
    config_echo: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def ruin_worst_quantile(self, tail: float = 0.05) -> Optional[float]:
        """Worst-``tail`` ruin: the UPPER ``(1 − tail)`` quantile of per-seed worse-ruin
        (higher ruin is adverse). ``tail=0.05`` → the 95th-percentile ruin."""
        return worst_quantile(self.ruin_worse_distribution, tail, lower_is_worse=False)

    def dsr_worst_quantile(self, tail: float = 0.05) -> Optional[float]:
        """Worst-``tail`` DSR: the LOWER ``tail`` quantile of the bootstrapped DSR
        (lower DSR is adverse). ``tail=0.05`` → the 5th-percentile DSR."""
        return worst_quantile(self.dsr_distribution, tail, lower_is_worse=True)

    def mean_ruin_worse(self) -> Optional[float]:
        """Mean of the per-seed worse-ruin estimates (reported, never the gate)."""
        if not self.ruin_worse_distribution:
            return None
        return math.fsum(self.ruin_worse_distribution) / len(self.ruin_worse_distribution)

    def mean_dsr(self) -> Optional[float]:
        """Mean of the bootstrapped DSR distribution (reported, never the gate)."""
        if not self.dsr_distribution:
            return None
        return math.fsum(self.dsr_distribution) / len(self.dsr_distribution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "ruin_results": [r.to_dict() for r in self.ruin_results],
            "ruin_worse_distribution": list(self.ruin_worse_distribution),
            "dsr_distribution": list(self.dsr_distribution),
            "n_effective_trials": self.n_effective_trials,
            "ruin_worst_quantile_tail05": self.ruin_worst_quantile(0.05),
            "dsr_worst_quantile_tail05": self.dsr_worst_quantile(0.05),
            "mean_ruin_worse": self.mean_ruin_worse(),
            "mean_dsr": self.mean_dsr(),
            "config_echo": self.config_echo,
            "warnings": list(self.warnings),
        }


def seed_sweep(
    returns: Sequence[Number],
    *,
    n_effective_trials: int,
    seeds: Sequence[int] = DEFAULT_SEED_SET,
    config: RuinConfig = DEFAULT_RUIN_CONFIG,
    trials_sharpe_variance: Optional[Number] = None,
) -> SeedSweepResult:
    """Sweep the ruin bootstrap + a bootstrapped DSR across a fixed seed set (S3).

    For each seed: run :func:`estimate_ruin` (collecting its worse-ruin estimate)
    and, on the same seed's block-bootstrap resample of ``returns``, recompute
    (per-period Sharpe, skew, raw kurtosis) and the Deflated Sharpe Ratio with the
    supplied punitive ``n_effective_trials`` — yielding a DSR *distribution*. Both
    distributions are exposed through worst-quantile accessors (SSOT: thresholds
    apply to the worst quantile, not the mean). Deterministic given the seed set +
    config.

    ``seeds`` must be a non-empty sequence of distinct non-negative ints (default
    :data:`DEFAULT_SEED_SET`, 25 seeds). Degenerate DATA propagates as ``None``
    per-seed ruin and an empty DSR distribution (never raises); invalid CONFIG
    raises (empty/duplicate/negative seeds, ``n_effective_trials < 1``).
    """
    n_trials = _require_int("n_effective_trials", n_effective_trials, minimum=1)
    if not isinstance(config, RuinConfig):
        raise ValueError(f"config must be a RuinConfig, got {type(config).__name__}")
    seed_list = [_require_int("seed", s, minimum=0) for s in seeds]
    if not seed_list:
        raise ValueError("seeds must be a non-empty sequence")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError(f"seeds must be distinct, got {seed_list}")

    rs = [_as_float(r, "return") for r in returns]
    n = len(rs)
    length = config.path_length if config.path_length is not None else n
    warnings: list[str] = []

    block_len: Optional[int] = None
    if n >= 2 and len(set(rs)) >= 2:
        block_len = (
            config.mean_block_length
            if config.mean_block_length is not None
            else _estimate_block_length(rs)
        )

    ruin_results: list[RuinResult] = []
    ruin_worse: list[float] = []
    dsr_values: list[float] = []
    undefined_dsr = 0
    for s in seed_list:
        rr = estimate_ruin(rs, seed=s, config=config)
        ruin_results.append(rr)
        if rr.ruin_prob_worse is not None:
            ruin_worse.append(rr.ruin_prob_worse)

        # Bootstrapped DSR: resample the returns (block bootstrap, so the dependence
        # structure carries into the moment estimates) and recompute the DSR.
        if block_len is not None:
            rng = random.Random(f"{s}:dsr")
            resample = _stationary_bootstrap_path(rng, rs, length, block_len)
            sr = per_period_sharpe(resample)
            sk = sample_skewness(resample)
            ku = sample_kurtosis(resample)
            if sr is not None and sk is not None and ku is not None:
                res = deflated_sharpe_ratio(
                    sr,
                    len(resample),
                    sk,
                    ku,
                    n_trials,
                    trials_sharpe_variance=trials_sharpe_variance,
                )
                if res.deflated_sharpe_ratio is not None:
                    dsr_values.append(res.deflated_sharpe_ratio)
                else:
                    undefined_dsr += 1
            else:
                undefined_dsr += 1

    if block_len is None:
        warnings.append("degenerate returns: no bootstrapped DSR distribution")
    elif undefined_dsr:
        warnings.append(
            f"{undefined_dsr}/{len(seed_list)} resamples produced an undefined DSR "
            "(excluded from the distribution)"
        )

    return SeedSweepResult(
        seeds=tuple(seed_list),
        ruin_results=tuple(ruin_results),
        ruin_worse_distribution=tuple(ruin_worse),
        dsr_distribution=tuple(dsr_values),
        n_effective_trials=n_trials,
        config_echo=config.to_dict(),
        warnings=tuple(warnings),
    )

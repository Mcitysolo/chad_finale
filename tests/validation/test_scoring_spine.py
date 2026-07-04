"""Known-answer + edge tests for chad.validation.scoring_spine (Phase 1).

Every numeric assertion is checked against a reference value derived **by hand**
in the test's docstring — no expected value is produced by re-running the module
under test. Tolerances use ``pytest.approx`` because the module uses IEEE-754
floating point (e.g. ``sqrt(252)``), so bit-exact literals are neither meaningful
nor portable; exactness is asserted only where the arithmetic is genuinely exact
(indices, counts, all-flat zeros).

Fixtures are tiny in-memory sequences; nothing here touches the real bar corpus,
the network, or any runtime state.
"""

from __future__ import annotations

import json
import math

import pytest

from chad.validation.scoring_spine import (
    DEFAULT_PERIODS_PER_YEAR,
    ScoreResult,
    equity_to_returns,
    score_equity,
    score_returns,
    score_trades,
)

SQRT_252 = math.sqrt(252)


# --------------------------------------------------------------------------- #
# 1. Total return.
# --------------------------------------------------------------------------- #
def test_total_return_known_answer():
    """returns = [0.10, -0.05, 0.20].

    total_return = (1.10)(0.95)(1.20) - 1 = 1.254 - 1 = 0.254.
    """
    r = score_returns([0.10, -0.05, 0.20])
    assert r.total_return == pytest.approx(0.254, rel=1e-12)
    assert r.n_periods == 3
    assert r.kind == "returns"


# --------------------------------------------------------------------------- #
# 2. CAGR — derived horizon and explicit horizon.
# --------------------------------------------------------------------------- #
def test_cagr_derived_horizon_known_answer():
    """returns = [0.20, 0.20, 0.20], periods_per_year = 1 (annual bars).

    final equity = 1.2^3 = 1.728, years = n/ppy = 3/1 = 3.
    CAGR = 1.728^(1/3) - 1 = 1.2 - 1 = 0.20.
    """
    r = score_returns([0.20, 0.20, 0.20], periods_per_year=1)
    assert r.years == pytest.approx(3.0)
    assert r.total_return == pytest.approx(0.728, rel=1e-12)
    assert r.cagr == pytest.approx(0.20, rel=1e-9)


def test_cagr_explicit_horizon_known_answer():
    """A single +100% period with an explicit years=2 horizon.

    final equity = 2.0, CAGR = 2^(1/2) - 1 = 0.41421356237...
    Also confirms the explicit ``years`` override path (vs derived).
    """
    r = score_returns([1.0], years=2.0, periods_per_year=1)
    assert r.total_return == pytest.approx(1.0)
    assert r.cagr == pytest.approx(math.sqrt(2.0) - 1.0, rel=1e-12)
    # Single element: risk metrics need n>=2, so they are sentinels.
    assert r.volatility is None
    assert r.sharpe is None
    assert r.sortino is None


def test_cagr_total_wipeout_is_minus_one():
    """returns = [-1.0, 0.5]: equity hits exactly 0 then can't recover.

    final equity = 1*(1-1.0)*(1+0.5) = 0.0 → CAGR annualizes to -100% = -1.0.
    """
    r = score_returns([-1.0, 0.5])
    assert r.total_return == pytest.approx(-1.0)
    assert r.cagr == -1.0


def test_cagr_negative_equity_is_none():
    """returns = [-1.5]: a >100% single-period loss drives equity below 0.

    final equity = 1*(1-1.5) = -0.5 < 0 → real fractional root undefined → None.
    """
    r = score_returns([-1.5])
    assert r.total_return == pytest.approx(-1.5)
    assert r.cagr is None


def test_cagr_overflow_returns_none_not_raise():
    """A huge gain over a tiny derived horizon overflows the annualized factor.

    score_returns([1000.0]) with ppy=252 → horizon 1/252 → exponent 252 →
    1001.0 ** 252 ≈ 10^756 > float64 max. The 'degenerate data never raises'
    contract requires this to collapse to the None sentinel, not raise
    OverflowError. total_return stays finite and correct (+100000%).
    """
    r = score_returns([1000.0])
    assert r.cagr is None
    assert r.total_return == pytest.approx(1000.0)
    # Large periods_per_year collapses the horizon so even a modest return overflows.
    r2 = score_returns([0.5], periods_per_year=100000)
    assert r2.cagr is None


# --------------------------------------------------------------------------- #
# 3. Volatility (annualized sample stdev).
# --------------------------------------------------------------------------- #
def test_volatility_known_answer():
    """returns = [0.01, 0.02, -0.01, 0.03, 0.00].

    mean = 0.05/5 = 0.01.
    sum of squared deviations = 0 + 0.0001 + 0.0004 + 0.0004 + 0.0001 = 0.001.
    sample variance = 0.001/(5-1) = 0.00025, sample stdev = sqrt(0.00025).
    annualized volatility = sqrt(0.00025) * sqrt(252) = 0.250998007...
    """
    r = score_returns([0.01, 0.02, -0.01, 0.03, 0.00])
    assert r.volatility == pytest.approx(math.sqrt(0.00025) * SQRT_252, rel=1e-12)


# --------------------------------------------------------------------------- #
# 4. Sharpe ratio (annualized).
# --------------------------------------------------------------------------- #
def test_sharpe_known_answer():
    """returns = [0.01, 0.02, -0.01, 0.03, 0.00], rf = 0, ppy = 252.

    mean excess = 0.01, sample stdev = sqrt(0.00025) = 0.0158113883...
    Sharpe (periodic) = 0.01 / 0.0158113883 = 0.6324555320...
    Sharpe (annual)   = 0.6324555320 * sqrt(252) = 10.0399203184...
    """
    r = score_returns([0.01, 0.02, -0.01, 0.03, 0.00])
    expected = (0.01 / math.sqrt(0.00025)) * SQRT_252
    assert r.sharpe == pytest.approx(expected, rel=1e-12)
    assert r.sharpe == pytest.approx(10.039920318408907, rel=1e-12)


def test_sharpe_with_risk_free_rate():
    """Same returns, but rf_annual = 2.52 → per-period rf = 2.52/252 = 0.01.

    excess returns = [0.00, 0.01, -0.02, 0.02, -0.01], mean excess = 0.0.
    mean excess = 0 → Sharpe periodic = 0 → annual Sharpe = 0.0 (a real zero,
    not a sentinel: the stdev denominator is non-zero).
    """
    r = score_returns([0.01, 0.02, -0.01, 0.03, 0.00], risk_free_rate=2.52)
    assert r.sharpe == pytest.approx(0.0, abs=1e-12)
    assert r.sharpe is not None


def test_sharpe_zero_variance_is_none():
    """Constant non-zero returns have zero sample stdev → Sharpe undefined → None."""
    r = score_returns([0.01, 0.01, 0.01])
    assert r.volatility == pytest.approx(0.0, abs=1e-15)
    assert r.sharpe is None  # 0.01 / 0 would be +inf; sentinel instead.


# --------------------------------------------------------------------------- #
# 5. Sortino ratio (annualized, target downside deviation).
# --------------------------------------------------------------------------- #
def test_sortino_mixed_returns_known_answer():
    """returns = [0.01, -0.02, 0.03, -0.01], rf = 0, ppy = 252.

    mean excess = 0.01/4 = 0.0025.
    downside diffs min(0, r) = [0, -0.02, 0, -0.01]; squares = [0, 0.0004, 0, 0.0001].
    downside deviation = sqrt((0.0004 + 0.0001)/4) = sqrt(0.000125) = 0.01118033988...
      (denominator is the FULL count n=4, the standard target-downside-deviation).
    Sortino periodic = 0.0025 / 0.01118033988 = 0.2236067977...
    Sortino annual   = 0.2236067977 * sqrt(252) = 3.5496478698...
    """
    r = score_returns([0.01, -0.02, 0.03, -0.01])
    dd = math.sqrt(0.000125)
    expected = (0.0025 / dd) * SQRT_252
    assert r.sortino == pytest.approx(expected, rel=1e-12)
    assert r.sortino == pytest.approx(3.5496478698597693, rel=1e-12)


def test_sortino_no_downside_is_none():
    """All returns >= target (rf=0) → zero downside deviation → Sortino None."""
    r = score_returns([0.01, 0.02, 0.00])
    assert r.sortino is None  # no downside → undefined denominator.


def test_sortino_all_negative_is_finite_negative():
    """returns = [-0.01, -0.02]: fully downside, so Sortino is a real negative.

    mean excess = -0.015. downside dev = sqrt((0.0001+0.0004)/2) = sqrt(0.00025).
    Sortino periodic = -0.015/sqrt(0.00025) = -0.9486832980..., annual *sqrt(252).
    """
    r = score_returns([-0.01, -0.02])
    expected = (-0.015 / math.sqrt(0.00025)) * SQRT_252
    assert r.sortino == pytest.approx(expected, rel=1e-12)
    assert r.sortino < 0


# --------------------------------------------------------------------------- #
# 6. Max drawdown (magnitude + peak/trough indices).
# --------------------------------------------------------------------------- #
def test_max_drawdown_known_answer():
    """returns = [0.10, -0.20, 0.05].

    equity curve = [1.00, 1.10, 0.88, 0.924] (index 0 is the start point).
    running peak reaches 1.10 at index 1; trough 0.88 at index 2.
    drawdown = 0.88/1.10 - 1 = -0.20 → magnitude 0.20, peak index 1, trough index 2.
    """
    r = score_returns([0.10, -0.20, 0.05])
    assert r.max_drawdown == pytest.approx(0.20, rel=1e-12)
    assert r.max_drawdown_peak_index == 1
    assert r.max_drawdown_trough_index == 2


def test_max_drawdown_monotonic_up_is_zero():
    """A strictly rising curve has no decline → magnitude exactly +0.0 (not -0.0)."""
    r = score_returns([0.01, 0.02, 0.03])
    assert r.max_drawdown == 0.0
    assert math.copysign(1.0, r.max_drawdown) == 1.0  # positive zero, not -0.0
    assert r.max_drawdown_peak_index == 0
    assert r.max_drawdown_trough_index == 0


def test_score_equity_matches_returns_path():
    """Scoring an equity curve == scoring its implied returns (drawdown identical).

    equity [100, 110, 88, 92.4] implies returns [0.10, -0.20, 0.05]; drawdown is
    scale-invariant so magnitude 0.20 and indices (1, 2) are preserved.
    """
    eq = [100.0, 110.0, 88.0, 92.4]
    from_equity = score_equity(eq)
    from_returns = score_returns([0.10, -0.20, 0.05])
    assert from_equity.max_drawdown == pytest.approx(from_returns.max_drawdown, rel=1e-12)
    assert from_equity.max_drawdown_peak_index == from_returns.max_drawdown_peak_index
    assert from_equity.max_drawdown_trough_index == from_returns.max_drawdown_trough_index
    assert from_equity.total_return == pytest.approx(from_returns.total_return, rel=1e-12)


def test_equity_to_returns_known_answer():
    """[100, 110, 88] → [110/100-1, 88/110-1] = [0.10, -0.20]."""
    out = equity_to_returns([100.0, 110.0, 88.0])
    assert out == pytest.approx([0.10, -0.20], rel=1e-12)


def test_equity_to_returns_rejects_nonpositive_base():
    """A zero/negative prior equity makes a return undefined → ValueError."""
    with pytest.raises(ValueError):
        equity_to_returns([100.0, 0.0, 50.0])


# --------------------------------------------------------------------------- #
# 7. Trade statistics.
# --------------------------------------------------------------------------- #
def test_score_trades_known_answer():
    """pnls = [100, -50, 200, -30, 0, 150].

    wins = [100, 200, 150] (gross 450); losses = [-50, -30] (gross -80); 1 breakeven.
    n_trades = 6, win_rate = 3/6 = 0.5.
    avg_win = 450/3 = 150.0, avg_loss = -80/2 = -40.0.
    profit_factor = 450 / 80 = 5.625, total_pnl = 370.
    """
    t = score_trades([100, -50, 200, -30, 0, 150])
    assert t.kind == "trades"
    assert t.n_trades == 6
    assert t.win_rate == pytest.approx(0.5)
    assert t.avg_win == pytest.approx(150.0)
    assert t.avg_loss == pytest.approx(-40.0)
    assert t.profit_factor == pytest.approx(5.625)
    assert t.total_pnl == pytest.approx(370.0)


def test_score_trades_all_wins_profit_factor_none():
    """No losing trades → profit_factor undefined (would be +inf) → None."""
    t = score_trades([10.0, 20.0])
    assert t.win_rate == pytest.approx(1.0)
    assert t.avg_win == pytest.approx(15.0)
    assert t.avg_loss is None
    assert t.profit_factor is None
    assert t.total_pnl == pytest.approx(30.0)


def test_score_trades_all_losses_profit_factor_zero():
    """Losses but no wins → gross_profit 0 → profit_factor 0.0 (a real zero)."""
    t = score_trades([-10.0, -20.0])
    assert t.win_rate == pytest.approx(0.0)
    assert t.avg_win is None
    assert t.avg_loss == pytest.approx(-15.0)
    assert t.profit_factor == 0.0
    assert t.total_pnl == pytest.approx(-30.0)


def test_score_trades_all_breakeven():
    """All-zero PnLs: no wins, no losses → ratios None, win_rate 0.0, sum 0.0."""
    t = score_trades([0.0, 0.0])
    assert t.win_rate == pytest.approx(0.0)
    assert t.avg_win is None
    assert t.avg_loss is None
    assert t.profit_factor is None
    assert t.total_pnl == 0.0


# --------------------------------------------------------------------------- #
# 8. Degenerate-input sentinels (must never raise).
# --------------------------------------------------------------------------- #
def test_empty_returns_all_none():
    """Empty returns → n_periods 0, every return/risk metric None (no data)."""
    r = score_returns([])
    assert r.n_periods == 0
    assert r.total_return is None
    assert r.cagr is None
    assert r.volatility is None
    assert r.sharpe is None
    assert r.sortino is None
    assert r.max_drawdown is None
    assert r.max_drawdown_peak_index is None
    assert r.max_drawdown_trough_index is None


def test_single_element_partial_metrics():
    """Single return: total_return/max_drawdown defined; sample-stdev metrics None."""
    r = score_returns([0.05])
    assert r.n_periods == 1
    assert r.total_return == pytest.approx(0.05)
    assert r.max_drawdown == 0.0  # a single up move has no decline
    assert r.volatility is None
    assert r.sharpe is None
    assert r.sortino is None


def test_single_negative_element_drawdown():
    """A single loss: equity [1.0, 0.97] → max drawdown 0.03 at trough index 1."""
    r = score_returns([-0.03])
    assert r.max_drawdown == pytest.approx(0.03, rel=1e-12)
    assert r.max_drawdown_peak_index == 0
    assert r.max_drawdown_trough_index == 1


def test_all_zero_returns_flat():
    """All-zero returns: total_return 0.0 and volatility 0.0 (computed zeros, not None),
    but Sharpe/Sortino are None (zero denominator). This distinguishes 'flat' (0.0)
    from 'no data' (None)."""
    r = score_returns([0.0, 0.0, 0.0])
    assert r.total_return == 0.0
    assert r.cagr == pytest.approx(0.0, abs=1e-15)
    assert r.volatility == pytest.approx(0.0, abs=1e-15)
    assert r.sharpe is None
    assert r.sortino is None
    assert r.max_drawdown == 0.0


def test_empty_trades_defined_sum():
    """Empty trade book: n_trades 0, total_pnl 0.0 (a defined sum), ratios None."""
    t = score_trades([])
    assert t.n_trades == 0
    assert t.total_pnl == 0.0
    assert t.win_rate is None
    assert t.avg_win is None
    assert t.avg_loss is None
    assert t.profit_factor is None


# --------------------------------------------------------------------------- #
# 9. Config validation (invalid config is a caller bug → raises).
# --------------------------------------------------------------------------- #
def test_nonpositive_periods_per_year_raises():
    with pytest.raises(ValueError):
        score_returns([0.01, 0.02], periods_per_year=0)
    with pytest.raises(ValueError):
        score_returns([0.01, 0.02], periods_per_year=-252)


def test_nonpositive_explicit_years_raises():
    with pytest.raises(ValueError):
        score_returns([0.01, 0.02], years=0.0)
    with pytest.raises(ValueError):
        score_returns([0.01, 0.02], years=-1.0)


def test_bool_input_rejected():
    """bool is a subclass of int but must not masquerade as a 1.0/0.0 return."""
    with pytest.raises(TypeError):
        score_returns([True, False])
    with pytest.raises(TypeError):
        score_trades([True])


# --------------------------------------------------------------------------- #
# 10. Determinism + serialisation.
# --------------------------------------------------------------------------- #
def test_determinism_byte_identical_to_dict():
    """Same input → byte-identical JSON (SSOT §3.8 determinism contract)."""
    returns = [0.011, -0.004, 0.02, -0.015, 0.007, 0.0]
    a = json.dumps(score_returns(returns).to_dict(), sort_keys=True)
    b = json.dumps(score_returns(returns).to_dict(), sort_keys=True)
    assert a == b


def test_to_dict_is_json_serialisable_and_complete():
    """to_dict() round-trips through json and exposes every dataclass field."""
    r = score_returns([0.01, -0.02, 0.03])
    d = r.to_dict()
    restored = json.loads(json.dumps(d))
    assert restored == d
    # Every dataclass field is present in the dict (no silent drops).
    from dataclasses import fields

    assert set(d.keys()) == {f.name for f in fields(ScoreResult)}


def test_label_is_carried_through():
    """The optional label is preserved for report embedding (both entry points)."""
    assert score_returns([0.01, 0.02], label="omega_macro").label == "omega_macro"
    assert score_trades([10.0, -5.0], label="alpha_intraday").label == "alpha_intraday"


def test_default_periods_per_year_constant():
    """Guard the documented daily default so a silent change is caught."""
    assert DEFAULT_PERIODS_PER_YEAR == 252

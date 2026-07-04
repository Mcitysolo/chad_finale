"""Tests for chad/validation/verdict.py — Phase 5 verdict mapping (SSOT Part 4).

Fixture-only. They pin: every verdict type fires on its condition; the fixed precedence
(NOT_REPLAYABLE → CONTAMINATED → INSUFFICIENT_DATA → FAIL → PASS); below-minimums ⇒
INSUFFICIENT_DATA; PASS is always labeled a *candidate*; the portfolio §4.2 pass logic;
and — critically — the verdict layer performs NO I/O (never writes ready_for_live or any
runtime/ file), proven by breaking every filesystem write path while it runs.
"""

from __future__ import annotations

import builtins
import dataclasses
from pathlib import Path

import pytest

from chad.validation import verdict as verdict_mod
from chad.validation.verdict import (
    PASS_LABEL,
    HeadMetrics,
    PortfolioMetrics,
    Verdict,
    VerdictThresholds,
    decide_portfolio_verdict,
    decide_verdict,
)


def _passing_metrics(**overrides) -> HeadMetrics:
    """A metrics record that clears every bar → PASS, mutated per-test via overrides."""
    base = dict(
        head="alpha",
        parity_status="REPLAYABLE",
        replayable=True,
        data_quality_status="CLEAN",
        oos_access_count=1,
        n_oos_trades=40,
        n_walk_forward_windows=8,
        n_regimes_in_oos=4,
        deflated_sharpe_worst=0.99,
        cost_adj_cagr=0.12,
        worst_quantile_ruin=0.0,
        regimes_with_edge=3,
        regime_scoped_sizing=False,
        final_run=True,
        oos_source="sealed_oos",
    )
    base.update(overrides)
    return HeadMetrics(**base)


# --------------------------------------------------------------------------- #
# PASS (candidate).
# --------------------------------------------------------------------------- #
def test_pass_is_labeled_candidate() -> None:
    res = decide_verdict(_passing_metrics())
    assert res.verdict is Verdict.PASS
    assert res.label == PASS_LABEL == "PASS (candidate)"
    assert res.is_candidate is True
    # Every survival check passed.
    assert all(c.passed for c in res.checks)


# --------------------------------------------------------------------------- #
# NOT_REPLAYABLE.
# --------------------------------------------------------------------------- #
def test_not_replayable_passthrough() -> None:
    res = decide_verdict(_passing_metrics(replayable=False, parity_status="NOT_REPLAYABLE"))
    assert res.verdict is Verdict.NOT_REPLAYABLE
    assert res.is_candidate is False


def test_not_replayable_precedes_contamination_and_minimums() -> None:
    # Even with a contaminated count AND below minimums, non-replayable wins first.
    res = decide_verdict(
        _passing_metrics(replayable=False, parity_status="UNKNOWN", oos_access_count=5, n_oos_trades=0)
    )
    assert res.verdict is Verdict.NOT_REPLAYABLE


# --------------------------------------------------------------------------- #
# CONTAMINATED.
# --------------------------------------------------------------------------- #
def test_contaminated_fires_on_double_open() -> None:
    res = decide_verdict(_passing_metrics(oos_access_count=2))
    assert res.verdict is Verdict.CONTAMINATED
    assert res.is_candidate is False


def test_contaminated_precedes_insufficient_and_fail() -> None:
    # Below minimums AND failing numbers, but a twice-opened box → CONTAMINATED first.
    res = decide_verdict(
        _passing_metrics(oos_access_count=3, n_oos_trades=0, deflated_sharpe_worst=0.0)
    )
    assert res.verdict is Verdict.CONTAMINATED


def test_single_open_is_not_contaminated() -> None:
    assert decide_verdict(_passing_metrics(oos_access_count=1)).verdict is Verdict.PASS
    assert decide_verdict(_passing_metrics(oos_access_count=0)).verdict is Verdict.PASS


# --------------------------------------------------------------------------- #
# INSUFFICIENT_DATA — each pre-registered minimum (SSOT §4.3).
# --------------------------------------------------------------------------- #
def test_below_n_min_is_insufficient() -> None:
    assert decide_verdict(_passing_metrics(n_oos_trades=29)).verdict is Verdict.INSUFFICIENT_DATA


def test_below_w_min_is_insufficient() -> None:
    assert decide_verdict(_passing_metrics(n_walk_forward_windows=5)).verdict is Verdict.INSUFFICIENT_DATA


def test_below_r_min_is_insufficient() -> None:
    assert decide_verdict(_passing_metrics(n_regimes_in_oos=2)).verdict is Verdict.INSUFFICIENT_DATA


def test_data_quality_fail_is_insufficient() -> None:
    assert decide_verdict(_passing_metrics(data_quality_status="FAIL")).verdict is Verdict.INSUFFICIENT_DATA


def test_data_quality_warn_is_acceptable() -> None:
    # WARN proceeds (only FAIL blocks) — the head is still judged on its numbers.
    assert decide_verdict(_passing_metrics(data_quality_status="WARN")).verdict is Verdict.PASS


def test_insufficient_precedes_fail() -> None:
    # Below minimums AND a failing number → INSUFFICIENT_DATA (honest default first).
    res = decide_verdict(_passing_metrics(n_oos_trades=10, cost_adj_cagr=-0.5))
    assert res.verdict is Verdict.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# FAIL — each §4.1 survival condition (minimums already met).
# --------------------------------------------------------------------------- #
def test_fail_low_deflated_sharpe() -> None:
    res = decide_verdict(_passing_metrics(deflated_sharpe_worst=0.5))
    assert res.verdict is Verdict.FAIL
    assert any(c.name == "deflated_sharpe" and not c.passed for c in res.checks)


def test_fail_none_deflated_sharpe_is_not_confirmed() -> None:
    # An undefined DSR cannot confirm the edge → FAIL (strict standard).
    res = decide_verdict(_passing_metrics(deflated_sharpe_worst=None))
    assert res.verdict is Verdict.FAIL


def test_fail_nonpositive_cagr() -> None:
    assert decide_verdict(_passing_metrics(cost_adj_cagr=0.0)).verdict is Verdict.FAIL
    assert decide_verdict(_passing_metrics(cost_adj_cagr=-0.01)).verdict is Verdict.FAIL


def test_fail_ruin_above_bound() -> None:
    res = decide_verdict(_passing_metrics(worst_quantile_ruin=0.02))  # > 0.01 bound
    assert res.verdict is Verdict.FAIL
    assert any(c.name == "worst_quantile_ruin" and not c.passed for c in res.checks)


def test_fail_regime_fragile_without_scoped_sizing() -> None:
    res = decide_verdict(_passing_metrics(regimes_with_edge=1, regime_scoped_sizing=False))
    assert res.verdict is Verdict.FAIL


def test_regime_scoped_sizing_rescues_single_regime() -> None:
    res = decide_verdict(_passing_metrics(regimes_with_edge=1, regime_scoped_sizing=True))
    assert res.verdict is Verdict.PASS


# --------------------------------------------------------------------------- #
# Dry-run warning surfaced (decoy source).
# --------------------------------------------------------------------------- #
def test_non_final_run_carries_dry_run_warning() -> None:
    res = decide_verdict(_passing_metrics(final_run=False, oos_source="decoy"))
    assert res.final_run is False
    assert any("dry-run" in w or "decoy" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# Portfolio (SSOT §4.2, S5).
# --------------------------------------------------------------------------- #
def test_portfolio_pass_requires_bar_and_capital_fraction() -> None:
    pm = PortfolioMetrics(
        portfolio=_passing_metrics(head="portfolio", parity_status="PORTFOLIO"),
        surviving_heads=3, total_heads=4, capital_fraction_in_surviving_heads=0.8,
    )
    res = decide_portfolio_verdict(pm)
    assert res.verdict is Verdict.PASS
    assert res.head == "portfolio"
    assert any(c.name == "surviving_capital_fraction" and c.passed for c in res.checks)


def test_portfolio_bar_met_but_low_capital_fraction_fails() -> None:
    pm = PortfolioMetrics(
        portfolio=_passing_metrics(head="portfolio", parity_status="PORTFOLIO"),
        surviving_heads=1, total_heads=5, capital_fraction_in_surviving_heads=0.2,
    )
    res = decide_portfolio_verdict(pm)
    assert res.verdict is Verdict.FAIL
    assert any(c.name == "surviving_capital_fraction" and not c.passed for c in res.checks)


def test_portfolio_propagates_insufficient() -> None:
    pm = PortfolioMetrics(
        portfolio=_passing_metrics(head="portfolio", n_oos_trades=0),
        surviving_heads=0, total_heads=1, capital_fraction_in_surviving_heads=0.0,
    )
    res = decide_portfolio_verdict(pm)
    assert res.verdict is Verdict.INSUFFICIENT_DATA
    assert res.head == "portfolio"


# --------------------------------------------------------------------------- #
# Thresholds config validation.
# --------------------------------------------------------------------------- #
def test_thresholds_reject_bad_config() -> None:
    with pytest.raises(ValueError):
        VerdictThresholds(dsr_confidence=1.5)
    with pytest.raises(ValueError):
        VerdictThresholds(ruin_bound=0.0)
    with pytest.raises(ValueError):
        VerdictThresholds(n_min=0)


def test_pre_registered_minimums_are_the_committed_values() -> None:
    t = VerdictThresholds()
    assert (t.n_min, t.w_min, t.r_min) == (30, 6, 3)
    assert (verdict_mod.N_MIN, verdict_mod.W_MIN, verdict_mod.R_MIN) == (30, 6, 3)


# --------------------------------------------------------------------------- #
# The hard guarantee: verdict.py performs NO I/O (never writes runtime / ready_for_live).
# --------------------------------------------------------------------------- #
def test_verdict_writes_nothing_even_with_all_write_paths_broken(monkeypatch, tmp_path) -> None:
    """If decide_verdict tried to open/write anything, these poisoned hooks would raise."""

    def _boom_open(*args, **kwargs):
        raise AssertionError("verdict.py must not open any file")

    def _boom_write_text(self, *args, **kwargs):
        raise AssertionError("verdict.py must not write any file")

    def _boom_write_bytes(self, *args, **kwargs):
        raise AssertionError("verdict.py must not write any file")

    monkeypatch.setattr(builtins, "open", _boom_open)
    monkeypatch.setattr(Path, "write_text", _boom_write_text)
    monkeypatch.setattr(Path, "write_bytes", _boom_write_bytes)

    # Exercise every branch; none may touch the filesystem.
    for m in (
        _passing_metrics(),
        _passing_metrics(replayable=False),
        _passing_metrics(oos_access_count=2),
        _passing_metrics(n_oos_trades=0),
        _passing_metrics(deflated_sharpe_worst=0.1),
    ):
        decide_verdict(m)
    decide_portfolio_verdict(
        PortfolioMetrics(
            portfolio=_passing_metrics(head="portfolio"),
            surviving_heads=1, total_heads=1, capital_fraction_in_surviving_heads=1.0,
        )
    )
    # No files were created in tmp cwd either.
    assert not list(tmp_path.iterdir())


def test_verdict_module_has_no_write_surface_in_source() -> None:
    """Static belt-and-suspenders: the module source calls no write/open primitive."""
    src = Path(verdict_mod.__file__).read_text(encoding="utf-8")
    # Split off docstrings/reason strings by checking for actual call syntax.
    for forbidden in ("open(", ".write(", ".write_text(", ".write_bytes(", "os.", "Path("):
        assert forbidden not in src, f"verdict.py unexpectedly references {forbidden!r}"

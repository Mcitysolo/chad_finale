"""Known-answer + independence tests for chad.validation.regime_labeler (Phase 2, §3.4).

Every regime assignment is derived by hand in the test docstring from a tiny price
fixture with a small ``lookback`` (so the two-point realized-vol ``|r1-r2|/sqrt(2)``
is checkable by inspection). The CRITICAL guarantee — the labeler is *independent*
of CHAD's own ``regime_classifier`` (SSOT §3.4 / V4) — is enforced by an AST scan of
this module's imports (a defendant may not pick the judge).

Fixtures are in-memory price lists; nothing here touches the bar corpus, network,
or runtime state.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

import chad.validation.regime_labeler as regime_labeler
from chad.validation.regime_labeler import (
    DEFAULT_REGIME_CONFIG,
    Regime,
    RegimeConfig,
    RegimeSeries,
    label_series,
)
from chad.validation.regime_labeler import _quantile  # white-box: the tercile math

SQRT2 = math.sqrt(2.0)
# A short-lookback config makes two-point realized vol hand-checkable.
CFG = RegimeConfig(lookback=2, flat_threshold=0.01, vol_high_quantile=2.0 / 3.0)


# --------------------------------------------------------------------------- #
# 1. Known-answer regime fixtures — all five labels + UNKNOWN.
# --------------------------------------------------------------------------- #
def test_fixture_a_flat_bull_calm_bear_vol():
    """prices = [100, 100, 100, 110, 88], lookback=2, flat=0.01, q=2/3.

    t=0,1 → UNKNOWN (t < lookback).
    t=2: window [100,100,100]; rets [0,0]; trailing 0.0 → FLAT; vol 0.
    t=3: window [100,100,110]; rets [0, 0.10]; trailing 0.10 → BULL; vol 0.10/sqrt2.
    t=4: window [100,110,88]; rets [0.10, -0.20]; trailing -0.12 → BEAR; vol 0.30/sqrt2.
    vols {0, 0.070711, 0.212132}; tercile(2/3) ≈ 0.117851.
    vol_high: only t=4 (0.212132 >= 0.117851) → BULL_CALM (t=3) / BEAR_VOL (t=4).
    """
    rs = label_series([100.0, 100.0, 100.0, 110.0, 88.0], config=CFG)
    assert isinstance(rs, RegimeSeries)
    regimes = [lab.regime for lab in rs.labels]
    assert regimes == [
        Regime.UNKNOWN,
        Regime.UNKNOWN,
        Regime.FLAT,
        Regime.BULL_CALM,
        Regime.BEAR_VOL,
    ]
    assert rs.n_labeled == 3
    assert rs.labels[2].realized_vol == pytest.approx(0.0)
    assert rs.labels[3].realized_vol == pytest.approx(0.10 / SQRT2)
    assert rs.labels[4].realized_vol == pytest.approx(0.30 / SQRT2)
    assert rs.vol_threshold == pytest.approx(0.117851, rel=1e-4)
    assert rs.labels[4].vol_high is True
    assert rs.labels[3].vol_high is False


def test_fixture_b_flat_bear_calm_bull_vol():
    """prices = [100, 100, 100, 90, 120], lookback=2, flat=0.01, q=2/3.

    t=2: trailing 0.0 → FLAT; vol 0.
    t=3: window [100,100,90]; rets [0,-0.10]; trailing -0.10 → BEAR; vol 0.10/sqrt2.
    t=4: window [100,90,120]; rets [-0.10, 0.333333]; trailing 0.20 → BULL;
         vol |−0.10−0.333333|/sqrt2 = 0.433333/sqrt2 ≈ 0.306413.
    vols {0, 0.070711, 0.306413}; tercile(2/3) ≈ 0.149278.
    vol_high: only t=4 → BEAR_CALM (t=3) / BULL_VOL (t=4).
    """
    rs = label_series([100.0, 100.0, 100.0, 90.0, 120.0], config=CFG)
    regimes = [lab.regime for lab in rs.labels]
    assert regimes == [
        Regime.UNKNOWN,
        Regime.UNKNOWN,
        Regime.FLAT,
        Regime.BEAR_CALM,
        Regime.BULL_VOL,
    ]
    assert rs.vol_threshold == pytest.approx(0.149278, rel=1e-4)
    assert rs.counts["bull_vol"] == 1
    assert rs.counts["bear_calm"] == 1
    assert rs.counts["flat"] == 1
    assert rs.counts["unknown"] == 2


def test_first_lookback_bars_are_unknown():
    """Every bar with index < lookback lacks trailing history → UNKNOWN sentinel."""
    rs = label_series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], config=RegimeConfig(lookback=3))
    for i in range(3):
        assert rs.labels[i].regime is Regime.UNKNOWN
        assert rs.labels[i].trailing_return is None
        assert rs.labels[i].realized_vol is None
        assert rs.labels[i].vol_high is None


# --------------------------------------------------------------------------- #
# 2. Vol tercile boundaries — the calm/vol cut is exactly the quantile.
# --------------------------------------------------------------------------- #
def test_quantile_known_answers():
    """Type-7 linear-interpolated quantile (numpy-default), checked by hand.

    sorted [10,20,30,40]:
      q=0     → 10
      q=1     → 40
      q=1/3   → pos 1.0        → 20
      q=2/3   → pos 2.0        → 30
      q=0.5   → pos 1.5        → 25
    single-element [7] → 7 for any q.
    """
    s = [10.0, 20.0, 30.0, 40.0]
    assert _quantile(s, 0.0) == 10.0
    assert _quantile(s, 1.0) == 40.0
    assert _quantile(s, 1.0 / 3.0) == pytest.approx(20.0)
    assert _quantile(s, 2.0 / 3.0) == pytest.approx(30.0)
    assert _quantile(s, 0.5) == pytest.approx(25.0)
    assert _quantile([7.0], 0.9) == 7.0


def test_vol_high_flag_matches_threshold_boundary():
    """Each labelable bar is 'vol' iff realized_vol >= vol_threshold (inclusive)."""
    rs = label_series([100.0, 100.0, 100.0, 110.0, 88.0], config=CFG)
    thr = rs.vol_threshold
    assert thr is not None
    for lab in rs.labels:
        if lab.realized_vol is None:
            continue
        assert lab.vol_high == (lab.realized_vol >= thr)


def test_higher_quantile_makes_fewer_bars_vol():
    """Raising the tercile cut can only shrink the 'vol' set (monotone)."""
    prices = [100.0, 101.0, 99.0, 103.0, 97.0, 105.0, 95.0, 108.0]
    low = label_series(prices, config=RegimeConfig(lookback=2, vol_high_quantile=0.5))
    high = label_series(prices, config=RegimeConfig(lookback=2, vol_high_quantile=0.9))
    n_low = sum(1 for lab in low.labels if lab.vol_high)
    n_high = sum(1 for lab in high.labels if lab.vol_high)
    assert n_high <= n_low


# --------------------------------------------------------------------------- #
# 3. Degenerate data → documented sentinels (never raises).
# --------------------------------------------------------------------------- #
def test_empty_series_is_sentinel():
    rs = label_series([], config=CFG)
    assert rs.labels == ()
    assert rs.n_labeled == 0
    assert rs.vol_threshold is None
    assert rs.warnings  # records the empty-series warning
    assert rs.counts["unknown"] == 0


def test_short_series_all_unknown():
    """n <= lookback → no bar has enough history → all UNKNOWN + warning."""
    rs = label_series([100.0, 101.0], config=CFG)  # n=2, lookback=2
    assert all(lab.regime is Regime.UNKNOWN for lab in rs.labels)
    assert rs.n_labeled == 0
    assert rs.vol_threshold is None
    assert rs.warnings


def test_constant_prices_zero_dispersion_all_flat_calm():
    """Constant prices → zero returns → zero vol, zero dispersion.

    trailing return 0 → FLAT for every labelable bar; the tercile is undefined so
    vol_threshold is None and a zero-dispersion warning is recorded.
    """
    rs = label_series([50.0, 50.0, 50.0, 50.0, 50.0], config=CFG)
    labelable = [lab for lab in rs.labels if lab.regime is not Regime.UNKNOWN]
    assert labelable and all(lab.regime is Regime.FLAT for lab in labelable)
    assert rs.vol_threshold is None
    assert any("dispersion" in w for w in rs.warnings)
    # No bar is spuriously 'vol' when there is no dispersion.
    assert all(lab.vol_high is False for lab in labelable)


def test_zero_return_is_flat_even_with_flat_threshold_zero():
    """A directionless (exactly-zero) trailing return is FLAT even when
    flat_threshold=0.0 — a zero return must not be asymmetrically folded into BEAR.

    Constant prices → every labelable bar has trailing_return == 0.0; with the
    (valid) config flat_threshold=0.0 they must still be FLAT, never bear_*.
    """
    cfg0 = RegimeConfig(lookback=2, flat_threshold=0.0, vol_high_quantile=2.0 / 3.0)
    rs = label_series([100.0, 100.0, 100.0, 100.0, 100.0], config=cfg0)
    labelable = [lab for lab in rs.labels if lab.regime is not Regime.UNKNOWN]
    assert labelable and all(lab.regime is Regime.FLAT for lab in labelable)
    assert rs.counts["bear_calm"] == 0 and rs.counts["bear_vol"] == 0


def test_non_positive_price_in_window_is_unknown():
    """A non-positive price makes returns undefined → those bars UNKNOWN, no raise.

    prices = [100, 0, 100, 101, 102], lookback=2:
      t=2 window [100,0,100]   → contains 0 → UNKNOWN
      t=3 window [0,100,101]   → contains 0 → UNKNOWN
      t=4 window [100,101,102] → clean       → labeled (trailing 0.02 > flat → BULL)
    """
    rs = label_series([100.0, 0.0, 100.0, 101.0, 102.0], config=CFG)
    assert rs.labels[2].regime is Regime.UNKNOWN
    assert rs.labels[3].regime is Regime.UNKNOWN
    assert rs.labels[4].regime is not Regime.UNKNOWN


# --------------------------------------------------------------------------- #
# 4. CHAD label passthrough — carried, NEVER authoritative.
# --------------------------------------------------------------------------- #
def test_chad_labels_passthrough_does_not_change_regime():
    """chad_labels ride along for side-by-side reporting only; a deliberately
    contradictory CHAD label must not flip the independent regime."""
    prices = [100.0, 100.0, 100.0, 110.0, 88.0]
    chad = ["x", "x", "trending_bear", "trending_bear", "trending_bull"]  # wrong on purpose
    rs = label_series(prices, config=CFG, chad_labels=chad)
    # Independent labels unchanged from fixture A.
    assert [lab.regime for lab in rs.labels] == [
        Regime.UNKNOWN, Regime.UNKNOWN, Regime.FLAT, Regime.BULL_CALM, Regime.BEAR_VOL
    ]
    # But the CHAD label is carried verbatim for comparison.
    assert rs.labels[3].chad_label == "trending_bear"
    assert rs.labels[4].chad_label == "trending_bull"


def test_chad_labels_without_passthrough_are_none():
    rs = label_series([100.0, 100.0, 100.0, 110.0, 88.0], config=CFG)
    assert all(lab.chad_label is None for lab in rs.labels)


def test_chad_labels_length_mismatch_raises():
    with pytest.raises(ValueError):
        label_series([100.0, 101.0, 102.0], config=CFG, chad_labels=["a", "b"])


# --------------------------------------------------------------------------- #
# 5. Determinism.
# --------------------------------------------------------------------------- #
def test_determinism_same_series_same_labels():
    prices = [100.0, 102.0, 101.0, 105.0, 99.0, 110.0, 95.0]
    a = label_series(prices, config=CFG)
    b = label_series(prices, config=CFG)
    assert a.to_dict() == b.to_dict()


# --------------------------------------------------------------------------- #
# 6. Invalid config → ValueError.
# --------------------------------------------------------------------------- #
def test_invalid_config_raises():
    with pytest.raises(ValueError):
        RegimeConfig(lookback=1)               # need >= 2 for a sample stdev
    with pytest.raises(ValueError):
        RegimeConfig(flat_threshold=-0.01)     # negative band
    with pytest.raises(ValueError):
        RegimeConfig(vol_high_quantile=0.0)    # must be in (0,1)
    with pytest.raises(ValueError):
        RegimeConfig(vol_high_quantile=1.0)
    with pytest.raises(ValueError):
        label_series([100.0, 101.0], config="not a config")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 7. INDEPENDENCE — the labeler must NOT import chad's regime_classifier (V4).
# --------------------------------------------------------------------------- #
def _imported_module_names(path: Path) -> set[str]:
    """Every module referenced by an ``import`` / ``from ... import`` in ``path``."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.add(node.module)
    return names


def test_regime_labeler_does_not_import_chad_regime_classifier():
    """CRITICAL (SSOT §3.4 / V4): no import of chad's regime classifier (or any
    chad.analytics module). The mention in the docstring is fine; the AST scan only
    inspects actual import statements, so it is precise, not a substring false-match.
    """
    imported = _imported_module_names(Path(regime_labeler.__file__))
    for name in imported:
        assert "regime_classifier" not in name, f"forbidden import of {name}"
        assert not name.startswith("chad.analytics"), f"forbidden analytics import {name}"


def test_regime_labeler_imports_are_stdlib_only():
    """Positive control: the labeler imports only stdlib / __future__ — proving its
    independence structurally (no first-party import can smuggle CHAD's classifier in)."""
    allowed_roots = {"__future__", "math", "dataclasses", "enum", "typing"}
    imported = _imported_module_names(Path(regime_labeler.__file__))
    for name in imported:
        assert name.split(".")[0] in allowed_roots, f"unexpected import: {name}"


def test_default_config_is_frozen_and_echoed():
    """Sanity: the default config round-trips through the report echo."""
    rs = label_series([100.0, 100.0, 100.0, 110.0, 88.0])
    assert rs.config_echo == DEFAULT_REGIME_CONFIG.to_dict()
    assert rs.config_echo["lookback"] == 20

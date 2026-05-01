"""Tests for Phase-8 Session 4 regime layer (G1 + G2 + G3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.analytics.regime_classifier import (
    DEFAULT_ADX_THRESHOLD,
    DEFAULT_VOL_PCTILE_THRESHOLD,
    classify_regime,
    read_regime_state,
    write_regime_state,
)
from chad.portfolio.regime_activation import (
    allowed_strategies_for_regime,
    filter_intents_by_regime,
    is_strategy_allowed,
    load_activation_matrix,
)
from chad.risk.regime_reduction import (
    DEFAULT_ADVERSE_REDUCTION_PCT,
    DEFAULT_VOLATILE_REDUCTION_PCT,
    generate_partial_close_intents,
    handle_regime_transition,
    should_reduce_on_transition,
)


# ---------------------------------------------------------------------------
# G1 — classifier
# ---------------------------------------------------------------------------


def test_high_vol_produces_volatile():
    result = classify_regime(realized_vol_percentile=0.9)
    assert result.regime == "volatile"
    assert "realized_vol_percentile" in result.inputs_used


def test_trending_adx_above_25_bull():
    result = classify_regime(adx=40.0, trend_slope=0.5)
    assert result.regime == "trending_bull"


def test_trending_adx_above_25_bear():
    result = classify_regime(adx=40.0, trend_slope=-0.5)
    assert result.regime == "trending_bear"


def test_ranging_when_adx_below_25_and_vol_calm():
    result = classify_regime(adx=15.0, realized_vol_percentile=0.3)
    assert result.regime == "ranging"


def test_unknown_when_no_data():
    result = classify_regime()
    assert result.regime == "unknown"
    assert result.inputs_used == []


def test_unknown_when_insufficient_data():
    """trend_slope alone — no adx, no vol — should not classify confidently."""
    result = classify_regime(trend_slope=0.5)
    assert result.regime == "unknown"


def test_volatile_overrides_trending():
    """High vol should win even if ADX signals a trend."""
    result = classify_regime(realized_vol_percentile=0.9, adx=40.0, trend_slope=0.5)
    assert result.regime == "volatile"


def test_classify_regime_never_raises_on_garbage():
    result = classify_regime(adx="nope", trend_slope=float("nan"))
    assert result.regime in {"unknown", "ranging"}  # degrades cleanly


# ---------------------------------------------------------------------------
# G1 — persistence
# ---------------------------------------------------------------------------


def test_regime_written_to_json(tmp_path: Path):
    path = tmp_path / "regime_state.json"
    result = classify_regime(adx=40.0, trend_slope=0.5)
    payload = write_regime_state(result, source="test", path=path)
    assert payload["regime"] == "trending_bull"
    assert payload["ok"] is True
    assert payload["source"] == "test"
    # File parses.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["regime"] == "trending_bull"


def test_regime_state_captures_previous(tmp_path: Path):
    path = tmp_path / "regime_state.json"
    write_regime_state(classify_regime(adx=40.0, trend_slope=0.5), path=path)  # trending_bull
    second = write_regime_state(
        classify_regime(realized_vol_percentile=0.9), path=path
    )  # volatile
    assert second["previous_regime"] == "trending_bull"
    assert second["regime"] == "volatile"


def test_read_regime_state_missing_returns_default(tmp_path: Path):
    path = tmp_path / "missing.json"
    data = read_regime_state(path)
    assert data["regime"] == "unknown"
    assert data["ok"] is False


# ---------------------------------------------------------------------------
# G2 — activation matrix
# ---------------------------------------------------------------------------


def test_g2_matrix_loads_with_calibrated_values():
    # Audit-O calibrated the matrix per regime (2026-04-22):
    # trending_bull/bear contain momentum+trend families; ranging has
    # mean-reversion; volatile has vol-family; adverse is empty.
    mat = load_activation_matrix()
    assert mat  # file exists and parsed
    # trending regimes: momentum families are present.
    assert "alpha" in mat["trending_bull"]
    assert "delta" in mat["trending_bear"]
    # ranging: mean-reversion families.
    assert "delta_pairs" in mat["ranging"]
    assert "gamma_reversion" in mat["ranging"]
    # volatile: narrow to vol-family only.
    assert "omega_vol" in mat["volatile"]
    assert len(mat["volatile"]) <= 16
    # adverse: hard silence.
    assert mat["adverse"] == []


def test_g2_strategy_allowed_when_in_list():
    mat = {"trending_bull": ["alpha", "gamma"]}
    assert is_strategy_allowed("alpha", "trending_bull", mat) is True
    assert is_strategy_allowed("gamma", "trending_bull", mat) is True


def test_g2_strategy_filtered_by_regime():
    mat = {"volatile": ["gamma"]}
    assert is_strategy_allowed("alpha", "volatile", mat) is False
    assert is_strategy_allowed("gamma", "volatile", mat) is True


def test_g2_unknown_regime_with_unknown_bucket_falls_back():
    mat = {"unknown": ["alpha"]}
    assert is_strategy_allowed("alpha", "garbage", mat) is True
    assert is_strategy_allowed("delta", "garbage", mat) is False


def test_g2_missing_matrix_fails_open():
    """No matrix at all → every strategy allowed (fail-open)."""
    assert is_strategy_allowed("anything", "any_regime", {}) is True


class _FakeIntent:
    def __init__(self, strategy: str, symbol: str = "SPY"):
        self.strategy = strategy
        self.symbol = symbol


def test_g2_filter_intents_partitions_allowed_and_rejected():
    mat = {"volatile": ["gamma"]}
    intents = [_FakeIntent("alpha"), _FakeIntent("gamma"), _FakeIntent("delta")]
    allowed, rejected = filter_intents_by_regime(intents, "volatile", mat)
    assert len(allowed) == 1
    assert allowed[0].strategy == "gamma"
    rejected_strategies = [i.strategy for i, _ in rejected]
    assert set(rejected_strategies) == {"alpha", "delta"}


def test_g2_filter_fails_open_with_empty_matrix():
    intents = [_FakeIntent("alpha")]
    allowed, rejected = filter_intents_by_regime(intents, "volatile", {})
    assert len(allowed) == 1
    assert rejected == []


# ---------------------------------------------------------------------------
# G3 — regime reduction
# ---------------------------------------------------------------------------


def test_g3_reduction_on_adverse_transition():
    d = should_reduce_on_transition("trending_bull", "adverse")
    assert d.should_reduce is True
    assert d.pct == pytest.approx(DEFAULT_ADVERSE_REDUCTION_PCT)


def test_g3_reduction_on_volatile_transition():
    d = should_reduce_on_transition("ranging", "volatile")
    assert d.should_reduce is True
    assert d.pct == pytest.approx(DEFAULT_VOLATILE_REDUCTION_PCT)


def test_g3_no_reduction_on_unknown_warning_only():
    d = should_reduce_on_transition("trending_bull", "unknown")
    assert d.should_reduce is False
    assert d.warn_only is True


def test_g3_no_reduction_on_neutral_transition():
    d = should_reduce_on_transition("ranging", "trending_bull")
    assert d.should_reduce is False
    assert d.warn_only is False


def test_g3_no_reduction_when_regimes_equal():
    d = should_reduce_on_transition("trending_bull", "trending_bull")
    assert d.should_reduce is False


def test_g3_no_reduction_when_no_previous_regime():
    d = should_reduce_on_transition(None, "adverse")
    assert d.should_reduce is False


def test_g3_generates_partial_close_intents():
    open_positions = {
        "alpha|SPY": {
            "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 100.0,
        },
        "gamma|QQQ": {
            "strategy": "gamma", "symbol": "QQQ",
            "side": "SELL", "quantity": 50.0,
        },
    }
    intents = generate_partial_close_intents(open_positions, 0.5, "test_reason")
    assert len(intents) == 2
    by_symbol = {i["symbol"]: i for i in intents}
    assert by_symbol["SPY"]["quantity"] == pytest.approx(50.0)
    assert by_symbol["SPY"]["close_side"] == "SELL"
    assert by_symbol["QQQ"]["quantity"] == pytest.approx(25.0)
    assert by_symbol["QQQ"]["close_side"] == "BUY"


def test_g3_skips_fractional_results_below_one_unit():
    open_positions = {
        "alpha|SPY": {
            "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 1.0,  # 0.5 × 1 = 0.5 → rounds to 0 → dropped
        },
    }
    intents = generate_partial_close_intents(open_positions, 0.5, "test")
    assert intents == []


def test_g3_handle_regime_transition_adverse_emits_closes():
    open_positions = {
        "alpha|SPY": {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "quantity": 100.0},
    }
    result = handle_regime_transition(
        from_regime="trending_bull",
        to_regime="adverse",
        open_positions=open_positions,
    )
    assert result["decision"]["should_reduce"] is True
    assert len(result["close_intents"]) == 1


def test_g3_handle_regime_transition_unknown_is_warn_only():
    open_positions = {
        "alpha|SPY": {"strategy": "alpha", "symbol": "SPY", "side": "BUY", "quantity": 100.0},
    }
    result = handle_regime_transition(
        from_regime="trending_bull",
        to_regime="unknown",
        open_positions=open_positions,
    )
    assert result["decision"]["warn_only"] is True
    assert result["close_intents"] == []

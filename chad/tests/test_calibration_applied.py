"""Verify Audit-O calibration values actually land in the runtime code.

Each test ties a config file value to the module that reads it; a
future session that silently mutates either side fails these tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config files present with expected values
# ---------------------------------------------------------------------------


def test_sizing_config_target_vol_is_0_015():
    cfg = json.loads((REPO_ROOT / "config" / "sizing_config.json").read_text())
    assert cfg["vol_adjusted_sizer"]["target_daily_vol"] == pytest.approx(0.015)


def test_sizing_config_correlation_threshold_is_0_65():
    cfg = json.loads((REPO_ROOT / "config" / "sizing_config.json").read_text())
    assert cfg["correlation_monitor"]["threshold"] == pytest.approx(0.65)


def test_sizing_config_max_per_symbol_is_300():
    cfg = json.loads((REPO_ROOT / "config" / "sizing_config.json").read_text())
    assert cfg["composite_cap"]["max_per_symbol"] == 300
    assert cfg["composite_cap"]["max_adv_pct"] == pytest.approx(0.005)
    assert cfg["composite_cap"]["max_position_pct"] == pytest.approx(0.05)


def test_signal_stacking_min_votes_is_1():
    """2026-04-22 revert: min_votes lowered back to 1 after Audit-O strategy-
    fixes pass observed quorum starvation with only 1-2 strategy families
    actively firing per cycle. Window extended to 300s so re-raising to 2
    later doesn't require re-editing the config. See
    reports/strategy_fixes_20260422.json for rationale.
    """
    cfg = json.loads((REPO_ROOT / "config" / "signal_stacking_config.json").read_text())
    assert cfg["min_votes"] == 1
    assert cfg["window_seconds"] == 300


def test_edge_decay_config_exists_and_threshold_is_5():
    cfg = json.loads((REPO_ROOT / "config" / "edge_decay_config.json").read_text())
    assert cfg["consecutive_threshold"] == 5
    assert cfg["min_trades"] == 20


def test_simulated_oms_config_has_class_bps():
    cfg = json.loads((REPO_ROOT / "config" / "simulated_oms_config.json").read_text())
    classes = cfg["slippage_bps_by_class"]
    assert classes["equity_etf"] == pytest.approx(3.0)
    assert classes["futures"] == pytest.approx(1.5)
    assert classes["crypto"] == pytest.approx(8.0)


def test_threshold_adapter_config_has_softened_multipliers():
    cfg = json.loads((REPO_ROOT / "config" / "threshold_adapter_config.json").read_text())
    mults = cfg["regime_multipliers"]
    assert mults["ranging"] == pytest.approx(0.9)
    assert mults["volatile"] == pytest.approx(1.3)
    assert mults["unknown"] == pytest.approx(1.1)
    # adverse stays hard-gated.
    assert mults["adverse"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Regime matrix has real (non-no-op) values
# ---------------------------------------------------------------------------


def test_regime_matrix_adverse_is_empty():
    """Adverse regime silences all strategies."""
    cfg = json.loads((REPO_ROOT / "config" / "regime_activation_matrix.json").read_text())
    assert cfg["regimes"]["adverse"] == []


def test_regime_matrix_volatile_includes_vol_and_momentum():
    """Volatile regime must ship vol-family AND the momentum/trend strategies
    that CHAD's classifier keeps calling 'volatile' when ADX>=25 (trending-
    volatile tape). Narrowing to vol-only starved the book in 2026-04 — this
    test locks in the 2026-04-22 Audit-O+P expansion."""
    cfg = json.loads((REPO_ROOT / "config" / "regime_activation_matrix.json").read_text())
    volatile = cfg["regimes"]["volatile"]
    assert "omega_vol" in volatile or "omega" in volatile
    assert "alpha" in volatile
    assert "alpha_futures" in volatile
    assert "delta" in volatile
    assert "beta_trend" in volatile


def test_regime_matrix_ranging_favors_mean_reversion():
    cfg = json.loads((REPO_ROOT / "config" / "regime_activation_matrix.json").read_text())
    ranging = cfg["regimes"]["ranging"]
    assert "delta_pairs" in ranging
    assert "gamma_reversion" in ranging


def test_regime_matrix_trending_favors_momentum():
    cfg = json.loads((REPO_ROOT / "config" / "regime_activation_matrix.json").read_text())
    for key in ("trending_bull", "trending_bear"):
        lst = cfg["regimes"][key]
        assert "alpha" in lst
        assert "delta" in lst
        assert "beta_trend" in lst


# ---------------------------------------------------------------------------
# Code consumers read the configured values
# ---------------------------------------------------------------------------


def test_vol_adjusted_sizer_reads_target_0_015():
    from chad.risk.vol_adjusted_sizer import VolAdjustedSizer
    sizer = VolAdjustedSizer.from_config()
    assert sizer.target_daily_vol == pytest.approx(0.015)


def test_correlation_monitor_reads_threshold_0_65():
    from chad.risk.correlation_monitor import CorrelationMonitor
    mon = CorrelationMonitor.from_config()
    assert mon.threshold == pytest.approx(0.65)


def test_composite_size_cap_reads_300_max_per_symbol():
    from chad.risk.composite_size_cap import CompositeSizeCap
    cap = CompositeSizeCap.from_config()
    assert cap.max_per_symbol == 300
    assert cap.max_adv_pct == pytest.approx(0.005)
    assert cap.max_position_pct == pytest.approx(0.05)


def test_edge_decay_monitor_reads_threshold_5():
    from chad.risk.edge_decay_monitor import EdgeDecayMonitor
    m = EdgeDecayMonitor()
    assert m.consecutive_threshold == 5
    assert m.min_trades == 20


def test_threshold_adapter_reads_config_softened_values():
    from chad.analytics.threshold_adapter import adjust, load_regime_multipliers
    # Force a fresh read — tests can be order-sensitive.
    load_regime_multipliers(force_reload=True)
    assert adjust(1.0, "volatile") == pytest.approx(1.3)
    assert adjust(1.0, "ranging") == pytest.approx(0.9)
    assert adjust(1.0, "unknown") == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# SimulatedOMS classifies + looks up slippage by asset class
# ---------------------------------------------------------------------------


def test_simulated_oms_classifies_and_applies_per_class_bps():
    from chad.execution.oms import SimulatedOMS, _classify_symbol
    oms = SimulatedOMS()
    # Classification first.
    assert _classify_symbol("SPY", oms._futures_symbols) == "equity_etf"
    assert _classify_symbol("MCL", oms._futures_symbols) == "futures"
    assert _classify_symbol("BTC-USD", oms._futures_symbols) == "crypto"
    # Per-class bps lookup.
    assert oms._resolve_slippage_bps("SPY") == pytest.approx(3.0)
    assert oms._resolve_slippage_bps("MCL") == pytest.approx(1.5)
    assert oms._resolve_slippage_bps("BTC-USD") == pytest.approx(8.0)


def test_simulated_oms_override_flag_disables_config_lookup():
    from chad.execution.oms import SimulatedOMS
    oms = SimulatedOMS(slippage_bps=5.0, use_config_overrides=False)
    # Override flag disables class lookup — constructor value wins.
    assert oms._resolve_slippage_bps("SPY") == 5.0
    assert oms._resolve_slippage_bps("MCL") == 5.0
    assert oms._resolve_slippage_bps("BTC-USD") == 5.0


# ---------------------------------------------------------------------------
# Code fixes
# ---------------------------------------------------------------------------


def test_signal_decay_dir_created_on_init(tmp_path):
    """SignalDecayRecorder.__init__ now ensures its ledger_dir exists."""
    from chad.analytics.signal_decay import SignalDecayRecorder
    ledger = tmp_path / "decay_ledger"
    # Directory does not exist yet.
    assert not ledger.is_dir()
    SignalDecayRecorder(ledger_dir=ledger)
    # After construction it does.
    assert ledger.is_dir()


def test_paper_exec_evidence_accepts_expected_price():
    """Evidence dataclass accepts expected_price kwarg — live_loop +
    position_reconciler pass it through to enable slippage computation."""
    from chad.execution.paper_exec_evidence_writer import PaperExecEvidence
    ev = PaperExecEvidence(
        symbol="SPY",
        side="BUY",
        quantity=100.0,
        fill_price=500.10,
        expected_price=500.0,
        strategy="alpha",
        source_strategies=["alpha"],
        broker="ibkr_paper",
        status="submitted",
        asset_class="equity",
        is_live=False,
        fill_time_utc="2026-04-22T12:00:00Z",
    )
    assert ev.expected_price == pytest.approx(500.0)

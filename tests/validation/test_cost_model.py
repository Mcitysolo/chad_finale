"""Known-answer + edge tests for chad.validation.cost_model (Phase 2, SSOT §3.5).

Every numeric assertion is derived by hand in the test docstring against the
conservative default :class:`CostConfig`; no expected value is produced by
re-running the module under test. Floating-point results use ``pytest.approx``;
exact values (commission floors, counts, flags, "no fill") are asserted exactly.

Fixtures are tiny in-memory records; nothing here touches the bar corpus, the
network, a broker, or any runtime state.
"""

from __future__ import annotations

import math

import pytest

from chad.validation.cost_model import (
    CostBreakdown,
    CostConfig,
    DEFAULT_COST_CONFIG,
    InstrumentClass,
    IntrabarResolution,
    LiquidityTier,
    Trade,
    apply_costs,
    resolve_intrabar,
)

SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
# 1. Commission — per instrument class, both legs.
# --------------------------------------------------------------------------- #
def test_stk_commission_min_floor_applies():
    """STK, qty=100 @ $0.005/share = $0.50/leg < $1.00 floor → floor per leg.

    commission = 2 legs * max(0.005*100, 1.00) = 2 * 1.00 = 2.00.
    No volatility / volume given → both a vol and a volume warning are recorded.
    """
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=100,
        entry_price=50.0,
        exit_price=52.0,
    )
    b = apply_costs(t)
    assert b.commission == pytest.approx(2.00)
    assert b.instrument_class == "STK"
    # Degenerate-but-valid: no vol, no volume → conservative zero-terms + warnings.
    assert len(b.warnings) == 2


def test_stk_commission_per_share_above_floor():
    """STK, qty=500 @ $0.005 = $2.50/leg > $1.00 floor → per-share governs.

    commission = 2 * max(0.005*500, 1.00) = 2 * 2.50 = 5.00.
    """
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=500,
        entry_price=20.0,
        exit_price=21.0,
    )
    assert apply_costs(t).commission == pytest.approx(5.00)


def test_fut_commission_flat_per_contract_independent_of_multiplier():
    """FUT, qty=3 contracts, $0.85/contract → 2 * (0.85*3) = 5.10, flat.

    The point ``multiplier`` scales notional (spread/slippage) but NOT commission.
    """
    t = Trade(
        instrument_class=InstrumentClass.FUT,
        quantity=3,
        entry_price=4000.0,
        exit_price=4010.0,
        multiplier=5.0,
        volatility=0.0,
        bar_volume=100_000,
    )
    assert apply_costs(t).commission == pytest.approx(5.10)


def test_crypto_commission_is_rate_times_notional_both_legs():
    """CRYPTO, qty=2 @ rate 0.18% of per-leg notional.

    entry notional 2*30000 = 60000 → 0.0018*60000 = 108.0
    exit  notional 2*31000 = 62000 → 0.0018*62000 = 111.6
    commission = 108.0 + 111.6 = 219.6.
    """
    t = Trade(
        instrument_class=InstrumentClass.CRYPTO,
        quantity=2,
        entry_price=30_000.0,
        exit_price=31_000.0,
        volatility=0.0,
        bar_volume=1_000,
    )
    assert apply_costs(t).commission == pytest.approx(219.6)


# --------------------------------------------------------------------------- #
# 2. Half-spread — per liquidity tier, charged on BOTH legs.
# --------------------------------------------------------------------------- #
def test_half_spread_scales_with_liquidity_tier():
    """Same trade at each tier: spread = tier_bps/1e4 * (entry_notional + exit_notional).

    notional sum = 100*100 + 100*100 = 20000 (entry=exit=100, qty=100).
    LIQUID   2 bps → 2e-4 * 20000 = 4.00
    MID      5 bps → 5e-4 * 20000 = 10.00
    ILLIQUID 15 bps → 15e-4 * 20000 = 30.00
    """
    expected = {
        LiquidityTier.LIQUID: 4.00,
        LiquidityTier.MID: 10.00,
        LiquidityTier.ILLIQUID: 30.00,
    }
    for tier, exp in expected.items():
        t = Trade(
            instrument_class=InstrumentClass.STK,
            quantity=100,
            entry_price=100.0,
            exit_price=100.0,
            liquidity_tier=tier,
            volatility=0.0,
            bar_volume=10_000_000,
        )
        assert apply_costs(t).spread_cost == pytest.approx(exp), tier


# --------------------------------------------------------------------------- #
# 3. Slippage — base + volatility term + participation term, both legs.
# --------------------------------------------------------------------------- #
def test_slippage_base_vol_and_participation_terms():
    """qty=1000, entry=exit=10 → per-leg notional 10000, notional sum 20000.

    participation = qty/bar_volume = 1000/100000 = 0.01.
    slip_frac = 1e-4 (base) + 0.10*0.02 (vol) + 0.10*0.01 (part)
              = 0.0001 + 0.0020 + 0.0010 = 0.0031.
    slippage = 0.0031 * 20000 = 62.00.
    """
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.02,
        bar_volume=100_000,
    )
    b = apply_costs(t)
    assert b.slippage_cost == pytest.approx(62.00)
    assert b.warnings == ()  # both vol and volume supplied → no warning


def test_missing_volatility_zeros_vol_term_and_warns():
    """No volatility → vol slippage term drops out; a warning is recorded.

    slip_frac = 1e-4 (base) + 0 (vol) + 0.10*0.01 (part) = 0.0011.
    slippage = 0.0011 * 20000 = 22.00.
    """
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        bar_volume=100_000,
    )
    b = apply_costs(t)
    assert b.slippage_cost == pytest.approx(22.00)
    assert any("volatility" in w for w in b.warnings)


def test_missing_volume_zeros_participation_term_and_warns():
    """No participation/volume → participation slippage term drops out; warns.

    slip_frac = 1e-4 (base) + 0.10*0.02 (vol) + 0 (part) = 0.0021.
    slippage = 0.0021 * 20000 = 42.00.
    """
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.02,
    )
    b = apply_costs(t)
    assert b.slippage_cost == pytest.approx(42.00)
    assert any("participation" in w or "volume" in w for w in b.warnings)


def test_zero_volume_bar_is_treated_as_missing_participation():
    """bar_volume == 0 (halted bar) is degenerate-but-valid → participation term 0 + warn,
    not a division-by-zero crash."""
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.0,
        bar_volume=0,
    )
    b = apply_costs(t)
    # slip = base only: 1e-4 * 20000 = 2.00
    assert b.slippage_cost == pytest.approx(2.00)
    assert any("participation" in w or "volume" in w for w in b.warnings)


def test_explicit_participation_overrides_bar_volume():
    """When both participation and bar_volume are given, participation wins."""
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.0,
        participation=0.05,      # explicit
        bar_volume=100_000,      # would imply 0.01; ignored
    )
    # slip = 1e-4 + 0.10*0.05 = 0.0001 + 0.005 = 0.0051 ; * 20000 = 102.00
    assert apply_costs(t).slippage_cost == pytest.approx(102.00)


# --------------------------------------------------------------------------- #
# 4. Market impact — OFF by default; sqrt law when enabled.
# --------------------------------------------------------------------------- #
def test_market_impact_off_by_default():
    """Default config leaves impact at exactly 0.0 and flags it disabled."""
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.0,
        bar_volume=100_000,
    )
    b = apply_costs(t)
    assert b.impact_cost == 0.0
    assert b.market_impact_enabled is False


def test_market_impact_sqrt_law_when_enabled():
    """impact_frac = impact_coeff * sqrt(participation), charged on both legs.

    participation = 1000/100000 = 0.01 → sqrt = 0.1 ; impact_frac = 0.10*0.1 = 0.01.
    impact = 0.01 * 20000 = 200.00.
    """
    cfg = CostConfig(enable_market_impact=True)
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.0,
        bar_volume=100_000,
    )
    b = apply_costs(t, cfg)
    assert b.impact_cost == pytest.approx(200.00)
    assert b.market_impact_enabled is True


def test_market_impact_zero_when_enabled_but_no_volume():
    """Impact enabled but no volume → participation 0 → impact 0.0 (+ warning)."""
    cfg = CostConfig(enable_market_impact=True)
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=1000,
        entry_price=10.0,
        exit_price=10.0,
        volatility=0.0,
    )
    b = apply_costs(t, cfg)
    assert b.impact_cost == 0.0
    assert any("participation" in w or "volume" in w for w in b.warnings)


# --------------------------------------------------------------------------- #
# 5. Totals, net PnL, config echo, determinism.
# --------------------------------------------------------------------------- #
def test_total_and_net_pnl():
    """total_cost = commission+spread+slippage+impact; net = gross - total.

    STK qty=100 @ 50/52, vol 0.02, bar_volume 1e6, gross 200.
    commission 2.00 ; spread LIQUID 2e-4*(5000+5200)=2.04 ;
    slip_frac 1e-4 + 0.10*0.02 + 0.10*(100/1e6)=0.0001+0.002+0.00001=0.00211 ;
      slippage 0.00211*(5000+5200)=0.00211*10200=21.522 ;
    total 2.00+2.04+21.522=25.562 ; net 200-25.562=174.438.
    """
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=100,
        entry_price=50.0,
        exit_price=52.0,
        volatility=0.02,
        bar_volume=1_000_000,
        gross_pnl=200.0,
    )
    b = apply_costs(t)
    assert b.commission == pytest.approx(2.00)
    assert b.spread_cost == pytest.approx(2.04)
    assert b.slippage_cost == pytest.approx(21.522)
    assert b.total_cost == pytest.approx(25.562)
    assert b.net_pnl == pytest.approx(174.438)
    assert b.total_cost >= 0.0


def test_net_pnl_none_when_no_gross_pnl():
    """No gross_pnl → net_pnl stays None (a sentinel, not a computed 0)."""
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=100,
        entry_price=50.0,
        exit_price=52.0,
        volatility=0.0,
        bar_volume=1_000_000,
    )
    b = apply_costs(t)
    assert b.gross_pnl is None
    assert b.net_pnl is None


def test_config_is_echoed_into_output():
    """The exact config that produced the numbers is embedded for reproducibility."""
    cfg = CostConfig(slippage_base_bps=3.0, enable_market_impact=True)
    t = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=100,
        entry_price=50.0,
        exit_price=50.0,
        volatility=0.0,
        bar_volume=1_000_000,
    )
    b = apply_costs(t, cfg)
    assert b.config_echo == cfg.to_dict()
    assert b.config_echo["slippage_base_bps"] == 3.0
    assert b.config_echo["enable_market_impact"] is True


def test_determinism_same_trade_same_breakdown():
    """Same inputs → identical breakdown (byte-identical dict)."""
    t = Trade(
        instrument_class=InstrumentClass.FUT,
        quantity=2,
        entry_price=4200.0,
        exit_price=4180.0,
        multiplier=5.0,
        volatility=0.015,
        bar_volume=50_000,
        gross_pnl=-200.0,
    )
    assert apply_costs(t).to_dict() == apply_costs(t).to_dict()


# --------------------------------------------------------------------------- #
# 6. ONE path for both stages — synthetic vs real-shaped fill (S4).
# --------------------------------------------------------------------------- #
def test_identical_costs_synthetic_vs_real_shaped_fill():
    """A synthetic Stage-1 Trade and a Stage-2 real-fill mapping with the same
    numbers must produce byte-identical cost breakdowns — proving the single
    ``apply_costs`` haircut is applied to both stages (SSOT §3.5 / S4)."""
    synthetic = Trade(
        instrument_class=InstrumentClass.STK,
        quantity=300,
        entry_price=45.0,
        exit_price=44.0,
        liquidity_tier=LiquidityTier.MID,
        volatility=0.018,
        bar_volume=250_000,
        gross_pnl=-300.0,
        label="alpha_intraday",
    )
    real_fill = {
        "instrument": "STK",             # string coerced to enum
        "qty": 300,
        "entry_price": 45.0,
        "exit_price": 44.0,
        "tier": "MID",
        "realized_vol": 0.018,
        "volume": 250_000,
        "pnl": -300.0,
        "strategy": "alpha_intraday",
        "irrelevant_broker_field": "ignored",
    }
    from_real = Trade.from_fill(real_fill)
    assert from_real == synthetic
    assert apply_costs(synthetic).to_dict() == apply_costs(from_real).to_dict()


# --------------------------------------------------------------------------- #
# 7. Pessimistic intrabar resolution (V2).
# --------------------------------------------------------------------------- #
def test_intrabar_both_in_range_returns_stop_and_flags_ambiguous():
    """bar [90,110] contains both stop 95 and target 105 → pessimistic STOP + flag."""
    r = resolve_intrabar({"high": 110.0, "low": 90.0}, stop=95.0, target=105.0)
    assert isinstance(r, IntrabarResolution)
    assert r.outcome == "stop"
    assert r.fill_price == pytest.approx(95.0)
    assert r.ambiguous is True
    assert r.stop_in_range and r.target_in_range


def test_intrabar_only_stop_in_range():
    """Target above the bar high → only stop hit, unambiguous."""
    r = resolve_intrabar({"high": 100.0, "low": 90.0}, stop=95.0, target=105.0)
    assert r.outcome == "stop"
    assert r.fill_price == pytest.approx(95.0)
    assert r.ambiguous is False


def test_intrabar_only_target_in_range():
    """Stop below the bar low → only target hit, unambiguous."""
    r = resolve_intrabar({"high": 110.0, "low": 100.0}, stop=95.0, target=105.0)
    assert r.outcome == "target"
    assert r.fill_price == pytest.approx(105.0)
    assert r.ambiguous is False


def test_intrabar_neither_in_range_is_no_fill():
    """Both levels outside the bar → no fill this bar (outcome 'none')."""
    r = resolve_intrabar({"high": 101.0, "low": 99.0}, stop=95.0, target=105.0)
    assert r.outcome == "none"
    assert r.fill_price is None
    assert r.ambiguous is False


def test_intrabar_is_direction_agnostic_for_shorts():
    """For a short (target below entry, stop above), both-in-range still → STOP.

    Pessimism is symmetric: the rule only asks which levels the bar traversed.
    bar [40,60] contains stop 55 and target 45 → stop (worse) + ambiguous.
    """
    r = resolve_intrabar({"high": 60.0, "low": 40.0}, stop=55.0, target=45.0)
    assert r.outcome == "stop"
    assert r.fill_price == pytest.approx(55.0)
    assert r.ambiguous is True


def test_intrabar_boundary_touch_is_inclusive():
    """A level exactly on the bar high/low counts as in-range (inclusive)."""
    r = resolve_intrabar({"high": 105.0, "low": 95.0}, stop=95.0, target=105.0)
    assert r.stop_in_range and r.target_in_range
    assert r.outcome == "stop" and r.ambiguous is True


# --------------------------------------------------------------------------- #
# 8. Fail-fast on invalid config / malformed input (never silent).
# --------------------------------------------------------------------------- #
def test_negative_config_parameter_raises():
    with pytest.raises(ValueError):
        CostConfig(stk_commission_per_share=-0.01)
    with pytest.raises(ValueError):
        CostConfig(half_spread_bps_liquid=-1.0)
    with pytest.raises(ValueError):
        CostConfig(slippage_vol_coeff=-0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantity": 0},
        {"quantity": -10},
        {"entry_price": 0.0},
        {"entry_price": -5.0},
        {"exit_price": -1.0},
        {"multiplier": 0.0},
        {"volatility": -0.01},
        {"participation": -0.1},
        {"bar_volume": -100},
    ],
)
def test_malformed_trade_raises(kwargs):
    """Structurally-invalid trade input is a caller bug → ValueError at construction."""
    base = dict(
        instrument_class=InstrumentClass.STK,
        quantity=100,
        entry_price=50.0,
        exit_price=52.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        Trade(**base)


def test_intrabar_malformed_bar_raises():
    """high < low, or missing/non-numeric high/low, is a data-integrity bug → raise."""
    with pytest.raises(ValueError):
        resolve_intrabar({"high": 90.0, "low": 110.0}, stop=95.0, target=105.0)
    with pytest.raises(ValueError):
        resolve_intrabar({"high": 110.0}, stop=95.0, target=105.0)  # missing low
    with pytest.raises(ValueError):
        resolve_intrabar({"high": "x", "low": 90.0}, stop=95.0, target=105.0)


def test_apply_costs_rejects_non_trade():
    with pytest.raises(ValueError):
        apply_costs({"not": "a trade"})  # type: ignore[arg-type]

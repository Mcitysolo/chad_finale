"""U3 CRYPTO-TRUST — lunch-money sizing: min-size bump / afford / skip.

Test matrix: $185 paper account x SCR 0.1 sizing across BTC/ETH/SOL, proving
each of the three paths fires. Pure decision function; no feed/broker.
"""

from __future__ import annotations

from chad.execution.kraken_min_size import (
    ACTION_BUMP,
    ACTION_PASS,
    ACTION_SKIP,
    MARKER_BELOW_MIN_SKIP,
    MARKER_MIN_SIZE_BUMP,
    decide_min_size,
)
from chad.execution.kraken_trading_config import load_kraken_trading_config

_CFG = load_kraken_trading_config()
_ACCOUNT_USD = 185.0
_SCR = 0.1

# Representative live-ish prices for the deterministic matrix.
_PRICE = {"XBTUSD": 64000.0, "ETHUSD": 3400.0, "SOLUSD": 170.0}


def _computed_volume(target_notional_usd: float, price: float) -> float:
    """Strategy target notional attenuated by SCR sizing_factor, as a base qty."""
    return (target_notional_usd * _SCR) / price


def test_sol_below_min_but_affordable_bumps():
    pair = "SOLUSD"
    price = _PRICE[pair]
    cv = _computed_volume(50.0, price)          # $50 * 0.1 = $5 -> 0.0294 SOL
    mn = _CFG.min_volume(pair)                  # 0.05
    assert cv < mn
    d = decide_min_size(pair=pair, computed_volume=cv, price=price, min_volume=mn,
                        available_notional=_ACCOUNT_USD, risk_cap_notional=50.0)
    assert d.action == ACTION_BUMP
    assert d.marker == MARKER_MIN_SIZE_BUMP
    assert d.final_volume == mn
    assert d.min_notional == mn * price  # $8.5, affordable within $185 + $50 cap


def test_btc_at_or_above_min_passes():
    pair = "XBTUSD"
    price = _PRICE[pair]
    cv = _computed_volume(200.0, price)         # $200 * 0.1 = $20 -> 0.0003125 BTC
    mn = _CFG.min_volume(pair)                  # 0.0001
    assert cv >= mn
    d = decide_min_size(pair=pair, computed_volume=cv, price=price, min_volume=mn,
                        available_notional=_ACCOUNT_USD, risk_cap_notional=50.0)
    assert d.action == ACTION_PASS
    assert d.marker is None
    assert d.final_volume == cv


def test_eth_below_min_and_cap_binds_skips():
    pair = "ETHUSD"
    price = _PRICE[pair]
    cv = _computed_volume(30.0, price)          # $30 * 0.1 = $3 -> 0.000882 ETH
    mn = _CFG.min_volume(pair)                  # 0.001 -> min_notional $3.40
    assert cv < mn
    # per-strategy risk cap only $2 -> cannot afford the $3.40 minimum -> SKIP
    d = decide_min_size(pair=pair, computed_volume=cv, price=price, min_volume=mn,
                        available_notional=_ACCOUNT_USD, risk_cap_notional=2.0)
    assert d.action == ACTION_SKIP
    assert d.marker == MARKER_BELOW_MIN_SKIP
    assert d.final_volume == 0.0


def test_skip_when_account_balance_cannot_afford_min():
    pair = "SOLUSD"
    price = _PRICE[pair]
    cv = _computed_volume(50.0, price)
    mn = _CFG.min_volume(pair)                  # min_notional $8.5
    d = decide_min_size(pair=pair, computed_volume=cv, price=price, min_volume=mn,
                        available_notional=5.0, risk_cap_notional=None)  # only $5 in wallet
    assert d.action == ACTION_SKIP
    assert d.marker == MARKER_BELOW_MIN_SKIP


def test_unconstrained_below_min_bumps():
    d = decide_min_size(pair="SOLUSD", computed_volume=0.01, price=170.0, min_volume=0.05,
                        available_notional=None, risk_cap_notional=None)
    assert d.action == ACTION_BUMP


def test_zero_min_passes_through():
    d = decide_min_size(pair="ZZZ", computed_volume=0.01, price=170.0, min_volume=0.0,
                        available_notional=None, risk_cap_notional=None)
    assert d.action == ACTION_PASS
    assert d.final_volume == 0.01

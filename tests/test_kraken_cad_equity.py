"""Kraken native-CAD equity valuation tests.

Root-cause fix: the Kraken account is natively CAD (kraken_balances.json
balances.CAD + a small crypto sliver). CHAD previously valued it in USD via a
0.73 CAD-per-USD fallback guess, then tried to convert back through a dark
USDCAD feed — a wrong value in an unconfirmable currency feeding risk/sizing.

These tests pin the native-CAD path end to end:
  - provider._value_balances_in_cad / KrakenBalanceSnapshot.to_json
  - publisher._read_kraken_cad_equity / _apply_kraken_cad_leg

All fixtures only — no network, no broker calls. stdlib + pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT
from chad.market_data.kraken_balance_provider import (
    KrakenBalanceProvider,
    KrakenBalanceSnapshot,
)


# ---------------------------------------------------------------------------
# (a)-(c) provider CAD valuation
# ---------------------------------------------------------------------------
def test_a_cad_cash_plus_crypto_sliver_valued_in_cad() -> None:
    """balances {CAD, BTC} + BTC-USD price => native CAD ≈ 357 (screenshot)."""
    provider = KrakenBalanceProvider()
    balances = {"CAD": 252.85, "BTC": 0.0012}
    prices = {"BTC-USD": 61253.6}

    cad = provider._value_balances_in_cad(balances, prices)

    # Exact, DRY against the canonical constant: CAD cash 1:1 + crypto*price*const.
    expected = 252.85 + 0.0012 * 61253.6 * USDCAD_CONVERSION_CONSTANT
    assert abs(cad - expected) < 1e-9
    # Within tolerance of the 357.33 screenshot value (computed ≈356.93).
    assert abs(cad - 357.33) <= 1.0
    assert 355.0 < cad < 359.0


def test_b_pure_cad_balance_is_one_to_one_no_conversion() -> None:
    """A pure CAD balance is valued exactly 1:1 — no FX applied."""
    provider = KrakenBalanceProvider()
    cad = provider._value_balances_in_cad({"CAD": 252.85}, {})
    assert cad == 252.85


def test_c_missing_crypto_price_skips_crypto_keeps_cad_cash() -> None:
    """No BTC price => crypto leg skipped, CAD cash still counted natively."""
    provider = KrakenBalanceProvider()
    cad = provider._value_balances_in_cad({"CAD": 252.85, "BTC": 0.0012}, {})
    assert cad == 252.85  # only the CAD cash, crypto sliver dropped (unpriced)


# ---------------------------------------------------------------------------
# (d) snapshot carries BOTH fields
# ---------------------------------------------------------------------------
def test_d_snapshot_to_json_carries_cad_and_usd_equivalent() -> None:
    """to_json() exposes both cad_equivalent (new) and usd_equivalent (back-compat)."""
    snap = KrakenBalanceSnapshot(
        ts_utc="2026-07-02T00:00:00Z",
        ok=True,
        balances={"CAD": 252.85, "BTC": 0.0012},
        raw={"ZCAD": "252.85", "XXBT": "0.0012"},
        usd_equivalent=184.58,
        cad_equivalent=356.93,
    )
    j = snap.to_json()
    assert "cad_equivalent" in j and "usd_equivalent" in j
    assert j["cad_equivalent"] == 356.93
    assert j["usd_equivalent"] == 184.58


# ---------------------------------------------------------------------------
# (e)-(f) publisher native-CAD leg (no usdcad, honest currency_ok)
# ---------------------------------------------------------------------------
def _write_snapshot(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_e_publisher_reads_native_cad_no_usdcad_needed(tmp_path: Path) -> None:
    """Given a snapshot with cad_equivalent, the leg is set to it, currency_ok
    True, with NO usdcad multiply and NO dead-feed dependency."""
    import chad.ops.portfolio_snapshot_publisher as pub

    snap = _write_snapshot(
        tmp_path / "kraken_balances.json",
        {"ok": True, "balances": {"CAD": 252.85, "BTC": 0.0012}, "cad_equivalent": 357.33},
    )
    payload: dict = {}
    pub._apply_kraken_cad_leg(payload, path=snap)

    assert payload["kraken_equity"] == 357.33  # native cad_equivalent, verbatim
    assert payload["kraken_equity_currency"] == "CAD"
    assert payload["kraken_equity_currency_ok"] is True
    # No usdcad key was ever introduced by the leg — the value IS native CAD.
    assert "usdcad_rate_used" not in payload

    # Direct reader returns the same native CAD value.
    assert pub._read_kraken_cad_equity(snap) == 357.33


def test_f_publisher_fail_closed_when_no_cad_and_no_balance(tmp_path: Path) -> None:
    """cad_equivalent absent AND no reconstructable balance => currency_ok False,
    prior kraken_equity preserved (fail-closed, never mis-tagged)."""
    import chad.ops.portfolio_snapshot_publisher as pub

    snap = _write_snapshot(
        tmp_path / "kraken_balances.json",
        {"ok": True, "balances": {}},  # no cad_equivalent, empty balances
    )
    payload: dict = {"kraken_equity": 999.0}  # prior value to preserve
    pub._apply_kraken_cad_leg(payload, path=snap)

    assert payload["kraken_equity"] == 999.0  # preserved, NOT overwritten
    assert payload["kraken_equity_currency_ok"] is False
    assert pub._read_kraken_cad_equity(snap) is None


def test_g_publisher_fallback_reconstructs_native_cad_from_balances(tmp_path: Path) -> None:
    """Bonus: cad_equivalent absent but balances present => reconstruct native
    CAD (cash 1:1, unpriced crypto skipped), currency_ok True — still no FX."""
    import chad.ops.portfolio_snapshot_publisher as pub

    snap = _write_snapshot(
        tmp_path / "kraken_balances.json",
        {"ok": True, "balances": {"CAD": 252.85, "BTC": 0.0012}},  # no cad_equivalent
    )
    payload: dict = {}
    pub._apply_kraken_cad_leg(payload, path=snap)

    assert payload["kraken_equity"] == 252.85  # CAD cash only (crypto unpriced)
    assert payload["kraken_equity_currency_ok"] is True

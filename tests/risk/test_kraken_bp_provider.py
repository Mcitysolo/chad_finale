"""Tests for chad/risk/kraken_bp_provider.py (Margin BLOCK Phase A, crypto lane).

Fixtures only — no network, no broker. Exercises: a realistic
kraken_balances.json-shaped fixture parses to native-CAD capacity; crypto is
valued via the sanctioned FX constant (DRY, not retyped) only when priced and
otherwise conservatively dropped; every fail-closed path (unreadable / missing
balances / unparseable / negative / no-capture-time / stale / not-a-mapping /
missing or malformed file); the cache re-checks freshness; and the provider
exposes NO order method and makes NO allow/block decision.
"""

from __future__ import annotations

import json

import pytest

from chad.constants.fx import USDCAD_CONVERSION_CONSTANT
from chad.risk.kraken_bp_provider import (
    ACCOUNT_CURRENCY,
    DEFAULT_FRESHNESS_TTL_SECONDS,
    KrakenBPReason,
    KrakenBuyingPowerProvider,
    KrakenBuyingPowerSnapshot,
    parse_kraken_balances,
)

# Real kraken_balances.json shape (verified against runtime/kraken_balances.json).
_TS = "2026-07-06T11:43:36Z"


def _payload(**overrides):
    base = {
        "ts_utc": _TS,
        "ok": True,
        "balances": {"BTC": 0.0012, "CAD": 252.8538},
        "raw": {"XXBT": "0.0012000000", "ZCAD": "252.8538"},
        "usd_equivalent": 184.583274,
        "cad_equivalent": 252.8538,
        "error": None,
    }
    base.update(overrides)
    return base


def _epoch(ts=_TS):
    # Derive the fixture's own capture epoch so tests can pin "fresh" precisely.
    from chad.risk.kraken_bp_provider import _iso_to_epoch

    e = _iso_to_epoch(ts)
    assert e is not None
    return e


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------
def test_realistic_unpriced_snapshot_counts_cad_cash_only():
    """No prices supplied → crypto sliver dropped (conservative); CAD cash native."""
    now = _epoch() + 5.0
    snap = parse_kraken_balances(_payload(), now_epoch=now, ttl_seconds=3600.0)
    assert snap.usable is True
    assert snap.reason == KrakenBPReason.OK
    assert snap.currency == ACCOUNT_CURRENCY == "CAD"
    assert snap.cad_cash == pytest.approx(252.8538)
    assert snap.crypto_cad == pytest.approx(0.0)
    assert snap.available_cad == pytest.approx(252.8538)
    assert snap.unpriced_symbols == ("BTC",)


def test_priced_crypto_valued_via_fx_constant_dry():
    """Priced crypto valued as qty*price*USDCAD_CONVERSION_CONSTANT (imported, not retyped)."""
    now = _epoch() + 5.0
    prices = {"BTC-USD": 61253.6}
    snap = parse_kraken_balances(
        _payload(), now_epoch=now, ttl_seconds=3600.0, prices_usd=prices
    )
    assert snap.usable is True
    expected_crypto = 0.0012 * 61253.6 * USDCAD_CONVERSION_CONSTANT
    assert snap.crypto_cad == pytest.approx(expected_crypto)
    assert snap.available_cad == pytest.approx(252.8538 + expected_crypto)
    assert snap.unpriced_symbols == ()


def test_pure_cad_cash_is_one_to_one():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(balances={"CAD": 252.8538}), now_epoch=now, ttl_seconds=3600.0
    )
    assert snap.usable is True
    assert snap.available_cad == pytest.approx(252.8538)
    assert snap.crypto_cad == pytest.approx(0.0)
    assert snap.unpriced_symbols == ()


def test_zero_price_is_treated_as_unpriced():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(), now_epoch=now, ttl_seconds=3600.0, prices_usd={"BTC-USD": 0.0}
    )
    assert snap.usable is True
    assert snap.crypto_cad == pytest.approx(0.0)
    assert snap.unpriced_symbols == ("BTC",)


# ---------------------------------------------------------------------------
# fail-closed paths
# ---------------------------------------------------------------------------
def test_ok_false_fails_closed():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(_payload(ok=False), now_epoch=now, ttl_seconds=3600.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNREADABLE
    assert snap.available_cad is None


def test_error_set_fails_closed():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(error="EAPI:Rate limit exceeded"), now_epoch=now, ttl_seconds=3600.0
    )
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNREADABLE


def test_missing_balances_fails_closed():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(_payload(balances={}), now_epoch=now, ttl_seconds=3600.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.MISSING_BALANCES


def test_balances_wrong_type_fails_closed():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(balances=["BTC", 0.0012]), now_epoch=now, ttl_seconds=3600.0
    )
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.MISSING_BALANCES


def test_unparseable_balance_fails_closed():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(balances={"CAD": "not-a-number"}), now_epoch=now, ttl_seconds=3600.0
    )
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNPARSEABLE


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_non_finite_balance_fails_closed(nonfinite):
    """A nan/+inf balance must fail closed (UNPARSEABLE), not sum into a usable
    non-finite available_cad. -inf is caught earlier by the negative guard."""
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(balances={"CAD": nonfinite}), now_epoch=now, ttl_seconds=3600.0
    )
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNPARSEABLE
    assert snap.available_cad is None


def test_load_from_file_with_json_nan_fails_closed(tmp_path):
    """json.loads accepts bare NaN by default; the file-load path must fail
    closed rather than surface a non-finite available_cad as usable."""
    f = tmp_path / "kraken_balances.json"
    f.write_text(
        '{"ts_utc":"2026-07-06T11:43:36Z","ok":true,"error":null,'
        '"balances":{"CAD":NaN},"usd_equivalent":0,"cad_equivalent":0}',
        encoding="utf-8",
    )
    p = KrakenBuyingPowerProvider(ttl_seconds=3600.0)
    snap = p.load_from_file(f, now_epoch=_epoch() + 5.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNPARSEABLE


def test_non_finite_price_treated_as_unpriced():
    """A +inf crypto price must not poison crypto_cad — dropped as unpriced."""
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(), now_epoch=now, ttl_seconds=3600.0, prices_usd={"BTC-USD": float("inf")}
    )
    assert snap.usable is True
    assert snap.crypto_cad == pytest.approx(0.0)
    assert snap.unpriced_symbols == ("BTC",)


def test_negative_balance_fails_closed():
    now = _epoch() + 1.0
    snap = parse_kraken_balances(
        _payload(balances={"CAD": -10.0}), now_epoch=now, ttl_seconds=3600.0
    )
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.NEGATIVE_BALANCE


def test_stale_snapshot_fails_closed():
    now = _epoch() + DEFAULT_FRESHNESS_TTL_SECONDS + 0.001
    snap = parse_kraken_balances(_payload(), now_epoch=now)  # default 30s TTL
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.STALE
    assert snap.available_cad is None


def test_future_capture_time_fails_closed():
    now = _epoch() - 100.0
    snap = parse_kraken_balances(_payload(), now_epoch=now, ttl_seconds=3600.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.STALE


@pytest.mark.parametrize("bad_ts", [None, "", "not-a-timestamp", 12345])
def test_bad_ts_fails_closed(bad_ts):
    snap = parse_kraken_balances(
        _payload(ts_utc=bad_ts), now_epoch=1_800_000_000.0, ttl_seconds=3600.0
    )
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.NO_CAPTURE_TIME


def test_not_a_mapping_fails_closed():
    snap = parse_kraken_balances("garbage", now_epoch=1000.0, ttl_seconds=3600.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.NOT_A_MAPPING


def test_bad_ttl_fails_closed():
    snap = parse_kraken_balances(_payload(), now_epoch=_epoch(), ttl_seconds=0.0)
    assert snap.usable is False
    # Harmonized with the IBKR provider's BAD_TTL code (same error, same reason).
    assert snap.reason == KrakenBPReason.BAD_TTL


# ---------------------------------------------------------------------------
# provider cache + file load
# ---------------------------------------------------------------------------
def test_provider_cache_and_freshness_recheck():
    p = KrakenBuyingPowerProvider(ttl_seconds=30.0)
    cap = _epoch()
    snap = p.update(_payload(), now_epoch=cap + 5.0)
    assert snap.usable is True
    fresh = p.get(now_epoch=cap + 20.0)
    assert fresh.usable is True
    assert fresh.age_seconds == pytest.approx(20.0)
    stale = p.get(now_epoch=cap + 31.0)
    assert stale.usable is False
    assert stale.reason == KrakenBPReason.STALE


def test_provider_get_without_data_fails_closed():
    p = KrakenBuyingPowerProvider()
    snap = p.get(now_epoch=1000.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.NO_DATA


def test_provider_does_not_cache_fail_closed():
    p = KrakenBuyingPowerProvider()
    bad = p.update(_payload(ok=False), now_epoch=_epoch())
    assert bad.usable is False
    assert p.get(now_epoch=_epoch()).reason == KrakenBPReason.NO_DATA


def test_load_from_missing_file_fails_closed(tmp_path):
    p = KrakenBuyingPowerProvider()
    snap = p.load_from_file(tmp_path / "does_not_exist.json", now_epoch=1000.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNREADABLE


def test_load_from_malformed_json_fails_closed(tmp_path):
    bad = tmp_path / "kraken_balances.json"
    bad.write_text("{not json", encoding="utf-8")
    p = KrakenBuyingPowerProvider()
    snap = p.load_from_file(bad, now_epoch=1000.0)
    assert snap.usable is False
    assert snap.reason == KrakenBPReason.UNREADABLE


def test_load_from_good_file_parses(tmp_path):
    good = tmp_path / "kraken_balances.json"
    good.write_text(json.dumps(_payload()), encoding="utf-8")
    p = KrakenBuyingPowerProvider(ttl_seconds=3600.0)
    snap = p.load_from_file(good, now_epoch=_epoch() + 5.0)
    assert snap.usable is True
    assert snap.available_cad == pytest.approx(252.8538)


# ---------------------------------------------------------------------------
# invariants: NO order method, NO allow/block decision
# ---------------------------------------------------------------------------
_FORBIDDEN = ("place_order", "placeorder", "submit", "cancel", "allow", "block", "decide", "order")


def _public_names(obj):
    return [n for n in dir(obj) if not n.startswith("_")]


def test_provider_exposes_no_order_or_decision_method():
    for name in _public_names(KrakenBuyingPowerProvider):
        low = name.lower()
        assert all(tok not in low for tok in _FORBIDDEN), name
    for attr in ("place_order", "placeOrder", "submit", "cancel", "allow", "block", "decide"):
        assert not hasattr(KrakenBuyingPowerProvider, attr)


def test_snapshot_exposes_no_order_or_decision_method():
    for name in _public_names(KrakenBuyingPowerSnapshot):
        low = name.lower()
        assert all(tok not in low for tok in _FORBIDDEN), name


def test_module_does_not_import_live_or_execution_paths():
    import chad.risk.kraken_bp_provider as mod

    with open(mod.__file__, "r", encoding="utf-8") as fh:
        text = fh.read()
    for banned in ("live_loop", "execution", "ibkr_adapter", "placeOrder", "place_order", "orchestrator"):
        assert banned not in text, banned

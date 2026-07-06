"""Tests for chad/risk/buying_power_provider.py (Margin BLOCK Phase A, IBKR lane).

Fixtures only — no network, no broker. Exercises: every field parsed from a
realistic accountSummary fixture; every fail-closed path (missing / unparseable
/ absent-, wrong-, mixed-currency / ambiguous / no-capture-time / stale / empty
/ bad-ttl); a legitimately-zero margin field stays usable (not a silent-zero
false alarm); the cache re-checks freshness; and the provider exposes NO order
method and makes NO allow/block decision.
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from chad.risk.buying_power_provider import (
    BPReason,
    BuyingPowerProvider,
    BuyingPowerSnapshot,
    DEFAULT_EXPECTED_CURRENCY,
    DEFAULT_FRESHNESS_TTL_SECONDS,
    REQUIRED_TAGS,
    parse_account_summary,
)

# Mirror the real ib_async row shape: AccountValue(account, tag, value, currency,
# modelCode) — exactly what chad/portfolio/ibkr_portfolio_collector_v2.py reads.
AccountValue = namedtuple("AccountValue", "account tag value currency modelCode")

_ACCT = "DU1234567"

# Realistic CAD paper-account values (magnitudes near the real snapshot's
# NetLiquidation ~1,000,265 CAD).
_REALISTIC = {
    "NetLiquidation": "1000265.14",
    "BuyingPower": "3999999.55",
    "ExcessLiquidity": "980210.00",
    "AvailableFunds": "500123.45",
    "FullInitMarginReq": "20055.14",
    "FullMaintMarginReq": "18000.00",
}


def _rows(values, *, currency="CAD", account=_ACCT):
    """Build accountSummary-shaped rows from a {tag: value_str} mapping."""
    return [
        AccountValue(account=account, tag=tag, value=val, currency=currency, modelCode="")
        for tag, val in values.items()
    ]


def _fresh(rows, **kw):
    """Parse rows at a fixed, fresh (captured==now) evaluation time."""
    kw.setdefault("captured_at_epoch", 1000.0)
    kw.setdefault("now_epoch", 1000.0)
    return parse_account_summary(rows, **kw)


# ---------------------------------------------------------------------------
# happy path — every field parsed, tagged, usable
# ---------------------------------------------------------------------------
def test_complete_fresh_snapshot_is_usable_all_fields_parsed():
    snap = _fresh(_rows(_REALISTIC))
    assert snap.usable is True
    assert snap.reason == BPReason.OK
    assert snap.currency == "CAD"
    assert snap.net_liquidation == pytest.approx(1000265.14)
    assert snap.buying_power == pytest.approx(3999999.55)
    assert snap.excess_liquidity == pytest.approx(980210.00)
    assert snap.available_funds == pytest.approx(500123.45)
    assert snap.full_init_margin_req == pytest.approx(20055.14)
    assert snap.full_maint_margin_req == pytest.approx(18000.00)
    assert snap.age_seconds == pytest.approx(0.0)
    assert snap.is_fail_closed is False


def test_required_tags_are_the_six_design_fields():
    assert set(REQUIRED_TAGS) == {
        "NetLiquidation",
        "BuyingPower",
        "ExcessLiquidity",
        "AvailableFunds",
        "FullInitMarginReq",
        "FullMaintMarginReq",
    }


def test_dict_rows_parse_identically_to_object_rows():
    dict_rows = [
        {"account": _ACCT, "tag": t, "value": v, "currency": "CAD"}
        for t, v in _REALISTIC.items()
    ]
    snap = _fresh(dict_rows)
    assert snap.usable is True
    assert snap.net_liquidation == pytest.approx(1000265.14)


def test_legitimate_zero_margin_is_usable_not_silent_zero():
    """A flat account reports FullInitMarginReq == 0 — a real value, not missing."""
    vals = dict(_REALISTIC, FullInitMarginReq="0", FullMaintMarginReq="0.00")
    snap = _fresh(_rows(vals))
    assert snap.usable is True
    assert snap.full_init_margin_req == 0.0
    assert snap.full_maint_margin_req == 0.0


# ---------------------------------------------------------------------------
# fail-closed: structural
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("drop", list(_REALISTIC.keys()))
def test_missing_any_field_fails_closed(drop):
    vals = {k: v for k, v in _REALISTIC.items() if k != drop}
    snap = _fresh(_rows(vals))
    assert snap.usable is False
    assert snap.reason == BPReason.MISSING_FIELD
    assert drop in snap.detail
    # No silent zero anywhere.
    assert snap.net_liquidation is None
    assert snap.buying_power is None


@pytest.mark.parametrize("bad", ["", "  ", "n/a", "abc", "1,000", None])
def test_unparseable_value_fails_closed(bad):
    vals = dict(_REALISTIC, BuyingPower=bad)
    snap = _fresh(_rows(vals))
    assert snap.usable is False
    # An empty/whitespace/None value is UNPARSEABLE, never a silent zero.
    assert snap.reason == BPReason.UNPARSEABLE
    assert snap.buying_power is None


@pytest.mark.parametrize("nonfinite", ["nan", "NaN", "inf", "+inf", "Infinity", "-inf", "-Infinity"])
def test_non_finite_value_fails_closed(nonfinite):
    """float() accepts 'nan'/'inf'/'Infinity' — a non-finite money field must
    fail closed (UNPARSEABLE), never surface a usable nan/inf figure."""
    vals = dict(_REALISTIC, ExcessLiquidity=nonfinite)
    snap = _fresh(_rows(vals))
    assert snap.usable is False
    assert snap.reason == BPReason.UNPARSEABLE
    assert snap.excess_liquidity is None


def test_empty_input_fails_closed():
    snap = _fresh([])
    assert snap.usable is False
    assert snap.reason == BPReason.EMPTY_INPUT


def test_ambiguous_duplicate_tag_fails_closed():
    """Same tag reported for two accounts with different values, no filter."""
    rows = _rows(_REALISTIC, account="ACCT_A")
    rows += [AccountValue("ACCT_B", "NetLiquidation", "42.0", "CAD", "")]
    snap = _fresh(rows)
    assert snap.usable is False
    assert snap.reason == BPReason.AMBIGUOUS_FIELD
    assert "NetLiquidation" in snap.detail


def test_ambiguous_same_value_different_currency_fails_closed():
    """Same tag, same magnitude, different currency across accounts must be
    AMBIGUOUS (dedup keys on (value, currency), not value alone) — otherwise the
    second row's currency is silently dropped, a currency-provenance hole."""
    rows = _rows(_REALISTIC, account="ACCT_A")
    rows += [
        AccountValue("ACCT_B", "NetLiquidation", _REALISTIC["NetLiquidation"], "USD", "")
    ]
    snap = _fresh(rows)
    assert snap.usable is False
    assert snap.reason == BPReason.AMBIGUOUS_FIELD
    assert "NetLiquidation" in snap.detail


def test_duplicate_identical_row_is_not_ambiguous():
    """A harmless exact-duplicate row (same value AND currency) stays usable."""
    rows = _rows(_REALISTIC, account="ACCT_A")
    rows += [
        AccountValue("ACCT_A", "NetLiquidation", _REALISTIC["NetLiquidation"], "CAD", "")
    ]
    snap = _fresh(rows, account="ACCT_A")
    assert snap.usable is True
    assert snap.net_liquidation == pytest.approx(1000265.14)


def test_account_filter_selects_the_right_account():
    rows = _rows(_REALISTIC, account="ACCT_A")
    # Noise rows for a different account must be ignored under the filter.
    rows += _rows({"NetLiquidation": "1.0"}, account="ACCT_B")
    snap = _fresh(rows, account="ACCT_A")
    assert snap.usable is True
    assert snap.net_liquidation == pytest.approx(1000265.14)


# ---------------------------------------------------------------------------
# fail-closed: currency
# ---------------------------------------------------------------------------
def test_wrong_currency_all_usd_fails_closed():
    snap = _fresh(_rows(_REALISTIC, currency="USD"))  # expected default CAD
    assert snap.usable is False
    assert snap.reason == BPReason.WRONG_CURRENCY
    assert "USD" in snap.detail


def test_absent_currency_fails_closed():
    rows = _rows(_REALISTIC)
    rows[2] = rows[2]._replace(currency="")  # blank out one row's currency
    snap = _fresh(rows)
    assert snap.usable is False
    assert snap.reason == BPReason.ABSENT_CURRENCY


def test_mixed_currency_fails_closed():
    rows = _rows(_REALISTIC)
    rows[1] = rows[1]._replace(currency="USD")  # one USD among CAD
    snap = _fresh(rows)
    assert snap.usable is False
    assert snap.reason == BPReason.MIXED_CURRENCY


def test_expected_currency_none_accepts_consistent_foreign_currency():
    """With expected_currency=None the parser only requires internal agreement."""
    snap = _fresh(_rows(_REALISTIC, currency="USD"), expected_currency=None)
    assert snap.usable is True
    assert snap.currency == "USD"


# ---------------------------------------------------------------------------
# fail-closed: freshness
# ---------------------------------------------------------------------------
def test_stale_snapshot_fails_closed():
    snap = parse_account_summary(
        _rows(_REALISTIC),
        captured_at_epoch=1000.0,
        now_epoch=1000.0 + DEFAULT_FRESHNESS_TTL_SECONDS + 0.001,
        ttl_seconds=DEFAULT_FRESHNESS_TTL_SECONDS,
    )
    assert snap.usable is False
    assert snap.reason == BPReason.STALE
    # currency still surfaced for diagnostics, but no numeric values.
    assert snap.currency == "CAD"
    assert snap.net_liquidation is None


def test_exactly_at_ttl_is_still_fresh():
    snap = parse_account_summary(
        _rows(_REALISTIC),
        captured_at_epoch=1000.0,
        now_epoch=1000.0 + DEFAULT_FRESHNESS_TTL_SECONDS,
        ttl_seconds=DEFAULT_FRESHNESS_TTL_SECONDS,
    )
    assert snap.usable is True


def test_future_capture_time_fails_closed():
    snap = parse_account_summary(
        _rows(_REALISTIC), captured_at_epoch=2000.0, now_epoch=1000.0
    )
    assert snap.usable is False
    assert snap.reason == BPReason.STALE


def test_no_capture_time_fails_closed():
    snap = parse_account_summary(
        _rows(_REALISTIC), captured_at_epoch=None, now_epoch=1000.0
    )
    assert snap.usable is False
    assert snap.reason == BPReason.NO_CAPTURE_TIME


def test_bad_ttl_fails_closed():
    snap = parse_account_summary(
        _rows(_REALISTIC), captured_at_epoch=1000.0, now_epoch=1000.0, ttl_seconds=0.0
    )
    assert snap.usable is False
    assert snap.reason == BPReason.BAD_TTL


# ---------------------------------------------------------------------------
# provider cache + freshness re-check
# ---------------------------------------------------------------------------
def test_provider_caches_usable_and_recheck_freshness():
    p = BuyingPowerProvider(ttl_seconds=30.0)
    snap = p.update(_rows(_REALISTIC), captured_at_epoch=1000.0, now_epoch=1000.0)
    assert snap.usable is True
    # Read back while fresh.
    fresh = p.get(now_epoch=1020.0)
    assert fresh.usable is True
    assert fresh.age_seconds == pytest.approx(20.0)
    # Read back after TTL — cached snapshot is now stale, must fail closed.
    stale = p.get(now_epoch=1000.0 + 31.0)
    assert stale.usable is False
    assert stale.reason == BPReason.STALE


def test_provider_get_without_data_fails_closed():
    p = BuyingPowerProvider()
    snap = p.get(now_epoch=1000.0)
    assert snap.usable is False
    assert snap.reason == BPReason.NO_DATA


def test_provider_does_not_cache_fail_closed():
    p = BuyingPowerProvider()
    # A fail-closed update must not become the cache.
    bad = p.update(_rows({"NetLiquidation": "1"}), captured_at_epoch=1000.0, now_epoch=1000.0)
    assert bad.usable is False
    assert p.get(now_epoch=1000.0).reason == BPReason.NO_DATA


def test_default_expected_currency_is_cad():
    assert DEFAULT_EXPECTED_CURRENCY == "CAD"


# ---------------------------------------------------------------------------
# invariants: NO order method, NO allow/block decision
# ---------------------------------------------------------------------------
_FORBIDDEN = ("place_order", "placeorder", "submit", "cancel", "allow", "block", "decide", "order")


def _public_names(obj):
    return [n for n in dir(obj) if not n.startswith("_")]


def test_provider_exposes_no_order_or_decision_method():
    for name in _public_names(BuyingPowerProvider):
        low = name.lower()
        assert all(tok not in low for tok in _FORBIDDEN), name
    # And explicitly: none of the classic order/decision entrypoints exist.
    for attr in ("place_order", "placeOrder", "submit", "cancel", "allow", "block", "decide"):
        assert not hasattr(BuyingPowerProvider, attr)


def test_snapshot_exposes_no_order_or_decision_method():
    for name in _public_names(BuyingPowerSnapshot):
        low = name.lower()
        assert all(tok not in low for tok in _FORBIDDEN), name


def test_module_does_not_import_live_or_execution_paths():
    import chad.risk.buying_power_provider as mod

    src = mod.__file__
    assert src is not None
    with open(src, "r", encoding="utf-8") as fh:
        text = fh.read()
    for banned in ("live_loop", "execution", "ibkr_adapter", "placeOrder", "place_order", "orchestrator"):
        assert banned not in text, banned

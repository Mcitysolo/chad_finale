#!/usr/bin/env python3
"""
chad/tests/test_options_quote_check.py

Phase D Item 2 Tier 3A — offline BAG quote-check engine tests.

All tests are pure / offline:

* No IBKR / ``ib_async`` / ``ib_insync`` import anywhere.
* No live socket, no broker connection, no order placement.
* No ``runtime/`` mutation.
* Synthetic ``OptionLegQuote`` / ``BagComboQuote`` fixtures only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chad.options.spread_spec import OptionsSpreadSpec
from chad.options.quote_check import (
    BagComboQuote,
    OptionLegQuote,
    SpreadQuoteCheckInput,
    SpreadQuoteCheckResult,
    check_spread_limit_price,
    compute_combo_mid_debit,
    compute_leg_mid_debit,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #

def _spec(net_debit_estimate: float = 1.85) -> OptionsSpreadSpec:
    return OptionsSpreadSpec(
        symbol="SPY",
        expiry="20260618",
        long_strike=737.0,
        short_strike=744.0,
        long_right="C",
        short_right="C",
        ratio_long=1,
        ratio_short=1,
        spread_type="BULL_CALL",
        net_debit_estimate=net_debit_estimate,
    )


def _long_leg(
    bid=None,
    ask=None,
    last=None,
    theo_price=None,
) -> OptionLegQuote:
    return OptionLegQuote(
        symbol="SPY",
        expiry="20260618",
        strike=737.0,
        right="C",
        bid=bid,
        ask=ask,
        last=last,
        theo_price=theo_price,
    )


def _short_leg(
    bid=None,
    ask=None,
    last=None,
    theo_price=None,
) -> OptionLegQuote:
    return OptionLegQuote(
        symbol="SPY",
        expiry="20260618",
        strike=744.0,
        right="C",
        bid=bid,
        ask=ask,
        last=last,
        theo_price=theo_price,
    )


# --------------------------------------------------------------------------- #
# 1. OptionLegQuote normalizes/accepts valid quote.                            #
# --------------------------------------------------------------------------- #

def test_option_leg_quote_accepts_valid_quote() -> None:
    leg = _long_leg(bid=3.40, ask=3.60, last=3.50)
    assert leg.bid == pytest.approx(3.40)
    assert leg.ask == pytest.approx(3.60)
    assert leg.last == pytest.approx(3.50)
    assert leg.strike == pytest.approx(737.0)
    assert leg.right == "C"


# --------------------------------------------------------------------------- #
# 2. Combo bid/ask mid preferred over leg mid.                                #
# --------------------------------------------------------------------------- #

def test_combo_mid_preferred_over_leg_mid() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)  # combo mid = 1.85
    long_q = _long_leg(bid=3.40, ask=3.60)  # leg long mid = 3.50
    short_q = _short_leg(bid=1.70, ask=1.80)  # leg short mid = 1.75
    # leg-diff would be 3.50 - 1.75 = 1.75; combo mid is 1.85
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.85,
            long_quote=long_q,
            short_quote=short_q,
            combo_quote=combo,
        )
    )
    assert result.ok is True
    assert result.source == "combo_mid"
    assert result.mid_debit == pytest.approx(1.85)


# --------------------------------------------------------------------------- #
# 3. Leg mid computes debit as long_mid - short_mid.                          #
# --------------------------------------------------------------------------- #

def test_leg_mid_debit_is_long_minus_short() -> None:
    long_q = _long_leg(bid=3.40, ask=3.60)
    short_q = _short_leg(bid=1.70, ask=1.80)
    debit, source, warnings = compute_leg_mid_debit(long_q, short_q)
    assert debit == pytest.approx(3.50 - 1.75)
    assert source == "leg_mid"
    assert "theo_only_quote_check" not in warnings


# --------------------------------------------------------------------------- #
# 4. Theo fallback computes debit and warns theo_only_quote_check.            #
# --------------------------------------------------------------------------- #

def test_theo_fallback_emits_advisory_warning() -> None:
    long_q = _long_leg(theo_price=3.50)
    short_q = _short_leg(theo_price=1.75)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.75,
            long_quote=long_q,
            short_quote=short_q,
        )
    )
    assert result.ok is True
    assert result.source == "theo_mid"
    assert result.mid_debit == pytest.approx(1.75)
    assert "theo_only_quote_check" in result.warnings


# --------------------------------------------------------------------------- #
# 5. No available mid returns ok=False and no_quote_mid_available.            #
# --------------------------------------------------------------------------- #

def test_no_mid_available_returns_error() -> None:
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.85,
            long_quote=None,
            short_quote=None,
            combo_quote=None,
        )
    )
    assert result.ok is False
    assert result.source == "none"
    assert result.mid_debit is None
    assert "no_quote_mid_available" in result.errors


# --------------------------------------------------------------------------- #
# 6. Limit equal to mid passes.                                                #
# --------------------------------------------------------------------------- #

def test_limit_equal_to_mid_passes() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.85,
            combo_quote=combo,
        )
    )
    assert result.ok is True
    assert result.deviation_abs == pytest.approx(0.0)
    assert result.errors == ()


# --------------------------------------------------------------------------- #
# 7. Limit below mid passes with warning limit_below_mid.                     #
# --------------------------------------------------------------------------- #

def test_limit_below_mid_passes_with_warning() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)  # mid = 1.85
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.50,
            combo_quote=combo,
        )
    )
    assert result.ok is True
    assert "limit_below_mid" in result.warnings
    assert result.deviation_abs is not None and result.deviation_abs < 0
    assert "limit_price_too_far_above_mid" not in result.errors


# --------------------------------------------------------------------------- #
# 8. Limit too far above mid fails.                                            #
# --------------------------------------------------------------------------- #

def test_limit_too_far_above_mid_fails() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)  # mid = 1.85
    # default tolerance: max(0.05, 0.10 * 1.85) = 0.185
    # limit 2.50 → deviation_abs = 0.65 > 0.185 → fail
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=2.50,
            combo_quote=combo,
        )
    )
    assert result.ok is False
    assert "limit_price_too_far_above_mid" in result.errors


# --------------------------------------------------------------------------- #
# 9. max_abs_deviation allows small absolute tolerance.                       #
# --------------------------------------------------------------------------- #

def test_max_abs_deviation_dominates_for_tight_mid() -> None:
    # mid = 0.10, pct tolerance = 0.01, abs tolerance = 0.05 → 0.05 wins
    combo = BagComboQuote(bid=0.09, ask=0.11)  # mid = 0.10
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=0.14,  # deviation 0.04 < 0.05
            combo_quote=combo,
            max_abs_deviation=0.05,
            max_pct_deviation=0.10,
        )
    )
    assert result.ok is True
    assert result.max_allowed_deviation == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# 10. max_pct_deviation allows percentage tolerance.                          #
# --------------------------------------------------------------------------- #

def test_max_pct_deviation_dominates_for_wide_mid() -> None:
    # mid = 10.0, pct tolerance = 1.0, abs tolerance = 0.05 → 1.0 wins
    combo = BagComboQuote(bid=9.80, ask=10.20)  # mid = 10.00
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=10.80,  # deviation 0.80 < 1.0
            combo_quote=combo,
            max_abs_deviation=0.05,
            max_pct_deviation=0.10,
        )
    )
    assert result.ok is True
    assert result.max_allowed_deviation == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 11. Invalid non-positive limit_price fails.                                 #
# --------------------------------------------------------------------------- #

def test_non_positive_limit_price_fails() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)
    for bad in (0.0, -0.5, float("nan"), float("inf")):
        result = check_spread_limit_price(
            SpreadQuoteCheckInput(
                spec=_spec(),
                limit_price=bad,
                combo_quote=combo,
            )
        )
        assert result.ok is False, f"limit_price={bad!r} should be rejected"
        assert "invalid_limit_price" in result.errors


# --------------------------------------------------------------------------- #
# 12. Crossed quote ask < bid is ignored and warning emitted.                 #
# --------------------------------------------------------------------------- #

def test_crossed_combo_quote_ignored_with_warning() -> None:
    # Crossed combo with no last → no combo mid; warning recorded.
    combo = BagComboQuote(bid=1.95, ask=1.80)  # crossed (ask < bid)
    long_q = _long_leg(bid=3.40, ask=3.60)
    short_q = _short_leg(bid=1.70, ask=1.80)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.75,
            long_quote=long_q,
            short_quote=short_q,
            combo_quote=combo,
        )
    )
    assert result.source == "leg_mid"
    assert "crossed_quote_ignored" in result.warnings


def test_crossed_leg_quote_ignored_with_warning() -> None:
    long_q = _long_leg(bid=3.60, ask=3.40, last=3.50)  # crossed
    short_q = _short_leg(bid=1.70, ask=1.80)
    debit, source, warnings = compute_leg_mid_debit(long_q, short_q)
    assert debit == pytest.approx(3.50 - 1.75)
    assert source == "leg_mid"
    assert "crossed_quote_ignored" in warnings


# --------------------------------------------------------------------------- #
# 13. Last price fallback works when bid/ask absent.                          #
# --------------------------------------------------------------------------- #

def test_last_price_fallback() -> None:
    long_q = _long_leg(last=3.50)
    short_q = _short_leg(last=1.75)
    debit, source, warnings = compute_leg_mid_debit(long_q, short_q)
    assert debit == pytest.approx(1.75)
    assert source == "leg_mid"
    assert "theo_only_quote_check" not in warnings


# --------------------------------------------------------------------------- #
# 14. Invalid negative bid/ask ignored.                                       #
# --------------------------------------------------------------------------- #

def test_negative_bid_ask_ignored_falls_through_to_last() -> None:
    long_q = _long_leg(bid=-1.0, ask=-2.0, last=3.50)
    short_q = _short_leg(bid=-1.0, ask=-2.0, last=1.75)
    debit, source, warnings = compute_leg_mid_debit(long_q, short_q)
    assert debit == pytest.approx(1.75)
    assert source == "leg_mid"
    # Negative bid/ask are simply absent — no crossed warning.
    assert "crossed_quote_ignored" not in warnings


# --------------------------------------------------------------------------- #
# 15. Combo quote unavailable falls back to leg quote.                        #
# --------------------------------------------------------------------------- #

def test_combo_unavailable_falls_back_to_leg() -> None:
    long_q = _long_leg(bid=3.40, ask=3.60)
    short_q = _short_leg(bid=1.70, ask=1.80)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.75,
            long_quote=long_q,
            short_quote=short_q,
            combo_quote=None,
        )
    )
    assert result.ok is True
    assert result.source == "leg_mid"
    assert result.mid_debit == pytest.approx(1.75)


# --------------------------------------------------------------------------- #
# 16. Source is combo_mid when combo used.                                    #
# --------------------------------------------------------------------------- #

def test_source_combo_mid_when_combo_used() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.85,
            combo_quote=combo,
        )
    )
    assert result.source == "combo_mid"


# --------------------------------------------------------------------------- #
# 17. Source is leg_mid when leg quote used.                                  #
# --------------------------------------------------------------------------- #

def test_source_leg_mid_when_leg_quote_used() -> None:
    long_q = _long_leg(bid=3.40, ask=3.60)
    short_q = _short_leg(bid=1.70, ask=1.80)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.75,
            long_quote=long_q,
            short_quote=short_q,
        )
    )
    assert result.source == "leg_mid"


# --------------------------------------------------------------------------- #
# 18. Source is theo_mid when theoretical quote used.                         #
# --------------------------------------------------------------------------- #

def test_source_theo_mid_when_theo_used() -> None:
    long_q = _long_leg(theo_price=3.50)
    short_q = _short_leg(theo_price=1.75)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.75,
            long_quote=long_q,
            short_quote=short_q,
        )
    )
    assert result.source == "theo_mid"
    assert "theo_only_quote_check" in result.warnings


# --------------------------------------------------------------------------- #
# 19. Result dataclass contains warnings/errors as tuples.                    #
# --------------------------------------------------------------------------- #

def test_result_uses_immutable_tuple_collections() -> None:
    combo = BagComboQuote(bid=1.80, ask=1.90)
    result = check_spread_limit_price(
        SpreadQuoteCheckInput(
            spec=_spec(),
            limit_price=1.85,
            combo_quote=combo,
        )
    )
    assert isinstance(result, SpreadQuoteCheckResult)
    assert isinstance(result.warnings, tuple)
    assert isinstance(result.errors, tuple)
    # Frozen dataclass: cannot mutate.
    with pytest.raises((AttributeError, TypeError)):
        result.ok = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 20. Module source contains no IBKR / ib_async / placeOrder imports.         #
# --------------------------------------------------------------------------- #

def test_module_source_has_no_ibkr_imports() -> None:
    """Forbid actual import/call usage of IBKR / strategy / execution modules.

    Patterns are intentionally precise — they match import statements and
    code-level references, not docstring text describing what the module
    deliberately does *not* do.
    """
    path = Path(__file__).resolve().parent.parent / "options" / "quote_check.py"
    src = path.read_text(encoding="utf-8")
    banned_patterns = [
        r"^\s*import\s+ib_async",
        r"^\s*from\s+ib_async\b",
        r"^\s*import\s+ib_insync",
        r"^\s*from\s+ib_insync\b",
        r"^\s*import\s+chad\.execution",
        r"^\s*from\s+chad\.execution\b",
        r"^\s*import\s+chad\.strategies",
        r"^\s*from\s+chad\.strategies\b",
        r"^\s*from\s+chad\.market_data\.ibkr_",
        r"^\s*import\s+chad\.market_data\.ibkr_",
        r"\bIbkrAdapter\(",
        r"\.placeOrder\(",
        r"\.connectAsync\(",
        r"\.reqMktData\(",
    ]
    for pat in banned_patterns:
        assert re.search(pat, src, flags=re.MULTILINE) is None, (
            f"Forbidden pattern {pat!r} found in chad/options/quote_check.py"
        )


# --------------------------------------------------------------------------- #
# 21. chad/options/__init__.py exports quote-check types.                     #
# --------------------------------------------------------------------------- #

def test_init_exports_quote_check_types() -> None:
    import chad.options as opt

    for name in (
        "OptionLegQuote",
        "BagComboQuote",
        "SpreadQuoteCheckInput",
        "SpreadQuoteCheckResult",
        "check_spread_limit_price",
        "OptionsSpreadSpec",
    ):
        assert hasattr(opt, name), f"chad.options should export {name}"
        assert name in opt.__all__, f"{name} missing from chad.options.__all__"

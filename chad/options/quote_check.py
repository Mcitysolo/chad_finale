#!/usr/bin/env python3
"""
chad/options/quote_check.py

Phase D Item 2 Tier 3A — offline BAG / combo spread quote-check engine.

A pure, deterministic, stdlib-only validator for the limit price of a
two-leg options spread. The module accepts caller-supplied leg quotes
and / or a combo quote, computes a debit mid from the best available
source, and reports whether a proposed limit price is within tolerance.

Hard safety contract
--------------------
* No IBKR, ``ib_async``, or ``ib_insync`` imports.
* No strategy, execution, or market-data live-quote imports.
* No order placement, no broker calls, no runtime mutation.
* No side effects on import.

This module is unwired — it is callable from preview / test surfaces
but no production code path consults it yet. A follow-on tier (Tier 3b)
adds the live-quote provider that feeds real broker bid/ask into this
checker. Tier 3c wires the adapter to enforce the result for live mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from chad.options.spread_spec import OptionsSpreadSpec


# --------------------------------------------------------------------------- #
# Source markers                                                              #
# --------------------------------------------------------------------------- #

_SRC_COMBO = "combo_mid"
_SRC_LEG = "leg_mid"
_SRC_THEO = "theo_mid"
_SRC_NONE = "none"

_ALL_SOURCES: Tuple[str, ...] = (_SRC_COMBO, _SRC_LEG, _SRC_THEO, _SRC_NONE)


# --------------------------------------------------------------------------- #
# Data shapes                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class OptionLegQuote:
    """A single option leg's broker-side quote, with an optional synthetic
    theoretical fallback. The checker treats theo_price as advisory only —
    a result that depended on theo_price emits the ``theo_only_quote_check``
    warning so live-mode wiring (Tier 3c) can refuse to act on it.
    """
    symbol: str
    expiry: str
    strike: float
    right: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    theo_price: Optional[float] = None
    ts_utc: Optional[str] = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class BagComboQuote:
    """Optional direct combo (BAG) quote. When present and usable the checker
    prefers this over leg-mid arithmetic.
    """
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    ts_utc: Optional[str] = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class SpreadQuoteCheckInput:
    """Inputs to ``check_spread_limit_price``. ``spec`` is the typed
    OptionsSpreadSpec from Tier 1 (used as a contract anchor; this offline
    checker does not currently consult any of its strike / right fields).
    """
    spec: OptionsSpreadSpec
    limit_price: float
    long_quote: Optional[OptionLegQuote] = None
    short_quote: Optional[OptionLegQuote] = None
    combo_quote: Optional[BagComboQuote] = None
    max_abs_deviation: float = 0.05
    max_pct_deviation: float = 0.10


@dataclass(frozen=True, slots=True)
class SpreadQuoteCheckResult:
    """Result of evaluating ``check_spread_limit_price``. ``ok=True`` iff a
    mid was computable and the proposed ``limit_price`` is within tolerance.
    A negative ``deviation_abs`` (limit below mid) is accepted with the
    ``limit_below_mid`` warning — a passive miss is preferable to overpaying.
    """
    ok: bool
    source: str
    mid_debit: Optional[float]
    limit_price: float
    deviation_abs: Optional[float]
    deviation_pct: Optional[float]
    max_allowed_deviation: Optional[float]
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _positive_float(value) -> Optional[float]:
    """Return ``float(value)`` iff value is a finite, strictly positive number.
    Returns ``None`` for ``None``, non-numeric, NaN, infinite, zero, and
    negative inputs.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    if out <= 0.0:
        return None
    return out


def _mid(
    bid,
    ask,
    last=None,
    theo_price=None,
) -> Tuple[Optional[float], str]:
    """Compute a single-quote mid price.

    Returns ``(price, source)`` where ``source`` is one of:

    * ``"bid_ask"`` — midpoint of a non-crossed positive bid/ask pair.
    * ``"last"`` — bid/ask unavailable (or crossed), last positive finite.
    * ``"theo"`` — bid/ask/last unavailable, theo_price positive finite.
    * ``"none"`` — nothing usable.

    Crossed quote handling (``ask < bid``): the bid/ask pair is dropped and
    the caller is expected to detect this separately and emit the
    ``crossed_quote_ignored`` warning. ``_mid`` itself is deterministic.
    """
    bid_v = _positive_float(bid)
    ask_v = _positive_float(ask)
    if bid_v is not None and ask_v is not None and ask_v >= bid_v:
        return ((bid_v + ask_v) / 2.0, "bid_ask")

    last_v = _positive_float(last)
    if last_v is not None:
        return (last_v, "last")

    theo_v = _positive_float(theo_price)
    if theo_v is not None:
        return (theo_v, "theo")

    return (None, "none")


def _is_crossed(bid, ask) -> bool:
    """True iff both bid and ask are positive finite and ``ask < bid``."""
    bid_v = _positive_float(bid)
    ask_v = _positive_float(ask)
    return bid_v is not None and ask_v is not None and ask_v < bid_v


# --------------------------------------------------------------------------- #
# Mid computation                                                             #
# --------------------------------------------------------------------------- #

def compute_combo_mid_debit(
    combo_quote: Optional[BagComboQuote],
) -> Tuple[Optional[float], str, Tuple[str, ...]]:
    """Compute the combo (BAG) debit from a single combo quote.

    Returns ``(mid_debit, source, warnings)``. ``source`` is ``"combo_mid"``
    when a mid is found, ``"none"`` otherwise. Combo bid/ask is preferred
    over combo.last (used only when bid/ask is unavailable or crossed).
    """
    if combo_quote is None:
        return (None, _SRC_NONE, ())

    warnings: list = []
    bid_v = _positive_float(combo_quote.bid)
    ask_v = _positive_float(combo_quote.ask)

    if bid_v is not None and ask_v is not None:
        if ask_v >= bid_v:
            return ((bid_v + ask_v) / 2.0, _SRC_COMBO, ())
        warnings.append("crossed_quote_ignored")

    last_v = _positive_float(combo_quote.last)
    if last_v is not None:
        return (last_v, _SRC_COMBO, tuple(warnings))

    return (None, _SRC_NONE, tuple(warnings))


def compute_leg_mid_debit(
    long_quote: Optional[OptionLegQuote],
    short_quote: Optional[OptionLegQuote],
) -> Tuple[Optional[float], str, Tuple[str, ...]]:
    """Compute the debit ``long_mid - short_mid`` from per-leg quotes.

    Returns ``(mid_debit, source, warnings)``. ``source`` is:

    * ``"leg_mid"`` — both legs resolved via bid/ask or last.
    * ``"theo_mid"`` — at least one leg fell back to theo_price (advisory).
    * ``"none"`` — at least one leg unresolvable.

    Warnings:

    * ``"crossed_quote_ignored"`` — at least one leg had ``ask < bid``.
    * ``"theo_only_quote_check"`` — at least one leg used theo_price.
    """
    if long_quote is None or short_quote is None:
        return (None, _SRC_NONE, ())

    warnings: list = []

    if _is_crossed(long_quote.bid, long_quote.ask) or _is_crossed(
        short_quote.bid, short_quote.ask
    ):
        warnings.append("crossed_quote_ignored")

    long_mid, long_src = _mid(
        long_quote.bid, long_quote.ask, long_quote.last, long_quote.theo_price
    )
    short_mid, short_src = _mid(
        short_quote.bid, short_quote.ask, short_quote.last, short_quote.theo_price
    )

    if long_mid is None or short_mid is None:
        return (None, _SRC_NONE, tuple(warnings))

    debit = long_mid - short_mid

    if long_src == "theo" or short_src == "theo":
        warnings.append("theo_only_quote_check")
        source = _SRC_THEO
    else:
        source = _SRC_LEG

    return (debit, source, tuple(warnings))


# --------------------------------------------------------------------------- #
# Main checker                                                                #
# --------------------------------------------------------------------------- #

def check_spread_limit_price(
    input: SpreadQuoteCheckInput,
) -> SpreadQuoteCheckResult:
    """Validate a proposed BAG / spread limit price against the best
    available mid. See module docstring for the safety contract.

    Decision tree:

    1. ``limit_price`` must be positive finite.
    2. If a combo quote yields a usable mid, use ``"combo_mid"``.
    3. Else, if both leg quotes yield a mid (via bid/ask or last), use
       ``"leg_mid"``.
    4. Else, if both legs supply theo_price, use ``"theo_mid"`` with the
       ``theo_only_quote_check`` warning.
    5. Else, fail with ``no_quote_mid_available``.

    Tolerance: ``deviation_abs <= max(max_abs_deviation,
    max_pct_deviation * mid_debit)``. A limit below mid is accepted with
    the ``limit_below_mid`` warning.
    """
    warnings: list = []
    errors: list = []

    raw_lp = input.limit_price
    try:
        lp_float = float(raw_lp)
    except (TypeError, ValueError):
        lp_float = 0.0

    lp = _positive_float(raw_lp)
    if lp is None:
        errors.append("invalid_limit_price")
        return SpreadQuoteCheckResult(
            ok=False,
            source=_SRC_NONE,
            mid_debit=None,
            limit_price=lp_float,
            deviation_abs=None,
            deviation_pct=None,
            max_allowed_deviation=None,
            warnings=(),
            errors=tuple(errors),
        )

    mid_debit: Optional[float] = None
    source = _SRC_NONE

    combo_mid, combo_src, combo_warn = compute_combo_mid_debit(input.combo_quote)
    if (
        combo_mid is not None
        and math.isfinite(combo_mid)
        and combo_mid > 0
    ):
        mid_debit = combo_mid
        source = combo_src
        warnings.extend(combo_warn)
    else:
        # combo not usable — record any combo-side warnings (e.g. crossed)
        warnings.extend(combo_warn)

        leg_mid, leg_src, leg_warn = compute_leg_mid_debit(
            input.long_quote, input.short_quote
        )
        if (
            leg_mid is not None
            and math.isfinite(leg_mid)
            and leg_mid > 0
        ):
            mid_debit = leg_mid
            source = leg_src
            warnings.extend(leg_warn)
        else:
            warnings.extend(leg_warn)

    if mid_debit is None:
        errors.append("no_quote_mid_available")
        return SpreadQuoteCheckResult(
            ok=False,
            source=_SRC_NONE,
            mid_debit=None,
            limit_price=lp,
            deviation_abs=None,
            deviation_pct=None,
            max_allowed_deviation=None,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    deviation_abs = lp - mid_debit
    deviation_pct = deviation_abs / mid_debit
    max_allowed = max(
        float(input.max_abs_deviation),
        float(input.max_pct_deviation) * mid_debit,
    )

    if deviation_abs < 0:
        warnings.append("limit_below_mid")
        ok = True
    elif deviation_abs <= max_allowed:
        ok = True
    else:
        ok = False
        errors.append("limit_price_too_far_above_mid")

    return SpreadQuoteCheckResult(
        ok=ok,
        source=source,
        mid_debit=mid_debit,
        limit_price=lp,
        deviation_abs=deviation_abs,
        deviation_pct=deviation_pct,
        max_allowed_deviation=max_allowed,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


__all__ = [
    "OptionLegQuote",
    "BagComboQuote",
    "SpreadQuoteCheckInput",
    "SpreadQuoteCheckResult",
    "compute_combo_mid_debit",
    "compute_leg_mid_debit",
    "check_spread_limit_price",
]

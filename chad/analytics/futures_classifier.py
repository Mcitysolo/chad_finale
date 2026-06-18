"""Canonical futures-row classifier (confidence-sample item 5b).

Single source of truth for "is this trade/fill row a futures contract", used by
the SCR confidence-sample builders (``trade_stats_engine`` and
``expectancy_tracker``) to keep Bug-B futures contamination out of the effective
performance sample.

Why a dedicated set instead of ``chad.utils.context_builder.DEFAULT_FUTURES_SYMBOLS``
-----------------------------------------------------------------------------------
That constant is env-overridable (``CHAD_FUTURES_SYMBOLS``) and scoped to *which*
futures to pull into every context build — its default ``{MES, MNQ, MCL, MGC}`` is
deliberately small and notably OMITS ``M6E``/``M2K`` that do land in the ledger.
A classifier used to *exclude* contamination must recognise every micro-futures
root that can appear, so it carries its own complete, non-tunable root set.

Matching is by symbol ROOT so it survives any contract-month suffix, plus an
authoritative ``sec_type == "FUT"`` short-circuit (sparse in the ledger, but
trustworthy when present). Pure module: no I/O, import-safe.
"""
from __future__ import annotations

from typing import Any, Optional

# Complete set of CME micro-futures roots observed in CHAD's ledgers.
CME_MICRO_FUTURES_ROOTS = frozenset(
    {
        "MES",  # Micro E-mini S&P 500
        "MNQ",  # Micro E-mini Nasdaq-100
        "MYM",  # Micro E-mini Dow
        "M2K",  # Micro E-mini Russell 2000
        "M6E",  # Micro EUR/USD
        "MGC",  # Micro Gold
        "MCL",  # Micro WTI Crude
    }
)

# CME contract month codes (Jan..Dec). Used only to recognise dated contract
# suffixes like ``MESZ5`` without false-positiving on equity tickers.
_CME_MONTH_CODES = frozenset("FGHJKMNQUVXZ")


def is_futures_row(symbol: Any, sec_type: Optional[Any] = None) -> bool:
    """Return True when a ledger row represents a futures contract.

    Two independent signals (belt-and-suspenders):

    * ``sec_type == "FUT"`` — authoritative when present, but most ledger rows
      carry no ``secType``, so it cannot stand alone.
    * symbol root in :data:`CME_MICRO_FUTURES_ROOTS` — the practical discriminator.
      Matches the bare root (e.g. ``MES``) and a dated contract whose suffix is a
      year digit optionally preceded by a CME month-code letter (e.g. ``MESZ5``,
      ``M6EM26``). A bare equity ticker such as ``MESA`` does not end in a digit,
      so it never enters the dated-contract branch and is not misclassified.
    """
    if sec_type is not None and str(sec_type).strip().upper() == "FUT":
        return True

    s = str(symbol or "").strip().upper()
    if not s:
        return False

    if s in CME_MICRO_FUTURES_ROOTS:
        return True

    # Dated-contract tolerance: a CME contract symbol ends in a year digit,
    # optionally preceded by a month-code letter. Only engage when the symbol
    # actually ends in a digit so plain equity tickers can never reach here.
    if s[-1].isdigit():
        core = s.rstrip("0123456789")
        if core in CME_MICRO_FUTURES_ROOTS:
            return True
        if core and core[-1] in _CME_MONTH_CODES and core[:-1] in CME_MICRO_FUTURES_ROOTS:
            return True

    return False


__all__ = ["CME_MICRO_FUTURES_ROOTS", "is_futures_row"]

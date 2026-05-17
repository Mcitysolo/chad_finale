"""chad.options — options chain + spread data types.

Public exports:
* ``OptionsSpreadSpec`` — typed two-leg combo/spread spec consumed by the
  IBKR adapter and paper-fill simulator (Phase D Item 2 Tier 1).
* ``OptionLegQuote`` / ``BagComboQuote`` / ``SpreadQuoteCheckInput`` /
  ``SpreadQuoteCheckResult`` / ``check_spread_limit_price`` — offline
  BAG quote-check engine (Phase D Item 2 Tier 3A).
"""

from chad.options.quote_check import (
    BagComboQuote,
    OptionLegQuote,
    SpreadQuoteCheckInput,
    SpreadQuoteCheckResult,
    check_spread_limit_price,
)
from chad.options.spread_spec import OptionsSpreadSpec

__all__ = [
    "OptionsSpreadSpec",
    "OptionLegQuote",
    "BagComboQuote",
    "SpreadQuoteCheckInput",
    "SpreadQuoteCheckResult",
    "check_spread_limit_price",
]

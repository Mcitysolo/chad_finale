"""chad.options — options chain + spread data types.

Public exports:
* ``OptionsSpreadSpec`` — typed two-leg combo/spread spec consumed by the
  IBKR adapter and paper-fill simulator (Phase D Item 2 Tier 1).
"""

from chad.options.spread_spec import OptionsSpreadSpec

__all__ = ["OptionsSpreadSpec"]

"""chad/constants/fx.py — canonical FX constants (Principle 9: one definition).

Import the constant defined here; never re-declare ``1.4160`` inline anywhere.
"""

from __future__ import annotations

# Band the sanctioned conversion constant must sit inside. A value outside this
# band is either inverted (~0.73 = USD-per-CAD) or garbage; the guard below
# refuses to import such a value rather than let a bad rate leak into valuation.
USDCAD_BAND_LOW: float = 1.20
USDCAD_BAND_HIGH: float = 1.50

USDCAD_CONVERSION_CONSTANT: float = 1.4160
"""C-ii sanctioned one-time USD->CAD conversion constant (CAD per USD).

The live USD.CAD IBKR feed is dark on the paper account (Error 10089); CHAD
does not trade forex. Used ONLY to value the small crypto sliver of Kraken
holdings into the CAD base. NEVER stored as a live rate. Band-valid
[1.20,1.50].
"""

# Import-time integrity guard (plain if/raise so ``python -O`` cannot strip it):
# a hand-edit that moves the constant out of the sanctioned band fails loudly at
# import instead of silently corrupting CAD valuation downstream.
if not (USDCAD_BAND_LOW <= USDCAD_CONVERSION_CONSTANT <= USDCAD_BAND_HIGH):
    raise ValueError(
        "USDCAD_CONVERSION_CONSTANT="
        f"{USDCAD_CONVERSION_CONSTANT} outside sanctioned band "
        f"[{USDCAD_BAND_LOW},{USDCAD_BAND_HIGH}] (CAD per USD)"
    )

__all__ = [
    "USDCAD_CONVERSION_CONSTANT",
    "USDCAD_BAND_LOW",
    "USDCAD_BAND_HIGH",
]

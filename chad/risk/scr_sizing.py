"""SCR sizing-factor application — single source of truth.

TIER1-B (PRE-LIVE BLOCKING). The SCR governor publishes a ``sizing_factor``
to ``runtime/scr_state.json`` for every state (WARMUP=0.10, CAUTIOUS=0.25,
CONFIDENT=1.00, PAUSED=0.00). Historically the execution path applied that
factor to intent quantity ONLY when the state was ``CAUTIOUS``
(chad/core/live_loop.py). In every other throttled state — notably WARMUP,
where 0.10 is published — the factor was displayed but never applied to any
order quantity: a decorative control.

This module makes the applied path uniform and self-attesting:

* ``scr_sizing_should_apply`` is the policy predicate. The live loop gates its
  scaling on it AND the Exterminator sentinel (EX021) mirrors it into the
  ``scr_sizing_application.v1`` evidence marker — so the applied path and the
  observability marker cannot drift. Re-narrowing the policy flips the marker
  and trips the sentinel; it can never go silent again.
* ``apply_scr_sizing`` is the scaling math, extracted verbatim from the former
  CAUTIOUS block so CAUTIOUS behaviour is byte-identical.

Kill-switch: ``CHAD_SCR_SIZING_APPLY`` (default ON). Set to 0/false/no/off to
disable application — which the sentinel then reports as a decorative factor.
"""

from __future__ import annotations

import math
import os

# Tokens that turn the (default-on) application kill-switch OFF.
_DISABLE_TOKENS = frozenset({"0", "false", "no", "off"})


def scr_sizing_apply_enabled() -> bool:
    """Application kill-switch. Default ON.

    ``CHAD_SCR_SIZING_APPLY`` in {0, false, no, off} disables application of the
    published SCR sizing factor. Disabling it re-creates a decorative control,
    which the EX021 sentinel reports as CRITICAL.
    """
    return (
        os.environ.get("CHAD_SCR_SIZING_APPLY", "1").strip().lower()
        not in _DISABLE_TOKENS
    )


def scr_sizing_should_apply(
    state: str,
    sizing_factor: float,
    *,
    enabled: bool | None = None,
) -> bool:
    """Policy predicate: is the published factor applied to intent quantity?

    True iff application is enabled, the state is not ``PAUSED`` (PAUSED is a
    hard-block handled upstream, not a scale) and the factor is a real throttle
    ``0 < f < 1.0``. A factor of exactly 1.0 (CONFIDENT) is a no-op and returns
    False. Any non-finite / unparseable factor returns False (fail-safe).

    The live loop and the sentinel marker both consume this predicate so the
    applied path and the observability evidence are guaranteed consistent.
    """
    if enabled is None:
        enabled = scr_sizing_apply_enabled()
    if not enabled:
        return False
    if str(state or "").upper() == "PAUSED":
        return False
    try:
        f = float(sizing_factor)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(f):
        return False
    return 0.0 < f < 1.0


def apply_scr_sizing(raw_qty: float, sizing_factor: float, sec_type: str) -> float:
    """Scale a raw intent quantity by the published SCR sizing factor.

    Futures (``sec_type == "FUT"``) round to the nearest whole contract with a
    floor of 1; every other instrument floors to whole units with a floor of 1.
    This is the former CAUTIOUS math, unchanged, so applying it in CAUTIOUS is
    byte-identical to prior behaviour.
    """
    raw = float(raw_qty or 0.0)
    f = float(sizing_factor)
    if str(sec_type or "").upper() == "FUT":
        return max(1.0, round(raw * f))
    return max(1.0, float(math.floor(raw * f)))

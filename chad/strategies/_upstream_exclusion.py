"""GAP-035 upstream exclusion helper for strategy emitters.

Single source of truth proxy: re-exports the canonical
``_EFFECTIVE_NON_CHAD_SYMBOLS`` frozenset and ``_EXCLUSION_SOURCE`` label
from ``chad.core.position_reconciler`` so strategy modules
(``delta``, ``delta_pairs``, ``alpha_options``) can filter operator-
excluded symbols *before* emitting ``TradeSignal`` objects.

Why this exists: the close-path chokepoints
(``position_reconciler.apply_close_intents`` and
``flip_executor.enforce_flip_close_first``) already refuse close intents
on excluded symbols. They do not, however, prevent CHAD strategies from
*opening* a position on an excluded symbol — that requires upstream
filtering at the signal emitter, which is GAP-035.

Leaf-dependency direction: this module imports from
``chad.core.position_reconciler``; ``chad.core.position_reconciler`` does
NOT import from ``chad.strategies``. Strategies are free to import this
helper without creating a cycle.

Fail-closed posture: if the canonical SSOT cannot be imported for any
reason (broken refactor, partial install) the fallback exclusion set is
**empty** and the source label is ``fallback_empty``. That means a missing
SSOT does NOT silently widen exclusions — strategies still emit; the
downstream chokepoints remain the last line of defence.
"""
from __future__ import annotations

from typing import FrozenSet

try:
    from chad.core.position_reconciler import (  # type: ignore
        _EFFECTIVE_NON_CHAD_SYMBOLS as _CANONICAL_EXCLUDED,
        _EXCLUSION_SOURCE as _CANONICAL_SOURCE,
    )
    OPERATOR_EXCLUDED_SYMBOLS: FrozenSet[str] = frozenset(
        str(s).upper() for s in _CANONICAL_EXCLUDED
    )
    EXCLUSION_SOURCE: str = str(_CANONICAL_SOURCE)
except Exception:  # noqa: BLE001 — fail-closed: never raise at import time
    OPERATOR_EXCLUDED_SYMBOLS = frozenset()
    EXCLUSION_SOURCE = "fallback_empty"


def is_operator_excluded(symbol: object) -> bool:
    """True if ``symbol`` is in the operator-exclusion SSOT.

    Case-insensitive; ``None`` / empty / non-string inputs return False.
    """
    if symbol is None:
        return False
    try:
        s = str(symbol).strip().upper()
    except Exception:  # noqa: BLE001
        return False
    if not s:
        return False
    return s in OPERATOR_EXCLUDED_SYMBOLS


def filter_operator_excluded(symbols):
    """Yield only the symbols that are NOT operator-excluded.

    Accepts any iterable; preserves order; ignores ``None`` / empty.
    """
    if not symbols:
        return
    for sym in symbols:
        if sym is None:
            continue
        try:
            s = str(sym).strip().upper()
        except Exception:  # noqa: BLE001
            continue
        if not s or s in OPERATOR_EXCLUDED_SYMBOLS:
            continue
        yield s


__all__ = [
    "EXCLUSION_SOURCE",
    "OPERATOR_EXCLUDED_SYMBOLS",
    "filter_operator_excluded",
    "is_operator_excluded",
]

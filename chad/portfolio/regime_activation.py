"""
chad/portfolio/regime_activation.py

Phase-8 Session 4 (G2): regime-driven strategy activation filter.

Loads config/regime_activation_matrix.json and exposes helpers that let
the routing layer drop intents whose strategy is not allowed under the
current regime label.

Default config ships with every strategy enabled in every regime, so
installing this module does not change behavior until an operator edits
the matrix. Missing config, unreadable JSON, or unknown regime labels
fail OPEN — the intent is allowed through — so a mis-configured matrix
never silently stops trading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = ROOT / "config" / "regime_activation_matrix.json"


def load_activation_matrix(
    path: Path = DEFAULT_MATRIX_PATH,
) -> Mapping[str, Sequence[str]]:
    """Read the regime → strategy-list map. Returns {} on any error."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOG.warning("regime_activation_matrix unreadable (%s); failing open", exc)
        return {}
    if not isinstance(data, dict):
        return {}
    regimes = data.get("regimes")
    if not isinstance(regimes, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in regimes.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
    return out


def allowed_strategies_for_regime(
    regime: str,
    matrix: Optional[Mapping[str, Sequence[str]]] = None,
) -> Optional[Sequence[str]]:
    """Return the allowed-strategies list for the given regime.

    Returns None if the regime is not present in the matrix — the caller
    should treat that as 'allow everything' (fail-open semantics).
    """
    mat = load_activation_matrix() if matrix is None else matrix
    if not mat:
        return None
    key = str(regime or "unknown").lower()
    if key in mat:
        return mat[key]
    # Unknown regime label falls back to the 'unknown' bucket if present,
    # otherwise allow-all.
    return mat.get("unknown")


def is_strategy_allowed(
    strategy: str,
    regime: str,
    matrix: Optional[Mapping[str, Sequence[str]]] = None,
) -> bool:
    """True if the strategy is allowed under the given regime (fail-open)."""
    allowed = allowed_strategies_for_regime(regime, matrix)
    if allowed is None:
        return True  # fail-open
    return str(strategy) in set(allowed)


def _intent_strategy(intent: Any) -> str:
    """Best-effort strategy identifier extraction from an intent object."""
    s = getattr(intent, "strategy", None)
    if s is None and isinstance(intent, Mapping):
        s = intent.get("strategy")
    if hasattr(s, "value"):
        return str(s.value)
    return str(s or "")


def filter_intents_by_regime(
    intents: Iterable[Any],
    regime: str,
    matrix: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[List[Any], List[Tuple[Any, str]]]:
    """Partition intents into (allowed, rejected).

    rejected is a list of (intent, reason) tuples so the caller can emit
    a structured log per dropped intent.
    """
    mat = load_activation_matrix() if matrix is None else matrix
    allowed = allowed_strategies_for_regime(regime, mat)

    if allowed is None:
        # Fail-open: no matrix or regime not present → let everything through.
        out: List[Any] = list(intents)
        return out, []

    allowed_set = set(allowed)
    kept: List[Any] = []
    dropped: List[Tuple[Any, str]] = []
    for intent in intents:
        strat = _intent_strategy(intent)
        if strat in allowed_set:
            kept.append(intent)
        else:
            dropped.append((intent, f"regime_strategy_mismatch:regime={regime};strategy={strat}"))
    return kept, dropped


__all__ = [
    "DEFAULT_MATRIX_PATH",
    "load_activation_matrix",
    "allowed_strategies_for_regime",
    "is_strategy_allowed",
    "filter_intents_by_regime",
]

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

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = ROOT / "config" / "regime_activation_matrix.json"

# --------------------------------------------------------------------------- #
# CRYPTO-EXPLORE-WIRE W2 — paper-epoch-only crypto exploration flag.
# (ops/pending_actions/UC1_crypto_exploration_mode_2026-07-17.md)
#
# The flag re-admits `alpha_crypto` intents that the regime matrix would drop, so CHAD can
# test the *assumption* that crypto has no edge in `ranging`. It is implemented HERE, at the
# regime-gate check — NOT by mutating config/regime_activation_matrix.json — so the matrix
# stays byte-identical and flag-off restores the exact stock GATING decision with zero residue
# (the rollback IS the default). Scope note: "zero residue" is about the routing/gating
# decision — W3 regime-tagging and the W1 overlay heartbeat are always-on observability and do
# NOT change which intents fill; they are intentionally not flag-gated.
# --------------------------------------------------------------------------- #
CRYPTO_EXPLORATION_ENV = "CHAD_CRYPTO_EXPLORATION"
EXEC_MODE_ENV = "CHAD_EXECUTION_MODE"
# The crypto lane has its OWN live/paper switch, DECOUPLED from the global execution mode
# (chad/core/kraken_execution.py:resolve_kraken_mode — an explicit CHAD_KRAKEN_MODE=live routes
# to real Kraken orders regardless of CHAD_EXECUTION_MODE). Exploration re-admits ONLY
# alpha_crypto, a Kraken-lane strategy, so the fail-closed guard must cover this axis too.
KRAKEN_MODE_ENV = "CHAD_KRAKEN_MODE"
EXPLORATION_STRATEGY = "alpha_crypto"

MARKER_EXPLORATION_PASS = "CRYPTO_EXPLORATION_PASS"
MARKER_EXPLORATION_FLAG = "exploration=true"
MARKER_EXPLORATION_REFUSED = "CRYPTO_EXPLORATION_REFUSED_NON_PAPER"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def crypto_exploration_state(
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, str]:
    """Resolve the exploration flag into (active, reason).

    - ("off"): flag unset/false — stock gating, the matrix edit (never made) is inert.
    - ("active"): flag on AND ``CHAD_EXECUTION_MODE=paper`` AND the crypto lane is not live —
      exploration entries honored.
    - ("refused_non_paper"): flag on but global ``CHAD_EXECUTION_MODE`` != paper — FAIL-CLOSED.
    - ("refused_kraken_live"): flag on, global mode paper, but ``CHAD_KRAKEN_MODE=live`` —
      FAIL-CLOSED on the SECOND axis. Exploration re-admits only ``alpha_crypto`` (a Kraken-lane
      strategy), and the Kraken lane's live/paper switch is decoupled from the global one, so a
      global paper posture is NOT sufficient: an explicit ``CHAD_KRAKEN_MODE=live`` would route
      re-admitted intents to REAL Kraken orders. Both refusals keep stock gating and re-admit
      nothing. Exploration is a paper-epoch-only measurement instrument; it must never — on
      either axis — silently downgrade into a live posture (UC1 §5, "it dies at live").
    """
    env = os.environ if env is None else env
    raw = str(env.get(CRYPTO_EXPLORATION_ENV, "") or "").strip().lower()
    if raw not in _TRUTHY:
        return False, "off"
    mode = str(env.get(EXEC_MODE_ENV, "") or "").strip().lower()
    if mode != "paper":
        return False, "refused_non_paper"
    # Second axis: an explicit CHAD_KRAKEN_MODE=live is the ONLY value that resolves the Kraken
    # lane to live while the global mode is paper (resolve_kraken_mode: unset/other falls back
    # to the global mode, which we have already pinned to paper here). Refuse on it.
    kmode = str(env.get(KRAKEN_MODE_ENV, "") or "").strip().lower()
    if kmode == "live":
        return False, "refused_kraken_live"
    return True, "active"


def stamp_intent_regime(intent: Any, regime: str) -> Any:
    """Return ``intent`` carrying the live ``regime`` (CRYPTO-EXPLORE-WIRE W3).

    Prefers an immutable ``dataclasses.replace`` when the intent exposes a ``regime`` field
    (the Kraken ``StrategyTradeIntent``); falls back to a best-effort ``setattr`` and finally to
    the unchanged intent. Never raises — a stamping failure must not drop a valid intent.
    """
    r = str(regime or "")
    try:
        if dataclasses.is_dataclass(intent) and not isinstance(intent, type):
            fields = {f.name for f in dataclasses.fields(intent)}
            if "regime" in fields:
                return dataclasses.replace(intent, regime=r)
    except Exception:  # noqa: BLE001 - stamping is best-effort
        pass
    try:
        setattr(intent, "regime", r)
    except Exception:  # noqa: BLE001 - frozen/other intent shapes keep their default
        pass
    return intent


def _tag_exploration(intent: Any, regime: str) -> Any:
    """Tag a re-admitted alpha_crypto intent: exploration markers + live regime.

    Adds ``CRYPTO_EXPLORATION_PASS regime=<r>`` and ``exploration=true`` to ``markers`` so the
    marks flow all the way into the trusted-fill evidence tags (the harness slices on them),
    and stamps ``regime`` so the fill is regime-attributable. One immutable replace for a frozen
    dataclass; best-effort for anything else.
    """
    r = str(regime or "")
    existing = tuple(str(m) for m in (getattr(intent, "markers", ()) or ()))
    new_markers = existing + (f"{MARKER_EXPLORATION_PASS} regime={r}", MARKER_EXPLORATION_FLAG)
    try:
        if dataclasses.is_dataclass(intent) and not isinstance(intent, type):
            fields = {f.name for f in dataclasses.fields(intent)}
            changes: dict = {}
            if "markers" in fields:
                changes["markers"] = new_markers
            if "regime" in fields:
                changes["regime"] = r
            if changes:
                return dataclasses.replace(intent, **changes)
    except Exception:  # noqa: BLE001 - tagging is best-effort
        pass
    try:
        setattr(intent, "markers", new_markers)
        setattr(intent, "regime", r)
    except Exception:  # noqa: BLE001
        pass
    return intent


def apply_crypto_exploration(
    kept: Sequence[Any],
    dropped: Sequence[Tuple[Any, str]],
    regime: str,
    *,
    env: Optional[Mapping[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Any], List[Tuple[Any, str]], dict]:
    """Post-process a regime-gate partition to honour crypto exploration (W2).

    Re-admits ``alpha_crypto`` intents that were dropped ONLY by the regime gate, tagging each
    with the exploration markers + live regime, when the flag is active (on + paper). Off →
    returns the partition untouched (byte-identical stock gating). Non-paper with the flag on →
    refuses loudly and keeps stock gating (fail-closed).

    Returns (kept2, dropped2, info) where info={"state", "readmitted"}.
    """
    log = logger or LOG
    active, reason = crypto_exploration_state(env)
    kept_out = list(kept)
    dropped_out = list(dropped)

    if reason == "off":
        return kept_out, dropped_out, {"state": "off", "readmitted": 0}

    if reason.startswith("refused"):
        would = [i for (i, _) in dropped_out if _intent_strategy(i) == EXPLORATION_STRATEGY]
        _env = env or os.environ
        if reason == "refused_kraken_live":
            axis = f"{KRAKEN_MODE_ENV}={_env.get(KRAKEN_MODE_ENV, '<unset>')} (crypto lane LIVE)"
        else:
            axis = f"{EXEC_MODE_ENV}={_env.get(EXEC_MODE_ENV, '<unset>')} (not paper)"
        log.error(
            "%s flag=%s=1 but %s — REFUSING to arm crypto exploration; keeping stock gating "
            "and dropping %d exploration entr%s.",
            MARKER_EXPLORATION_REFUSED, CRYPTO_EXPLORATION_ENV, axis,
            len(would), "y" if len(would) == 1 else "ies",
        )
        return kept_out, dropped_out, {"state": reason, "readmitted": 0}

    # active: re-admit dropped alpha_crypto intents, tagged.
    readmitted: List[Any] = []
    still_dropped: List[Tuple[Any, str]] = []
    for intent, drop_reason in dropped_out:
        if _intent_strategy(intent) == EXPLORATION_STRATEGY:
            tagged = _tag_exploration(intent, regime)
            readmitted.append(tagged)
            log.info(
                "%s regime=%s strategy=%s pair=%s (paper exploration re-admit)",
                MARKER_EXPLORATION_PASS, regime, _intent_strategy(intent),
                getattr(intent, "pair", None) or getattr(intent, "symbol", None),
            )
        else:
            still_dropped.append((intent, drop_reason))
    return kept_out + readmitted, still_dropped, {"state": "active", "readmitted": len(readmitted)}


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
    # CRYPTO-EXPLORE-WIRE W2/W3
    "CRYPTO_EXPLORATION_ENV",
    "EXEC_MODE_ENV",
    "KRAKEN_MODE_ENV",
    "EXPLORATION_STRATEGY",
    "MARKER_EXPLORATION_PASS",
    "MARKER_EXPLORATION_FLAG",
    "MARKER_EXPLORATION_REFUSED",
    "crypto_exploration_state",
    "stamp_intent_regime",
    "apply_crypto_exploration",
]

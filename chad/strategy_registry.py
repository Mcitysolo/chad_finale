#!/usr/bin/env python3
"""
chad/strategy_registry.py

Single canonical strategy registry — the SSOT that resolves the historical
three-way drift between:

  - ``chad.types.StrategyName``                       — the enum (18 declared names)
  - ``chad.risk.tier_manager._CANONICAL_STRATEGY_NAMES`` — tier active-eligible set
  - ``config/strategy_weights.json``                  — the weighted / trading set

Before this module those three lists disagreed (enum 18 / tier 17 / weights 16)
with no guard against further silent divergence. This module declares every
strategy exactly once with two orthogonal axes, and `assert_registry_consistency`
fails the process loudly the moment any consumer drifts.

Two axes per strategy
---------------------
``track``  : ``main``        — runs in the main StrategyEngine (registered handler)
             ``side_engine`` — wired only into a side engine (e.g. dominance_allocator),
                               not registered in the main engine
``status`` : ``active``      — part of the real trading set; carries an allocator
                               weight in ``config/strategy_weights.json``
             ``dormant``      — declared and known, but deliberately NOT trading
                               (no weight). Never silently dropped; never
                               auto-activated. Future activation is a deliberate
                               operator action (assign a weight + flip to active).

Derived sets (the single source consumers must agree with)
----------------------------------------------------------
  ACTIVE  (status == active)   == ``config/strategy_weights.json`` weights keys (16)
                               == ``tier_manager`` active-eligible set (16)
  DORMANT (status == dormant)  == {alpha_intraday_micro (main, pending weight),
                                   alpha_forex (side_engine)}                  (2)
  enum    == ACTIVE ∪ DORMANT                                                  (18)

Relationship to the handler-registration SSOT (``chad.strategies``)
-------------------------------------------------------------------
The ``track`` axis projects cleanly onto the existing handler-registration
partition, so this registry stays consistent with it (cross-checked by the
assertion):

  track == main         <=> ``chad.strategies.active_strategy_names()``   (17 registered)
  track == side_engine  <=> ``chad.strategies.DEFERRED_STRATEGIES``       ({alpha_forex})

Note that ``alpha_intraday_micro`` is ``track=main`` (its handler IS registered
and runs every cycle) yet ``status=dormant`` (it carries no weight, so the
allocator never issues it a cap and it produces 0 fills). The two axes are
independent: registration governs whether a handler executes; status governs
whether the strategy is part of the weighted trading set.

This module does NOT duplicate the enum — every entry references a
``StrategyName`` member. It has no import-time side effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from chad.types import StrategyName

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "config" / "strategy_weights.json"


class Track(str, Enum):
    """Engine track a strategy is wired into."""

    MAIN = "main"
    SIDE_ENGINE = "side_engine"


class Status(str, Enum):
    """Trading status: whether the strategy is part of the weighted set."""

    ACTIVE = "active"
    DORMANT = "dormant"


@dataclass(frozen=True)
class StrategyEntry:
    """Immutable declaration of one strategy on both axes."""

    name: StrategyName
    track: Track
    status: Status
    note: str = ""


# ---------------------------------------------------------------------------
# THE canonical declaration. Every StrategyName appears exactly once.
#   16 ACTIVE  (status=active)  — identical to config/strategy_weights.json keys
#    2 DORMANT (status=dormant) — declared, known, deliberately not trading
# ---------------------------------------------------------------------------
_REGISTRY: Tuple[StrategyEntry, ...] = (
    # --- ACTIVE: the 16 weighted strategies (the real trading set) ----------
    StrategyEntry(StrategyName.ALPHA, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.BETA, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.BETA_TREND, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.GAMMA, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.OMEGA, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.DELTA, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.ALPHA_CRYPTO, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.ALPHA_INTRADAY, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.ALPHA_FUTURES, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.GAMMA_FUTURES, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.OMEGA_MACRO, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.GAMMA_REVERSION, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.ALPHA_OPTIONS, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.OMEGA_VOL, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.DELTA_PAIRS, Track.MAIN, Status.ACTIVE),
    StrategyEntry(StrategyName.OMEGA_MOMENTUM_OPTIONS, Track.MAIN, Status.ACTIVE),
    # --- DORMANT: declared, known, NOT trading (no weight) ------------------
    StrategyEntry(
        StrategyName.ALPHA_INTRADAY_MICRO,
        Track.MAIN,
        Status.DORMANT,
        note=(
            "pending weight assignment; handler IS registered (main track) and "
            "runs every cycle, but carries no allocator weight so it produces "
            "0 fills. Activation is a deliberate future operator action — "
            "assign a weight in config/strategy_weights.json AND flip to active."
        ),
    ),
    StrategyEntry(
        StrategyName.ALPHA_FOREX,
        Track.SIDE_ENGINE,
        Status.DORMANT,
        note=(
            "side-engine only (dominance_allocator); not registered in the main "
            "StrategyEngine (chad.strategies.DEFERRED_STRATEGIES). FX universe "
            "not mapped to active bar/price context; 0 fills."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Accessors (deterministic, alphabetical for stable downstream output)
# ---------------------------------------------------------------------------

def all_entries() -> Tuple[StrategyEntry, ...]:
    """Return every declared strategy entry (declaration order)."""
    return _REGISTRY


def active_entries() -> Tuple[StrategyEntry, ...]:
    """Entries with status == active (the weighted trading set)."""
    return tuple(e for e in _REGISTRY if e.status is Status.ACTIVE)


def dormant_entries() -> Tuple[StrategyEntry, ...]:
    """Entries with status == dormant (declared, known, not trading)."""
    return tuple(e for e in _REGISTRY if e.status is Status.DORMANT)


def active_strategy_values() -> Tuple[str, ...]:
    """Alphabetical tuple of ACTIVE strategy string values (the 16)."""
    return tuple(sorted(e.name.value for e in _REGISTRY if e.status is Status.ACTIVE))


def dormant_strategy_values() -> Tuple[str, ...]:
    """Alphabetical tuple of DORMANT strategy string values (the 2)."""
    return tuple(sorted(e.name.value for e in _REGISTRY if e.status is Status.DORMANT))


def main_track_values() -> Tuple[str, ...]:
    """Alphabetical tuple of main-track strategy string values (registered handlers)."""
    return tuple(sorted(e.name.value for e in _REGISTRY if e.track is Track.MAIN))


def side_engine_values() -> Tuple[str, ...]:
    """Alphabetical tuple of side-engine strategy string values."""
    return tuple(sorted(e.name.value for e in _REGISTRY if e.track is Track.SIDE_ENGINE))


# ---------------------------------------------------------------------------
# Startup consistency assertion — the loud guard against future drift
# ---------------------------------------------------------------------------

class RegistryConsistencyError(RuntimeError):
    """Raised when a consumer has drifted from the canonical registry."""


def _load_weight_keys(weights_path: Optional[Path]) -> set:
    path = weights_path if weights_path is not None else WEIGHTS_PATH
    if not Path(path).is_file():
        raise RegistryConsistencyError(
            f"strategy registry consistency: weights file missing at {path}"
        )
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - corrupt file is an error path
        raise RegistryConsistencyError(
            f"strategy registry consistency: weights file unreadable at {path}: {exc}"
        ) from exc
    weights = doc.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise RegistryConsistencyError(
            f"strategy registry consistency: weights file {path} has no "
            "non-empty 'weights' object"
        )
    return set(weights.keys())


def assert_registry_consistency(*, weights_path: Optional[Path] = None) -> None:
    """Fail LOUD if any consumer has drifted from the canonical registry.

    Validates, raising :class:`RegistryConsistencyError` with a precise
    message on the first violated invariant:

      1. Every ``StrategyName`` enum member is declared exactly once here
         (no duplicates, no orphans).
      2. ACTIVE and DORMANT are disjoint and ACTIVE ∪ DORMANT == enum.
      3. ACTIVE == ``config/strategy_weights.json`` weights keys.
      4. ACTIVE == ``tier_manager._CANONICAL_STRATEGY_NAMES`` (tier-eligible).
      5. Track axis agrees with the handler-registration SSOT:
         main-track == ``chad.strategies.active_strategy_names()`` and
         side-engine == ``chad.strategies.DEFERRED_STRATEGIES``.

    This is intended to run at service/app init so registry drift can never
    again be silent. It performs only read-only checks and never mutates state.
    """
    # 1. exactly-once declaration; no orphans -------------------------------
    declared = [e.name for e in _REGISTRY]
    dupes = sorted({n.value for n in declared if declared.count(n) > 1})
    if dupes:
        raise RegistryConsistencyError(
            f"strategy registry: names declared more than once: {dupes}"
        )
    declared_set = set(declared)
    enum_set = set(StrategyName)
    missing = sorted(s.value for s in (enum_set - declared_set))
    if missing:
        raise RegistryConsistencyError(
            f"strategy registry: StrategyName members not declared: {missing} — "
            "every enum member must be declared exactly once (active or dormant)."
        )

    # 2. ACTIVE / DORMANT partition the enum --------------------------------
    active = {e.name for e in _REGISTRY if e.status is Status.ACTIVE}
    dormant = {e.name for e in _REGISTRY if e.status is Status.DORMANT}
    overlap = sorted(s.value for s in (active & dormant))
    if overlap:
        raise RegistryConsistencyError(
            f"strategy registry: names both active and dormant: {overlap}"
        )
    if (active | dormant) != enum_set:
        unclassified = sorted(s.value for s in (enum_set - (active | dormant)))
        raise RegistryConsistencyError(
            "strategy registry: ACTIVE ∪ DORMANT != enum; "
            f"unclassified={unclassified}"
        )

    active_vals = {s.value for s in active}

    # 3. ACTIVE == weights keys ---------------------------------------------
    weight_keys = _load_weight_keys(weights_path)
    if active_vals != weight_keys:
        only_registry = sorted(active_vals - weight_keys)
        only_weights = sorted(weight_keys - active_vals)
        raise RegistryConsistencyError(
            "strategy registry: ACTIVE set != config/strategy_weights.json keys; "
            f"active_without_weight={only_registry} "
            f"weight_without_active={only_weights} — assign/remove a weight or "
            "flip the registry status."
        )

    # 4. ACTIVE == tier active-eligible set ---------------------------------
    import chad.risk.tier_manager as _tm  # lazy: avoids import cycle at module load

    tier_vals = set(_tm._CANONICAL_STRATEGY_NAMES)
    if active_vals != tier_vals:
        only_registry = sorted(active_vals - tier_vals)
        only_tier = sorted(tier_vals - active_vals)
        raise RegistryConsistencyError(
            "strategy registry: ACTIVE set != tier_manager._CANONICAL_STRATEGY_NAMES; "
            f"active_not_in_tier={only_registry} tier_not_active={only_tier}."
        )

    # 5. track axis agrees with handler-registration SSOT -------------------
    from chad.strategies import (  # lazy: chad.strategies imports heavy handlers
        active_strategy_names as _registered,
        deferred_strategy_names as _deferred,
    )

    main_vals = {e.name.value for e in _REGISTRY if e.track is Track.MAIN}
    side_vals = {e.name.value for e in _REGISTRY if e.track is Track.SIDE_ENGINE}
    registered_vals = {s.value for s in _registered()}
    deferred_vals = {s.value for s in _deferred()}
    if main_vals != registered_vals:
        raise RegistryConsistencyError(
            "strategy registry: main-track set != chad.strategies registered "
            f"handlers; only_main={sorted(main_vals - registered_vals)} "
            f"only_registered={sorted(registered_vals - main_vals)}."
        )
    if side_vals != deferred_vals:
        raise RegistryConsistencyError(
            "strategy registry: side-engine set != chad.strategies.DEFERRED_"
            f"STRATEGIES; only_side={sorted(side_vals - deferred_vals)} "
            f"only_deferred={sorted(deferred_vals - side_vals)}."
        )

"""GAP-003/008/026 — Box 033 strategy registry parity tests.

Asserts the four-way reconciliation across:
  1. ``chad.types.StrategyName`` enum
  2. ``chad.strategies.DEFERRED_STRATEGIES`` (handler-not-registered set)
  3. ``chad.strategies.active_strategy_names()`` (the live _REGISTRY)
  4. ``chad.risk.tier_manager._CANONICAL_STRATEGY_NAMES``
  5. ``config/strategy_weights.json`` weights

Invariants enforced:
  - Every enum value is in (active_registry | deferred); no orphans.
  - active_registry == tier_canonical (the wildcard-expansion list MUST
    match the registered handlers — otherwise tier_state.enabled_strategies
    will silently drop registered strategies on SCALE).
  - active_registry ⊇ weights keys (every weighted strategy is registered).
  - weights keys ∩ deferred == ∅ (we never assign weight to a deferred
    strategy).
  - active_registry - weights ⊆ WEIGHT_DEFERRED_ALLOWLIST (strategies that
    are registered but intentionally have no allocator weight; see the
    pending-action artifact for each entry).

WEIGHT_DEFERRED_ALLOWLIST is the explicit, audited set of strategies that
are present in the active registry but NOT in strategy_weights.json. As
of Box 033 (2026-05-20) it contains exactly one entry:

  - ``alpha_intraday_micro`` — per CHAD Unified SSOT v9.1 §residual
    line 1285: "registered but conditionally inactive at runtime ...
    REQUIRES CONTEXT-BUILDER WIRING VERIFICATION before
    MICRO/STARTER/PRO_GROWTH testing." Adding it to the allowlist is
    the deterministic record that this state is *known*, not an
    accidental gap. See
    ``ops/pending_actions/BOX-033_alpha_intraday_micro_weight_policy.md``
    for the operator decision required to remove it from the allowlist.

If a new strategy is added to the enum/registry, this test will fail
loudly until either (a) it is given a weight in
``config/strategy_weights.json`` OR (b) it is added to
WEIGHT_DEFERRED_ALLOWLIST with a documented operator note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import FrozenSet

import pytest

from chad.risk.tier_manager import _CANONICAL_STRATEGY_NAMES
from chad.strategies import (
    DEFERRED_STRATEGIES,
    active_strategy_names,
    deferred_strategy_names,
)
from chad.types import StrategyName


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = REPO_ROOT / "config" / "strategy_weights.json"


# Strategies that are registered (active) but intentionally have NO entry
# in config/strategy_weights.json. Every addition here MUST be paired
# with a pending-action artifact documenting the operator decision.
WEIGHT_DEFERRED_ALLOWLIST: FrozenSet[str] = frozenset({
    # See ops/pending_actions/BOX-033_alpha_intraday_micro_weight_policy.md
    # SSOT v9.1 line 1285 residual: "registered but conditionally inactive
    # at runtime ... REQUIRES CONTEXT-BUILDER WIRING VERIFICATION before
    # MICRO/STARTER/PRO_GROWTH testing."
    "alpha_intraday_micro",
})


def _load_weights() -> dict:
    assert WEIGHTS_PATH.is_file(), (
        f"config/strategy_weights.json missing at {WEIGHTS_PATH}"
    )
    payload = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    weights = payload.get("weights")
    assert isinstance(weights, dict) and weights, (
        "strategy_weights.v1 payload must include a non-empty 'weights' dict"
    )
    return weights


def test_enum_orphans_none() -> None:
    """Every StrategyName enum value is either active-registered or deferred."""
    enum_vals = {s.value for s in StrategyName}
    active = {s.value for s in active_strategy_names()}
    deferred = {s.value for s in deferred_strategy_names()}
    orphans = enum_vals - (active | deferred)
    assert orphans == set(), (
        f"Enum members not in active OR deferred: {sorted(orphans)} — "
        "every StrategyName must be classified."
    )


def test_active_registry_matches_tier_canonical() -> None:
    """The tier_manager wildcard-expansion list MUST equal the active registry."""
    active = {s.value for s in active_strategy_names()}
    tier = set(_CANONICAL_STRATEGY_NAMES)
    only_in_active = active - tier
    only_in_tier = tier - active
    assert only_in_active == set() and only_in_tier == set(), (
        "tier_manager._CANONICAL_STRATEGY_NAMES out of sync with active "
        f"registry; only_in_active={sorted(only_in_active)} "
        f"only_in_tier={sorted(only_in_tier)}"
    )


def test_weights_are_subset_of_active_registry() -> None:
    """Every weight key MUST be an active strategy (no ghost weights)."""
    weights = _load_weights()
    active = {s.value for s in active_strategy_names()}
    ghosts = set(weights.keys()) - active
    assert ghosts == set(), (
        f"strategy_weights.json contains keys not in active registry: "
        f"{sorted(ghosts)} — remove them or re-register the strategy."
    )


def test_weights_disjoint_from_deferred() -> None:
    """Deferred strategies MUST NOT receive a weight."""
    weights = _load_weights()
    deferred = {s.value for s in deferred_strategy_names()}
    conflict = set(weights.keys()) & deferred
    assert conflict == set(), (
        f"Deferred strategies have weights assigned: {sorted(conflict)} — "
        "either remove DEFERRED_STRATEGIES membership or remove the weight."
    )


def test_active_without_weight_is_in_allowlist() -> None:
    """Active strategies without a weight MUST be in WEIGHT_DEFERRED_ALLOWLIST.

    This is the gate that catches "new strategy added to registry but
    forgotten in weights" mistakes. Either give it a weight or add to the
    allowlist with a pending-action note explaining why.
    """
    weights = _load_weights()
    active = {s.value for s in active_strategy_names()}
    missing_weight = active - set(weights.keys())
    undocumented = missing_weight - WEIGHT_DEFERRED_ALLOWLIST
    assert undocumented == set(), (
        f"Active strategies have no weight and are not in the documented "
        f"WEIGHT_DEFERRED_ALLOWLIST: {sorted(undocumented)} — either add a "
        "weight in config/strategy_weights.json or add to "
        "WEIGHT_DEFERRED_ALLOWLIST with a pending-action note."
    )


def test_allowlist_entries_are_still_active() -> None:
    """Every WEIGHT_DEFERRED_ALLOWLIST entry MUST still be in the active
    registry. If a name is here but no longer registered, the entry is
    stale and must be removed."""
    active = {s.value for s in active_strategy_names()}
    stale = WEIGHT_DEFERRED_ALLOWLIST - active
    assert stale == set(), (
        f"WEIGHT_DEFERRED_ALLOWLIST has stale entries no longer in active "
        f"registry: {sorted(stale)} — remove them."
    )


def test_pending_action_artifact_exists_for_each_allowlist_entry() -> None:
    """Each WEIGHT_DEFERRED_ALLOWLIST entry must point to a real pending-
    action artifact under ops/pending_actions/. The convention is
    BOX-<box-id>_<name>_weight_policy.md."""
    pending_dir = REPO_ROOT / "ops" / "pending_actions"
    assert pending_dir.is_dir(), f"missing pending-actions dir {pending_dir}"
    for name in WEIGHT_DEFERRED_ALLOWLIST:
        # Match by suffix to allow box-id prefix variation, but require
        # the strategy name to appear in some pending-action filename.
        candidates = sorted(pending_dir.glob(f"*{name}*policy.md"))
        assert candidates, (
            f"No pending-action artifact found for weight-deferred "
            f"strategy {name!r} (expected something matching "
            f"ops/pending_actions/*{name}*policy.md)."
        )


def test_deferred_set_is_a_frozenset_of_enum() -> None:
    """Sanity: DEFERRED_STRATEGIES must be a frozenset of StrategyName."""
    assert isinstance(DEFERRED_STRATEGIES, frozenset)
    for x in DEFERRED_STRATEGIES:
        assert isinstance(x, StrategyName)

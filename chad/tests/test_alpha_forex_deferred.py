"""
GAP-014 closure tests: ALPHA_FOREX deferred-by-design parity.

Pre-GAP-014, ALPHA_FOREX existed as a StrategyName enum member and had a
StrategyRiskLimits entry in chad/policy.py, but was absent from the
chad/strategies/__init__.py registry with no formal signal that this was
intentional. This left parity checks unable to distinguish "deferred by
design" from "missing by accident".

GAP-014 formalizes the deferred status by introducing
``chad.strategies.DEFERRED_STRATEGIES`` as the single source of truth.
These tests assert the invariants that resolution implies:

1. The active registry is exactly ``StrategyName - DEFERRED_STRATEGIES``.
2. ALPHA_FOREX is in DEFERRED_STRATEGIES, has a disabled policy entry,
   and is absent from the active registry by design.
3. The dynamic-caps weight config (config/strategy_weights.json) does not
   list alpha_forex, so the allocator's active-strategy set excludes it.
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.policy import build_default_strategy_limits
from chad.strategies import (
    DEFERRED_STRATEGIES,
    _build_registry,
    active_strategy_names,
    deferred_strategy_names,
)
from chad.types import StrategyName


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_strategy_registry_matches_active_strategy_policy() -> None:
    """Registry keys must equal StrategyName members minus DEFERRED_STRATEGIES."""
    registry_keys = set(_build_registry().keys())
    expected_active = set(StrategyName) - set(DEFERRED_STRATEGIES)
    assert registry_keys == expected_active, (
        f"registry/active-policy drift: "
        f"registry_only={sorted(s.value for s in registry_keys - expected_active)} "
        f"active_only={sorted(s.value for s in expected_active - registry_keys)}"
    )
    # Helper functions must agree with the same invariant.
    assert active_strategy_names() == registry_keys
    assert deferred_strategy_names() == DEFERRED_STRATEGIES
    # Active and deferred sets must be a clean partition of the enum.
    assert active_strategy_names().isdisjoint(deferred_strategy_names())
    assert active_strategy_names() | deferred_strategy_names() == set(StrategyName)


def test_alpha_forex_absent_from_active_registry_by_design() -> None:
    """ALPHA_FOREX is intentionally deferred, not accidentally missing."""
    # The enum member still exists for historical/runtime/data compatibility.
    assert StrategyName.ALPHA_FOREX in set(StrategyName)

    # It is recorded as deferred — the authoritative signal of intent.
    assert StrategyName.ALPHA_FOREX in DEFERRED_STRATEGIES
    assert StrategyName.ALPHA_FOREX in deferred_strategy_names()

    # Therefore it is NOT in the active registry.
    assert StrategyName.ALPHA_FOREX not in _build_registry()
    assert StrategyName.ALPHA_FOREX not in active_strategy_names()

    # The risk policy still carries a block for it (so any leaked signal
    # would be hard-blocked), but the block must be disabled.
    limits = build_default_strategy_limits()
    assert StrategyName.ALPHA_FOREX in limits, (
        "ALPHA_FOREX policy entry removed — the disabled block is the "
        "belt-and-suspenders guard against historical/leaked signals."
    )
    assert limits[StrategyName.ALPHA_FOREX].enabled is False


def test_dynamic_caps_active_strategies_do_not_include_alpha_forex() -> None:
    """
    config/strategy_weights.json is the source-of-truth allocator input.
    It must not list alpha_forex, so DynamicRiskAllocator never produces
    a non-zero cap for it on the active path.
    """
    weights_path = REPO_ROOT / "config" / "strategy_weights.json"
    assert weights_path.is_file(), f"missing config: {weights_path}"
    with weights_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    weights = doc.get("weights", {})
    assert isinstance(weights, dict) and weights, "strategy_weights.weights empty"
    assert "alpha_forex" not in weights, (
        "alpha_forex is in config/strategy_weights.json but is a deferred "
        "strategy — DynamicRiskAllocator would issue a cap for it."
    )
    # Every weighted strategy must correspond to an active enum member.
    active_values = {s.value for s in active_strategy_names()}
    unknown = sorted(set(weights.keys()) - active_values)
    assert not unknown, (
        f"strategy_weights.json contains keys not in the active registry: {unknown}"
    )

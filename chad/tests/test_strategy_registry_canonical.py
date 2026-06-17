"""Canonical strategy registry tests (chad.strategy_registry).

Covers the single-source-of-truth model that resolves the historical
three-way drift (enum 18 / tier 17 / weights 16):

  - registry internal consistency (declared once, clean active/dormant split);
  - the startup assertion PASSES on the current committed config;
  - the startup assertion FAILS LOUD on an injected divergence
    (a phantom weight, and a dropped tier name);
  - the runtime ACTIVE set is unchanged == the 16 weighted strategies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import chad.risk.tier_manager as tier_manager
from chad.strategy_registry import (
    RegistryConsistencyError,
    Status,
    Track,
    active_entries,
    active_strategy_values,
    all_entries,
    assert_registry_consistency,
    dormant_strategy_values,
    main_track_values,
    side_engine_values,
)
from chad.types import StrategyName


REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = REPO_ROOT / "config" / "strategy_weights.json"


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------

def test_every_enum_member_declared_exactly_once() -> None:
    declared = [e.name for e in all_entries()]
    assert len(declared) == len(set(declared)), "duplicate declarations present"
    assert set(declared) == set(StrategyName), "registry does not cover the enum"
    assert len(all_entries()) == 18


def test_active_dormant_partition_the_enum() -> None:
    active = {e.name for e in all_entries() if e.status is Status.ACTIVE}
    dormant = {e.name for e in all_entries() if e.status is Status.DORMANT}
    assert active.isdisjoint(dormant)
    assert active | dormant == set(StrategyName)
    assert len(active) == 16
    assert len(dormant) == 2


def test_dormant_set_is_exactly_micro_and_forex() -> None:
    assert set(dormant_strategy_values()) == {
        "alpha_intraday_micro",
        "alpha_forex",
    }


def test_dormant_tracks_are_as_declared() -> None:
    by_name = {e.name: e for e in all_entries()}
    micro = by_name[StrategyName.ALPHA_INTRADAY_MICRO]
    forex = by_name[StrategyName.ALPHA_FOREX]
    assert micro.track is Track.MAIN and micro.status is Status.DORMANT
    assert forex.track is Track.SIDE_ENGINE and forex.status is Status.DORMANT


def test_active_values_sorted_and_sixteen() -> None:
    vals = active_strategy_values()
    assert len(vals) == 16
    assert list(vals) == sorted(vals)
    assert len(active_entries()) == 16


# ---------------------------------------------------------------------------
# ACTIVE == the 16 weighted strategies (runtime trading set unchanged)
# ---------------------------------------------------------------------------

def _weight_keys() -> set:
    doc = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    return set(doc["weights"].keys())


def test_active_equals_weight_keys() -> None:
    assert set(active_strategy_values()) == _weight_keys()
    assert len(_weight_keys()) == 16


def test_active_equals_tier_canonical() -> None:
    assert set(active_strategy_values()) == set(tier_manager._CANONICAL_STRATEGY_NAMES)
    assert len(tier_manager._CANONICAL_STRATEGY_NAMES) == 16


def test_track_axis_matches_handler_registration_ssot() -> None:
    from chad.strategies import active_strategy_names, deferred_strategy_names

    assert set(main_track_values()) == {s.value for s in active_strategy_names()}
    assert set(side_engine_values()) == {s.value for s in deferred_strategy_names()}


# ---------------------------------------------------------------------------
# Startup assertion: passes on current config
# ---------------------------------------------------------------------------

def test_assert_passes_on_current_config() -> None:
    # Must not raise against the real committed config + live tier list.
    assert_registry_consistency()


# ---------------------------------------------------------------------------
# Startup assertion: FAILS LOUD on injected divergence
# ---------------------------------------------------------------------------

def _write_weights(path: Path, weights: dict) -> None:
    path.write_text(
        json.dumps({"schema_version": "strategy_weights.v1", "weights": weights}),
        encoding="utf-8",
    )


def test_assert_fails_on_phantom_weight(tmp_path: Path) -> None:
    """A weight for a name outside ACTIVE must trip the guard."""
    weights = {v: 0.05 for v in active_strategy_values()}
    weights["ghost_strategy"] = 0.01  # phantom: not in ACTIVE
    bad = tmp_path / "strategy_weights.json"
    _write_weights(bad, weights)
    with pytest.raises(RegistryConsistencyError) as exc:
        assert_registry_consistency(weights_path=bad)
    assert "weights" in str(exc.value).lower()


def test_assert_fails_on_missing_weight(tmp_path: Path) -> None:
    """An ACTIVE strategy dropped from the weights file must trip the guard."""
    weights = {v: 0.05 for v in active_strategy_values()}
    weights.pop("alpha")  # drop a real active strategy
    bad = tmp_path / "strategy_weights.json"
    _write_weights(bad, weights)
    with pytest.raises(RegistryConsistencyError):
        assert_registry_consistency(weights_path=bad)


def test_assert_fails_on_dropped_tier_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the tier active set drops a name from ACTIVE, the guard must fail."""
    shrunk = [n for n in tier_manager._CANONICAL_STRATEGY_NAMES if n != "alpha"]
    monkeypatch.setattr(tier_manager, "_CANONICAL_STRATEGY_NAMES", shrunk)
    with pytest.raises(RegistryConsistencyError) as exc:
        assert_registry_consistency()
    assert "tier" in str(exc.value).lower()


def test_assert_fails_on_extra_tier_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the tier active set gains a non-ACTIVE name (e.g. the dormant micro
    creeping back in), the guard must fail."""
    grown = list(tier_manager._CANONICAL_STRATEGY_NAMES) + ["alpha_intraday_micro"]
    monkeypatch.setattr(tier_manager, "_CANONICAL_STRATEGY_NAMES", grown)
    with pytest.raises(RegistryConsistencyError):
        assert_registry_consistency()


def test_assert_fails_on_missing_weights_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(RegistryConsistencyError):
        assert_registry_consistency(weights_path=missing)

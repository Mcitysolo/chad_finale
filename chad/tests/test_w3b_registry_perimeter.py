"""W3B-9 — strategy-registry perimeter guards (hard-in-tests leg of D6).

P1-1 shipped the canonical registry + 6-invariant startup tripwire
(2026-06-17, ff58803/1ff9a62), but seven hand-maintained lists sat outside
it. These pins are the HARD leg; the runtime leg (invariants 7/8 in
assert_registry_consistency) is warn-tier by decision D6 — config drift
must never brick the engine, CI must still fail loud.

Perimeter covered here:
- strategy_routing_diagnostics.KNOWN_STRATEGIES (hand tuple)
- chassis sleeve frozensets (dynamic_risk_allocator 50/30/20)
- dominance_allocator.DEFAULT_BASE_WEIGHTS (side-engine track)
- config/per_strategy_loss_limits.json keys
- config/regime_activation_matrix.json names
- dashboard + daily-report display maps (completed in this commit)
- the dead REAL_STRATEGIES allowlist stays dead (W3B-8)
"""

from __future__ import annotations

import json
from pathlib import Path

from chad.strategy_registry import (
    active_strategy_values,
    dormant_strategy_values,
)
from chad.types import StrategyName

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ACTIVE = set(active_strategy_values())
DORMANT = set(dormant_strategy_values())
DECLARED = {s.value for s in StrategyName}


def test_registry_shape_sanity():
    assert len(DECLARED) == 18 and len(ACTIVE) == 16 and len(DORMANT) == 2
    assert DORMANT == {"alpha_intraday_micro", "alpha_forex"}


# ---------------------------------------------------------------------------
# code constants
# ---------------------------------------------------------------------------


def test_known_strategies_bounded_by_registry():
    """DECLARED ⊇ KNOWN_STRATEGIES ⊇ ACTIVE: no undeclared name may enter the
    diagnostics universe, and every ACTIVE strategy must be diagnosable."""
    from chad.ops.strategy_routing_diagnostics import KNOWN_STRATEGIES

    known = set(KNOWN_STRATEGIES)
    assert known <= DECLARED, f"undeclared names: {sorted(known - DECLARED)}"
    assert ACTIVE <= known, f"ACTIVE names missing: {sorted(ACTIVE - known)}"


def test_chassis_sleeves_partition_active_exactly():
    """The 50/30/20 sleeves must partition ACTIVE: disjoint, union == ACTIVE.
    Currently true but unguarded — a strategy added to the registry without a
    sleeve (or landed in two) silently distorts enforce_chassis."""
    from chad.risk.dynamic_risk_allocator import (
        ADAPTIVE_STRATEGIES,
        ALPHA_STRATEGIES,
        BETA_STRATEGIES,
    )

    union = set(ALPHA_STRATEGIES) | set(BETA_STRATEGIES) | set(ADAPTIVE_STRATEGIES)
    assert union == ACTIVE, (
        f"sleeves != ACTIVE; missing={sorted(ACTIVE - union)} "
        f"extra={sorted(union - ACTIVE)}"
    )
    assert not (set(ALPHA_STRATEGIES) & set(BETA_STRATEGIES))
    assert not (set(ALPHA_STRATEGIES) & set(ADAPTIVE_STRATEGIES))
    assert not (set(BETA_STRATEGIES) & set(ADAPTIVE_STRATEGIES))


def test_dominance_base_weights_are_declared():
    from chad.risk.dominance_allocator import DEFAULT_BASE_WEIGHTS

    keys = set(DEFAULT_BASE_WEIGHTS)
    assert keys <= DECLARED, f"undeclared names: {sorted(keys - DECLARED)}"


def test_real_strategies_allowlist_stays_dead():
    """W3B-8 removed the unreferenced 7-name allowlist; attribution is
    denylist-based (PLACEHOLDER_STRATEGIES). Guard against resurrection."""
    from chad.execution import paper_exec_evidence_writer as pev

    assert not hasattr(pev, "REAL_STRATEGIES")
    assert pev.PLACEHOLDER_STRATEGIES == {"", "unknown", "manual", "paper_exec"}


# ---------------------------------------------------------------------------
# config files
# ---------------------------------------------------------------------------


def _load(rel):
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def test_loss_limit_keys_bounded_by_registry():
    limits = set(_load("config/per_strategy_loss_limits.json")["limits_usd"])
    assert limits <= DECLARED, f"undeclared names: {sorted(limits - DECLARED)}"
    assert ACTIVE <= limits, (
        f"ACTIVE names with no explicit loss limit: {sorted(ACTIVE - limits)} "
        "(falling to default_limit_usd silently)"
    )


def test_regime_matrix_names_bounded_by_registry():
    regimes = _load("config/regime_activation_matrix.json")["regimes"]
    for regime, names in regimes.items():
        extra = set(names) - DECLARED
        assert not extra, f"regime {regime!r} activates undeclared: {sorted(extra)}"


# ---------------------------------------------------------------------------
# display maps (completed in W3B-9 — every declared name renders)
# ---------------------------------------------------------------------------


def test_dashboard_display_map_covers_all_declared(monkeypatch):
    # chad.dashboard.api refuses to import without the dashboard password
    # (Step-9 deployment guard); satisfy it for this read-only pin.
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "w3b-perimeter-pin")
    import importlib

    api = importlib.import_module("chad.dashboard.api")

    missing = DECLARED - set(api.STRATEGY_NAMES)
    assert not missing, f"dashboard display names missing: {sorted(missing)}"
    assert set(api.STRATEGY_NAMES) <= DECLARED  # no stale/renamed keys


def test_report_display_map_covers_all_declared():
    from chad.ops.daily_chad_report import STRATEGY_NAMES

    missing = DECLARED - set(STRATEGY_NAMES)
    assert not missing, f"report display names missing: {sorted(missing)}"
    # legacy 'crypto' alias is the one intentional extra key
    assert set(STRATEGY_NAMES) - DECLARED == {"crypto"}


# ---------------------------------------------------------------------------
# runtime warn-tier invariants (7/8) exist and never raise on current config
# ---------------------------------------------------------------------------


def test_invariants_7_and_8_warn_never_raise(recwarn):
    """D6: the runtime leg emits UserWarning on perimeter drift and NEVER
    raises for config-side issues beyond the pre-existing hard invariants.
    On the current (consistent) config it must pass clean."""
    from chad.strategy_registry import assert_registry_consistency

    assert_registry_consistency()  # must not raise
    # current config: only the known dormant-in-tier warnings may appear
    for w in recwarn.list:
        assert "dormant" in str(w.message)


def test_invariant_7_warns_on_unknown_loss_limit_key(tmp_path):
    """A typo'd loss-limit key warns (never raises) at the runtime tripwire."""
    import warnings as w

    from chad.strategy_registry import assert_registry_consistency

    bad = {"limits_usd": {"alpha": 100.0, "alpha_typo": 50.0}, "default_limit_usd": 75.0}
    p = tmp_path / "per_strategy_loss_limits.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        assert_registry_consistency(loss_limits_path=p)
    msgs = [str(c.message) for c in caught]
    assert any("alpha_typo" in m for m in msgs), msgs


def test_invariant_8_warns_on_unknown_regime_name(tmp_path):
    import warnings as w

    from chad.strategy_registry import assert_registry_consistency

    bad = {"regimes": {"trending_bull": ["alpha", "no_such_strategy"]}}
    p = tmp_path / "regime_activation_matrix.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        assert_registry_consistency(regime_matrix_path=p)
    msgs = [str(c.message) for c in caught]
    assert any("no_such_strategy" in m for m in msgs), msgs

"""Regression tests for the halt/boost contradiction fix.

Locks the invariant that a strategy halted in
runtime/strategy_allocations.json never receives an aggressive winner
boost (>1.0) when the dynamic risk allocator builds dynamic_caps,
even if winner_scaling.json still publishes a stale boost.

The suppression must happen on the read path inside the allocator.
This module is forbidden from mutating winner_scaling.json or
strategy_allocations.json — the test fixture writes synthetic copies
in a tmp dir and points the allocator's runtime helpers at it via
monkeypatch.

The four required tests live here:
  - test_halted_strategy_boost_suppressed_to_neutral
  - test_non_halted_strategy_boost_preserved
  - test_halted_strategy_does_not_receive_aggressive_dynamic_caps_multiplier
  - test_delta_halt_boost_contradiction_regression
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chad.risk import dynamic_risk_allocator as dra
from chad.risk.dynamic_risk_allocator import (
    DynamicRiskAllocator,
    PortfolioSnapshot,
    StrategyAllocation,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture()
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand up a fake runtime/ directory and point the allocator at it.

    Writes a fresh winner_scaling.json with delta=1.5x boost and a
    strategy_allocations.json halting delta. Leaves regime_booster and
    tier_state absent (allocator falls back to neutral).
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    winner = {
        "schema_version": "winner_scaling.v1",
        "multipliers": {
            "alpha": 1.0,
            "beta": 1.0,
            "delta": 1.5,
            "gamma": 1.0,
        },
        "ts_utc": _now_iso(),
    }
    (runtime / "winner_scaling.json").write_text(json.dumps(winner))

    allocations = {
        "schema_version": "strategy_allocations.v1",
        "updated_at": _now_iso(),
        "allocations": {
            "delta": {
                "halted": True,
                "halt_reason": "consecutive_negative_15",
                "halted_at": _now_iso(),
                "cleared_at": None,
                "cleared_by": "",
                "consecutive_negative": 15,
            },
        },
    }
    (runtime / "strategy_allocations.json").write_text(json.dumps(allocations))

    monkeypatch.setattr(dra, "_runtime_dir", lambda: runtime)
    return runtime


def _build_allocator() -> DynamicRiskAllocator:
    weights = {
        "alpha": 0.4,
        "beta": 0.3,
        "delta": 0.2,
        "gamma": 0.1,
    }
    alloc = StrategyAllocation(weights=weights, source="test")
    return DynamicRiskAllocator(strategy_allocation=alloc, daily_risk_fraction=0.05)


def _snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(ibkr_equity=1_000_000.0, coinbase_equity=0.0)


def test_halted_strategy_boost_suppressed_to_neutral(
    fake_runtime: Path,
) -> None:
    """A halted strategy with a 1.5x winner_scaling boost must be
    clamped to 1.0 (neutral) inside _compute_caps_with_overlays. The
    suppression must come from the halt set the allocator reads at
    runtime, not from winner_scaling.json.
    """
    allocator = _build_allocator()
    details = allocator._compute_caps_with_overlays(snapshot=_snapshot())

    overlay = details["per_strategy_overlay"]["delta"]
    assert overlay["halt_clamp_applied"] is True
    assert overlay["winner_factor"] == 1.0, (
        "halted strategy must be clamped to neutral 1.0, not 1.5"
    )

    # winner_scaling.json must NOT have been mutated by the allocator.
    raw = json.loads((fake_runtime / "winner_scaling.json").read_text())
    assert raw["multipliers"]["delta"] == 1.5, (
        "allocator must suppress boost on the read path only — "
        "winner_scaling.json must remain unmodified"
    )


def test_non_halted_strategy_boost_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-halted strategy with a 1.5x winner_scaling boost must
    keep its 1.5x — the clamp must only fire for halted strategies.
    """
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    winner = {
        "schema_version": "winner_scaling.v1",
        "multipliers": {
            "alpha": 1.5,
            "beta": 1.0,
            "delta": 1.0,
            "gamma": 1.0,
        },
        "ts_utc": _now_iso(),
    }
    (runtime / "winner_scaling.json").write_text(json.dumps(winner))
    allocations = {
        "schema_version": "strategy_allocations.v1",
        "updated_at": _now_iso(),
        "allocations": {},
    }
    (runtime / "strategy_allocations.json").write_text(json.dumps(allocations))
    monkeypatch.setattr(dra, "_runtime_dir", lambda: runtime)

    allocator = _build_allocator()
    details = allocator._compute_caps_with_overlays(snapshot=_snapshot())

    alpha_overlay = details["per_strategy_overlay"]["alpha"]
    assert alpha_overlay["halt_clamp_applied"] is False
    assert alpha_overlay["winner_factor"] == 1.5, (
        "non-halted strategy must keep its full winner boost"
    )


def test_halted_strategy_does_not_receive_aggressive_dynamic_caps_multiplier(
    fake_runtime: Path,
) -> None:
    """build_payload (the function that materializes
    runtime/dynamic_caps.json) must surface the suppressed
    winner_factor. The published artifact must not show a halted
    strategy with winner_factor > 1.0.
    """
    allocator = _build_allocator()
    payload = allocator.build_payload(snapshot=_snapshot())

    delta_entry = payload["strategies"]["delta"]
    assert delta_entry["winner_factor"] <= 1.0, (
        f"dynamic_caps published winner_factor={delta_entry['winner_factor']} "
        "for halted strategy delta — boost suppression failed"
    )
    assert delta_entry["halt_clamp_applied"] is True

    # The dollar cap must reflect the clamped winner_factor: it must
    # be strictly smaller than what the un-clamped 1.5x would have
    # produced (base_cap * 1.0 * regime_mult <= base_cap * 1.5 * regime_mult).
    base = delta_entry["base_cap_pre_overlay"]
    regime = delta_entry["regime_factor"]
    unclamped = base * 1.5 * regime
    assert delta_entry["dollar_cap"] < unclamped + 1e-9, (
        "halted strategy dollar_cap must be smaller than the unclamped "
        "1.5x boost would have produced"
    )


def test_delta_halt_boost_contradiction_regression(
    fake_runtime: Path,
) -> None:
    """Specific regression for the 2026-05-05 contradiction: delta
    halted (consecutive_negative_15) while winner_scaling.json still
    showed delta=1.5. After the fix, both the per-strategy overlay
    and the dynamic_caps payload must show delta with winner_factor
    clamped to 1.0 and halt_clamp_applied=True. winner_scaling.json
    and strategy_allocations.json must not be mutated.
    """
    allocator = _build_allocator()

    # Snapshot state before
    winner_before = json.loads((fake_runtime / "winner_scaling.json").read_text())
    allocations_before = json.loads(
        (fake_runtime / "strategy_allocations.json").read_text()
    )

    # Compute and publish payload
    payload = allocator.build_payload(snapshot=_snapshot())
    delta = payload["strategies"]["delta"]
    assert delta["winner_factor"] == 1.0
    assert delta["halt_clamp_applied"] is True

    # Snapshot state after — runtime files must be byte-identical
    winner_after = json.loads((fake_runtime / "winner_scaling.json").read_text())
    allocations_after = json.loads(
        (fake_runtime / "strategy_allocations.json").read_text()
    )
    assert winner_after == winner_before, (
        "winner_scaling.json was mutated — suppression must be read-only"
    )
    assert allocations_after == allocations_before, (
        "strategy_allocations.json was mutated — halt must be preserved untouched"
    )

    # Halt itself preserved in the source file
    assert (
        allocations_after["allocations"]["delta"]["halted"] is True
    ), "delta halt must remain set after allocator runs"

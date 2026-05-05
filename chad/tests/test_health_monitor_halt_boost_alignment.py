"""
Health-monitor halt+boost alignment tests (Channel 2 Stage 2).

Locks the contract that the health monitor must classify halt+boost
contradictions using the *effective* per-strategy winner_factor
published in ``dynamic_caps.json`` — not the raw multiplier from
``winner_scaling.json``.

Background: the orchestrator's halt clamp publishes
``dynamic_caps.strategies.{name}.winner_factor<=1.0`` with
``halt_clamp_applied=true`` for any halted strategy. A halted strategy
whose raw winner_scaling multiplier is 1.5x but whose effective
winner_factor has been clamped to 1.0 is NOT a contradiction — the
allocator never sizes the strategy at 1.5x. The health monitor must
align with that semantic so it does not page on a resolved condition.

Tests pin:
- R14 stays silent when a halted strategy's raw boost has been
  effectively clamped to <=1.0 in dynamic_caps.
- R14 warns when a halted strategy's effective winner_factor remains
  >1.0 (true contradiction).
- R14 warns when dynamic_caps is missing/empty so the clamp cannot be
  verified.
- R13 surfaces the SCR exclusion classification so the gap can be
  triaged (pnl_zero / pnl_untrusted / rejected / partial / quarantine)
  without manual log inspection.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from chad.ops import health_monitor_rules as hmr


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def runtime(tmp_path: Path):
    """Redirect the rule engine's RUNTIME path to a tmp dir."""
    rt = tmp_path / "runtime"
    rt.mkdir()
    with patch.object(hmr, "RUNTIME", rt):
        yield rt


def _halted_alloc(strategy: str = "delta") -> dict:
    return {
        "schema_version": "strategy_allocations.v1",
        "allocations": {
            strategy: {
                "halted": True,
                "halt_reason": "consecutive_negative_15",
                "halted_at": "2026-05-04T13:30:15+00:00",
                "consecutive_negative": 15,
                "cleared_at": None,
                "cleared_by": "",
            }
        },
    }


def _raw_winner_scaling(strategy: str, mult: float) -> dict:
    return {
        "schema_version": "winner_scaling.v1",
        "multipliers": {strategy: mult, "alpha": 1.0},
        "max_multiplier": 1.5,
        "min_multiplier": 0.5,
    }


def _dynamic_caps_with(strategy: str, winner_factor: float,
                       halt_clamp_applied: bool) -> dict:
    return {
        "strategies": {
            strategy: {
                "winner_factor": winner_factor,
                "halt_clamp_applied": halt_clamp_applied,
                "dollar_cap": 100.0,
            },
            "alpha": {
                "winner_factor": 1.0,
                "halt_clamp_applied": False,
                "dollar_cap": 500.0,
            },
        }
    }


# ─── R14: halt+boost rule ───────────────────────────────────────────────────

def test_health_monitor_does_not_warn_when_halted_boost_is_effectively_clamped(
    runtime: Path,
):
    """The live runtime state: delta is halted, raw winner_scaling=1.5,
    but dynamic_caps clamps winner_factor=1.0 with halt_clamp_applied=true.
    R14 must NOT emit a halt+boost finding — the contradiction has
    already been resolved by the allocator's clamp."""
    _write(runtime / "strategy_allocations.json", _halted_alloc("delta"))
    _write(runtime / "winner_scaling.json", _raw_winner_scaling("delta", 1.5))
    _write(
        runtime / "dynamic_caps.json",
        _dynamic_caps_with("delta", winner_factor=1.0,
                           halt_clamp_applied=True),
    )

    findings: list = []
    hmr.rule_halted_with_unclamped_boost(findings)

    r14 = [f for f in findings if f.rule_id == "R14"]
    assert r14 == [], (
        "R14 must stay silent when the halt clamp has already suppressed "
        f"the raw boost (effective winner_factor<=1.0). Got: {r14}"
    )


def test_health_monitor_warns_when_halted_strategy_effective_boost_above_one(
    runtime: Path,
):
    """A halted strategy whose effective winner_factor in dynamic_caps
    remains >1.0 is a real contradiction: the allocator would size it
    aggressively while it is supposedly halted. R14 must warn."""
    _write(runtime / "strategy_allocations.json", _halted_alloc("delta"))
    _write(runtime / "winner_scaling.json", _raw_winner_scaling("delta", 1.5))
    _write(
        runtime / "dynamic_caps.json",
        _dynamic_caps_with("delta", winner_factor=1.5,
                           halt_clamp_applied=False),
    )

    findings: list = []
    hmr.rule_halted_with_unclamped_boost(findings)

    r14 = [f for f in findings if f.rule_id == "R14"]
    assert len(r14) == 1, f"expected exactly one R14 finding, got {r14}"
    f = r14[0]
    assert f.severity == "WARNING"
    assert "delta" in f.title
    assert "winner_factor" in f.evidence
    assert "halt_clamp_applied=False" in f.evidence


def test_health_monitor_warns_when_dynamic_caps_missing_and_raw_boost_above_one(
    runtime: Path,
):
    """If dynamic_caps.json is missing or has no per-strategy entry, the
    health monitor cannot verify the halt clamp suppressed the raw boost.
    R14 must warn so the operator investigates rather than silently
    trusting a stale raw value."""
    _write(runtime / "strategy_allocations.json", _halted_alloc("delta"))
    _write(runtime / "winner_scaling.json", _raw_winner_scaling("delta", 1.5))
    # NOTE: dynamic_caps.json deliberately not written.

    findings: list = []
    hmr.rule_halted_with_unclamped_boost(findings)

    r14 = [f for f in findings if f.rule_id == "R14"]
    assert len(r14) == 1, (
        f"R14 must warn when dynamic_caps is missing AND raw boost>1.0; "
        f"got {r14}"
    )
    assert "unverified" in r14[0].title.lower()
    assert "dynamic_caps.json missing" in r14[0].evidence


def test_health_monitor_does_not_warn_when_halted_strategy_has_no_raw_boost(
    runtime: Path,
):
    """A halted strategy with no raw boost (raw=1.0) is not a candidate
    for the halt+boost rule — there is nothing to clamp. R14 must stay
    silent. Prevents R14 spam every cycle."""
    _write(runtime / "strategy_allocations.json", _halted_alloc("delta"))
    _write(runtime / "winner_scaling.json", _raw_winner_scaling("delta", 1.0))
    _write(
        runtime / "dynamic_caps.json",
        _dynamic_caps_with("delta", winner_factor=1.0,
                           halt_clamp_applied=True),
    )

    findings: list = []
    hmr.rule_halted_with_unclamped_boost(findings)
    assert [f for f in findings if f.rule_id == "R14"] == []


def test_health_monitor_r14_silent_when_no_halts(runtime: Path):
    """No halted strategies → R14 is a no-op."""
    _write(runtime / "strategy_allocations.json",
           {"schema_version": "strategy_allocations.v1",
            "allocations": {}})
    _write(runtime / "winner_scaling.json", _raw_winner_scaling("delta", 1.5))
    _write(
        runtime / "dynamic_caps.json",
        _dynamic_caps_with("delta", winner_factor=1.5,
                           halt_clamp_applied=False),
    )

    findings: list = []
    hmr.rule_halted_with_unclamped_boost(findings)
    assert findings == []


# ─── R13: SCR gap classification ───────────────────────────────────────────

def test_scr_gap_classification_surfaces_pnl_zero_if_available_or_other_gap(
    runtime: Path,
):
    """R13 must surface the SCR exclusion classification (untrusted /
    manual / nonfinite / partial / pnl_zero) so the operator can confirm
    the gap is composed of legitimate exclusions rather than a counting
    bug. The current paper runtime has 28 raw vs 7 effective with most
    of the gap being pnl_untrusted/rejected fills — that classification
    must appear in the finding description, not just a raw gap count."""
    _write(runtime / "pnl_state.json", {
        "schema_version": "pnl_state.v1",
        "trade_count": 28,
        "realized_pnl": 5.65,
    })
    _write(runtime / "scr_state.json", {
        "schema_version": "scr_state.v1",
        "state": "WARMUP",
        "stats": {
            "total_trades": 18,
            "effective_trades": 7,
            "excluded_untrusted": 18,
            "excluded_manual": 0,
            "excluded_nonfinite": 0,
        },
    })

    findings: list = []
    hmr.rule_scr_effective_trades_gap(findings)

    r13 = [f for f in findings if f.rule_id == "R13"]
    assert len(r13) == 1, f"expected R13 to fire on 28-vs-7 gap, got {r13}"
    desc = r13[0].description
    assert "excluded_untrusted=18" in desc, (
        "R13 must surface the SCR untrusted-exclusion count so the "
        "operator can tell at a glance whether the gap is legitimate. "
        f"Got: {desc}"
    )
    # Severity must remain INFO — the gap is informational while in WARMUP.
    assert r13[0].severity == "INFO"


def test_scr_gap_classification_surfaces_unclassified_when_breakdown_missing(
    runtime: Path,
):
    """If SCR publishes a gap but no exclusion breakdown, R13 must flag
    the residual as 'unclassified=N' so the operator knows the gap is
    real but the classification is incomplete."""
    _write(runtime / "pnl_state.json", {
        "schema_version": "pnl_state.v1",
        "trade_count": 30,
        "realized_pnl": 0.0,
    })
    _write(runtime / "scr_state.json", {
        "schema_version": "scr_state.v1",
        "state": "WARMUP",
        "stats": {"effective_trades": 5},
    })

    findings: list = []
    hmr.rule_scr_effective_trades_gap(findings)

    r13 = [f for f in findings if f.rule_id == "R13"]
    assert len(r13) == 1
    assert "unclassified=25" in r13[0].description

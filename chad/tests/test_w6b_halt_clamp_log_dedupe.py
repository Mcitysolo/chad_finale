"""
chad/tests/test_w6b_halt_clamp_log_dedupe.py

W6B-9 (P2-3) — HALT_BOOST_SUPPRESSED log de-duplication.

The clamp behaviour is covered by test_dynamic_risk_allocator_halt_boost.py and
is NOT re-tested here. These tests pin only the narration contract:

  1. the structural record (applied_overlays[...]["halt_clamp_applied"]) is
     untouched — it is what health_monitor and the exterminator actually read,
     so a logging change must not disturb it;
  2. a sustained clamp narrates ONCE, not once per cycle (the actual P2-3
     complaint);
  3. a released clamp narrates once too — a signal that did not exist before;
  4. the line is INFO, not WARNING.
"""

from __future__ import annotations

import logging

import pytest

from chad.risk import dynamic_risk_allocator as dra


@pytest.fixture(autouse=True)
def _clean_dedupe_cache():
    """The dedupe cache is module-level and per-process by design; isolate it."""
    dra._HALT_CLAMP_LOGGED.clear()
    yield
    dra._HALT_CLAMP_LOGGED.clear()


def _halt_lines(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "HALT_BOOST_SUPPRESSED" in r.getMessage()]


def test_sustained_clamp_narrates_once_not_once_per_cycle(caplog):
    """The P2-3 complaint: a strategy halted for days logged every cycle."""
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    for _ in range(50):  # 50 allocator cycles with the clamp still engaged
        dra._log_halt_clamp_transition("gamma", True)

    lines = _halt_lines(caplog)
    assert len(lines) == 1, f"expected 1 line for 50 clamped cycles, got {len(lines)}"
    assert "transition=engaged" in lines[0].getMessage()


def test_engage_and_release_each_narrate_once(caplog):
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    for _ in range(5):
        dra._log_halt_clamp_transition("delta", True)
    for _ in range(5):
        dra._log_halt_clamp_transition("delta", False)

    msgs = [r.getMessage() for r in _halt_lines(caplog)]
    assert len(msgs) == 2, msgs
    assert "transition=engaged" in msgs[0]
    assert "transition=released" in msgs[1]


def test_never_clamped_strategy_is_silent(caplog):
    """An unclamped strategy must not announce a release it never engaged."""
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    for _ in range(10):
        dra._log_halt_clamp_transition("alpha", False)

    assert _halt_lines(caplog) == []


def test_line_is_info_not_warning(caplog):
    """P2-3's residue was WARNING-level noise; the record is structural."""
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    dra._log_halt_clamp_transition("omega_macro", True)

    lines = _halt_lines(caplog)
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert lines[0].levelno < logging.WARNING


def test_re_engage_after_release_narrates_again(caplog):
    """Dedupe must not silence a genuinely new halt episode."""
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    dra._log_halt_clamp_transition("gamma", True)
    dra._log_halt_clamp_transition("gamma", False)
    dra._log_halt_clamp_transition("gamma", True)

    msgs = [r.getMessage() for r in _halt_lines(caplog)]
    assert len(msgs) == 3, msgs
    assert "engaged" in msgs[0] and "released" in msgs[1] and "engaged" in msgs[2]


def test_strategies_dedupe_independently(caplog):
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    for _ in range(3):
        dra._log_halt_clamp_transition("gamma", True)
        dra._log_halt_clamp_transition("delta", True)

    msgs = [r.getMessage() for r in _halt_lines(caplog)]
    assert len(msgs) == 2, msgs
    assert any("strategy=gamma" in m for m in msgs)
    assert any("strategy=delta" in m for m in msgs)


def test_structural_record_survives_the_logging_change(
    tmp_path, monkeypatch, caplog
):
    """per_strategy_overlay is the consumed evidence (health_monitor_rules.py:591)
    — prove it still carries the clamp on EVERY cycle even though only the first
    cycle narrates. Mirrors test_dynamic_risk_allocator_halt_boost.py's fixture:
    synthetic runtime in tmp_path, never the live files."""
    import json
    from datetime import datetime, timezone

    from chad.risk.dynamic_risk_allocator import (
        DynamicRiskAllocator,
        PortfolioSnapshot,
        StrategyAllocation,
    )

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "winner_scaling.json").write_text(
        json.dumps({
            "schema_version": "winner_scaling.v1",
            "multipliers": {"delta": 1.5, "alpha": 1.0},
            "ts_utc": now,
        })
    )
    (runtime / "strategy_allocations.json").write_text(
        json.dumps({
            "schema_version": "strategy_allocations.v1",
            "updated_at": now,
            "allocations": {"delta": {"halted": True, "halt_reason": "test"}},
        })
    )
    monkeypatch.setattr(dra, "_runtime_dir", lambda: runtime)
    caplog.set_level(logging.INFO, logger=dra.logger.name)

    allocator = DynamicRiskAllocator(
        strategy_allocation=StrategyAllocation(
            weights={"alpha": 0.5, "delta": 0.5}, source="test"
        ),
        daily_risk_fraction=0.05,
    )
    snap = PortfolioSnapshot(ibkr_equity=1_000_000.0, coinbase_equity=0.0)

    # Three consecutive cycles: the log dedupes after the first, the structural
    # record must NOT.
    for _ in range(3):
        details = allocator._compute_caps_with_overlays(snapshot=snap)
        overlay = details["per_strategy_overlay"]["delta"]
        assert overlay["halt_clamp_applied"] is True
        assert overlay["winner_factor"] == 1.0

    assert len(_halt_lines(caplog)) == 1, "3 cycles must narrate once"

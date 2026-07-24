"""TIER1-B (PRE-LIVE BLOCKING): the published SCR sizing_factor must be applied
to intent quantity in EVERY non-PAUSED throttled state, not only CAUTIOUS.

These are the regression locks that make a published-but-unapplied (decorative)
factor impossible to re-introduce silently:

1. ``apply_scr_sizing`` / ``scr_sizing_should_apply`` — the single source of
   truth, per state, plus the kill-switch.
2. Intent-quantity assertion per state (WARMUP/CAUTIOUS/CONFIDENT/PAUSED) — the
   quantity actually reflects the published factor.
3. Source pin — the live loop gates on ``scr_sizing_should_apply`` and no longer
   carries the old CAUTIOUS-only literal gate.
4. EX021 sentinel — a published sub-unity factor that is disabled or unapplied
   trips a CRITICAL finding.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chad.risk.scr_sizing import (
    apply_scr_sizing,
    scr_sizing_apply_enabled,
    scr_sizing_should_apply,
)

ext = importlib.import_module("chad.ops.exterminator")

# The real governor ladder (chad/analytics/shadow_confidence_router.py).
STATE_FACTORS = {
    "WARMUP": 0.10,
    "CAUTIOUS": 0.25,
    "CONFIDENT": 1.00,
    "PAUSED": 0.00,
}


# ---------------------------------------------------------------------------
# 1. Policy predicate + kill-switch
# ---------------------------------------------------------------------------

def test_should_apply_true_for_throttled_states(monkeypatch):
    monkeypatch.delenv("CHAD_SCR_SIZING_APPLY", raising=False)
    assert scr_sizing_should_apply("WARMUP", 0.10) is True
    assert scr_sizing_should_apply("CAUTIOUS", 0.25) is True


def test_should_apply_false_for_noop_and_hardblock(monkeypatch):
    monkeypatch.delenv("CHAD_SCR_SIZING_APPLY", raising=False)
    # 1.0 (CONFIDENT) is a no-op; 0.0 (PAUSED) is a hard-block, not a scale.
    assert scr_sizing_should_apply("CONFIDENT", 1.00) is False
    assert scr_sizing_should_apply("PAUSED", 0.00) is False


def test_should_apply_false_on_nonfinite_or_garbage(monkeypatch):
    monkeypatch.delenv("CHAD_SCR_SIZING_APPLY", raising=False)
    assert scr_sizing_should_apply("WARMUP", float("nan")) is False
    assert scr_sizing_should_apply("WARMUP", None) is False  # type: ignore[arg-type]


def test_kill_switch_default_on_and_disables(monkeypatch):
    monkeypatch.delenv("CHAD_SCR_SIZING_APPLY", raising=False)
    assert scr_sizing_apply_enabled() is True
    for token in ("0", "false", "no", "off", "OFF", "False"):
        monkeypatch.setenv("CHAD_SCR_SIZING_APPLY", token)
        assert scr_sizing_apply_enabled() is False
        assert scr_sizing_should_apply("WARMUP", 0.10) is False
    monkeypatch.setenv("CHAD_SCR_SIZING_APPLY", "1")
    assert scr_sizing_apply_enabled() is True


# ---------------------------------------------------------------------------
# 2. Scaling math + intent-quantity per state
# ---------------------------------------------------------------------------

def test_apply_scaling_math_equities_floor():
    # Equities floor to whole units, minimum 1.
    assert apply_scr_sizing(100, 0.10, "STK") == 10.0
    assert apply_scr_sizing(37, 0.25, "STK") == 9.0  # floor(9.25)
    assert apply_scr_sizing(3, 0.10, "STK") == 1.0   # floor(0.3)=0 -> min 1


def test_apply_scaling_math_futures_round():
    # Futures round to nearest whole contract, minimum 1.
    assert apply_scr_sizing(10, 0.25, "FUT") == 2.0   # round(2.5) -> banker's 2
    assert apply_scr_sizing(100, 0.10, "FUT") == 10.0
    assert apply_scr_sizing(2, 0.10, "FUT") == 1.0    # round(0.2)=0 -> min 1


def test_cautious_math_byte_identical_to_former_block():
    # The former CAUTIOUS block: FUT round(min1), else floor(min1). Unchanged.
    import math
    for raw in (1, 7, 10, 33, 250):
        for sec in ("STK", "FUT", "OPT", ""):
            f = 0.25
            if sec == "FUT":
                expected = max(1.0, round(raw * f))
            else:
                expected = max(1.0, float(math.floor(raw * f)))
            assert apply_scr_sizing(raw, f, sec) == expected


def _scale_intent_like_live_loop(state, factor, raw_qty, sec_type):
    """Mirror the live_loop.py TIER1-B block exactly."""
    intent = SimpleNamespace(quantity=float(raw_qty), sec_type=sec_type, symbol="X")
    if scr_sizing_should_apply(state, factor):
        scaled = apply_scr_sizing(intent.quantity, factor, sec_type)
        if scaled != intent.quantity:
            intent.quantity = scaled
    return intent.quantity


def test_intent_quantity_reflects_published_factor_per_state(monkeypatch):
    monkeypatch.delenv("CHAD_SCR_SIZING_APPLY", raising=False)
    # WARMUP: 100 * 0.10 = 10 (the bug class — previously NOT applied)
    assert _scale_intent_like_live_loop("WARMUP", STATE_FACTORS["WARMUP"], 100, "STK") == 10.0
    # CAUTIOUS: 100 * 0.25 = 25 (unchanged behaviour)
    assert _scale_intent_like_live_loop("CAUTIOUS", STATE_FACTORS["CAUTIOUS"], 100, "STK") == 25.0
    # CONFIDENT: factor 1.0 -> untouched
    assert _scale_intent_like_live_loop("CONFIDENT", STATE_FACTORS["CONFIDENT"], 100, "STK") == 100.0
    # PAUSED: factor 0.0 -> not scaled here (hard-blocked upstream) -> untouched
    assert _scale_intent_like_live_loop("PAUSED", STATE_FACTORS["PAUSED"], 100, "STK") == 100.0


def test_warmup_no_longer_full_size(monkeypatch):
    monkeypatch.delenv("CHAD_SCR_SIZING_APPLY", raising=False)
    # The decorative-control regression: WARMUP at 0.10 must shrink the order.
    raw = 100
    scaled = _scale_intent_like_live_loop("WARMUP", 0.10, raw, "STK")
    assert scaled < raw
    assert scaled == 10.0


# ---------------------------------------------------------------------------
# 3. Source pin — live loop wiring cannot silently re-narrow
# ---------------------------------------------------------------------------

def test_live_loop_uses_predicate_not_cautious_literal():
    src = Path("chad/core/live_loop.py").read_text(encoding="utf-8")
    assert "_scr_should_apply(_scr_state_val, _scr_sizing)" in src, (
        "live_loop must gate SCR sizing on the shared predicate"
    )
    assert 'scr_sizing_application.json' in src, (
        "live_loop must write the EX021 evidence marker each cycle"
    )
    # The former CAUTIOUS-only literal gate must be gone.
    assert '_scr_state_val == "CAUTIOUS" and _scr_sizing' not in src


# ---------------------------------------------------------------------------
# 4. EX021 sentinel — decorative / unapplied factor trips CRITICAL
# ---------------------------------------------------------------------------

def _mk_sentinel(tmp_path: Path):
    return ext.Exterminator(runtime_dir=tmp_path)


def _scr(state="WARMUP", factor=0.10):
    return {"state": state, "sizing_factor": factor,
            "stats": {"effective_trades": 71}}


def _write_marker(tmp_path: Path, **overrides):
    payload = {
        "schema_version": "scr_sizing_application.v1",
        "scr_state": "WARMUP",
        "published_sizing_factor": 0.10,
        "application_enabled": True,
        "applied": True,
        "intents_in_cycle": 3,
    }
    payload.update(overrides)
    (tmp_path / "scr_sizing_application.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_ex021_clean_when_applied(tmp_path):
    _write_marker(tmp_path, applied=True, application_enabled=True)
    findings = _mk_sentinel(tmp_path).check_scr_sizing_applied(_scr())
    assert findings == []


def test_ex021_ignores_unity_factor(tmp_path):
    # CONFIDENT publishes 1.0 — nothing to apply, no finding even without a marker.
    findings = _mk_sentinel(tmp_path).check_scr_sizing_applied(_scr("CONFIDENT", 1.0))
    assert findings == []


def test_ex021_critical_when_disabled(tmp_path):
    _write_marker(tmp_path, application_enabled=False, applied=False)
    findings = _mk_sentinel(tmp_path).check_scr_sizing_applied(_scr())
    assert len(findings) == 1
    assert findings[0].id == "EX021"
    assert findings[0].severity == ext.SEVERITY_CRITICAL
    assert "DECORATIVE" in findings[0].title


def test_ex021_critical_when_published_not_applied(tmp_path):
    _write_marker(tmp_path, application_enabled=True, applied=False)
    findings = _mk_sentinel(tmp_path).check_scr_sizing_applied(_scr())
    assert len(findings) == 1
    assert findings[0].id == "EX021"
    assert findings[0].severity == ext.SEVERITY_CRITICAL
    assert "NOT applied" in findings[0].title


def test_ex021_warns_when_marker_missing(tmp_path):
    findings = _mk_sentinel(tmp_path).check_scr_sizing_applied(_scr())
    assert len(findings) == 1
    assert findings[0].id == "EX021"
    assert findings[0].severity == ext.SEVERITY_WARNING


def test_ex021_warns_on_stale_marker_factor(tmp_path):
    _write_marker(tmp_path, published_sizing_factor=0.25, applied=True)
    findings = _mk_sentinel(tmp_path).check_scr_sizing_applied(_scr("WARMUP", 0.10))
    assert len(findings) == 1
    assert findings[0].severity == ext.SEVERITY_WARNING
    assert "stale" in findings[0].title.lower()

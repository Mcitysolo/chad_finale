"""TIER1-A (cosmetic): the dashboard Training-card "Next:" line must track the
current SCR STATE, never the runaway sharpe-derived score.

Reproduces the reported bug: WARMUP with sharpe_like=0.444 (score 53) rendered
"Next: Confident — Unlocks 50% size at score 60", skipping the real next rung
(Cautious/25%). After the clamp it must read Cautious while the state is WARMUP,
regardless of how high the cosmetic score runs.
"""
from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def api(monkeypatch):
    # The dashboard module requires CHAD_DASHBOARD_PASSWORD at import time.
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test_value_12345")
    sys.modules.pop("chad.dashboard.api", None)
    return importlib.import_module("chad.dashboard.api")


def _scr(state: str, sharpe: float, *, sizing: float = 0.10, trades: int = 71):
    return {
        "state": state,
        "sizing_factor": sizing,
        "stats": {
            "effective_trades": trades,
            "win_rate": 0.042,
            "sharpe_like": sharpe,
        },
    }


def test_reported_bug_warmup_high_score_shows_cautious_not_confident(api):
    # sharpe 0.444 -> score 53; pre-fix this skipped to "Confident".
    assert api._sharpe_to_score(0.444) == 53
    status = api.StateBuilder()._chad_status(_scr("WARMUP", 0.444))
    assert status["next_level"] == "Cautious"
    assert "25%" in status["next_level_detail"]
    assert "score 30" in status["next_level_detail"]


def test_warmup_low_score_also_shows_cautious(api):
    status = api.StateBuilder()._chad_status(_scr("WARMUP", 0.05))  # score < 30
    assert status["next_level"] == "Cautious"


def test_cautious_state_shows_confident_next(api):
    # Even with a very high score, CAUTIOUS's immediate successor is Confident.
    status = api.StateBuilder()._chad_status(_scr("CAUTIOUS", 0.95, sizing=0.25))
    assert status["next_level"] == "Confident"
    assert "50%" in status["next_level_detail"]


def test_performance_score_is_unclamped_raw_score(api):
    # The clamp only steers the "Next" label; the displayed score stays real.
    status = api.StateBuilder()._chad_status(_scr("WARMUP", 0.444))
    assert status["performance_score"] == 53


def test_progress_never_exceeds_100(api):
    status = api.StateBuilder()._chad_status(_scr("WARMUP", 0.444))
    assert 0 <= status["progress_to_next_level_pct"] <= 100


def test_confident_state_passes_score_through_unclamped(api):
    # No ceiling for CONFIDENT/other — raw score drives the (fictional) upper
    # rungs; this documents that TIER1-A only fixes the WARMUP/CAUTIOUS skip.
    status = api.StateBuilder()._chad_status(_scr("CONFIDENT", 1.0, sizing=1.0))
    assert status["next_level"] in {"Full Send", "Max"}

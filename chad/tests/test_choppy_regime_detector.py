"""Tests for chad/analytics/choppy_regime_detector.py."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.analytics import choppy_regime_detector as crd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trending_bars(n: int = 35, start: float = 100.0, step: float = 1.0) -> list:
    """Strong, monotonic uptrend — high ADX, no flips, no failed breakouts."""
    bars = []
    p = start
    for i in range(n):
        o = p
        c = p + step
        h = c + 0.05
        l = o - 0.05
        bars.append({"open": o, "high": h, "low": l, "close": c, "ts_utc": f"d{i}"})
        p = c
    return bars


def _flippy_bars(n: int = 35, mid: float = 100.0, amp: float = 1.0) -> list:
    """Tight oscillation around `mid` — high direction-flip count."""
    bars = []
    for i in range(n):
        c = mid + (amp if i % 2 == 0 else -amp)
        o = mid - (amp if i % 2 == 0 else -amp)
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        bars.append({"open": o, "high": h, "low": l, "close": c, "ts_utc": f"d{i}"})
    return bars


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Redirect STATE_PATH to a tmp file so tests don't touch runtime/."""
    p = tmp_path / "choppy_regime_state.json"
    monkeypatch.setattr(crd, "STATE_PATH", p)
    return p


# ---------------------------------------------------------------------------
# 1 — ADX weakness contributes to score
# ---------------------------------------------------------------------------


def test_adx_weak_contributes_to_score(monkeypatch):
    """A flippy series with weak trend should produce a non-zero ADX-weak signal."""
    monkeypatch.setattr(crd, "_load_daily_bars", lambda *a, **k: _flippy_bars(40))
    monkeypatch.setattr(crd, "_compute_small_loss_churn", lambda: 0.0)
    out = crd.compute_choppy_score()
    # ADX on the oscillator should be low (well under 20).
    assert out["indicators"]["adx_weak"] is True
    assert out["choppy_score"] >= 0.30  # at least the ADX-weak weight


# ---------------------------------------------------------------------------
# 2 — Direction flips contribute
# ---------------------------------------------------------------------------


def test_high_direction_flips_contributes(monkeypatch):
    bars = _flippy_bars(30)
    monkeypatch.setattr(crd, "_load_daily_bars", lambda *a, **k: bars)
    monkeypatch.setattr(crd, "_compute_small_loss_churn", lambda: 0.0)
    out = crd.compute_choppy_score()
    assert out["indicators"]["direction_flips_5d"] >= crd.DIRECTION_FLIP_THRESHOLD
    assert out["indicators"]["direction_flip_high"] is True


# ---------------------------------------------------------------------------
# 3 — Score below threshold => not choppy
# ---------------------------------------------------------------------------


def test_score_below_threshold_not_choppy(monkeypatch):
    monkeypatch.setattr(crd, "_load_daily_bars", lambda *a, **k: _trending_bars(40))
    monkeypatch.setattr(crd, "_compute_small_loss_churn", lambda: 0.0)
    out = crd.compute_choppy_score()
    assert out["choppy_score"] < crd.CHOPPY_THRESHOLD
    assert out["is_choppy_raw"] is False


# ---------------------------------------------------------------------------
# 4 — Score above threshold => choppy raw
# ---------------------------------------------------------------------------


def test_score_above_threshold_choppy(monkeypatch):
    """Force every indicator on and verify is_choppy_raw fires."""
    monkeypatch.setattr(crd, "_load_daily_bars", lambda *a, **k: _flippy_bars(40))
    monkeypatch.setattr(crd, "_compute_adx", lambda *a, **k: 8.0)
    monkeypatch.setattr(crd, "_compute_direction_flips", lambda *a, **k: 4)
    monkeypatch.setattr(crd, "_compute_failed_breakouts", lambda *a, **k: 3)
    monkeypatch.setattr(crd, "_compute_small_loss_churn", lambda: 0.8)
    monkeypatch.setattr(crd, "_compute_followthrough", lambda *a, **k: 0.2)
    out = crd.compute_choppy_score()
    assert out["choppy_score"] >= crd.CHOPPY_THRESHOLD
    assert out["is_choppy_raw"] is True


# ---------------------------------------------------------------------------
# 5 — Hysteresis: requires N consecutive reads to enter
# ---------------------------------------------------------------------------


def test_hysteresis_requires_consecutive_reads_to_enter(monkeypatch, _isolated_state):
    """One choppy read must NOT activate; CONSECUTIVE_READS_TO_ENTER must."""
    monkeypatch.setattr(
        crd, "compute_choppy_score",
        lambda *a, **k: {
            "choppy_score": 0.80,
            "is_choppy_raw": True,
            "indicators": {},
            "proxy_symbol": "TEST",
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    for i in range(crd.CONSECUTIVE_READS_TO_ENTER - 1):
        st = crd.evaluate_and_persist()
        assert st["choppy_active"] is False, f"activated too early at read {i+1}"
        assert st["consecutive_choppy_reads"] == i + 1

    # Nth read must activate
    st = crd.evaluate_and_persist()
    assert st["choppy_active"] is True
    assert st["consecutive_choppy_reads"] == crd.CONSECUTIVE_READS_TO_ENTER


# ---------------------------------------------------------------------------
# 6 — Hysteresis: requires M consecutive clean reads to exit (after min hold)
# ---------------------------------------------------------------------------


def test_hysteresis_requires_consecutive_reads_to_exit(monkeypatch, _isolated_state):
    """
    Once active and past min-hold, exit only after CONSECUTIVE_READS_TO_EXIT
    clean reads — not after one or two.
    """
    # Seed an active state, entered far enough in the past to clear min-hold.
    past = datetime.now(timezone.utc) - timedelta(
        minutes=crd.MIN_CHOPPY_HOLD_MINUTES + 5
    )
    _isolated_state.write_text(
        json.dumps(
            {
                "choppy_active": True,
                "choppy_score": 0.80,
                "consecutive_choppy_reads": 5,
                "consecutive_clean_reads": 0,
                "entered_choppy_at_utc": past.isoformat(),
                "ts_utc": past.isoformat(),
                "ttl_seconds": 300,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        crd, "compute_choppy_score",
        lambda *a, **k: {
            "choppy_score": 0.10,  # cleanly under exit threshold
            "is_choppy_raw": False,
            "indicators": {},
            "proxy_symbol": "TEST",
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    for i in range(crd.CONSECUTIVE_READS_TO_EXIT - 1):
        st = crd.evaluate_and_persist()
        assert st["choppy_active"] is True, f"deactivated too early at read {i+1}"

    st = crd.evaluate_and_persist()
    assert st["choppy_active"] is False
    assert st["consecutive_clean_reads"] == crd.CONSECUTIVE_READS_TO_EXIT


# ---------------------------------------------------------------------------
# 7 — Min hold time prevents early exit
# ---------------------------------------------------------------------------


def test_min_hold_time_prevents_early_exit(monkeypatch, _isolated_state):
    """
    Even with M+ consecutive clean reads, if entered_choppy_at_utc is recent,
    state stays choppy until min-hold elapses.
    """
    just_now = datetime.now(timezone.utc) - timedelta(minutes=1)
    _isolated_state.write_text(
        json.dumps(
            {
                "choppy_active": True,
                "choppy_score": 0.80,
                "consecutive_choppy_reads": 5,
                "consecutive_clean_reads": 0,
                "entered_choppy_at_utc": just_now.isoformat(),
                "ts_utc": just_now.isoformat(),
                "ttl_seconds": 300,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        crd, "compute_choppy_score",
        lambda *a, **k: {
            "choppy_score": 0.05,
            "is_choppy_raw": False,
            "indicators": {},
            "proxy_symbol": "TEST",
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Run far more clean reads than M — must STILL be choppy due to min-hold.
    for _ in range(crd.CONSECUTIVE_READS_TO_EXIT + 5):
        st = crd.evaluate_and_persist()
    assert st["choppy_active"] is True, (
        "min_hold did not block early exit"
    )


# ---------------------------------------------------------------------------
# 8 — get_choppy_state safe default on missing file
# ---------------------------------------------------------------------------


def test_get_choppy_state_returns_safe_default_on_missing_file(
    _isolated_state, monkeypatch,
):
    if _isolated_state.exists():
        _isolated_state.unlink()
    out = crd.get_choppy_state()
    assert out["choppy_active"] is False
    assert out["choppy_score"] == 0.0


# ---------------------------------------------------------------------------
# 9 — Clean market produces a low score
# ---------------------------------------------------------------------------


def test_clean_market_zero_score(monkeypatch):
    """
    A strongly trending series with no churn and no failures should score
    well below CHOPPY_THRESHOLD and below the entry threshold of the
    consecutive_choppy counter.
    """
    monkeypatch.setattr(crd, "_load_daily_bars", lambda *a, **k: _trending_bars(60))
    monkeypatch.setattr(crd, "_compute_small_loss_churn", lambda: 0.0)
    out = crd.compute_choppy_score()
    assert out["choppy_score"] < crd.CLEAN_THRESHOLD
    assert out["is_choppy_raw"] is False
    assert out["indicators"]["direction_flip_high"] is False

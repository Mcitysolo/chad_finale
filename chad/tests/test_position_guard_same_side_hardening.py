"""Same-side guard hardening regressions.

Covers the 2026-05-09 same-side audit follow-ups:

  - is_same_side_open and is_flip_signal compare sides case-insensitively
    so a record-side "BUY" never bypasses guard against an intent side "buy".
  - is_same_side_open does not rewrite position_guard.json when last_state
    is already MAINTAINED (write-amplification guard).
  - The live_loop SKIP-suppression log emits the strategy field so we can
    attribute over-blocking back to the originating strategy.
  - Existing flip and pyramid behavior is unchanged: the suppression is not
    relaxed by this hardening.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from chad.core import position_guard


@dataclass
class _Intent:
    strategy: str
    symbol: str
    side: str
    quantity: float = 0.0


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    state_path = tmp_path / "position_guard.json"
    monkeypatch.setattr(position_guard, "STATE_PATH", state_path)
    return state_path


def _seed(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state), encoding="utf-8")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_position_guard_same_side_case_insensitive(tmp_state):
    """Mixed-case sides must not bypass same-side suppression."""
    _seed(tmp_state, {
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 5.0, "last_state": "OPEN",
        },
    })
    # Intent side carries lowercase whitespace — must still match record BUY.
    assert position_guard.is_same_side_open(_Intent("alpha", "SPY", " buy ")) is True
    # And likewise mixed case.
    assert position_guard.is_same_side_open(_Intent("alpha", "SPY", "Buy")) is True


def test_position_guard_same_side_idempotent_writes(tmp_state):
    """Repeat MAINTAINED signals must not rewrite position_guard.json."""
    _seed(tmp_state, {
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 5.0, "last_state": "OPEN",
        },
    })
    intent = _Intent("alpha", "SPY", "BUY")

    # First call promotes OPEN → MAINTAINED (one write).
    assert position_guard.is_same_side_open(intent) is True
    after_first = _load(tmp_state)
    assert after_first["alpha|SPY"]["last_state"] == "MAINTAINED"
    first_version = after_first["_version"]

    # Subsequent calls observe last_state == MAINTAINED and must NOT rewrite.
    for _ in range(5):
        assert position_guard.is_same_side_open(intent) is True

    after_repeat = _load(tmp_state)
    assert after_repeat["_version"] == first_version, (
        "is_same_side_open rewrote position_guard.json despite no state "
        "change — write amplification guard regressed."
    )


def test_position_guard_same_side_does_not_block_flip(tmp_state):
    """Opposite-side intent is still detected as a flip, not same-side."""
    _seed(tmp_state, {
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 5.0, "last_state": "OPEN",
        },
    })
    flip_intent = _Intent("alpha", "SPY", "SELL")
    assert position_guard.is_same_side_open(flip_intent) is False
    assert position_guard.is_flip_signal(flip_intent) is True

    # Case-normalized flip detection: lowercase "sell" against record "BUY"
    # is still a flip, not a missed same-side.
    flip_lower = _Intent("alpha", "SPY", " sell ")
    assert position_guard.is_same_side_open(flip_lower) is False
    assert position_guard.is_flip_signal(flip_lower) is True


def test_position_guard_same_side_blocks_pyramid(tmp_state):
    """Current behavior: same-side add (pyramid) without explicit bypass is blocked.

    Pinned here so a future scale-in/pyramid bypass change has to consciously
    update this test rather than silently relax suppression.
    """
    _seed(tmp_state, {
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "SELL", "quantity": 28.0, "last_state": "OPEN",
        },
    })
    pyramid_intent = _Intent("alpha", "SPY", "SELL", 14.0)
    assert position_guard.is_same_side_open(pyramid_intent) is True


def test_guard_vs_broker_truth_drift_detector(tmp_state):
    """detect_guard_vs_broker_truth_drift is a pure detector — no disk writes.

    Covers three states:
      - alpha|SPY BUY vs broker_sync|SPY BUY    → no drift
      - omega|GLD BUY but broker_sync|GLD missing → broker_truth_missing
      - gamma|TLT SELL vs broker_sync|TLT BUY  → side_mismatch
    """
    seed_state = {
        "alpha|SPY": {
            "open": True, "strategy": "alpha", "symbol": "SPY",
            "side": "BUY", "quantity": 5.0, "last_state": "MAINTAINED",
        },
        "broker_sync|SPY": {
            "open": True, "strategy": "broker_sync", "symbol": "SPY",
            "side": "BUY", "quantity": 5.0,
        },
        "omega|GLD": {
            "open": True, "strategy": "omega", "symbol": "GLD",
            "side": "BUY", "quantity": 3.0, "last_state": "OPEN",
        },
        "gamma|TLT": {
            "open": True, "strategy": "gamma", "symbol": "TLT",
            "side": "SELL", "quantity": 7.0, "last_state": "OPEN",
        },
        "broker_sync|TLT": {
            "open": True, "strategy": "broker_sync", "symbol": "TLT",
            "side": "BUY", "quantity": 7.0,
        },
        # Closed entries must be ignored by the detector.
        "alpha|QQQ": {
            "open": False, "strategy": "alpha", "symbol": "QQQ",
            "side": "BUY", "last_state": "CLOSED",
        },
    }
    _seed(tmp_state, seed_state)
    pre_mtime = tmp_state.stat().st_mtime_ns

    drift = position_guard.detect_guard_vs_broker_truth_drift(seed_state)

    # Detector must NOT touch position_guard.json at all.
    assert tmp_state.stat().st_mtime_ns == pre_mtime, (
        "detect_guard_vs_broker_truth_drift mutated position_guard.json — "
        "it must be a read-only detector."
    )

    by_kind = {d["drift_kind"]: d for d in drift}
    assert set(by_kind) == {"broker_truth_missing", "side_mismatch"}, (
        f"unexpected drift kinds: {sorted(by_kind)}"
    )

    missing = by_kind["broker_truth_missing"]
    assert missing["strategy"] == "omega"
    assert missing["symbol"] == "GLD"
    assert missing["guard_side"] == "BUY"
    assert missing["broker_side"] is None
    assert missing["broker_present"] is False

    mismatch = by_kind["side_mismatch"]
    assert mismatch["strategy"] == "gamma"
    assert mismatch["symbol"] == "TLT"
    assert mismatch["guard_side"] == "SELL"
    assert mismatch["broker_side"] == "BUY"
    assert mismatch["broker_present"] is True

    # alpha|SPY (matching broker truth) must NOT appear in drift.
    assert all(d["key"] != "alpha|SPY" for d in drift)
    # Closed entry must NOT appear in drift.
    assert all(d["key"] != "alpha|QQQ" for d in drift)


def test_live_loop_same_side_log_includes_strategy():
    """The SAME_SIDE_POSITION_OPEN suppression log must carry strategy=%s.

    Source-text assertion (matching the test_live_loop_edge_decay_alert_scope
    pattern) — invoking run_once requires the full live-loop dependency
    graph; the log format itself is what we are pinning.
    """
    import chad.core.live_loop as live_loop

    src = Path(live_loop.__file__).read_text(encoding="utf-8")

    # The suppression log block must mention strategy=%s in the format string
    # AND pass strategy in the extra dict for structured log consumers.
    assert "SAME_SIDE_POSITION_OPEN" in src
    same_side_idx = src.index("SAME_SIDE_POSITION_OPEN.value")
    # Look at a window large enough to cover the full logger.info(...) call.
    window = src[max(0, same_side_idx - 200): same_side_idx + 800]
    assert "strategy=%s" in window, (
        "same-side suppression log no longer includes 'strategy=%s' — "
        "operators lose the ability to attribute over-blocking back to "
        "the originating strategy."
    )
    assert '"strategy"' in window, (
        "same-side suppression log no longer puts strategy into the extra "
        "dict — structured log consumers will stop seeing the field."
    )

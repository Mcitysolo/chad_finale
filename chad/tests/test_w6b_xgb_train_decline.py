"""
chad/tests/test_w6b_xgb_train_decline.py

W6B-7 (D3) — chad-xgb-train refuses LOUDLY but exits GREEN.

The unit has sat in systemd `failed` since 2026-07-19 — EXS5's only red. It is
not crashing; it is refusing, correctly, at 72 trusted rows against a required
100 (2153 untrusted, 6 quarantined).

D3 ratified "declined, exit 0": refusing to train on untrusted rows is the
system honouring its own rules. The tests below pin the three properties that
make that safe:

  1. a DECLINE exits 0; a genuine ERROR still exits 1 (the distinction is the
     whole design — collapsing it would hide real failures);
  2. every outcome publishes xgb_train_state.v1 with the countdown, so the
     refusal is visible without journal archaeology;
  3. MIN_TRAINING_SAMPLES is 100 and is a trust floor, not a tuning knob.

(3) is a guard against the most tempting wrong fix: lowering the threshold
until the sentinel goes green would train a live-money veto model on rows the
system has already declared untrusted.
"""

from __future__ import annotations

import json

import pytest

from chad.analytics import train_xgb_model as txm


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Never write to the live runtime/ tree."""
    monkeypatch.setattr(txm, "XGB_TRAIN_STATE_PATH", tmp_path / "xgb_train_state.json")
    return tmp_path / "xgb_train_state.json"


def _state(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The threshold is a trust floor
# --------------------------------------------------------------------------

def test_min_training_samples_is_still_100():
    """Named test, deliberately brittle. If a future change lowers this to make
    a red sentinel green, that change should have to delete this test and say
    so in a commit message."""
    assert txm.MIN_TRAINING_SAMPLES == 100


def test_state_file_records_the_threshold_policy(_isolate_state):
    txm._write_train_state(
        outcome="declined", reason="Not enough samples for training (72 < 100)",
        usable_rows=72, excluded={"untrusted": 2153, "quarantined": 6},
    )
    st = _state(_isolate_state)
    assert "trust floor" in st["threshold_policy"]
    assert "not a tuning knob" in st["threshold_policy"]


# --------------------------------------------------------------------------
# Decline vs error
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reason", [
    "Not enough samples for training (72 < 100)",
    "No trade rows available",
    "No usable samples after feature building",
])
def test_data_quality_reasons_are_declines(reason):
    assert txm._is_decline(reason) is True


@pytest.mark.parametrize("reason", [
    "xgboost not installed in venv",
    "Invalid train/validation split",
    "some unexpected explosion",
])
def test_genuine_failures_are_not_declines(reason):
    """Collapsing this distinction would make the unit exit 0 on a broken
    install — a far worse outcome than the red it replaces."""
    assert txm._is_decline(reason) is False


def test_insufficient_samples_exits_zero(monkeypatch, _isolate_state):
    """The headline behaviour: EXS5 clears honestly."""
    _stub_pipeline(
        monkeypatch,
        result=txm.TrainingResult(
            ok=False,
            reason="Not enough samples for training (72 < 100)",
            n_samples=72, n_features=9, metrics={},
            excluded={"untrusted": 2153, "quarantined": 6},
        ),
    )
    assert txm.main([]) == 0
    st = _state(_isolate_state)
    assert st["outcome"] == "declined"
    assert st["usable_rows"] == 72
    assert st["required_rows"] == 100
    assert st["rows_short"] == 28
    assert st["exclusions"]["untrusted"] == 2153


def test_hard_error_still_exits_one(monkeypatch, _isolate_state):
    _stub_pipeline(
        monkeypatch,
        result=txm.TrainingResult(
            ok=False, reason="xgboost not installed in venv",
            n_samples=0, n_features=0, metrics={},
        ),
    )
    assert txm.main([]) == 1
    assert _state(_isolate_state)["outcome"] == "error"


def test_successful_training_exits_zero_and_records_trained(monkeypatch, _isolate_state):
    _stub_pipeline(
        monkeypatch,
        result=txm.TrainingResult(
            ok=True, reason="", n_samples=150, n_features=9,
            metrics={"val_veto_rate_at_0.65": 0.4},
            model_version="xgb_veto_20260724_000000", dataset_hash="sha256:abc",
        ),
    )
    assert txm.main([]) == 0
    st = _state(_isolate_state)
    assert st["outcome"] == "trained"
    assert st["model_version"] == "xgb_veto_20260724_000000"
    assert st["rows_short"] == 0


# --------------------------------------------------------------------------
# Early-exit paths also publish state
# --------------------------------------------------------------------------

def test_no_trade_rows_declines_and_publishes(monkeypatch, _isolate_state):
    monkeypatch.setattr(txm, "_load_trade_rows", lambda: [])
    assert txm.main([]) == 0
    st = _state(_isolate_state)
    assert st["outcome"] == "declined"
    assert st["usable_rows"] == 0
    assert st["rows_short"] == 100


# --------------------------------------------------------------------------
# The state file is a valid runtime contract
# --------------------------------------------------------------------------

def test_unavailable_exclusion_sets_decline_rather_than_train(monkeypatch, _isolate_state):
    """W6B-7 fail-closed. _load_exclusion_sets has always degraded to EMPTY
    sets on failure and merely logged a warning — its own docstring admitted
    "the manifest will then record zero exclusions for all reasons". But a run
    that cannot tell untrusted rows from trusted ones would train a live-money
    veto model on the whole contaminated book (2153 untrusted rows) while
    reporting zero exclusions. That silently inverts the principle D3 rests on.

    _train_model must never be reached in this state."""
    monkeypatch.setattr(txm, "_load_trade_rows", lambda: [{"row": 1}])
    monkeypatch.setattr(txm, "_load_exclusion_sets", lambda: (set(), set(), False))

    def _explode(*_a, **_kw):
        raise AssertionError("must not train without exclusion sets")

    monkeypatch.setattr(txm, "_build_dataset", _explode)

    assert txm.main([]) == 0
    st = _state(_isolate_state)
    assert st["outcome"] == "declined"
    assert st["exclusions_loaded"] is False
    assert "Exclusion sets unavailable" in st["reason"]


def test_normal_run_records_exclusions_loaded_true(monkeypatch, _isolate_state):
    _stub_pipeline(
        monkeypatch,
        result=txm.TrainingResult(
            ok=False, reason="Not enough samples for training (72 < 100)",
            n_samples=72, n_features=9, metrics={}, excluded={"untrusted": 2153},
        ),
    )
    assert txm.main([]) == 0
    assert _state(_isolate_state)["exclusions_loaded"] is True


def test_state_carries_a_pinned_schema_and_ttl(_isolate_state):
    txm._write_train_state(outcome="declined", reason="r", usable_rows=72)
    st = _state(_isolate_state)
    assert st["schema_version"] == "xgb_train_state.v1"
    assert st["ttl_seconds"] > 24 * 3600, "daily timer needs >24h TTL plus slack"
    assert st["ts_utc"].endswith("Z")


def test_state_write_failure_never_breaks_the_run(monkeypatch, tmp_path):
    """A state-write problem must not turn a successful training run into a
    failed one."""
    monkeypatch.setattr(
        txm, "XGB_TRAIN_STATE_PATH", tmp_path / "no" / "such" / "dir" / "s.json"
    )

    def _boom(*_a, **_kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(txm.os, "replace", _boom)
    txm._write_train_state(outcome="trained", reason="ok", usable_rows=150)  # must not raise


# --------------------------------------------------------------------------
# helper
# --------------------------------------------------------------------------

def _stub_pipeline(monkeypatch, *, result):
    """Drive main() to _train_model without touching real trade files."""
    import numpy as np

    monkeypatch.setattr(txm, "_load_trade_rows", lambda: [{"row": 1}])
    monkeypatch.setattr(txm, "_load_exclusion_sets", lambda: (set(), set(), True))
    monkeypatch.setattr(
        txm, "_build_dataset",
        lambda *_a, **_kw: (
            np.ones((5, 9)), np.array([0, 1, 0, 1, 0]), dict(result.excluded),
        ),
    )
    monkeypatch.setattr(txm, "_train_model", lambda *_a, **_kw: result)

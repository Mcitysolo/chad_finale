"""Tests for chad/validation/config_freeze.py — Phase 5 config freeze (SSOT §3.2, F2).

Fixture-only. They pin the two mechanisms: (1) a deterministic, order-free, change-\
sensitive config hash + verify, and (2) the goalpost-moving penalty — a config change
AFTER a FAIL invalidates the prior seal and increments the deflation trial count, while
pre-result / non-FAIL tuning does not.
"""

from __future__ import annotations

import pytest

from chad.validation.config_freeze import (
    FreezeLedger,
    FrozenConfig,
    config_hash,
    deflation_trials,
)

_THRESH_A = {"dsr_confidence": 0.95, "min_cagr": 0.0, "ruin_bound": 0.01}
_THRESH_B = {"dsr_confidence": 0.90, "min_cagr": 0.0, "ruin_bound": 0.01}  # loosened
_COST = {"stk_commission_per_share": 0.005, "half_spread_bps_liquid": 2.0}
_COST2 = {"stk_commission_per_share": 0.001, "half_spread_bps_liquid": 2.0}  # cheaper
_TS = "2026-07-04T00:00:00Z"


def _ledger(tmp_path) -> FreezeLedger:
    return FreezeLedger(tmp_path / "freeze" / "ledger.json")


# --------------------------------------------------------------------------- #
# Hash + verify.
# --------------------------------------------------------------------------- #
def test_config_hash_is_deterministic_order_free_and_change_sensitive() -> None:
    h1 = config_hash(_THRESH_A, _COST)
    h2 = config_hash(dict(reversed(list(_THRESH_A.items()))), _COST)  # key order shuffled
    assert h1 == h2
    assert config_hash(_THRESH_B, _COST) != h1     # threshold change
    assert config_hash(_THRESH_A, _COST2) != h1    # cost change


def test_frozen_config_verify(tmp_path) -> None:
    state = _ledger(tmp_path).freeze(_THRESH_A, _COST, timestamp=_TS)
    assert isinstance(state.frozen, FrozenConfig)
    assert state.frozen.verify(_THRESH_A, _COST) is True
    assert state.frozen.verify(_THRESH_B, _COST) is False


# --------------------------------------------------------------------------- #
# Freeze lifecycle.
# --------------------------------------------------------------------------- #
def test_freeze_starts_at_base_trials(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    state = ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    assert state.trial_count == 0
    assert state.last_verdict is None
    assert state.superseded_hashes == ()
    # base_trials pre-load.
    state2 = ledger.freeze(_THRESH_A, _COST, timestamp=_TS, base_trials=3)
    assert state2.trial_count == 3


def test_persistence_across_instances(tmp_path) -> None:
    path = tmp_path / "freeze" / "ledger.json"
    FreezeLedger(path).freeze(_THRESH_A, _COST, timestamp=_TS)
    loaded = FreezeLedger(path).load()
    assert loaded is not None
    assert loaded.frozen.verify(_THRESH_A, _COST)


def test_record_and_amend_require_prior_freeze(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.record_verdict("FAIL")
    with pytest.raises(ValueError):
        ledger.amend(_THRESH_B, _COST, timestamp=_TS)


# --------------------------------------------------------------------------- #
# The penalty — post-FAIL change invalidates seal + bumps trial count.
# --------------------------------------------------------------------------- #
def test_post_fail_change_invalidates_seal_and_bumps_trials(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    s0 = ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    old_hash = s0.frozen.config_hash
    ledger.record_verdict("FAIL")

    s1 = ledger.amend(_THRESH_B, _COST, timestamp=_TS)
    # trial count incremented.
    assert s1.trial_count == 1
    # seal invalidated: the OLD config no longer verifies, the new one does.
    assert s1.frozen.config_hash != old_hash
    assert s1.frozen.verify(_THRESH_A, _COST) is False
    assert s1.frozen.verify(_THRESH_B, _COST) is True
    # the old hash is recorded as superseded (the invalidated seal).
    assert old_hash in s1.superseded_hashes


def test_post_fail_same_config_is_no_bump(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    ledger.record_verdict("FAIL")
    s1 = ledger.amend(_THRESH_A, _COST, timestamp=_TS)  # identical config
    assert s1.trial_count == 0
    assert s1.superseded_hashes == ()


def test_pre_result_tuning_is_not_penalised(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    # No verdict recorded yet → changing config is legitimate pre-registration tuning.
    s1 = ledger.amend(_THRESH_B, _COST, timestamp=_TS)
    assert s1.trial_count == 0
    assert s1.frozen.verify(_THRESH_B, _COST) is True


def test_non_fail_verdict_change_is_not_penalised(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    ledger.record_verdict("INSUFFICIENT_DATA")
    s1 = ledger.amend(_THRESH_B, _COST, timestamp=_TS)
    assert s1.trial_count == 0


def test_two_post_fail_changes_accumulate(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    ledger.record_verdict("FAIL")
    ledger.amend(_THRESH_B, _COST, timestamp=_TS)      # +1
    ledger.record_verdict("FAIL")
    s2 = ledger.amend(_THRESH_B, _COST2, timestamp=_TS)  # +1
    assert s2.trial_count == 2
    assert len(s2.superseded_hashes) == 2


# --------------------------------------------------------------------------- #
# Deflation coupling.
# --------------------------------------------------------------------------- #
def test_deflation_trials_adds_ledger_count(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.freeze(_THRESH_A, _COST, timestamp=_TS)
    ledger.record_verdict("FAIL")
    state = ledger.amend(_THRESH_B, _COST, timestamp=_TS)
    assert state.trial_count == 1
    assert deflation_trials(7, state) == 8  # punitive base 7 + 1 goalpost move
    with pytest.raises(ValueError):
        deflation_trials(0, state)

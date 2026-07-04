"""Tests for chad/validation/oos_lockbox.py — Phase 5 OOS lockbox (SSOT §3.1, F1).

Fixture-only. They pin the four mechanical guarantees that turn OOS discipline from a
promise into a lock:
  * the sealed OOS is REFUSED without ``final_run=True`` (the reviewer's leak attempt);
  * each logged open increments the access count; count > 1 maps to CONTAMINATED;
  * the run-log is append-only and hash-chained (tamper-evident);
  * dev runs use a deterministic decoy and never open — or touch — the real box.
"""

from __future__ import annotations

import json

import pytest

from chad.validation.oos_lockbox import (
    DEFAULT_DECOY_SEED,
    OOSAccessError,
    OOSLockbox,
    OOSSealError,
    returns_hash,
    synthetic_decoy_returns,
)
from chad.validation.verdict import HeadMetrics, Verdict, decide_verdict

_OOS = [0.01, -0.02, 0.015, 0.0, -0.005, 0.03]
_CFG_HASH = "cfg" * 10
_COMMIT = "abc123def456"
_TS = "2026-07-04T00:00:00Z"


def _box(tmp_path) -> OOSLockbox:
    return OOSLockbox(tmp_path / "lockbox")


def _seal(box: OOSLockbox, values=_OOS):
    return box.seal(values, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)


# --------------------------------------------------------------------------- #
# Hashing / decoy helpers.
# --------------------------------------------------------------------------- #
def test_returns_hash_deterministic_and_order_sensitive() -> None:
    assert returns_hash(_OOS) == returns_hash(list(_OOS))
    assert returns_hash(_OOS) != returns_hash(list(reversed(_OOS)))
    assert returns_hash([0.01, 0.02]) != returns_hash([0.02, 0.01])


def test_returns_hash_rejects_bool_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        returns_hash([0.1, True])
    with pytest.raises(ValueError):
        returns_hash([0.1, float("nan")])
    with pytest.raises(ValueError):
        returns_hash([0.1, float("inf")])


def test_synthetic_decoy_is_deterministic_per_seed() -> None:
    a = synthetic_decoy_returns(20, seed=7)
    b = synthetic_decoy_returns(20, seed=7)
    c = synthetic_decoy_returns(20, seed=8)
    assert a == b
    assert a != c
    assert len(a) == 20
    assert synthetic_decoy_returns(0, seed=7) == []
    with pytest.raises(ValueError):
        synthetic_decoy_returns(-1, seed=7)


# --------------------------------------------------------------------------- #
# Seal.
# --------------------------------------------------------------------------- #
def test_seal_records_hash_and_is_idempotent_on_identical_content(tmp_path) -> None:
    box = _box(tmp_path)
    assert not box.is_sealed()
    seal = _seal(box)
    assert box.is_sealed()
    assert seal.oos_hash == returns_hash(_OOS)
    assert seal.n_oos == len(_OOS)
    assert seal.config_hash == _CFG_HASH
    # Re-sealing identical content returns the same seal, no error.
    again = _seal(box)
    assert again.oos_hash == seal.oos_hash


def test_conflicting_reseal_raises(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    with pytest.raises(OOSSealError):
        box.seal([0.9, 0.9], config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)


def test_open_before_seal_raises(tmp_path) -> None:
    box = _box(tmp_path)
    with pytest.raises(OOSSealError):
        box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)


# --------------------------------------------------------------------------- #
# The gate — refuse without final_run.
# --------------------------------------------------------------------------- #
def test_open_without_final_run_is_refused_and_not_logged(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    with pytest.raises(OOSAccessError):
        box.open_oos(_OOS, final_run=False, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    # A refused attempt must NOT append to the log (count stays 0).
    assert box.access_count() == 0


def test_open_with_final_run_returns_and_logs(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    got = box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    assert got == [float(x) for x in _OOS]
    assert box.access_count() == 1


def test_open_with_mismatched_content_raises(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    with pytest.raises(OOSSealError):
        box.open_oos([9.0, 9.0], final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    assert box.access_count() == 0


# --------------------------------------------------------------------------- #
# Access count + contamination mapping.
# --------------------------------------------------------------------------- #
def test_access_count_increments_each_open(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    assert box.access_count() == 0
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    assert box.access_count() == 1
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp="2026-07-04T01:00:00Z")
    assert box.access_count() == 2


def test_double_open_maps_to_contaminated_verdict(tmp_path) -> None:
    """count > 1 ⇒ the Phase-5 verdict auto-flags CONTAMINATED (SSOT §3.1)."""
    box = _box(tmp_path)
    _seal(box)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp="2026-07-04T02:00:00Z")
    assert box.access_count() == 2
    metrics = HeadMetrics(
        head="h", parity_status="REPLAYABLE", replayable=True, data_quality_status="CLEAN",
        oos_access_count=box.access_count(), n_oos_trades=40, n_walk_forward_windows=8,
        n_regimes_in_oos=4, deflated_sharpe_worst=0.99, cost_adj_cagr=0.1,
        worst_quantile_ruin=0.0, regimes_with_edge=3, regime_scoped_sizing=False,
        final_run=True, oos_source="sealed_oos",
    )
    assert decide_verdict(metrics).verdict is Verdict.CONTAMINATED


# --------------------------------------------------------------------------- #
# Append-only + hash-chain integrity.
# --------------------------------------------------------------------------- #
def test_run_log_is_append_only_and_chained(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    first_line = box.access_log_path.read_text(encoding="utf-8")
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp="2026-07-04T03:00:00Z")
    both = box.access_log_path.read_text(encoding="utf-8")
    # The second open only appended — the first line is byte-for-byte unchanged.
    assert both.startswith(first_line)
    records = box.access_records()
    assert [r.seq for r in records] == [0, 1]
    assert box.verify_log_integrity() is True


def test_tampering_a_log_line_breaks_integrity(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp="2026-07-04T04:00:00Z")
    lines = box.access_log_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    payload["code_commit"] = "TAMPERED"
    lines[0] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    box.access_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert box.verify_log_integrity() is False


def test_tail_truncation_cannot_hide_an_open(tmp_path) -> None:
    """Deleting the LAST log line must NOT reduce access_count or pass integrity.

    A bare forward chain leaves a self-consistent prefix after tail-truncation; the
    high-water anchor closes that hole so CONTAMINATED cannot be silently reverted.
    """
    box = _box(tmp_path)
    _seal(box)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp="2026-07-04T06:00:00Z")
    assert box.access_count() == 2
    lines = box.access_log_path.read_text(encoding="utf-8").splitlines()
    box.access_log_path.write_text(lines[0] + "\n", encoding="utf-8")  # drop the tail (2nd) line
    # Count must NOT drop below the anchored high-water mark; integrity must fail.
    assert box.access_count() == 2
    assert box.verify_log_integrity() is False


def test_whole_log_deletion_is_detected(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    assert box.access_head_path.is_file()  # the anchor was written
    box.access_log_path.unlink()
    # Anchor still records the open → count preserved, integrity fails.
    assert box.access_count() == 1
    assert box.verify_log_integrity() is False


def test_deleting_a_log_line_breaks_integrity(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp=_TS)
    box.open_oos(_OOS, final_run=True, config_hash=_CFG_HASH, code_commit=_COMMIT, timestamp="2026-07-04T05:00:00Z")
    lines = box.access_log_path.read_text(encoding="utf-8").splitlines()
    # Drop the FIRST record: the remaining record still carries seq=1, so its position
    # (index 0) no longer matches its seq — the chain root is broken.
    box.access_log_path.write_text(lines[1] + "\n", encoding="utf-8")
    assert box.verify_log_integrity() is False


# --------------------------------------------------------------------------- #
# Decoy — dev runs never open the real box.
# --------------------------------------------------------------------------- #
def test_decoy_never_touches_real_box(tmp_path) -> None:
    box = _box(tmp_path)
    _seal(box)
    decoy = box.decoy_oos(len(_OOS))
    assert len(decoy) == len(_OOS)
    # Decoy uses the sealed decoy_seed and does NOT log an access.
    assert decoy == synthetic_decoy_returns(len(_OOS), seed=DEFAULT_DECOY_SEED)
    assert box.access_count() == 0
    assert box.verify_log_integrity() is True

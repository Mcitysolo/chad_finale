"""
chad/tests/test_w6b_envelope_contracts.py

W6B-2 (D1) — envelope contracts for free-keyspace runtime files.

D1 ruling (2026-07-24): pin the ENVELOPE — top-level contract plus per-entry
value shape — and never enumerate identity keys. The key space stays free.

The tests that matter most here are the NEGATIVE ones: that the envelope does
not constrain the key space, and that it does not require fields on closed
entries. Both are ways a well-meaning tightening could turn a working system
red, and both are properties the position_guard writer explicitly relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chad.ops.exterminator_sentinel import _check_envelope

# The envelope this lane proposes for position_guard.json, mirroring
# chad/core/position_guard.py::_validate_position_guard_schema.
GUARD_ENVELOPE = {
    "required_keys": ["_version", "_written_by"],
    "reserved_key_prefix": "_",
    "entry_value_type": "object",
    "entry_required_keys_when": {
        "field": "open",
        "equals": True,
        "required_keys": ["strategy", "symbol", "side"],
    },
}

TRADE_CLOSER_ENVELOPE = {
    "required_keys": ["queues", "processed_fill_ids", "saved_at_utc"],
}


def _guard(entries: dict, *, meta: bool = True) -> dict:
    doc = dict(entries)
    if meta:
        doc["_version"] = 1753318000000
        doc["_written_by"] = "position_guard"
    return doc


def _open_entry(strategy="gamma", symbol="BAC"):
    return {
        "open": True, "strategy": strategy, "symbol": symbol, "side": "BUY",
        "quantity": 11.0, "opened_at": "2026-07-23T14:42:23Z",
        "updated_at_utc": "2026-07-24T00:55:54Z",
    }


# --------------------------------------------------------------------------
# The key space must stay free — this is the D1 ruling itself
# --------------------------------------------------------------------------

def test_arbitrary_identity_keys_are_accepted():
    """Any "<strategy>|<SYMBOL>" is valid. The envelope must never enumerate,
    pattern-match, or whitelist identity keys."""
    doc = _guard({
        "gamma|BAC": _open_entry("gamma", "BAC"),
        "omega_macro|M6E": _open_entry("omega_macro", "M6E"),
        "broker_sync|TLT": _open_entry("broker_sync", "TLT"),
        "a_strategy_invented_tomorrow|WEIRD.SYM": _open_entry("x", "WEIRD.SYM"),
    })
    assert _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE) == []


def test_empty_guard_is_valid():
    """A flat book is a legitimate state, not a contract break."""
    assert _check_envelope("runtime/position_guard.json", _guard({}), GUARD_ENVELOPE) == []


def test_no_minimum_entry_count_is_imposed():
    one = _check_envelope("runtime/position_guard.json", _guard({"g|A": _open_entry()}), GUARD_ENVELOPE)
    many = _check_envelope(
        "runtime/position_guard.json",
        _guard({f"g|S{i}": _open_entry("g", f"S{i}") for i in range(200)}),
        GUARD_ENVELOPE,
    )
    assert one == [] and many == []


# --------------------------------------------------------------------------
# Closed entries may be sparse — the near-miss this design corrected
# --------------------------------------------------------------------------

def test_closed_entries_may_be_sparse():
    """position_guard.py:459 (reset_from_broker_truth) builds a closed entry
    with no `opened_at`. An envelope inferred from today's data — where all 39
    live entries happen to carry it — would go red the first time that path
    ran. The writer's contract says "closed entries can be sparse"; the read
    side must agree."""
    doc = _guard({
        "gamma|SPY": {
            "open": False, "updated_at_utc": "2026-07-24T00:00:00Z",
            "strategy": "gamma", "symbol": "SPY", "side": "", "quantity": 0.0,
        },
        "gamma|MINIMAL": {"open": False},
    })
    assert _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE) == []


def test_open_entry_missing_identification_is_a_break():
    """The conditional half: an OPEN entry without strategy/symbol/side is the
    partial record that crashes reconciliation_publisher and net_exposure_gate."""
    doc = _guard({"gamma|BAC": {"open": True, "quantity": 11.0}})
    breaks = _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE)
    assert len(breaks) == 1
    assert breaks[0]["break"] == "envelope_entry_keys_missing"
    assert set(breaks[0]["entries"][0]["missing_keys"]) == {"strategy", "symbol", "side"}


def test_open_is_matched_by_value_not_truthiness():
    """`open` must equal True. A string "true" or 1 is not an open position,
    and treating it as one would apply the strict rule to the wrong entries."""
    doc = _guard({"gamma|X": {"open": "yes"}})
    assert _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE) == []


# --------------------------------------------------------------------------
# Structural breaks
# --------------------------------------------------------------------------

def test_missing_meta_keys_is_a_break():
    doc = _guard({"gamma|BAC": _open_entry()}, meta=False)
    breaks = _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE)
    assert breaks[0]["break"] == "envelope_required_keys_missing"
    assert set(breaks[0]["missing_keys"]) == {"_version", "_written_by"}


def test_non_dict_entry_is_a_break():
    doc = _guard({"gamma|BAC": "not-a-dict"})
    breaks = _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE)
    assert breaks[0]["break"] == "envelope_entry_not_an_object"
    assert breaks[0]["entry_count"] == 1


def test_reserved_prefix_keys_are_never_treated_as_entries():
    """_version is an int. Without the reserved-prefix skip it would be
    reported as a non-object entry every single cycle."""
    doc = _guard({})
    doc["_some_future_meta_key"] = 12345
    assert _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE) == []


def test_entry_detail_is_capped_but_count_is_not():
    """A structurally broken file must not dump hundreds of rows into the
    sentinel report — but the operator still needs the true magnitude."""
    doc = _guard({f"g|S{i}": {"open": True} for i in range(50)})
    breaks = _check_envelope("runtime/position_guard.json", doc, GUARD_ENVELOPE)
    assert breaks[0]["entry_count"] == 50
    assert len(breaks[0]["entries"]) == 10


# --------------------------------------------------------------------------
# trade_closer_state: fixed top level, no free key space
# --------------------------------------------------------------------------

def test_trade_closer_envelope_accepts_the_real_shape():
    doc = {"queues": {}, "processed_fill_ids": [], "saved_at_utc": "2026-07-24T00:00:00Z"}
    assert _check_envelope("runtime/trade_closer_state.json", doc, TRADE_CLOSER_ENVELOPE) == []


def test_trade_closer_envelope_catches_a_dropped_queue():
    doc = {"processed_fill_ids": [], "saved_at_utc": "2026-07-24T00:00:00Z"}
    breaks = _check_envelope("runtime/trade_closer_state.json", doc, TRADE_CLOSER_ENVELOPE)
    assert breaks[0]["missing_keys"] == ["queues"]


# --------------------------------------------------------------------------
# The live artifacts must pass today
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rel,envelope", [
    ("position_guard.json", GUARD_ENVELOPE),
    ("trade_closer_state.json", TRADE_CLOSER_ENVELOPE),
])
def test_live_artifact_satisfies_its_proposed_envelope(rel, envelope):
    """Pre-flight: adopting these envelopes must be inert on day one. Skips
    rather than fails when the live tree is unavailable (CI, fresh clone)."""
    path = Path("/home/ubuntu/chad_finale/runtime") / rel
    if not path.is_file():
        pytest.skip(f"{rel} not present on this host")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert _check_envelope(f"runtime/{rel}", doc, envelope) == []


# --------------------------------------------------------------------------
# The envelope must not disturb the schema_version path
# --------------------------------------------------------------------------

def test_envelope_only_contract_skips_version_validation(tmp_path):
    """A contract with no schema_version/accepts must NOT report
    schema_version_absent — that is the whole point of leaving position_guard
    unmodified."""
    from chad.ops.exterminator_sentinel import ExterminatorSentinel

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "position_guard.json").write_text(json.dumps(_guard({"g|A": _open_entry()})))

    s = ExterminatorSentinel.__new__(ExterminatorSentinel)
    s.repo_root = tmp_path
    s.data_dir = tmp_path / "data"
    s.config = {"schema_contracts": {"enforced": {
        "runtime/position_guard.json": {
            "envelope": GUARD_ENVELOPE,
            "pinned_at": "test",
        }
    }}}
    result = s.check_schema_breaks()
    assert result.evidence["break_count"] == 0, result.evidence["breaks"]


def test_versioned_contract_still_validates_version(tmp_path):
    """Regression guard: making version checking opt-in must not silently
    disable it for the 10 contracts that already depend on it."""
    from chad.ops.exterminator_sentinel import ExterminatorSentinel

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "scr_state.json").write_text(json.dumps({"schema_version": "scr_state.v99"}))

    s = ExterminatorSentinel.__new__(ExterminatorSentinel)
    s.repo_root = tmp_path
    s.data_dir = tmp_path / "data"
    s.config = {"schema_contracts": {"enforced": {
        "runtime/scr_state.json": {
            "schema_version": "scr_state.v1", "accepts": ["scr_state.v1"],
        }
    }}}
    result = s.check_schema_breaks()
    assert result.evidence["break_count"] == 1
    assert result.evidence["breaks"][0]["break"] == "schema_version_unrecognised"

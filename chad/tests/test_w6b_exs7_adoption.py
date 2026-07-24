"""
chad/tests/test_w6b_exs7_adoption.py

W6B-1 — EXS7 enforcement-coverage tooling.

The point of these tests is not that the tool runs; it is that the EXCLUSION
rules are the ones we argued for, and that the emitted contract stays minimal.
Both are things a future edit could quietly widen:

  * widening the exclusions silently shrinks enforcement coverage;
  * widening required_keys silently manufactures future false FAILs.

The tool is read-only, so every test builds a synthetic runtime dir. None of
them touch the live tree.
"""

from __future__ import annotations

import json

import pytest

from ops import exs7_adoption as ad


@pytest.fixture()
def fake_tree(tmp_path):
    """A synthetic repo: runtime/ dir plus a config with a known enforced set."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    config = tmp_path / "exterminator.json"
    config.write_text(json.dumps({
        "schema_contracts": {
            "enforced": {
                "runtime/already.json": {
                    "schema_version": "already.v1",
                    "accepts": ["already.v1"],
                    "required_keys": ["schema_version"],
                }
            }
        }
    }))
    return runtime, config


def _write(runtime, name, payload):
    p = runtime / name
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return p


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------

def test_pinned_and_live_file_is_eligible(fake_tree):
    runtime, config = fake_tree
    _write(runtime, "regime_state.json", {"schema_version": "regime_state.v1"})
    res = ad.classify(runtime, config)
    assert [e["file"] for e in res["eligible"]] == ["runtime/regime_state.json"]


def test_already_enforced_is_not_re_proposed(fake_tree):
    runtime, config = fake_tree
    _write(runtime, "already.json", {"schema_version": "already.v1"})
    res = ad.classify(runtime, config)
    assert res["already_enforced"] == ["runtime/already.json"]
    assert res["eligible"] == []


def test_unpinned_file_is_not_eligible(fake_tree):
    """Adoption is for files that already carry a schema_version. Pinning new
    ones is a separate, riskier exercise (W6B-2)."""
    runtime, config = fake_tree
    _write(runtime, "no_version.json", {"ts_utc": "2026-07-24T00:00:00Z"})
    res = ad.classify(runtime, config)
    assert res["eligible"] == []
    assert res["unpinned"] == ["runtime/no_version.json"]


def test_unreadable_file_is_reported_not_swallowed(fake_tree):
    runtime, config = fake_tree
    _write(runtime, "ndjson_under_json_name.json", '{"a":1}\n{"a":2}\n')
    res = ad.classify(runtime, config)
    assert res["eligible"] == []
    assert len(res["unreadable"]) == 1
    assert res["unreadable"][0]["file"] == "runtime/ndjson_under_json_name.json"


def test_json_array_is_unreadable_not_eligible(fake_tree):
    runtime, config = fake_tree
    _write(runtime, "a_list.json", ["schema_version"])
    res = ad.classify(runtime, config)
    assert res["unreadable"][0]["error"] == "not a JSON object"


# --------------------------------------------------------------------------
# Exclusion class 1 — dated one-offs
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "quarantine_manifest_20260511.json",
    "quarantine_manifest_pff1_ghost_scrub.json",
    "broker_truth_snapshot_20260419.json",
    "__guard_probe_should_not_exist__.json",
])
def test_dated_and_one_off_names_are_excluded(fake_tree, name):
    """These are prunable by design; enforcing them turns housekeeping red."""
    runtime, config = fake_tree
    _write(runtime, name, {"schema_version": "whatever.v1"})
    res = ad.classify(runtime, config)
    assert res["eligible"] == [], f"{name} must not be eligible"
    assert res["excluded"][0]["class"] == "dated_one_off"


def test_ordinary_name_containing_digits_is_not_mistaken_for_dated(fake_tree):
    """The dated pattern must be tight enough not to eat real contracts —
    e.g. a v2/v3 suffix or an embedded number is not a date."""
    runtime, config = fake_tree
    _write(runtime, "tier_state.json", {"schema_version": "tier_state.v3"})
    _write(runtime, "allocator_v3_state.json", {"schema_version": "allocator_v3_state.v1"})
    res = ad.classify(runtime, config)
    assert {e["file"] for e in res["eligible"]} == {
        "runtime/tier_state.json", "runtime/allocator_v3_state.json",
    }


# --------------------------------------------------------------------------
# Exclusion class 2 — event-conditional
# --------------------------------------------------------------------------

def test_event_conditional_files_are_excluded_with_a_reason(fake_tree):
    runtime, config = fake_tree
    _write(runtime, "epoch_reset_state.json", {"schema_version": "epoch_reset_state.v1"})
    res = ad.classify(runtime, config)
    assert res["eligible"] == []
    ex = next(e for e in res["excluded"] if e["file"] == "runtime/epoch_reset_state.json")
    assert ex["class"] == "event_conditional"
    assert ex["reason"], "an exclusion without a stated reason is a curation problem"


def test_every_declared_exclusion_carries_a_reason():
    """Guards against someone adding a bare filename to the exclusion maps."""
    for rel, reason in {**ad.EVENT_CONDITIONAL, **ad.PUBLISHER_NOT_YET_FIRED}.items():
        assert rel.startswith("runtime/"), rel
        assert reason and len(reason) > 20, f"{rel} needs a real reason"


# --------------------------------------------------------------------------
# Exclusion class 3 — publisher landed, artifact not yet produced
# --------------------------------------------------------------------------

def test_publisher_not_yet_fired_is_excluded_while_absent(fake_tree):
    """The W6A bars_refresh_state case: enforcing a file whose publisher has
    landed but never run manufactures a red for working code."""
    runtime, config = fake_tree
    res = ad.classify(runtime, config)
    classes = {e["class"] for e in res["excluded"]}
    assert "publisher_not_yet_fired" in classes
    entry = next(e for e in res["excluded"] if e["class"] == "publisher_not_yet_fired")
    assert entry["file"] == "runtime/bars_refresh_state.json"


def test_publisher_not_yet_fired_becomes_eligible_once_it_exists(fake_tree):
    """The exclusion must be conditional on absence, not permanent — otherwise
    the contract can never be adopted after the first nightly run."""
    runtime, config = fake_tree
    _write(runtime, "bars_refresh_state.json", {"schema_version": "bars_refresh_state.v1"})
    res = ad.classify(runtime, config)
    assert "runtime/bars_refresh_state.json" in {e["file"] for e in res["eligible"]}
    assert not any(e["class"] == "publisher_not_yet_fired" for e in res["excluded"])


# --------------------------------------------------------------------------
# The emitted contract must stay minimal
# --------------------------------------------------------------------------

def test_emitted_contract_requires_only_schema_version():
    """Inferring richer required_keys from one observation is how an optional
    field becomes a permanent false FAIL. Keep it minimal on purpose."""
    contract = ad.build_contract({"file": "runtime/x.json", "schema_version": "x.v1"})
    assert contract["required_keys"] == ["schema_version"]
    assert contract["accepts"] == ["x.v1"]


def test_emitted_contract_accepts_only_the_observed_version():
    """A silent schema bump is exactly what enforcement should catch."""
    contract = ad.build_contract({"file": "runtime/x.json", "schema_version": "x.v2"})
    assert contract["accepts"] == ["x.v2"]
    assert "x.v1" not in contract["accepts"]


# --------------------------------------------------------------------------
# Pre-flight verification
# --------------------------------------------------------------------------

def test_verify_is_clean_when_contracts_hold(fake_tree):
    runtime, config = fake_tree
    _write(runtime, "a.json", {"schema_version": "a.v1"})
    res = ad.classify(runtime, config)
    assert ad.verify(runtime, res["eligible"]) == []


def test_verify_catches_a_version_drift(fake_tree):
    """Adopt against v1, then the publisher bumps to v2 -> break."""
    runtime, config = fake_tree
    _write(runtime, "a.json", {"schema_version": "a.v1"})
    res = ad.classify(runtime, config)
    _write(runtime, "a.json", {"schema_version": "a.v2"})
    breaks = ad.verify(runtime, res["eligible"])
    assert breaks and breaks[0]["break"] == "schema_version_unrecognised"


def test_verify_catches_a_vanished_file(fake_tree):
    runtime, config = fake_tree
    p = _write(runtime, "a.json", {"schema_version": "a.v1"})
    res = ad.classify(runtime, config)
    p.unlink()
    breaks = ad.verify(runtime, res["eligible"])
    assert breaks and breaks[0]["break"] == "missing"


def test_verify_mirrors_the_sentinels_break_vocabulary():
    """If the sentinel's break names drift from this tool's, the pre-flight
    stops predicting the real check. Pin the shared vocabulary."""
    from chad.ops import exterminator_sentinel as sentinel
    import inspect

    src = inspect.getsource(sentinel.ExterminatorSentinel.check_schema_breaks)
    for token in (
        "missing",
        "unreadable_or_not_an_object",
        "schema_version_absent",
        "schema_version_unrecognised",
        "required_keys_missing",
    ):
        assert f'"{token}"' in src, f"sentinel no longer emits break={token}"

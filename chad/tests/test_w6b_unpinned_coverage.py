"""
chad/tests/test_w6b_unpinned_coverage.py

W6B-3 — EXS7's unpinned surface is computed, not curated.

The pre-existing `unpinned_known` is an exact-path dict with 5 entries. A full
enumeration of runtime/ on 2026-07-24 found 127 files with no schema_version
plus 4 unparseable as JSON — so the WARN described 5 of 131.

Enumerating the rest is not the fix: 65 of the 127 are `telegram_dedupe_*.json`
whose filenames are generated from alert text, an UNBOUNDED namespace that
would need a commit per new alert string.

These tests pin the behaviours that make the replacement honest:
  * pattern classes cover unbounded namespaces without enumeration
  * a genuinely new unpinned artifact is UNDOCUMENTED and trips the warn
  * declaring a class with a reason clears it
  * unreadable files are surfaced instead of being invisible
  * the warn no longer fires merely because unpinned_known is non-empty
"""

from __future__ import annotations

import json

from chad.ops.exterminator_sentinel import (
    STATUS_OK,
    STATUS_WARN,
    ExterminatorSentinel,
    _classify_unpinned,
)


def _runtime(tmp_path, files: dict):
    rt = tmp_path / "runtime"
    rt.mkdir(exist_ok=True)
    for name, content in files.items():
        (rt / name).write_text(
            content if isinstance(content, str) else json.dumps(content)
        )
    return rt


# --------------------------------------------------------------------------
# Pattern classes vs unbounded namespaces
# --------------------------------------------------------------------------

def test_one_pattern_covers_an_unbounded_namespace(tmp_path):
    """65 dedupe markers today, more tomorrow, all from one declared class."""
    files = {
        f"telegram_dedupe_health_R{i:02d}_whatever.json": {"ts": i} for i in range(40)
    }
    rt = _runtime(tmp_path, files)
    res = _classify_unpinned(
        rt, {}, {"telegram_dedupe_*.json": {"reason": "ephemeral alert marker"}}, {}
    )
    assert res["undocumented_count"] == 0
    assert res["by_class"]["telegram_dedupe_*.json"] == 40


def test_a_new_unpinned_artifact_is_undocumented(tmp_path):
    """The actionable event: someone ships a publisher with no schema_version."""
    rt = _runtime(tmp_path, {"brand_new_state.json": {"ts_utc": "2026-07-24T00:00:00Z"}})
    res = _classify_unpinned(rt, {}, {"telegram_dedupe_*.json": {"reason": "x"}}, {})
    assert res["undocumented"] == ["runtime/brand_new_state.json"]
    assert res["undocumented_count"] == 1


def test_declaring_a_class_clears_the_undocumented_entry(tmp_path):
    rt = _runtime(tmp_path, {"scratch_thing.json": {"a": 1}})
    before = _classify_unpinned(rt, {}, {}, {})
    after = _classify_unpinned(rt, {}, {"scratch_*.json": {"reason": "scratch space"}}, {})
    assert before["undocumented_count"] == 1
    assert after["undocumented_count"] == 0
    assert after["documented_count"] == 1


def test_exact_unpinned_known_entries_still_work(tmp_path):
    """Backwards compatibility: the pre-existing exact-path dict is honoured."""
    rt = _runtime(tmp_path, {"legacy.json": {"a": 1}})
    res = _classify_unpinned(rt, {"runtime/legacy.json": "documented reason"}, {}, {})
    assert res["undocumented_count"] == 0
    assert res["by_class"]["exact:unpinned_known"] == 1


# --------------------------------------------------------------------------
# Pinned and enforced files are out of scope
# --------------------------------------------------------------------------

def test_pinned_files_are_not_counted_as_unpinned(tmp_path):
    rt = _runtime(tmp_path, {"pinned.json": {"schema_version": "pinned.v1"}})
    res = _classify_unpinned(rt, {}, {}, {})
    assert res["undocumented_count"] == 0
    assert res["documented_count"] == 0


def test_enforced_files_are_skipped_entirely(tmp_path):
    """An enforced contract is validated by the break loop; counting it here
    too would double-report it."""
    rt = _runtime(tmp_path, {"enf.json": {"no_version": True}})
    res = _classify_unpinned(rt, {}, {}, {"runtime/enf.json": {"envelope": {}}})
    assert res["undocumented_count"] == 0


# --------------------------------------------------------------------------
# Unreadable files were previously invisible
# --------------------------------------------------------------------------

def test_ndjson_under_a_json_name_is_reported(tmp_path):
    """runtime/flip_executor_audit.json and signal_throttle_audit.json are real
    instances: NDJSON bodies under a .json name, so json.load() raises. They
    were invisible to EXS7 — not enforced so never opened, not schema-bearing
    so never counted."""
    rt = _runtime(tmp_path, {"audit.json": '{"a":1}\n{"a":2}\n'})
    res = _classify_unpinned(rt, {}, {}, {})
    assert res["unreadable_count"] == 1
    assert res["unreadable"][0]["file"] == "runtime/audit.json"
    assert res["undocumented_count"] == 0, "unreadable is its own class, not undocumented"


def test_zero_byte_file_is_reported_unreadable(tmp_path):
    rt = _runtime(tmp_path, {"empty.json": ""})
    res = _classify_unpinned(rt, {}, {}, {})
    assert res["unreadable_count"] == 1


def test_missing_runtime_dir_degrades_quietly(tmp_path):
    res = _classify_unpinned(tmp_path / "nope", {}, {}, {})
    assert res["undocumented_count"] == 0 and res["scanned"] == 0


# --------------------------------------------------------------------------
# Bounded output
# --------------------------------------------------------------------------

def test_undocumented_list_is_capped_but_count_is_true(tmp_path):
    rt = _runtime(tmp_path, {f"f{i:03d}.json": {"a": i} for i in range(60)})
    res = _classify_unpinned(rt, {}, {}, {})
    assert res["undocumented_count"] == 60
    assert len(res["undocumented"]) == 25


# --------------------------------------------------------------------------
# The warn now describes the actionable gap
# --------------------------------------------------------------------------

def _sentinel(tmp_path, config):
    s = ExterminatorSentinel.__new__(ExterminatorSentinel)
    s.repo_root = tmp_path
    s.data_dir = tmp_path / "data"
    s.config = config
    return s


def test_warn_no_longer_fires_merely_because_unpinned_known_is_nonempty(tmp_path):
    """The old rule was `if unpinned_known: WARN` — permanent by construction,
    since that dict never shrinks on its own. A permanent warn trains operators
    to skim, which is the same failure mode P2-3's log noise had."""
    _runtime(tmp_path, {"documented.json": {"a": 1}})
    s = _sentinel(tmp_path, {"schema_contracts": {
        "enforced": {},
        "unpinned_known": {"runtime/documented.json": "known and accepted"},
    }})
    result = s.check_schema_breaks()
    assert result.status == STATUS_OK
    assert result.evidence["unpinned_known_count"] == 1


def test_warn_fires_on_an_unclassified_file(tmp_path):
    _runtime(tmp_path, {"surprise.json": {"a": 1}})
    s = _sentinel(tmp_path, {"schema_contracts": {"enforced": {}, "unpinned_known": {}}})
    result = s.check_schema_breaks()
    assert result.status == STATUS_WARN
    assert result.evidence["unpinned_coverage"]["undocumented_count"] == 1


def test_warn_fires_on_an_unreadable_file_even_when_all_else_is_clean(tmp_path):
    _runtime(tmp_path, {"broken.json": "not json"})
    s = _sentinel(tmp_path, {"schema_contracts": {"enforced": {}, "unpinned_known": {}}})
    result = s.check_schema_breaks()
    assert result.status == STATUS_WARN
    assert result.evidence["unpinned_coverage"]["unreadable_count"] == 1


def test_breaks_still_outrank_coverage(tmp_path):
    """A real contract break must stay FAIL regardless of coverage state."""
    _runtime(tmp_path, {
        "enf.json": {"schema_version": "wrong.v9"},
        "surprise.json": {"a": 1},
    })
    s = _sentinel(tmp_path, {"schema_contracts": {
        "enforced": {"runtime/enf.json": {"schema_version": "enf.v1", "accepts": ["enf.v1"]}},
        "unpinned_known": {},
    }})
    from chad.ops.exterminator_sentinel import STATUS_FAIL
    assert s.check_schema_breaks().status == STATUS_FAIL


def test_clean_tree_with_full_class_coverage_reports_ok(tmp_path):
    _runtime(tmp_path, {
        "telegram_dedupe_a.json": {"x": 1},
        "telegram_dedupe_b.json": {"x": 2},
        "pinned.json": {"schema_version": "pinned.v1"},
    })
    s = _sentinel(tmp_path, {"schema_contracts": {
        "enforced": {},
        "unpinned_known": {},
        "unpinned_classes": {"telegram_dedupe_*.json": {"reason": "ephemeral"}},
    }})
    result = s.check_schema_breaks()
    assert result.status == STATUS_OK

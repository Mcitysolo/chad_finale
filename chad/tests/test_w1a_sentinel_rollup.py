"""
W1A-3 — Exterminator Sentinel rollup driver-attribution.

The overall_status rollup (ok/warn/fail) already existed; what was missing was
WHICH checks drove it. These tests pin the additive `rollup` block:
{overall, first_fail, driver_check_ids, driver_reason}. Driver-attribution ONLY
(GO decision D1a) — no downstream consumer health rule this wave.

The all-ok world is unreachable through run() (EXS8 ml_anomalies always warns
no_baseline), so the rollup logic is exercised as a pure function over synthetic
CheckResult lists, plus one integration test over the healthy fixture that
asserts rollup consistency with overall_status and that the anti-auto-heal locks
still hold.
"""

from __future__ import annotations

import importlib

import pytest

sentinel_mod = importlib.import_module("chad.ops.exterminator_sentinel")
CheckResult = sentinel_mod.CheckResult
worst_status = sentinel_mod.worst_status
_build_rollup = sentinel_mod._build_rollup

# Reuse the healthy-runtime fixtures/helpers from the main sentinel test module.
from chad.tests.test_exterminator_sentinel import _fresh_runtime, _make, NOW  # noqa: E402


def _cr(check_id: str, status: str, name: str = "x") -> "sentinel_mod.CheckResult":
    return CheckResult(check_id, name, status, "title", "summary", {"e": 1})


# ---------------------------------------------------------------------------
# Pure-function rollup logic
# ---------------------------------------------------------------------------


def test_all_ok_has_no_drivers() -> None:
    checks = [_cr("EXS1", "ok"), _cr("EXS2", "ok")]
    overall = worst_status(c.status for c in checks)
    rollup = _build_rollup(checks, overall)
    assert rollup["overall"] == "ok"
    assert rollup["driver_check_ids"] == []
    assert rollup["first_fail"] is None
    assert rollup["driver_reason"] == "all checks ok"


def test_single_fail_names_that_check() -> None:
    checks = [_cr("EXS1", "ok"), _cr("EXS3", "fail", "untrusted_fills"), _cr("EXS8", "warn")]
    overall = worst_status(c.status for c in checks)
    rollup = _build_rollup(checks, overall)
    assert rollup["overall"] == "fail"
    assert rollup["driver_check_ids"] == ["EXS3"]  # only the fail, not the warn
    assert rollup["first_fail"] == "EXS3"
    assert "EXS3:untrusted_fills" in rollup["driver_reason"]


def test_multiple_fails_first_fail_is_first_in_order() -> None:
    checks = [_cr("EXS3", "fail"), _cr("EXS5", "fail"), _cr("EXS8", "warn")]
    overall = worst_status(c.status for c in checks)
    rollup = _build_rollup(checks, overall)
    assert rollup["overall"] == "fail"
    assert rollup["driver_check_ids"] == ["EXS3", "EXS5"]
    assert rollup["first_fail"] == "EXS3"


def test_warn_only_rolls_up_to_warn_with_warners_as_drivers() -> None:
    # EXS6/EXS8 top out at warn; a warn-only world must roll up to warn, and the
    # drivers are the warners (not an empty set).
    checks = [_cr("EXS1", "ok"), _cr("EXS6", "warn", "dirty_git"), _cr("EXS8", "warn", "ml_anomalies")]
    overall = worst_status(c.status for c in checks)
    rollup = _build_rollup(checks, overall)
    assert rollup["overall"] == "warn"
    assert rollup["driver_check_ids"] == ["EXS6", "EXS8"]
    assert rollup["first_fail"] is None  # warn is not a fail


def test_fail_dominates_warn_drivers_are_only_fails() -> None:
    # When both warn and fail are present, overall=fail and drivers are ONLY the
    # checks at the overall (fail) severity — warns are not drivers.
    checks = [_cr("EXS6", "warn"), _cr("EXS4", "fail", "reconciliation_drift"), _cr("EXS8", "warn")]
    overall = worst_status(c.status for c in checks)
    rollup = _build_rollup(checks, overall)
    assert rollup["driver_check_ids"] == ["EXS4"]


# ---------------------------------------------------------------------------
# Integration: rollup is consistent with overall_status + anti-heal intact
# ---------------------------------------------------------------------------


@pytest.fixture()
def clock():
    return lambda: NOW


@pytest.fixture()
def quiet_providers():
    return {
        "systemctl_provider": lambda query: {"failed_units": [], "error": None, "query": list(query)},
        "git_provider": lambda: {"head": "abc123", "branch": "main", "entries": [], "error": None},
        "notifier": lambda message, dedupe_key: False,
    }


def test_report_has_rollup_consistent_with_overall(tmp_path, clock, quiet_providers) -> None:
    runtime = tmp_path / "runtime"
    _fresh_runtime(runtime)
    report = _make(tmp_path, clock, quiet_providers).run()

    assert "rollup" in report
    rollup = report["rollup"]

    # 1. rollup.overall mirrors the existing overall_status exactly.
    assert rollup["overall"] == report["overall_status"]

    # 2. driver_check_ids are exactly the checks AT the overall severity.
    checks = report["checks"]
    expected_drivers = (
        [c["check_id"] for c in checks if c["status"] == report["overall_status"]]
        if report["overall_status"] != "ok"
        else []
    )
    assert rollup["driver_check_ids"] == expected_drivers

    # 3. first_fail is the first fail (or None) — self-consistent with checks.
    fails = [c["check_id"] for c in checks if c["status"] == "fail"]
    assert rollup["first_fail"] == (fails[0] if fails else None)

    # 4. Additive only: the pre-existing schema + anti-auto-heal locks hold.
    assert report["schema_version"] == "exterminator_sentinel.v1"
    assert report["read_only_confirmed"] is True
    assert report["runtime_files_modified"] == []
    assert len(report["checks"]) == 8

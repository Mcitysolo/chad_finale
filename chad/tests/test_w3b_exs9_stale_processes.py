"""W3B-5 — EXS9 stale-process check (service start vs import-graph code age).

The :9618 lesson: chad-backend served month-old SCR math (2026-06-19 ->
07-20) behind fresh scr_state.json timestamps; BOX-034A recorded the ~15-day
orchestrator variant and named this check as the missing ops item.

Locked properties:
- WARN-CAPPED (D2): even a month of lag never produces fail — deploy-pending
  is governed operator state, and maybe_notify pages on fail only, so EXS9
  can never page;
- import-graph scoping (D1/BOX-034A section 3): each unit compares against
  ITS code_paths via the injected provider, never bare HEAD;
- not-running services are skipped (EXS5's jurisdiction);
- provider errors degrade to warn, never crash (and never fail);
- the read-only anti-mutation locks still hold with the new providers.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.ops import exterminator_sentinel as sentinel_mod
from chad.ops.exterminator_sentinel import ExterminatorSentinel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOW = datetime(2026, 7, 22, 21, 0, 0, tzinfo=timezone.utc)


def _uptime(started_ago_s, state="active", error=None, pid=4242):
    def provider(unit):
        if error:
            return {"unit": unit, "active_state": "", "active_enter_unix": None,
                    "main_pid": None, "error": error}
        return {
            "unit": unit, "active_state": state,
            "active_enter_unix": (NOW - timedelta(seconds=started_ago_s)).timestamp(),
            "main_pid": pid, "error": None,
        }
    return provider


def _code(committed_ago_s, error=None):
    def provider(paths):
        if error:
            return {"paths": list(paths), "commit_unix": None, "commit_hash": "",
                    "error": error}
        return {
            "paths": list(paths),
            "commit_unix": (NOW - timedelta(seconds=committed_ago_s)).timestamp(),
            "commit_hash": "abc123def456", "error": None,
        }
    return provider


def _sentinel(tmp_path, cfg, *, uptime, code):
    runtime = tmp_path / "runtime"
    runtime.mkdir(exist_ok=True)
    config_path = tmp_path / "exterminator.json"
    config_path.write_text(json.dumps({"stale_processes": cfg}), encoding="utf-8")
    return ExterminatorSentinel(
        repo_root=tmp_path,
        runtime_dir=runtime,
        reports_dir=tmp_path / "reports",
        config_path=config_path,
        clock=lambda: NOW,
        service_uptime_provider=uptime,
        code_timestamp_provider=code,
    )


_ONE_UNIT = {
    "grace_seconds": 900,
    "units": [{"unit": "chad-backend.service", "code_paths": ["backend/"]}],
    "excluded_units": [{"unit": "chad-ibgateway.service", "reason": "non-repo"}],
}


def _row(result, unit):
    return next(r for r in result.evidence["services"] if r["unit"] == unit)


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def test_fresh_service_is_ok(tmp_path):
    # started 1h ago, newest commit 2 days ago -> current
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(3600), code=_code(2 * 86400))
    result = s.check_stale_processes()
    assert result.status == "ok"
    row = _row(result, "chad-backend.service")
    assert row["status"] == "ok"
    assert row["code_newer_by_seconds"] < 0


def test_stale_service_warns(tmp_path):
    # the :9618 shape: process 31 days old, code committed 2 days ago
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(31 * 86400), code=_code(2 * 86400))
    result = s.check_stale_processes()
    assert result.status == "warn"
    row = _row(result, "chad-backend.service")
    assert row["reason"] == "process_predates_code"
    assert row["code_newer_by_seconds"] > 28 * 86400
    assert "chad-backend.service" in result.summary


def test_never_fails_even_at_extreme_lag(tmp_path):
    """D2 lock: a YEAR of lag is still warn — deploy-pending is governed
    state and EXS9 must never page (maybe_notify fires on fail only)."""
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(365 * 86400), code=_code(60))
    result = s.check_stale_processes()
    assert result.status == "warn"
    assert all(r["status"] != "fail" for r in result.evidence["services"])


def test_grace_absorbs_fresh_merge(tmp_path):
    # commit landed 5 min after start -> inside the 900s grace -> ok
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(600), code=_code(300))
    assert s.check_stale_processes().status == "ok"


def test_not_running_service_is_skipped(tmp_path):
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(0, state="inactive"),
                  code=_code(60))
    result = s.check_stale_processes()
    assert result.status == "ok"
    row = _row(result, "chad-backend.service")
    assert row["skipped"] is True
    assert row["reason"].startswith("not_running:")


def test_uptime_probe_error_degrades_to_warn(tmp_path):
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(0, error="TimeoutExpired"),
                  code=_code(60))
    result = s.check_stale_processes()
    assert result.status == "warn"
    assert _row(result, "chad-backend.service")["reason"] == "uptime_probe_error"
    assert "degraded" in result.summary


def test_code_timestamp_error_degrades_to_warn(tmp_path):
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(3600),
                  code=_code(0, error="git unavailable"))
    result = s.check_stale_processes()
    assert result.status == "warn"
    assert _row(result, "chad-backend.service")["reason"] == "code_timestamp_unavailable"


def test_missing_config_section_warns(tmp_path):
    s = _sentinel(tmp_path, {}, uptime=_uptime(3600), code=_code(60))
    result = s.check_stale_processes()
    assert result.status == "warn"
    assert "declares no stale_processes.units" in result.summary


def test_exclusions_carried_in_evidence(tmp_path):
    s = _sentinel(tmp_path, _ONE_UNIT, uptime=_uptime(3600), code=_code(2 * 86400))
    result = s.check_stale_processes()
    assert result.evidence["excluded_units"] == _ONE_UNIT["excluded_units"]


def test_per_unit_paths_reach_the_provider(tmp_path):
    """D1 lock: the provider receives each unit's own code_paths — the
    import-graph scoping, not bare HEAD."""
    seen = []

    def code(paths):
        seen.append(tuple(paths))
        return {"paths": list(paths), "commit_unix": (NOW - timedelta(days=2)).timestamp(),
                "commit_hash": "abc", "error": None}

    cfg = {
        "grace_seconds": 900,
        "units": [
            {"unit": "a.service", "code_paths": ["backend/"]},
            {"unit": "b.service", "code_paths": ["chad/web/", "chad/utils/"]},
        ],
    }
    s = _sentinel(tmp_path, cfg, uptime=_uptime(3600), code=code)
    s.check_stale_processes()
    assert seen == [("backend/",), ("chad/web/", "chad/utils/")]


# ---------------------------------------------------------------------------
# real-config shape + provider read-only locks
# ---------------------------------------------------------------------------


def test_real_config_declares_the_section():
    cfg = json.loads((REPO_ROOT / "config" / "exterminator.json").read_text(encoding="utf-8"))
    sp = cfg["stale_processes"]
    assert sp["grace_seconds"] == 900
    units = {u["unit"] for u in sp["units"]}
    # the two documented incident services must be covered
    assert "chad-backend.service" in units
    assert "chad-orchestrator.service" in units
    # every unit entry carries import-graph paths
    assert all(u.get("code_paths") for u in sp["units"])
    # justified exclusions present
    excluded = {e["unit"] for e in sp["excluded_units"]}
    assert "chad-ibgateway.service" in excluded
    assert all(e.get("reason") for e in sp["excluded_units"])


def test_default_uptime_provider_uses_readonly_show_only():
    import inspect

    src = inspect.getsource(sentinel_mod.default_service_uptime_provider)
    assert '"show"' in src
    for forbidden in ('"start"', '"stop"', '"restart"', '"reset-failed"'):
        assert forbidden not in src


def test_default_code_provider_uses_readonly_log_only():
    import inspect

    src = inspect.getsource(sentinel_mod.default_code_timestamp_provider)
    assert '"log"' in src
    for forbidden in ('"commit"', '"push"', '"merge"', '"rebase"', '"clean"'):
        assert forbidden not in src

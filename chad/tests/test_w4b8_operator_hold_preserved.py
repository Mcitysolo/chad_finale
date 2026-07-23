"""W4B-8d (INCIDENT-0723 D4a) — auto-refresh must not stomp an operator hold.

During the incident the operator-granted EXIT_ONLY (TTL 24h, 13:43:03Z) was
rewritten to ALLOW_LIVE by the 10-minute chad-operator-intent-refresh timer
at 13:43:11Z. These tests pin the new contract: `refresh` PRESERVES an
explicitly-set, unexpired EXIT_ONLY / DENY_ALL — judged by the state's OWN
ts_utc + ttl_seconds, not the store's 900s freshness default — and only
falls back to ALLOW_LIVE once the hold's declared TTL has elapsed.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.ops import operator_intent_refresher as oir


def _ctx(tmp_path: Path) -> oir.RuntimeContext:
    return oir.RuntimeContext(
        repo_root=tmp_path,
        runtime_path=tmp_path / "operator_intent.json",
        execution_mode="paper",
        hostname="test-host",
        pid=os.getpid(),
        now_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _refresh_args() -> argparse.Namespace:
    return argparse.Namespace(ttl_seconds=900, reason="")


def _backdate(path: Path, seconds: float) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    ts = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    for key in ("ts_utc", "timestamp_utc"):
        if key in doc:
            doc[key] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(doc), encoding="utf-8")


def _state(ctx) -> dict:
    return json.loads(ctx.runtime_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("held", ["EXIT_ONLY", "DENY_ALL"])
def test_refresh_preserves_unexpired_hold(tmp_path, held):
    ctx = _ctx(tmp_path)
    oir.write_intent(ctx, mode=held, reason="incident_hold_test",
                     ttl_seconds=86400, allow_live_write=False)

    rc = oir.cmd_refresh(ctx, _refresh_args())

    assert rc == oir.ExitCode.SUCCESS
    st = _state(ctx)
    mode = str(st.get("operator_mode", st.get("mode", ""))).upper()
    assert mode == held, f"refresh stomped the {held} hold: {st}"
    assert "incident_hold_test" in str(st)


def test_refresh_hold_expiry_deadline_does_not_extend(tmp_path):
    """Each preserving refresh re-stamps ts with the REMAINING ttl — the
    original expiry deadline never moves later."""
    ctx = _ctx(tmp_path)
    oir.write_intent(ctx, mode="EXIT_ONLY", reason="hold",
                     ttl_seconds=1000, allow_live_write=False)
    _backdate(ctx.runtime_path, 400)

    oir.cmd_refresh(ctx, _refresh_args())

    st = _state(ctx)
    ttl = int(st.get("ttl_seconds", 0))
    assert 0 < ttl <= 600, f"remaining ttl must shrink past time: {st}"


def test_refresh_reverts_after_hold_expiry(tmp_path):
    """An EXPIRED hold is not preserved — refresh returns to the non-live
    default (ALLOW_LIVE) exactly as before."""
    ctx = _ctx(tmp_path)
    oir.write_intent(ctx, mode="EXIT_ONLY", reason="hold",
                     ttl_seconds=60, allow_live_write=False)
    _backdate(ctx.runtime_path, 120)

    rc = oir.cmd_refresh(ctx, _refresh_args())

    assert rc == oir.ExitCode.SUCCESS
    st = _state(ctx)
    mode = str(st.get("operator_mode", st.get("mode", ""))).upper()
    assert mode == "ALLOW_LIVE"


def test_refresh_normal_path_unchanged_for_allow_live(tmp_path):
    """No hold in place -> refresh behaves exactly as before."""
    ctx = _ctx(tmp_path)
    oir.write_intent(ctx, mode="ALLOW_LIVE", reason="baseline",
                     ttl_seconds=900, allow_live_write=False)

    rc = oir.cmd_refresh(ctx, _refresh_args())

    assert rc == oir.ExitCode.SUCCESS
    st = _state(ctx)
    mode = str(st.get("operator_mode", st.get("mode", ""))).upper()
    assert mode == "ALLOW_LIVE"


def test_refresh_absent_state_still_refreshes(tmp_path):
    """Missing state file -> normal refresh (fail-open to the default), the
    pre-fix behavior for a fresh host."""
    ctx = _ctx(tmp_path)
    rc = oir.cmd_refresh(ctx, _refresh_args())
    assert rc == oir.ExitCode.SUCCESS
    mode = str(_state(ctx).get("operator_mode", "")).upper()
    assert mode == "ALLOW_LIVE"

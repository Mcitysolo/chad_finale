"""GAP-005 — live_readiness TTL alignment.

Pins the contract that LiveGate honors live_readiness.json's weekly
evaluation cadence rather than treating the pointer as stale after the
generic 600s default. The fix MUST NOT relax fail-closed behavior:
ready_for_live=false must still deny LIVE execution, regardless of
freshness.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chad.core import live_gate as lg


def _write_readiness(runtime: Path, *, ready: bool, ts_utc: datetime, ttl_seconds: int | None) -> None:
    payload = {
        "schema_version": "live_readiness_state.v1",
        "ready_for_live": bool(ready),
        "ts_utc": ts_utc.isoformat().replace("+00:00", "Z"),
        "next_evaluation_cadence": "weekly",
    }
    if ttl_seconds is not None:
        payload["ttl_seconds"] = int(ttl_seconds)
    (runtime / "live_readiness.json").write_text(json.dumps(payload), encoding="utf-8")


def test_live_readiness_default_ttl_is_weekly():
    assert lg.LIVE_READINESS_DEFAULT_TTL_SECONDS == 7 * 24 * 60 * 60
    assert lg.LIVE_READINESS_DEFAULT_TTL_SECONDS == 604800


def test_live_readiness_weekly_ttl_not_stale_after_10_minutes(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    _write_readiness(runtime, ready=False, ts_utc=ts, ttl_seconds=None)
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)

    state = lg._load_live_readiness_state()

    assert state.ok is True, f"expected fresh under weekly default, got reason={state.reason}"
    assert state.ready_for_live is False
    assert state.reason == "LIVE_READINESS_FALSE"


def test_live_readiness_ready_false_remains_not_live_approved(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    _write_readiness(runtime, ready=False, ts_utc=ts, ttl_seconds=604800)
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)

    state = lg._load_live_readiness_state()

    assert state.ok is True
    assert state.ready_for_live is False
    assert state.reason == "LIVE_READINESS_FALSE"


def test_live_readiness_expired_after_weekly_ttl(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ts = datetime.now(timezone.utc) - timedelta(days=8)
    _write_readiness(runtime, ready=False, ts_utc=ts, ttl_seconds=None)
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)

    state = lg._load_live_readiness_state()

    assert state.ok is False
    assert state.ready_for_live is False
    assert "stale" in state.reason


def test_live_gate_does_not_approve_live_when_ready_false_even_if_fresh(tmp_path, monkeypatch):
    """LiveGate must DENY_ALL (no allow_ibkr_live) when readiness is fresh
    but ready_for_live=false. This guards against the TTL fix accidentally
    being read as a permission widening."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    _write_readiness(runtime, ready=False, ts_utc=ts, ttl_seconds=604800)
    monkeypatch.setattr(lg, "_runtime_dir", lambda: runtime)

    decision = lg.evaluate_live_gate()

    assert decision.allow_ibkr_live is False
    assert any("LIVE_READINESS_FALSE" in r or "LIVE_READINESS_UNAVAILABLE" in r or "OPERATOR" in r or "STOP" in r or "PROFIT" in r for r in decision.reasons), decision.reasons
    assert decision.context.live_readiness.ready_for_live is False

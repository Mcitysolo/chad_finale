"""GAP-016 / BOX-037 — lock ALLOW_LIVE semantics.

These tests pin the four safety contracts that make the system fail
closed for live trading even though ``operator_intent.json`` currently
carries ``ALLOW_LIVE`` (set by the every-10-minute refresher in paper
mode):

  1. ``refresh_mode_for_execution`` returns ALLOW_LIVE in paper/dry_run
     and **raises** in live — auto-refresh cannot widen permissions
     to live trading.
  2. ``write_intent`` refuses to mutate operator_intent.json in live
     execution mode unless ``allow_live_write=True`` is explicitly
     supplied (matches the ``--allow-live-write`` CLI flag).
  3. ``OperatorIntentStore.load_fail_closed`` collapses to
     ``DENY_ALL`` for every degraded path: missing file, malformed
     JSON, unknown mode, missing TTL, stale TTL.
  4. ``live_gate._load_operator_intent`` collapses to ``DENY_ALL`` when
     the runtime file is missing or carries an unrecognised mode —
     the gate is fail-closed regardless of refresher behaviour.

None of these tests authorise live trading, mutate live runtime state,
or restart any service. They use tmp_path + monkeypatched constants.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.operator_intent_store import (
    OperatorIntentStore,
    OperatorMode,
)
from chad.core import live_gate
from chad.ops import operator_intent_refresher as refresher


# ---------------------------------------------------------------------------
# 1. Refresher policy: paper/dry_run may auto-refresh ALLOW_LIVE; live cannot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exec_mode", ["paper", "dry_run"])
def test_refresh_mode_returns_allow_live_in_non_live(exec_mode: str):
    """Documented paper-mode auto-refresh: refresher writes ALLOW_LIVE
    because the execution-mode gate still blocks actual live trading.
    """
    assert refresher.refresh_mode_for_execution(exec_mode) == OperatorMode.ALLOW_LIVE


def test_refresh_mode_refuses_to_widen_in_live_mode():
    """If execution mode flips to live, the timer-driven refresher must
    not silently keep refreshing ALLOW_LIVE — the operator must use the
    explicit set/--allow-live-write path."""
    with pytest.raises(RuntimeError, match="refresh_refused_live_mode"):
        refresher.refresh_mode_for_execution("live")


# ---------------------------------------------------------------------------
# 2. write_intent guard: live mode requires explicit allow_live_write
# ---------------------------------------------------------------------------


def _ctx(tmp_path: Path, exec_mode: str) -> refresher.RuntimeContext:
    return refresher.RuntimeContext(
        repo_root=tmp_path,
        runtime_path=tmp_path / "operator_intent.json",
        execution_mode=exec_mode,
        hostname="test-host",
        pid=1,
        now_utc="2026-05-20T00:00:00Z",
    )


def test_write_intent_blocks_live_mode_without_allow_live_write(tmp_path: Path):
    ctx = _ctx(tmp_path, exec_mode="live")
    with pytest.raises(RuntimeError, match="write_refused_live_mode_without_allow_live_write"):
        refresher.write_intent(
            ctx,
            mode="ALLOW_LIVE",
            reason="operator_promotes_to_live",
            ttl_seconds=900,
            allow_live_write=False,
        )
    assert not ctx.runtime_path.exists(), "no file may be written under refusal path"


def test_write_intent_allows_live_mode_with_explicit_allow_live_write(tmp_path: Path):
    """The explicit deliberate-human-intent path: live mode write is
    permitted only when the operator passes allow_live_write=True
    (matches the --allow-live-write CLI flag)."""
    ctx = _ctx(tmp_path, exec_mode="live")
    payload = refresher.write_intent(
        ctx,
        mode="ALLOW_LIVE",
        reason="ratified_live_promotion_2026-05-20",
        ttl_seconds=900,
        allow_live_write=True,
    )
    assert payload["ok"] is True
    assert ctx.runtime_path.is_file()
    data = json.loads(ctx.runtime_path.read_text(encoding="utf-8"))
    assert data["operator_mode"] == OperatorMode.ALLOW_LIVE
    assert "ratified_live_promotion_2026-05-20" in data["operator_reason"]


def test_write_intent_paper_mode_does_not_require_allow_live_write(tmp_path: Path):
    ctx = _ctx(tmp_path, exec_mode="paper")
    payload = refresher.write_intent(
        ctx,
        mode="ALLOW_LIVE",
        reason="paper_auto_refresh",
        ttl_seconds=900,
        allow_live_write=False,
    )
    assert payload["ok"] is True
    assert ctx.runtime_path.is_file()


# ---------------------------------------------------------------------------
# 3. OperatorIntentStore fail-closed paths
# ---------------------------------------------------------------------------


def test_store_load_missing_file_returns_deny_all(tmp_path: Path):
    """Missing operator_intent.json must collapse to DENY_ALL. The
    shared-lock helper open()s with 'a+' so the path may surface as
    either ``missing`` or ``json_decode_error`` depending on whether
    the open created an empty file — both are fail-closed paths and
    must yield DENY_ALL with freshness.ok=False."""
    store = OperatorIntentStore(path=tmp_path / "operator_intent.json")
    st = store.load_fail_closed()
    assert st.mode == OperatorMode.DENY_ALL
    assert st.reason.startswith("operator_intent_")
    assert st.freshness.ok is False


def test_store_load_malformed_json_returns_deny_all(tmp_path: Path):
    p = tmp_path / "operator_intent.json"
    p.write_text("{not valid json", encoding="utf-8")
    store = OperatorIntentStore(path=p)
    st = store.load_fail_closed()
    assert st.mode == OperatorMode.DENY_ALL
    assert "json_decode_error" in st.reason


def test_store_load_unknown_mode_returns_deny_all(tmp_path: Path):
    p = tmp_path / "operator_intent.json"
    p.write_text(json.dumps({
        "operator_mode": "GO_WILD",
        "operator_reason": "rogue",
        "ts_utc": "2026-05-20T00:00:00Z",
        "ttl_seconds": 900,
    }), encoding="utf-8")
    store = OperatorIntentStore(path=p)
    st = store.load_fail_closed()
    assert st.mode == OperatorMode.DENY_ALL
    assert "invalid_or_missing_mode" in st.reason


def test_store_load_stale_ttl_returns_deny_all(tmp_path: Path):
    p = tmp_path / "operator_intent.json"
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    p.write_text(json.dumps({
        "operator_mode": "ALLOW_LIVE",
        "operator_reason": "old_ratification",
        "ts_utc": stale_ts,
        "ttl_seconds": 900,
    }), encoding="utf-8")
    store = OperatorIntentStore(path=p)
    st = store.load_fail_closed()
    assert st.mode == OperatorMode.DENY_ALL
    assert "stale_or_missing" in st.reason
    assert st.freshness.ok is False


def test_store_load_fresh_allow_live_returns_allow_live(tmp_path: Path):
    """Positive control: fresh ALLOW_LIVE is returned faithfully."""
    p = tmp_path / "operator_intent.json"
    fresh_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    p.write_text(json.dumps({
        "operator_mode": "ALLOW_LIVE",
        "operator_reason": "fresh_ratified",
        "ts_utc": fresh_ts,
        "ttl_seconds": 900,
    }), encoding="utf-8")
    store = OperatorIntentStore(path=p)
    st = store.load_fail_closed()
    assert st.mode == OperatorMode.ALLOW_LIVE
    assert st.freshness.ok is True


# ---------------------------------------------------------------------------
# 4. LiveGate fail-closed loader contracts
# ---------------------------------------------------------------------------


def _patch_runtime_dir(monkeypatch, tmp_path: Path) -> None:
    """Point live_gate at a hermetic runtime dir."""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(live_gate, "_runtime_dir", lambda: runtime_dir)


def test_live_gate_operator_intent_missing_file_yields_deny_all(monkeypatch, tmp_path: Path):
    _patch_runtime_dir(monkeypatch, tmp_path)
    intent = live_gate._load_operator_intent()
    assert intent.operator_mode == "DENY_ALL"
    assert intent.operator_reason.startswith("operator_intent_")


def test_live_gate_operator_intent_unknown_mode_yields_deny_all(monkeypatch, tmp_path: Path):
    _patch_runtime_dir(monkeypatch, tmp_path)
    runtime_dir = tmp_path / "runtime"
    fresh_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (runtime_dir / "operator_intent.json").write_text(json.dumps({
        "operator_mode": "GO_WILD",
        "operator_reason": "rogue",
        "ts_utc": fresh_ts,
        "ttl_seconds": 900,
    }), encoding="utf-8")
    intent = live_gate._load_operator_intent()
    assert intent.operator_mode == "DENY_ALL"
    assert "unknown_mode" in intent.operator_reason


def test_live_gate_operator_intent_stale_yields_deny_all(monkeypatch, tmp_path: Path):
    """A fresh-looking ALLOW_LIVE that has aged past its TTL must not
    authorize live behaviour through the gate."""
    _patch_runtime_dir(monkeypatch, tmp_path)
    runtime_dir = tmp_path / "runtime"
    stale_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    (runtime_dir / "operator_intent.json").write_text(json.dumps({
        "operator_mode": "ALLOW_LIVE",
        "operator_reason": "stale_allow_live",
        "ts_utc": stale_ts,
        "ttl_seconds": 900,
    }), encoding="utf-8")
    intent = live_gate._load_operator_intent()
    assert intent.operator_mode == "DENY_ALL", (
        "stale ALLOW_LIVE must fail closed at the LiveGate boundary (GAP-016)"
    )
    assert "stale_or_invalid" in intent.operator_reason


def test_live_gate_normalizes_legacy_allow_to_allow_live(monkeypatch, tmp_path: Path):
    """LiveGate accepts legacy ALLOW alias and normalizes to ALLOW_LIVE.
    This proves the back-compat path documented in live_gate.py:436-438."""
    _patch_runtime_dir(monkeypatch, tmp_path)
    runtime_dir = tmp_path / "runtime"
    fresh_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    (runtime_dir / "operator_intent.json").write_text(json.dumps({
        "operator_mode": "ALLOW",
        "operator_reason": "legacy_alias",
        "ts_utc": fresh_ts,
        "ttl_seconds": 900,
    }), encoding="utf-8")
    intent = live_gate._load_operator_intent()
    assert intent.operator_mode == "ALLOW_LIVE"


# ---------------------------------------------------------------------------
# 5. End-to-end via evaluate_live_gate: stale ALLOW_LIVE never authorizes
# ---------------------------------------------------------------------------


def test_evaluate_live_gate_stale_allow_live_returns_deny_all(monkeypatch, tmp_path: Path):
    """Composite proof: even with stop_state.stop=false and a stale
    ALLOW_LIVE, the gate denies — the operator-intent gate fires first.
    """
    _patch_runtime_dir(monkeypatch, tmp_path)
    runtime_dir = tmp_path / "runtime"
    now = datetime.now(timezone.utc)
    fresh = now.isoformat().replace("+00:00", "Z")
    stale_iso = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")

    # stop disabled, fresh
    (runtime_dir / "stop_state.json").write_text(json.dumps({
        "stop": False, "reason": "ok", "ts_utc": fresh, "ttl_seconds": 86400,
    }), encoding="utf-8")
    # profit lock fresh & not stopping new entries
    (runtime_dir / "profit_lock_state.json").write_text(json.dumps({
        "stop_new_entries": False, "profit_lock_active": False, "mode": "NORMAL",
        "sizing_factor": 1.0, "ts_utc": fresh, "ttl_seconds": 120,
    }), encoding="utf-8")
    # STALE operator intent — ALLOW_LIVE that has expired
    (runtime_dir / "operator_intent.json").write_text(json.dumps({
        "operator_mode": "ALLOW_LIVE",
        "operator_reason": "stale_ratification",
        "ts_utc": stale_iso, "ttl_seconds": 900,
    }), encoding="utf-8")

    # Stub execution config / mode so we don't depend on env
    monkeypatch.setattr(live_gate, "_load_execution_config", lambda: live_gate.ExecutionConfig(
        exec_mode="paper", raw_mode_enum="paper",
        ibkr_enabled=True, ibkr_dry_run=True, kraken_enabled=False,
    ))

    decision = live_gate.evaluate_live_gate()
    assert decision.mode == "DENY_ALL"
    assert decision.allow_ibkr_live is False
    # Gate ordering: STOP / PROFIT_LOCK pass; OPERATOR_INTENT trips on
    # the DENY_ALL produced by the stale loader.
    joined = " ".join(decision.reasons)
    assert "OperatorIntent=DENY_ALL" in joined
    assert "stale_or_invalid" in joined or "operator_intent_" in joined


def test_evaluate_live_gate_paper_mode_blocks_live_even_with_fresh_allow_live(
    monkeypatch, tmp_path: Path,
):
    """The live state of the production host today: operator_intent is
    fresh ALLOW_LIVE (paper auto-refresh), but exec_mode=paper. The
    gate must still deny live — the EXEC_MODE gate (or an upstream
    readiness/scr/quality gate) fires."""
    _patch_runtime_dir(monkeypatch, tmp_path)
    runtime_dir = tmp_path / "runtime"
    now = datetime.now(timezone.utc)
    fresh = now.isoformat().replace("+00:00", "Z")

    for fname, payload in {
        "stop_state.json": {"stop": False, "reason": "ok", "ts_utc": fresh, "ttl_seconds": 86400},
        "profit_lock_state.json": {
            "stop_new_entries": False, "profit_lock_active": False, "mode": "NORMAL",
            "sizing_factor": 1.0, "ts_utc": fresh, "ttl_seconds": 120,
        },
        "operator_intent.json": {
            "operator_mode": "ALLOW_LIVE",
            "operator_reason": "auto_refresh_allow_entries_non_live",
            "ts_utc": fresh, "ttl_seconds": 900,
        },
        "live_readiness.json": {
            "ready_for_live": False, "ts_utc": fresh,
            "ttl_seconds": 7 * 24 * 60 * 60,
        },
    }.items():
        (runtime_dir / fname).write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(live_gate, "_load_execution_config", lambda: live_gate.ExecutionConfig(
        exec_mode="paper", raw_mode_enum="paper",
        ibkr_enabled=True, ibkr_dry_run=True, kraken_enabled=False,
    ))

    decision = live_gate.evaluate_live_gate()
    assert decision.mode == "DENY_ALL"
    assert decision.allow_ibkr_live is False

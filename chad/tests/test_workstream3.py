"""Workstream-3 acceptance tests.

Covers:
  * ops/micro_eod_flatten.py — tier-gated EOD flatten emitter
  * chad/ops/health_monitor_rules.py — additive R15 + R16 rules

The EOD flatten tests redirect the module's RUNTIME_DIR (and the
position_reconciler / position_guard STATE_PATH it delegates to) into a
per-test tmp_path so the suite never touches the live runtime tree.
"""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────
def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_tier_state(
    runtime: Path,
    *,
    tier_name: str,
    flatten_before_eod: bool,
    age_seconds: float = 0.0,
    ttl_seconds: int = 900,
    include_ttl: bool = True,
) -> None:
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    profile = {
        "max_contracts_per_trade": 1,
        "max_risk_per_trade_usd": 10,
        "max_daily_loss_usd": 20,
        "max_weekly_loss_usd": 30,
        "max_trades_per_day": 2,
        "primary_session_only": True,
        "flatten_before_eod": flatten_before_eod,
        "flatten_eod_minutes_before_close": 30,
        "stop_width_gate_enabled": True,
    }
    payload: Dict[str, Any] = {
        "schema_version": "tier_state.v2",
        "tier_name": tier_name,
        "tier_description": f"test {tier_name}",
        "current_equity_usd": 100.0,
        "tier_min_equity": 0.0,
        "tier_max_equity": 2500.0,
        "enabled_strategies": ["alpha_intraday_micro"],
        "allowed_instruments": ["MES"],
        "risk_profile": profile,
        "previous_tier": None,
        "promoted_at_utc": None,
        "demotion_pending": False,
        "demotion_pending_to": None,
        "demotion_pending_reason": None,
        "demotion_pending_since_utc": None,
        "demotion_applies_at": None,
        "ts_utc": _iso(ts),
    }
    if include_ttl:
        payload["ttl_seconds"] = ttl_seconds
    _write_json(runtime / "tier_state.json", payload)


def _write_position_guard(runtime: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    _write_json(runtime / "position_guard.json", entries)


@pytest.fixture
def eod_module(tmp_path, monkeypatch):
    """Import (or reload) ops.micro_eod_flatten with RUNTIME pinned to tmp_path."""
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    # Pin chad.core.position_guard.STATE_PATH so load_open_positions()
    # delegates to our tmp guard file.
    from chad.core import position_guard
    monkeypatch.setattr(
        position_guard, "STATE_PATH", runtime / "position_guard.json"
    )

    import ops.micro_eod_flatten as mod
    mod = importlib.reload(mod)
    monkeypatch.setattr(mod, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(mod, "TIER_STATE_PATH", runtime / "tier_state.json")
    monkeypatch.setattr(
        mod, "STATUS_OUT_PATH", runtime / "micro_eod_flatten.json"
    )
    monkeypatch.setattr(
        mod, "INTENT_LEDGER_PATH", runtime / "eod_flatten_intents.json"
    )
    return mod, runtime


def _run(eod_module):
    mod, runtime = eod_module
    mod.run()
    status = json.loads((runtime / "micro_eod_flatten.json").read_text())
    return status


# ── Test 1: SCALE tier — flatten skipped ────────────────────────────────────
def test_t1_scale_tier_skipped(eod_module):
    mod, runtime = eod_module
    _write_tier_state(runtime, tier_name="SCALE", flatten_before_eod=False)
    status = _run(eod_module)
    assert status["status"] == "SKIPPED"
    assert status["positions_closed"] == 0
    assert status["flatten_required"] is False


# ── Test 2: MICRO, no qualifying positions ──────────────────────────────────
def test_t2_micro_no_positions(eod_module):
    mod, runtime = eod_module
    _write_tier_state(runtime, tier_name="MICRO", flatten_before_eod=True)
    _write_position_guard(runtime, {})
    status = _run(eod_module)
    assert status["status"] == "OK"
    assert status["positions_found"] == 0
    assert status["positions_closed"] == 0


# ── Test 3: STARTER, one MES alpha_intraday_micro long ──────────────────────
def test_t3_starter_one_mes_long(eod_module):
    mod, runtime = eod_module
    _write_tier_state(runtime, tier_name="STARTER", flatten_before_eod=True)
    _write_position_guard(runtime, {
        "alpha_intraday_micro|MES": {
            "open": True,
            "strategy": "alpha_intraday_micro",
            "symbol": "MES",
            "side": "BUY",
            "quantity": 1.0,
            "opened_at": _iso(datetime.now(timezone.utc)),
        }
    })
    status = _run(eod_module)
    assert status["status"] == "OK"
    assert status["positions_found"] == 1
    assert status["positions_closed"] == 1

    ledger = json.loads(
        (runtime / "eod_flatten_intents.json").read_text(encoding="utf-8")
    )
    pending = [
        i for i in ledger["intents"]
        if i["position_key"] == "alpha_intraday_micro|MES"
    ]
    assert len(pending) == 1
    assert pending[0]["close_side"] == "SELL"
    assert pending[0]["quantity"] == 1.0
    assert pending[0]["status"] == "pending"


# ── Test 4: tier_state.json missing -> ERROR, exit 0 ────────────────────────
def test_t4_tier_state_missing(eod_module):
    mod, runtime = eod_module
    rc = mod.main()
    assert rc == 0
    status = json.loads((runtime / "micro_eod_flatten.json").read_text())
    assert status["status"] == "ERROR"
    assert status["skipped_reason"] == "EOD_FLATTEN_TIER_STATE_MISSING_OR_STALE"


# ── Test 5: tier_state.json stale beyond ttl -> ERROR ───────────────────────
def test_t5_tier_state_stale(eod_module):
    mod, runtime = eod_module
    _write_tier_state(
        runtime,
        tier_name="MICRO",
        flatten_before_eod=True,
        age_seconds=2400,  # 40min — beyond the explicit 300s ttl below
        ttl_seconds=300,
    )
    rc = mod.main()
    assert rc == 0
    status = json.loads((runtime / "micro_eod_flatten.json").read_text())
    assert status["status"] == "ERROR"
    assert status["skipped_reason"] == "EOD_FLATTEN_TIER_STATE_MISSING_OR_STALE"


# ── Test 6: status JSON written atomically and parses cleanly ───────────────
def test_t6_status_artifact_atomic(eod_module):
    mod, runtime = eod_module
    _write_tier_state(runtime, tier_name="MICRO", flatten_before_eod=True)
    _write_position_guard(runtime, {})
    _run(eod_module)
    status_path = runtime / "micro_eod_flatten.json"
    assert status_path.is_file()
    # File must be valid JSON.
    parsed = json.loads(status_path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "micro_eod_flatten.v1"
    # No tmp file leftover after atomic write.
    assert not (runtime / "micro_eod_flatten.json.tmp").exists()


# ── R15 / R16 tests ─────────────────────────────────────────────────────────
@pytest.fixture
def health_module(tmp_path, monkeypatch):
    """Import chad.ops.health_monitor_rules with RUNTIME redirected to tmp."""
    import chad.ops.health_monitor_rules as hmr
    hmr = importlib.reload(hmr)
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hmr, "RUNTIME", runtime)
    return hmr, runtime


def _write_tier_enforcement(
    runtime: Path,
    *,
    tier: str,
    max_daily_loss_usd,
    budget_remaining_today_usd: float,
    daily_loss_limit_hit: bool = False,
) -> None:
    payload = {
        "tier": tier,
        "max_daily_loss_usd": max_daily_loss_usd,
        "budget_remaining_today_usd": budget_remaining_today_usd,
        "daily_loss_limit_hit": daily_loss_limit_hit,
        "ts_utc": _iso(datetime.now(timezone.utc)),
        "ttl_seconds": 300,
    }
    _write_json(runtime / "tier_enforcement_state.json", payload)


# ── Test 7: R15 fires when budget below 30% threshold ───────────────────────
def test_t7_r15_fires(health_module):
    hmr, runtime = health_module
    _write_tier_enforcement(
        runtime, tier="MICRO",
        max_daily_loss_usd=20.0,
        budget_remaining_today_usd=5.50,
    )
    findings: List[hmr.Finding] = []
    hmr.rule_tier_daily_loss_approaching(findings)
    ids = [f.rule_id for f in findings]
    assert "R15" in ids


# ── Test 8: R15 does NOT fire when budget at/above 30% threshold ────────────
def test_t8_r15_does_not_fire(health_module):
    hmr, runtime = health_module
    _write_tier_enforcement(
        runtime, tier="MICRO",
        max_daily_loss_usd=20.0,
        budget_remaining_today_usd=7.00,  # 35% — above the 30% threshold (6.0)
    )
    findings: List[hmr.Finding] = []
    hmr.rule_tier_daily_loss_approaching(findings)
    assert "R15" not in [f.rule_id for f in findings]


# ── Test 9: R15 silent on SCALE tier with null max_daily_loss ───────────────
def test_t9_r15_silent_on_scale_null_cap(health_module):
    hmr, runtime = health_module
    _write_tier_enforcement(
        runtime, tier="SCALE",
        max_daily_loss_usd=None,
        budget_remaining_today_usd=0.0,
    )
    findings: List[hmr.Finding] = []
    hmr.rule_tier_daily_loss_approaching(findings)
    assert "R15" not in [f.rule_id for f in findings]


def _write_setup_family(
    runtime: Path, family: str, trades: int, skip_count: int
) -> None:
    payload = {
        "families": {
            family: {
                "trades": trades,
                "skip_count_stop_too_wide": skip_count,
            }
        }
    }
    _write_json(runtime / "setup_family_expectancy.json", payload)


# ── Test 10: R16 fires when skip_count > trades*2 with trades>=5 ────────────
def test_t10_r16_fires(health_module):
    hmr, runtime = health_module
    _write_setup_family(runtime, "ORB", trades=5, skip_count=11)
    findings: List[hmr.Finding] = []
    hmr.rule_setup_family_skip_rate(findings)
    ids = [f.rule_id for f in findings]
    assert "R16" in ids


# ── Test 11: R16 does NOT fire at exact 2× boundary ─────────────────────────
def test_t11_r16_does_not_fire_at_boundary(health_module):
    hmr, runtime = health_module
    _write_setup_family(runtime, "ORB", trades=5, skip_count=10)
    findings: List[hmr.Finding] = []
    hmr.rule_setup_family_skip_rate(findings)
    assert "R16" not in [f.rule_id for f in findings]


# ── Test 12: R16 silent when trades < 5 (sample-size floor) ─────────────────
def test_t12_r16_silent_below_sample_floor(health_module):
    hmr, runtime = health_module
    _write_setup_family(runtime, "ORB", trades=4, skip_count=11)
    findings: List[hmr.Finding] = []
    hmr.rule_setup_family_skip_rate(findings)
    assert "R16" not in [f.rule_id for f in findings]


# ── Test 13: Non-alpha_intraday_micro positions are ignored ─────────────────
def test_t13_ignores_other_strategies(eod_module):
    mod, runtime = eod_module
    _write_tier_state(runtime, tier_name="MICRO", flatten_before_eod=True)
    _write_position_guard(runtime, {
        # alpha — equity, must be ignored
        "alpha|AAPL": {
            "open": True, "strategy": "alpha", "symbol": "AAPL",
            "side": "BUY", "quantity": 10.0,
        },
        # alpha_futures — futures but not micro, must be ignored
        "alpha_futures|MES": {
            "open": True, "strategy": "alpha_futures", "symbol": "MES",
            "side": "BUY", "quantity": 1.0,
        },
        # broker_sync residual — not alpha_intraday_micro, must be ignored
        "broker_sync|CVX": {
            "open": True, "strategy": "broker_sync", "symbol": "CVX",
            "side": "SELL", "quantity": 54.0,
        },
        # discretionary / manual
        "manual|TSLA": {
            "open": True, "strategy": "manual", "symbol": "TSLA",
            "side": "BUY", "quantity": 1.0,
        },
        # delta_pairs — must be ignored even if symbol is MES
        "delta_pairs|MES": {
            "open": True, "strategy": "delta_pairs", "symbol": "MES",
            "side": "BUY", "quantity": 2.0,
        },
    })
    status = _run(eod_module)
    assert status["status"] == "OK"
    assert status["positions_found"] == 0
    assert status["positions_closed"] == 0
    # Ledger may not even exist if no intents emitted; if it does, intents=[].
    ledger_path = runtime / "eod_flatten_intents.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger.get("intents") == []


# ── Test 14: Idempotency — second run does not duplicate intents ────────────
def test_t14_idempotent_no_duplicate_intent(eod_module):
    mod, runtime = eod_module
    _write_tier_state(runtime, tier_name="MICRO", flatten_before_eod=True)
    _write_position_guard(runtime, {
        "alpha_intraday_micro|MES": {
            "open": True,
            "strategy": "alpha_intraday_micro",
            "symbol": "MES",
            "side": "BUY",
            "quantity": 1.0,
        }
    })
    s1 = _run(eod_module)
    assert s1["positions_closed"] == 1
    s2 = _run(eod_module)
    # Same qualifying position, same pending intent already in ledger.
    # positions_closed stays stable because the second run treated the
    # existing pending intent as already satisfying the close requirement.
    assert s2["positions_closed"] == 1
    ledger = json.loads(
        (runtime / "eod_flatten_intents.json").read_text(encoding="utf-8")
    )
    # Only ONE pending intent for this position_key across both runs.
    pending_for_pk = [
        i for i in ledger["intents"]
        if i.get("position_key") == "alpha_intraday_micro|MES"
        and str(i.get("status", "")).lower() == "pending"
    ]
    assert len(pending_for_pk) == 1

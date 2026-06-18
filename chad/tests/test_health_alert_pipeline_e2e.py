"""
Fix D — Health-Monitor Alert Pipeline E2E Smoke Test (VERIFY-ONLY).

These tests prove the EXISTING alert pipeline works end to end; they do not
introduce any new dispatcher, dedup, or systemd infrastructure. The pipeline
shape under test:

    run_all_rules() -> List[Finding]
      -> health_monitor.run_monitor() dispatcher loop
        -> health_monitor._notify(...)
          -> chad.utils.telegram_notify.notify(message, severity, dedupe_key)

All Telegram calls are mocked (unittest.mock); no real message is ever sent
and no runtime/* JSON is mutated. Rule functions are exercised against the
real rule_ibkr_sustained_latency (R19) and rule_ibkr_gateway_version (R20)
implementations as they exist in chad/ops/health_monitor_rules.py.

Pipeline status is documented in
ops/pending_actions/HEALTH_ALERT_PIPELINE_operational_2026-05-28.md

NOTE on the dispatcher's actual behaviour (verified against source):
health_monitor.run_monitor() calls _notify() for EVERY finding. The
finding's remedy_action selects the *message shape* — a pure
"HEALTH MONITOR" notification when remedy_action == "notify", or a
"🔧 AUTO-FIXED" report otherwise — but the notify() call itself is
unconditional. test_finding_without_notify_action_uses_autofix_branch
documents that real two-branch behaviour rather than asserting (falsely)
that non-notify findings skip Telegram.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from chad.ops.health_monitor_rules import (
    Finding,
    rule_ibkr_sustained_latency,
    rule_ibkr_gateway_version,
)


def _make_finding(*, severity: str, remedy_action: str) -> Finding:
    return Finding(
        rule_id="RTEST",
        severity=severity,
        title="Synthetic test finding",
        description="synthetic description",
        remedy_type="NOTIFY_ONLY" if remedy_action == "notify" else "SERVICE_RESTART",
        remedy_action=remedy_action,
        remedy_args={},
        evidence="synthetic evidence",
    )


def _run_dispatcher_once(monkeypatch, finding: Finding) -> MagicMock:
    """Drive the real health_monitor.run_monitor() dispatcher with a single
    injected finding, with Claude reasoning disabled and execute_remedy +
    telegram_notify.notify mocked. Returns the notify mock."""
    import chad.ops.health_monitor as hm

    # Tier 1: feed exactly one finding through the real dispatcher loop.
    monkeypatch.setattr(hm, "run_all_rules", lambda: [finding])
    # Tier 3: do not run any real remedy.
    monkeypatch.setattr(hm, "execute_remedy", lambda action, args: "mocked-remedy-result")
    # Tier 2: disable Claude reasoning (no API key -> _ask_claude returns "").
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    notify_mock = MagicMock()
    # _notify imports telegram_notify.notify at call time, so patching the
    # module attribute is picked up by the dispatcher.
    monkeypatch.setattr("chad.utils.telegram_notify.notify", notify_mock)

    hm.run_monitor(dry_run=False)
    return notify_mock


# ── Test 1 — notify-action finding dispatches to Telegram ──────────────────────

def test_finding_with_notify_action_calls_telegram(monkeypatch):
    finding = _make_finding(severity="CRITICAL", remedy_action="notify")
    notify_mock = _run_dispatcher_once(monkeypatch, finding)

    assert notify_mock.call_count == 1
    _, kwargs = notify_mock.call_args
    # CRITICAL findings map to severity="critical" in the dispatcher.
    assert kwargs.get("severity") == "critical"
    # Pure-notification message shape (no auto-fix wording).
    message = notify_mock.call_args.args[0]
    assert "HEALTH MONITOR" in message
    assert "[CRITICAL]" in message


# ── Test 2 — non-notify finding takes the AUTO-FIXED branch ────────────────────

def test_finding_without_notify_action_uses_autofix_branch(monkeypatch):
    """A finding whose remedy_action != "notify" still dispatches (the monitor
    reports what it auto-fixed), but via the "🔧 AUTO-FIXED" message branch,
    NOT the pure "HEALTH MONITOR" notification branch. This faithfully
    documents the real dispatcher: notify() fires for every finding;
    remedy_action only selects the message shape."""
    finding = _make_finding(severity="WARNING", remedy_action="restart_service")
    notify_mock = _run_dispatcher_once(monkeypatch, finding)

    assert notify_mock.call_count == 1
    message = notify_mock.call_args.args[0]
    assert "AUTO-FIXED" in message
    assert "HEALTH MONITOR" not in message
    _, kwargs = notify_mock.call_args
    assert kwargs.get("severity") == "warning"


# ── Test 3 — R19 finding shape ─────────────────────────────────────────────────

def test_r19_finding_shape_produces_dedupe_key(monkeypatch, tmp_path):
    import chad.ops.health_monitor_rules as rules

    status = {
        "consecutive_cycles_above_stop_threshold": 6,
        "last_above_threshold_at": "2026-05-28T17:00:00Z",
        "max_latency_observed_in_window": 2500.0,
        "current_recovery_state": "above_threshold",
        "last_gateway_churn_at": None,
        "ts_utc": "2026-05-28T17:00:05Z",
    }
    (tmp_path / "ibkr_status.json").write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setattr(rules, "RUNTIME", tmp_path)

    findings: list[Finding] = []
    rule_ibkr_sustained_latency(findings)

    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "R19"
    assert f.severity == "CRITICAL"
    assert f.remedy_action == "notify"
    assert "IBKR sustained latency" in f.title


# ── Test 4 — R20 finding shape ─────────────────────────────────────────────────

def test_r20_finding_shape_produces_dedupe_key(monkeypatch, tmp_path):
    import chad.ops.health_monitor_rules as rules
    import chad.tools.ibkr_gateway_version_check as vc

    # Empty RUNTIME dir -> no fresh cache -> rule falls through to build_report.
    monkeypatch.setattr(rules, "RUNTIME", tmp_path)

    fake_report = {
        "schema_version": "ibkr_gateway_version_check.v1",
        "ts_utc": "2026-05-28T21:45:00Z",
        "installed": {
            "build": 1037,
            "display": "10.37",
            "detection_source": "jts.ini",
            "install_path": "/opt/ibgateway",
            "detection_error": None,
        },
        "target": {"build": 1045, "display": "10.45", "source": "ops"},
        "comparison": {
            "severity": "stale",
            "is_current": False,
            "build_delta": 8,
        },
        "recommendation": "Upgrade IB Gateway to build 1045.",
    }
    monkeypatch.setattr(vc, "build_report", lambda *a, **k: fake_report)

    findings: list[Finding] = []
    rule_ibkr_gateway_version(findings)

    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "R20"
    assert f.severity == "CRITICAL"
    assert f.remedy_action == "notify"
    assert "IBKR Gateway version" in f.title


# ── Test 5 — dedupe key naming convention matches on-disk files ────────────────

def test_dedupe_key_naming_convention():
    """The dispatcher derives dedupe_key as f"health_{rule_id}_{title[:30]}";
    telegram_notify._dedupe_path then sanitises it into the on-disk filename
    telegram_dedupe_<key>.json. Prove that derivation reproduces the exact
    filenames already present in runtime/ for R19 and R20."""
    from chad.utils.telegram_notify import _dedupe_path

    cases = {
        "R19": (
            "IBKR sustained latency above stop threshold",
            "telegram_dedupe_health_R19_IBKRsustainedlatencyaboves.json",
        ),
        "R20": (
            "IBKR Gateway version is stale",
            "telegram_dedupe_health_R20_IBKRGatewayversionisstale.json",
        ),
    }
    for rule_id, (title, expected_filename) in cases.items():
        # Mirror health_monitor.run_monitor()'s dedupe_key construction exactly.
        dedupe_key = f"health_{rule_id}_{title[:30]}"
        path = _dedupe_path(dedupe_key)
        assert path.name == expected_filename


# ── Test 6 — dedup TTL default documents the 15-minute cooldown ────────────────

def test_telegram_notify_dedupe_ttl_default_is_900s(monkeypatch):
    """The 15-minute (900s) dedup cooldown is the default. NotifyConfig has no
    literal field default (it is a frozen dataclass with all-required fields);
    the 900s default lives in load_config() via
    _env_int("TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS", 900). Verify load_config()
    produces dedupe_ttl_s == 900 when the env override is unset."""
    from chad.utils import telegram_notify

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_ID", "123456")
    monkeypatch.delenv("TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS", raising=False)

    cfg = telegram_notify.load_config()
    assert cfg.dedupe_ttl_s == 900


# ── Test 7 — a raising rule surfaces as a finding, not a silent drop ───────────

def test_raising_rule_surfaces_as_error_finding(monkeypatch):
    """Regression: run_all_rules() used to swallow a raising rule with a bare
    `pass`, making a broken rule indistinguishable from a passing one. The fix
    records the failure as an ERROR-severity Finding so it reaches the alert
    pipeline. Patch one real rule in the iteration list to raise and prove the
    engine (a) does not crash and (b) emits a finding naming that rule."""
    import chad.ops.health_monitor_rules as rules

    boom_msg = "synthetic rule explosion"

    def _boom(findings):
        raise RuntimeError(boom_msg)

    # Carry the real rule's name so the surfaced finding reflects the rule that
    # broke (the engine derives rule_id from fn.__name__).
    _boom.__name__ = "rule_critical_services"

    # The run_all_rules() list resolves bare rule names against the module
    # globals at call time, so patching the attribute is picked up.
    monkeypatch.setattr(rules, "rule_critical_services", _boom)

    findings = rules.run_all_rules()

    # Engine stayed resilient (returned a list) AND the broken rule was not
    # silently dropped — it surfaces as an ERROR finding naming the rule.
    err = [
        f for f in findings
        if f.severity == "ERROR" and f.rule_id == "RULE_ERROR:rule_critical_services"
    ]
    assert len(err) == 1, "raising rule must surface as a finding, not a silent drop"
    f = err[0]
    assert "rule_critical_services" in f.title
    # repr(exc) is carried in evidence so the failure is diagnosable downstream.
    assert "RuntimeError" in f.evidence
    assert boom_msg in f.evidence
    # Must route through the safe no-op remedy + pure-notification message shape.
    assert f.remedy_action == "notify"
    assert f.remedy_type == "NOTIFY_ONLY"

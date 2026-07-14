"""COACH-VOICE-L1 U4 — golden-text tests for the human alert formatter.

Covers, per template per mode:
  * the four coach questions are answered (What / Why / What CHAD did / Act),
  * SIMPLE (and STANDARD) never leak codes (unit names, rule ids, filenames,
    raw second counts, JSON paths, the PRO 'raw:' marker),
  * any term definition is <=10 words,
  * the <=6-line budget,
  * the unknown-kind safe fallback,
  * mode switching via config file + CHAD_COACH_MODE env + explicit arg,
  * the wired-in senders preserve dedupe keys and (service_failure) the A1
    delivery-audit fields.

Presentation only: no network, no LLM, no runtime reads.
"""
from __future__ import annotations

import json
import re

import pytest

from chad.utils import coach_voice as cv


# ── shared fact fixtures (mirror the real artifact/finding fields) ─────────────
VAR_FEED_FACTS = {
    "rule_id": "R02",
    "severity": "WARNING",
    "title": "Feed STALE (notify): var_state.json (5877963s old, TTL=86400s)",
    "description": (
        "var_state.json is 5877963s old — more than 2× TTL. Auto-restart "
        "suppressed: publisher is a trading engine or has no scheduled runner."
    ),
    "evidence": "mtime age=5877963s TTL=86400s feed=var_state.json",
    "remedy_action": "notify",
}

FEED_RESTART_FACTS = {
    "rule_id": "R02",
    "severity": "CRITICAL",
    "title": "Feed STALE: price_cache.json (400s old, TTL=180s)",
    "evidence": "mtime age=400s TTL=180s service=chad-ibkr-price-refresh.timer",
    "remedy_action": "restart_feed_publisher",
}

SERVICE_FAIL_FACTS = {
    "failed_unit": "chad-ibkr-bar-provider.service",
    "severity": "HIGH",
    "active_unit_status": "failed",
    "journal_tail": ["boom", "Traceback ...", "RuntimeError"],
    "artifact_path": "reports/service_failures/x.json",
}

SERVICE_TEST_FACTS = {
    "failed_unit": "chad-orchestrator.service",
    "severity": "HIGH",
    "active_unit_status": "active",
    "journal_tail": [],
    "journal_error": None,
}

STOP_BUS_FACTS = {"reason": "broker_latency"}
DRAWDOWN_FACTS = {"drawdown_pct": -6.2, "threshold_pct": 5.0}
EDGE_DECAY_FACTS = {"strategy": "omega_macro", "consecutive_losses": 4}
HEALTH_FINDING_FACTS = {
    "rule_id": "R05",
    "severity": "CRITICAL",
    "title": "Reconciliation RED — position mismatch",
    "description": "Reconciliation is RED. worst_diff=3 mismatches=2",
    "evidence": "reconciliation_state.json status=RED",
}
HEALTH_AUTOFIX_FACTS = {
    "rule_id": "R08",
    "severity": "CRITICAL",
    "title": "Zero-byte runtime file: scr_state.json",
    "remedy_result": "restored scr_state.json from backup",
}
HEALTH_ANALYSIS_FACTS = {"analysis": "Alpha cluster minor churn. SCR stable. Watch fills."}

ALL_CASES = [
    ("feed_stale", VAR_FEED_FACTS),
    ("feed_stale", FEED_RESTART_FACTS),
    ("service_failure", SERVICE_FAIL_FACTS),
    ("service_failure", SERVICE_TEST_FACTS),
    ("stop_bus", STOP_BUS_FACTS),
    ("drawdown", DRAWDOWN_FACTS),
    ("edge_decay", EDGE_DECAY_FACTS),
    ("health_finding", HEALTH_FINDING_FACTS),
    ("health_autofix", HEALTH_AUTOFIX_FACTS),
    ("health_analysis", HEALTH_ANALYSIS_FACTS),
    ("market_closed", {}),
]

# Literal codes that must never surface in SIMPLE/STANDARD for the fixtures above.
FORBIDDEN_IN_SIMPLE = [
    "var_state.json",
    "price_cache.json",
    "chad-ibkr-bar-provider.service",
    "chad-orchestrator.service",
    "chad-ibkr-price-refresh.timer",
    "reconciliation_state.json",
    "scr_state.json",
    "reports/service_failures",
    "5877963",
    "R02",
    "R05",
    "R08",
    "omega_macro",
    "broker_latency",
    "raw:",
    "evidence=",
    "TTL=",
]


def _lines(text: str):
    return [ln for ln in text.split("\n") if ln.strip()]


def _assert_four_questions(text: str):
    """Structurally verify the four coach questions are answered."""
    lines = _lines(text)
    assert len(lines) >= 4, f"expected >=4 coach lines, got {len(lines)}: {text!r}"
    # Q1/Q2/Q3 are distinct lines.
    assert lines[0] != lines[1] != lines[2], f"coach lines not distinct: {text!r}"
    # Q4 — the action answer is always present.
    assert "no action needed" in text.lower(), f"missing action answer: {text!r}"


# ── golden: four questions + line budget across every kind and mode ────────────
@pytest.mark.parametrize("kind,facts", ALL_CASES)
@pytest.mark.parametrize("mode", ["SIMPLE", "STANDARD", "PRO"])
def test_four_questions_and_line_budget(kind, facts, mode):
    out = cv.format_alert(kind, facts, mode)
    _assert_four_questions(out)
    lines = _lines(out)
    assert len(lines) <= cv.MAX_LINES, f"{kind}/{mode} exceeded line budget: {out!r}"
    # Leading severity emoji so the notifier delivers it verbatim.
    assert out.startswith(("ℹ️", "⚠️", "\U0001f6a8")), f"no severity emoji: {out!r}"


@pytest.mark.parametrize("kind,facts", ALL_CASES)
@pytest.mark.parametrize("mode", ["SIMPLE", "STANDARD"])
def test_no_codes_in_simple_or_standard(kind, facts, mode):
    out = cv.format_alert(kind, facts, mode)
    # Pattern-based scrub contract.
    for pat in cv._CODE_PATTERNS:
        m = pat.search(out)
        assert m is None, f"{kind}/{mode} leaked code {m.group(0)!r} via /{pat.pattern}/: {out!r}"
    # Literal fixture codes must be absent.
    for token in FORBIDDEN_IN_SIMPLE:
        assert token not in out, f"{kind}/{mode} leaked literal code {token!r}: {out!r}"


@pytest.mark.parametrize("kind,facts", ALL_CASES)
def test_pro_may_show_raw_marker(kind, facts):
    out = cv.format_alert(kind, facts, "PRO")
    # PRO carries a trailing raw marker/metric line for every templated kind.
    assert "raw:" in out, f"PRO missing raw marker line: {out!r}"
    _assert_four_questions(out)
    assert len(_lines(out)) <= cv.MAX_LINES


# ── golden: the VaR example target (the spec's worked example) ─────────────────
def test_var_stale_simple_matches_target_shape():
    out = cv.format_alert("feed_stale", VAR_FEED_FACTS, "SIMPLE")
    assert "VaR" in out
    assert "68 days" in out                      # 5877963s translated to days
    assert "scheduler was never built" in out    # the cause, in plain English
    assert "(it estimates your worst-case daily loss)" in out  # inline definition
    assert "informational" in out
    assert "No action needed" in out
    assert "5877963" not in out                  # no raw seconds
    assert "var_state.json" not in out           # no filename code


# ── golden: definitions are <=10 words ─────────────────────────────────────────
def _parenthetical_defs(text: str):
    return re.findall(r"\(([^)]+)\)", text)


@pytest.mark.parametrize("kind,facts", ALL_CASES)
@pytest.mark.parametrize("mode", ["SIMPLE", "STANDARD"])
def test_inline_definitions_within_ten_words(kind, facts, mode):
    out = cv.format_alert(kind, facts, mode)
    for definition in _parenthetical_defs(out):
        # Only definitions (lowercase, explanatory) — not code parentheticals,
        # which the no-codes test already forbids.
        words = definition.split()
        assert len(words) <= 10, f"{kind}/{mode} definition >10 words: {definition!r}"


def test_specific_term_definitions_are_short():
    dd = cv.format_alert("drawdown", DRAWDOWN_FACTS, "SIMPLE")
    assert "how far below the recent peak we are" in dd
    assert len("how far below the recent peak we are".split()) <= 10

    ed = cv.format_alert("edge_decay", EDGE_DECAY_FACTS, "SIMPLE")
    assert "a strategy that stopped working as well as before" in ed
    assert len("a strategy that stopped working as well as before".split()) <= 10

    var = cv.format_alert("feed_stale", VAR_FEED_FACTS, "SIMPLE")
    assert len("it estimates your worst-case daily loss".split()) <= 10
    assert "it estimates your worst-case daily loss" in var


# ── golden: service_failure distinguishes real vs test/manual ──────────────────
def test_service_failure_real_reads_as_a_failure():
    out = cv.format_alert("service_failure", SERVICE_FAIL_FACTS, "SIMPLE")
    assert out.startswith("\U0001f6a8")  # 🚨 for a real HIGH failure
    assert "stopped" in out.lower()
    assert "price-data feed" in out       # friendly name, not the unit code
    assert "test or manual" not in out.lower()


def test_service_failure_test_or_manual_is_reassuring():
    out = cv.format_alert("service_failure", SERVICE_TEST_FACTS, "SIMPLE")
    assert out.startswith("ℹ️")           # info, not alarm
    assert "test or manual run" in out.lower()
    assert "No action needed" in out


# ── unknown-kind fallback ──────────────────────────────────────────────────────
def test_unknown_kind_uses_safe_generic_and_answers_four_questions():
    out = cv.format_alert(
        "totally_unheard_of_kind",
        {"title": "Something novel", "description": "A plain detail."},
        "SIMPLE",
    )
    _assert_four_questions(out)
    assert "raw:" not in out  # SIMPLE hides the artifact path


def test_unknown_kind_pro_exposes_artifact_path_only_in_pro():
    facts = {"title": "Novel", "artifact_path": "reports/x.json"}
    simple = cv.format_alert("mystery_kind", facts, "SIMPLE")
    pro = cv.format_alert("mystery_kind", facts, "PRO")
    assert "reports/x.json" not in simple
    assert "reports/x.json" in pro


def test_format_alert_never_raises_on_garbage_facts():
    # None facts, weird types — must degrade, not crash.
    assert cv.format_alert("stop_bus", None, "SIMPLE")
    assert cv.format_alert("drawdown", {"drawdown_pct": "not-a-number"}, "SIMPLE")
    assert cv.format_alert("edge_decay", {"strategy": None, "consecutive_losses": "x"}, "PRO")


# ── mode resolution: explicit arg > env > config file > default ────────────────
def test_explicit_mode_beats_env(monkeypatch):
    monkeypatch.setenv("CHAD_COACH_MODE", "PRO")
    out = cv.format_alert("drawdown", DRAWDOWN_FACTS, "SIMPLE")
    assert "raw:" not in out  # explicit SIMPLE wins over env PRO


def test_env_overrides_config(monkeypatch, tmp_path):
    cfg = tmp_path / "coach_voice.json"
    cfg.write_text(json.dumps({"mode": "SIMPLE"}), encoding="utf-8")
    monkeypatch.setattr(cv, "COACH_CONFIG_PATH", cfg)
    monkeypatch.setenv("CHAD_COACH_MODE", "PRO")
    # mode=None → resolve; env PRO beats config SIMPLE.
    out = cv.format_alert("drawdown", DRAWDOWN_FACTS)
    assert "raw:" in out


def test_config_file_mode_used_when_no_env(monkeypatch, tmp_path):
    cfg = tmp_path / "coach_voice.json"
    cfg.write_text(json.dumps({"mode": "STANDARD"}), encoding="utf-8")
    monkeypatch.setattr(cv, "COACH_CONFIG_PATH", cfg)
    monkeypatch.delenv("CHAD_COACH_MODE", raising=False)
    assert cv.resolve_mode() == "STANDARD"
    out = cv.format_alert("drawdown", DRAWDOWN_FACTS)
    # STANDARD adds its extra context line and no raw marker.
    assert "raw:" not in out
    assert "Watch for another update only if it deepens" in out


def test_default_mode_is_simple(monkeypatch, tmp_path):
    monkeypatch.setattr(cv, "COACH_CONFIG_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.delenv("CHAD_COACH_MODE", raising=False)
    assert cv.resolve_mode() == "SIMPLE"


def test_invalid_mode_values_fall_through(monkeypatch, tmp_path):
    monkeypatch.setattr(cv, "COACH_CONFIG_PATH", tmp_path / "nope.json")
    monkeypatch.setenv("CHAD_COACH_MODE", "LOUD")  # not a valid mode
    assert cv.resolve_mode() == "SIMPLE"
    # invalid explicit arg also falls through to resolution
    out = cv.format_alert("drawdown", DRAWDOWN_FACTS, "BOGUS")
    assert out  # renders in resolved (SIMPLE) mode without raising


def test_standard_is_simple_plus_one_context_line():
    simple = _lines(cv.format_alert("stop_bus", STOP_BUS_FACTS, "SIMPLE"))
    standard = _lines(cv.format_alert("stop_bus", STOP_BUS_FACTS, "STANDARD"))
    assert standard[: len(simple)] == simple
    assert len(standard) == len(simple) + 1


# ── the shipped config file is valid and defaults to SIMPLE ────────────────────
def test_shipped_config_is_valid_simple():
    raw = json.loads(cv.COACH_CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw.get("mode") == "SIMPLE"
    assert raw.get("schema_version") == "coach_voice.v1"


# ── humanizers ─────────────────────────────────────────────────────────────────
def test_humanize_duration():
    assert cv.humanize_duration(5877963) == "68 days"
    assert cv.humanize_duration(86400) == "1 day"
    assert cv.humanize_duration(3600) == "1 hour"
    assert cv.humanize_duration(400) == "7 minutes"
    assert cv.humanize_duration(1) == "1 second"
    assert cv.humanize_duration(None) == "an unknown time"
    assert cv.humanize_duration("abc") == "an unknown time"


def test_humanize_name():
    assert cv.humanize_name("omega_macro") == "Omega Macro"
    assert cv.humanize_name("chad-ibkr-bar-provider.service") == "Ibkr Bar Provider"
    assert cv.humanize_name("") == "one of CHAD's parts"


# ── wire-in: hot-path sends preserve dedupe keys and route through coach ───────
def test_hot_path_sends_preserve_dedupe_keys_and_use_coach(monkeypatch):
    import chad.utils.telegram_notify as tn

    captured = {}

    def _capture(text, dedupe_key=None):
        captured[dedupe_key] = text
        return True

    monkeypatch.setattr(tn, "_send_raw_telegram", _capture)

    assert tn.send_stop_bus_alert("broker_latency") is True
    assert tn.send_drawdown_alert(-6.2, 5.0) is True
    assert tn.send_edge_decay_alert("omega_macro", 4) is True

    # Dedupe keys are exactly the pre-coach keys (A1 behavior preserved).
    assert "stop_bus_triggered" in captured
    assert "drawdown_threshold" in captured
    assert "edge_decay_omega_macro" in captured

    # And the composed text is the coach voice, not the legacy machine text.
    assert "paused all trading" in captured["stop_bus_triggered"]
    assert "STOP BUS TRIGGERED" not in captured["stop_bus_triggered"]
    assert "recent high" in captured["drawdown_threshold"]
    assert "No action needed" in captured["edge_decay_omega_macro"]


def test_hot_path_send_falls_back_to_legacy_if_coach_unavailable(monkeypatch):
    import chad.utils.telegram_notify as tn

    captured = {}

    def _capture(text, dedupe_key=None):
        captured[dedupe_key] = text
        return True

    monkeypatch.setattr(tn, "_send_raw_telegram", _capture)
    # Force the coach layer to be unavailable.
    monkeypatch.setattr(tn, "_coach_format", lambda kind, facts: None)

    assert tn.send_stop_bus_alert("broker_latency") is True
    # Legacy text is used, delivery is NOT lost.
    assert "STOP BUS TRIGGERED" in captured["stop_bus_triggered"]


# ── wire-in: service_failure preserves A1 delivery-audit fields + dedupe key ───
def test_service_failure_coach_preserves_a1_audit_fields(monkeypatch, tmp_path):
    import chad.ops.service_failure_alert as sfa
    from chad.utils.telegram_notify import NotifyOutcome, DeliveryStatus

    # Stub journal + systemctl so run() is hermetic.
    monkeypatch.setattr(sfa, "_read_journal_tail", lambda unit, n: (["line one", "line two"], None))
    monkeypatch.setattr(sfa, "_systemctl_active", lambda unit: "failed")

    seen = {}

    def _fake_notify_detailed(message, *, severity="info", dedupe_key=None):
        seen["message"] = message
        seen["dedupe_key"] = dedupe_key
        seen["severity"] = severity
        return NotifyOutcome(DeliveryStatus.SENT, None)

    monkeypatch.setattr(
        "chad.utils.telegram_notify.notify_detailed", _fake_notify_detailed, raising=False
    )

    res = sfa.run(
        failed_unit="chad-ibkr-bar-provider.service",
        severity="HIGH",
        include_runtime_snapshot=False,
        dry_run=False,
        artifact_dir=tmp_path,
    )

    # A1 delivery-audit fields intact and derived from the notifier outcome.
    assert res.payload["telegram_sent"] is True
    assert res.payload["telegram_delivery_status"] == "sent"
    assert res.payload["delivery_error"] is None
    assert res.exit_code == sfa.EXIT_OK
    # Dedupe key unchanged (per-unit).
    assert seen["dedupe_key"] == "service_failure:chad-ibkr-bar-provider.service"
    # The delivered message is the coach voice, not the legacy journal-tail dump.
    assert "No action needed" in seen["message"]
    assert "--- journal tail" not in seen["message"]
    assert seen["message"].startswith(("ℹ️", "⚠️", "\U0001f6a8"))


def test_service_failure_falls_back_to_legacy_when_coach_unavailable(monkeypatch):
    import chad.ops.service_failure_alert as sfa

    payload = {
        "failed_unit": "chad-backend.service",
        "severity": "HIGH",
        "active_unit_status": "failed",
        "journal_tail": ["boom"],
        "host": "h",
        "ts_utc": "2026-07-14T00:00:00Z",
        "artifact_path": "reports/service_failures/x.json",
    }
    monkeypatch.setattr(sfa, "_coach_message", lambda p: None)
    out = sfa._format_telegram_message(payload)
    # Legacy machine format still works as a hard fallback.
    assert "--- journal tail" in out
    assert "chad-backend.service" in out


# ── wire-in: health finding routing picks the right coach kind ─────────────────
def test_kind_for_finding_mapping():
    from chad.ops.health_monitor_rules import Finding

    feed = Finding("R02", "WARNING", "Feed STALE (notify): var_state.json (1s old, TTL=1s)",
                   "d", "NOTIFY_ONLY", "notify", evidence="feed=var_state.json")
    assert cv.kind_for_finding(feed) == "feed_stale"

    notify_only = Finding("R05", "CRITICAL", "Reconciliation RED", "d",
                          "NOTIFY_ONLY", "notify", evidence="e")
    assert cv.kind_for_finding(notify_only) == "health_finding"

    autofix = Finding("R08", "CRITICAL", "Zero-byte runtime file: scr_state.json", "d",
                      "SAFE_AUTO", "restore_from_backup", evidence="e")
    assert cv.kind_for_finding(autofix) == "health_autofix"

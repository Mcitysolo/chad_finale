"""NEW-GAP-044 — options-chain refresh failure / stale Greeks must alert loudly.

Pre-fix, when chad-options-chain-refresh.service exited 1 (e.g. SPY contract
details timeout) it wrote ``runtime/options_chains_cache.json`` with an
``error`` field but no rule re-surfaced that failure to the Telegram pipeline.
options_greeks_gate.get_option_greeks is deliberately failure-soft (returns
delta=±0.5 with source="default"), so strategies see "silent staleness" —
exactly what the Box-20 acceptance criterion forbids.

Two new health_monitor_rules entries (R17, R18) detect every degraded state
and emit a Finding(remedy_type="NOTIFY_ONLY") so the existing health monitor
loop sends a Telegram alert.

This file pins:
  * R17 detects ``error`` field on the chain cache (CRITICAL).
  * R17 detects empty chains map (CRITICAL).
  * R17 detects missing file (INFO).
  * R17 detects stale ts_utc on a weekday (WARNING).
  * R17 stays quiet on a healthy fresh cache.
  * R18 detects missing Greeks file (CRITICAL).
  * R18 detects stale Greeks beyond declared ttl_seconds (CRITICAL).
  * R18 detects status=partial (WARNING).
  * R18 detects empty symbols map (WARNING).
  * R18 stays quiet on a healthy fresh Greeks file.
  * Both rules emit remedy_type="NOTIFY_ONLY" so the existing notify path
    in chad/ops/health_monitor.py picks them up.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from chad.ops import health_monitor_rules as hmr


# ---------------------------------------------------------------------------
# Test fixture: redirect module-level RUNTIME at tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    return tmp_path


def _utc_iso(offset_seconds: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.isoformat().replace("+00:00", "Z")


def _write_chain(runtime: Path, *, payload: dict) -> Path:
    p = runtime / "options_chains_cache.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _write_greeks(runtime: Path, *, payload: dict) -> Path:
    p = runtime / "options_greeks.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _r17(runtime: Path) -> list[hmr.Finding]:
    findings: list[hmr.Finding] = []
    hmr.rule_options_chain_refresh_health(findings)
    return findings


def _r18(runtime: Path) -> list[hmr.Finding]:
    findings: list[hmr.Finding] = []
    hmr.rule_options_greeks_freshness(findings)
    return findings


# ---------------------------------------------------------------------------
# R17 — chain refresh health
# ---------------------------------------------------------------------------


def test_r17_error_field_emits_critical_notify(_runtime: Path) -> None:
    """Headline GAP-044 reproducer: today's failure persisted ``error``."""
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-60),
        "chains": {},
        "error": "all_symbols_failed: SPY=timeout_after_30.0s",
    })
    findings = _r17(_runtime)
    assert len(findings) == 1
    f = findings[0]
    assert f.rule_id == "R17"
    assert f.severity == "CRITICAL"
    assert f.remedy_type == "NOTIFY_ONLY"
    assert f.remedy_action == "notify"
    assert "all_symbols_failed" in f.evidence
    assert "SPY" in f.evidence


def test_r17_empty_chains_no_error_emits_critical(_runtime: Path) -> None:
    """Degenerate path: chains={} but no error field — still fail loudly."""
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-60),
        "chains": {},
    })
    findings = _r17(_runtime)
    assert len(findings) == 1
    assert findings[0].severity == "CRITICAL"
    assert "empty" in findings[0].title.lower()


def test_r17_missing_file_emits_info(_runtime: Path) -> None:
    """Missing file is not as severe as failed refresh — INFO is enough,
    but still goes through the notify pipeline."""
    findings = _r17(_runtime)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "INFO"
    assert f.remedy_action == "notify"


def test_r17_fresh_cache_with_chains_emits_no_finding(_runtime: Path) -> None:
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-60),
        "chains": {"SPY": {"strikes": [400, 410], "expirations": ["2026-06-20"]}},
    })
    findings = _r17(_runtime)
    assert findings == []


def test_r17_stale_cache_on_weekday_emits_warning(
    _runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ts_utc older than 26h with `is_weekday()` true must produce a WARNING."""
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-27 * 3600),
        "chains": {"SPY": {"strikes": [400], "expirations": ["2026-06-20"]}},
    })
    monkeypatch.setattr(hmr, "is_weekday", lambda: True)
    findings = _r17(_runtime)
    # The healthy `chains` short-circuit returns before the ts staleness check
    # only if error+empty branches don't fire. ts staleness emits WARNING.
    assert len(findings) == 1
    assert findings[0].severity == "WARNING"
    assert "stale" in findings[0].title.lower()


def test_r17_stale_cache_on_weekend_emits_no_finding(
    _runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weekend grace: don't page operators on Saturday/Sunday for a daily
    Mon-Fri refresher that legitimately won't run."""
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-50 * 3600),  # ~2 days old
        "chains": {"SPY": {"strikes": [400], "expirations": ["2026-06-20"]}},
    })
    monkeypatch.setattr(hmr, "is_weekday", lambda: False)
    findings = _r17(_runtime)
    assert findings == []


# ---------------------------------------------------------------------------
# R18 — Greeks freshness
# ---------------------------------------------------------------------------


def test_r18_missing_greeks_file_emits_critical(_runtime: Path) -> None:
    findings = _r18(_runtime)
    assert len(findings) == 1
    f = findings[0]
    assert f.severity == "CRITICAL"
    assert f.remedy_action == "notify"
    assert "missing" in f.title.lower()


def test_r18_stale_greeks_beyond_ttl_emits_critical(_runtime: Path) -> None:
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-100000),  # ~27.8 hours old
        "ttl_seconds": 90000,
        "status": "ok",
        "symbols": {"SPY": {}},
    })
    findings = _r18(_runtime)
    assert len(findings) == 1
    assert findings[0].severity == "CRITICAL"
    assert "stale" in findings[0].title.lower()


def test_r18_status_partial_emits_warning(_runtime: Path) -> None:
    """The exact production state today: synthetic provider, partial result."""
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-60),
        "ttl_seconds": 90000,
        "status": "partial",
        "symbols": {"SPY": {"calls": {}}},
    })
    findings = _r18(_runtime)
    titles = [f.title for f in findings]
    assert any("partial" in t.lower() for t in titles)
    # Must include a WARNING-severity notify finding for partial status
    assert any(f.severity == "WARNING" for f in findings)


def test_r18_empty_symbols_emits_warning(_runtime: Path) -> None:
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-60),
        "ttl_seconds": 90000,
        "status": "ok",
        "symbols": {},
    })
    findings = _r18(_runtime)
    assert any("empty" in f.title.lower() for f in findings)
    assert any(f.severity == "WARNING" for f in findings)


def test_r18_failed_status_emits_critical(_runtime: Path) -> None:
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-60),
        "ttl_seconds": 90000,
        "status": "failed",
        "symbols": {},
    })
    findings = _r18(_runtime)
    assert len(findings) == 1
    assert findings[0].severity == "CRITICAL"
    assert findings[0].remedy_action == "notify"


def test_r18_fresh_healthy_greeks_emits_no_finding(_runtime: Path) -> None:
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-60),
        "ttl_seconds": 90000,
        "status": "ok",
        "symbols": {"SPY": {"calls": {}, "puts": {}}},
    })
    findings = _r18(_runtime)
    assert findings == []


# ---------------------------------------------------------------------------
# Wiring: both rules are registered in run_all_rules
# ---------------------------------------------------------------------------


def test_run_all_rules_includes_new_options_rules(
    _runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Belt-and-braces: run_all_rules must dispatch through R17/R18, not
    just expose them as helpers. We trigger both failure modes and assert
    at least one finding from each rule_id shows up."""
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-60),
        "chains": {},
        "error": "all_symbols_failed: SPY=timeout_after_30.0s",
    })
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-100000),
        "ttl_seconds": 90000,
        "status": "partial",
        "symbols": {},
    })
    findings = hmr.run_all_rules()
    rule_ids = {f.rule_id for f in findings}
    assert "R17" in rule_ids
    assert "R18" in rule_ids


# ---------------------------------------------------------------------------
# Anti-regression: no rule can silence the alert by changing remedy_type
# ---------------------------------------------------------------------------


def test_every_options_finding_routes_through_notify_pipeline(_runtime: Path) -> None:
    """Failure cases must always go through NOTIFY_ONLY → telegram. A
    drive-by patch that changed remedy_type to SAFE_AUTO would silently
    suppress the alert without breaking the test suite — pin it."""
    _write_chain(_runtime, payload={
        "schema_version": "options_chain_cache.v2",
        "ts_utc": _utc_iso(-60),
        "chains": {},
        "error": "all_symbols_failed: SPY=timeout_after_30.0s",
    })
    _write_greeks(_runtime, payload={
        "schema_version": "options_greeks.v1",
        "ts_utc": _utc_iso(-100000),
        "ttl_seconds": 90000,
        "status": "failed",
        "symbols": {},
    })
    findings = _r17(_runtime) + _r18(_runtime)
    assert findings, "expected at least one finding from the failure fixture"
    for f in findings:
        assert f.remedy_action == "notify"
        assert f.remedy_type == "NOTIFY_ONLY"

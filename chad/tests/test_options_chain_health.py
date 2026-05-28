"""Unit tests for OPTIONS-CHAIN-1 freshness validator + health rule + strategy fail-closed."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from chad.market_data import options_chain_freshness as fresh

NOW = datetime(2026, 5, 27, 20, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_cache(path: Path, *, ts: datetime, chains: dict | None = None, error: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts_utc": _iso(ts), "chains": chains or {}}
    if error is not None:
        payload["error"] = error
    path.write_text(json.dumps(payload))


def _write_failure(path: Path, *, ts: datetime, reason: str = "ibkr_contract_details_unresponsive"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "options_chain_refresh_failure.v1",
        "ts_utc": _iso(ts),
        "status": "failed",
        "blocked_reason": reason,
        "error_type": "all_symbols_failed",
    }
    path.write_text(json.dumps(payload))


# --- is_chain_fresh ---------------------------------------------------------

def test_chain_fresh_when_recent_and_populated(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    _write_cache(cache, ts=NOW - timedelta(minutes=30), chains={"SPY": {"strikes": []}})
    ok, reason, _ = fresh.is_chain_fresh(cache_path=cache, now=NOW)
    assert ok and reason == "fresh"


def test_chain_stale_when_too_old(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    _write_cache(cache, ts=NOW - timedelta(hours=48), chains={"SPY": {"strikes": []}})
    ok, reason, details = fresh.is_chain_fresh(cache_path=cache, now=NOW)
    assert not ok and reason == "cache_stale"
    assert details["cache_age_seconds"] > 24 * 3600


def test_chain_missing(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    ok, reason, _ = fresh.is_chain_fresh(cache_path=cache, now=NOW)
    assert not ok and reason == "cache_missing"


def test_chain_error_field_blocks_freshness(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    _write_cache(cache, ts=NOW, chains={}, error="ibkr_unresponsive")
    ok, reason, _ = fresh.is_chain_fresh(cache_path=cache, now=NOW)
    assert not ok and reason == "cache_error_present"


def test_chain_empty_chains_blocks_freshness(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    _write_cache(cache, ts=NOW, chains={})
    ok, reason, _ = fresh.is_chain_fresh(cache_path=cache, now=NOW)
    assert not ok and reason == "cache_empty_chains"


# --- is_failure_artifact_fresh ----------------------------------------------

def test_failure_artifact_fresh_within_window(tmp_path):
    fa = tmp_path / "options_chain_refresh_failure.json"
    _write_failure(fa, ts=NOW - timedelta(minutes=10))
    is_fresh, details = fresh.is_failure_artifact_fresh(failure_path=fa, now=NOW)
    assert is_fresh
    assert details["failure_artifact_reason"] == "ibkr_contract_details_unresponsive"


def test_failure_artifact_stale_outside_window(tmp_path):
    fa = tmp_path / "options_chain_refresh_failure.json"
    _write_failure(fa, ts=NOW - timedelta(hours=12))
    is_fresh, _ = fresh.is_failure_artifact_fresh(failure_path=fa, now=NOW)
    assert not is_fresh


def test_failure_artifact_missing(tmp_path):
    fa = tmp_path / "options_chain_refresh_failure.json"
    is_fresh, details = fresh.is_failure_artifact_fresh(failure_path=fa, now=NOW)
    assert not is_fresh
    assert not details["failure_artifact_exists"]


# --- chain_usability aggregate ---------------------------------------------

def test_usability_blocked_by_fresh_failure_even_when_cache_recent(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    fa = tmp_path / "options_chain_refresh_failure.json"
    _write_cache(cache, ts=NOW - timedelta(minutes=5), chains={"SPY": {"strikes": []}})
    _write_failure(fa, ts=NOW - timedelta(minutes=10))
    v = fresh.chain_usability(cache_path=cache, failure_path=fa, now=NOW)
    assert v.usable is False
    assert v.reason == "fresh_failure_artifact"


def test_usability_ok_when_cache_fresh_and_no_failure(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    fa = tmp_path / "options_chain_refresh_failure.json"
    _write_cache(cache, ts=NOW, chains={"SPY": {"strikes": []}})
    v = fresh.chain_usability(cache_path=cache, failure_path=fa, now=NOW)
    assert v.usable is True
    assert v.reason == "usable"


def test_usability_blocked_by_stale_cache_when_no_failure(tmp_path):
    cache = tmp_path / "options_chains_cache.json"
    fa = tmp_path / "options_chain_refresh_failure.json"
    _write_cache(cache, ts=NOW - timedelta(hours=72), chains={"SPY": {}})
    v = fresh.chain_usability(cache_path=cache, failure_path=fa, now=NOW)
    assert v.usable is False
    assert v.reason == "cache_stale"


# --- health monitor rule integration ---------------------------------------

def test_health_rule_critical_when_failure_artifact_fresh(monkeypatch, tmp_path):
    from chad.ops import health_monitor_rules as hmr
    from chad.market_data import options_chain_freshness as ocf

    fa = tmp_path / "options_chain_refresh_failure.json"
    _write_failure(fa, ts=datetime.now(timezone.utc) - timedelta(minutes=5))

    original = ocf.is_failure_artifact_fresh

    def _patched(failure_path=None, fresh_window_seconds=None, now=None):
        return original(
            failure_path=fa,
            fresh_window_seconds=fresh_window_seconds or ocf.DEFAULT_FAILURE_FRESH_SECONDS,
            now=now,
        )

    monkeypatch.setattr(ocf, "is_failure_artifact_fresh", _patched)

    findings: list = []
    hmr.rule_options_chain_refresh_failure_artifact(findings)
    r17b = [f for f in findings if f.rule_id == "R17b"]
    assert r17b, "rule did not emit R17b finding"
    assert r17b[0].severity == "CRITICAL"


def test_health_rule_no_finding_when_artifact_missing(monkeypatch, tmp_path):
    from chad.ops import health_monitor_rules as hmr
    from chad.market_data import options_chain_freshness as ocf

    nonexistent = tmp_path / "no_such_file.json"
    original = ocf.is_failure_artifact_fresh

    def _patched(failure_path=None, fresh_window_seconds=None, now=None):
        return original(
            failure_path=nonexistent,
            fresh_window_seconds=fresh_window_seconds or ocf.DEFAULT_FAILURE_FRESH_SECONDS,
            now=now,
        )

    monkeypatch.setattr(ocf, "is_failure_artifact_fresh", _patched)

    findings: list = []
    hmr.rule_options_chain_refresh_failure_artifact(findings)
    r17b = [f for f in findings if f.rule_id == "R17b"]
    assert not r17b


# --- strategy fail-closed --------------------------------------------------

def test_alpha_options_chain_load_returns_none_on_fresh_failure(monkeypatch, tmp_path):
    from chad.strategies import alpha_options
    from chad.market_data import options_chain_freshness as ocf

    def _unusable(*a, **kw):
        return ocf.FreshnessVerdict(
            usable=False,
            reason="fresh_failure_artifact",
            cache_exists=True,
            cache_age_seconds=300.0,
            cache_error=None,
            cache_chains_count=5,
            failure_artifact_exists=True,
            failure_artifact_age_seconds=600.0,
            failure_artifact_reason="ibkr_contract_details_unresponsive",
        )

    monkeypatch.setattr(
        "chad.market_data.options_chain_freshness.chain_usability", _unusable
    )
    result = alpha_options._load_chain_from_cache("SPY")
    assert result is None

"""
IR1 (INTEL-REPAIR) regression tests.

Covers the thaw and the loud-staleness behaviour introduced when the advisory
refresh was found to have zero working LLM tiers (Anthropic credit-exhaustion
400, Ollama timeout, and an OpenAI fallback that could never fire because
/etc/chad/openai.env was never loaded).

R4(a): a FRESH cache entry passes the 300s freshness gate and
       _apply_intelligence_bias() actually applies the bias (and a STALE entry
       is ignored — proving the gate).
R4(b): the failure path emits the INTEL_REFRESH_FAILED marker and does NOT
       overwrite last-good cache data.

Deterministic and offline — no network, no real LLM calls.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc))


class _Intent:
    """Minimal stand-in for a trade intent (symbol/strategy/confidence)."""

    def __init__(self, symbol: str, strategy: str, confidence: float):
        self.symbol = symbol
        self.strategy = strategy
        self.confidence = confidence


# --------------------------------------------------------------------------- #
# R4(a) — fresh entry passes the 300s gate and _apply_intelligence_bias applies
# --------------------------------------------------------------------------- #

def _seed_cache(path, key: str, adjustment: float, ts_utc: str):
    path.write_text(
        json.dumps({
            "confidence": {key: {
                "symbol": key.split("|")[0],
                "strategy": key.split("|")[1],
                "adjustment": adjustment,
                "reason": "test",
                "macro_risk": "low",
                "regime": "neutral",
                "ts_utc": ts_utc,
            }},
            "regime": {},
            "last_updated_utc": ts_utc,
        }),
        encoding="utf-8",
    )


def test_fresh_negative_bias_suppresses_intent(tmp_path, monkeypatch):
    from chad.core import live_loop
    cache = tmp_path / "strategy_intelligence_cache.json"
    # Fresh entry, strong negative adjustment: 0.50 + (-0.40) = 0.10 < 0.20 -> drop
    _seed_cache(cache, "AAPL|alpha", -0.40, _now_iso())
    monkeypatch.setattr(live_loop, "_INTELLIGENCE_CACHE_PATH", cache)

    intents = [_Intent("AAPL", "alpha", 0.50)]
    out = live_loop._apply_intelligence_bias(intents, logging.getLogger("test"))
    assert out == [], "fresh sub-threshold intent must be suppressed by the bias"


def test_fresh_small_bias_keeps_intent(tmp_path, monkeypatch):
    from chad.core import live_loop
    cache = tmp_path / "strategy_intelligence_cache.json"
    # Fresh entry, small adjustment: 0.50 + (-0.05) = 0.45 >= 0.20 -> keep
    _seed_cache(cache, "AAPL|alpha", -0.05, _now_iso())
    monkeypatch.setattr(live_loop, "_INTELLIGENCE_CACHE_PATH", cache)

    intents = [_Intent("AAPL", "alpha", 0.50)]
    out = live_loop._apply_intelligence_bias(intents, logging.getLogger("test"))
    assert len(out) == 1, "fresh above-threshold intent must survive"


def test_stale_entry_ignored_by_300s_gate(tmp_path, monkeypatch):
    from chad.core import live_loop
    cache = tmp_path / "strategy_intelligence_cache.json"
    # Same strong negative adjustment but STALE (well beyond 300s) -> gate rejects
    stale_ts = _iso(datetime.now(timezone.utc) - timedelta(seconds=3600))
    _seed_cache(cache, "AAPL|alpha", -0.40, stale_ts)
    monkeypatch.setattr(live_loop, "_INTELLIGENCE_CACHE_PATH", cache)

    intents = [_Intent("AAPL", "alpha", 0.50)]
    out = live_loop._apply_intelligence_bias(intents, logging.getLogger("test"))
    assert len(out) == 1, "stale entry must NOT bias (fail-open past the 300s gate)"


# --------------------------------------------------------------------------- #
# R4(b) — failure emits the marker and does not overwrite last-good
# --------------------------------------------------------------------------- #

def test_failed_run_writes_marker_and_increments(tmp_path, monkeypatch):
    from chad.ops import strategy_intelligence_refresh as sir
    state = tmp_path / "intel_refresh_state.json"
    monkeypatch.setattr(sir, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(sir, "STATE_PATH", state)

    s1 = sir._write_refresh_state(outcome="FAILED", real_calls=0, fallback_calls=9,
                                  last_error_class="anthropic_credit_exhausted")
    assert s1["marker"] == "INTEL_REFRESH_FAILED"
    assert s1["consecutive_failures"] == 1
    assert s1["last_error_class"] == "anthropic_credit_exhausted"

    # A second consecutive failure increments; last_success_utc stays None.
    s2 = sir._write_refresh_state(outcome="FAILED", real_calls=0, fallback_calls=9,
                                  last_error_class="anthropic_credit_exhausted")
    assert s2["consecutive_failures"] == 2
    assert s2["last_success_utc"] is None

    on_disk = json.loads(state.read_text(encoding="utf-8"))
    assert on_disk["marker"] == "INTEL_REFRESH_FAILED"
    assert on_disk["consecutive_failures"] == 2


def test_ok_run_clears_marker_and_stamps_success(tmp_path, monkeypatch):
    from chad.ops import strategy_intelligence_refresh as sir
    state = tmp_path / "intel_refresh_state.json"
    monkeypatch.setattr(sir, "RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(sir, "STATE_PATH", state)

    sir._write_refresh_state(outcome="FAILED", real_calls=0, fallback_calls=9,
                             last_error_class="x")
    ok = sir._write_refresh_state(outcome="OK", real_calls=14, fallback_calls=0)
    assert ok["marker"] is None
    assert ok["outcome"] == "OK"
    assert ok["consecutive_failures"] == 0
    assert ok["last_success_utc"] is not None


def test_health_rule_notifies_after_threshold(tmp_path, monkeypatch):
    from chad.ops import health_monitor_rules as hmr
    monkeypatch.setattr(hmr, "RUNTIME", tmp_path)
    state = tmp_path / "intel_refresh_state.json"

    # Below threshold: no finding.
    state.write_text(json.dumps({
        "marker": "INTEL_REFRESH_FAILED", "consecutive_failures": 1,
        "last_error_class": "anthropic_credit_exhausted", "last_success_utc": None,
    }), encoding="utf-8")
    findings: list = []
    hmr.rule_intel_refresh_stale(findings)
    assert findings == []

    # At/above threshold: exactly one NOTIFY_ONLY finding, dedupe-stable title.
    state.write_text(json.dumps({
        "marker": "INTEL_REFRESH_FAILED",
        "consecutive_failures": hmr._INTEL_REFRESH_FAIL_THRESHOLD,
        "last_error_class": "anthropic_credit_exhausted", "last_success_utc": None,
    }), encoding="utf-8")
    findings = []
    hmr.rule_intel_refresh_stale(findings)
    assert len(findings) == 1
    f = findings[0]
    assert f.remedy_type == "NOTIFY_ONLY"
    # Title must be stable (no fluctuating count/timestamp in the dedupe prefix).
    assert f.title == "Advisory intel refresh FAILED — no live LLM tier"
    assert "3" not in f.title  # count lives in the description, not the title


class _FallbackClient:
    """Fake ClaudeClient whose chat_json always returns the neutral stub."""

    def chat_json(self, *args, **kwargs):
        return {
            "error": "all_ai_providers_unavailable",
            "fallback": True,
            "message": "unavailable",
            "ts_utc": _now_iso(),
        }


def test_failure_does_not_overwrite_last_good(tmp_path, monkeypatch):
    from chad.intel.strategy_intelligence import StrategyIntelligence
    # Seed a distinctive last-good entry with an OLD ts (so it is not fresh and
    # get_confidence_bias proceeds to _fetch, which then hits the stub).
    cache = tmp_path / "strategy_intelligence_cache.json"
    old_ts = _iso(datetime.now(timezone.utc) - timedelta(seconds=3600))
    _seed_cache(cache, "AAPL|alpha", -0.12, old_ts)
    before = cache.read_text(encoding="utf-8")

    si = StrategyIntelligence(_FallbackClient(), tmp_path)
    bias = si.get_confidence_bias(symbol="AAPL", strategy_name="alpha", base_confidence=0.5)

    # Neutral is returned on failure...
    assert bias.adjustment == 0.0
    # ...and the on-disk last-good entry is byte-for-byte unchanged (not clobbered
    # with a fresh-timestamped neutral 0.0).
    after = cache.read_text(encoding="utf-8")
    assert after == before, "failure path must not overwrite last-good cache data"


# --------------------------------------------------------------------------- #
# R1 — billing-400 latch + short-circuit (no network)
# --------------------------------------------------------------------------- #

class _FakeMessages:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self.exc


class _FakeAnthropic:
    def __init__(self, exc):
        self.messages = _FakeMessages(exc)


def test_credit_exhaustion_latches_and_short_circuits():
    from chad.intel.claude_client import ClaudeClient, ClaudeConfig, ClaudeAPIError
    c = ClaudeClient(config=ClaudeConfig(api_key="test"))
    fake = _FakeAnthropic(Exception(
        "Error code: 400 - {'type':'error','error':{'type':'invalid_request_error',"
        "'message':'Your credit balance is too low to access the Anthropic API.'}}"
    ))
    c._client = fake

    # First call: 400 -> raises ClaudeAPIError, latches the tier off.
    with pytest.raises(ClaudeAPIError):
        c._call_claude(prompt="hi", task_type="standard")
    assert c.anthropic_unavailable is True
    assert c.last_error_class == "anthropic_credit_exhausted"
    assert fake.messages.calls == 1  # no wasted retries on a billing 400

    # Second call: short-circuits BEFORE hitting the API (stops the retry storm).
    with pytest.raises(ClaudeAPIError):
        c._call_claude(prompt="hi again", task_type="standard")
    assert fake.messages.calls == 1  # create() was NOT called again


# --------------------------------------------------------------------------- #
# R2 — model-ID source of truth is fail-safe
# --------------------------------------------------------------------------- #

def test_llm_models_fallback_safe(monkeypatch):
    monkeypatch.setenv("CHAD_LLM_MODELS_PATH", "/does/not/exist.json")
    import importlib
    from chad.intel import llm_models
    importlib.reload(llm_models)
    try:
        assert llm_models.anthropic_tiers()["complex"] == "claude-sonnet-4-6"
        assert llm_models.openai_model() == "gpt-4.1"
        assert "claude-haiku-4-5-20251001" in llm_models.anthropic_cost_per_1k()
    finally:
        monkeypatch.delenv("CHAD_LLM_MODELS_PATH", raising=False)
        importlib.reload(llm_models)

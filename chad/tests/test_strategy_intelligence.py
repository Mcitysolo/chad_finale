"""
Tests for chad.intel.strategy_intelligence — StrategyIntelligence

All Claude API calls are mocked. No real API calls are made.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chad.intel.strategy_intelligence import (
    StrategyIntelligence,
    ConfidenceBias,
    UniverseFilter,
    CONFIDENCE_BIAS_MIN,
    CONFIDENCE_BIAS_MAX,
    _clamp,
    _neutral_bias,
    _neutral_universe_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime_dir(tmp_path):
    """Create a temporary runtime directory with mock state files."""
    rd = tmp_path / "runtime"
    rd.mkdir()

    # macro_state.json
    (rd / "macro_state.json").write_text(json.dumps({
        "vix": 18.5,
        "risk_label": "low",
        "schema_version": "macro_state.v1",
    }))

    # execution_quality.json
    (rd / "execution_quality.json").write_text(json.dumps({
        "avg_slippage_bps": 1.2,
        "fill_rate": 0.98,
    }))

    return rd


@pytest.fixture
def mock_client():
    """Create a mock ClaudeClient."""
    client = MagicMock()
    client.chat_json.return_value = {
        "adjustment": 0.02,
        "reason": "Low VIX, clean technicals",
        "macro_risk": "low",
        "regime": "risk_on",
    }
    return client


@pytest.fixture
def si(mock_client, runtime_dir):
    """Create a StrategyIntelligence instance with mocked client."""
    with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
        return StrategyIntelligence(mock_client, runtime_dir)


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_confidence_bias_returns_neutral_on_error(self, runtime_dir):
        """If Claude client throws, return neutral bias (0.0)."""
        broken_client = MagicMock()
        broken_client.chat_json.side_effect = RuntimeError("API down")

        with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
            si = StrategyIntelligence(broken_client, runtime_dir)
            bias = si.get_confidence_bias("MES", "alpha_futures", 0.7)

        assert bias.adjustment == 0.0
        assert "error" in bias.reason

    def test_universe_filter_returns_empty_on_error(self, runtime_dir):
        """If Claude client throws, return empty filter."""
        broken_client = MagicMock()
        broken_client.chat_json.side_effect = RuntimeError("API down")

        with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
            si = StrategyIntelligence(broken_client, runtime_dir)
            uf = si.get_universe_filter("alpha_futures", ["MES", "MNQ"])

        assert uf.avoid_symbols == []
        assert uf.prefer_symbols == []

    def test_regime_profile_returns_normal_on_error(self, runtime_dir):
        """If Claude client throws, return 'normal'."""
        broken_client = MagicMock()
        broken_client.chat_json.side_effect = RuntimeError("API down")

        si = StrategyIntelligence(broken_client, runtime_dir)
        profile = si.get_regime_profile("alpha_futures")
        assert profile == "normal"


# ---------------------------------------------------------------------------
# Confidence bias bounds
# ---------------------------------------------------------------------------


class TestConfidenceBiasBounds:
    def test_clamp_enforces_min(self):
        assert _clamp(-0.5, CONFIDENCE_BIAS_MIN, CONFIDENCE_BIAS_MAX) == CONFIDENCE_BIAS_MIN

    def test_clamp_enforces_max(self):
        assert _clamp(0.5, CONFIDENCE_BIAS_MIN, CONFIDENCE_BIAS_MAX) == CONFIDENCE_BIAS_MAX

    def test_clamp_passes_through_valid(self):
        assert _clamp(0.05, CONFIDENCE_BIAS_MIN, CONFIDENCE_BIAS_MAX) == 0.05

    def test_bias_adjustment_clamped_from_claude(self, runtime_dir):
        """Even if Claude returns out-of-range, it gets clamped."""
        client = MagicMock()
        client.chat_json.return_value = {
            "adjustment": 0.99,  # Way too high
            "reason": "Very bullish",
            "macro_risk": "low",
            "regime": "risk_on",
        }

        with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
            si = StrategyIntelligence(client, runtime_dir)
            bias = si.get_confidence_bias("MES", "alpha_futures", 0.7)

        assert bias.adjustment == CONFIDENCE_BIAS_MAX  # Clamped to +0.10

    def test_bias_negative_clamped(self, runtime_dir):
        """Negative bias clamped to -0.15."""
        client = MagicMock()
        client.chat_json.return_value = {
            "adjustment": -0.50,
            "reason": "Crisis",
            "macro_risk": "extreme",
            "regime": "crisis",
        }

        with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
            si = StrategyIntelligence(client, runtime_dir)
            bias = si.get_confidence_bias("MES", "alpha_futures", 0.7)

        assert bias.adjustment == CONFIDENCE_BIAS_MIN  # Clamped to -0.15


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCache:
    def test_confidence_bias_cached(self, si, mock_client):
        """Second call for same symbol/strategy should use cache, not call Claude."""
        bias1 = si.get_confidence_bias("MES", "alpha_futures", 0.7)
        bias2 = si.get_confidence_bias("MES", "alpha_futures", 0.7)

        # Claude should only be called once
        assert mock_client.chat_json.call_count == 1
        assert bias1.adjustment == bias2.adjustment

    def test_different_symbols_not_cached(self, si, mock_client):
        """Different symbols should each call Claude."""
        si.get_confidence_bias("MES", "alpha_futures", 0.7)
        si.get_confidence_bias("MNQ", "alpha_futures", 0.7)

        assert mock_client.chat_json.call_count == 2

    def test_regime_profile_cached(self, runtime_dir):
        """Regime profile should be cached for 15 minutes."""
        client = MagicMock()
        client.chat_json.return_value = {
            "profile": "conservative",
            "reasoning": "VIX elevated",
        }

        si = StrategyIntelligence(client, runtime_dir)
        p1 = si.get_regime_profile("alpha_futures")
        p2 = si.get_regime_profile("alpha_futures")

        assert p1 == "conservative"
        assert p2 == "conservative"
        assert client.chat_json.call_count == 1

    def test_cache_persisted_to_disk(self, si, runtime_dir):
        """Cache should be written to runtime/strategy_intelligence_cache.json."""
        si.get_confidence_bias("MES", "alpha_futures", 0.7)

        cache_path = runtime_dir / "strategy_intelligence_cache.json"
        assert cache_path.exists()

        data = json.loads(cache_path.read_text())
        assert "confidence" in data
        assert "MES|alpha_futures" in data["confidence"]


# ---------------------------------------------------------------------------
# Universe filter
# ---------------------------------------------------------------------------


class TestUniverseFilter:
    def test_universe_filter_basic(self, runtime_dir):
        client = MagicMock()
        client.chat_json.return_value = {
            "avoid_symbols": ["MES"],
            "prefer_symbols": ["MNQ", "MCL"],
            "reasons": {"MES": "Earnings within 24h"},
        }

        with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
            si = StrategyIntelligence(client, runtime_dir)
            uf = si.get_universe_filter("alpha_futures", ["MES", "MNQ", "MCL", "MGC"])

        assert "MES" in uf.avoid_symbols
        assert "MNQ" in uf.prefer_symbols
        assert "MES" in uf.reasons

    def test_universe_filter_strips_non_proposed(self, runtime_dir):
        """Symbols not in proposed list should be filtered out."""
        client = MagicMock()
        client.chat_json.return_value = {
            "avoid_symbols": ["MES", "AAPL"],  # AAPL not proposed
            "prefer_symbols": ["MNQ"],
            "reasons": {},
        }

        with patch("chad.intel.strategy_intelligence.StrategyIntelligence._load_news_headlines", return_value=[]):
            si = StrategyIntelligence(client, runtime_dir)
            uf = si.get_universe_filter("alpha_futures", ["MES", "MNQ"])

        assert "AAPL" not in uf.avoid_symbols
        assert "MES" in uf.avoid_symbols


# ---------------------------------------------------------------------------
# Regime profile
# ---------------------------------------------------------------------------


class TestRegimeProfile:
    def test_regime_profile_normal(self, runtime_dir):
        client = MagicMock()
        client.chat_json.return_value = {
            "profile": "normal",
            "reasoning": "VIX low, no crisis",
        }

        si = StrategyIntelligence(client, runtime_dir)
        assert si.get_regime_profile("alpha_futures") == "normal"

    def test_regime_profile_conservative(self, runtime_dir):
        client = MagicMock()
        client.chat_json.return_value = {
            "profile": "conservative",
            "reasoning": "VIX > 25",
        }

        si = StrategyIntelligence(client, runtime_dir)
        assert si.get_regime_profile("alpha_futures") == "conservative"

    def test_regime_profile_invalid_returns_normal(self, runtime_dir):
        """Invalid profile string should default to 'normal'."""
        client = MagicMock()
        client.chat_json.return_value = {
            "profile": "aggressive",  # Not a valid option
            "reasoning": "test",
        }

        si = StrategyIntelligence(client, runtime_dir)
        assert si.get_regime_profile("alpha_futures") == "normal"


# ---------------------------------------------------------------------------
# Neutral defaults
# ---------------------------------------------------------------------------


class TestNeutralDefaults:
    def test_neutral_bias(self):
        bias = _neutral_bias("MES", "alpha_futures")
        assert bias.adjustment == 0.0
        assert bias.symbol == "MES"

    def test_neutral_universe_filter(self):
        uf = _neutral_universe_filter()
        assert uf.avoid_symbols == []
        assert uf.prefer_symbols == []


# ---------------------------------------------------------------------------
# Macro / event placeholder containment (Batch 7)
# ---------------------------------------------------------------------------


def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stamp(payload, age_seconds=0, ttl=1800):
    payload = dict(payload)
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_seconds)).isoformat().replace(
        "+00:00", "Z"
    )
    payload["ts_utc"] = ts
    payload["ttl_seconds"] = ttl
    return payload


def test_strategy_intelligence_uses_runtime_vix_when_available(tmp_path):
    """Real VIX from price_cache.json should be the source — never 'unknown'."""
    rd = tmp_path / "runtime"
    rd.mkdir()
    (rd / "macro_state.json").write_text(json.dumps(_stamp({
        "risk_label": "risk_on",
        "schema_version": "macro_state.v1",
        "source": {"provider": "FredYieldProvider"},
    })))
    (rd / "price_cache.json").write_text(json.dumps(_stamp({
        "prices": {"VIX": 16.99, "SPY": 717.58},
    }, ttl=300)))
    (rd / "event_risk.json").write_text(json.dumps(_stamp({
        "severity": "low",
        "elevated_risk": False,
        "next_event": {
            "name": "FOMC Rate Decision",
            "ts_utc": _now_iso(),
            "hours_until": 69.0,
            "severity": "high",
            "source": "operator_calendar",
        },
        "source": {"provider": "EconomicCalendarRiskProvider"},
    })))

    si = StrategyIntelligence(MagicMock(), rd)
    ctx = si._load_market_context()

    assert ctx["vix"]["vix"] == 16.99
    assert ctx["vix"]["vix_source"] == "VIX"
    assert ctx["vix"]["provider_status"] == "real"
    # Sanity: nowhere in context is the placeholder string 'unknown' substituted in.
    assert ctx["vix"]["vix"] != "unknown"


def test_strategy_intelligence_marks_macro_unavailable_when_stale(tmp_path):
    """Macro/VIX past ts_utc+ttl must be marked stale — not silently passed as fresh."""
    rd = tmp_path / "runtime"
    rd.mkdir()
    # Stale: age 7200s with ttl 1800s
    (rd / "macro_state.json").write_text(json.dumps(_stamp({
        "risk_label": "risk_on",
    }, age_seconds=7200, ttl=1800)))
    (rd / "price_cache.json").write_text(json.dumps(_stamp({
        "prices": {"VIX": 18.0},
    }, age_seconds=600, ttl=300)))
    # No event_risk file at all
    si = StrategyIntelligence(MagicMock(), rd)
    ctx = si._load_market_context()

    assert ctx["macro_meta"]["provider_status"] == "stale"
    assert ctx["vix"]["provider_status"] == "stale"
    assert ctx["event_risk"]["provider_status"] == "unavailable"


def test_event_context_uses_structured_event_risk_next_event(tmp_path):
    """Structured next_event from EconomicCalendarRiskProvider must surface intact."""
    rd = tmp_path / "runtime"
    rd.mkdir()
    (rd / "event_risk.json").write_text(json.dumps(_stamp({
        "severity": "low",
        "elevated_risk": False,
        "next_event": {
            "name": "FOMC Rate Decision",
            "ts_utc": "2026-05-07T18:00:00Z",
            "hours_until": 69.48,
            "severity": "high",
            "source": "operator_calendar",
        },
        "source": {"provider": "EconomicCalendarRiskProvider"},
    })))
    si = StrategyIntelligence(MagicMock(), rd)
    ev = si._load_event_context()
    assert ev["provider_status"] == "real"
    assert ev["next_event"]["name"] == "FOMC Rate Decision"
    assert ev["next_event"]["severity"] == "high"
    assert ev["source_provider"] == "EconomicCalendarRiskProvider"


def test_event_context_does_not_treat_time_of_day_placeholder_as_real(tmp_path):
    """MarketHoursRiskProvider time-of-day stub must NOT pass as real event data."""
    rd = tmp_path / "runtime"
    rd.mkdir()
    (rd / "event_risk.json").write_text(json.dumps(_stamp({
        "severity": "medium",
        "elevated_risk": False,
        "next_event": None,
        "windows": [
            {"label": "US_MARKET_OPEN", "severity": "medium", "start_utc": "x", "end_utc": "y"},
        ],
        "source": {
            "provider": "MarketHoursRiskProvider",
            "provider_status": "placeholder_or_unavailable",
            "kind": "time_of_day_placeholder",
        },
    })))
    si = StrategyIntelligence(MagicMock(), rd)
    ev = si._load_event_context()
    assert ev["provider_status"] == "placeholder_or_unavailable"
    assert ev["next_event"] is None


def test_macro_event_context_degrades_conservatively_when_missing(tmp_path):
    """No runtime files at all → all sections marked unavailable, prompt steers to conservative."""
    rd = tmp_path / "runtime"
    rd.mkdir()
    si = StrategyIntelligence(MagicMock(), rd)
    ctx = si._load_market_context()
    assert ctx["macro_meta"]["provider_status"] == "unavailable"
    assert ctx["vix"]["provider_status"] in ("unavailable", "unavailable_no_symbol")
    assert ctx["event_risk"]["provider_status"] == "unavailable"

    # Verify the regime prompt path declares unavailable rather than treating
    # missing data as neutral. We exercise it by having Claude return 'normal'
    # and confirming the prompt itself flagged unavailability — the regime
    # rule says "default to conservative if any input is missing".
    client = MagicMock()
    captured = {}
    def _capture(prompt, system=None, task_type=None):
        captured["prompt"] = prompt
        return {"profile": "conservative", "reasoning": "All inputs unavailable — degrade conservatively"}
    client.chat_json.side_effect = _capture
    si2 = StrategyIntelligence(client, rd)
    profile = si2.get_regime_profile("alpha")
    assert profile == "conservative"
    assert "unavailable" in captured["prompt"]
    assert "degrade conservatively" in captured["prompt"].lower()

"""
Tests for chad.intel.claude_client — ClaudeClient

All Anthropic API calls are mocked. No real API calls are made.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from chad.intel.claude_client import (
    ClaudeClient,
    ClaudeConfig,
    ClaudeClientError,
    ClaudeConfigError,
    ClaudeRateLimitError,
    ClaudeAPIError,
    ConfigurationError,
    TIER_MODELS,
    _load_claude_env_file,
    _approximate_tokens,
    _RateLimiterState,
)


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------


class TestKeyLoading:
    def test_load_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123")
        key = ClaudeClient.load_key()
        assert key == "sk-ant-test-key-123"

    def test_load_key_missing_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("chad.intel.claude_client._load_claude_env_file"):
            with pytest.raises(ClaudeConfigError, match="ANTHROPIC_API_KEY"):
                ClaudeClient.load_key()

    def test_load_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / "claude.env"
        env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _load_claude_env_file(str(env_file))
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-from-file"
        # Cleanup
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_load_env_file_missing_is_noop(self):
        _load_claude_env_file("/nonexistent/path/claude.env")


# ---------------------------------------------------------------------------
# Tier routing
# ---------------------------------------------------------------------------


class TestTierRouting:
    def test_routine_tier(self):
        assert ClaudeClient.model_for_tier("routine") == "claude-haiku-4-5-20251001"

    def test_standard_tier(self):
        assert ClaudeClient.model_for_tier("standard") == "claude-haiku-4-5-20251001"

    def test_complex_tier(self):
        assert ClaudeClient.model_for_tier("complex") == "claude-sonnet-4-6"

    def test_unknown_tier_defaults_to_standard(self):
        assert ClaudeClient.model_for_tier("unknown") == TIER_MODELS["standard"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def _make_client(self):
        config = ClaudeConfig(api_key="sk-ant-test", max_requests_per_min=10)
        with patch("chad.intel.claude_client.ClaudeClient._build_anthropic_client", return_value=None):
            return ClaudeClient(config=config)

    def test_rate_limit_per_minute(self):
        client = self._make_client()
        # Exhaust 10 requests
        for _ in range(10):
            client._enforce_rate_limits(estimated_tokens=10)
        # 11th should fail
        with pytest.raises(ClaudeRateLimitError, match="requests per minute"):
            client._enforce_rate_limits(estimated_tokens=10)

    def test_rate_limit_daily_tokens(self):
        client = self._make_client()
        # Use up daily cap in one shot
        client._enforce_rate_limits(estimated_tokens=100_000)
        # Next request should fail
        with pytest.raises(ClaudeRateLimitError, match="daily token cap"):
            client._enforce_rate_limits(estimated_tokens=1)

    def test_rate_limit_resets_after_minute(self):
        client = self._make_client()
        for _ in range(10):
            client._enforce_rate_limits(estimated_tokens=10)
        # Simulate minute passing
        client._rate_state.minute_window_start = time.monotonic() - 61
        # Should succeed now
        client._enforce_rate_limits(estimated_tokens=10)


# ---------------------------------------------------------------------------
# chat_json
# ---------------------------------------------------------------------------


class TestChatJson:
    def _make_client_with_mock(self):
        config = ClaudeConfig(api_key="sk-ant-test")

        mock_anthropic = MagicMock()
        mock_response = MagicMock()

        block = MagicMock()
        block.text = '{"regime": "neutral", "vix": 23.5}'
        mock_response.content = [block]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_anthropic.messages.create.return_value = mock_response

        with patch("chad.intel.claude_client.ClaudeClient._build_anthropic_client", return_value=mock_anthropic):
            client = ClaudeClient(config=config)

        return client, mock_anthropic

    def test_chat_json_returns_dict(self):
        client, mock = self._make_client_with_mock()
        result = client.chat_json("What is VIX?", task_type="routine")
        assert isinstance(result, dict)
        assert result["regime"] == "neutral"

    def test_chat_json_gpt_compatible_kwargs(self):
        client, mock = self._make_client_with_mock()
        result = client.chat_json(
            system_prompt="You are a test.",
            user_prompt="What is VIX?",
            temperature=0.1,
            max_output_tokens=1024,
            extra_context={"role": "test"},
        )
        assert isinstance(result, dict)
        assert result["regime"] == "neutral"

    def test_chat_json_empty_prompt_raises(self):
        client, _ = self._make_client_with_mock()
        with pytest.raises(ClaudeClientError, match="prompt must not be empty"):
            client.chat_json("")

    def test_chat_json_strips_markdown_fences(self):
        config = ClaudeConfig(api_key="sk-ant-test")
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        block = MagicMock()
        block.text = '```json\n{"status": "ok"}\n```'
        mock_response.content = [block]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 20
        mock_anthropic.messages.create.return_value = mock_response

        with patch("chad.intel.claude_client.ClaudeClient._build_anthropic_client", return_value=mock_anthropic):
            client = ClaudeClient(config=config)

        result = client.chat_json("test")
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_returns_structured_error_when_all_unavailable(self):
        config = ClaudeConfig(api_key="sk-ant-test")
        with patch("chad.intel.claude_client.ClaudeClient._build_anthropic_client", return_value=None):
            client = ClaudeClient(config=config)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            result = client.chat_json("test prompt")
            assert result["fallback"] is True
            assert "error" in result


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_synthesize_returns_string(self):
        config = ClaudeConfig(api_key="sk-ant-test")
        mock_anthropic = MagicMock()
        mock_response = MagicMock()
        block = MagicMock()
        block.text = "Buy small position in MES. VIX is moderate."
        mock_response.content = [block]
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 30
        mock_anthropic.messages.create.return_value = mock_response

        with patch("chad.intel.claude_client.ClaudeClient._build_anthropic_client", return_value=mock_anthropic):
            client = ClaudeClient(config=config)

        result = client.synthesize(
            engine_outputs={"research": {"ok": True, "data": {}}},
            user_request="Should I buy MES?",
            runtime_context={"vix": 23.5},
            coaching_profile="Be concise",
        )
        assert isinstance(result, str)
        assert "MES" in result

    def test_synthesize_returns_fallback_on_error(self):
        config = ClaudeConfig(api_key="sk-ant-test")
        with patch("chad.intel.claude_client.ClaudeClient._build_anthropic_client", return_value=None):
            client = ClaudeClient(config=config)

        result = client.synthesize(
            engine_outputs={},
            user_request="test",
            runtime_context={},
            coaching_profile="",
        )
        assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# Token approximation
# ---------------------------------------------------------------------------


class TestTokenApprox:
    def test_approximate_tokens(self):
        assert _approximate_tokens("hello world") >= 1
        assert _approximate_tokens("a" * 400) == 100
        assert _approximate_tokens("") == 1

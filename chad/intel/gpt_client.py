from __future__ import annotations

"""
CHAD GPT Client (Phase 10 – Global Intelligence Layer)

This module provides a safe, rate-limited wrapper around the OpenAI Chat
Completions API for CHAD's "Global Intelligence Layer".

Design goals:
- NEVER execute trades or touch broker / risk state directly.
- Only produce advisory, structured outputs for higher-level modules.
- Enforce strict timeouts, rate limits, and logging for auditability.

Config loading:
- Primary source: /etc/chad/openai.env (simple KEY=VALUE lines).
- Secondary: process environment (os.environ), allowing overrides.
- Backwards compatibility for older variable names used in earlier phases.

Expected environment variables (after normalisation):
    OPENAI_API_KEY               (required)
    OPENAI_MODEL_NAME            (default: "gpt-4.1-mini")
    OPENAI_REQUEST_TIMEOUT_SEC   (default: 8)
    OPENAI_MAX_REQUESTS_PER_MIN  (default: 10)
    OPENAI_DAILY_TOKEN_CAP       (default: 50000)

Usage:
- A long-lived process should create ONE GPTClient instance and reuse it.
- Higher-level modules (research, lessons, risk explainer) call:

    client.chat_json(
        system_prompt=...,
        user_prompt=...,
        temperature=0.1,
        max_output_tokens=1024,
        extra_context={"role": "research_scenario"},
    )

  and receive a dict parsed from the model's JSON response.

Safety notes:
- Rate limiting is in-process only; it guards against runaway loops.
- All calls are logged (metadata only, no raw secrets) to logs/gpt/gpt_client.log.
- No trading, risk, or execution modules are imported from here.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests import Response, Session

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_DIR_DEFAULT = "/home/ubuntu/CHAD FINALE/logs/gpt"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_ENV_PATH = "/etc/chad/openai.env"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GPTClientError(Exception):
    """Base error type for GPT client-related failures."""


class GPTConfigError(GPTClientError):
    """Raised when required configuration is missing or invalid."""


class GPTRateLimitError(GPTClientError):
    """Raised when local rate limits would be exceeded."""


class GPTAPIError(GPTClientError):
    """Raised when the OpenAI API returns an error or invalid response."""


# ---------------------------------------------------------------------------
# Configuration & Rate Limiting State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPTConfig:
    api_key: str
    model_name: str
    request_timeout_sec: float
    max_requests_per_min: int
    daily_token_cap: int


@dataclass
class _RateLimiterState:
    minute_window_start: float
    minute_requests: int
    day_window_start: float
    day_tokens_used: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_openai_env_file(path: str) -> None:
    """
    Best-effort loader for an .env-style file.

    Reads KEY=VALUE lines and populates os.environ for any missing keys.
    Ignores comments and malformed lines.

    This function is intentionally minimal: it only exists to ensure that
    /etc/chad/openai.env is visible to GPTClient even if systemd does not
    explicitly ExportEnvironment for those variables.
    """
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                if key not in os.environ and value:
                    os.environ[key] = value
    except Exception as exc:  # noqa: BLE001
        logger = logging.getLogger("chad.gpt_client")
        logger.exception("Failed to load OpenAI env file %s: %s", path, exc)


def _get_logger() -> logging.Logger:
    """
    Create or return a logger for GPT client operations.

    - Logs to stdout via the root logger (already configured by CHAD),
      and additionally to a dedicated rotating file in LOG_DIR_DEFAULT.
    - Ensures the log directory exists.
    """
    logger = logging.getLogger("chad.gpt_client")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Ensure directory exists
    log_dir = Path(LOG_DIR_DEFAULT)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "gpt_client.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Do not duplicate messages to root unless it's configured differently.
    logger.propagate = True
    logger.info("GPT client logging initialised at %s", log_path)
    return logger


def _extract_text_and_usage(resp_json: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Extract the assistant message content text and optional usage dictionary
    from a Chat Completions API response.

    Raises:
        KeyError if the expected structure is not present.
    """
    choices = resp_json["choices"]
    if not isinstance(choices, list) or not choices:
        raise KeyError("choices[0].message.content")

    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise KeyError("choices[0].message.content")

    usage = resp_json.get("usage")
    if usage is not None and not isinstance(usage, dict):
        usage = None

    return content, usage


def _approximate_tokens(*texts: str) -> int:
    """
    Very rough token estimator: approximate 4 characters per token.

    This is intentionally conservative and used ONLY for local rate limiting.
    """
    total_chars = sum(len(t) for t in texts if t)
    return max(1, total_chars // 4)


# ---------------------------------------------------------------------------
# GPT Client
# ---------------------------------------------------------------------------


class GPTClient:
    """
    Safe, rate-limited wrapper around OpenAI Chat Completions for CHAD.

    - Reads config from /etc/chad/openai.env and environment variables.
    - Enforces per-minute and per-day quotas.
    - Uses synchronous HTTP (requests) with timeouts and retries.
    - Provides a JSON-oriented "chat_json" helper for structured responses.
    """

    def __init__(
        self,
        config: Optional[GPTConfig] = None,
        session: Optional[Session] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or self._load_config_from_env()
        self._session: Session = session or requests.Session()
        self._logger = logger or _get_logger()

        now = time.monotonic()
        self._rate_state = _RateLimiterState(
            minute_window_start=now,
            minute_requests=0,
            day_window_start=now,
            day_tokens_used=0,
        )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_config_from_env() -> GPTConfig:
        """
        Load GPT configuration from environment variables.

        Steps:
        1. Best-effort load from /etc/chad/openai.env into os.environ.
        2. Normalise older variable names:
           - OPENAI_MODEL      -> OPENAI_MODEL_NAME
           - OPENAI_TIMEOUT    -> OPENAI_REQUEST_TIMEOUT_SEC
        3. Validate presence of OPENAI_API_KEY.
        4. Parse numeric limits with strong validation.
        """
        # 1) Load env file if OPENAI_API_KEY not already present
        if not os.environ.get("OPENAI_API_KEY"):
            _load_openai_env_file(OPENAI_ENV_PATH)

        # 2) Normalise older names
        if "OPENAI_MODEL_NAME" not in os.environ and os.environ.get("OPENAI_MODEL"):
            os.environ["OPENAI_MODEL_NAME"] = os.environ["OPENAI_MODEL"]

        if "OPENAI_REQUEST_TIMEOUT_SEC" not in os.environ and os.environ.get("OPENAI_TIMEOUT"):
            os.environ["OPENAI_REQUEST_TIMEOUT_SEC"] = os.environ["OPENAI_TIMEOUT"]

        # 3) Required API key
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise GPTConfigError("OPENAI_API_KEY is not set or empty")

        # 4) Optional model
        model_name = os.environ.get("OPENAI_MODEL_NAME", "").strip() or "gpt-4.1-mini"

        def _int_env(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if not raw:
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise GPTConfigError(f"{name} must be an integer, got {raw!r}") from exc
            if value <= 0:
                raise GPTConfigError(f"{name} must be positive, got {value}")
            return value

        def _float_env(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if not raw:
                return default
            try:
                value = float(raw)
            except ValueError as exc:
                raise GPTConfigError(f"{name} must be a float, got {raw!r}") from exc
            if value <= 0.0:
                raise GPTConfigError(f"{name} must be positive, got {value}")
            return value

        timeout = _float_env("OPENAI_REQUEST_TIMEOUT_SEC", 8.0)
        max_per_min = _int_env("OPENAI_MAX_REQUESTS_PER_MIN", 10)
        daily_cap = _int_env("OPENAI_DAILY_TOKEN_CAP", 50000)

        return GPTConfig(
            api_key=api_key,
            model_name=model_name,
            request_timeout_sec=timeout,
            max_requests_per_min=max_per_min,
            daily_token_cap=daily_cap,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> GPTConfig:
        return self._config

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call the model with a system + user prompt and expect a JSON object.

        The model is instructed to produce a single JSON object in the response.
        We parse the first choice as JSON and return it.

        Raises:
            GPTRateLimitError
            GPTAPIError
            GPTClientError (including GPTConfigError)
        """
        if not system_prompt.strip():
            raise GPTClientError("system_prompt must not be empty")
        if not user_prompt.strip():
            raise GPTClientError("user_prompt must not be empty")
        if max_output_tokens <= 0:
            raise GPTClientError("max_output_tokens must be positive")
        if not (0.0 <= temperature <= 2.0):
            raise GPTClientError("temperature must be between 0.0 and 2.0")

        payload = self._build_payload(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        estimated_tokens = _approximate_tokens(system_prompt, user_prompt)

        with self._lock:
            self._enforce_rate_limits(estimated_tokens=estimated_tokens)

        t0 = time.monotonic()
        resp_json = self._post_chat_with_retries(payload, extra_context=extra_context)
        elapsed = time.monotonic() - t0

        role = extra_context.get("role") if extra_context else "unknown"
        self._logger.info(
            "GPT chat_json call completed in %.3fs (model=%s, role=%s)",
            elapsed,
            self._config.model_name,
            role,
        )

        try:
            content_text, usage = _extract_text_and_usage(resp_json)
        except KeyError as exc:
            raise GPTAPIError(f"Unexpected API response structure: missing {exc}") from exc

        if usage is not None:
            with self._lock:
                self._update_token_usage(usage)

        try:
            parsed: Any = json.loads(content_text)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON is not an object")
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failed to parse JSON from GPT response: %s; raw=%r",
                exc,
                content_text[:512],
            )
            raise GPTAPIError("Failed to parse GPT JSON response") from exc

        return parsed

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        """
        Build the JSON payload for the Chat Completions API.

        We use a simple two-message exchange: system + user.
        """
        return {
            "model": self._config.model_name,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    def _post_chat_with_retries(
        self,
        payload: Dict[str, Any],
        *,
        extra_context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        POST the chat payload to the OpenAI API with basic retry logic.

        Retries on:
        - HTTP 429 (rate limit)
        - HTTP 500, 502, 503, 504 (transient server errors)
        """
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_retries:
            attempt += 1
            try:
                return self._post_chat_once(payload)
            except GPTAPIError as exc:
                # If error is due to HTTP status, we may or may not retry.
                last_error = exc
                if not self._is_retryable_api_error(exc):
                    raise
                backoff = min(2 ** attempt, 10) + 0.1 * attempt
                role = extra_context.get("role") if extra_context else "unknown"
                self._logger.warning(
                    "Retryable GPT API error (attempt %d/%d, role=%s): %s; backing off %.2fs",
                    attempt,
                    max_retries,
                    role,
                    exc,
                )
                time.sleep(backoff)
            except requests.RequestException as exc:
                last_error = exc
                backoff = min(2 ** attempt, 10) + 0.1 * attempt
                self._logger.warning(
                    "Network error in GPT client (attempt %d/%d): %s; backing off %.2fs",
                    attempt,
                    max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        # If we reach here, all retries have failed.
        if last_error:
            raise GPTAPIError(f"Failed to call OpenAI API after {max_retries} attempts") from last_error
        raise GPTAPIError("Failed to call OpenAI API: unknown error")

    def _post_chat_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single attempt at sending payload to OpenAI Chat Completions API.
        """
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp: Response = self._session.post(
                OPENAI_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=self._config.request_timeout_sec,
            )
        except requests.RequestException as exc:
            self._logger.exception("HTTP request to OpenAI failed: %s", exc)
            raise GPTAPIError(f"Network error: {exc}") from exc

        if resp.status_code != 200:
            # Try to parse OpenAI-style error
            try:
                err_json = resp.json()
            except Exception:  # noqa: BLE001
                err_json = {}
            message = (
                err_json.get("error", {}).get("message")
                if isinstance(err_json, dict)
                else None
            )
            raise GPTAPIError(
                f"OpenAI API error (status={resp.status_code}): {message or resp.text[:256]}"
            )

        try:
            resp_json: Dict[str, Any] = resp.json()
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("Failed to decode JSON from OpenAI response: %s", exc)
            raise GPTAPIError("Failed to decode JSON from OpenAI response") from exc

        return resp_json

    @staticmethod
    def _is_retryable_api_error(exc: GPTAPIError) -> bool:
        """
        Decide whether a GPTAPIError is likely retryable based on its message.
        """
        msg = str(exc).lower()
        retryable_fragments = [
            "rate limit",
            "429",
            "temporarily unavailable",
            "timeout",
            "try again",
            "overloaded",
            "server error",
            "status=500",
            "status=502",
            "status=503",
            "status=504",
        ]
        return any(fragment in msg for fragment in retryable_fragments)

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #

    def _enforce_rate_limits(self, *, estimated_tokens: int) -> None:
        """
        Enforce per-minute and per-day limits.

        Uses monotonic time and a simple sliding window per minute and per day.
        Raises GPTRateLimitError if limits would be exceeded.
        """
        now = time.monotonic()
        state = self._rate_state

        # Per-minute window
        minute_elapsed = now - state.minute_window_start
        if minute_elapsed >= 60.0:
            state.minute_window_start = now
            state.minute_requests = 0

        # Per-day window (24h rolling)
        day_elapsed = now - state.day_window_start
        if day_elapsed >= 24 * 3600.0:
            state.day_window_start = now
            state.day_tokens_used = 0

        if state.minute_requests + 1 > self._config.max_requests_per_min:
            raise GPTRateLimitError(
                f"Local GPT rate limit exceeded: {self._config.max_requests_per_min} requests per minute"
            )

        if state.day_tokens_used + estimated_tokens > self._config.daily_token_cap:
            raise GPTRateLimitError(
                f"Local GPT daily token cap exceeded: {self._config.daily_token_cap} tokens"
            )

        # If we reach here, we can proceed and update counters.
        state.minute_requests += 1
        state.day_tokens_used += estimated_tokens

    def _update_token_usage(self, usage: Dict[str, Any]) -> None:
        """
        Update daily token usage based on the 'usage' field returned by OpenAI.

        We prefer 'total_tokens' if present, otherwise fall back to sum of
        prompt_tokens + completion_tokens, if available.
        """
        total_tokens = usage.get("total_tokens")
        if isinstance(total_tokens, int) and total_tokens > 0:
            tokens = total_tokens
        else:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            tokens = 0
            if isinstance(prompt_tokens, int) and prompt_tokens > 0:
                tokens += prompt_tokens
            if isinstance(completion_tokens, int) and completion_tokens > 0:
                tokens += completion_tokens
            if tokens <= 0:
                # Fallback: do not adjust counters if usage is malformed.
                return

        state = self._rate_state
        state.day_tokens_used += tokens


# ---------------------------------------------------------------------------
# Convenience utility for diagnostics
# ---------------------------------------------------------------------------


def test_connection() -> Dict[str, Any]:
    """
    Lightweight connectivity + configuration test for GPTClient.

    - Attempts to instantiate a GPTClient.
    - Returns a dict describing configuration and whether API key is present.
    - Does NOT make a network call, so it is safe to run in diagnostics.
    """
    info: Dict[str, Any] = {
        "config_loaded": False,
        "api_key_present": False,
        "model_name": None,
        "error": None,
    }
    try:
        client = GPTClient()
        info["config_loaded"] = True
        info["api_key_present"] = bool(client.config.api_key)
        info["model_name"] = client.config.model_name
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info

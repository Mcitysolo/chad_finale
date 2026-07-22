from __future__ import annotations

"""
CHAD Phase 9 — Claude AI Client (Global Intelligence Layer)

Unified client for Anthropic Claude API with three-tier model routing,
rate limiting, usage logging, retry with backoff, and OpenAI fallback.

Design goals:
- NEVER execute trades or touch broker / risk state directly.
- Only produce advisory, structured outputs for higher-level modules.
- Enforce strict timeouts, rate limits, and logging for auditability.
- Implements both chat_json() (GPTClient-compatible) and synthesize()
  (LLMClient Protocol from advisory_engine.py).

Tier routing:
    TIER 0 — "routine":  OllamaClient (phi3:mini, local, free) if available
    TIER 1 — "routine":  claude-haiku-4-5-20251001  (fast, cheap — fallback)
    TIER 2 — "standard": claude-haiku-4-5-20251001  (default for most advisory)
    TIER 3 — "complex":  claude-sonnet-4-6           (full portfolio analysis)

Config loading:
    Primary source: /etc/chad/claude.env (simple KEY=VALUE lines).
    Secondary: process environment (os.environ).

Expected environment variables (after normalisation):
    ANTHROPIC_API_KEY  (required)
"""

import json
import logging
import logging.handlers
import os
import threading
import time

import psutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLAUDE_ENV_PATH = "/etc/chad/claude.env"
OPENAI_ENV_PATH = "/etc/chad/openai.env"
LOG_DIR_DEFAULT = "/home/ubuntu/chad_finale/logs/claude"
USAGE_FILE = "/home/ubuntu/chad_finale/runtime/claude_usage.json"

# GAP-037/038: model IDs are centralised in config/llm_models.json (read via
# chad.intel.llm_models). The accessors fall back to the exact pre-IR1 literals
# when the config is missing/corrupt, so behaviour is unchanged on load failure.
from chad.intel import llm_models as _llm_models  # noqa: E402

TIER_MODELS = _llm_models.anthropic_tiers()

DEFAULT_REQUEST_TIMEOUT_SEC = 15.0
DEFAULT_MAX_REQUESTS_PER_MIN = int(os.environ.get("CHAD_MAX_REQUESTS_PER_MIN", "30"))
DEFAULT_DAILY_TOKEN_CAP: int = int(
    os.environ.get("CHAD_ADVISORY_TOKEN_CAP", "100000")
)
DEFAULT_DAILY_DOLLAR_CAP: float = float(
    os.environ.get("CHAD_ADVISORY_DAILY_DOLLAR_CAP", "5.0")
)
OLLAMA_MIN_RAM_MB: int = int(
    os.environ.get("CHAD_OLLAMA_MIN_RAM_MB", "1500")
)

_MODEL_COST_PER_1K_TOKENS: dict = _llm_models.anthropic_cost_per_1k()
_DEFAULT_COST_PER_1K = 0.003  # conservative default

MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = {429, 529, 500, 502, 503, 504}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ClaudeClientError(Exception):
    """Base error type for Claude client failures."""


class ClaudeConfigError(ClaudeClientError):
    """Raised when required configuration is missing or invalid."""


class ClaudeRateLimitError(ClaudeClientError):
    """Raised when local rate limits would be exceeded."""


class ClaudeAPIError(ClaudeClientError):
    """Raised when the Claude API returns an error or invalid response."""


class ConfigurationError(ClaudeClientError):
    """Raised when no AI provider is available."""


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama inference is skipped due to system constraints."""
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaudeConfig:
    api_key: str
    request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC
    max_requests_per_min: int = DEFAULT_MAX_REQUESTS_PER_MIN
    daily_token_cap: int = DEFAULT_DAILY_TOKEN_CAP


@dataclass
class _RateLimiterState:
    minute_window_start: float = 0.0
    minute_requests: int = 0
    day_window_start: float = 0.0
    day_tokens_used: int = 0
    day_dollars_used: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_claude_env_file(path: str = CLAUDE_ENV_PATH) -> None:
    """Best-effort loader for .env-style file into os.environ."""
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
    except Exception:
        pass


def _openai_fallback_enabled() -> bool:
    """Kill-switch for the OpenAI advisory fallback (default ON).

    Set CHAD_INTEL_OPENAI_FALLBACK=0 (or false/no) to disable — reverts the
    client to the pre-IR1 behaviour where /etc/chad/openai.env is not loaded
    and the OpenAI fallback cannot fire.
    """
    return os.environ.get("CHAD_INTEL_OPENAI_FALLBACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _load_openai_env_file(path: str = OPENAI_ENV_PATH) -> None:
    """Load /etc/chad/openai.env so the OpenAI advisory fallback has its key.

    IR1 root-cause fix: the refresh service unit has no EnvironmentFile and the
    client previously loaded only claude.env, so OPENAI_API_KEY was never in the
    process and _try_openai_fallback bailed at the key check — leaving zero
    working advisory tiers once Anthropic credits were exhausted. Gated by the
    CHAD_INTEL_OPENAI_FALLBACK kill-switch.
    """
    if not _openai_fallback_enabled():
        return
    _load_claude_env_file(path)


def _get_logger() -> logging.Logger:
    """Create or return a logger for Claude client operations."""
    logger = logging.getLogger("chad.claude_client")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir = Path(LOG_DIR_DEFAULT)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "claude_client.log"

    # W3B-11: plain FileHandler grew unbounded (12.4MB and active at the
    # 2026-07-22 inventory). Same 10MB x 5 bound as the telegram_bot
    # precedent (chad/utils/telegram_bot.py).
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = True
    logger.info("Claude client logging initialised at %s", log_path)
    return logger


def _approximate_tokens(*texts: str) -> int:
    """Rough token estimator: ~4 chars per token. For local rate limiting only."""
    total_chars = sum(len(t) for t in texts if t)
    return max(1, total_chars // 4)


def _write_usage_log(entry: Dict[str, Any]) -> None:
    """Append a usage entry to runtime/claude_usage.json."""
    usage_path = Path(USAGE_FILE)
    usage_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if usage_path.exists():
            data = json.loads(usage_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {"entries": []}
        else:
            data = {"entries": []}

        entries = data.get("entries", [])
        if not isinstance(entries, list):
            entries = []

        entries.append(entry)

        # Keep last 1000 entries
        if len(entries) > 1000:
            entries = entries[-1000:]

        data["entries"] = entries
        data["last_updated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        tmp = usage_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(usage_path)
    except Exception:
        pass


def _write_call_log(
    *,
    model: str,
    task_type: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Write per-call log to logs/claude/ directory."""
    log_dir = Path(LOG_DIR_DEFAULT)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    date_str = ts.strftime("%Y%m%d")
    log_path = log_dir / f"calls_{date_str}.ndjson"

    cost_per_1k = _MODEL_COST_PER_1K_TOKENS.get(model, _DEFAULT_COST_PER_1K)
    estimated_cost = (input_tokens + output_tokens) / 1000.0 * cost_per_1k

    entry = {
        "ts_utc": ts.isoformat().replace("+00:00", "Z"),
        "model": model,
        "task_type": task_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "duration_ms": duration_ms,
        "success": success,
    }
    if error:
        entry["error"] = error

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass

    # Also update usage tracking
    _write_usage_log(entry)


# ---------------------------------------------------------------------------
# Claude Client
# ---------------------------------------------------------------------------


class ClaudeClient:
    """
    Production-grade Claude API client for CHAD.

    - Three-tier model routing based on task complexity.
    - Rate limiting (10 req/min, 100K tokens/day).
    - 15s timeout per call.
    - 2 retries with exponential backoff on 529/overload.
    - Fallback to OpenAI if Claude unavailable.
    - Full usage logging.
    """

    def __init__(
        self,
        config: Optional[ClaudeConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or self._load_config_from_env()
        self._logger = logger or _get_logger()
        self._client = self._build_anthropic_client()

        now = time.monotonic()
        self._rate_state = _RateLimiterState(
            minute_window_start=now,
            minute_requests=0,
            day_window_start=now,
            day_tokens_used=0,
        )
        self._lock = threading.Lock()

        # IR1 provider-outcome telemetry (per process run). "real" = a genuine
        # LLM (Anthropic/OpenAI/Ollama) answered; "fallback" = every tier failed
        # and chat_json returned the neutral structured stub. Read by the
        # strategy-intelligence refresh to detect a fully-dead advisory tier and
        # raise the INTEL_REFRESH_FAILED marker instead of failing silently.
        self._provider_calls = {"real": 0, "fallback": 0}
        # IR1 billing short-circuit: once Anthropic reports credit exhaustion,
        # skip further doomed Anthropic calls this run (stops the ~14-calls/cycle
        # retry storm) and go straight to the OpenAI fallback.
        self._anthropic_unavailable = False
        self._last_error_class = ""

    def provider_call_stats(self) -> Dict[str, int]:
        """Return this run's provider-outcome counters (copy)."""
        return dict(self._provider_calls)

    @property
    def last_error_class(self) -> str:
        """Coarse classification of the most recent provider failure."""
        return self._last_error_class

    @property
    def anthropic_unavailable(self) -> bool:
        """True once Anthropic has reported credit exhaustion this run."""
        return self._anthropic_unavailable

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls) -> "ClaudeClient":
        """Factory classmethod: load key from /etc/chad/claude.env or environment."""
        return cls()

    @classmethod
    def load_key(cls) -> str:
        """Read ANTHROPIC_API_KEY from /etc/chad/claude.env or environment."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            _load_claude_env_file()
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ClaudeConfigError("ANTHROPIC_API_KEY is not set or empty")
        return key

    @staticmethod
    def _load_config_from_env() -> ClaudeConfig:
        """Load Claude configuration from environment variables."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            _load_claude_env_file()
        # IR1: also source /etc/chad/openai.env so the OpenAI advisory fallback
        # has its key (the systemd unit provides no EnvironmentFile for it).
        if not os.environ.get("OPENAI_API_KEY"):
            _load_openai_env_file()

        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ClaudeConfigError("ANTHROPIC_API_KEY is not set or empty")

        return ClaudeConfig(api_key=api_key)

    def _build_anthropic_client(self) -> Any:
        """Build the anthropic.Anthropic client instance."""
        try:
            import anthropic
            return anthropic.Anthropic(
                api_key=self._config.api_key,
                timeout=self._config.request_timeout_sec,
            )
        except Exception as exc:
            self._logger.warning("Failed to build Anthropic client: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Tier routing
    # ------------------------------------------------------------------ #

    @staticmethod
    def model_for_tier(task_type: str) -> str:
        """Return model ID for a given task type tier."""
        return TIER_MODELS.get(task_type, TIER_MODELS["standard"])

    # ------------------------------------------------------------------ #
    # Rate limiting
    # ------------------------------------------------------------------ #

    def _enforce_rate_limits(self, *, estimated_tokens: int) -> None:
        """Enforce per-minute and per-day limits. Raises ClaudeRateLimitError."""
        now = time.monotonic()
        state = self._rate_state

        minute_elapsed = now - state.minute_window_start
        if minute_elapsed >= 60.0:
            state.minute_window_start = now
            state.minute_requests = 0

        day_elapsed = now - state.day_window_start
        if day_elapsed >= 24 * 3600.0:
            state.day_window_start = now
            state.day_tokens_used = 0
            state.day_dollars_used = 0.0

        if state.minute_requests + 1 > self._config.max_requests_per_min:
            raise ClaudeRateLimitError(
                f"Local Claude rate limit exceeded: {self._config.max_requests_per_min} requests per minute"
            )

        if state.day_tokens_used + estimated_tokens > self._config.daily_token_cap:
            raise ClaudeRateLimitError(
                f"Local Claude daily token cap exceeded: {self._config.daily_token_cap} tokens"
            )

        state.minute_requests += 1
        state.day_tokens_used += estimated_tokens

    def _update_token_usage(self, input_tokens: int, output_tokens: int, model: str = "") -> None:
        """Update daily token and dollar usage from API response."""
        total = input_tokens + output_tokens
        if total > 0:
            self._rate_state.day_tokens_used += total
            cost_per_1k = _MODEL_COST_PER_1K_TOKENS.get(model, _DEFAULT_COST_PER_1K)
            cost = total / 1000.0 * cost_per_1k
            self._rate_state.day_dollars_used += cost
            if self._rate_state.day_dollars_used >= DEFAULT_DAILY_DOLLAR_CAP:
                raise ClaudeRateLimitError(
                    f"Daily dollar cap ${DEFAULT_DAILY_DOLLAR_CAP:.2f} exceeded "
                    f"(used ${self._rate_state.day_dollars_used:.3f})"
                )

    # ------------------------------------------------------------------ #
    # Core API call
    # ------------------------------------------------------------------ #

    def _call_claude(
        self,
        *,
        prompt: str,
        system: Optional[str] = None,
        task_type: str = "standard",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Tuple[str, int, int]:
        """
        Make a single Claude API call with retries.

        Returns: (response_text, input_tokens, output_tokens)
        Raises: ClaudeAPIError on failure after retries.
        """
        if self._client is None:
            raise ClaudeAPIError("Anthropic client not initialised")

        # IR1 billing short-circuit: a credit-exhaustion 400 is a persistent
        # account state, not a transient error — once seen, every further call
        # this run 400s identically. Skip them and fall through to OpenAI.
        if self._anthropic_unavailable:
            raise ClaudeAPIError("anthropic_credit_exhausted (short-circuit)")

        model = self.model_for_tier(task_type)
        messages = [{"role": "user", "content": prompt}]

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES
            try:
                response = self._client.messages.create(**kwargs)

                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)

                text = "\n".join(text_parts).strip()
                input_tok = getattr(response.usage, "input_tokens", 0)
                output_tok = getattr(response.usage, "output_tokens", 0)

                return text, input_tok, output_tok

            except Exception as exc:
                last_error = exc
                exc_str = str(exc).lower()

                # IR1: Anthropic credit exhaustion — a persistent, non-retryable
                # account state. Latch it so the rest of this run short-circuits
                # the Anthropic tier instead of re-issuing doomed 400 calls.
                if "credit balance is too low" in exc_str or (
                    "credit" in exc_str and "too low" in exc_str
                ):
                    self._anthropic_unavailable = True
                    self._last_error_class = "anthropic_credit_exhausted"
                    self._logger.warning(
                        "anthropic_credit_exhausted — latching Anthropic tier off "
                        "for this run; falling through to OpenAI fallback"
                    )
                    break

                # Check if retryable
                is_retryable = any(
                    frag in exc_str
                    for frag in ["529", "overload", "rate_limit", "timeout", "500", "502", "503", "504"]
                )

                if not is_retryable or attempt > MAX_RETRIES:
                    if not self._last_error_class:
                        self._last_error_class = "anthropic_api_error"
                    break

                backoff = min(2 ** attempt, 8) + 0.1 * attempt
                self._logger.warning(
                    "Retryable Claude API error (attempt %d/%d): %s; backing off %.2fs",
                    attempt, MAX_RETRIES + 1, exc, backoff,
                )
                time.sleep(backoff)

        raise ClaudeAPIError(f"Claude API failed after {MAX_RETRIES + 1} attempts: {last_error}") from last_error

    # ------------------------------------------------------------------ #
    # OpenAI fallback
    # ------------------------------------------------------------------ #

    def _try_openai_fallback(
        self,
        *,
        prompt: str,
        system: Optional[str] = None,
    ) -> Optional[str]:
        """Attempt to call OpenAI as fallback if key is available."""
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            from chad.intel.gpt_client import GPTClient
            client = GPTClient()
            result = client.chat_json(
                system_prompt=system or "You are a helpful financial analyst. Respond in JSON.",
                user_prompt=prompt,
                temperature=0.1,
                max_output_tokens=1024,
            )
            return json.dumps(result)
        except Exception as exc:
            self._logger.warning("OpenAI fallback also failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Public API: chat_json
    # ------------------------------------------------------------------ #

    def chat_json(
        self,
        prompt: Optional[str] = None,
        *,
        system: Optional[str] = None,
        task_type: str = "standard",
        schema: Optional[Dict[str, Any]] = None,
        # GPTClient-compatible kwargs for drop-in replacement
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call Claude with a prompt and expect a JSON object response.

        Compatible with both ClaudeClient and GPTClient call signatures:
        - ClaudeClient style: chat_json(prompt, system=..., task_type=...)
        - GPTClient style: chat_json(system_prompt=..., user_prompt=..., extra_context=...)

        Returns: Parsed JSON dict.
        Raises: ClaudeRateLimitError, ClaudeAPIError, ClaudeClientError.
        """
        # Normalize GPTClient-compatible kwargs
        if prompt is None and user_prompt is not None:
            prompt = user_prompt
        if system is None and system_prompt is not None:
            system = system_prompt

        if not prompt or not prompt.strip():
            raise ClaudeClientError("prompt must not be empty")

        # Tier 0: try Ollama for routine tasks; fall through to Haiku on failure
        if task_type == "routine":
            try:
                if _ollama_ram_ok():
                    _ollama = OllamaClient()
                    _olm_res = _ollama.chat_json(
                        prompt=prompt,
                        system=system,
                    )
                    self._provider_calls["real"] += 1  # IR1: Ollama answered
                    return _olm_res
            except Exception as _olm_err:
                logging.getLogger("chad.intel.claude_client").warning(
                    "ollama_tier0_failed fallback_to_haiku err=%s",
                    _olm_err,
                )
            # Fall through to Claude Haiku below

        # Build system prompt with JSON instruction
        sys_parts = []
        if system:
            sys_parts.append(system)
        sys_parts.append("Respond with a single valid JSON object. No markdown fences, no commentary outside the JSON.")
        if schema:
            sys_parts.append(f"Expected JSON schema:\n{json.dumps(schema, indent=2)}")
        full_system = "\n\n".join(sys_parts)

        estimated_tokens = _approximate_tokens(prompt, full_system)

        with self._lock:
            self._enforce_rate_limits(estimated_tokens=estimated_tokens)

        model = self.model_for_tier(task_type)
        t0 = time.monotonic()
        input_tok = 0
        output_tok = 0

        try:
            text, input_tok, output_tok = self._call_claude(
                prompt=prompt,
                system=full_system,
                task_type=task_type,
            )
            self._provider_calls["real"] += 1  # IR1: Anthropic answered
        except ClaudeAPIError:
            # Try OpenAI fallback
            fallback = self._try_openai_fallback(prompt=prompt, system=full_system)
            if fallback is not None:
                self._provider_calls["real"] += 1  # IR1: OpenAI answered
                text = fallback
                model = "openai_fallback"
                self._logger.info("Used OpenAI fallback for chat_json")
            else:
                # Return structured fallback — every tier failed.
                self._provider_calls["fallback"] += 1  # IR1: no real provider
                if not self._last_error_class:
                    self._last_error_class = "all_providers_unavailable"
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                _write_call_log(
                    model=model, task_type=task_type,
                    input_tokens=0, output_tokens=0,
                    duration_ms=elapsed_ms, success=False,
                    error="all_providers_unavailable",
                )
                return {
                    "error": "all_ai_providers_unavailable",
                    "fallback": True,
                    "message": "Claude and OpenAI both unavailable. Manual analysis required.",
                    "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        with self._lock:
            self._update_token_usage(input_tok, output_tok, model=model)

        _write_call_log(
            model=model, task_type=task_type,
            input_tokens=input_tok, output_tokens=output_tok,
            duration_ms=elapsed_ms, success=True,
        )

        self._logger.info(
            "chat_json completed in %dms (model=%s, task_type=%s, in=%d, out=%d)",
            elapsed_ms, model, task_type, input_tok, output_tok,
        )

        # Parse JSON from response
        try:
            # Strip markdown fences if present
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                # Remove first and last fence lines
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean = "\n".join(lines).strip()

            parsed = json.loads(clean)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON is not an object")
            return parsed
        except Exception as exc:
            self._logger.error(
                "Failed to parse JSON from Claude response: %s; raw=%r",
                exc, text[:512],
            )
            raise ClaudeAPIError("Failed to parse Claude JSON response") from exc

    # ------------------------------------------------------------------ #
    # Public API: synthesize (LLMClient Protocol)
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        engine_outputs: Dict[str, Any],
        user_request: Any,
        runtime_context: Dict[str, Any],
        coaching_profile: str,
    ) -> str:
        """
        Synthesize engine outputs into a user-facing advisory text.

        Matches the LLMClient Protocol signature from advisory_engine.py.
        This is the raw synthesis method — ClaudeAdvisoryClient wraps this
        with the full advisory_engine protocol.
        """
        prompt = f"""You are CHAD's advisory synthesis layer.

Coaching instructions:
{coaching_profile}

User request:
{json.dumps({"symbol": getattr(user_request, "symbol", str(user_request)), "question": getattr(user_request, "user_question", str(user_request))}, default=str)}

Runtime context:
{json.dumps(runtime_context, indent=2, default=str)[:8000]}

Engine outputs:
{json.dumps(engine_outputs, indent=2, default=str)[:16000]}

Provide a clear, practical advisory response. Be direct and risk-aware."""

        estimated_tokens = _approximate_tokens(prompt)

        with self._lock:
            self._enforce_rate_limits(estimated_tokens=estimated_tokens)

        t0 = time.monotonic()
        try:
            text, input_tok, output_tok = self._call_claude(
                prompt=prompt,
                system="You are CHAD — a sophisticated trading system's intelligence layer. You have the precision of a quant, the macro awareness of a global macro trader, and the directness of someone who respects the operator's time. When the signal is clear, say so directly. When data is mixed, say that too. Give specific rotation recommendations, regime assessments, and portfolio observations. Never execute — always advise. The operator makes final decisions. No hedging when the signal is clear. No jargon. Plain English always. Keep responses under 600 characters. CRITICAL: Only discuss symbols the user explicitly asked about. Never mention, recommend, or analyze tickers not present in the user request or engine outputs. Never invent stock symbols, prices, or positions not in the provided data. CRITICAL SELF-REFERENCE RULE: Questions using 'we', 'our', 'us', or 'the system' refer to CHAD's own trading performance (see runtime_context.system_status — scr_state, scr_win_rate, scr_sharpe, scr_reasons). Never interpret these as questions about external stock tickers. 'Will we hit our target' means 'will CHAD's win rate reach its SCR promotion threshold', not anything about a stock.",
                task_type="complex",
                max_tokens=1024,
                temperature=0.2,
            )

            elapsed_ms = int((time.monotonic() - t0) * 1000)

            with self._lock:
                self._update_token_usage(input_tok, output_tok, model=self.model_for_tier("complex"))

            _write_call_log(
                model=self.model_for_tier("complex"),
                task_type="synthesize",
                input_tokens=input_tok, output_tokens=output_tok,
                duration_ms=elapsed_ms, success=True,
            )

            return text

        except Exception as exc:
            self._logger.warning("Claude synthesize failed: %s", exc)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _write_call_log(
                model=self.model_for_tier("complex"),
                task_type="synthesize",
                input_tokens=0, output_tokens=0,
                duration_ms=elapsed_ms, success=False,
                error=str(exc),
            )
            return f"CHAD advisory synthesis unavailable: {type(exc).__name__}"

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> ClaudeConfig:
        return self._config


# ===================================================================== #
# OllamaClient — local model tier for routine tasks
# ===================================================================== #

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
OLLAMA_TIMEOUT_SEC = float(os.environ.get("CHAD_OLLAMA_TIMEOUT_SEC", "30.0"))

_ollama_logger = logging.getLogger("chad.ollama")


def _ollama_ram_ok() -> bool:
    """Return True if sufficient RAM is available for Ollama inference.

    Reads CHAD_OLLAMA_MIN_RAM_MB (default 1500MB). Fail-open: if psutil
    raises for any reason, returns True to allow the call through.
    """
    try:
        available_mb = psutil.virtual_memory().available // 1024 // 1024
        if available_mb < OLLAMA_MIN_RAM_MB:
            logging.getLogger("chad.intel.claude_client").warning(
                "ollama_ram_circuit_breaker available_mb=%d threshold_mb=%d",
                available_mb,
                OLLAMA_MIN_RAM_MB,
            )
            return False
        return True
    except Exception:
        return True


class OllamaClient:
    """
    Local Ollama model client for lightweight routine tasks.

    Provides the same chat_json() interface as ClaudeClient for tier routing.
    Falls back gracefully if Ollama is not running.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT_SEC,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @staticmethod
    def is_available() -> bool:
        """Check if Ollama is running and has a model loaded."""
        import urllib.request
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/tags",
                headers={"User-Agent": "chad-ollama/1.0"},
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
        except Exception:
            return False

    def chat_json(
        self,
        prompt: str,
        *,
        task_type: str = "routine",
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt to Ollama and parse JSON response.

        Compatible with ClaudeClient.chat_json() interface.
        """
        if not _ollama_ram_ok():
            raise OllamaUnavailableError(
                "RAM circuit breaker: insufficient available memory "
                f"(threshold={OLLAMA_MIN_RAM_MB}MB)"
            )
        import urllib.request

        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        payload = json.dumps({
            "model": self._model,
            "prompt": full_prompt + "\n\nRespond ONLY with valid JSON, no extra text.",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "chad-ollama/1.0"},
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            text = str(raw.get("response", "")).strip()
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _ollama_logger.info(
                "ollama.ok model=%s elapsed_ms=%d len=%d",
                self._model, elapsed_ms, len(text),
            )

            # Extract JSON from response (may have markdown wrapping)
            return _extract_json(text)

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            _ollama_logger.warning("ollama.error model=%s elapsed_ms=%d err=%s", self._model, elapsed_ms, exc)
            raise

    def generate(self, prompt: str, *, max_tokens: int = 256) -> str:
        """Raw text generation without JSON parsing."""
        if not _ollama_ram_ok():
            raise OllamaUnavailableError(
                "RAM circuit breaker: insufficient available memory "
                f"(threshold={OLLAMA_MIN_RAM_MB}MB)"
            )
        import urllib.request

        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "chad-ollama/1.0"},
        )

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        return str(raw.get("response", "")).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may contain markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return {"raw_text": text}

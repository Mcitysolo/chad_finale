"""
Single source of truth loader for LLM model IDs (GAP-037/038).

All model-ID sites in the intelligence layer read from config/llm_models.json
via this module. The config is advisory-only: if it is missing or corrupt,
every accessor returns a hardcoded default identical to the pre-IR1 literals,
so behaviour is unchanged on a load failure (fail-safe). This module NEVER
raises to callers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

_CONFIG_PATH = Path(
    os.environ.get(
        "CHAD_LLM_MODELS_PATH",
        "/home/ubuntu/chad_finale/config/llm_models.json",
    )
)

# Hardcoded defaults preserve the exact pre-IR1 model IDs so a missing or
# corrupt config file leaves behaviour identical to before centralisation.
_DEFAULTS: Dict[str, Any] = {
    "anthropic": {
        "tiers": {
            "routine": "claude-haiku-4-5-20251001",
            "standard": "claude-haiku-4-5-20251001",
            "complex": "claude-sonnet-4-6",
        },
        "cost_per_1k_usd": {
            "claude-haiku-4-5-20251001": 0.001,
            "claude-sonnet-4-6": 0.003,
            "claude-opus-4-7": 0.015,
        },
    },
    "openai": {"model": "gpt-4.1"},
    "fallback_order": ["ollama", "anthropic", "openai"],
}


def _load() -> Dict[str, Any]:
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return _DEFAULTS


def anthropic_tiers() -> Dict[str, str]:
    """Return the task_type -> Claude model-ID map (routine/standard/complex)."""
    data = _load()
    anthropic = data.get("anthropic") if isinstance(data.get("anthropic"), dict) else {}
    tiers = anthropic.get("tiers")
    if isinstance(tiers, dict) and tiers:
        return {str(k): str(v) for k, v in tiers.items()}
    return dict(_DEFAULTS["anthropic"]["tiers"])


def anthropic_cost_per_1k() -> Dict[str, float]:
    """Return the Claude model-ID -> USD-per-1k-tokens cost map."""
    data = _load()
    anthropic = data.get("anthropic") if isinstance(data.get("anthropic"), dict) else {}
    cost = anthropic.get("cost_per_1k_usd")
    out: Dict[str, float] = {}
    if isinstance(cost, dict) and cost:
        for k, v in cost.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out or dict(_DEFAULTS["anthropic"]["cost_per_1k_usd"])


def openai_model(default: str = "gpt-4.1") -> str:
    """Return the canonical OpenAI advisory model (documented choice: gpt-4.1)."""
    data = _load()
    openai = data.get("openai") if isinstance(data.get("openai"), dict) else {}
    model = openai.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return default


def fallback_order() -> list:
    """Return the documented provider fallback order (informational)."""
    data = _load()
    order = data.get("fallback_order")
    if isinstance(order, list) and order:
        return [str(x) for x in order]
    return list(_DEFAULTS["fallback_order"])

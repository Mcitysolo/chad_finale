"""GAP-005: OLLAMA_TIMEOUT_SEC must be env-configurable for fast-fail."""
from __future__ import annotations

import importlib

import chad.intel.claude_client as claude_client


def test_default_timeout_is_30s(monkeypatch):
    monkeypatch.delenv("CHAD_OLLAMA_TIMEOUT_SEC", raising=False)
    reloaded = importlib.reload(claude_client)
    try:
        assert reloaded.OLLAMA_TIMEOUT_SEC == 30.0
    finally:
        importlib.reload(claude_client)


def test_env_var_overrides_timeout(monkeypatch):
    monkeypatch.setenv("CHAD_OLLAMA_TIMEOUT_SEC", "5.0")
    reloaded = importlib.reload(claude_client)
    try:
        assert reloaded.OLLAMA_TIMEOUT_SEC == 5.0
    finally:
        monkeypatch.delenv("CHAD_OLLAMA_TIMEOUT_SEC", raising=False)
        importlib.reload(claude_client)

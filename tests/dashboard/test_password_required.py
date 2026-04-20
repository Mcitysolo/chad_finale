"""Step 9: verify fail-closed when CHAD_DASHBOARD_PASSWORD unset."""
import importlib
import sys

import pytest


def test_password_env_required(monkeypatch):
    """api module must raise RuntimeError if env var is missing."""
    monkeypatch.delenv("CHAD_DASHBOARD_PASSWORD", raising=False)
    sys.modules.pop("chad.dashboard.api", None)
    with pytest.raises(RuntimeError, match="CHAD_DASHBOARD_PASSWORD"):
        importlib.import_module("chad.dashboard.api")


def test_password_env_accepted(monkeypatch):
    """api module imports successfully when env var is set."""
    monkeypatch.setenv("CHAD_DASHBOARD_PASSWORD", "test_value_12345")
    sys.modules.pop("chad.dashboard.api", None)
    mod = importlib.import_module("chad.dashboard.api")
    assert mod.DASHBOARD_PASSWORD == "test_value_12345"

"""Tests for chad.tools.kraken_futures_auth_smoke (Phase C Item 1C)."""

from __future__ import annotations

import ast
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

import pytest

from chad.exchanges.kraken_futures_client import (
    KrakenFuturesClient,
    KrakenFuturesCredentials,
)
from chad.tools import kraken_futures_auth_smoke as smoke

_TOOL_PATH = Path(smoke.__file__)
_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "PHASE_C_C1C_KRAKEN_FUTURES_AUTH_SMOKE.md"

_FUTURES_ENV_KEYS = ("KRAKEN_FUTURES_API_KEY", "KRAKEN_FUTURES_API_SECRET")


@pytest.fixture(autouse=True)
def _isolate_credentials(monkeypatch, tmp_path):
    """Ensure tests never see real Futures credentials and never read /etc."""
    for key in _FUTURES_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    sentinel = tmp_path / "kraken.env.missing"
    monkeypatch.setattr(
        "chad.exchanges.kraken_futures_client._FALLBACK_ENV_FILE",
        sentinel,
        raising=True,
    )


def _run_and_capture(argv, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = smoke.run(argv, **kwargs)
    return code, stdout.getvalue(), stderr.getvalue()


def test_dry_run_exits_zero_with_status_line():
    code, out, _err = _run_and_capture(["--dry-run"])
    assert code == 0
    assert "KRAKEN_FUTURES_AUTH_SMOKE status=dry_run_ok" in out
    assert "credentials_present=false" in out
    assert "dry_run=true" in out


def test_default_mode_no_credentials_exits_two():
    code, out, _err = _run_and_capture([])
    assert code == 2
    assert "KRAKEN_FUTURES_AUTH_SMOKE status=missing_credentials" in out
    assert "credentials_present=false" in out


def test_live_readonly_no_credentials_exits_two():
    code, out, _err = _run_and_capture(["--live-readonly"])
    assert code == 2
    assert "KRAKEN_FUTURES_AUTH_SMOKE status=missing_credentials" in out
    assert "live_readonly=true" in out


def test_live_readonly_with_credentials_but_uncertified_endpoint_exits_three(monkeypatch):
    fake = KrakenFuturesCredentials(api_key="FAKE_KEY", api_secret="FAKE_SECRET")
    monkeypatch.setattr(smoke, "load_credentials_from_env", lambda: fake)

    def fake_probe(_client: KrakenFuturesClient) -> Dict[str, Any]:
        return {"status": "not_certified", "error": "uncertified"}

    code, out, _err = _run_and_capture(["--live-readonly"], probe=fake_probe)
    assert code == 3
    assert (
        "KRAKEN_FUTURES_AUTH_SMOKE status=credentials_present_endpoint_not_certified"
        in out
    )
    assert "credentials_present=true" in out


def test_default_mode_with_credentials_exits_three(monkeypatch):
    fake = KrakenFuturesCredentials(api_key="FAKE_KEY", api_secret="FAKE_SECRET")
    monkeypatch.setattr(smoke, "load_credentials_from_env", lambda: fake)

    code, out, _err = _run_and_capture([])
    assert code == 3
    assert (
        "KRAKEN_FUTURES_AUTH_SMOKE status=credentials_present_endpoint_not_certified"
        in out
    )


def test_no_secrets_printed(monkeypatch):
    secret_token = "SECRET-TOKEN-MUST-NOT-LEAK-9XYZ"
    fake = KrakenFuturesCredentials(api_key="API-TOKEN-LEAK-CHECK-7Q", api_secret=secret_token)
    monkeypatch.setattr(smoke, "load_credentials_from_env", lambda: fake)

    for argv in ([], ["--live-readonly"], ["--dry-run"]):
        _code, out, err = _run_and_capture(argv)
        assert secret_token not in out
        assert secret_token not in err
        assert fake.api_key not in out
        assert fake.api_key not in err


def test_module_does_not_import_strategies_or_live_loop():
    source = _TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    forbidden_prefixes = (
        "chad.strategies",
        "chad.core.live_loop",
        "chad.execution.execution_pipeline",
        "chad.execution.kraken_executor",
        "chad.execution.kraken_trade_router",
    )
    for mod in modules:
        for prefix in forbidden_prefixes:
            assert not mod.startswith(prefix), f"forbidden import: {mod}"


def test_module_does_not_call_submit_order():
    source = _TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                assert func.attr != "submit_order", "tool must never invoke submit_order"
            if isinstance(func, ast.Name):
                assert func.id != "submit_order", "tool must never invoke submit_order"

    test_source = Path(__file__).read_text(encoding="utf-8")
    test_tree = ast.parse(test_source)
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                assert func.attr != "submit_order", "test must never invoke submit_order"
            if isinstance(func, ast.Name):
                assert func.id != "submit_order", "test must never invoke submit_order"


def test_main_handles_unexpected_exception_returns_one(monkeypatch):
    def boom(_argv=None, **_kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(smoke, "run", boom)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = smoke.main([])
    assert code == 1
    assert "KRAKEN_FUTURES_AUTH_SMOKE status=unexpected_error" in stdout.getvalue()


def test_documentation_file_present_with_required_phrases():
    assert _DOC_PATH.exists(), f"missing doc at {_DOC_PATH}"
    text = _DOC_PATH.read_text(encoding="utf-8")
    assert "never places orders" in text
    assert "AUTHENTICATED SMOKE TEST SCAFFOLD ONLY" in text


def test_conflicting_flags_exits_one():
    code, out, _err = _run_and_capture(["--dry-run", "--live-readonly"])
    assert code == 1
    assert "KRAKEN_FUTURES_AUTH_SMOKE status=conflicting_flags" in out

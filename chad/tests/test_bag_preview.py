"""
Tier 1 Phase D Item 2 — preview_bag_intent CLI behavior + safety tests.

This module verifies the dry-run BAG preview CLI:
  * emits the expected JSON shape on success
  * refuses any live execution mode
  * imports no IBKR connection / placeOrder code paths
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_SCRIPT = REPO_ROOT / "scripts" / "preview_bag_intent.py"


def _run_preview(args: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Default to paper mode unless the test overrides it.
    env.setdefault("CHAD_EXECUTION_MODE", "paper")
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(PREVIEW_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )


# ---------- 1. happy path ----------

def test_preview_emits_expected_json() -> None:
    result = _run_preview(
        [
            "--symbol", "SPY",
            "--expiry", "20260618",
            "--long-strike", "737",
            "--short-strike", "744",
            "--long-right", "C",
            "--short-right", "C",
            "--contracts", "1",
            "--spread-type", "BULL_CALL",
            "--net-debit-estimate", "350",
            "--max-loss-per-contract", "700",
        ]
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["symbol"] == "SPY"
    assert payload["sec_type"] == "BAG"
    assert payload["contracts"] == 1
    assert payload["spread_spec"]["long_strike"] == 737.0
    assert payload["spread_spec"]["short_strike"] == 744.0
    assert payload["legacy_meta"]["sec_type"] == "BAG"
    assert payload["legacy_meta"]["long_right"] == "C"
    assert payload["legacy_meta"]["short_right"] == "C"
    assert len(payload["bag_legs"]) == 2
    assert payload["bag_legs"][0]["action"] == "BUY"
    assert payload["bag_legs"][1]["action"] == "SELL"


# ---------- 2. mode gate ----------

def test_preview_refuses_live_mode() -> None:
    result = _run_preview(
        [
            "--symbol", "SPY",
            "--expiry", "20260618",
            "--long-strike", "737",
            "--short-strike", "744",
            "--long-right", "C",
            "--short-right", "C",
            "--contracts", "1",
        ],
        extra_env={"CHAD_EXECUTION_MODE": "live"},
    )
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "preview_refused_non_paper_mode"
    assert payload["CHAD_EXECUTION_MODE"] == "live"


@pytest.mark.parametrize("mode", ["paper", "dry_run", "preview", ""])
def test_preview_accepts_paper_dryrun_and_preview_modes(mode) -> None:
    extra: dict[str, str] = {}
    if mode == "":
        # Remove CHAD_EXECUTION_MODE to simulate unset.
        extra = {"CHAD_EXECUTION_MODE": ""}
    else:
        extra = {"CHAD_EXECUTION_MODE": mode}
    result = _run_preview(
        [
            "--symbol", "SPY",
            "--expiry", "20260618",
            "--long-strike", "737",
            "--short-strike", "744",
            "--long-right", "C",
            "--short-right", "C",
            "--contracts", "1",
        ],
        extra_env=extra,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


# ---------- 3. validation surface ----------

def test_preview_reports_validator_error_for_same_strike() -> None:
    result = _run_preview(
        [
            "--symbol", "SPY",
            "--expiry", "20260618",
            "--long-strike", "737",
            "--short-strike", "737",
            "--long-right", "C",
            "--short-right", "C",
            "--contracts", "1",
        ]
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "spread_spec_validation_failed"
    assert "differ" in payload["detail"]


def test_preview_reports_validator_error_for_bad_expiry() -> None:
    result = _run_preview(
        [
            "--symbol", "SPY",
            "--expiry", "2026-06-18",
            "--long-strike", "737",
            "--short-strike", "744",
            "--long-right", "C",
            "--short-right", "C",
            "--contracts", "1",
        ]
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "spread_spec_validation_failed"
    assert "expiry" in payload["detail"]


# ---------- 4. source safety ----------

def test_preview_source_has_no_ibkr_imports_or_placeorder() -> None:
    """The preview script must never reach a live broker path.

    We check the *executable* portion of the source (strip docstrings and
    line comments) so the docstring's plain-English description of what
    the script does NOT do is not flagged.
    """
    import ast

    text = PREVIEW_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)

    # Collect all import names + every attribute / Name access in the AST.
    imports: list[str] = []
    name_uses: list[str] = []
    call_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.Name):
            name_uses.append(node.id)
        elif isinstance(node, ast.Attribute):
            name_uses.append(node.attr)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                call_names.append(func.id)
            elif isinstance(func, ast.Attribute):
                call_names.append(func.attr)

    forbidden_modules = {"ib_async", "ib_insync"}
    for mod in imports:
        root = mod.split(".")[0]
        assert root not in forbidden_modules, (
            f"preview script must not import {mod!r}"
        )
        assert "ibkr_adapter" not in mod, (
            f"preview script must not import the IBKR adapter ({mod!r})"
        )

    forbidden_calls = {"placeOrder", "connectAsync", "connect"}
    for call in call_names:
        assert call not in forbidden_calls, (
            f"preview script must not call {call!r}"
        )

    forbidden_names = {"IbkrAdapter"}
    for name in name_uses:
        assert name not in forbidden_names, (
            f"preview script must not reference {name!r}"
        )


def test_preview_test_module_has_no_runtime_ibkr_imports() -> None:
    """Audit the test module itself via AST so this safety surface is
    captured even if someone later adds a copy-paste mistake."""
    import ast

    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in {"ib_async", "ib_insync"}, (
                    f"preview test must not import {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0] if mod else ""
            assert root not in {"ib_async", "ib_insync"}, (
                f"preview test must not import {mod!r}"
            )
            assert "ibkr_adapter" not in mod, (
                f"preview test must not import IBKR adapter ({mod!r})"
            )

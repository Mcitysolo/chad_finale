"""
ISSUE-78 — execution-mode canonicalization tests.

Pin three contracts:
  1) Hot-path callers route execution-mode decisions through the canonical
     reader chad.execution.execution_config.get_execution_mode().
  2) Invalid / missing CHAD_EXECUTION_MODE values fail safe to DRY_RUN —
     no implicit upgrade to a riskier posture.
  3) No duplicate direct CHAD_EXECUTION_MODE env reads remain in hot
     execution / risk / orchestration code paths. Only execution_config.py
     and the show_execution_config diagnostic CLI may reference the env
     name directly; everywhere else must go through get_execution_mode().
"""

from __future__ import annotations

import re
from importlib import reload
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HOT_PATH_DIRS = (
    REPO_ROOT / "chad" / "core",
    REPO_ROOT / "chad" / "execution",
    REPO_ROOT / "chad" / "risk",
    REPO_ROOT / "chad" / "ops",
    REPO_ROOT / "chad" / "portfolio",
    REPO_ROOT / "chad" / "strategies",
)

ENV_READ_PATTERN = re.compile(
    r"""(?:os\.getenv|os\.environ\.get|os\.environ\[)\s*\(?\s*["']CHAD_EXECUTION_MODE["']""",
)

# Files allowed to read CHAD_EXECUTION_MODE directly:
#   - execution_config.py  → canonical reader implementation
ALLOWED_DIRECT_READERS = {
    REPO_ROOT / "chad" / "execution" / "execution_config.py",
}


def test_canonical_reader_dry_run_default(monkeypatch) -> None:
    """Missing CHAD_EXECUTION_MODE must fall back to DRY_RUN."""
    from chad.execution import execution_config as ec

    monkeypatch.delenv("CHAD_EXECUTION_MODE", raising=False)
    assert ec.get_execution_mode() == ec.ExecutionMode.DRY_RUN
    assert ec.is_paper_mode() is True
    assert ec.is_live_mode() is False


def test_execution_mode_invalid_value_fails_safe_or_policy_default(monkeypatch) -> None:
    """Invalid CHAD_EXECUTION_MODE values must fail closed to DRY_RUN.

    Crucially: an invalid value must NOT silently upgrade to ibkr_live.
    """
    from chad.execution import execution_config as ec

    for bogus in ("garbage", "LIVE_TRADING", "yolo", "  ", "ibkr_unknown"):
        monkeypatch.setenv("CHAD_EXECUTION_MODE", bogus)
        m = ec.get_execution_mode()
        assert m == ec.ExecutionMode.DRY_RUN, (
            f"invalid CHAD_EXECUTION_MODE={bogus!r} must fail closed to DRY_RUN, got {m}"
        )
        assert ec.is_live_mode() is False
        cfg = ec.get_execution_config()
        assert cfg.mode == ec.ExecutionMode.DRY_RUN
        assert cfg.ibkr_dry_run is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("dry_run", "dry_run"),
        ("paper", "ibkr_paper"),
        ("ibkr_paper", "ibkr_paper"),
        ("live", "ibkr_live"),
        ("ibkr_live", "ibkr_live"),
        ("PAPER", "ibkr_paper"),
        ("  Live  ", "ibkr_live"),
    ],
)
def test_canonical_reader_alias_normalization(monkeypatch, raw, expected) -> None:
    """The alias map normalizes paper/live to ibkr_paper/ibkr_live."""
    from chad.execution import execution_config as ec

    monkeypatch.setenv("CHAD_EXECUTION_MODE", raw)
    assert ec.get_execution_mode().value == expected


def test_execution_mode_hot_paths_use_canonical_reader() -> None:
    """Each affected hot-path module must import get_execution_mode from
    chad.execution.execution_config (the canonical reader) — not roll its
    own env read.
    """
    affected = [
        REPO_ROOT / "chad" / "core" / "live_gate.py",
        REPO_ROOT / "chad" / "core" / "live_loop.py",
        REPO_ROOT / "chad" / "core" / "orchestrator.py",
        REPO_ROOT / "chad" / "core" / "full_execution_cycle.py",
        REPO_ROOT / "chad" / "core" / "paper_position_closer.py",
        REPO_ROOT / "chad" / "ops" / "operator_intent_refresher.py",
    ]
    for p in affected:
        src = p.read_text(encoding="utf-8")
        assert "from chad.execution.execution_config import" in src, (
            f"{p} does not import from chad.execution.execution_config"
        )
        # Canonical helper must appear (either get_execution_mode or
        # is_paper_mode / is_live_mode which themselves call it).
        assert any(
            tok in src
            for tok in ("get_execution_mode", "is_paper_mode", "is_live_mode")
        ), f"{p} does not use a canonical execution_config helper"


def test_no_direct_chad_execution_mode_reads_in_hot_paths() -> None:
    """No hot-path module should call os.getenv("CHAD_EXECUTION_MODE") /
    os.environ.get("CHAD_EXECUTION_MODE") / os.environ["CHAD_EXECUTION_MODE"].

    Only execution_config.py is permitted to read the env var directly.
    """
    offenders: list[str] = []
    for hot_dir in HOT_PATH_DIRS:
        if not hot_dir.is_dir():
            continue
        for py in hot_dir.rglob("*.py"):
            if py in ALLOWED_DIRECT_READERS:
                continue
            if "__pycache__" in py.parts:
                continue
            if py.name.endswith(".bak") or ".bak" in py.suffixes:
                continue
            text = py.read_text(encoding="utf-8", errors="ignore")
            if ENV_READ_PATTERN.search(text):
                offenders.append(str(py.relative_to(REPO_ROOT)))

    assert offenders == [], (
        "Direct CHAD_EXECUTION_MODE env reads found in hot paths "
        "(should go through chad.execution.execution_config.get_execution_mode): "
        f"{offenders}"
    )


def test_kraken_mode_remains_separate_env(monkeypatch) -> None:
    """CHAD_KRAKEN_MODE is intentionally a separate Kraken-only override.

    Reason: the Kraken executor lane must be controllable independently of
    the IBKR execution mode (e.g. paper_kraken validate-only without
    flipping IBKR posture, or running IBKR live with Kraken off). Document
    the contract so this test pins the separation.
    """
    from chad.core import kraken_execution as ke

    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    monkeypatch.delenv("CHAD_EXECUTION_MODE", raising=False)
    assert ke.resolve_kraken_mode() == "off"

    monkeypatch.setenv("CHAD_KRAKEN_MODE", "paper_kraken")
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "dry_run")
    assert ke.resolve_kraken_mode() == "paper_kraken"

    # When CHAD_KRAKEN_MODE is unset, kraken falls back to the canonical
    # execution_config.is_live_mode() — confirming the canonical reader is
    # the only IBKR-mode source the kraken module consults.
    monkeypatch.delenv("CHAD_KRAKEN_MODE", raising=False)
    monkeypatch.setenv("CHAD_EXECUTION_MODE", "live")
    assert ke.resolve_kraken_mode() == "live"

    monkeypatch.setenv("CHAD_EXECUTION_MODE", "paper")
    assert ke.resolve_kraken_mode() == "off"

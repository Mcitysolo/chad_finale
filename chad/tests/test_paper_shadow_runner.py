from __future__ import annotations

import os
import sys

from chad.core.paper_shadow_runner import (
    ARM_ENV_NAME,
    ARM_PHRASE,
    PaperShadowConfig,
    is_armed,
    should_place_paper_orders,
)


def test_is_armed_false_by_default(monkeypatch) -> None:
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    assert is_armed() is False


def test_is_armed_true_only_with_exact_phrase(monkeypatch) -> None:
    monkeypatch.setenv(ARM_ENV_NAME, ARM_PHRASE)
    assert is_armed() is True


def test_should_place_paper_orders_disabled(monkeypatch) -> None:
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    cfg = PaperShadowConfig(enabled=False)
    allowed, reasons = should_place_paper_orders(cfg)
    assert allowed is False
    assert any("enabled is false" in r for r in reasons)


def test_should_place_paper_orders_requires_arm(monkeypatch) -> None:
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    cfg = PaperShadowConfig(enabled=True)
    allowed, reasons = should_place_paper_orders(cfg)
    assert allowed is False
    assert any("not set to arm phrase" in r for r in reasons)


def test_module_does_not_import_ib_insync_on_safe_paths(monkeypatch) -> None:
    """
    Import safety: calling the gating functions should not import ib_insync.
    (We import ib_insync lazily only inside preview pipeline builder, and
    even that should not run during these gating tests.)
    """
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    _ = should_place_paper_orders(PaperShadowConfig(enabled=False))
    assert "ib_insync" not in sys.modules

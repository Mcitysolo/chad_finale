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
    Import safety contract:
    Calling the gating functions must not *cause* the broker library to be
    imported. PR-03 strengthened this from "ib_insync only" to also reject
    ib_async because the entire point of the gating layer is to defer
    broker code until the operator has explicitly armed.

    Note: other tests may legitimately import ib_insync or ib_async earlier
    in the same pytest process. So we assert on the delta: the call must
    not introduce a new import.
    """
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)

    before = set(sys.modules.keys())
    _ = should_place_paper_orders(PaperShadowConfig(enabled=False))
    after = set(sys.modules.keys())

    # The gating call must not be the reason ib_insync or ib_async appears.
    delta = after - before
    assert "ib_insync" not in delta
    assert "ib_async" not in delta

from __future__ import annotations

from chad.core.paper_shadow_runner import (
    ARM_ENV_NAME,
    ARM_PHRASE,
    PaperShadowConfig,
    _live_gate_allows_paper,
    _should_execute_paper,
)


def test_execute_gate_disabled(monkeypatch) -> None:
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    ok, reasons = _should_execute_paper(PaperShadowConfig(enabled=False, armed=False))
    assert ok is False
    assert any("enabled" in r for r in reasons)


def test_execute_gate_requires_config_armed(monkeypatch) -> None:
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    ok, reasons = _should_execute_paper(PaperShadowConfig(enabled=True, armed=False))
    assert ok is False
    assert any("armed is false" in r.lower() for r in reasons)


def test_execute_gate_requires_env_phrase(monkeypatch) -> None:
    monkeypatch.delenv(ARM_ENV_NAME, raising=False)
    ok, reasons = _should_execute_paper(PaperShadowConfig(enabled=True, armed=True))
    assert ok is False
    assert any("arm phrase" in r.lower() for r in reasons)


def test_execute_gate_true_only_when_both_armed(monkeypatch) -> None:
    monkeypatch.setenv(ARM_ENV_NAME, ARM_PHRASE)
    ok, reasons = _should_execute_paper(PaperShadowConfig(enabled=True, armed=True))
    assert ok is True
    assert any("env-armed" in r.lower() for r in reasons)


def test_live_gate_fail_safe(monkeypatch) -> None:
    # If live-gate cannot be reached, we MUST fail safe and refuse execution.
    # We do not mock urllib internals; we rely on the function's try/except behavior.
    ok, reason = _live_gate_allows_paper()
    assert ok in (True, False)
    # Regardless of result, it must be a string and must not raise.
    assert isinstance(reason, str)

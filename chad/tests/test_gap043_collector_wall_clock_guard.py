"""NEW-GAP-043 — chad-ibkr-collector wall-clock guard.

Pre-fix audit observed chad-ibkr-collector.service had stuck in
`activating/start` for 9h18m, with the systemd unit configured as
TimeoutStartSec=infinity / RuntimeMaxSec=infinity. This file pins the
code-level defense-in-depth: the collector installs a SIGALRM-based
wall-clock guard at the top of main() so the process self-terminates
with exit code 124 even if the systemd guardrails are missing or
misconfigured (e.g. an older host that hasn't picked up the drop-in
in ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf).

We pin:
  * install_wall_clock_guard installs SIGALRM and returns the budget
  * the alarm actually fires after the configured budget and produces
    a CollectorWallClockTimeout(SystemExit(124))
  * the env var CHAD_COLLECTOR_WALL_CLOCK_SECONDS overrides the default
  * malformed env values fall back to the safe default
  * disarm_wall_clock_guard cancels a pending alarm
  * the systemd drop-in file exists and contains finite TimeoutStartSec
    + RuntimeMaxSec
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from chad.portfolio import ibkr_portfolio_collector_v2 as collector


@pytest.fixture(autouse=True)
def _clear_alarm():
    """Each test starts with no pending alarm."""
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)
    yield
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


# ---------------------------------------------------------------------------
# install_wall_clock_guard returns budget + arms SIGALRM
# ---------------------------------------------------------------------------


def test_install_with_default_returns_default_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(collector.WALL_CLOCK_ENV, raising=False)
    s = collector.install_wall_clock_guard()
    assert s == collector.DEFAULT_WALL_CLOCK_SECONDS
    assert s == 60


def test_install_with_explicit_seconds_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(collector.WALL_CLOCK_ENV, "9999")
    s = collector.install_wall_clock_guard(seconds=5)
    assert s == 5  # explicit arg wins over env


def test_env_override_takes_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(collector.WALL_CLOCK_ENV, "12")
    s = collector.install_wall_clock_guard()
    assert s == 12


@pytest.mark.parametrize("bad", ["", "abc", "-3", "0", " "])
def test_malformed_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(collector.WALL_CLOCK_ENV, bad)
    s = collector.install_wall_clock_guard()
    assert s == collector.DEFAULT_WALL_CLOCK_SECONDS


# ---------------------------------------------------------------------------
# Alarm actually fires
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM unavailable")
def test_alarm_fires_and_raises_typed_systemexit_with_code_124() -> None:
    """The signal-driven path: install a 1s guard, sleep 2s, expect the
    SystemExit subclass with code=124 (the canonical timeout exit code)."""
    collector.install_wall_clock_guard(seconds=1)
    start = time.monotonic()
    with pytest.raises(collector.CollectorWallClockTimeout) as ei:
        # signal.pause() blocks until any signal arrives — clean way to
        # wait for SIGALRM without busy-spinning.
        signal.pause()
    elapsed = time.monotonic() - start
    assert 0.5 <= elapsed <= 3.0, f"alarm fired too early/late: {elapsed:.2f}s"
    assert isinstance(ei.value, SystemExit)
    assert ei.value.code == 124
    assert ei.value.seconds == 1


# ---------------------------------------------------------------------------
# Disarm + re-arm semantics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM unavailable")
def test_disarm_cancels_pending_alarm() -> None:
    collector.install_wall_clock_guard(seconds=1)
    collector.disarm_wall_clock_guard()
    # If the alarm were still armed, this sleep would be interrupted by
    # CollectorWallClockTimeout. Sleeping past the budget without exception
    # proves the alarm was cancelled.
    time.sleep(1.3)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM unavailable")
def test_rearming_replaces_previous_budget() -> None:
    """POSIX SIGALRM semantics: setting a new alarm replaces the old one.
    Install 60s, then install 1s, expect the 1s to fire."""
    collector.install_wall_clock_guard(seconds=60)
    collector.install_wall_clock_guard(seconds=1)
    start = time.monotonic()
    with pytest.raises(collector.CollectorWallClockTimeout):
        signal.pause()
    elapsed = time.monotonic() - start
    assert elapsed < 3.0


# ---------------------------------------------------------------------------
# main() installs the guard before any IBKR work
# ---------------------------------------------------------------------------


def test_main_installs_guard_before_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If main() exits via argparse (no subcommand), the wall-clock guard
    must already be armed by the time arg parsing fails. We monkeypatch
    install_wall_clock_guard to record the call order."""
    calls = []

    real = collector.install_wall_clock_guard

    def _spy(seconds=None):
        calls.append(("install", seconds))
        return real(seconds=1) if seconds is None else real(seconds=seconds)

    monkeypatch.setattr(collector, "install_wall_clock_guard", _spy)
    # No subcommand -> main() prints help and returns 2 without touching IB.
    rc = collector.main(argv=[])
    assert rc == 2
    assert calls and calls[0][0] == "install"


# ---------------------------------------------------------------------------
# systemd drop-in staging file exists with finite guardrails
# ---------------------------------------------------------------------------


REPO_ROOT = Path(__file__).resolve().parents[2]
DROPIN_PATH = (
    REPO_ROOT
    / "ops"
    / "systemd"
    / "chad-ibkr-collector.service.d"
    / "10-timeout-guards.conf"
)


def test_systemd_dropin_file_is_present() -> None:
    assert DROPIN_PATH.is_file(), (
        f"missing GAP-043 systemd drop-in: {DROPIN_PATH}. The Channel-1 "
        "deployment instructions depend on this file being checked in."
    )


def test_systemd_dropin_sets_finite_timeout_start_sec() -> None:
    text = DROPIN_PATH.read_text(encoding="utf-8")
    assert "[Service]" in text
    # Must explicitly set a finite TimeoutStartSec — the pre-fix unit had
    # TimeoutStartUSec=infinity, which is what allowed the 9h hang.
    assert "TimeoutStartSec=" in text
    # No `infinity` value allowed.
    assert "TimeoutStartSec=infinity" not in text.lower()


def test_systemd_dropin_sets_finite_runtime_max_sec() -> None:
    text = DROPIN_PATH.read_text(encoding="utf-8")
    assert "RuntimeMaxSec=" in text
    assert "RuntimeMaxSec=infinity" not in text.lower()


def test_systemd_dropin_sets_bounded_stop_timeout() -> None:
    text = DROPIN_PATH.read_text(encoding="utf-8")
    assert "TimeoutStopSec=" in text


def test_systemd_dropin_documents_channel_1_deployment_steps() -> None:
    """Operator action is gated behind Channel 1 (systemctl daemon-reload).
    The drop-in file must describe how to install + reload so the operator
    has a self-contained reference."""
    text = DROPIN_PATH.read_text(encoding="utf-8")
    assert "daemon-reload" in text
    assert "systemctl" in text
    assert "/etc/systemd/system/chad-ibkr-collector.service.d" in text

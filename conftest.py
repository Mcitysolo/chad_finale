"""Repo-root pytest conftest.

Two jobs, applied to EVERY test tree (``tests/``, ``chad/tests/``, ``tests/validation/``)
because this file is their common ancestor:

  1. Put the repo root on ``sys.path`` so ``import chad...`` resolves everywhere.
  2. Install the G3C-HF repo-write leak guard for the whole session and enforce it per
     test — any test that creates/modifies a file under the working-tree ``data/`` or
     ``runtime/`` tree fails loudly (see ``chad/testing/repo_write_guard.py``). This kills
     the entire class of "test wrote through a production default path instead of
     tmp_path" leaks — the ``margin_shadow_20270115.ndjson`` incident (2026-07-10).
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chad.testing.repo_write_guard import (  # noqa: E402  (after sys.path insert)
    install as _install_write_guard,
    set_enforcing as _set_write_guard_enforcing,
    take_attempts as _take_write_guard_attempts,
    uninstall as _uninstall_write_guard,
)


def pytest_configure(config):
    # Patch the write primitives once for the session. Enforcement stays OFF until a test
    # is actually running (the autouse fixture below), so collection / import-time wiring
    # — e.g. chad.core.live_loop building its module-level shadow gate — is never blocked.
    _install_write_guard()


def pytest_unconfigure(config):
    _uninstall_write_guard()


@pytest.fixture(autouse=True)
def _rth_gate_off_by_default(monkeypatch):
    """WKF U2: neutralise the market-hours (RTH) gate by default in tests.

    In production ``CHAD_RTH_GATE`` defaults ON, which blocks any equity/ETF
    intent submitted outside regular trading hours. Left unset, every existing
    adapter test that submits an equity intent (open-order guard, margin gate,
    idempotency, ...) would become wall-clock-dependent — green in-session, red
    off-hours/weekends. Setting it to ``"0"`` here makes those tests
    deterministic; the WKF-U2 suite (``test_wkf_u2_rth_gate.py``) re-enables the
    gate explicitly via ``monkeypatch.delenv("CHAD_RTH_GATE")`` to exercise the
    default-ON path. This is the "env-disable for tests" contract of the gate.
    """
    monkeypatch.setenv("CHAD_RTH_GATE", "0")
    yield


@pytest.fixture(autouse=True)
def _repo_write_guard():
    """Fail this test if it creates/modifies files under the repo's data/ | runtime/ tree.

    Writes are blocked at the primitive (they never touch disk); this teardown check is the
    backstop that also fails a test whose writer *swallows* the block (best-effort writers).
    """
    _take_write_guard_attempts()  # clear any residue from collection-time paranoia
    _set_write_guard_enforcing(True)
    try:
        yield
    finally:
        _set_write_guard_enforcing(False)
    attempts = _take_write_guard_attempts()
    if attempts:
        detail = "\n".join(f"  - {prim} {mode!r} -> {path}" for path, mode, prim in attempts)
        pytest.fail(
            "G3C-HF repo-write-guard: this test created/modified working-tree files under "
            "data/ | runtime/ (BLOCKED before touching disk — point the writer at tmp_path):"
            f"\n{detail}",
            pytrace=False,
        )

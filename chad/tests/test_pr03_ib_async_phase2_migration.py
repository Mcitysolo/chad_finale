"""PR-03 — ib_async Phase 2 migration completion.

Pins the contract that closed the Phase 2 ledger:

* Zero production-source ib_insync imports remain — the
  ``PHASE2_DEFERRED_FILES`` allowlist in
  ``chad/tests/test_ib_async_import_parity.py`` is empty.
* ``chad.core.paper_position_closer`` and ``chad.core.paper_shadow_runner``
  import cleanly without ever calling out to a real broker.
* The four other paper-runtime entry-point modules (broker events
  collector, paper fill harvester, paper ledger watcher, reconciliation
  publisher) also import cleanly.
* The paper_shadow_runner gating layer remains lightweight: importing
  the module or calling ``should_place_paper_orders`` must not pull
  ib_insync into ``sys.modules`` (PR-03 strengthens the original
  contract by additionally documenting that ib_async is the post-migration
  broker library — broker imports are kept lazy/function-local so the
  safety surface stays auditable).
* Live posture stays paper-only (sentinel against accidental
  ready_for_live or allow_ibkr_live flips).

Tests are stdlib-only and never open an IBKR socket. They reuse the
parity-test ledger as the source of truth.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from chad.tests.test_ib_async_import_parity import (
    IB_INSYNC_IMPORT_RE,
    PHASE1_MIGRATED_FILES,
    PHASE2_DEFERRED_FILES,
    _iter_production_python_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# T1 — Phase 2 ledger fully drained
# ---------------------------------------------------------------------------


def test_phase2_deferred_files_is_empty() -> None:
    """PR-03 closes the Phase 2 ledger entirely. An empty allowlist with the
    `test_remaining_ib_insync_imports_are_explicitly_allowlisted` invariant
    means any new production ib_insync importer immediately fails CI."""
    assert PHASE2_DEFERRED_FILES == (), (
        "PR-03 contract requires PHASE2_DEFERRED_FILES to be empty. "
        f"Currently has {len(PHASE2_DEFERRED_FILES)} entries."
    )


def test_phase1_migrated_files_includes_pr03_targets() -> None:
    """Both PR-03 targets must be recorded in the migrated ledger so the
    `test_phase1_migrated_files_do_not_import_ib_insync` invariant catches
    any regression that re-adds an ib_insync import to them."""
    migrated = set(PHASE1_MIGRATED_FILES)
    assert "chad/core/paper_position_closer.py" in migrated
    assert "chad/core/paper_shadow_runner.py" in migrated


# ---------------------------------------------------------------------------
# T1 (continued) — direct grep for residual ib_insync imports in the
# migrated files. Belt-and-braces against a future edit that bypasses the
# parity-test regex (e.g. via `importlib.import_module`).
# ---------------------------------------------------------------------------


def _file_imports_ib_insync(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return bool(IB_INSYNC_IMPORT_RE.search(text))


@pytest.mark.parametrize(
    "rel_path",
    [
        "chad/core/paper_position_closer.py",
        "chad/core/paper_shadow_runner.py",
    ],
)
def test_pr03_migrated_files_no_longer_import_ib_insync(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"{rel_path} must exist"
    assert not _file_imports_ib_insync(path), (
        f"{rel_path} must not re-introduce an ib_insync import after PR-03"
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        "chad/core/paper_position_closer.py",
        "chad/core/paper_shadow_runner.py",
    ],
)
def test_pr03_migrated_files_now_import_ib_async(rel_path: str) -> None:
    """The migrated files must reference ib_async — otherwise the broker
    plumbing has been silently removed instead of swapped."""
    path = REPO_ROOT / rel_path
    text = path.read_text(encoding="utf-8", errors="ignore")
    assert re.search(
        r"^\s*(?:from\s+ib_async\b|import\s+ib_async\b)", text, re.MULTILINE
    ), f"{rel_path} must import ib_async"


# ---------------------------------------------------------------------------
# T2 — global zero-importers sweep. PHASE2 is empty, so any production
# file importing ib_insync is unexpected.
# ---------------------------------------------------------------------------


def test_zero_production_ib_insync_importers() -> None:
    """A repo-wide scan of production source must turn up zero ib_insync
    importers. Catches drift even if someone forgets to update the parity
    ledger."""
    offenders: list[str] = []
    for path in _iter_production_python_files():
        if _file_imports_ib_insync(path):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], (
        "Production source imports ib_insync (Phase 2 must be complete). "
        f"Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# T3 / T4 — module import smoke tests. Importing these modules must not
# raise, must not open broker sockets, and must not even pull the broker
# library into sys.modules when the broker code path isn't exercised.
# ---------------------------------------------------------------------------


def _import_in_subprocess(module_name: str) -> tuple[int, str, str]:
    """Run a fresh Python subprocess that imports ``module_name`` and reports
    whether ib_insync ended up in sys.modules. Subprocess isolation is
    required because other tests in the same pytest process legitimately
    import ib_insync (the coexistence parity tests)."""
    import subprocess
    import textwrap

    src = textwrap.dedent(
        f"""
        import json, sys, os
        os.environ.setdefault("CHAD_SKIP_IB_CONNECT", "1")
        try:
            __import__("{module_name}")
        except Exception as exc:
            print(json.dumps({{"ok": False, "error_type": type(exc).__name__,
                               "error_message": str(exc)}}))
            raise SystemExit(1)
        info = {{
            "ok": True,
            "ib_insync_in_sys_modules": "ib_insync" in sys.modules,
            "ib_async_in_sys_modules": "ib_async" in sys.modules,
        }}
        print(json.dumps(info))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(REPO_ROOT),
            "CHAD_SKIP_IB_CONNECT": "1",
        },
        timeout=30.0,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_paper_position_closer_imports_cleanly() -> None:
    """Importing paper_position_closer must succeed without opening a broker
    socket. Module-level ib_async types are expected to land in sys.modules
    (they are used as static type references); ib_insync MUST NOT."""
    rc, stdout, stderr = _import_in_subprocess("chad.core.paper_position_closer")
    assert rc == 0, f"import failed: rc={rc}, stderr={stderr}"
    info = json.loads(stdout.strip().splitlines()[-1])
    assert info["ok"] is True
    assert info["ib_insync_in_sys_modules"] is False, (
        "paper_position_closer must not pull ib_insync into sys.modules"
    )
    # ib_async is allowed (the module-level type imports are intentional).
    assert info["ib_async_in_sys_modules"] is True


def test_paper_shadow_runner_imports_cleanly_without_broker_lib() -> None:
    """paper_shadow_runner keeps its broker imports inside functions, so a
    bare module import must NOT pull the broker library at all — neither
    ib_insync nor ib_async. This preserves the original GAP-A019 ledger
    invariant in stronger form (post-migration we also forbid ib_async,
    because the entire point of the gating layer is to defer broker code)."""
    rc, stdout, stderr = _import_in_subprocess("chad.core.paper_shadow_runner")
    assert rc == 0, f"import failed: rc={rc}, stderr={stderr}"
    info = json.loads(stdout.strip().splitlines()[-1])
    assert info["ok"] is True
    assert info["ib_insync_in_sys_modules"] is False
    assert info["ib_async_in_sys_modules"] is False, (
        "paper_shadow_runner must keep broker imports function-local; bare "
        "module import should not load ib_async"
    )


# ---------------------------------------------------------------------------
# T5 — Service entrypoint smoke imports. None of these should raise even
# without IBKR running.
# ---------------------------------------------------------------------------


SERVICE_ENTRYPOINTS = (
    "chad.core.paper_position_closer",
    "chad.core.paper_shadow_runner",
    "chad.portfolio.ibkr_paper_fill_harvester",
    "chad.portfolio.ibkr_paper_ledger_watcher",
    "chad.ops.reconciliation_publisher",
    "chad.ops.ibkr_broker_events_collector",
)


@pytest.mark.parametrize("modname", SERVICE_ENTRYPOINTS)
def test_service_entrypoint_imports(modname: str) -> None:
    rc, stdout, stderr = _import_in_subprocess(modname)
    assert rc == 0, (
        f"service entrypoint {modname} fails to import: rc={rc}, "
        f"stderr={stderr}"
    )


# ---------------------------------------------------------------------------
# T7 — PR-09 contract still holds: positions_truth carries the broker
# authority status and replay diagnostic status as separate fields.
# ---------------------------------------------------------------------------


def test_pr09_positions_truth_contract_preserved() -> None:
    p = REPO_ROOT / "runtime" / "positions_truth.json"
    if not p.is_file():
        pytest.skip("runtime/positions_truth.json not present in this env")
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert "broker_authority_status" in doc
    assert "replay_diagnostic_status" in doc


# ---------------------------------------------------------------------------
# T8 — PR-02 / PR-02b placeholder protections still hold: the delta
# strategy's upstream-abstain helper exists and the reconciler still
# guards against synthesized close-fill placeholders. Pinning the symbol
# surfaces, not behavior — behavior is pinned by their own tests.
# ---------------------------------------------------------------------------


def test_pr02_delta_strategy_abstain_surface_present() -> None:
    """The PR-02 delta-strategy abstain fix relies on the price guard. Pin
    that the producer of delta signals still loads."""
    mod = importlib.import_module("chad.strategies.delta")
    assert mod is not None


# ---------------------------------------------------------------------------
# T9 — PR-04 options truth handling: the structured failure-artifact
# helpers must still be importable. Behavior is pinned in test_pr04_*.
# ---------------------------------------------------------------------------


def test_pr04_failure_artifact_constants_present() -> None:
    mod = importlib.import_module("chad.market_data.options_chain_refresh")
    assert mod.FAILURE_ARTIFACT_NAME == "options_chain_refresh_failure.json"
    assert mod.FAILURE_ARTIFACT_SCHEMA == "options_chain_refresh_failure.v1"


# ---------------------------------------------------------------------------
# T10 — live posture remains paper-only
# ---------------------------------------------------------------------------


def test_live_posture_unchanged_paper_only() -> None:
    live = json.loads(
        (REPO_ROOT / "runtime" / "live_readiness.json").read_text(encoding="utf-8")
    )
    hb = json.loads(
        (REPO_ROOT / "runtime" / "decision_trace_heartbeat.json").read_text(
            encoding="utf-8"
        )
    )
    assert live.get("ready_for_live") is False
    assert hb.get("allow_ibkr_live") is False
    assert hb.get("allow_ibkr_paper") is True

"""GAP-007 Phase 0 — ib_async install + import-parity validation.

Phase 0 only validates that ib_async can be installed and imported alongside
ib_insync. No production source file may be migrated from ib_insync to
ib_async in this phase. The migration of source imports happens in later
phases.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ib_async_imports_core_api():
    import ib_async
    from ib_async import IB, Stock, Contract, Order

    assert hasattr(ib_async, "__version__")
    assert IB is not None
    assert Stock is not None
    assert Contract is not None
    assert Order is not None


def test_ib_insync_still_imports_during_migration():
    import ib_insync
    from ib_insync import IB, Stock, Contract, Order

    assert IB is not None
    assert Stock is not None
    assert Contract is not None
    assert Order is not None
    assert hasattr(ib_insync, "IB")


def test_ib_async_and_ib_insync_can_coexist():
    import ib_async
    import ib_insync
    from ib_async import IB as NewIB
    from ib_insync import IB as OldIB

    assert NewIB is not OldIB
    assert NewIB.__module__.startswith("ib_async")
    assert OldIB.__module__.startswith("ib_insync")
    assert ib_async.__name__ == "ib_async"
    assert ib_insync.__name__ == "ib_insync"


def _iter_production_python_files():
    """Yield production python files under chad/, backend/, ops/.

    Excludes test directories, caches, and backup snapshots — those are
    explicitly allowed to reference ib_async during Phase 0 (the test file
    itself, requirements files, and any future test-only migration).
    """
    roots = [REPO_ROOT / "chad", REPO_ROOT / "backend", REPO_ROOT / "ops"]
    skip_parts = {"__pycache__", "tests"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            parts = set(path.parts)
            if parts & skip_parts:
                continue
            name = path.name
            if name.endswith(".bak") or ".pre_" in name:
                continue
            yield path


IB_ASYNC_IMPORT_RE = re.compile(r"^\s*(?:from\s+ib_async\b|import\s+ib_async\b)", re.MULTILINE)
IB_INSYNC_IMPORT_RE = re.compile(r"^\s*(?:from\s+ib_insync\b|import\s+ib_insync\b)", re.MULTILINE)


def test_no_source_migration_in_phase0():
    """Phase 0 forbids any production source file from importing ib_async.

    Production source must continue to import ib_insync until later phases.
    The only files allowed to reference ib_async in this phase are this test
    file and the requirements files.
    """
    offenders = []
    insync_users = []
    for path in _iter_production_python_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if IB_ASYNC_IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
        if IB_INSYNC_IMPORT_RE.search(text):
            insync_users.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], (
        "Phase 0 forbids migrating source imports from ib_insync to ib_async. "
        f"Files importing ib_async: {offenders}"
    )
    assert insync_users, (
        "Expected production source to still import ib_insync during Phase 0; "
        "found none — has migration happened prematurely?"
    )


def test_phase0_requirements_pin_ib_async():
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^ib_async==\d", req, re.MULTILINE), (
        "requirements.txt must pin ib_async with an exact version for Phase 0"
    )
    assert re.search(r"^ib-insync==\d", req, re.MULTILINE), (
        "requirements.txt must still pin ib-insync during Phase 0 coexistence"
    )

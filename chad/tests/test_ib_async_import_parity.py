"""GAP-007 / GAP-A019 ib_async migration — install + import-parity validation.

Phase 0 validated that ib_async can be installed and imported alongside
ib_insync. Phase 1 begins migrating low-risk source imports.

GAP-A019 Phase 1A re-classification (audit only — no production imports
were edited in this pass):

The previous classification recorded every ib_insync importer in
PHASE2_DEFERRED_FILES on the basis that every importer is on an active
broker path. A finer-grained re-audit (read-only grep of every importer)
shows three distinct usage profiles:

- EXECUTION_PATH — places, cancels, or modifies orders (placeOrder,
  cancelOrder, MarketOrder/LimitOrder/whatIfOrder). These remain
  Phase 2 and MUST NOT migrate without a dedicated execution-path
  safety harness.
- MARKET_DATA / READ_ONLY_BROKER / PORTFOLIO_RECONCILIATION — connects
  to IBKR but only reads (positions(), fills(), accountSummary(),
  reqHistoricalData, reqMktData, reqContractDetails, reqCurrentTime,
  managedAccounts, qualifyContracts). No order-affecting calls.
  Verified by grep across each file: zero matches for placeOrder,
  cancelOrder, modifyOrder, MarketOrder, LimitOrder, StopOrder,
  bracketOrder, whatIfOrder, oneCancelsAll, reqGlobalCancel, or reqIds.

The 18 read-only files are recorded as PROPOSED_PHASE1_CANDIDATES.
They are NOT migrated by this audit — PHASE2_DEFERRED_FILES is the
only enforced ledger and remains unchanged. Migrating each candidate
is a separate, single-file change with its own verification.

Files explicitly excluded from any Phase 1 candidacy regardless of
audit findings (mandated Phase 2): chad/execution/ibkr_adapter.py,
chad/core/live_loop.py, chad/core/orchestrator.py,
chad/execution/execution_pipeline.py.

The Phase 1 contract enforced here:
- No file in PHASE1_MIGRATED_FILES may import ib_insync.
- Every file in PHASE2_DEFERRED_FILES must still import ib_insync (so
  removing one is an explicit, reviewed action).
- Every production source file that imports ib_insync must appear in
  PHASE2_DEFERRED_FILES (no silent additions).
- PROPOSED_PHASE1_CANDIDATES is a documentation-only subset of
  PHASE2_DEFERRED_FILES; it carries no enforcement until a file is
  promoted into PHASE1_MIGRATED_FILES by an explicit migration commit.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Phase 1 migration ledger
# ---------------------------------------------------------------------------

# Files migrated from ib_insync to ib_async in Phase 1. Empty in this batch:
# no production source file met the strict low-risk criteria (see module
# docstring). Future Phase 1 follow-ups append paths here as new low-risk
# candidates are validated.
PHASE1_MIGRATED_FILES: tuple[str, ...] = ()

# Production source files still importing ib_insync that are explicitly
# deferred to Phase 2. Each entry is on an active broker path or in a
# disallow-listed category. Phase 2 will migrate these with dedicated
# IBKR safety tests.
PHASE2_DEFERRED_FILES: tuple[str, ...] = (
    "backend/ibkr.py",
    "chad/core/broker_position_sync.py",
    "chad/core/ibkr_healthcheck.py",
    "chad/core/live_loop.py",
    "chad/core/paper_position_closer.py",
    "chad/core/paper_shadow_runner.py",
    "chad/dashboard/api.py",
    "chad/execution/ibkr_adapter.py",
    "chad/execution/ibkr_trade_router.py",
    "chad/intel/advisory_engine.py",
    "chad/market_data/ibkr_bar_provider.py",
    "chad/market_data/ibkr_historical_provider.py",
    "chad/market_data/ibkr_price_provider.py",
    "chad/market_data/nightly_bars_refresh.py",
    "chad/market_data/options_chain_refresh.py",
    "chad/market_data/price_cache_refresh.py",
    "chad/ops/ibkr_broker_events_collector.py",
    "chad/ops/portfolio_snapshot_publisher.py",
    "chad/ops/reconciliation_publisher.py",
    "chad/options/chain_provider.py",
    "chad/portfolio/ibkr_paper_fill_harvester.py",
    "chad/portfolio/ibkr_paper_ledger_watcher.py",
    "chad/portfolio/ibkr_portfolio_collector_v2.py",
)


# GAP-A019 Phase 1A re-classification candidates. Documentation-only.
#
# These files were verified by re-audit to have NO order-affecting calls
# (no placeOrder / cancelOrder / modifyOrder / MarketOrder / LimitOrder /
# StopOrder / bracketOrder / whatIfOrder / oneCancelsAll /
# reqGlobalCancel / reqIds). They use the IBKR connection only for
# market data, account/positions/fills reads, or contract qualification.
#
# Listing here records that they are eligible for Phase 1 migration in
# subsequent single-file commits — it does NOT migrate them. They remain
# in PHASE2_DEFERRED_FILES until each is moved by an explicit migration
# commit (and added to PHASE1_MIGRATED_FILES).
#
# Files explicitly excluded from candidacy regardless of read-only status
# (mandated Phase 2 by GAP-A019 scope): chad/execution/ibkr_adapter.py,
# chad/core/live_loop.py, chad/core/orchestrator.py,
# chad/execution/execution_pipeline.py.
#
# Files NOT in this list because they are EXECUTION_PATH (place orders):
#   chad/core/paper_position_closer.py, chad/core/paper_shadow_runner.py,
#   chad/execution/ibkr_trade_router.py.
PROPOSED_PHASE1_CANDIDATES: tuple[str, ...] = (
    "backend/ibkr.py",
    "chad/core/broker_position_sync.py",
    "chad/core/ibkr_healthcheck.py",
    "chad/dashboard/api.py",
    "chad/intel/advisory_engine.py",
    "chad/market_data/ibkr_bar_provider.py",
    "chad/market_data/ibkr_historical_provider.py",
    "chad/market_data/ibkr_price_provider.py",
    "chad/market_data/nightly_bars_refresh.py",
    "chad/market_data/options_chain_refresh.py",
    "chad/market_data/price_cache_refresh.py",
    "chad/ops/ibkr_broker_events_collector.py",
    "chad/ops/portfolio_snapshot_publisher.py",
    "chad/ops/reconciliation_publisher.py",
    "chad/options/chain_provider.py",
    "chad/portfolio/ibkr_paper_fill_harvester.py",
    "chad/portfolio/ibkr_paper_ledger_watcher.py",
    "chad/portfolio/ibkr_portfolio_collector_v2.py",
)


# ---------------------------------------------------------------------------
# Library presence + coexistence
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Source-tree scanning helpers
# ---------------------------------------------------------------------------

def _iter_production_python_files():
    """Yield production python files under chad/, backend/, ops/.

    Excludes test directories, caches, and backup snapshots — those are
    explicitly allowed to reference either library (the parity test itself,
    requirements files, and any future test-only migration).
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


def _imports_ib_insync(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(IB_INSYNC_IMPORT_RE.search(text))


def _imports_ib_async(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return bool(IB_ASYNC_IMPORT_RE.search(text))


# ---------------------------------------------------------------------------
# Phase 1 invariants
# ---------------------------------------------------------------------------

def test_phase1_migrated_files_do_not_import_ib_insync():
    """Every file in PHASE1_MIGRATED_FILES must have shed ib_insync.

    A regression here means a Phase 1 file was rewritten to re-introduce
    an ib_insync import — that has to be caught immediately.
    """
    regressions: list[str] = []
    missing: list[str] = []
    for rel in PHASE1_MIGRATED_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        if _imports_ib_insync(path):
            regressions.append(rel)

    assert missing == [], (
        f"PHASE1_MIGRATED_FILES references files that no longer exist: {missing}"
    )
    assert regressions == [], (
        "Phase 1 migrated files must not re-introduce ib_insync imports. "
        f"Regressed: {regressions}"
    )


def test_phase2_high_risk_files_remain_allowlisted_on_ib_insync():
    """Every Phase 2 deferred file must still import ib_insync.

    If a file disappears from the import list it should also be removed
    from PHASE2_DEFERRED_FILES (and likely added to PHASE1_MIGRATED_FILES).
    Catching this prevents silent drift between the ledger and reality.
    """
    missing_file: list[str] = []
    no_longer_imports: list[str] = []
    for rel in PHASE2_DEFERRED_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            missing_file.append(rel)
            continue
        if not _imports_ib_insync(path):
            no_longer_imports.append(rel)

    assert missing_file == [], (
        f"PHASE2_DEFERRED_FILES references files that no longer exist: {missing_file}"
    )
    assert no_longer_imports == [], (
        "Phase 2 deferred files must still import ib_insync until their "
        "migration is reviewed and they move to PHASE1_MIGRATED_FILES. "
        f"No longer importing: {no_longer_imports}"
    )


def test_remaining_ib_insync_imports_are_explicitly_allowlisted():
    """Any production file importing ib_insync must be on an explicit list.

    The union of PHASE1_MIGRATED_FILES (which must NOT import ib_insync)
    and PHASE2_DEFERRED_FILES (which MUST import ib_insync) is the only
    allowed surface for ib_insync references in production source. New
    importers that are not added to PHASE2_DEFERRED_FILES are rejected.
    """
    deferred = {rel for rel in PHASE2_DEFERRED_FILES}
    unexpected: list[str] = []

    for path in _iter_production_python_files():
        if not _imports_ib_insync(path):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        if rel not in deferred:
            unexpected.append(rel)

    assert unexpected == [], (
        "Production files importing ib_insync must appear in "
        "PHASE2_DEFERRED_FILES (or be migrated and recorded in "
        f"PHASE1_MIGRATED_FILES). Unexpected importers: {unexpected}"
    )


def test_proposed_phase1_candidates_are_subset_of_phase2_deferred():
    """PROPOSED_PHASE1_CANDIDATES is a documentation-only audit artifact.

    Until a candidate is migrated by an explicit single-file commit,
    it must remain in PHASE2_DEFERRED_FILES. Drift here would mean the
    audit list silently became a migration ledger.
    """
    deferred = set(PHASE2_DEFERRED_FILES)
    migrated = set(PHASE1_MIGRATED_FILES)
    candidates = set(PROPOSED_PHASE1_CANDIDATES)

    not_deferred = sorted(candidates - deferred)
    leaked_into_migrated = sorted(candidates & migrated)

    assert not_deferred == [], (
        "PROPOSED_PHASE1_CANDIDATES entries must still be tracked in "
        "PHASE2_DEFERRED_FILES until they are migrated by an explicit "
        f"commit. Missing from PHASE2_DEFERRED_FILES: {not_deferred}"
    )
    assert leaked_into_migrated == [], (
        "Once a file is migrated it should be removed from "
        "PROPOSED_PHASE1_CANDIDATES (move it cleanly into "
        "PHASE1_MIGRATED_FILES). Found in both: "
        f"{leaked_into_migrated}"
    )


def test_phase0_requirements_pin_ib_async():
    req = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^ib_async==\d", req, re.MULTILINE), (
        "requirements.txt must pin ib_async with an exact version"
    )
    assert re.search(r"^ib-insync==\d", req, re.MULTILINE), (
        "requirements.txt must still pin ib-insync during coexistence"
    )

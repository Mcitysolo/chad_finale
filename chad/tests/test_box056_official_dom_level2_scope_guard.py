"""Official Matrix Box 056 — IBKR DOM / Level 2 scope guard.

Acceptance criterion (Evidence-Locked Completion Matrix v0.1):
    "DOM rows proven or formally excluded from live-readiness scope."

CHAD's documented stance (docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md
and CHAD_UNIFIED_SSOT_v9_3 §C3) is that IBKR DOM / Level 2 / market depth
is **BLOCKED / DEFERRED** pending a paid IBKR Level 2 entitlement. Until
then, CHAD must remain **DOM-free**:

  - no IBKR DOM API calls (reqMktDepth / cancelMktDepth / reqMktDepthExchanges)
  - no DOM publisher / daemon / state file
  - no strategy that consumes DOM-derived signals
  - no live_readiness gate that requires DOM availability

These guards are STATIC — they grep production source under `chad/` and
`ops/` for the canonical IBKR DOM API surface and for `live_readiness`
gate references. They are entirely offline: no IBKR connection, no network
I/O, no runtime JSON / SQLite mutation. If the entitlement is later
provisioned and a DOM consumer is built, these tests must be relaxed in
the same change that proves DOM rows are populated (see
docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md §4 unlock condition).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Production directories scanned for DOM API surface. Tests, docs, and
# archived material are excluded because they may legitimately mention
# DOM in commentary (the scoping decision itself lives in docs/).
PRODUCTION_DIRS = (
    REPO_ROOT / "chad",
    REPO_ROOT / "ops",
)

# Subdirectories to skip even inside PRODUCTION_DIRS.
SKIP_DIRS = {
    "tests",         # tests may exercise DOM mocks; out of guard scope
    "__pycache__",
    "_archive",
    "_backup_staging",
    "backups",
    ".git",
    "completion_matrix_evidence",  # under runtime/, but defensive
}

# Exact IBKR DOM / Level 2 API symbols. We grep for these as words (\b...\b)
# so substrings like "domestic" or "dominance" do not false-positive.
DOM_API_SYMBOLS = (
    "reqMktDepth",
    "cancelMktDepth",
    "reqMktDepthExchanges",
    "domBids",
    "domAsks",
    "updateMktDepth",
    "updateMktDepthL2",
)

# Live-readiness publisher path. Box-056 requires that *no* check inside
# this publisher silently depends on DOM rows / market depth.
LIVE_READINESS_PUBLISHER = REPO_ROOT / "ops" / "live_readiness_publish.py"

# Substrings forbidden inside the live-readiness publisher. These are
# narrower than DOM_API_SYMBOLS because the publisher is a single file
# we can vet word-by-word.
LIVE_READINESS_FORBIDDEN = (
    "reqMktDepth",
    "domBids",
    "domAsks",
    "orderflow_state",
    "orderflow_gate",
)


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            # Skip excluded subdirs anywhere in the path.
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            yield p


def _scan_for_symbols(files: Iterable[Path], symbols: Iterable[str]) -> List[Tuple[Path, int, str, str]]:
    """Return (path, lineno, symbol, line) for every match. Word-boundary."""
    patterns = [(s, re.compile(rf"\b{re.escape(s)}\b")) for s in symbols]
    hits: List[Tuple[Path, int, str, str]] = []
    for fp in files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for sym, pat in patterns:
                if pat.search(line):
                    hits.append((fp, lineno, sym, line.strip()))
    return hits


# ===========================================================================
# Gate 5 — no production code path silently depends on IBKR DOM
# ===========================================================================

def test_box056_no_ibkr_dom_api_in_production_code() -> None:
    """No production module under chad/ or ops/ may call the IBKR
    market-depth API surface (reqMktDepth, domBids, domAsks, etc.) while
    the IBKR Level 2 entitlement is unproven."""
    files = list(_iter_python_files(PRODUCTION_DIRS))
    assert files, "production scan must find at least one .py file"
    hits = _scan_for_symbols(files, DOM_API_SYMBOLS)
    if hits:
        formatted = "\n".join(
            f"  {fp.relative_to(REPO_ROOT)}:{ln}  [{sym}]  {snippet}"
            for fp, ln, sym, snippet in hits
        )
        raise AssertionError(
            "IBKR DOM API surface found in production code while the Level 2 "
            "entitlement remains unproven (see "
            "docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md). If the entitlement "
            "is now provisioned, relax this guard in the SAME change that proves "
            "DOM rows populate.\n" + formatted
        )


def test_box056_no_dom_dependency_in_live_readiness_publisher() -> None:
    """ops/live_readiness_publish.py must NOT condition ready_for_live on
    DOM / market-depth / orderflow availability. The publisher's gates as
    of Box-056 are: stop, feed, reconciliation, lifecycle_truth,
    execution_quality, mutation_state, canary_state, chad_mode,
    operator_intent, scr. None of those gates requires DOM."""
    assert LIVE_READINESS_PUBLISHER.exists(), (
        f"live readiness publisher missing: {LIVE_READINESS_PUBLISHER}"
    )
    text = LIVE_READINESS_PUBLISHER.read_text(encoding="utf-8", errors="ignore")
    offenders = [s for s in LIVE_READINESS_FORBIDDEN if s in text]
    assert not offenders, (
        f"live_readiness_publish.py references forbidden DOM-related symbols "
        f"{offenders!r}. Box-056 requires DOM to remain out of the live-readiness "
        f"gate set until the IBKR Level 2 entitlement is proven."
    )


# ===========================================================================
# Documentation guard — the formal scoping decision must remain on disk
# ===========================================================================

def test_box056_dom_blocker_decision_doc_present() -> None:
    """The formal DOM scoping decision (Phase-C item 3) must remain in
    docs/. Removing or moving it would orphan the Box-056 evidence."""
    doc = REPO_ROOT / "docs" / "PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md"
    assert doc.exists(), (
        f"Phase-C/C3 DOM blocker decision missing at {doc}. "
        f"This document is the canonical scoping decision for Box-056; "
        f"it must remain on disk."
    )
    text = doc.read_text(encoding="utf-8", errors="ignore")
    # Spot-check that the key blocker assertions are still present.
    for needle in ("BLOCKED / DEFERRED", "Error 354", "domBids", "domAsks"):
        assert needle in text, (
            f"Phase-C/C3 DOM blocker decision is missing expected text {needle!r}; "
            f"the document may have been edited or partially overwritten."
        )

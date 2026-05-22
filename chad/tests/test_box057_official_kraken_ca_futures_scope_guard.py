"""Official Matrix Box 057 — Kraken-CA futures scope guard.

Acceptance criterion (Evidence-Locked Completion Matrix v0.1):
    "jurisdiction issue is either solved or strategy remains spot-only."

CHAD's documented stance
(docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md) is that Kraken
Futures / perpetuals / derivatives / margin trading is **BLOCKED FOR
CURRENT CANADIAN DEPLOYMENT** (regulatory / jurisdictional, not technical).
The C1B / C1C scaffolds remain in the repository for code-archaeology but
must NOT be wired into live routing while this block stands.

These guards are STATIC — they grep production source under `chad/core/`,
`chad/strategies/`, `chad/execution/` (excluding the futures scaffold
files themselves), and `ops/` for any importer of the Kraken Futures
scaffold. They also verify that the futures client and adapter retain
their `dry_run=True` fail-closed defaults, and that the futures-intel
service is read-only (no `submit_order` / `AddOrder` / private endpoint).

If the jurisdiction block is later lifted and Kraken Futures is wired
into live routing under an eligible entity, these tests must be relaxed
in the **same change** that lands the operator's written approval and
proof of eligibility (see PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED §4).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files that *legitimately* implement / mention the dormant futures
# scaffold and must be excluded when looking for production importers.
SCAFFOLD_FILES = {
    REPO_ROOT / "chad" / "execution" / "kraken_futures_adapter.py",
    REPO_ROOT / "chad" / "exchanges" / "kraken_futures_client.py",
    REPO_ROOT / "chad" / "tools" / "kraken_futures_auth_smoke.py",
    REPO_ROOT / "chad" / "market_data" / "kraken_futures_intel_publisher.py",
}

# Directories scanned for production importers. Tests / archives skipped.
PRODUCTION_DIRS = (
    REPO_ROOT / "chad" / "core",
    REPO_ROOT / "chad" / "strategies",
    REPO_ROOT / "chad" / "execution",
    REPO_ROOT / "chad" / "risk",
    REPO_ROOT / "chad" / "ops",
    REPO_ROOT / "ops",
)

SKIP_DIRS = {
    "tests", "__pycache__", "_archive", "_backup_staging",
    "backups", ".git", "completion_matrix_evidence",
}

# Symbols that, if imported by a production module, would wire the
# dormant Kraken Futures scaffold into live order routing.
KRAKEN_FUTURES_IMPORT_SYMBOLS = (
    "KrakenFuturesAdapter",
    "KrakenFuturesClient",
    "KrakenFuturesOrderRequest",
    "KrakenFuturesOrderResult",
    "KrakenFuturesIntent",
    "kraken_futures_adapter",
    "kraken_futures_client",
)

# Files whose dry_run defaults must remain True (fail-closed).
DRY_RUN_DEFAULT_FILES = (
    REPO_ROOT / "chad" / "execution" / "kraken_futures_adapter.py",
    REPO_ROOT / "chad" / "exchanges" / "kraken_futures_client.py",
)

# Read-only intel publisher path. Box-057 requires this publisher to
# never call private order endpoints or signing.
INTEL_PUBLISHER = REPO_ROOT / "chad" / "market_data" / "kraken_futures_intel_publisher.py"

# Symbols forbidden inside the intel publisher (would mean it's not
# read-only). Each entry is a concrete API surface — bare words like
# "private" are excluded because they appear in *comments* that prohibit
# the very thing this guard is checking for.
INTEL_PUBLISHER_FORBIDDEN = (
    "submit_order",
    "AddOrder",
    "sendorder",
    "sign_request",
    "Authent",
    "/derivatives/api/v3/sendorder",
    "/derivatives/api/v3/cancelorder",
    "/derivatives/api/v3/orders",
)

# Documentation guard — the formal scoping decision must remain on disk.
DECISION_DOC = REPO_ROOT / "docs" / "PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md"
DECISION_NEEDLES = (
    "BLOCKED FOR CURRENT CANADIAN DEPLOYMENT",
    "KrakenFuturesAdapter",
    "KrakenFuturesClient",
    "Kraken Futures live trading is blocked",
)


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p in SCAFFOLD_FILES:
                continue
            yield p


def _scan_for_symbols(files: Iterable[Path], symbols: Iterable[str]) -> List[Tuple[Path, int, str, str]]:
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
# Gate 5 — no production module wires Kraken Futures into live routing
# ===========================================================================

def test_box057_no_production_importer_of_kraken_futures_scaffold() -> None:
    """No module under chad/core, chad/strategies, chad/execution,
    chad/risk, chad/ops, or ops/ may import the dormant Kraken Futures
    scaffold while the Canadian jurisdiction block stands."""
    files = list(_iter_python_files(PRODUCTION_DIRS))
    assert files, "production scan must find at least one .py file"
    hits = _scan_for_symbols(files, KRAKEN_FUTURES_IMPORT_SYMBOLS)
    if hits:
        formatted = "\n".join(
            f"  {fp.relative_to(REPO_ROOT)}:{ln}  [{sym}]  {snippet}"
            for fp, ln, sym, snippet in hits
        )
        raise AssertionError(
            "Kraken Futures scaffold imported by production code while the "
            "Canadian jurisdiction block stands (see "
            "docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md). If the "
            "block is now lifted under an eligible entity, relax this guard "
            "in the SAME change that lands the operator's written approval "
            "and the eligibility proof.\n" + formatted
        )


def test_box057_kraken_futures_dry_run_defaults_remain_true() -> None:
    """KrakenFuturesAdapter.__init__(dry_run: bool = True) and
    KrakenFuturesClient.__init__(dry_run: bool = True) must keep their
    fail-closed defaults. Flipping these defaults to False would mean any
    accidental import in production routes live."""
    for fp in DRY_RUN_DEFAULT_FILES:
        assert fp.exists(), f"required scaffold file missing: {fp}"
        text = fp.read_text(encoding="utf-8", errors="ignore")
        assert "dry_run: bool = True" in text, (
            f"{fp.relative_to(REPO_ROOT)} must keep `dry_run: bool = True` "
            f"as the fail-closed default. Found content does not contain that "
            f"exact declaration."
        )


def test_box057_intel_publisher_is_read_only() -> None:
    """The Phase-C C1A intel publisher must remain a pure public-ticker
    reader — no order submission, no private endpoint, no signing."""
    assert INTEL_PUBLISHER.exists(), (
        f"Kraken Futures intel publisher missing at {INTEL_PUBLISHER}"
    )
    text = INTEL_PUBLISHER.read_text(encoding="utf-8", errors="ignore")
    offenders = [s for s in INTEL_PUBLISHER_FORBIDDEN if s in text]
    assert not offenders, (
        f"kraken_futures_intel_publisher.py contains forbidden symbols "
        f"{offenders!r}. Box-057 requires the intel publisher to remain a "
        f"read-only public-ticker reader (no order submission, no private "
        f"endpoint, no signing)."
    )


def test_box057_jurisdiction_block_decision_doc_present() -> None:
    """The formal Kraken-CA jurisdiction block decision must remain in
    docs/. Removing or moving it would orphan the Box-057 evidence."""
    assert DECISION_DOC.exists(), (
        f"Phase-C Kraken-Futures Canadian jurisdiction block decision "
        f"missing at {DECISION_DOC}. This document is the canonical "
        f"scoping decision for Box-057; it must remain on disk."
    )
    text = DECISION_DOC.read_text(encoding="utf-8", errors="ignore")
    for needle in DECISION_NEEDLES:
        assert needle in text, (
            f"Decision document missing expected text {needle!r}; "
            f"the document may have been edited or partially overwritten."
        )

"""Official Matrix Box 058 — Coinglass external setup scope guard.

Acceptance criterion (Evidence-Locked Completion Matrix v0.1):
    "paid API/key procured or Coinglass removed from required scope."

CHAD's documented stance
(`docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md §5`,
`docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md §4`,
`docs/CHAD_UNIFIED_SSOT_v9_3_2026-05-17.md`) is that Coinglass (Phase C
Item 2) is **BLOCKED** until a paid API key is procured. Until then:

  - No publisher, adapter, or consumer may be scaffolded against
    Coinglass.
  - No public Coinglass endpoint may be used as a stand-in.
  - No live-readiness gate may require Coinglass.

CHAD's crypto-derivatives intel (funding rate, open interest, crowding
bias) is sourced from the **public Kraken Futures ticker endpoint**
(`chad/market_data/crypto_derivatives_publisher.py` and
`chad/market_data/kraken_futures_intel_publisher.py`) — a free,
keyless feed. Coinglass is therefore not required.

These guards are STATIC — they grep production source under `chad/`,
`ops/`, `scripts/`, `config/`, and `deploy/` for any reference to
Coinglass, and verify that `ops/live_readiness_publish.py` does not
condition `ready_for_live` on Coinglass. Entirely offline: no network,
no API calls, no runtime mutation.

If a paid Coinglass key is later procured, these tests must be relaxed
in the **same commit** that lands the publisher / consumer build AND
the operator's procurement decision record.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

PRODUCTION_DIRS = (
    REPO_ROOT / "chad",
    REPO_ROOT / "ops",
    REPO_ROOT / "scripts",
    REPO_ROOT / "config",
    REPO_ROOT / "deploy",
)

SKIP_DIRS = {
    "tests", "__pycache__", "_archive", "_backup_staging",
    "backups", ".git", "completion_matrix_evidence",
}

# Production source extensions to scan. Documentation (docs/) and the
# Box-058 evidence file itself are intentionally NOT scanned — they
# legitimately describe the Coinglass scoping decision.
PRODUCTION_GLOBS = ("*.py", "*.json", "*.yaml", "*.yml", "*.toml", "*.service", "*.timer", "*.sh")

# Coinglass-specific symbols. We use word-boundary regex so substrings
# like "coin" (in coingecko, coinbase, etc.) do not false-positive.
COINGLASS_SYMBOLS = (
    "coinglass",
    "Coinglass",
    "CoinGlass",
    "COINGLASS",
    "COINGLASS_API_KEY",
    "open-api.coinglass",
    "api.coinglass",
)

LIVE_READINESS_PUBLISHER = REPO_ROOT / "ops" / "live_readiness_publish.py"

DECISION_DOCS = (
    REPO_ROOT / "docs" / "CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md",
    REPO_ROOT / "docs" / "CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md",
)

DECISION_NEEDLES_BY_DOC = {
    REPO_ROOT / "docs" / "CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md": (
        "Coinglass (Phase C Item 2",
        "BLOCKED",
        "paid API plan",
    ),
    REPO_ROOT / "docs" / "CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md": (
        "C2 Coinglass status — BLOCKED",
        "No Coinglass implementation is authorized",
    ),
}


def _iter_production_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for pattern in PRODUCTION_GLOBS:
            for p in root.rglob(pattern):
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                yield p


def _scan_for_symbols(
    files: Iterable[Path], symbols: Iterable[str]
) -> List[Tuple[Path, int, str, str]]:
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
# Gate 5 — no Coinglass dependency in production
# ===========================================================================

def test_box058_no_coinglass_reference_in_production_tree() -> None:
    """No file under chad/, ops/, scripts/, config/, or deploy/ may
    reference Coinglass while the Phase-C C2 block stands. Crypto
    derivatives intel (funding, OI, crowding) must continue to be
    sourced from the public Kraken Futures ticker endpoint
    (chad/market_data/crypto_derivatives_publisher.py)."""
    files = list(_iter_production_files(PRODUCTION_DIRS))
    assert files, "production scan must find at least one file"
    hits = _scan_for_symbols(files, COINGLASS_SYMBOLS)
    if hits:
        formatted = "\n".join(
            f"  {fp.relative_to(REPO_ROOT)}:{ln}  [{sym}]  {snippet}"
            for fp, ln, sym, snippet in hits
        )
        raise AssertionError(
            "Coinglass reference found in production code/config while the "
            "C2 paid-API block stands (see "
            "docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md §5 and "
            "docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md §4). If a paid "
            "Coinglass key is now procured, relax this guard in the SAME "
            "change that lands the operator procurement decision record "
            "and the new publisher build.\n" + formatted
        )


def test_box058_live_readiness_publisher_has_no_coinglass_dependency() -> None:
    """ops/live_readiness_publish.py must NOT condition ready_for_live on
    Coinglass. The publisher's gates as of Box-058 are: stop, feed,
    reconciliation, lifecycle_truth, execution_quality, mutation_state,
    canary_state, chad_mode, operator_intent, scr. None reference
    Coinglass."""
    assert LIVE_READINESS_PUBLISHER.exists(), (
        f"live readiness publisher missing: {LIVE_READINESS_PUBLISHER}"
    )
    text = LIVE_READINESS_PUBLISHER.read_text(encoding="utf-8", errors="ignore")
    offenders = [s for s in COINGLASS_SYMBOLS if s in text]
    assert not offenders, (
        f"live_readiness_publish.py references forbidden Coinglass symbols "
        f"{offenders!r}. Box-058 requires Coinglass to remain out of the "
        f"live-readiness gate set while the paid-key block stands."
    )


def test_box058_blocker_decision_docs_present() -> None:
    """The formal Coinglass scoping decision must remain in docs/.
    Removing or moving these documents would orphan the Box-058
    evidence."""
    for doc, needles in DECISION_NEEDLES_BY_DOC.items():
        assert doc.exists(), (
            f"Coinglass scoping decision document missing at {doc}. "
            f"It is canonical for Box-058 and must remain on disk."
        )
        text = doc.read_text(encoding="utf-8", errors="ignore")
        for needle in needles:
            assert needle in text, (
                f"{doc.relative_to(REPO_ROOT)} is missing expected text "
                f"{needle!r}; the document may have been edited or "
                f"partially overwritten."
            )

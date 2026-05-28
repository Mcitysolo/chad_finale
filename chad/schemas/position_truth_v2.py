"""Schema for ``position_truth_v2.v1`` — Decision 1 / R1 closeout.

Pure dataclass schema with ``serialize_to_dict`` / ``parse_from_dict``
helpers. The engine in ``chad.core.position_truth_engine`` populates
these structures; the validator in ``chad.validators.position_truth_v2``
consumes them. No production wiring in this phase — see §12 of the
design document.

See: ``docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md`` §8.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA_VERSION = "position_truth_v2.v1"
ENGINE_VERSION = "1.0.0"
AUTHORITY_MODE = "merged_with_provenance"
DEFAULT_TTL_SECONDS = 360

# Source freshness budgets — design §14 ("Open questions for operator").
SNAPSHOT_TTL_SECONDS = 600  # 2× the 5-min poll cadence
LEDGER_TTL_SECONDS = 60     # event-driven; older than 60s means writer hung

# ``side`` vocabulary
SIDE_LONG = "LONG"
SIDE_SHORT = "SHORT"
SIDE_FLAT = "FLAT"
SIDE_UNKNOWN = "UNKNOWN"
VALID_SIDES = frozenset({SIDE_LONG, SIDE_SHORT, SIDE_FLAT, SIDE_UNKNOWN})

# ``value_source`` vocabulary
VS_BOTH = "both"
VS_SNAPSHOT = "snapshot"
VS_LEDGER = "ledger"
VS_DISAGREEMENT = "DISAGREEMENT"
VS_FAIL_CLOSED = "FAIL_CLOSED"
VALID_VALUE_SOURCES = frozenset({VS_BOTH, VS_SNAPSHOT, VS_LEDGER, VS_DISAGREEMENT, VS_FAIL_CLOSED})

# Merge-rule IDs (design §9). The engine selects one per symbol.
RULE_M1 = "M1"
RULE_M2 = "M2"
RULE_M3 = "M3"
RULE_M4 = "M4"
RULE_M5 = "M5"
VALID_MERGE_RULES = frozenset({RULE_M1, RULE_M2, RULE_M3, RULE_M4, RULE_M5})

# ``authority_decision`` vocabulary
AD_SNAPSHOT = "snapshot"
AD_LEDGER = "ledger"
AD_FAIL_CLOSED = "FAIL_CLOSED"
VALID_AUTHORITY_DECISIONS = frozenset({AD_SNAPSHOT, AD_LEDGER, AD_FAIL_CLOSED})

# ``global_authority_health`` ordered by severity (low → high).
HEALTH_GREEN = "GREEN"
HEALTH_YELLOW = "YELLOW"
HEALTH_DEGRADED = "DEGRADED"
HEALTH_RED = "RED"
HEALTH_SEVERITY: dict[str, int] = {
    HEALTH_GREEN: 0,
    HEALTH_YELLOW: 1,
    HEALTH_DEGRADED: 2,
    HEALTH_RED: 3,
}

# Per-merge-rule contribution to global health (design §10).
RULE_TO_HEALTH: dict[str, str] = {
    RULE_M1: HEALTH_GREEN,
    RULE_M2: HEALTH_YELLOW,
    RULE_M3: HEALTH_DEGRADED,
    RULE_M4: HEALTH_RED,
    RULE_M5: HEALTH_RED,
}


@dataclass
class SourceArtifact:
    """Metadata of one of the engine's input artifacts."""
    path: str
    ts_utc: str | None
    sha256: str
    age_seconds: float | None


@dataclass
class ProvenanceEntry:
    """One row of a per-symbol provenance chain."""
    surface: str          # "snapshot" | "ledger"
    ref: str              # conId for snapshot; hash key (or symbol) for ledger
    ts_utc: str | None


@dataclass
class PositionEntry:
    """Per-symbol merged truth entry (design §8)."""
    qty: float | int | None
    side: str
    value_source: str
    snapshot_value: float | int | None
    ledger_value: float | int | None
    agreement: bool
    delta: float | int | None
    delta_reason: str
    merge_rule: str
    authority_decision: str
    fail_closed: bool
    last_reconciled_utc: str
    provenance_chain: list[ProvenanceEntry] = field(default_factory=list)


@dataclass
class PositionTruthV2:
    """Top-level ``position_truth_v2.v1`` document."""
    ts_utc: str
    source_artifacts: dict[str, SourceArtifact]
    positions: dict[str, PositionEntry]
    global_authority_health: str
    fail_closed_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION
    authority_mode: str = AUTHORITY_MODE
    ttl_seconds: int = DEFAULT_TTL_SECONDS


# ---------------------------------------------------------------------------
# (de)serialization helpers
# ---------------------------------------------------------------------------

def derive_side(qty: float | int | None) -> str:
    """Derive ``LONG``/``SHORT``/``FLAT``/``UNKNOWN`` from a signed qty."""
    if qty is None:
        return SIDE_UNKNOWN
    if qty > 0:
        return SIDE_LONG
    if qty < 0:
        return SIDE_SHORT
    return SIDE_FLAT


def health_from_rules(rules: list[str]) -> str:
    """Compute ``global_authority_health`` = max severity across rules.

    Empty input ⇒ ``GREEN`` (no symbols = nothing to disagree about).
    """
    if not rules:
        return HEALTH_GREEN
    worst = HEALTH_GREEN
    worst_sev = HEALTH_SEVERITY[HEALTH_GREEN]
    for r in rules:
        h = RULE_TO_HEALTH.get(r, HEALTH_RED)
        if HEALTH_SEVERITY[h] > worst_sev:
            worst = h
            worst_sev = HEALTH_SEVERITY[h]
    return worst


def serialize_to_dict(doc: PositionTruthV2) -> dict[str, Any]:
    """Convert a ``PositionTruthV2`` into a JSON-ready dict (sorted keys
    at serialisation time are the caller's responsibility)."""
    return asdict(doc)


def parse_from_dict(raw: dict[str, Any]) -> PositionTruthV2:
    """Parse a JSON-loaded dict back into a ``PositionTruthV2`` instance.

    Raises ``ValueError`` if ``schema_version`` is missing or unsupported.
    """
    sv = raw.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise ValueError(
            f"position_truth_v2: unsupported schema_version {sv!r} "
            f"(expected {SCHEMA_VERSION!r})"
        )
    src_raw = raw.get("source_artifacts") or {}
    sources = {
        k: SourceArtifact(**v) for k, v in src_raw.items()
    }
    pos_raw = raw.get("positions") or {}
    positions: dict[str, PositionEntry] = {}
    for sym, p in pos_raw.items():
        chain_raw = p.get("provenance_chain") or []
        p_chain = [ProvenanceEntry(**c) for c in chain_raw]
        positions[sym] = PositionEntry(
            qty=p.get("qty"),
            side=p.get("side", SIDE_UNKNOWN),
            value_source=p.get("value_source", VS_FAIL_CLOSED),
            snapshot_value=p.get("snapshot_value"),
            ledger_value=p.get("ledger_value"),
            agreement=bool(p.get("agreement", False)),
            delta=p.get("delta"),
            delta_reason=p.get("delta_reason", ""),
            merge_rule=p.get("merge_rule", RULE_M5),
            authority_decision=p.get("authority_decision", AD_FAIL_CLOSED),
            fail_closed=bool(p.get("fail_closed", True)),
            last_reconciled_utc=p.get("last_reconciled_utc", ""),
            provenance_chain=p_chain,
        )
    return PositionTruthV2(
        ts_utc=raw.get("ts_utc", ""),
        source_artifacts=sources,
        positions=positions,
        global_authority_health=raw.get("global_authority_health", HEALTH_RED),
        fail_closed_symbols=list(raw.get("fail_closed_symbols") or []),
        warnings=list(raw.get("warnings") or []),
        errors=list(raw.get("errors") or []),
        schema_version=sv,
        engine_version=raw.get("engine_version", ENGINE_VERSION),
        authority_mode=raw.get("authority_mode", AUTHORITY_MODE),
        ttl_seconds=int(raw.get("ttl_seconds", DEFAULT_TTL_SECONDS)),
    )

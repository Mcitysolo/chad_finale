"""chad.validation — the offline edge-validation harness (isolated package).

Hard isolation guarantee (SSOT §2): nothing in this package may import the live
trading loop, broker adapters, or any ``runtime/`` reader, and nothing in the
live trading path may import this package. Phase 0 (``bar_audit``) is the
forensic data-quality gate: no bar is backtested until it has been audited.

Public API is re-exported here so later phases embed the data-quality section by
importing from :mod:`chad.validation` rather than reaching into submodules.
"""

from __future__ import annotations

from chad.validation.bar_audit import (
    AuditConfig,
    CorpusAudit,
    Finding,
    Status,
    SymbolAudit,
    audit_bar_file,
    audit_corpus,
    audit_symbol,
    render_corpus_summary,
    render_symbol_audit,
)

__all__ = [
    "AuditConfig",
    "CorpusAudit",
    "Finding",
    "Status",
    "SymbolAudit",
    "audit_bar_file",
    "audit_corpus",
    "audit_symbol",
    "render_corpus_summary",
    "render_symbol_audit",
]

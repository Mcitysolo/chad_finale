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
from chad.validation.cost_model import (
    CostBreakdown,
    CostConfig,
    DEFAULT_COST_CONFIG,
    InstrumentClass,
    IntrabarResolution,
    LiquidityTier,
    Trade,
    apply_costs,
    resolve_intrabar,
)
from chad.validation.regime_labeler import (
    DEFAULT_REGIME_CONFIG,
    Regime,
    RegimeConfig,
    RegimeLabel,
    RegimeSeries,
    label_series,
)
from chad.validation.scoring_spine import (
    DEFAULT_PERIODS_PER_YEAR,
    ScoreResult,
    equity_to_returns,
    score_equity,
    score_returns,
    score_trades,
)
from chad.validation.splits import (
    Partition,
    WalkForwardWindow,
    generate_walk_forward,
    partition,
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
    # Phase 1 — shared scoring spine (SSOT §1.3).
    "DEFAULT_PERIODS_PER_YEAR",
    "ScoreResult",
    "equity_to_returns",
    "score_equity",
    "score_returns",
    "score_trades",
    # Phase 2 — cost model + pessimistic execution (SSOT §3.5).
    "CostBreakdown",
    "CostConfig",
    "DEFAULT_COST_CONFIG",
    "InstrumentClass",
    "IntrabarResolution",
    "LiquidityTier",
    "Trade",
    "apply_costs",
    "resolve_intrabar",
    # Phase 2 — splits + purged/embargoed walk-forward (SSOT §3.7).
    "Partition",
    "WalkForwardWindow",
    "generate_walk_forward",
    "partition",
    # Phase 2 — independent regime labeler (SSOT §3.4).
    "DEFAULT_REGIME_CONFIG",
    "Regime",
    "RegimeConfig",
    "RegimeLabel",
    "RegimeSeries",
    "label_series",
]

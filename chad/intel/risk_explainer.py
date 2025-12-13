from __future__ import annotations

"""
CHAD Phase 10 – Risk Explainer

This module turns CHAD’s internal risk state (SCR, CHAD_MODE, caps, reasons)
into a structured RiskExplanation object PLUS a concise human-friendly
summary string for the Coach / API layer.

STRICT GUARANTEES:
- NEVER adjusts risk.
- NEVER flips CHAD_MODE or execution mode.
- READ-ONLY: consumes JSON from existing runtime files and analytics.
- Returns advisory explanations only.

Used by:
- Telegram Coach (/ai_risk_now)
- Status API (/ai/risk_explain)
- Internal reporting jobs (Phase 10)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from chad.intel.schemas import RiskExplanation
from chad.analytics.trade_stats_engine import load_and_compute
from chad.analytics.shadow_confidence_router import evaluate_confidence
from chad.core.mode import get_chad_mode, is_live_mode_enabled

# runtime paths used by orchestrator + collectors
DYNAMIC_CAPS_PATH = Path("/home/ubuntu/CHAD FINALE/runtime/dynamic_caps.json")
PORTFOLIO_SNAPSHOT_PATH = Path("/home/ubuntu/CHAD FINALE/runtime/portfolio_snapshot.json")


def _load_dynamic_caps() -> Optional[Dict[str, Any]]:
    """Load dynamic caps JSON. Return dict or None."""
    if not DYNAMIC_CAPS_PATH.is_file():
        return None
    try:
        return json.loads(DYNAMIC_CAPS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_portfolio_equity() -> Optional[float]:
    """Total equity for explanation context."""
    if not PORTFOLIO_SNAPSHOT_PATH.is_file():
        return None
    try:
        snap = json.loads(PORTFOLIO_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        return float(
            (snap.get("ibkr_equity") or 0.0)
            + (snap.get("kraken_equity") or 0.0)
            + (snap.get("coinbase_equity") or 0.0)
        )
    except Exception:
        return None


def build_risk_explanation() -> RiskExplanation:
    """
    Produce a typed RiskExplanation object summarizing:

    - CHAD_MODE
    - live_enabled flag
    - SCR state + reasons + sizing_factor
    - total_equity (if available)
    
    This is READ-ONLY and has NO trading effects.
    """

    # 1. Mode
    mode = get_chad_mode()
    live_flag = is_live_mode_enabled()

    # 2. SCR (Shadow Confidence Router)
    stats = load_and_compute(
        max_trades=500,
        days_back=30,
        include_paper=True,
        include_live=True,
    )
    scr = evaluate_confidence(stats)

    # 3. Caps + Equity context
    dyn = _load_dynamic_caps()
    total_equity = _load_portfolio_equity()

    explanation = RiskExplanation(
        chad_mode=str(mode.value),
        live_enabled=bool(live_flag),
        scr_state=str(scr.state),
        reasons=list(scr.reasons),
        sizing_factor=float(scr.sizing_factor),
        total_equity=total_equity,
        risk_notes=None,
    )

    return explanation


def render_risk_summary_text(exp: RiskExplanation) -> str:
    """
    Convert a RiskExplanation object into a concise readable summary.

    This is what Telegram Coach replies for /ai_risk_now,
    and what the API endpoint returns for human text consumption.
    """

    lines = []
    lines.append("CHAD Risk Summary")
    lines.append("------------------")
    lines.append(f"Mode: {exp.chad_mode}  (live_enabled={exp.live_enabled})")
    lines.append(f"SCR State: {exp.scr_state}")
    lines.append(f"Sizing Factor: {exp.sizing_factor:.2f}")

    if exp.total_equity is not None:
        lines.append(f"Total Equity: {exp.total_equity:,.2f} USD")

    if exp.reasons:
        lines.append("")
        lines.append("Reasons:")
        for r in exp.reasons:
            lines.append(f"  • {r}")

    if exp.risk_notes:
        lines.append("")
        lines.append("Notes:")
        lines.append(f"{exp.risk_notes}")

    return "\n".join(lines)

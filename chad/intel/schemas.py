from __future__ import annotations

"""
CHAD Phase 10 — Intelligence Layer Schemas

These Pydantic models define STRICT, typed schemas for all GPT-driven
analytical outputs used by the new Global Intelligence Layer:

- ResearchScenario:     Narrative & scenario analysis per symbol/sector.
- ModelDoctorReport:    Feature ideas, hyperparams, blacklists for ML.
- MacroImpactAssessment:Regime classification + cross-strategy impact.
- LessonsLearned:       Structured post-trade “teacher” insights.
- RiskExplanation:      Human-friendly risk & SCR explanations.

JSON coming from GPT must ALWAYS match these schemas before CHAD consumes it.

All schemas are:
- Advisory-only
- Non-executable
- Immutable inputs for downstream logic
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


# ============================================================================ #
# Research Scenario (per symbol)
# ============================================================================ #
class ResearchRequestInput(BaseModel):
    """
    Input schema for requesting research via API or CLI.

    This is what /ai/research should accept from HTTP clients and what
    research_engine.run_research_scenario_from_request will consume.
    """

    symbol: str
    # 1w = 1 week, 1m = 1 month, 3m = 3 months
    scenario_timeframe: str = Field(pattern=r"^(1w|1m|3m)$")
    question: str

    @validator("symbol")
    def symbol_upper(cls, v: str) -> str:
        return v.upper().strip()

class ResearchScenario(BaseModel):
    """Structured research analysis for a single symbol."""

    symbol: str
    macro_risks: List[str] = Field(default_factory=list)
    bull_case: str
    bear_case: str
    base_case: str

    # Pydantic v2 uses `pattern=` instead of `regex=`
    scenario_timeframe: str = Field(pattern=r"^(1w|1m|3m)$")

    test_ideas: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)

    @validator("symbol")
    def symbol_upper(cls, v: str) -> str:
        return v.upper().strip()


# ============================================================================ #
# Model Doctor — ML Feature Engineering Advisor
# ============================================================================ #

class ModelDoctorReport(BaseModel):
    """GPT's advisory output for improving CHAD's ML models."""

    new_feature_ideas: List[str] = Field(default_factory=list)
    symbol_blacklists_suggestions: List[str] = Field(default_factory=list)
    hyperparam_tuning_ideas: List[str] = Field(default_factory=list)
    notes: Optional[str] = ""


# ============================================================================ #
# Macro / News Narrative — Cross-Strategy Impact Analysis
# ============================================================================ #

class StrategyImpact(BaseModel):
    strategy: str
    impact: str = Field(pattern=r"^(positive|negative|neutral)$")


class SuggestedAction(BaseModel):
    strategy: str
    action: str = Field(pattern=r"^(reduce_size|increase_size|increase_hedge|reduce_hedge|hold)$")
    factor: float = Field(gt=0.0, le=5.0)


class MacroImpactAssessment(BaseModel):
    """Narrative mapping of macro regime → strategy impacts."""

    regime: str
    impact_assessment: List[StrategyImpact] = Field(default_factory=list)
    suggested_actions: List[SuggestedAction] = Field(default_factory=list)
    notes: Optional[str] = ""


# ============================================================================ #
# Lessons Learned — Trade Post-Mortem
# ============================================================================ #

class LessonItem(BaseModel):
    issue: str
    example: Optional[str] = ""
    recommendation: Optional[str] = ""


class LessonsLearned(BaseModel):
    """GPT-generated lessons from recent trades."""

    date_range: str
    top_patterns: List[str] = Field(default_factory=list)
    lessons: List[LessonItem] = Field(default_factory=list)
    summary_markdown: Optional[str] = ""


# ============================================================================ #
# Risk Explanation — Human-Friendly SCR / Mode / Risk Commentary
# ============================================================================ #

class RiskExplanation(BaseModel):
    """Human explanation for CHAD’s risk posture."""

    chad_mode: str
    live_enabled: bool
    scr_state: str
    reasons: List[str] = Field(default_factory=list)
    sizing_factor: float
    total_equity: Optional[float] = None
    risk_notes: Optional[str] = ""


# ============================================================================ #
# Generic wrapper for misc GPT outputs
# ============================================================================ #

class GenericGPTOutput(BaseModel):
    """Fallback for GPT responses not matching a defined schema."""
    data: Dict[str, Any] = Field(default_factory=dict)

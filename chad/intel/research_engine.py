from __future__ import annotations

"""
CHAD Phase 10 — Research & Scenario Engine

This module produces STRUCTURED, ADVISORY-ONLY research reports for a symbol.
It uses GPTClient to generate JSON matching the ResearchScenario schema.

Guarantees
----------
- NEVER touches execution, SCR, brokers, or mode.
- Advisory-only: writes outputs into reports/research/.
- Validates ALL GPT responses via ResearchScenario before saving.
- Logs all interactions for audit.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from chad.intel.gpt_client import GPTClient, GPTClientError
from chad.intel.schemas import (
    ResearchScenario,
    ResearchRequestInput,
)


# ------------------------------------------------------------------------------
# Constants & paths
# ------------------------------------------------------------------------------

REPORTS_DIR = Path("/home/ubuntu/CHAD FINALE/reports/research")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are CHAD's Global Intelligence Layer — a senior analyst providing
structured, advisory-only research.

CRITICAL RULES:
- NEVER recommend adjusting risk limits directly.
- ONLY produce a JSON object matching the provided schema.
- Keep reasoning concise, factual, and actionable.
- NO disclaimers, no meta-talk, no apologies.

Your purpose:
- Summarize narrative macro risk for the symbol.
- Provide bull/bear/base case scenarios.
- Suggest testable ideas for CHAD's engineers.
- Output must validate under the ResearchScenario Pydantic schema.

JSON ONLY.
"""

USER_PROMPT_TEMPLATE = """
Symbol: {symbol}

Context Summary:
{context}

Price/Volume Summary (high-level):
{price_summary}

Now produce a structured JSON object with:
- macro_risks:   list[str]
- bull_case:     string
- bear_case:     string
- base_case:     string
- scenario_timeframe: "1w" | "1m" | "3m"
- test_ideas:    list[str]
- confidence_score: float in [0.0, 1.0]

JSON ONLY.
"""


# ------------------------------------------------------------------------------
# Core: the original research engine entry point
# ------------------------------------------------------------------------------

def run_research_scenario(
    symbol: str,
    *,
    context: str = "",
    price_summary: Optional[Dict[str, Any]] = None,
    client: Optional[GPTClient] = None,
    save: bool = True,
) -> ResearchScenario:
    """
    Generate a ResearchScenario for a symbol using GPT.

    Args:
        symbol:          Ticker symbol, e.g., "NVDA"
        context:         Narrative or notes (optional)
        price_summary:   Dict with high-level OHLC, vol, etc. (optional)
        client:          Optional GPTClient instance; will create one if None.
        save:            If True, write report to reports/research/.

    Returns:
        ResearchScenario instance.

    Raises:
        GPTClientError, GPTAPIError, GPTConfigError, ValueError
    """
    if client is None:
        client = GPTClient()  # loads config from /etc/chad/openai.env

    symbol = symbol.upper().strip()
    if not symbol:
        raise ValueError("symbol must be a non-empty string")

    price_summary_str = json.dumps(price_summary or {}, indent=2)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        symbol=symbol,
        context=context or "No additional context provided.",
        price_summary=price_summary_str,
    )

    # ----------------------------------------------------------------------
    # Call GPT
    # ----------------------------------------------------------------------
    raw_json = client.chat_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_output_tokens=1024,
        extra_context={"role": "research_scenario"},
    )

    if not isinstance(raw_json, dict):
        raise ValueError(
            f"GPT research response is not a JSON object, got type={type(raw_json).__name__}"
        )

    # ----------------------------------------------------------------------
    # Harden the payload before Pydantic validation
    # ----------------------------------------------------------------------
    # 1) Ensure symbol is present and normalised
    if "symbol" not in raw_json or not isinstance(raw_json.get("symbol"), str) or not raw_json["symbol"].strip():
        raw_json["symbol"] = symbol

    # 2) Ensure scenario_timeframe is present and valid; if missing/invalid,
    #    default to "1m" which matches the ResearchScenario pattern.
    stf = raw_json.get("scenario_timeframe")
    allowed_timeframes = {"1w", "1m", "3m"}
    if not isinstance(stf, str) or stf not in allowed_timeframes:
        raw_json["scenario_timeframe"] = "1m"

    # ----------------------------------------------------------------------
    # Validate using Pydantic
    # ----------------------------------------------------------------------
    try:
        scenario = ResearchScenario(**raw_json)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"GPT output did not match ResearchScenario: {exc}") from exc

    # ----------------------------------------------------------------------
    # Save result
    # ----------------------------------------------------------------------
    if save:
        utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = REPORTS_DIR / f"RESEARCH_{symbol}_{utc}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(scenario.model_dump(), f, indent=2)  # Pydantic v2 compatible
        log_line = f"[RESEARCH_ENGINE] Saved research scenario for {symbol} -> {out_path}"
        print(log_line)

    return scenario


# ------------------------------------------------------------------------------
# NEW: Wrapper for API usage — accepts ResearchRequestInput, not raw fields
# ------------------------------------------------------------------------------

def run_research_scenario_from_request(
    request: ResearchRequestInput,
    *,
    client: Optional[GPTClient] = None,
    save: bool = True,
) -> ResearchScenario:
    """
    Wrapper used by the /ai/research API route.

    Converts the input fields:
      - symbol
      - scenario_timeframe
      - question

    Into a structured context string, then delegates to run_research_scenario().
    """

    context_parts = [
        f"Requested timeframe: {request.scenario_timeframe}",
        "",
        "User question:",
        request.question,
    ]
    context = "\n".join(context_parts)

    return run_research_scenario(
        symbol=request.symbol,
        context=context,
        price_summary=None,
        client=client,
        save=save,
    )

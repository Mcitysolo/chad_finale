from __future__ import annotations

"""
CHAD Phase 9 — Strategy Intelligence Engine Functions

Advisory engine entry points for Claude-powered strategy intelligence.
These are called by the EngineRunner via DEFAULT_ENGINE_SPECS.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

LOG = logging.getLogger("chad.intel.strategy_intelligence_engine")


def _get_client_and_si():
    """Lazy-load ClaudeClient and StrategyIntelligence."""
    from chad.intel.claude_client import ClaudeClient
    from chad.intel.strategy_intelligence import StrategyIntelligence

    client = ClaudeClient.load()
    runtime_dir = Path("/home/ubuntu/chad_finale/runtime")
    return client, StrategyIntelligence(client, runtime_dir)


def run_strategy_intelligence(symbol: str) -> Dict[str, Any]:
    """Get confidence bias report for a symbol."""
    try:
        client, si = _get_client_and_si()
        bias = si.get_confidence_bias(
            symbol=symbol,
            strategy_name="advisory",
            base_confidence=0.5,
        )
        return {
            "symbol": bias.symbol,
            "adjustment": bias.adjustment,
            "reason": bias.reason,
            "macro_risk": bias.macro_risk,
            "regime": bias.regime,
            "ts_utc": bias.ts_utc,
        }
    except Exception as exc:
        return {"error": str(exc), "adjustment": 0.0, "reason": "intelligence_unavailable"}


def run_regime_classifier(symbol: str) -> Dict[str, Any]:
    """Classify current market regime with reasoning."""
    try:
        client, si = _get_client_and_si()
        profile = si.get_regime_profile("advisory")
        market_ctx = si._load_market_context()
        macro = market_ctx.get("macro_state", {})
        return {
            "profile": profile,
            "vix": macro.get("vix", macro.get("vix_close", "unknown")),
            "risk_label": macro.get("risk_label", "unknown"),
            "regime_source": "claude_intelligence",
        }
    except Exception as exc:
        return {"error": str(exc), "profile": "normal", "regime_source": "fallback"}


def run_cross_strategy_correlation(symbol: str) -> Dict[str, Any]:
    """Detect when strategies are over-correlated."""
    try:
        client, si = _get_client_and_si()

        # Read recent signals from position guard
        runtime_dir = Path("/home/ubuntu/chad_finale/runtime")
        guard_path = runtime_dir / "position_guard.json"
        guard_data = {}
        if guard_path.exists():
            try:
                guard_data = json.loads(guard_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        open_positions = {
            k: v for k, v in guard_data.items()
            if isinstance(v, dict) and v.get("open")
        }

        # Count strategies per symbol
        symbol_strategies: Dict[str, list] = {}
        for key, entry in open_positions.items():
            sym = entry.get("symbol", "")
            strat = entry.get("strategy", "unknown")
            if sym:
                symbol_strategies.setdefault(sym, []).append(strat)

        over_correlated = {
            sym: strats for sym, strats in symbol_strategies.items()
            if len(strats) > 1
        }

        return {
            "open_position_count": len(open_positions),
            "symbols_with_multiple_strategies": over_correlated,
            "correlation_risk": "high" if over_correlated else "low",
            "source": "position_guard_analysis",
        }
    except Exception as exc:
        return {"error": str(exc), "correlation_risk": "unknown"}

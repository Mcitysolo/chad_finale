"""
chad/risk/dynamic_caps.py

Lightweight, broker-agnostic risk-cap helpers used by KrakenExecutor (and any
future non-IBKR executor that needs a simple per-strategy notional cap check
against runtime/dynamic_caps.json).

The IBKR path uses a richer check_risk in chad.execution.ibkr_executor that
also tracks IBKR-specific quantity normalization. This module intentionally
provides a minimal subset:

  load_dynamic_caps(path)        -> dict
  check_risk(caps_data, intent)  -> RiskCheckResult

The intent is duck-typed: we only require .strategy and .notional_estimate.

Strategy key resolution
-----------------------
For crypto intents we accept either of the following keys in
strategy_caps (in order of preference):
  1. The intent's exact strategy name (e.g. 'alpha_crypto')
  2. The generic 'crypto' key (current canonical CHAD key for crypto cap)
This keeps the executor working whether dynamic_caps.json names the cap
'crypto' (current state) or 'alpha_crypto' (future state).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class RiskCheckResult:
    allowed: bool
    reason: str
    adjusted_notional: float


def load_dynamic_caps(path: Path) -> Dict[str, Any]:
    """
    Load dynamic_caps.json. Returns an empty dict if the file is missing,
    so callers fail closed via the cap-not-found path.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_strategy_cap(caps_data: Dict[str, Any], strategy: str) -> float | None:
    strategy_caps = caps_data.get("strategy_caps", {}) if isinstance(caps_data, dict) else {}
    if not isinstance(strategy_caps, dict):
        return None

    candidates = []
    if strategy:
        candidates.append(str(strategy))
    # Crypto fallback chain: any 'alpha_crypto'-style intent should also
    # accept the generic 'crypto' cap.
    if str(strategy).startswith("alpha_crypto") or str(strategy).startswith("crypto"):
        candidates.append("crypto")
        candidates.append("alpha_crypto")

    for key in candidates:
        if key in strategy_caps:
            v = strategy_caps[key]
            if isinstance(v, dict):
                v = v.get("cap_usd") or v.get("notional_cap") or v.get("max_notional")
            try:
                fv = float(v)
                if fv > 0:
                    return fv
            except (TypeError, ValueError):
                continue
    return None


def check_risk(*, caps_data: Dict[str, Any], intent: Any) -> RiskCheckResult:
    """
    Minimal per-strategy notional cap check.

    - Looks up the strategy cap in caps_data['strategy_caps'].
    - If the strategy is not found, fails closed (allowed=False).
    - If requested notional exceeds the cap, blocks the trade.
    """
    strategy = str(getattr(intent, "strategy", "") or "")
    requested = float(getattr(intent, "notional_estimate", 0.0) or 0.0)

    if requested <= 0:
        return RiskCheckResult(
            allowed=False,
            reason="non_positive_notional",
            adjusted_notional=0.0,
        )

    cap = _resolve_strategy_cap(caps_data, strategy)
    if cap is None:
        return RiskCheckResult(
            allowed=False,
            reason=f"strategy_cap_not_found:{strategy}",
            adjusted_notional=0.0,
        )

    if requested <= cap:
        return RiskCheckResult(
            allowed=True,
            reason=f"within_cap:requested={requested:.2f}<=cap={cap:.2f}",
            adjusted_notional=requested,
        )

    return RiskCheckResult(
        allowed=False,
        reason=f"exceeds_cap:requested={requested:.2f}>cap={cap:.2f}",
        adjusted_notional=cap,
    )

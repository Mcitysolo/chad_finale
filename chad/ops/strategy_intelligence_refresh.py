#!/usr/bin/env python3
"""
CHAD Strategy Intelligence Refresh — scheduled CLI wrapper.

Refreshes runtime/strategy_intelligence.json and the bias cache at
runtime/strategy_intelligence_cache.json by invoking the existing
StrategyIntelligence engine across a small universe.

Fail-soft: writes a neutral stub if the Claude client is unavailable,
so downstream consumers (live_loop _apply_intelligence_bias) never see
an empty or missing cache.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("chad.ops.strategy_intelligence_refresh")

RUNTIME_DIR = Path("/home/ubuntu/chad_finale/runtime")
OUTPUT_PATH = RUNTIME_DIR / "strategy_intelligence.json"
CACHE_PATH = RUNTIME_DIR / "strategy_intelligence_cache.json"

UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "AMD", "META", "BTC"]
STRATEGIES = ["alpha", "gamma", "alpha_futures", "gamma_futures", "gamma_reversion"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_neutral_stub(reason: str) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "confidence": {},
        "regime": {"advisory": {"profile": "normal", "reasoning": reason, "ts_utc": _utc_now_iso()}},
        "last_updated_utc": _utc_now_iso(),
        "stub_reason": reason,
    }
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        from chad.intel.claude_client import ClaudeClient
        from chad.intel.strategy_intelligence import StrategyIntelligence
    except Exception as exc:
        LOG.warning("Imports failed, writing neutral stub: %s", exc)
        _write_neutral_stub(f"import_error:{exc}")
        return 0

    try:
        client = ClaudeClient.load()
    except Exception as exc:
        LOG.warning("Claude client unavailable, writing neutral stub: %s", exc)
        _write_neutral_stub(f"claude_unavailable:{exc}")
        return 0

    si = StrategyIntelligence(client, RUNTIME_DIR)

    ok = 0
    err = 0
    for strategy in STRATEGIES:
        try:
            si.get_regime_profile(strategy)
            ok += 1
        except Exception as exc:
            LOG.warning("regime_profile(%s) failed: %s", strategy, exc)
            err += 1

    for sym in UNIVERSE:
        try:
            si.get_confidence_bias(symbol=sym, strategy_name="alpha", base_confidence=0.5)
            ok += 1
        except Exception as exc:
            LOG.warning("confidence_bias(%s) failed: %s", sym, exc)
            err += 1

    LOG.info("strategy_intelligence refresh done ok=%d err=%d", ok, err)
    return 0


if __name__ == "__main__":
    sys.exit(main())

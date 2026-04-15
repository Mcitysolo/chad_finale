#!/usr/bin/env python3
"""
CHAD Pre-Market Advisory Refresh

Runs once daily at 8:45 AM ET (12:45 UTC) — 45 minutes before US equity open —
to refresh strategy_intelligence.json with fresh confidence biases and regime
classification for the trading day.

Fail-soft: never raises. Writes a summary line to reports/ops/journal_*.log
so the morning brief has fresh intelligence to read.

CLI:
    python -m chad.ops.advisory_pre_market
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("chad.ops.advisory_pre_market")

REPO_ROOT = Path("/home/ubuntu/chad_finale")
RUNTIME_DIR = REPO_ROOT / "runtime"
REPORTS_DIR = REPO_ROOT / "reports" / "ops"

UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA", "AMD", "META", "BTC"]
STRATEGIES = ["alpha", "gamma", "alpha_futures", "gamma_futures", "gamma_reversion"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _journal(summary: str) -> None:
    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = REPORTS_DIR / f"advisory_pre_market_{day}.log"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{_utc_now_iso()} {summary}\n")
    except Exception as exc:
        LOG.warning("journal write failed: %s", exc)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOG.info("advisory pre-market refresh starting")

    try:
        from chad.intel.claude_client import ClaudeClient
        from chad.intel.strategy_intelligence import StrategyIntelligence
    except Exception as exc:
        LOG.warning("imports failed: %s", exc)
        _journal(f"SKIP import_error={exc}")
        return 0

    try:
        client = ClaudeClient.load()
    except Exception as exc:
        LOG.warning("claude client unavailable: %s", exc)
        _journal(f"SKIP claude_unavailable={exc}")
        return 0

    try:
        si = StrategyIntelligence(client, RUNTIME_DIR)
    except Exception as exc:
        LOG.warning("StrategyIntelligence init failed: %s", exc)
        _journal(f"SKIP init_error={exc}")
        return 0

    regime_ok = 0
    regime_err = 0
    bias_ok = 0
    bias_err = 0

    for strategy in STRATEGIES:
        try:
            si.get_regime_profile(strategy)
            regime_ok += 1
        except Exception as exc:
            LOG.warning("regime_profile(%s) failed: %s", strategy, exc)
            regime_err += 1

    for sym in UNIVERSE:
        try:
            si.get_confidence_bias(symbol=sym, strategy_name="alpha", base_confidence=0.5)
            bias_ok += 1
        except Exception as exc:
            LOG.warning("confidence_bias(%s) failed: %s", sym, exc)
            bias_err += 1

    summary = (
        f"DONE regimes={regime_ok}/{regime_ok + regime_err} "
        f"biases={bias_ok}/{bias_ok + bias_err}"
    )
    LOG.info(summary)
    _journal(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

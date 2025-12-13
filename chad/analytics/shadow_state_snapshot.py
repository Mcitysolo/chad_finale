"""
Shadow State Snapshot (Phase 5 – Confidence Snapshot for Coach / Status)

This module computes the current Shadow Confidence Router (SCR) state from
recent trade history and writes a compact JSON snapshot to disk.

Snapshot path:
    data/shadow/shadow_state.json

Snapshot schema (example):
    {
      "timestamp_utc": "2025-12-01T23:40:00.123456+00:00",
      "state": "WARMUP",
      "sizing_factor": 0.1,
      "paper_only": true,
      "reasons": [
        "Warmup: only 1 trades (< 50 required).",
        "Current win_rate=1.000, sharpe_like=0.000, max_drawdown=0.00, total_pnl=5.25"
      ],
      "stats": {
        "total_trades": 1,
        "winners": 1,
        "losers": 0,
        "win_rate": 1.0,
        "avg_pnl": 5.25,
        "total_pnl": 5.25,
        "sharpe_like": 0.0,
        "max_drawdown": 0.0,
        "live_trades": 0,
        "paper_trades": 1,
        "per_strategy": {
          "beta": {
            "total_trades": 1,
            "winners": 1,
            "losers": 0,
            "win_rate": 1.0,
            "avg_pnl": 5.25,
            "total_pnl": 5.25,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0
          }
        }
      }
    }

Intended usage
--------------
* Called periodically by a systemd timer or cron job to refresh the snapshot.
* Called on-demand by the Telegram coach or status HTTP endpoint.

CLI:
    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.analytics.shadow_state_snapshot
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from chad.analytics.trade_stats_engine import load_and_compute
from chad.analytics.shadow_confidence_router import evaluate_confidence, ShadowState


LOGGER_NAME = "chad.shadow_state_snapshot"
SHADOW_DIR = Path("data") / "shadow"
SHADOW_STATE_PATH = SHADOW_DIR / "shadow_state.json"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _shadow_state_to_dict(shadow_state: ShadowState) -> Dict[str, Any]:
    """
    Convert a ShadowState into a JSON-serializable dict.

    We rely on dataclass asdict for most fields, but ensure all nested
    objects are JSON-friendly.
    """
    raw = {
        "state": shadow_state.state,
        "sizing_factor": float(shadow_state.sizing_factor),
        "paper_only": bool(shadow_state.paper_only),
        "reasons": list(shadow_state.reasons),
        "stats": shadow_state.stats,
    }
    return raw


def write_shadow_snapshot() -> Path:
    """
    Compute stats + ShadowState and write a JSON snapshot to disk.

    Returns:
        Path to the snapshot file.
    """
    logger = _get_logger()
    _ensure_directory(SHADOW_DIR)

    logger.info("Loading recent trade stats for shadow state snapshot...")
    stats = load_and_compute(
        max_trades=200,
        days_back=30,
        include_paper=True,
        include_live=True,
    )

    shadow_state = evaluate_confidence(stats)

    logger.info(
        "ShadowState snapshot: state=%s sizing_factor=%.3f paper_only=%s",
        shadow_state.state,
        shadow_state.sizing_factor,
        shadow_state.paper_only,
    )

    snapshot: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "state": shadow_state.state,
        "sizing_factor": float(shadow_state.sizing_factor),
        "paper_only": bool(shadow_state.paper_only),
        "reasons": list(shadow_state.reasons),
        "stats": stats,
    }

    # Write atomically via temp file + rename.
    tmp_path = SHADOW_STATE_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")

    tmp_path.replace(SHADOW_STATE_PATH)

    logger.info("Shadow state snapshot written to: %s", SHADOW_STATE_PATH)
    return SHADOW_STATE_PATH


def main() -> None:
    """
    CLI entrypoint for:

        python -m chad.analytics.shadow_state_snapshot
    """
    try:
        path = write_shadow_snapshot()
        print(f"SHADOW SNAPSHOT WRITTEN: {path}")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger = _get_logger()
        logger.exception("Failed to write shadow state snapshot: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

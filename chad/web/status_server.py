"""
CHAD Status Server (Phase 6/7 – Read-Only HTTP Status + Shadow State + Mode)

This module exposes a minimal HTTP /status endpoint that reports:

    * Basic service health
    * Current UTC time
    * Global mode (CHAD_MODE):
        - DRY_RUN or LIVE (normalized)
        - live_enabled flag
    * Shadow Confidence state (if snapshot exists), including:
        - state
        - sizing_factor
        - paper_only
        - reasons
        - key stats (total_trades, win_rate, total_pnl)

It is STRICTLY READ-ONLY – it does NOT execute trades, modify config,
or change system mode.

Usage (dev):

    PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.web.status_server

By default it listens on 0.0.0.0:9618, but you can override via env:

    CHAD_STATUS_HOST
    CHAD_STATUS_PORT
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, Response

from chad.analytics.shadow_state_snapshot import SHADOW_STATE_PATH
from chad.core.mode import get_chad_mode, is_live_mode_enabled


LOGGER_NAME = "chad.status_server"
logger = logging.getLogger(LOGGER_NAME)


def _get_logger() -> logging.Logger:
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def _load_shadow_snapshot(path: Path) -> Dict[str, Any]:
    """
    Load the shadow_state.json snapshot if it exists.

    If not present or invalid, return a stub dict indicating unavailability.
    """
    if not path.exists():
        return {
            "available": False,
            "reason": f"snapshot file not found: {str(path)}",
        }

    try:
        with path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"error reading snapshot: {exc}",
        }

    # Extract a compact view for status.
    state = snapshot.get("state", "UNKNOWN")
    sizing_factor = snapshot.get("sizing_factor", 0.0)
    paper_only = snapshot.get("paper_only", True)
    reasons = snapshot.get("reasons", [])
    stats = snapshot.get("stats", {}) or {}
    total_trades = int(stats.get("total_trades", 0))
    win_rate = float(stats.get("win_rate", 0.0))
    total_pnl = float(stats.get("total_pnl", 0.0))

    return {
        "available": True,
        "state": state,
        "sizing_factor": float(sizing_factor),
        "paper_only": bool(paper_only),
        "reasons": reasons,
        "stats": {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        },
        "raw_path": str(path),
    }


def create_app() -> Flask:
    """
    Create and configure the Flask app.
    """
    _get_logger()
    app = Flask(__name__)

    @app.route("/status", methods=["GET"])
    def status() -> Response:
        """
        Return CHAD status as JSON.

        This is safe and read-only. It only reports:
            - current time
            - global CHAD mode (DRY_RUN / LIVE)
            - shadow state snapshot (if available)
        """
        now = datetime.now(timezone.utc).isoformat()

        # Mode block
        mode = get_chad_mode()
        live_enabled = is_live_mode_enabled()
        mode_block: Dict[str, Any] = {
            "chad_mode": mode.value,
            "live_enabled": live_enabled,
        }

        # Shadow state block
        shadow = _load_shadow_snapshot(SHADOW_STATE_PATH)

        payload: Dict[str, Any] = {
            "service": "CHAD Status Server",
            "time_utc": now,
            "healthy": True,
            "mode": mode_block,
            "shadow": shadow,
        }

        return jsonify(payload)

    @app.route("/", methods=["GET"])
    def root() -> Response:
        """
        Root path – suggests /status for full JSON.
        """
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "service": "CHAD Status Server",
            "time_utc": now,
            "message": "See /status for full JSON status.",
        }
        return jsonify(payload)

    return app


def main() -> None:
    """
    CLI entrypoint for:

        python -m chad.web.status_server
    """
    log = _get_logger()
    app = create_app()

    host = os.environ.get("CHAD_STATUS_HOST", "0.0.0.0")
    port_raw = os.environ.get("CHAD_STATUS_PORT", "9618")
    try:
        port = int(port_raw)
    except ValueError:
        log.warning("Invalid CHAD_STATUS_PORT=%s – defaulting to 9618", port_raw)
        port = 9618

    log.info("Starting CHAD Status Server on %s:%d ...", host, port)
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()

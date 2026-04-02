"""
CHAD Status Server (Phase 6/7 – Read-Only HTTP Status + Shadow/SCR + Mode)

This module exposes a minimal HTTP /status endpoint that reports:

    * Basic service health
    * Current UTC time
    * Global mode (CHAD_MODE):
        - DRY_RUN or LIVE (normalized)
        - live_enabled flag
    * Shadow/SCR state (authoritative runtime first), including:
        - state
        - sizing_factor
        - paper_only
        - reasons
        - key stats (total_trades, win_rate, total_pnl, etc.)

It is STRICTLY READ-ONLY – it does NOT execute trades, modify config,
or change system mode.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, Response

from chad.analytics.shadow_state_snapshot import SHADOW_STATE_PATH as LEGACY_DATA_SHADOW_PATH
from chad.core.mode import get_chad_mode, is_live_mode_enabled


LOGGER_NAME = "chad.status_server"
logger = logging.getLogger(LOGGER_NAME)

# Runtime artifacts (preferred). We align status to SSOT runtime truth.
DEFAULT_RUNTIME_DIR = Path(os.environ.get("CHAD_RUNTIME_DIR", "/home/ubuntu/chad_finale/runtime"))
RUNTIME_SCR_PATH = DEFAULT_RUNTIME_DIR / "scr_state.json"         # canonical SCR artifact
RUNTIME_SHADOW_PATH = DEFAULT_RUNTIME_DIR / "shadow_state.json"   # compatibility artifact (kept fresh by scr_state_sync.py)


def _get_logger() -> logging.Logger:
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger


def _read_json_file(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, f"not_found:{path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if not isinstance(obj, dict):
            return None, f"invalid_json_type:{path}"
        return obj, None
    except Exception as exc:  # noqa: BLE001
        return None, f"read_error:{type(exc).__name__}:{exc}"


def _normalize_shadow_payload(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts either:
      - {"shadow": {...}}  (API gateway shape)
      - {...}              (runtime scr_state/shadow_state shape)
    Returns the inner shadow dict.
    """
    if "shadow" in obj and isinstance(obj.get("shadow"), dict):
        return obj["shadow"]  # type: ignore[return-value]
    return obj


def _compact_shadow_view(*, payload: Dict[str, Any], raw_path: str, available: bool, err: Optional[str]) -> Dict[str, Any]:
    if not available:
        return {
            "available": False,
            "reason": err or "unavailable",
            "raw_path": raw_path,
        }

    p = _normalize_shadow_payload(payload)

    state = p.get("state", "UNKNOWN")
    sizing_factor = p.get("sizing_factor", 0.0)
    paper_only = p.get("paper_only", True)
    reasons = p.get("reasons", [])
    stats = p.get("stats", {}) or {}

    return {
        "available": True,
        "state": state,
        "sizing_factor": float(sizing_factor or 0.0),
        "paper_only": bool(paper_only),
        "reasons": list(reasons) if isinstance(reasons, list) else [str(reasons)],
        "stats": {
            # Raw counts (everything parsed)
            "total_trades": int(stats.get("total_trades", 0)),
            "live_trades": int(stats.get("live_trades", 0)),
            "paper_trades": int(stats.get("paper_trades", 0)),
            # Effective sample (what SCR trusts)
            "effective_trades": int(stats.get("effective_trades", 0)),
            "excluded_manual": int(stats.get("excluded_manual", 0)),
            "excluded_untrusted": int(stats.get("excluded_untrusted", 0)),
            "excluded_nonfinite": int(stats.get("excluded_nonfinite", 0)),
            # Performance metrics over effective sample
            "win_rate": float(stats.get("win_rate", 0.0)),
            "total_pnl": float(stats.get("total_pnl", 0.0)),
            "max_drawdown": float(stats.get("max_drawdown", 0.0)),
            "sharpe_like": float(stats.get("sharpe_like", 0.0)),
        },
        "ts_utc": p.get("ts_utc") or p.get("timestamp_utc"),
        "ttl_seconds": p.get("ttl_seconds"),
        "raw_path": raw_path,
    }


def _load_shadow_best_effort() -> Dict[str, Any]:
    """
    Load shadow/SCR status using SSOT-aligned precedence:
      1) runtime/scr_state.json (canonical SCR)
      2) runtime/shadow_state.json (compat, kept fresh by scr_state_sync)
      3) data/shadow/shadow_state.json (legacy snapshot)
    """
    # 1) runtime/scr_state.json
    obj, err = _read_json_file(RUNTIME_SCR_PATH)
    if obj is not None:
        return _compact_shadow_view(payload=obj, raw_path=str(RUNTIME_SCR_PATH), available=True, err=None)

    # 2) runtime/shadow_state.json
    obj2, err2 = _read_json_file(RUNTIME_SHADOW_PATH)
    if obj2 is not None:
        return _compact_shadow_view(payload=obj2, raw_path=str(RUNTIME_SHADOW_PATH), available=True, err=None)

    # 3) legacy data snapshot
    obj3, err3 = _read_json_file(LEGACY_DATA_SHADOW_PATH)
    if obj3 is not None:
        return _compact_shadow_view(payload=obj3, raw_path=str(LEGACY_DATA_SHADOW_PATH), available=True, err=None)

    # Nothing available
    return _compact_shadow_view(
        payload={},
        raw_path=f"{RUNTIME_SCR_PATH} | {RUNTIME_SHADOW_PATH} | {LEGACY_DATA_SHADOW_PATH}",
        available=False,
        err=err or err2 or err3 or "no_shadow_sources_available",
    )


def create_app() -> Flask:
    _get_logger()
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> Response:
        now = datetime.now(timezone.utc).isoformat()
        return jsonify({"service": "CHAD Status Server", "healthy": True, "time_utc": now})

    @app.route("/status", methods=["GET"])
    def status() -> Response:
        now = datetime.now(timezone.utc).isoformat()

        mode = get_chad_mode()
        live_enabled = is_live_mode_enabled()
        mode_block: Dict[str, Any] = {
            "chad_mode": mode.value,
            "live_enabled": live_enabled,
        }

        shadow = _load_shadow_best_effort()

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
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "service": "CHAD Status Server",
            "time_utc": now,
            "message": "See /status for full JSON status.",
        }
        return jsonify(payload)

    return app


def main() -> None:
    log = _get_logger()
    app = create_app()

    host = os.environ.get("CHAD_STATUS_HOST", "0.0.0.0")
    port_raw = os.environ.get("CHAD_STATUS_PORT", "9619")
    try:
        port = int(port_raw)
    except ValueError:
        log.warning("Invalid CHAD_STATUS_PORT=%s – defaulting to 9619", port_raw)
        port = 9619

    log.info("Starting CHAD Status Server on %s:%d ...", host, port)
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()

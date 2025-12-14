"""
CHAD Backend FastAPI Application (Phase 7 – Gateway Wrapper)

This module is the ONLY FastAPI app served by:

    chad-backend.service → uvicorn backend.app:app --port 9618

Design goals for Phase 7:

* The backend on port 9618 must be **read-only** and **risk-aware**.
* All real logic is delegated to the hardened CHAD API Gateway:
      backend.api_gateway:app
* No dynamic inclusion of routers like backend.orders, backend.ibkr, etc.
* No HTTP path can bypass:
      - ExecutionConfig (ibkr_dry_run hard-lock),
      - CHAD_MODE,
      - SCR + Shadow Router,
      - LiveGate evaluation.

This file is intentionally minimal and conservative:
it primarily mounts the API Gateway at root and exposes a tiny
diagnostic shim at /__backend__ for ops verification.
"""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import FastAPI

from backend.api_gateway import app as api_gateway_app

# Optional systemd watchdog integration.
# The module may or may not exist; we degrade gracefully to no-op.
try:
    from ops.watchdog import notify_ready, start_watchdog  # type: ignore[import]
except Exception:  # noqa: BLE001

    def start_watchdog(_interval: float = 10.0) -> None:  # type: ignore[override]
        return None

    def notify_ready() -> None:  # type: ignore[override]
        return None


LOGGER = logging.getLogger("chad.backend.app")


# ---------------------------------------------------------------------------
# Primary backend app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CHAD Backend (Phase 7)",
    version="0.1.0-backend7",
    description=(
        "CHAD Backend wrapper that delegates all functionality to the "
        "hardened CHAD API Gateway. No legacy routers are mounted here."
    ),
)


@app.on_event("startup")
async def _on_startup() -> None:
    """
    Backend startup hook.

    * Ensures logging is configured.
    * Starts systemd watchdog heartbeat if available.
    * Notifies systemd that the service is READY.
    """
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    LOGGER.info("CHAD Backend (Phase 7) startup: mounting API Gateway at root '/'.")

    # Start watchdog heartbeat (no-op if ops.watchdog is not present).
    start_watchdog(10.0)
    notify_ready()


# Mount the API Gateway at the root.
# All routes such as /health, /risk-state, /live-gate, /shadow, and the
# Phase-7-disabled /orders endpoint come from backend.api_gateway.
app.mount("/", api_gateway_app)


# ---------------------------------------------------------------------------
# Backend-level diagnostic shim
# ---------------------------------------------------------------------------

@app.get("/__backend__", include_in_schema=False, tags=["system"])
async def backend_info() -> Dict[str, str]:
    """
    Diagnostic endpoint to confirm that:

    * Systemd is serving backend.app:app on port 9618.
    * The API Gateway is mounted at root.

    This endpoint exposes no trading functionality and exists solely for
    ops verification and debugging.
    """
    return {
        "service": "CHAD Backend (Phase 7)",
        "message": (
            "API Gateway (backend.api_gateway:app) is mounted at root '/'. "
            "Use /health, /risk-state, /live-gate, and /shadow for status."
        ),
        "gateway_module": "backend.api_gateway",
    }

"""
IBKR integration router for CHAD backend.

Provides lightweight, safe endpoints to verify connectivity to the
IB Gateway running on localhost (paper or live).

Endpoints
---------
GET /ibkr/health
    - Checks:
        * we can connect to host:port
        * accounts list is non-empty
        * server time is readable
    - Returns: {"ok": bool, ...}

GET /ibkr/serverTime
    - Returns the current IBKR server time.

Configuration (via environment)
-------------------------------
IBKR_HOST=127.0.0.1          # default
IBKR_PORT=4002               # 4002 = paper, 4001 = live
IBKR_CLIENT_ID=9001          # dedicated client id for health checks
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from ib_insync import IB

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4002"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "9001"))

router = APIRouter(prefix="/ibkr", tags=["ibkr"])

# Shared global IB instance + lock
_ib = IB()
_ib_lock = threading.Lock()


def _ensure_event_loop() -> None:
    """
    Ensure the current thread has an asyncio event loop.

    FastAPI runs path handlers in AnyIO worker threads that may not have
    a default event loop. ib_insync internally calls asyncio.get_event_loop(),
    so we create and set one if needed.
    """
    try:
        # If there's already a running loop in this thread, we're good.
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop in this thread -> create and set one.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def _ensure_connected() -> IB:
    """
    Ensure global IB client is connected to the IB Gateway.
    Reconnects if disconnected. Converts failures into clean HTTP errors.
    """
    _ensure_event_loop()

    with _ib_lock:
        if _ib.isConnected():
            return _ib

        try:
            _ib.disconnect()
        except Exception:
            pass

        try:
            _ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=10)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "ok": False,
                    "reason": "ibkr_connect_failed",
                    "host": IBKR_HOST,
                    "port": IBKR_PORT,
                    "clientId": IBKR_CLIENT_ID,
                    "error": repr(exc),
                },
            ) from exc

        if not _ib.isConnected():
            raise HTTPException(
                status_code=503,
                detail={
                    "ok": False,
                    "reason": "ibkr_not_connected_after_connect",
                    "host": IBKR_HOST,
                    "port": IBKR_PORT,
                    "clientId": IBKR_CLIENT_ID,
                },
            )

        return _ib


@router.get("/health")
def ibkr_health() -> Dict[str, Any]:
    """
    Full IBKR health check: managed accounts + server time.
    """
    ib = _ensure_connected()

    # Accounts
    try:
        accounts: List[str] = list(ib.managedAccounts())
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "reason": "ibkr_accounts_failed",
                "error": repr(exc),
            },
        ) from exc

    # Server time
    try:
        server_time = ib.reqCurrentTime()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "reason": "ibkr_server_time_failed",
                "error": repr(exc),
            },
        ) from exc

    return {
        "ok": bool(accounts),
        "host": IBKR_HOST,
        "port": IBKR_PORT,
        "clientId": IBKR_CLIENT_ID,
        "accounts": accounts,
        "serverTime": server_time.isoformat() if hasattr(server_time, "isoformat") else str(server_time),
    }


@router.get("/serverTime")
def ibkr_server_time() -> Dict[str, Any]:
    """
    Thin wrapper around reqCurrentTime.
    """
    ib = _ensure_connected()

    try:
        server_time = ib.reqCurrentTime()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "ok": False,
                "reason": "ibkr_server_time_failed",
                "error": repr(exc),
            },
        ) from exc

    return {
        "ok": True,
        "host": IBKR_HOST,
        "port": IBKR_PORT,
        "clientId": IBKR_CLIENT_ID,
        "serverTime": server_time.isoformat() if hasattr(server_time, "isoformat") else str(server_time),
    }

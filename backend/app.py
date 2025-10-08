"""
CHAD Backend — FastAPI app bootstrap (production ready)

Features
--------
- Systemd watchdog heartbeat (READY=1 then WATCHDOG=1 every 10s)
- Dynamic router inclusion (imports if available, skips if missing)
- Shadow/canary safety: block mutating routes in MODE=shadow
- Stable /health and /status endpoints (always available)

Environment
-----------
MODE=prod|shadow                 # shadow blocks POST/PUT/PATCH/DELETE
CHAD_BUILD_SHA=<git sha>         # optional, shown in /status
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# --- Systemd watchdog (READY + periodic WATCHDOG) ---
try:
    from ops.watchdog import notify_ready, start_watchdog  # provided earlier
except Exception:  # graceful no-op if module missing
    def start_watchdog(_interval: float = 10.0) -> None:  # type: ignore
        pass
    def notify_ready() -> None:  # type: ignore
        pass

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="CHAD Backend", version="1.0")

# Systemd watchdog heartbeat
start_watchdog(10.0)
notify_ready()

MODE = os.getenv("MODE", "prod").lower()
BUILD_SHA = os.getenv("CHAD_BUILD_SHA", "unknown")
WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# -----------------------------------------------------------------------------
# Shadow/canary write-block (safe default)
# -----------------------------------------------------------------------------
@app.middleware("http")
async def shadow_block_mutations(request: Request, call_next):
    """
    When running in MODE=shadow, block mutating methods so canary stacks
    never hit real state-changing endpoints. Health checks remain allowed.
    """
    if MODE == "shadow" and request.method in WRITE_METHODS and not request.url.path.startswith("/health"):
        return JSONResponse(
            {"ok": False, "reason": "shadow_mode_mutation_blocked", "path": str(request.url.path)},
            status_code=403,
        )
    return await call_next(request)

# -----------------------------------------------------------------------------
# Minimal, always-on health & status endpoints
# -----------------------------------------------------------------------------
START_TS = time.time()

@app.get("/health")

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "mode": MODE}

async def health() -> dict[str, Any]:
    return {"ok": True, "mode": MODE}

@app.get("/status")
async def status() -> dict[str, Any]:
    return {"ok": True, "mode": MODE, "uptime_s": int(time.time() - START_TS), "build_sha": BUILD_SHA}

# -----------------------------------------------------------------------------
# Utility: include router if module exposes `router`
# -----------------------------------------------------------------------------
def _include_if_present(dotted: str, *, prefix: str | None = None) -> bool:
    """
    import <dotted> and add .router to app if present.
    Returns True if included, False if missing or invalid.
    """
    try:
        mod = importlib.import_module(dotted)
        router = getattr(mod, "router", None)
        if router is None:
            return False
        if prefix:
            # FastAPI routers usually carry their own prefix; we only include here.
            pass
        app.include_router(router)
        return True
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Dynamic router loading: include whatever exists in your codebase
# -----------------------------------------------------------------------------
# Core modules commonly present in your repo
_INCLUDES = [
    "backend.news",           # router: /news/*
    "backend.status",         # router: /status/* (your existing richer status)
    "backend.portfolio",      # router: /portfolio/*
    "backend.orders",         # router: /orders/*
    "backend.risk",           # router: /risk/*
    "backend.price",          # router: /price/*
    "backend.fx",             # router: /fx/*
    "backend.idea",           # router: /idea/*
    "backend.symbols",        # router: /symbols/*
    "backend.ibkr",           # optional: /ibkr/*
    "backend.crypto",         # optional: /crypto/*
    "backend.guards",         # new: /risk/guards/status
    "backend.health_status",  # adds /health, /status, /metrics/latency, /metrics/llm
    "backend.health",         # optional: your separate health router if you keep one
]

for dotted in _INCLUDES:
    _include_if_present(dotted)

# -----------------------------------------------------------------------------
# Optional: API latency middleware (cheap percentile metrics live in health_status)
# -----------------------------------------------------------------------------
try:
    # If metrics_latency.observe_api_latency exists (from previous step), record it.
    from .metrics_latency import observe_api_latency  # type: ignore

    @app.middleware("http")
    async def _latency_mw(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        dt_ms = int((time.perf_counter() - t0) * 1000)
        try:
            observe_api_latency(dt_ms)
        except Exception:
            pass
        return resp
except Exception:
    # metrics endpoint not present; silently skip
    pass

# -----------------------------------------------------------------------------
# Done — FastAPI app ready
# -----------------------------------------------------------------------------
_include_if_present("backend.brains")
_include_if_present("backend.kill_switch")

# --- TEAM CHAD: mount health & metrics ---
try:
    from backend.data_health import router as _data_router
    app.include_router(_data_router)
except Exception:
    pass
try:
    from backend.metrics_exporter import router as _metrics_router
    app.include_router(_metrics_router)
except Exception:
    pass
_include_if_present("backend.perf_summary")

# --- TEAM CHAD: mount /healthz ---
try:
    from backend.healthz import router as _hz_router
    app.include_router(_hz_router)
except Exception:
    pass

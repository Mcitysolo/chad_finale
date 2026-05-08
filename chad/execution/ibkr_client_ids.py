"""
chad/execution/ibkr_client_ids.py

Canonical IBKR client ID registry for CHAD (GAP-010).

Each IB Gateway / TWS connection identifies itself with a `clientId`. The
gateway rejects a second concurrent connect from the same id with Error 326
("client id is already in use"), so collisions silently disable a service.
Historically these ids were hardcoded in scattered Python modules, env
files, and systemd unit files, which made it easy to pick a new id that
clashed with an existing one.

This module is the single source of truth for the named client ids that
CHAD processes use against IB Gateway. New services must add a constant
here and reference it; do not introduce raw integer client ids in code.

Scope of this file
------------------
* The constants below mirror the values currently in use across the
  codebase and systemd units. **The values are documentation of existing
  reality, not a re-assignment.** GAP-010 deliberately preserves every
  value so behavior is unchanged.
* For services whose client id is supplied via env var (IBKR_CLIENT_ID)
  injected by a systemd unit, the constant here documents the unit's
  current value. The Python code for those services still reads the env
  var at runtime — the constant is a reference point, not a behavior
  change.
* This module must not import anything that talks to IBKR. It is pure
  data so it is safe to import from any layer (including tests that run
  with CHAD_SKIP_IB_CONNECT=1).
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Live trading / orchestrator path
# ---------------------------------------------------------------------------

# The autonomous live loop process. Holds the primary IB connection used by
# the orchestrator for placing/cancelling orders and reading positions.
# Source: chad/core/live_loop.py — `ib.connect(..., clientId=99, ...)`.
# Note: chad/execution/ibkr_adapter.py IbkrConfig.client_id default is also
# 99 by design — when the adapter is constructed inside the live loop it
# reuses the already-connected IB instance via ib_factory, so the default
# value mirrors LIVE_LOOP rather than introducing a separate connection.
LIVE_LOOP: int = 99

# ---------------------------------------------------------------------------
# Market data providers
# ---------------------------------------------------------------------------

# IBKR snapshot price provider (reqMktData). Used in the docstring example
# for chad/market_data/ibkr_price_provider.py. The provider itself accepts
# a pre-connected IB instance, so this id only matters for stand-alone use
# of that module.
PRICE_PROVIDER: int = 9030

# IBKR historical bars (reqHistoricalData). Docstring example for
# chad/market_data/ibkr_historical_provider.py.
HISTORICAL_PROVIDER: int = 9034

# Periodic price_cache.json refresh job.
# Source: chad/market_data/price_cache_refresh.py
#         `ib.connect(..., clientId=9035, timeout=15)`.
PRICE_CACHE_REFRESH: int = 9035

# Long-lived bar provider service (live cache feeder).
# Source: chad/market_data/ibkr_bar_provider.py default + systemd unit
# /etc/systemd/system/chad-ibkr-bar-provider.service (`--client-id 9021`).
# Not modified by this batch; documented here for completeness.
BAR_PROVIDER: int = 9021

# Nightly bars refresh job.
# Source: chad/market_data/nightly_bars_refresh.py — `IB_CLIENT_ID = 9053`.
NIGHTLY_BARS_REFRESH: int = 9053

# Options chain snapshot job.
# Source: chad/market_data/options_chain_refresh.py — `IBKR_CLIENT_ID = 88`.
OPTIONS_CHAIN: int = 88

# ---------------------------------------------------------------------------
# Portfolio / broker truth
# ---------------------------------------------------------------------------

# Portfolio collector service (account summary, positions).
# Source: systemd /etc/systemd/system/chad-ibkr-collector.service —
# `Environment=IBKR_CLIENT_ID=9003`. The collector reads IBKR_CLIENT_ID
# from env at runtime; the constant documents the current unit value.
PORTFOLIO_COLLECTOR: int = 9003

# Paper-fill harvester (reads execution history into evidence files).
# Source: chad/portfolio/ibkr_paper_fill_harvester.py — `IBKR_CLIENT_ID = 79`.
PAPER_FILL_HARVESTER: int = 79

# Portfolio snapshot publisher (read-only equity / positions snapshot).
# Source: chad/ops/portfolio_snapshot_publisher.py — `IBKR_CLIENT_ID = 84`.
PORTFOLIO_SNAPSHOT_PUBLISHER: int = 84

# Reconciliation publisher (compares local position_guard vs broker truth).
# Source: chad/ops/reconciliation_publisher.py — `IBKR_CLIENT_ID = 83`.
RECONCILER: int = 83

# Broker events collector (orderStatus / execDetails event tap).
# Source: chad/ops/ibkr_broker_events_collector.py default —
# `IBKR_CLIENT_ID = 118` (overridable via env).
BROKER_EVENTS_COLLECTOR: int = 118

# ---------------------------------------------------------------------------
# Operational / supporting
# ---------------------------------------------------------------------------

# Health check probe.
# Source: backend/ibkr.py default + systemd unit
# /etc/systemd/system/chad-ibkr-health.service `Environment=IBKR_CLIENT_ID=9001`.
HEALTHCHECK: int = 9001

# Paper shadow runner (parallel paper-mode validator).
# Source: chad/core/paper_shadow_runner.py default + systemd drop-in
# /etc/systemd/system/chad-paper-shadow-runner.service.d/96-ibkr-env.conf —
# `Environment=IBKR_CLIENT_ID=9013`.
PAPER_SHADOW_RUNNER: int = 9013

# Advisory engine (read-only intelligence module).
# Source: chad/intel/advisory_engine.py — `clientId=9036`.
ADVISORY_ENGINE: int = 9036

# Dashboard API (best-effort lightweight account ping).
# Source: chad/dashboard/api.py — `clientId=80`.
DASHBOARD_API: int = 80

# Paper ledger watcher (read-only oneshot timer service that reconciles
# runtime ledger state with broker truth). Configured via the `ibkr` block
# in runtime/ibkr_paper_ledger.json; LedgerConfig.load() reads
# ibkr.client_id from there. Explicit non-zero id required because
# clientId=0 is the IB Gateway wildcard and is unsafe.
LEDGER_WATCHER: int = 9040


# ---------------------------------------------------------------------------
# Programmatic access
# ---------------------------------------------------------------------------

def client_id_map() -> Dict[str, int]:
    """Return name -> client id for every registered constant."""
    return {
        "LIVE_LOOP": LIVE_LOOP,
        "PRICE_PROVIDER": PRICE_PROVIDER,
        "HISTORICAL_PROVIDER": HISTORICAL_PROVIDER,
        "PRICE_CACHE_REFRESH": PRICE_CACHE_REFRESH,
        "BAR_PROVIDER": BAR_PROVIDER,
        "NIGHTLY_BARS_REFRESH": NIGHTLY_BARS_REFRESH,
        "OPTIONS_CHAIN": OPTIONS_CHAIN,
        "PORTFOLIO_COLLECTOR": PORTFOLIO_COLLECTOR,
        "PAPER_FILL_HARVESTER": PAPER_FILL_HARVESTER,
        "PORTFOLIO_SNAPSHOT_PUBLISHER": PORTFOLIO_SNAPSHOT_PUBLISHER,
        "RECONCILER": RECONCILER,
        "BROKER_EVENTS_COLLECTOR": BROKER_EVENTS_COLLECTOR,
        "HEALTHCHECK": HEALTHCHECK,
        "PAPER_SHADOW_RUNNER": PAPER_SHADOW_RUNNER,
        "ADVISORY_ENGINE": ADVISORY_ENGINE,
        "DASHBOARD_API": DASHBOARD_API,
        "LEDGER_WATCHER": LEDGER_WATCHER,
    }


def all_client_ids() -> List[int]:
    """Return the list of all registered client id values, sorted ascending."""
    return sorted(client_id_map().values())


def assert_no_collisions() -> None:
    """
    Raise ValueError if two registry names share the same client id.

    IB Gateway rejects a second concurrent connect with the same client id
    (Error 326). A collision in this registry would mean two named services
    are configured to compete for the same connection slot.
    """
    seen: Dict[int, str] = {}
    collisions: List[str] = []
    for name, cid in client_id_map().items():
        if cid in seen:
            collisions.append(f"{seen[cid]} and {name} both use clientId={cid}")
        else:
            seen[cid] = name
    if collisions:
        raise ValueError(
            "IBKR client id collisions detected: " + "; ".join(collisions)
        )


# Fail fast at import time if the registry itself is inconsistent.
assert_no_collisions()

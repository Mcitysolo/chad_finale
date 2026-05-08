# Qualify Timeout — ib_async Phase 2 Dependency

Status: Pending (blocked on GAP-A019 Phase 2)
Created: 2026-05-08

## Issue
ibkr_adapter._qualify_if_possible() calls qualifyContracts() on every
submission attempt. All calls timeout at 10s. Root cause: ib_insync
event-loop architecture — qualifyContracts cannot complete when called
from a daemon thread while the main event loop runs. The qualify cache
(commit 9ccbebb) is wired correctly but cannot populate until at least
one qualify call succeeds.

## Impact
~68 timeouts/day, each adding 10s latency to paper submissions.
Paper fills still complete (fallback to unqualified contract).
No live capital risk.

## Fix Path
Migrate ibkr_adapter.py to ib_async (GAP-A019 Phase 2).
ib_async uses native asyncio — eliminates the daemon-thread conflict.
Once migrated, qualifyContracts will resolve correctly and the cache
will populate on first call per symbol per session.

## Interim Mitigation
The qualify cache is in place. When ib_async migration completes,
the cache will eliminate repeat calls immediately.

## Dependencies
- GAP-A019 Phase 2 (ibkr_adapter.py migration)
- Verification: watch for IBKR_QUALIFY_CACHE_STORE in journal

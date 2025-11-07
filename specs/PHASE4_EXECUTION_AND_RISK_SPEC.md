# Phase 4 — Execution & Risk Engine (Implementation Spec)

## Objective
Implement a live-safe execution layer with broker adapters, unified ExecutionRouter, and RiskManager v2 (VaR, drawdown, Kelly), with **DRY_RUN** default and explicit guardrails. All order flow must be deterministic, idempotent, and fully logged to `delta_trades.ndjson`.

## Deliverables
1. **Broker Adapters**
   - `adapters/ibkr_adapter.py`
     - Connect via ib_insync; connection lifecycle; health probe.
     - Place/cancel bracket and marketable limit orders with idempotent client-assigned IDs.
     - Dry-run mode routes to ledger only (no API).
     - Round-trip timing metrics (p50/p95).
   - `adapters/coinbase_adapter.py`
     - Advanced Trade API (REST + optional websocket for fills).
     - Same contract as IBKR adapter (place/cancel, idempotent IDs, dry-run path).
     - JWT handling is **never** stored in logs; redact in traces.

2. **Unified ExecutionRouter**
   - `services/execution_router.py`
     - Factory: `get_adapter(broker: Literal["ibkr","coinbase"])`.
     - Retries with exponential backoff + jitter for transient errors.
     - Circuit breaker on adapter-level error budget.
     - Exactly-once semantics via deterministic `order_key` and ledger de-dup.
     - Structured result object with broker refs and timings.

3. **RiskManager v2**
   - `services/risk_manager.py`
     - Inputs: `TradeIntent` (side, qty, symbol, price, strategy, confidence), portfolio snapshot, rolling risk metrics.
     - Enforce: global max DD, per-strategy max risk, Kelly fraction cap, per-symbol exposure cap.
     - Output: approved position size (can be 0), stop/target bands, optional denial reason.
     - DRY_RUN respected at the top-level gate.

4. **CapitalAllocator v2**
   - `services/capital_allocator.py`
     - Honor a capital map (e.g., 50/30/20) and a per-strategy max allocation.
     - Integrates with RiskManager v2; returns final allowed notional/qty.

5. **Order Ledger & Atomic Writes**
   - `data/ledger.py`
     - Append-only NDJSON with atomic temp→mv writes.
     - De-dup by `order_key`.
     - Persist request, decision (risk/alloc), adapter result, and timings.

6. **CLI & Service Glue**
   - `bin/dryrun_execute.py` to simulate Phase 4 pipeline end-to-end with DRY_RUN=1.
   - `services/health.py` with `probe_brokers()` (dry-run friendly).

7. **Configs & Guards**
   - `config/phase4.yaml` (limits, caps, endpoints).
   - `DRY_RUN` default true; live flips only via config + explicit flag.

8. **Tests**
   - Unit tests for RiskManager sizing/denials and allocator split.
   - Adapter tests (mocked IO) verifying idempotent IDs and retry/jitter logic.
   - Router integration test: intent→risk→alloc→ledger→(adapter or dry-run).

9. **Docs**
   - `OPERATIONS.md` section: how to run DRY_RUN, inspect ledger, view timing metrics.

## Non-Goals (Phase 4)
- No strategy logic (Phase 3).
- No SCR/canary (Phase 5).
- No direct Telegram controls (Phase 6).

## Acceptance Criteria
- `pytest -q` passes with new tests.
- Dry-run e2e: produces `delta_trades.ndjson` with idempotent order entries.
- Risk limits enforced; denial reasons logged; allocations respect caps.
- Adapters never log secrets; timing metrics present.
- No live orders unless DRY_RUN is explicitly false **and** config permit is set.


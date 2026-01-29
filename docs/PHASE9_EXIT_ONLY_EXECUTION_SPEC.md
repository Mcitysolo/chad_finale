# CHAD Phase 9.1 — Exit-Only Execution (Paper) — Design Spec

## Timestamp (UTC)
2026-01-29T02:10:29+00:00

## Scope (Design Only)
This document defines Phase 9.1 behavior.
It introduces NO implementation and makes NO changes to current Phase 8 runtime.

## Objective
Allow CHAD to perform **paper exits only** when LiveGate indicates:
- allow_exits_only = true
- allow_ibkr_paper = false
- allow_ibkr_live = false

No new entries are ever permitted in Phase 9.1.

## Preconditions (Must Already Be True)
- Phase 8 CLOSED (LAW)
- LiveGate authoritative
- Reconciliation GREEN
- Exactly-once semantics enforced
- STOP remains globally authoritative

## Execution Rules (Exit-Only)
- Only positions that already exist may be closed.
- No position may be increased, flipped, or re-opened.
- Each exit is idempotent (exactly-once).

## Authority Chain (Unchanged)
1. Strategy Brains → **signal intent only**
2. RiskManager → validates exit eligibility
3. SCR → confirms confidence constraints
4. LiveGate → authorizes exit-only
5. Executor → submits paper exit orders
6. Ledger → records exactly-once result

## Safety Guarantees
- If LiveGate flips back to DENY_ALL, exit execution halts immediately.
- STOP kills all execution, including exits.
- No lane or strategy may bypass this flow.

## Lane Context (Future Only)
- Phase 9.1 assumes a single global execution path.
- A future optional `lane_id` parameter may be passed through interfaces,
  but no lane logic is applied in Phase 9.1.

## Non-Goals (Explicit)
- No live trading
- No new entries
- No partial exits
- No strategy prioritization
- No performance optimization

## Promotion Criteria (Future)
Phase 9.1 may only be implemented after:
- explicit operator promotion
- test harness approval
- audit of idempotency paths
- Phase 9 Active Freeze creation


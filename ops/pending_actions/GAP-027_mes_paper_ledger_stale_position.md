# GAP-027 — MES Paper Ledger Stale Position

Status: Documented known issue
Risk: Zero (paper account only)
Created: 2026-05-07

## Finding
runtime/ibkr_paper_ledger_state.json contains an MES futures position with
source_strategies=["manual"], opened 2026-04-03. The position's avg_cost
(~$31,845) does not match current MES price ($7,384) because the contract
(conId=272093, March 2026 expiry) has expired and rolled. This is a stale
paper ledger artifact, not a live position with real capital.

## Positions in paper ledger (all manual, all paper):
- MES (FUT): stale expired contract — avg_cost inconsistent with current price
- MSFT (STK): avg_cost $477.93 — manual paper position
- AAPL (STK): avg_cost $261.99 — manual paper position
- SPY  (STK): avg_cost $690.56 — manual paper position

## Reconciliation
These 4 positions are excluded from CHAD's active reconciliation via
config/reconciliation_exclusions.json (AAPL, MSFT, NVDA explicitly excluded;
SPY broker drift formally excluded as pre-existing). MES is in the paper
ledger only — not in broker live positions.

## Resolution
- No action required before live promotion (paper only, no real capital)
- Optional: close MES position via IBKR Gateway TWS UI at next maintenance window
- Optional: reset paper account to eliminate legacy manual positions

Operator decision deferred. Not a live-trading blocker.

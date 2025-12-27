# CHAD Phase 11 — 16 Brain Roadmap (User Target)

This doc tracks the user’s target roster of 16 brains and the definition of “functioning” (no corners).

## Current reality (implemented today)
These 7 exist in code and are registered in StrategyName + registry:

1. ALPHA (equities)
2. BETA (equities)
3. GAMMA (equities)
4. OMEGA (equities hedge / macro overlay)
5. DELTA (execution/meta)
6. ALPHA_CRYPTO (crypto)
7. ALPHA_FOREX (FX)

## Phase 11 expansion target (add 9 to reach 16)
These are the additional brains required to reach the user’s “16 brain” roster:

8.  ALPHA_CRYPTO_ALT
9.  ALPHA_FUTURES
10. GAMMA_FUTURES
11. OMEGA_MACRO
12. ALPHA_OPTIONS
13. OMEGA_VOL
14. GAMMA_REVERSION
15. DELTA_PAIRS
16. ALPHA_STATARB

## Definition of “FUNCTIONING” (must pass all gates)
A brain is only counted as “functioning” when ALL of the following are true:

A) Code + Registry
- Strategy module exists (typed, lint-clean, unit-tested).
- StrategyName enum includes it.
- Strategy registry registers it deterministically.

B) Signal pipeline proof (no execution)
- `python3 -m chad.core.full_execution_cycle` shows it can emit signals (at least in a controlled test mode).

C) Execution-plan proof (no broker)
- A planned order is produced with correct strategy attribution.

D) Execution proof (market-appropriate)
- Equities/options/futures: paper execution during market hours.
- Crypto: validate-only + live canary (tiny) or exchange sandbox if available.
- FX: paper execution during FX open hours.

E) Ledger proof (traceable)
- A TradeResult record is written into `data/trades/trade_history_YYYYMMDD.ndjson`.
- `strategy != manual`
- Not marked `pnl_untrusted`
- Counts toward `effective_trades`.

F) Safety proof (non-negotiable)
- SCR/LiveGate/caps must be applied and reasons visible in /live-gate.
- Kill switch behavior proven (STOP / DRY_RUN prevents live execution).

## Sequence constraint (don’t jump)
Phase 11 build-out begins ONLY after:
- Phase 8 go-live mechanics are proven (paper canaries + logging + STOP drills),
- Phase 9 observability is stable,
- Base trio profitability is demonstrated with effective trades (not warmup churn).

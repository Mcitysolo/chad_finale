# CHAD PM Gate — Required 10/10 Before Phase 11 Expansion

Phase 11 (16-brain expansion) begins ONLY when these are true.

## Gate A — Safety + Execution (must be 10/10)
- /live-gate is stable and always returns 200
- SCR uses effective_trades and no longer poisoned by warmup/manual churn
- STOP/DRY_RUN guardrail proven (live_mode.json + LiveGate reasons)
- At least 10 effective trades exist in ledger from a real market (not manual warmup)

## Gate B — Unified Ledger (must be 10/10)
- IBKR paper trades log to TradeResult NDJSON
- Kraken live trades log to TradeResult NDJSON
- TradeResult records have correct strategy labels (not manual)
- Stats engine computes effective_trades > 0

## Gate C — Market Proof (must be 10/10)
- Equities: at least 1 executed paper fill during market hours logged as non-manual
- Crypto: at least 1 live canary logged (done)
- FX: at least 1 paper fill logged during FX open hours

## Gate D — Ops + Observability (must be 10/10)
- /metrics stable (no NaN/Inf poisoning)
- systemd services survive restart (backend/orchestrator/ibgateway/collectors)
- backup exists and succeeds
- alerts/health checks exist for feed stale + broker down

## Gate E — Expansion Discipline (policy)
- Add brains one at a time with acceptance tests:
  signals -> plan -> (paper/live-canary) -> ledger -> effective_trades


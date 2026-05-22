# BOX-057 — Kraken-CA futures scope policy (spot-only)

**Status:** Pending Action (paper / governance — no operator-applied change required).
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 057.
**Title:** Kraken-CA futures blocker resolved or scoped out
**Acceptance criterion (verbatim):**
> "jurisdiction issue is either solved or strategy remains spot-only."

## Decision

**SCOPED OUT — CHAD remains SPOT-ONLY on Kraken under the current
Canadian deployment.** Kraken Futures / perpetuals / derivatives /
margin live trading is **not** authorized and is **not** required for
`ready_for_live`. This is a regulatory / jurisdictional block, not a
technical one — Kraken Futures products are not available to retail or
unregistered accounts under the operator's current Canadian
jurisdiction.

This decision is the official Box-057 closure of the C1 external
blocker documented in
`docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`. It does NOT
log in to Kraken, does NOT change account settings, does NOT create or
modify API keys, does NOT enable futures, does NOT place orders, and
does NOT authorize live trading. CHAD remains PAPER.
`ready_for_live=false`.

## Why Kraken Futures is excluded from live-readiness

1. **No production importer.** No module under `chad/core/`,
   `chad/strategies/`, `chad/execution/`, `chad/risk/`, `chad/ops/`, or
   `ops/` imports `KrakenFuturesAdapter`, `KrakenFuturesClient`, or any
   futures symbol. The trade router
   (`chad/execution/kraken_trade_router.py`) and executor
   (`chad/execution/kraken_executor.py`) wrap `KrakenClient` (Kraken
   spot REST: `/0/private/AddOrder`) exclusively. Enforced statically
   by `chad/tests/test_box057_official_kraken_ca_futures_scope_guard.py`.
2. **Fail-closed defaults.** `KrakenFuturesAdapter.__init__(dry_run:
   bool = True)` and `KrakenFuturesClient.__init__(dry_run: bool =
   True)` keep `dry_run=True` as the default. Live submission requires
   `dry_run=False` **AND** credentials **AND** a wired opener — three
   independent gates, all currently fail-closed.
3. **Intel publisher is read-only.** The C1A
   `kraken_futures_intel_publisher` consumes the public
   `https://futures.kraken.com/derivatives/api/v3/tickers` endpoint
   only. No `submit_order` / `AddOrder` / `sendorder` / `sign_request`
   / `Authent`. Output is `runtime/kraken_futures_intel.json` —
   strategies treat it as a *confidence-only crowding signal*
   (alpha_crypto.py:730-739) and never route futures orders against it.
4. **No live-readiness gate requires Kraken Futures.** The publisher's
   gate set (`stop`, `feed`, `reconciliation`, `lifecycle_truth`,
   `execution_quality`, `mutation_state`, `canary_state`, `chad_mode`,
   `operator_intent`, `scr`) does not reference Kraken Futures.

## What this policy forbids (until the jurisdiction block is lifted)

1. No production code may import `KrakenFuturesAdapter`,
   `KrakenFuturesClient`, `KrakenFuturesOrderRequest`,
   `KrakenFuturesOrderResult`, `KrakenFuturesIntent`,
   `kraken_futures_adapter`, or `kraken_futures_client`.
2. The `dry_run: bool = True` default on the futures adapter and
   futures client must remain True.
3. The intel publisher must remain a public-ticker reader (no
   `submit_order` / `AddOrder` / `sendorder` / `sign_request` /
   `Authent` / private-endpoint URL).
4. `chad/tools/kraken_futures_auth_smoke.py` must not be run against
   the operator's Canadian Kraken account.
5. No systemd unit may execute Kraken Futures order placement.

Clauses 1–3 are enforced statically by
`chad/tests/test_box057_official_kraken_ca_futures_scope_guard.py`.
Clauses 4–5 are governance only.

## Unlock condition (manual external)

Per `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md §4`,
Kraken Futures may move from "scoped out" to "in scope" only after
**all** of the following hold simultaneously:

1. A legally eligible entity, account, and jurisdiction exists for
   Kraken Futures / derivatives / margin trading (i.e. not the current
   Canadian retail posture).
2. Operator has explicitly approved Kraken Futures live trading under
   that eligible entity **in writing**.
3. The eight technical unlock conditions documented in
   `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md §3.D` still hold:
   credentials present, certified read-only endpoint, no-order smoke
   passing, payload validation, risk-manager approval, explicit
   execution-pipeline authorization, kill-switch and reconciliation
   rules.
4. Operator validates dry-run / paper futures executions before any
   live order is permitted.

The unlock requires operator action **outside CHAD** —
jurisdiction/account eligibility (legal, not technical), explicit
written approval, and credential provisioning. **No assistant action
may be taken** to log into Kraken, change account settings, create or
modify API keys, or enable Kraken Futures.

If unlock is later granted, the guard tests in
`chad/tests/test_box057_official_kraken_ca_futures_scope_guard.py`
must be relaxed in the **same commit** that lands the operator's
written approval, the eligibility proof, and the paper/dry-run
validation evidence — never in advance.

## Verification command

```
source venv/bin/activate
python3 -m pytest chad/tests/test_box057_official_kraken_ca_futures_scope_guard.py -v
```

Last run (2026-05-21): 4 passed in 0.51s.

## Pending Action

This document is a **policy** only. There is **no runtime config change**
to apply. No service restart is required. No operator approval is needed
beyond review. CHAD remains PAPER. Live trading remains NOT authorized.
`ready_for_live=false`. Box 58+ remain open.

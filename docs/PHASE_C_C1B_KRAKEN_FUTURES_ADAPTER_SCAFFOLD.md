# Phase C Item 1B — Kraken Futures Adapter Scaffold

## Status

**SCAFFOLD ONLY — NOT LIVE TRADING ENABLED.**

This work establishes the structural foundation for future Kraken Futures
authenticated trading. It does not enable live order submission, does not
require credentials, and is not wired into any live execution path.

Date: 2026-05-16
Predecessor: Phase C Item 1A (Kraken Futures public intel publisher — committed)

## What was built

1. `chad/exchanges/kraken_futures_client.py` — low-level client scaffold.
   - Constants: `PUBLIC_BASE_URL`, `PRIVATE_BASE_URL`,
     `DEFAULT_TIMEOUT_SECONDS`.
   - Credential loader: env-first, fallback to `/etc/chad/kraken.env`,
     returns `None` if `KRAKEN_FUTURES_API_KEY` or `KRAKEN_FUTURES_API_SECRET`
     are absent.
   - Dataclasses: `KrakenFuturesCredentials`, `KrakenFuturesOrderRequest`,
     `KrakenFuturesOrderResult`.
   - Class `KrakenFuturesClient`:
     - `has_credentials()`
     - `validate_order(...)` — enforces `symbol` starts with `PF_`,
       `side ∈ {buy, sell}`, `order_type ∈ {mkt, lmt}`, `size > 0`,
       `limit_price` required for `lmt`.
     - `build_order_payload(...)` — `symbol/side/orderType/size`,
       plus `limitPrice/reduceOnly/cliOrdId` when applicable.
     - `sign_request(path, post_data, nonce)` — Kraken Futures
       `Authent` scheme structure (SHA-256 of `post_data + nonce + path`,
       HMAC-SHA-512 with base64-decoded secret, base64-encoded).
       Returns `APIKey`, `Authent`, `Nonce` headers.
     - `submit_order(...)` — dry-run by default; fails closed when
       `dry_run=False` without credentials or wired `opener`.

2. `chad/execution/kraken_futures_adapter.py` — execution adapter scaffold.
   - Dataclass `KrakenFuturesIntent`.
   - Class `KrakenFuturesAdapter`:
     - `normalize_symbol(...)` — `BTC-USD`/`XBT-USD` → `PF_XBTUSD`,
       `ETH-USD` → `PF_ETHUSD`, `SOL-USD` → `PF_SOLUSD`, `PF_*` passthrough.
     - `build_order_request(...)` — translates intent → request.
     - `submit_intent(...)` — delegates to the client; dry-run by default.
   - Default `dry_run=True`; default client has no credentials.

3. `chad/tests/test_kraken_futures_client.py` — 13 tests covering
   credential loading, validation guards, payload construction, signing
   headers, dry-run path, fail-closed paths, and no-network invariant.

4. `chad/tests/test_kraken_futures_adapter.py` — 11 tests covering
   symbol normalization, intent → request mapping, dry-run submission,
   invalid-symbol rejection, default dry-run state, and proof that the
   adapter does not import live execution modules.

5. This document.

## What was NOT built

- **No live order routing.** `submit_order` and `submit_intent` return
  dry-run results without network activity in the default configuration.
- **No strategy wiring.** No strategy imports the adapter. The adapter
  imports no strategy code.
- **No systemd unit.** No `chad-kraken-futures-adapter.*` service or timer.
- **No authenticated Kraken Futures smoke test.** The `Authent` signing
  scheme is structurally implemented per public Kraken Futures docs but
  is NOT certified against a real authenticated response yet.
- **No live execution authorization.** The adapter is not registered with
  `chad.execution.execution_pipeline`, `chad.core.live_loop`, the trade
  router, or any LiveGate / risk-cap path.
- **No production endpoint depended upon.** `submit_order`'s scaffolded
  endpoint path (`/sendorder`) is structurally placeholder-grade and
  must be validated against current Kraken Futures REST docs before
  any live use.

## Required unlock conditions before live trading

All of the following must be satisfied before this adapter may transmit
a single live order:

1. Operator creates a dedicated Kraken Futures API key/secret with
   trading permissions (NOT a reused spot key).
2. `/etc/chad/kraken.env` contains:
   - `KRAKEN_FUTURES_API_KEY=...`
   - `KRAKEN_FUTURES_API_SECRET=...`
   (Spot keys remain unaffected.)
3. Authenticated private endpoint smoke test passes — e.g. a
   read-only `/accounts` or `/openpositions` call, executed manually
   under operator supervision, returns an authenticated success.
4. Sandbox/dry-run order validation passes — Kraken Futures supports
   pre-trade validation; a `validate`-only round-trip must succeed.
5. Risk manager / LiveGate approval path wired:
   - Notional caps respected.
   - SCR / sizing factor gating applied.
   - OperatorIntent + Pending Action path for first live activation.
6. Execution pipeline integration reviewed:
   - Idempotency keys plumbed (parity with `chad.execution.kraken_executor`'s
     `ExecStateStore` discipline).
   - Reconciliation / fills writeback path defined.
   - paper_exec_evidence parity if running in paper mode first.
7. Kill switch defined: a single operator action that disables further
   Kraken Futures submissions without restarting CHAD.
8. Reconciliation rules defined for Kraken Futures positions vs.
   `broker_sync|*` truth, including how perp PnL flows into attribution.

## Next recommended steps

- Keep the public intel publisher (`chad-kraken-futures-intel-refresh.*`)
  running. It continues to publish `runtime/kraken_futures_intel.json`
  every 5 minutes with no credentials required.
- Defer building an authenticated smoke test until operator has
  provisioned Kraken Futures private keys.
- Keep C2 Coinglass deferred (paid API / deprecated public endpoints).
- Keep C3 IBKR DOM deferred (missing Level 2 entitlement).

## Files touched

- New: `chad/exchanges/kraken_futures_client.py`
- New: `chad/execution/kraken_futures_adapter.py`
- New: `chad/tests/test_kraken_futures_client.py`
- New: `chad/tests/test_kraken_futures_adapter.py`
- New: `docs/PHASE_C_C1B_KRAKEN_FUTURES_ADAPTER_SCAFFOLD.md`
- Modified: none.

## Verification

- `python3 -m py_compile` on all four new modules: clean.
- Targeted pytest on the two new test files: passes.
- Full pytest suite: must remain ≥ baseline (1894 + 21 new tests = 1915).
- `full_cycle_preview` unaffected (adapter not imported by hot path).
- Grep confirms no live integration: the adapter is not referenced by
  `chad/core/live_loop.py`, `chad/execution/execution_pipeline.py`, or
  any strategy.

## Rollback

This change is scaffold-only and adds files only. To roll back, delete
the five new files listed above. No state, no systemd, no config, and
no commit are produced by this task.

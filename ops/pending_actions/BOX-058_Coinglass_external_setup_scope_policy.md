# BOX-058 — Coinglass external setup scope policy (not required)

**Status:** Pending Action (paper / governance — no operator-applied change required).
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 058.
**Title:** Coinglass external setup decision
**Acceptance criterion (verbatim):**
> "paid API/key procured or Coinglass removed from required scope."

## Decision

**SCOPED OUT of required live-readiness.** Coinglass is **not** required
for CHAD `ready_for_live`. CHAD's crypto-derivatives intel (funding
rate, open interest, crowding bias) is sourced from the **public Kraken
Futures ticker endpoint** (`chad/market_data/crypto_derivatives_publisher.py`
and `chad/market_data/kraken_futures_intel_publisher.py`) — a free,
keyless feed that already supplies the equivalent signal set.

This decision is the official Box-058 closure of the C2 external
blocker documented in
`docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md §5` and
`docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md §4`. It does NOT buy any
API plan, does NOT create or modify any API key, does NOT call any
paid Coinglass endpoint, and does NOT authorize live trading. CHAD
remains PAPER. `ready_for_live=false`.

## Why Coinglass is not required

1. **Zero production reference.** No file under `chad/`, `ops/`,
   `scripts/`, `config/`, or `deploy/` references Coinglass. Enforced
   statically by `chad/tests/test_box058_official_coinglass_scope_guard.py`.
2. **No live-readiness gate.** `ops/live_readiness_publish.py`
   evaluates exactly these gates: `stop`, `feed`, `reconciliation`,
   `lifecycle_truth`, `execution_quality`, `mutation_state`,
   `canary_state`, `chad_mode`, `operator_intent`, `scr`. None
   references Coinglass.
3. **No strategy edge depends on Coinglass.** `alpha_crypto`,
   `omega_macro`, `delta`, `delta_pairs`, and the alpha/beta crypto
   modules build signals from bar data, public Kraken Futures
   derivatives intel, and IBKR futures price action — not from
   Coinglass-specific endpoints.
4. **Alternative provider already in use.** The Phase-B Item 4
   crypto-derivatives publisher consumes
   `https://futures.kraken.com/derivatives/api/v3/tickers` (free
   public ticker) to produce funding-rate, open-interest, and
   crowding-bias signals. This is the same shape of intel a paid
   Coinglass plan would provide.

## What this policy forbids (until / unless a paid key is procured)

1. No production code may add a Coinglass publisher, adapter, or
   consumer.
2. No public Coinglass endpoint may be used as a stand-in (the public
   endpoints returned HTTP 500 / deprecation responses during the
   Phase-C probe — `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md §4`).
3. No live-readiness gate may reference Coinglass.
4. No `COINGLASS_API_KEY` (or equivalent) may be referenced in
   production code while the key does not exist — keyless scaffolding
   is explicitly forbidden by
   `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md §5`.

Clauses 1–3 are enforced statically by
`chad/tests/test_box058_official_coinglass_scope_guard.py`. Clause 4
is governance only.

## Optional procurement path (manual external, not performed)

A Coinglass integration may move from "scoped out" to "in scope" only
after all of the following hold:

1. Operator has explicitly decided to subscribe to a **paid Coinglass
   API plan** under an eligible account.
2. The API key has been provisioned by Coinglass and securely stored
   outside the repository (e.g. via the operator's secrets manager —
   never committed).
3. A stable Coinglass endpoint contract has been documented (the prior
   probe showed deprecation / 500 on public endpoints; paid endpoints
   must be re-verified).
4. A new Coinglass publisher (Phase-C Channel-2 design) is built with
   the same shape as the existing Kraken Futures intel publisher:
   fail-soft, atomic-write, schema-versioned, with rate-limit and
   retry policy.
5. A consumer-side fallback policy is documented: if Coinglass is
   unavailable, the existing public Kraken Futures derivatives intel
   remains the SSOT — no strategy may hard-block on Coinglass.

The procurement requires operator action **outside CHAD** —
subscription purchase and key provisioning. **No assistant action may
be taken** to buy the API plan, create API keys, or call paid
Coinglass endpoints.

If procurement is later granted, the guard tests in
`chad/tests/test_box058_official_coinglass_scope_guard.py` must be
relaxed in the **same commit** that lands the publisher build, the
procurement decision record, and the fallback policy — never in
advance.

## Verification command

```
source venv/bin/activate
python3 -m pytest chad/tests/test_box058_official_coinglass_scope_guard.py -v
```

Last run (2026-05-21): 3 passed in 1.14s.

## Pending Action

This document is a **policy** only. There is **no runtime config change**
to apply. No service restart is required. No operator approval is
needed beyond review. CHAD remains PAPER. Live trading remains NOT
authorized. `ready_for_live=false`. Box 59+ remain open.

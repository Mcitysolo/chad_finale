# BOX-056 — IBKR DOM / Level 2 scope policy

**Status:** Pending Action (paper / governance — no operator-applied change required).
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 056.
**Title:** IBKR DOM / Level 2 external blocker resolved or scoped out
**Acceptance criterion (verbatim):**
> "DOM rows proven or formally excluded from live-readiness scope."

## Decision

**SCOPED OUT of live-readiness.** IBKR DOM / Level 2 / market-depth data is
**not** required for CHAD `ready_for_live`. CHAD has zero production
consumers of any IBKR DOM API (`reqMktDepth`, `cancelMktDepth`,
`reqMktDepthExchanges`, `domBids`, `domAsks`, `updateMktDepth*`) and zero
live-readiness gates condition on DOM availability.

This decision is the official Box-056 closure of the C3 external blocker
documented in `docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md`. It does NOT
change any IBKR account settings, does NOT request new market-data
subscriptions, does NOT place orders, and does NOT authorize live trading.
CHAD remains PAPER. `ready_for_live=false`.

## Why DOM is optional, not required

1. **No consumer.** No module under `chad/` calls any IBKR DOM API. The
   guard test `chad/tests/test_box056_official_dom_level2_scope_guard.py`
   enforces this statically.
2. **No live-readiness gate.** `ops/live_readiness_publish.py` evaluates
   exactly these gates: `stop`, `feed`, `reconciliation`,
   `lifecycle_truth`, `execution_quality`, `mutation_state`,
   `canary_state`, `chad_mode`, `operator_intent`, `scr`. None of them
   reads any DOM / depth / orderflow / order-book state.
3. **No strategy edge depends on DOM.** `alpha_options`, `alpha_futures`,
   `alpha_intraday`, `delta`, `delta_pairs`, `omega_macro`, and the
   alpha/beta crypto modules build signals from bar data, options chains,
   IV / Greeks, and macro features — not from Level 2 book imbalance.
4. **No paid data subscription is required for live activation.** The IBKR
   real-time L2 entitlement remains an *optional future edge*. The
   `PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md` decision explicitly defers
   the consumer build until the entitlement is provisioned and a live
   probe shows non-empty MES + MNQ `domBids` / `domAsks` with no
   IBKR Error 354.

## What this policy forbids (until DOM is proven)

1. No production code may call any IBKR DOM API:
   `reqMktDepth`, `cancelMktDepth`, `reqMktDepthExchanges`,
   `domBids`, `domAsks`, `updateMktDepth`, `updateMktDepthL2`.
2. No new live-readiness check may reference DOM / depth / orderflow /
   order-book.
3. No strategy may consume DOM-derived metadata.
4. No `chad-dom-daemon` / `ibkr_dom_provider` daemon, no
   `runtime/orderflow_state.json` publisher, no `orderflow_gate.py` may
   be created.
5. No IBKR client ID may be reserved for a future DOM provider.

The first two clauses are enforced statically by
`chad/tests/test_box056_official_dom_level2_scope_guard.py`. The remainder
are governance only.

## Unlock condition (manual external)

DOM may move from "scoped out" to "in scope" only after **all** of the
following hold simultaneously, at a CME market-hours probe (~Sunday 22:00
UTC or later):

- The trading account carries an active **IBKR CME Real-Time L2** (or
  equivalent CME market-depth) subscription.
- `reqMktDepth` returns no IBKR Error 354 for MES and MNQ.
- `MES.domBids > 0`, `MES.domAsks > 0`, `MNQ.domBids > 0`, `MNQ.domAsks > 0`.
- The probe evidence (raw log excerpt + timestamp) is attached to a
  follow-up unlock decision record under `ops/pending_actions/`.

The operator action required for unlock is **purchasing the IBKR L2
entitlement in the IBKR Client Portal**. This is a **manual external**
step that cannot be performed by CHAD or by this assistant. No assistant
action is permitted to enable, modify, or purchase market-data
subscriptions on the IBKR account.

If unlock is later granted, the guard tests in
`chad/tests/test_box056_official_dom_level2_scope_guard.py` must be
relaxed in the **same commit** that lands the DOM probe evidence — never
in advance.

## Verification command

```
source venv/bin/activate
python3 -m pytest chad/tests/test_box056_official_dom_level2_scope_guard.py -v
```

Last run (2026-05-21): 3 passed in 0.95s.

## Pending Action

This document is a **policy** only. There is **no runtime config change**
to apply. No service restart is required. No operator approval is needed
beyond review. CHAD remains PAPER. Live trading remains NOT authorized.
`ready_for_live=false`. Box 57+ remain open.

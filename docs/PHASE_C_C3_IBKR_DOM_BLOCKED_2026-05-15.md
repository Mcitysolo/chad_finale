# Phase C Item 3 — IBKR DOM Consumer Blocked

**Decision date:** 2026-05-15
**Author:** CHAD Engineering (Phase C pre-flight audit + live entitlement probe)
**Scope:** Phase C Item 3 (C3) — IBKR DOM / Level 2 consumer

---

## 1. Status

**BLOCKED / DEFERRED**

C3 is not authorized for implementation. Implementation is gated on a paid IBKR
market-depth (Level 2) entitlement that does not currently exist on the trading
account. The decision is recorded here so future audits do not interpret the
absence of a DOM consumer as an oversight.

---

## 2. Evidence

The following observations come from the Phase C pre-flight audit and a live
terminal probe against the running IBKR Gateway on 2026-05-15.

- `reqMktDepthExchanges` is available in `ib_async`, confirming the client-side
  API surface is present.
- IBKR Gateway is reachable at `127.0.0.1:4002` (paper account, default Gateway
  port).
- `MES JUN 2026` qualified successfully via contract qualification.
- `MNQ JUN 2026` qualified successfully via contract qualification.
- `reqMktDepth` returned **IBKR Error 354: "Requested market data is not
  subscribed."** for both MES and MNQ.
- `domBids` and `domAsks` were both empty (length 0) for MES and MNQ throughout
  the probe window.
- Conclusion: the IBKR account lacks the market-depth entitlement required for
  any DOM / Level 2 consumer.

Phase C pre-flight audit independently confirmed:

- `ib_async` exposes `reqMktDepth` and `reqMktDepthExchanges`.
- CHAD has no existing DOM / order-book implementation to extend or reuse.
- Current IBKR posture has no paid market-depth subscription.
- C3 therefore requires the paid IBKR Level 2 / market-depth entitlement
  before any implementation work begins.

---

## 3. Decision

- C3 implementation is **not authorized** until an IBKR market-depth
  subscription is confirmed against the trading account.
- **No strategy wiring** is to be attempted. `alpha_futures` and
  `alpha_intraday` must not consume DOM-derived signals while this block is in
  force.
- **No client ID is to be reserved** for a DOM provider daemon. Reserving a
  client ID now would create a dormant slot and a misleading signal that DOM
  is in flight.
- No DOM daemon, gate, or publisher is to be created.

This decision applies to the entire CHAD codebase; nothing under `chad/`,
`deploy/`, `ops/`, `config/`, `runtime/`, the systemd units, or tests is to be
modified as part of recording the block.

---

## 4. Unlock Condition

C3 may reopen only after a live entitlement probe shows **all** of the
following simultaneously:

- `MES.domBids > 0`
- `MES.domAsks > 0`
- `MNQ.domBids > 0`
- `MNQ.domAsks > 0`
- No `Error 354` and no other subscription error returned from
  `reqMktDepth` during the probe window.

The probe evidence (raw log excerpt and timestamp) must be attached to the
unlock decision record before any implementation work begins.

---

## 5. Future Implementation Outline (post-unlock)

When (and only when) the unlock condition above is satisfied, the following is
the intended shape of the C3 build. Recorded here so the unlock work can be
scoped without re-deriving the design.

1. Reserve a dedicated `DOM_PROVIDER` IBKR client ID, distinct from the
   execution and market-data client IDs already in use.
2. Build an isolated `ibkr_dom_provider.py` daemon. Isolation is required so
   DOM subscription churn cannot impact the execution or market-data
   connections.
3. Publish `runtime/orderflow_state.json` from the daemon as the SSOT for any
   downstream consumer. Treat this file like the other Phase B publishers
   (schema_version, atomic write, monotonic timestamps).
4. Add `orderflow_gate.py` to translate raw DOM into a gate signal. This gate
   must be **soft** — it contributes confidence metadata, it does not
   hard-block trades.
5. Only after (1)–(4) are stable should `alpha_futures` and `alpha_intraday`
   read the orderflow confidence metadata. No hard blocking in v1; orderflow
   is purely an additive signal.

---

## 6. Recommended Immediate Phase C Path

Given C3 is blocked, the recommended Phase C ordering is:

- **Proceed:** C1 — read-only Kraken Futures intel publisher. No entitlement
  block; data is public.
- **Defer:** Coinglass integration until an API key / paid plan is in place.
- **Defer:** IBKR DOM (this item, C3) until the Level 2 subscription
  entitlement is confirmed via the unlock probe in section 4.

This keeps Phase C forward progress on the only currently-unblocked item
without manufacturing a half-implemented DOM consumer that would have to be
unwound later.

---

## 7. References

- Phase C pre-flight audit (2026-05-15).
- Live IBKR terminal probe (2026-05-15) — MES JUN 2026, MNQ JUN 2026,
  Error 354 observed.
- CHAD Unified SSOT v9.2 (2026-05-15) — Phase B publishers live; Phase C
  planning underway.

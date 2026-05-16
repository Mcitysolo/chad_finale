# Phase C Kraken Futures — Canadian Jurisdiction Block

**Decision date:** 2026-05-16
**Author:** CHAD Engineering (Phase C jurisdiction review)
**Scope:** Phase C Item 1 (C1A / C1B / C1C) — Kraken Futures
**Deployment context:** Montréal / Canada

---

## 1. Status

**BLOCKED FOR CURRENT CANADIAN DEPLOYMENT**

Kraken Futures live trading is **not authorized** for the current CHAD
deployment. Kraken Futures / perpetuals / derivatives / margin products are
not available to retail or unregistered accounts under the operator's current
Canadian jurisdiction. This is a regulatory / jurisdictional block, not a
technical block. The Phase C C1A / C1B / C1C scaffolds remain in the
repository but must not be promoted to live routing for this deployment.

This decision supersedes any prior wording in Phase C status documents that
described the C1 unlock as solely contingent on Kraken Futures credentials.
Possession of credentials is not sufficient when the underlying product is
not legally available to the account in question.

---

## 2. What remains allowed

- **Kraken spot remains allowed.** Kraken spot trading is not affected by
  this decision. Any future spot integration is out of scope for this
  document but is not blocked by it.
- **C1A public Kraken Futures intel remains allowed as read-only market
  data.** The `kraken_futures_intel_publisher` continues to consume the
  public derivatives ticker endpoint. No credentials are touched. No private
  endpoint is called. The artifact `runtime/kraken_futures_intel.json` and
  the timer `chad-kraken-futures-intel-refresh.timer` may continue to
  operate as intelligence-only inputs.
- Treating public Kraken Futures intel as a **read-only market intelligence
  signal** for strategies that trade other venues (e.g. CME micros via IBKR)
  remains acceptable in principle, subject to normal strategy review.

---

## 3. What is blocked

- **Kraken Futures live trading is blocked for the current Canadian
  deployment.** No live orders, no margin positions, no perpetuals, no
  derivatives routing through Kraken under this deployment.
- **C1B adapter scaffold (`KrakenFuturesAdapter`, `KrakenFuturesClient`)
  remains dormant.** It must not be wired into live order routing,
  strategy execution, the orchestrator, or any systemd service while this
  block stands. The scaffold persists for code-archaeology and future
  optionality only.
- **C1C authenticated smoke scaffold (`kraken_futures_auth_smoke`) must not
  be used against the operator's Canadian Kraken account.** Running the
  authenticated smoke under the current jurisdiction provides no useful
  signal because the account cannot legally hold Kraken Futures positions
  regardless of API capability.
- **No strategy routing to Kraken Futures is authorized.** No strategy may
  import `KrakenFuturesAdapter` or `KrakenFuturesClient` for the purpose
  of order placement, sizing, or position management while this block
  stands.

---

## 4. Future unlock

Unlock is **only** possible when **all** of the following hold:

1. A legally eligible entity, account, and jurisdiction exists for Kraken
   Futures / derivatives / margin trading (i.e. not the current Canadian
   retail posture).
2. Operator has explicitly approved Kraken Futures live trading under that
   eligible entity in writing.
3. The eight technical unlock conditions previously documented in
   `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md` §3.D still hold:
   credentials present, certified read-only endpoint, no-order smoke
   passing, payload validation, risk-manager approval, explicit
   execution-pipeline authorization, kill-switch and reconciliation rules.

Possession of API keys alone is **not** sufficient to unlock. Eligibility
of the underlying entity / account / jurisdiction is a separate and
prerequisite gate.

---

## 5. Forbidden actions while this block stands

- Do not wire `KrakenFuturesAdapter` or `KrakenFuturesClient` into the live
  routing path.
- Do not enable any Kraken Futures live order placement.
- Do not run `chad/tools/kraken_futures_auth_smoke.py` against the
  operator's Canadian Kraken account.
- Do not introduce a strategy whose execution path depends on Kraken
  Futures order submission.
- Do not document the C1 unlock as "credentials-only" — eligibility comes
  first.

---

## 6. Audit pointers

- Phase C C1A publisher: `chad/market_data/kraken_futures_intel_publisher.py`
- Phase C C1B adapter scaffold: `chad/execution/kraken_futures_adapter.py`
  and `chad/exchanges/kraken_futures_client.py`
- Phase C C1C auth smoke scaffold: `chad/tools/kraken_futures_auth_smoke.py`
- Status lock referencing this block:
  `docs/CHAD_PHASE_C_STATUS_LOCK_2026-05-16.md`
- Prior scaffold docs (unchanged by this decision):
  `docs/PHASE_C_C1B_KRAKEN_FUTURES_ADAPTER_SCAFFOLD.md`,
  `docs/PHASE_C_C1C_KRAKEN_FUTURES_AUTH_SMOKE.md`

End of jurisdiction block notice.

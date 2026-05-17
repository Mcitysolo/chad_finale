# CHAD External Blockers Pinned — 2026-05-17

**Date:** 2026-05-17
**Author:** CHAD Engineering (pre-SSOT external-blocker audit)
**Purpose:** Pin the *only* items that remain unresolvable inside Channel 2
(code/docs/tests) so that the next SSOT can clearly distinguish engineering
work from external dependencies.

---

## 1. Status

**ONLY TRUE EXTERNAL BLOCKERS REMAIN**

Every other Phase A/B/C/D item is either complete or has a clearly bounded
Channel 2 closure path. The four items listed below are gated by *external*
dependencies — market hours, jurisdiction, paid subscriptions, or operator
authorisation — and cannot be closed by Claude or by additional engineering
work alone.

---

## 2. Channel map (precondition for understanding the rest of the doc)

CHAD uses three orthogonal channels for unblocking work. Each blocker below is
tagged with the channel(s) that can resolve it.

- **Channel 1 — Runtime probes.** Live runtime checks: `systemctl`, IBKR
  Gateway probes, `scripts/probe_*` read-only scripts, runtime artefact
  inspection. Cannot perform paid-subscription unlocks. Cannot grant
  jurisdictional approval.
- **Channel 2 — Code, docs, tests.** All engineering work that lives in the
  repository: source, tests, design docs, status docs. This is the only
  channel Claude operates in.
- **Channel 3 — External / subscriptions / accounts.** API keys, market-data
  subscriptions, account permissions, jurisdictional onboarding, broker
  entitlements. Requires the operator (and frequently a counterparty).

---

## 3. IBKR DOM (Phase C Item 3 — C3)

- **Status:** **PENDING / BLOCKED** until live MES/MNQ DOM rows return from
  the IBKR Gateway during regular CME hours.
- **Channel:** Channel 1 (probe) + Channel 3 (entitlement, if probe still
  fails).
- **Trigger to retry:** Channel 1 probe of MES + MNQ depth book after the CME
  Sunday/Monday open at approximately **Sunday 6 PM ET / 22:00 UTC**.
- **Unlock proof (all four conditions must hold):**
  - `MES domBids > 0`
  - `MES domAsks > 0`
  - `MNQ domBids > 0`
  - `MNQ domAsks > 0`
  - No `Error 354` ("Requested market data is not subscribed") observed.
- **Until then, do NOT build:**
  - A DOM publisher / `chad-dom-daemon`.
  - An `orderflow_gate` consumer.
  - Any strategy that consumes the (non-existent) DOM feed.
- **Reference:** `docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md`.

---

## 4. Kraken Futures live trading (Phase C Item 1 — C1, live execution leg)

- **Status:** **PERMANENTLY BLOCKED** for the current Canadian Kraken
  deployment.
- **Channel:** Channel 3 only (jurisdiction). No Channel 2 work can unblock
  this.
- Kraken remains **spot-only** for the current account.
- The Kraken Futures **public intelligence feed** (`kraken_futures_intel.v1`,
  `chad-kraken-futures-intel-refresh.timer`) remains **read-only allowed** and
  is currently healthy.
- **Forbidden:**
  - Do not configure the current Canadian Kraken account for futures, perps,
    derivatives, or margin.
  - Do not enable the `kraken_futures_adapter` scaffold for live execution.
  - Do not run the C1C authenticated smoke against a Canadian account with
    futures-trading intent.
- **Future unlock path (out of scope for current deployment):** a legally
  eligible entity, account, and jurisdiction, plus a separate operator-led
  approval. Until that is in place, treat this as permanently blocked.
- **References:**
  - `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`
  - `docs/PHASE_C_C1B_KRAKEN_FUTURES_ADAPTER_SCAFFOLD.md`
  - `docs/PHASE_C_C1C_KRAKEN_FUTURES_AUTH_SMOKE.md`

---

## 5. Coinglass (Phase C Item 2 — C2)

- **Status:** **BLOCKED** until a paid API plan / key is procured.
- **Channel:** Channel 3 (subscription decision and key).
- Public endpoints are unusable / deprecated for the data CHAD would consume.
- **Forbidden:**
  - **No code should be built keyless.** Do not scaffold a publisher, adapter,
    or consumer against Coinglass while no key exists.
  - Do not consume any public Coinglass endpoint as a stand-in.
- **Trigger to retry:** operator decision on whether to subscribe; if yes,
  Channel 3 delivers a key, after which Channel 2 designs the publisher.

---

## 6. Live BAG execution (Phase D Item 2)

This item is **listed here for clarity**, but it is **NOT** a true external
blocker. It is engineering-gated and is being closed through Channel 2 in
incremental Tiers.

- **Status:** Engineering-gated, **not external-only**.
- **Channel:** Channel 2 (multi-tier design + adapter + tests). Channel 3 is a
  *secondary* dependency for full broker-mid quote enforcement (Error 10091
  / IBKR market-data subscription).
- **Current Tier ladder:**
  - Tier 1 (typed `OptionsSpreadSpec`) — **complete**.
  - Tier 2 (LMT discipline; no MKT BAG submits) — **complete**.
  - Tier 3A (offline `bag_quote_check`) — **complete**.
  - Tier 3B (live-readonly probe script) — **complete and validated 2026-05-17**.
  - Tier 3C (BAG limit-price unit normalisation) — **next authorised
    workstream; design + docs only**.
  - Tier 3D (adapter quote enforcement) — depends on Tier 3C.
  - Tier 4 (bracket / failsafe exit + `spread_id` reconciliation) — pending.
  - Tier 5 (live BAG fill harness) — pending; requires operator GO.
- **Forbidden until all tiers pass:**
  - No live BAG execution.
  - No adapter quote enforcement until Tier 3C lands.
  - No use of contract-dollar debit values as IBKR BAG `lmtPrice`.
- **Reference:** `docs/PHASE_D_ITEM2_BAG_QUOTE_PROBE_RESULTS_2026-05-17.md`.

---

## 7. Channel map summary table

| Blocker | Channel | Resolvable by Claude alone? | Trigger to retry |
|---|---|---|---|
| IBKR DOM (C3) | 1 (+3) | No (requires market-hours probe) | CME open ~Sun 6 PM ET / 22:00 UTC |
| Kraken Futures live (C1) | 3 only | No (jurisdiction) | New eligible entity/account + operator approval |
| Coinglass (C2) | 3 only | No (paid key) | Operator decision on paid plan |
| Live BAG execution (D2) | 2 (+3) | Partially — Channel 2 tiers continue | Full Tier 3C→5 ladder + operator GO |

---

## 8. Implications for the next SSOT

The next SSOT can state, with full justification, that:

- **All Phase A items** are closed.
- **All Phase B items** are live and healthy.
- **Phase C** is closed except for the three external blockers above.
- **Phase D** has its first two items in flight (dynamic universe scanner v1
  read-only; BAG hardening through Tier 3B), with bounded Channel 2 work
  remaining and **no** assumption that any external blocker is resolved.

No engineering work currently in flight depends on any of the three pinned
external blockers being unblocked.

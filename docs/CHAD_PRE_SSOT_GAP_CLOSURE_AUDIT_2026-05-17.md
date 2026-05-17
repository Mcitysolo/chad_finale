# CHAD Pre-SSOT Gap Closure Audit — 2026-05-17

**Date:** 2026-05-17
**Author:** CHAD Engineering (pre-SSOT closure audit)
**Purpose:** Confirm that the only remaining unresolved items are true
external blockers, and that the system is ready for a new SSOT.

---

## 1. Status

**READY TO CREATE NEW SSOT AFTER THIS DOC IS COMMITTED**

Phase A, B, and C are closed (with three external blockers pinned in
`docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`). Phase D has both
in-flight items at a clean, bounded stopping point: the dynamic universe
scanner is live in read-only observation mode, and the BAG hardening ladder is
through Tier 3B with the Tier 3C design as the next authorised workstream.

The full-suite baseline is **2114 passed**, matching the most recent green
baseline, and `full_cycle_preview` runs clean with no broker calls.

---

## 2. Baseline

- **HEAD commit:** `a9fb34a Fix BAG quote probe async contract qualification`.
- **Full-suite baseline:** **2114 passed, 4 warnings, 69.39s**
  (`python3 -m pytest chad/tests/ -q --tb=short`).
- **`full_cycle_preview` result:** clean — no broker calls, no orders,
  `orders_count: 0`, `total_notional: 0.00`, `intents_count: 0`. The preview
  itself confirms "No broker calls were made. This is a logical preview only."
- **Working tree state at the start of this audit:** clean
  (`git status --short` empty).

---

## 3. Phase A status — Signal quality

**Complete (5 / 5 items).** Delivered signal-intelligence items:

- **A1 — Stop-distance sizing** (`57c791c`): `alpha_futures` / `alpha` /
  `alpha_intraday` size on a stop-aligned basis when the tier budget activates;
  legacy behaviour preserved as the fallback.
- **A2 — Session-zone gating** (`dd6d7fa`): `alpha`, `alpha_intraday`,
  `alpha_futures` honour `primary_session_only` for entries; exits are never
  blocked; MCL/MGC handling preserved; shared `chad/utils/session.py`.
- **A3 — Pre-entry risk/reward gate** (`4eadf6e`): R/R floor enforced before
  entry in the signal pipeline.
- **A4 — Setup-family tagging** (`759ec5d`): setup-family metadata propagated
  through trades; consumed by `setup_family_expectancy`.
- **A5 — Float-aware liquidity gate** (`6c73191`): float / ADV constraints
  enforced ahead of entry sizing.

---

## 4. Phase B status — Intelligence feeds

**Complete / live (6 publishers + FMP scaffold).** Each publisher has a
systemd timer, a runtime artefact, a dashboard surface (where applicable), and
freshness/TTL classification:

| Publisher | Runtime artefact | Timer | Status |
|---|---|---|---|
| News intel (catalyst news) | `runtime/news_intel.json` | `chad-news-intel-refresh.timer` | live, fresh, `status=ok` |
| Relative strength | `runtime/relative_strength.json` | `chad-rs-refresh.timer` | live, fresh, `status=ok` |
| Intraday RVOL scanner | `runtime/volume_scan.json` | `chad-volume-scan.timer` | live, fresh, `status=ok` |
| Crypto derivatives | `runtime/crypto_derivatives.json` | `chad-crypto-derivatives-refresh.timer` | live, fresh, `status=ok` |
| Futures roll calendar | `runtime/futures_roll_state.json` | `chad-futures-roll-refresh.timer` | live, fresh, `status=ok` |
| Options Greeks metadata | `runtime/options_greeks.json` | `chad-options-greeks-refresh.timer` | live; daily cadence aligned with TTL (`759fd32`); fresh by current TTL |
| FMP earnings intelligence | `runtime/earnings_intel.json` | `chad-fmp-earnings-intel-refresh.timer` | live, `status=partial` (expected; FMP free-tier coverage), dashboard-visible (`fcb8656`) |

All Phase B feeds are consumed by `dynamic_universe_candidates` and `event_risk`
(both also fresh).

---

## 5. Phase C status — Cross-venue / depth intelligence

| Item | Status | Notes |
|---|---|---|
| **C1A — Kraken Futures public intel** | **Complete / live** | `runtime/kraken_futures_intel.json` (`kraken_futures_intel.v1`), 306 perps, fresh; `chad-kraken-futures-intel-refresh.timer` healthy. |
| **C1B — Kraken Futures adapter scaffold** | **Complete / DORMANT** | `chad/execution/kraken_futures_adapter.py` — scaffold only, never wired. |
| **C1C — Kraken Futures authenticated smoke** | **Complete / fail-closed** | `chad/tools/kraken_futures_auth_smoke.py` — fails closed in the absence of credentials. |
| **C1 live trading** | **BLOCKED** | Canadian jurisdiction; permanent for current deployment. See `docs/PHASE_C_KRAKEN_FUTURES_CANADA_BLOCKED_2026-05-16.md`. |
| **C2 — Coinglass** | **BLOCKED** | Requires paid API key. No keyless scaffold permitted. |
| **C3 — IBKR DOM** | **BLOCKED / PENDING** | Awaiting actual DOM rows / entitlement proof at the next CME open (~Sun 6 PM ET / 22:00 UTC). See `docs/PHASE_C_C3_IBKR_DOM_BLOCKED_2026-05-15.md`. |

External-blocker pinning lives in
`docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`.

---

## 6. Phase D status — Universe + complex execution hardening

| Item | Status | Evidence |
|---|---|---|
| **D1 — Dynamic universe scanner** (v1 observation-only) | **Complete (observation-only)** | Design `99a5084`; publisher `ce0c4d1`; dashboard surface `c4144f5`; `runtime/dynamic_universe_candidates.json` fresh, 25 candidates, all input feeds healthy. **Promotion logic intentionally deferred** — current implementation does not mutate `runtime/universe.json`. |
| **D2 Tier 1 — Typed `OptionsSpreadSpec`** | **Complete** | `51166d4`. |
| **D2 Tier 2 — LMT discipline** | **Complete** | `2ab03e7` (design `c7c7406`). |
| **D2 Tier 3A — Offline `bag_quote_check`** | **Complete** | `a5c5b35` (design `c620119`). |
| **D2 Tier 3B — Live-readonly quote probe** | **Complete and live-readonly validated 2026-05-17** | `9fb163b`, async-qualification fix `a9fb34a`; results in `docs/PHASE_D_ITEM2_BAG_QUOTE_PROBE_RESULTS_2026-05-17.md`. |
| **XGB veto model promotion workflow** | **Complete** | `5463c63` (plan `e7e083e`; dirty-tree decision `be0155d`). |

---

## 7. Open gaps that remain intentionally open

These are not regressions; they are scoped, bounded work items that the
current SSOT intentionally leaves open. Each has an owner (Channel) and a
documented path forward.

- **Dynamic scanner promotion logic** (D1.2) — observation-only today; any
  writeback into `runtime/universe.json` must protect the static
  `config/universe.json` fallback (Channel 2).
- **BAG unit normalisation / Tier 3C** — design-doc only; introduce
  `net_debit_contract_dollars` vs `net_debit_per_share` split (Channel 2).
- **BAG adapter quote enforcement** (Tier 3D) — depends on Tier 3C
  (Channel 2).
- **BAG bracket / failsafe exit** (Tier 4) — pending (Channel 2).
- **`spread_id`-aware reconciliation** — currently reconciliation is per-leg;
  BAG needs a spread-level identifier carried through `trade_closer_state`
  and `position_guard` (Channel 2).
- **Live BAG fill harness** (Tier 5) — pending; requires operator GO
  (Channel 2 + operator).
- **IBKR DOM entitlement / probe** (C3) — pending CME open probe
  (Channel 1 + 3).
- **Coinglass key** (C2) — pending operator decision on paid plan
  (Channel 3).
- **Kraken Futures Canada block** (C1 live) — permanent for current
  deployment (Channel 3).

---

## 8. Closed gaps

These gaps are closed and require no further action in the new SSOT beyond
referencing them:

- **XGB dirty-tree issue** — closed by `be0155d` decision record and the
  promotion workflow (`5463c63`).
- **XGB silent promotion issue** — closed by the explicit promotion workflow
  (`5463c63`) + artefact hygiene plan (`e7e083e`).
- **BAG MKT footgun** — closed by Tier 2 LMT discipline (`2ab03e7`).
- **BAG typed metadata gap** — closed by Tier 1 typed `OptionsSpreadSpec`
  (`51166d4`).
- **BAG quote validation module gap** — closed by Tier 3A offline
  `bag_quote_check` (`a5c5b35`).
- **BAG live-readonly quote probe script gap** — closed by Tier 3B probe
  (`9fb163b`, fix `a9fb34a`) and the 2026-05-17 results doc.
- **FMP earnings intelligence visibility gap** — closed by `a98a165` publisher
  and `fcb8656` dashboard surface.
- **Dynamic universe read-only visibility gap** — closed by `ce0c4d1`
  publisher and `c4144f5` dashboard surface.

---

## 9. Next authorized workstream

**Recommendation:** Phase D Item 2 **Tier 3C — BAG limit-price unit
normalisation design**.

- This is **design + docs only**, in line with the Phase D BAG hardening
  ladder. It must not modify the adapter, must not place orders, and must not
  enable adapter quote enforcement.
- Tier 3C is the only Phase D BAG step that can land without depending on an
  external blocker (it does not need broker-mid bid/ask data; it only needs
  the per-share / contract-dollar field split).
- Adapter enforcement (Tier 3D) is deliberately **not** the next step.

---

## 10. Forbidden assumptions for the new SSOT

The new SSOT **must not** assume any of the following:

- Do **not** assume Kraken Futures live trading is available.
- Do **not** assume IBKR DOM is available.
- Do **not** assume Coinglass is available.
- Do **not** assume BAG live execution is authorized.
- Do **not** assume the dynamic scanner may replace `runtime/universe.json`
  (it is observation-only).
- Do **not** assume the contract-dollar debit equals IBKR BAG lmtPrice
  (i.e. the contract-dollar debit equals IBKR BAG lmtPrice **only** when
  normalised per-share — see Tier 3C).

---

## 11. New SSOT readiness

**The system is ready for a new SSOT.**

- The full suite is green at the expected baseline (**2114 passed**).
- `full_cycle_preview` runs clean with no broker calls.
- Every consumed runtime feed is fresh by its own TTL (or, where stale, the
  stale file is non-blocking and explicitly explained — e.g.
  `setup_family_expectancy.json` is on a 24 h cadence and currently within the
  expected refresh window of the daily timer).
- All open gaps are **clearly documented, bounded, and non-blocking** for
  SSOT creation.
- The only remaining unresolved items are true external blockers, pinned in
  `docs/CHAD_EXTERNAL_BLOCKERS_PINNED_2026-05-17.md`.

After this doc is committed, the operator may proceed to create the new SSOT
with confidence that the open-gap surface is fully understood and bounded.

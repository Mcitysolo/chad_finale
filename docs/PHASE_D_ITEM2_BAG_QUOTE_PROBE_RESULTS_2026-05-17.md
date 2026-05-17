# Phase D Item 2 — BAG Quote Probe Results — 2026-05-17

**Date:** 2026-05-17
**Author:** CHAD Engineering (Phase D Item 2 Tier 3B live-readonly validation)
**Scope:** Phase D Item 2 — BAG/COMBO hardening, Tier 3B (live-readonly quote probe)
**Channel:** Channel 1 (runtime probe) + Channel 2 (script result analysis)

---

## 1. Status

**READ-ONLY LIVE QUOTE PROBE VALIDATED — NO LIVE ORDERS AUTHORIZED**

The Tier 3B probe script (`scripts/probe_bag_quotes.py`, committed at `9fb163b`,
async qualification fix at `a9fb34a`) has been exercised in both `--dry-run-fake`
and `--live-readonly` modes against IBKR paper. Connectivity, contract
qualification, and the offline `quote_check` path are validated end-to-end with
no order placement, no adapter mutation, and no production wiring touched.

This document is the artefact closing out the Tier 3B workstream and the entry
point for designing Tier 3C (BAG limit-price unit normalisation).

---

## 2. Probe summary

- `scripts/probe_bag_quotes.py` was used.
- `--dry-run-fake` validated successfully (synthetic quotes, no network).
- `--live-readonly` validated script connectivity and the quote_check path
  against IBKR paper (read-only, no orders).
- No orders were submitted.
- No adapter enforcement is active. The probe operates in **observation-only**
  mode; the production BAG path remains untouched by the probe.
- The probe explicitly does **not** subscribe to live order data and does not
  call any execution adapter; it only qualifies contracts, reads quote
  snapshots, and runs the offline `quote_check` validator on the computed
  spread debit.

---

## 3. Key finding — BAG limit price unit mismatch

The most important result of the probe is a **unit mismatch** between the
intuitive "contract-dollar debit" and the IBKR BAG `lmtPrice` field.

- `--limit-price 350` failed (rejected as out-of-band by the offline
  `quote_check` validator).
- `--limit-price 3.97` passed.

**Interpretation:**

- IBKR `BAG` / option-spread `lmtPrice` is **per-share option-price debit**
  (i.e. the same units in which the bid/ask/last quotes are denominated for
  each individual option leg), not the total contract-dollar debit.
- A `$350` *contract-dollar* debit corresponds to a `3.50` option-price
  per-share debit (a standard US equity option contract represents 100 shares,
  so `option_price × 100 ≈ contract dollars` for a single contract).
- Passing `350` as `lmtPrice` therefore expresses an absurd marketable level
  (>> any reasonable option mid), which the offline `quote_check` correctly
  rejected as out-of-band relative to leg mids.
- Passing `3.97` correctly expresses the per-share spread debit and falls
  within the legs' implied mid band.

This finding is **not** a bug in the probe — the probe revealed exactly the
unit footgun that Tier 3C must normalise before any adapter-level enforcement
can be safely turned on.

---

## 4. Live-readonly quote findings

- Both SPY option leg contracts qualified successfully via `ib_async.qualifyContractsAsync`.
- IBKR returned **delayed last prices** for both legs (paper/no-live-subscription).
- **Bid/ask were unavailable** for the snapshot the probe took.
- IBKR emitted **`Error 10091`** — additional market-data subscription
  required for some option quote fields (bid/ask depth-of-book / unrestricted
  quotes).
- The offline `quote_check` validator used the **leg last-price fallback** to
  compute a per-share mid and a per-share computed debit.
- Because we are fallback-on-last (not mid), the implied band the validator
  uses for the spread is necessarily slack; this is acceptable for Tier 3B
  validation but is **not** acceptable as the basis for live enforcement.

---

## 5. Result examples

Direct probe outputs (paraphrased, structured for clarity):

- Long leg last:   **15.34**
- Short leg last:  **11.37**
- Computed per-share spread debit: **3.97** (= 15.34 − 11.37)
- `--limit-price 350` → **REJECTED** (out-of-band; treats as ~$350/share debit,
  which is >> the leg-mid implied band).
- `--limit-price 3.97` → **ACCEPTED** (within the leg-mid implied band).

---

## 6. Implication for Tier 3C

The Tier 3C workstream (BAG limit-price unit normalisation design) must do
**all** of the following before any adapter quote enforcement is wired:

- **Do not** pass `net_debit_estimate` contract-dollar values directly to the
  IBKR adapter's `lmtPrice` field.
- Introduce an **explicit unit-normalisation layer** between strategy-level
  debit estimates (typically expressed in contract dollars or per-share
  premium) and the IBKR adapter's `lmtPrice`.
- Recommended field split on `OptionsSpreadSpec` (or its downstream order
  intent):
  - `net_debit_contract_dollars` — the strategy-facing field, in $/contract.
  - `net_debit_per_share` — the broker-facing field, in $/option-price unit.
- The adapter's quote enforcement (when Tier 3D arrives) **must** consume
  `net_debit_per_share` only. Contract-dollar values must never reach
  `lmtPrice`.
- The offline `quote_check` validator (committed at `a5c5b35`) must continue to
  be the canonical pre-order gate; the per-share/contract-dollar split is a
  precondition for adapter enforcement, not a replacement for `quote_check`.
- Tier 3C is **design + docs only**. No adapter mutation, no order placement,
  no Channel 1 runtime probes beyond what Tier 3B already established.

---

## 7. Remaining blockers

The following remain open after Tier 3B closeout and must be tracked
independently in the pre-SSOT gap-closure audit:

- **True bid/ask broker-mid unavailable** without additional IBKR market-data
  subscription (Error 10091; Channel 3 decision).
- **Adapter quote enforcement not implemented** (Tier 3D; depends on Tier 3C).
- **Bracket / failsafe exit not implemented** for BAG entries.
- **`spread_id` reconciliation not implemented** in the position guard /
  reconciliation publisher (currently only per-leg).
- **Live BAG fill harness not implemented** (no Channel 1 closed-loop
  validation of a live BAG submit→fill→close round-trip).
- **Live BAG execution not authorized** by the operator and not gated open by
  `LiveGate`.

---

## 8. Forbidden actions

Until each Tier 3C/3D piece lands and is independently verified, the
following are explicitly **forbidden**:

- **No live BAG orders.** The probe is `--live-readonly` only. The production
  BAG path is dormant.
- **No adapter quote enforcement** until unit normalisation (Tier 3C) is
  designed, reviewed, and merged.
- **No use of 350-style contract-dollar values** as `lmtPrice` anywhere in the
  execution path. Any caller that constructs a BAG order intent must route its
  debit through the Tier 3C normalisation layer first.

---

## 9. Cross-references

- Probe script: `scripts/probe_bag_quotes.py` (`9fb163b`, fix `a9fb34a`).
- Tier 3B design: `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER3B_LIVE_QUOTE_PROBE_DESIGN_2026-05-17.md`.
- Tier 3 offline validator design: `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER3_QUOTE_CHECK_DESIGN_2026-05-17.md`.
- Tier 3 offline validator implementation: `a5c5b35` (`chad/options/bag_quote_check.py`).
- Tier 2 LMT discipline: `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER2_DESIGN_2026-05-17.md` + `2ab03e7`.
- Tier 1 typed `OptionsSpreadSpec`: `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER1.md` + `51166d4`.
- Phase D readiness baseline: `docs/CHAD_PHASE_C_CLOSEOUT_AND_PHASE_D_READINESS_2026-05-16.md`.

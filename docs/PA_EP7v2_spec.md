# PA-EP7v2 — Kill the $100 placeholder emission at root (delta + router + reconciler + marker-aware rejection)

**Status:** BACKLOG (post-Day-0). Supersedes the withdrawn delta-only PA-EP7.
**Posture:** Repo + surgical production changes, test-gated, one block at a time. No service actions. Instrumentation/correctness only — must not change risk-path sizing/gating behavior.
**Authority for deferral→activation:** operator GO required before any sub-PA in this spec is implemented.

---

## 1. Objective

Eliminate the `$100` no-live-price placeholder fingerprint **at the point of emission**, so validation-grade evidence no longer depends on downstream rejection/exclusion. Today the leak is LIVE (delta ~2–6/day, reconciler ~1–3/day, latest 06-12) but 100% contained as `status=rejected`. This PA removes the emission and hardens the containment as defense-in-depth.

---

## 2. Established evidence base (from the 2026-06-12 verification + PA-EP7-T)

| # | Fact | Evidence (file:line) |
|---|------|---------------------|
| E1 | **`TradeSignal` has NO `price` field** — `TradeSignal(price=…)` raises `TypeError`. The original PA-EP7 premise was false. | `types.py:102-122`; runtime `TypeError` reproduced |
| E2 | The real root: the **router never populates `RoutedSignal.price`**, so it defaults `0.0`. | `signal_router.py:196-206` (no `price=` kwarg); `types.py:152` (`RoutedSignal.price=0.0`) |
| E3 | `TradeSignal.meta` survives to `merged_meta` **for the primary strategy only**. | capture `signal_router.py:130-131`; retrieve `:187`; pass `:205` |
| E4 | delta `px` is in scope at all 5 exit sites (570/616/632/649/665) + entry (721). | `delta.py:552` resolve; consumed 590/645/661/678 |
| E5 | **Containment is numeric-fingerprint-only.** `normalize` rejects placeholders via 5a (`cached<=0` AND `symbol∈_LIQUID_PRICED_EQUITIES` AND `fill_price≈100`) or 5b (`cached>0` AND `>50%` deviation). The `trust_state`/`placeholder_fill_price`/tag markers are **outputs**, never rejection inputs. | 5a `paper_exec_evidence_writer.py:~1725-1746`; 5b `:~1763-1789`; markers set `:1741/:1786` |
| E6 | **Gap:** a placeholder the detector misses passes as a fill **and now gets a modeled fee** (post-PA-EP1). | locked in `test_pa_ep7t_placeholder_containment.py` cases C1/C2/C3 |
| E7 | reconciler emits ~1/3 of the records via its **own** path, not delta→router. | scan 14-day window; `position_reconciler.py:514-515` (normalize→write); abstain at `:468-476` does not cover all cases |

---

## 3. Gate 0 — Blocking pre-verify (no code until ALL pass)

The PA-EP7 false-premise lesson: **verify the field actually flows end-to-end before writing the fix.**

- **G0.1 — Confirm `RoutedSignal.price` actually reaches the placeholder-assignment point.** Trace `RoutedSignal.price` → intent/order build in `live_loop.py` → the line that assigns `_PLACEHOLDER_FILL_PRICE`/`$100`. Quote the exact assignment site and prove that a nonzero `RoutedSignal.price` would prevent it. **If `live_loop` ignores `RoutedSignal.price` and derives price elsewhere, STOP — populating `RoutedSignal.price` is necessary but not sufficient; re-scope to the actual assignment site.**
- **G0.2 — Locate the `$100` literal.** Find where the priceless order resolves to `100.0` (the writer constant `_PLACEHOLDER_FILL_PRICE=100.0` is the *forensic* value; confirm whether `live_loop`/order-build assigns 100.0 independently or inherits it). Quote it.
- **G0.3 — Confirm the router's price sources.** Determine what price the router can read at routing time (does it have `ctx`/`price_cache.json`/a price provider in scope at `signal_router.py:~180`?). This decides Architecture A vs B in §4.
- **G0.4 — Confirm the reconciler emission mechanism.** Establish *why* `position_reconciler` still emits $100 placeholders despite the `:468-476` abstain — is it a different close path, a race, or pre-fix residue? Quote the construction site feeding `normalize` at `:514`.

Any G0 mismatch → STOP, report, re-scope. Do not proceed on assumption.

---

## 4. Architecture decision (resolve in Gate 0, then lock)

**The merge-robustness requirement (E3) rules out a primary-only meta carrier as the sole mechanism.** Two viable architectures:

- **Arch-A (router-authoritative, recommended):** the router populates `RoutedSignal.price` from a **symbol-keyed price lookup it controls** (price_cache.json / price provider), independent of which strategy is primary. Merge-robust by construction. delta meta-carry becomes an optional *preference* (strategy-resolved price used when present, lookup as fallback).
- **Arch-B (meta-carrier only):** delta stamps `meta["price"]=px`; router reads `merged_meta.get("price")`. Simpler, but **brittle** — price dropped when delta isn't primary (E3). Acceptable only if G0.3 shows the router has no price source AND the operator accepts the multi-strategy caveat.

Recommendation: **Arch-A with meta-preference.** Final call is an operator decision (§8).

---

## 5. Sub-PA decomposition (one block at a time)

| Sub-PA | Scope (files) | Risk class | Depends on | Summary |
|--------|--------------|-----------|-----------|---------|
| **A — Router populates `RoutedSignal.price`** | `chad/utils/signal_router.py` + test | repo-only, test-gated (router is a pure transform; no broker) | Gate 0 | At `:196-206`, set `price=` from the chosen source (Arch-A lookup, or `merged_meta.get("price",0.0)` for Arch-B). **This is the load-bearing fix.** |
| **B — delta carries `px` in meta** | `chad/strategies/delta.py` + test | repo-only, test-gated | A (only meaningful once router reads it) | Stamp `meta["price"]=px` on the 5 exit sites **and** the entry site (721) — entry has the same `RoutedSignal.price=0` exposure. `px` in scope (E4). Pure additive to `meta`. |
| **C — reconciler emission** | `chad/core/position_reconciler.py` + test | **requires service restart** (live reconciler path) | Gate 0 G0.4 | Populate the close-evidence price (or abstain) at the `:514` construction so reconciler stops emitting priceless $100 records. Exact fix shape pending G0.4. |
| **D — marker-aware rejection (defense-in-depth)** | `chad/execution/paper_exec_evidence_writer.py` + flip PA-EP7-T 1c tests | repo-only, test-gated | independent | In `normalize`, treat `extra.trust_state=="PLACEHOLDER"` / `placeholder_fill_price≈100` / tag `placeholder` as a **rejection input** (not just output), and ensure `_apply_modeled_commission` never fees a placeholder. Closes the E6 gap (C1/C2/C3). |

**Why D is separate and still wanted even with A/B/C:** A/B/C remove the *emission*; D removes the *latent exposure* that any future non-allowlisted symbol carrying a $100 placeholder slips through as a fee-bearing fill. Belt and suspenders.

---

## 6. Sequencing & dependencies

1. **Gate 0** (read-only) — resolves Arch-A/B and the reconciler mechanism. Blocking.
2. **A** then **B** (B is inert until A reads the price). If Arch-A, A is self-sufficient and B is an enhancement that can follow or be dropped.
3. **D** — independent; can land first (it's the cheapest, repo-only, and immediately hardens containment). Reasonable to do D **first** as a safety net before touching the emission paths.
4. **C** — last (restart-gated); needs its own GO per governance rule 7.

Recommended order: **D → A → B → C**.

---

## 7. Test strategy

- **A:** unit on `SignalRouter.route` — given input signals for a symbol, the emitted `RoutedSignal.price` is the expected resolved price (>0). Multi-strategy merge test proving merge-robustness (Arch-A) or documenting the primary-only limitation (Arch-B). Red-then-green (price 0.0 → resolved).
- **B:** per-exit + entry test that the delta signal's `meta["price"]==px` (mirror the removed PA-EP7 per-branch harness — monkeypatch `_ema`/`_atr`/`_extract_ohlc`, drive each of the 5 exits + entry). Red-then-green.
- **C:** reconciler close-evidence carries a real price (or abstains) — no $100 placeholder emitted. Red-then-green per G0.4 shape.
- **D:** **flip the three locked GAP tests** in `test_pa_ep7t_placeholder_containment.py` (C1/C2/C3) from "stays fill / fee modeled" → "force-rejected / no fee." These tests were deliberately written to go red the moment D lands — that is the tripwire's payoff. Add the marker-aware rejection assertions.
- **Integration:** end-to-end — a delta exit → router → (mocked) order build no longer yields a `$100` placeholder / `status=rejected` record. This is the test the original PA-EP7 could not write because the fix was in the wrong layer.
- **Every sub-PA:** full suite green; `full_cycle_preview` clean; diff scoped to the named files + test only.

---

## 8. Open operator decisions

| # | Decision | One-line recommendation |
|---|----------|------------------------|
| 1 | Arch-A vs Arch-B (router pricing source) | **Arch-A** (router-authoritative, merge-robust) — pending G0.3 confirming the router can reach a price source. |
| 2 | Does B (delta entry path) get fixed, or exits only? | Fix **both** entry + exits — entry has identical `RoutedSignal.price=0` exposure; cost is one line per site. |
| 3 | D rejection semantics: reject vs `pnl_untrusted`-only on marker hit | **Reject** (mirror 5a/5b) — keeps a single containment contract; markers already co-occur with rejection in production. |
| 4 | C scheduling (restart-gated) | Sequence C **last**, behind explicit GO + a monitored restart per rule 7. |
| 5 | Day-0 relationship | Per the Evidence Pipeline gating set, EP-7 **kill gates Day-0** (clean fill stream at source). If Day-0 proceeds before PA-EP7v2, document that exclusion-by-tag is the interim Day-0 mechanism. |

---

## 9. Stop conditions

- Gate 0 mismatch (esp. G0.1: `RoutedSignal.price` does not flow to the placeholder site) → STOP, re-scope.
- Any sub-PA whose diff touches risk-path sizing/gating behavior → STOP (instrumentation only).
- Router change that alters routed `net_size`/attribution (not just `price`) → STOP.
- Test failure or out-of-scope diff → STOP, report, no commit.

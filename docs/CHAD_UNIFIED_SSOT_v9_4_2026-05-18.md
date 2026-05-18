# CHAD Unified SSOT v9.4
# Remediation-Batch Consolidation — GAP-001/002 Behavioral Pin, GAP-032 Lint Guard, Baseline Erratum

**Version:** 9.4
**Date:** 2026-05-18
**Status:** Active — Paper Epoch 2 / post-remediation forward-errata
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v9_3_2026-05-17.md` (v9.1/v9.2/v9.3 remain FROZEN; this doc states corrections forward)
**HEAD commit:** `a5311d27777d57c0e2fb2ac6dafa0326883e5834` (short: `a5311d27`)
**Branch:** `main`
**Live order posture:** Unchanged — `CHAD_EXECUTION_MODE=paper`, `ready_for_live: false`
**Lock type:** Forward-errata SSOT — no rewrite of prior frozen docs

---

## 0. Preamble — Why This Document Exists

This SSOT consolidates the remediation batch executed 2026-05-09 →
2026-05-18 that began with the GAP-001 / GAP-002 runtime fixes and
finished with the GAP-032 preventive lint guard. It also issues a
**forward erratum** correcting a test-count claim in v9.3 that did
not survive an independent A/B reconciliation.

The frozen v9.1 / v9.2 / v9.3 docs are **not modified**. Corrections
appear here only; readers of the older docs should follow the
forward-errata pointer in §6.

This document does NOT grant any new live-order authorization. It
does NOT alter live-gate state. Live promotion remains gated by the
items in §7 (deferred operator decisions) and §8 (operator-domain
blocker). The publisher continues to emit
`runtime/live_readiness.json` with `ready_for_live: false`.

---

## 1. Closures With Evidence Anchors

Every closure below carries a commit SHA and/or a verifier phase
number from this batch. No closure is recorded without an anchor.

### 1.1 GAP-001 — position_reconciler operator exclusion SSOT

| Aspect | Anchor |
| --- | --- |
| Runtime fix (unify on canonical operator exclusion SSOT) | commit `f02b0ed` — "Fix GAP-001: unify position_reconciler onto canonical operator exclusion SSOT" |
| Integrity re-audit (no pre-existing failure introduced) | Phase 38 A/B reconciliation (per-suite isolation across BASE=f02b0ed^, A=f02b0ed, B=6abf16b, HEAD=a5311d27 — identical profiles, zero regressions) |
| Behavioral verdict | **PINNED to GAP-020** weekday re-audit. The runtime structural unification is complete; behavioral final sign-off cannot proceed until the GAP-020 IBKR stop-bus latency operator-domain blocker (§8) is resolved on a weekday. |

### 1.2 GAP-002 — canonical SCR PnL Prometheus + truthful labels

| Aspect | Anchor |
| --- | --- |
| Runtime fix | commit `6abf16b` — "Fix GAP-002: expose canonical SCR PnL in Prometheus + truthful metric labeling" |
| Integrity re-audit | Phase 38 A/B reconciliation — same per-suite invocation at BASE/A/B/HEAD shows zero pass→fail transition across `6abf16b`; classification PRE_EXISTING for the one kraken failure observed (`test_intent_builder_btc_basic`) at all four commits |
| Status | **CLOSED** — runtime change is integrity-clean. |

### 1.3 GAP-005, GAP-006, GAP-014, GAP-024, GAP-027, GAP-031

Closed via the orchestrator + bar-provider material-fix workstream
captured in operator-led phases prior to this batch's docs slice:

| GAP | Anchor |
| --- | --- |
| GAP-005, GAP-006, GAP-014, GAP-024, GAP-027 | Phase 34 (orchestrator material fixes); Phase 36 (bar-provider restart soak + materiality proof — `chad-ibkr-bar-provider` MainPID=2880191 started 2026-05-18T12:11:21Z, universe materiality confirmed against `runtime/universe.json` screened 25-symbol set vs `config/universe.json` static 38; bar refresh observed for screened set, static-only equities NOT refreshed post-restart) |
| GAP-031 | Shadow-status no-restart treatment under the same workstream — observability change only, no service restart issued |

### 1.4 GAP-032 — `.wants/` regular-file corruption signature

| Aspect | Anchor |
| --- | --- |
| Elimination of the on-host corruption | Phases 26 / 29 / 31 (operator removal + re-enable as proper enable-symlinks; chad-scoped regular files = 0 on current host, confirmed Phase 37 §3) |
| Preventive lint guard (this batch) | commit `a5311d27` — "GAP-032: preventive systemd .wants/ lint guard (+ Phase-27 monotonic-only warning)". New artifacts: `chad/ops/systemd_wants_lint.py` (319 lines, stdlib-only, strictly read-only — no sudo, no systemctl mutation), `scripts/lint_systemd_wants_symlinks.sh` (13-line wrapper), `chad/tests/test_gap_032_systemd_wants_lint.py` (248 lines, 8/0 pass). Wired into `chad/core/full_cycle_preview.py` (preview payload field) and `chad/ops/reconciliation_publisher.py` (~5-min cadence refresh of `runtime/systemd_wants_lint.json`). No new systemd timer added (Phase-25 constraint: a new timer could itself become corrupted). One additive Pre-Live Operator step inserted into `CLAUDE.md` between former items 6 and 7. |
| Status | **CLOSED** — on-host eliminated; preventive guard live; chad-scoped regression detection now exists on the existing publisher cadence and in every preview run. |

### 1.5 NOT-A-BUG / corrected-memory closures

| GAP | Disposition | Anchor |
| --- | --- | --- |
| GAP-015 | NOT-A-BUG / corrected-memory | Pre-batch operator audit |
| GAP-017 | NOT-A-BUG / corrected-memory | Pre-batch operator audit |
| GAP-025 | Corrected-memory with **forward erratum** | See §4.4 |
| GAP-029 | **NOT-A-BUG** — premise refuted at both function level (Phase 30; `_lookup_paper_fill_price` is a pure read) AND full-surface level (Phase 39; zero read-modify-write callers of `runtime/price_cache.json` exist in production code; both writers — `chad/market_data/price_cache_refresh.py:317` and `chad/market_data/ibkr_price_provider.py:362` — use PID-suffixed tmp + `os.replace` atomic rename; the second writer's `IBKRPriceProvider.write_price_cache` has zero callers in production/tests/scripts and is effectively dead code). The race premise does not exist in the code. |

---

## 2. Corrected Test Baseline — Forward Erratum to v9.3

### 2.1 What v9.3 claimed

v9.3 stated the test baseline as **2,114 passing**, and a related
prompt referenced **"2124/0"** as the clean baseline. Both numbers
are now retracted as written.

### 2.2 What is actually true (Phase 38 evidence)

A Phase-38 isolation run in a scratch `git worktree` at HEAD
(`a5311d27`) measured:

- **Full `chad/tests/` clean-room (fresh worktree, no live runtime
  state)**: approximately **6 failed / 2,107 passed** at BASE
  (`f02b0ed^` = `2a523b6`), and **6 failed / 2,125 passed** at HEAD
  (`a5311d27`). The +18 pass delta from BASE → HEAD reconciles to:
  8 new tests in `test_gap_032_systemd_wants_lint.py` plus 10 from
  intervening commits.
- **Live tree at HEAD** (`/home/ubuntu/chad_finale`): **18 failed /
  2,114 passed**.

### 2.3 What the delta means

The 18-vs-6 delta between the live tree and the clean worktree is
**commit-independent**: it reflects pytest test-ordering / live-tree
session-cached state, not a regression introduced by any commit in
this batch. Per-suite invocation in isolation at all four commits
(`BASE`, `f02b0ed`, `6abf16b`, `a5311d27`) returns IDENTICAL
profiles — `alpha_options_meta_preservation` 8/0,
`futures_contract_resolver` 21/0, `kraken_execution` 13/1 (single
pre-existing failure `test_intent_builder_btc_basic`),
`omega_macro_execution_lane` 11/0.

### 2.4 Erratum

- The v9.3 figure **"2,114 passing"** describes the live-tree run
  with `CHAD_SKIP_IB_CONNECT=1` and is consistent with the
  live-tree observation in this batch; it is retained as
  contextually accurate for the live-tree invocation but is **NOT
  the clean-room number**.
- Any prior reference to **"2124/0"** as a clean baseline is
  **RETRACTED**. Use the Phase-38 clean-room figures
  (≈ 6 failed / 2,125 passed at HEAD; 6 failed / 2,107 passed at
  BASE) for any future regression comparisons.
- The retraction does **not** weaken the GAP-001 / GAP-002 closures:
  Phase 38 proves both are integrity-clean independent of the
  baseline number, by per-suite A/B at BASE/A/B/HEAD.

---

## 3. New GAPs Logged OPEN

### 3.1 GAP-030 — tests/ops uncollected + beta fixture (known-independent)

- **Status:** OPEN. Pre-existing test failure in `tests/ops/` /
  beta-fixture surface. Did not regress in this batch; explicitly
  carved out of Phase-37 by note: "do not 'fix' it here; only
  ensure THIS task adds no new failures to `pytest chad/tests/`".
- **Severity:** triage — does not affect runtime correctness.
- **Owner:** to be assigned at next operator triage.

### 3.2 GAP-033 — baseline unreliability / live-tree pollution (NEW)

- **Status:** OPEN. The live tree at `/home/ubuntu/chad_finale`
  shows 18 full-suite failures that do NOT reproduce in a clean
  `git worktree` at the same SHA (which shows 6 — itself a different
  set). Cause is consistent with pytest test-ordering interactions
  with cached `runtime/*.json` state, `__pycache__` residue, or
  ordering-dependent fixtures.
- **Implication for future audits:** Future "did this commit break
  X?" questions must use a clean worktree A/B (see Phase 38) rather
  than the live tree, until the root cause is isolated.
- **Severity:** triage — does not block live promotion directly, but
  any future test-count assertion in an SSOT must cite a clean-room
  invocation per §2.

### 3.3 (Both items above are logged here for v9.4 — no commit-level fix in this batch.)

---

## 4. Sub-Findings Logged

### 4.1 logrotate config-target coverage

Existing CHAD logrotate configuration covers the canonical service
log targets but does not blanket-cover every newly added publisher's
log path. Tracked as a sub-finding; remediation deferred. No
runtime impact today.

### 4.2 Macro-state no-op-skip mtime semantics

The macro-state publisher's no-op-skip path updates the artifact's
`ts_utc` field but does not always touch the file's filesystem
mtime, which can cause downstream freshness readers (that key off
mtime rather than payload `ts_utc`) to flag the artifact as stale
even though its content is fresh. Sub-finding only — no consumer is
currently flipping to RED on this discrepancy.

### 4.3 Monotonic-only-timer fragility (Phase-27 sub-finding, now permanently surfaced)

The Phase 37 lint guard's secondary check (chad `*.timer` with
`OnUnitActiveSec`/`OnBootSec`, no `OnCalendar`, AND empty
`NextElapseUSecRealtime` AND empty `NextElapseUSecMonotonic`) caught
**one** live warning on the current host:

```
{"unit": "chad-ibkr-price-refresh.timer",
 "reason": "monotonic_only_no_anchor"}
```

This is informational only — it does not flip the lint exit code
(exit code is keyed to chad-scoped regular-file corruption, of which
there are zero). It indicates the timer is monotonic-only without a
calendar anchor, which is the Phase-27-class fragility pattern.
**Follow-up TODO** (not in this batch): add an `OnCalendar` anchor
or otherwise prove the timer has actually armed an elapse — current
state simply means the lint has not observed it arm.

### 4.4 GAP-025 forward erratum

The v9.x SSOT line that recorded GAP-025 should be read in light of
the corrected-memory disposition in §1.5. No prior commit is being
revised; this paragraph is the forward-errata pointer.

---

## 5. Verification Ledger

| GAP | Evidence anchor | Verifier phase | Disposition |
| --- | --- | --- | --- |
| GAP-001 | commit `f02b0ed` | Phase 32 (operator) / Phase 38 (A/B) | Runtime CLOSED; behavioral PINNED to GAP-020 |
| GAP-002 | commit `6abf16b` | Phase 38 (A/B) | CLOSED |
| GAP-005 | bar-provider workstream | Phase 34 / Phase 36 | CLOSED |
| GAP-006 | bar-provider workstream | Phase 34 / Phase 36 | CLOSED |
| GAP-014 | bar-provider workstream | Phase 34 / Phase 36 | CLOSED |
| GAP-015 | corrected-memory | pre-batch | NOT-A-BUG |
| GAP-017 | corrected-memory | pre-batch | NOT-A-BUG |
| GAP-024 | bar-provider workstream | Phase 34 / Phase 36 | CLOSED |
| GAP-025 | corrected-memory + §4.4 errata | this doc | NOT-A-BUG (with forward note) |
| GAP-027 | bar-provider workstream | Phase 34 / Phase 36 | CLOSED |
| GAP-029 | price-cache full-surface read-only proof | Phase 30 (function) / Phase 39 (surface) | NOT-A-BUG |
| GAP-030 | (open) | — | OPEN — triage |
| GAP-031 | shadow-status no-restart | Phase 34 / Phase 36 | CLOSED |
| GAP-032 | host elimination + lint guard | Phase 26 / 29 / 31 + commit `a5311d27` (Phase 37) | CLOSED |
| GAP-033 | live-tree pollution observation | Phase 38 | OPEN — triage |

---

## 6. Frozen Predecessor Docs (Forward-Errata Pointer)

The following SSOTs are **FROZEN** and have **NOT** been modified by
this batch:

- `docs/CHAD_UNIFIED_SSOT_v9.1_2026-05-13.md`
- `docs/CHAD_UNIFIED_SSOT_v9.2_2026-05-15.md`
- `docs/CHAD_UNIFIED_SSOT_v9_3_2026-05-17.md`

Where v9.4 corrects a prior statement (notably the test-count
baseline; see §2), the correction is recorded here. The frozen docs
remain authoritative for the points-in-time they describe. Future
auditors should treat v9.4 §2 as the current canonical baseline
statement.

---

## 7. Deferred Operator Decisions (Tracked, Not Closed)

### 7.1 chad-full-cycle-refresh — epoch-boundary enable

Awaiting operator decision at next paper-epoch boundary. Not a code
change — purely an enable operation, deferred to keep the current
epoch's measurement clean.

### 7.2 chad-crypto-risk-off — epoch-boundary enable

Same disposition as 7.1. Awaiting operator decision at next
paper-epoch boundary.

Neither item is a closure; they are **deferred decisions**, tracked
here so future SSOTs can pick them up without re-derivation.

---

## 8. Operator-Domain Blocker (Top Pre-Live Operator Task)

### 8.1 IBKR stop-bus intermittent ~8s latency

The IBKR stop-bus surface intermittently exhibits ~8-second latency
spikes. While intermittent, this materially affects:

- **GAP-020 weekday re-audit** — cannot proceed cleanly until the
  latency is characterized on a weekday session.
- **GAP-001 behavioral final** — depends on the GAP-020 re-audit
  result.
- **Live promotion** — gated by GAP-001 behavioral final.

The blocker is **operator-domain** (broker / network / IB Gateway),
not code. No commit in this batch addresses it. It is recorded here
as the **top pre-live operator task** so the next operator-mode
session can pick it up.

### 8.2 Order of operations once the blocker is cleared

1. GAP-020 weekday re-audit (operator)
2. GAP-001 behavioral final sign-off (operator)
3. Re-evaluation of `ready_for_live` via the live-readiness
   publisher (no manual flip)
4. Live activation sequence per `CLAUDE.md` (which now includes the
   GAP-032 lint exit-0 requirement added in commit `a5311d27`)

Until step 3 reports `ready_for_live: true` on its own, the live
posture remains paper.

---

## 9. State of the Untouched Backlog

The following GAP IDs were **not touched by this batch** and carry
**no status claim** from v9.4. They retain whatever status v9.3 and
its predecessors assigned them. Future audits should re-verify
before changing disposition:

- GAP-003
- GAP-004
- GAP-007
- GAP-008
- GAP-009
- GAP-010
- GAP-011
- GAP-012
- GAP-013
- GAP-016
- GAP-018
- GAP-019
- GAP-021
- GAP-022
- GAP-023
- GAP-026

This is a **listing**, not a status update. v9.4 makes no claim
about any of these.

---

## 10. Live-Gate State at v9.4 Cutover

- `CHAD_EXECUTION_MODE` = `paper`
- `live_readiness.ready_for_live` = `false` (publisher-controlled,
  unchanged by this SSOT)
- SCR band: `CONFIDENT` (per v9.3 baseline)
- `paper_only`: `false`; `sizing_factor`: `1.0` (per v9.3 baseline)
- No live-gate state altered by this batch.

The retraction of the v9.3 test-count claim (§2) does NOT modify the
live-gate state and does NOT weaken the GAP-001 / GAP-002 closures
(see §1.1, §1.2, §2.3).

---

## 11. Batch Provenance

- **Batch window:** 2026-05-09 through 2026-05-18.
- **HEAD at v9.4 cutover:** `a5311d27777d57c0e2fb2ac6dafa0326883e5834`.
- **Anchoring commits in this batch:**
  - `f02b0ed` — GAP-001 runtime
  - `6abf16b` — GAP-002 runtime
  - `a5311d27` — GAP-032 preventive lint guard
- **Verifier phases referenced:** 26, 27, 29, 30, 31, 34, 36, 37, 38, 39.
- **Tag policy:** no new git tag created by v9.4 (this is a
  forward-errata SSOT; the active tags listed in `CLAUDE.md`
  remain authoritative).
- **Push status:** v9.4 is local-only at commit time. No push by
  this SSOT.

---

*End of v9.4. See §6 forward-errata pointer for relationship to
frozen v9.1 / v9.2 / v9.3.*

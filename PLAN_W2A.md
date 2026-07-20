# PLAN_W2A — Wave 2 Lane A: Books Cleanup (PHASE 1, PLAN ONLY)

**Branch:** `goal/wave2-books-cleanup` (forked from `main` @ `31f5517`, which already
contains the merged PFF1 Q1+Q2 fix).
**Worktree:** `/home/ubuntu/chad_w2a` — ALL Wave-2A build work happens here.
**Live checkout:** `/home/ubuntu/chad_finale` is on `main` (canonical); the box's
5-min timers execute *that* tree. **No edits to the live tree, and no runtime
mutation, in Phase 1–2.** The two runtime-mutating scripts (items 1 & 2) execute
only from a Phase-3 operator GO, and even then run **dry-run by default** against
the live `runtime/`.

**Status of the world (verified 2026-07-20):**
- `main` = `31f5517` = merge of the PFF1 branch → Q1 (harvester mirror dedup) and
  Q2 (untrusted total_pnl gate) are on main and in the live tree.
- BUT the live SCR reads from a long-running shadow server (`scr_state.json.source
  = http://127.0.0.1:9618/shadow`). It has **not reloaded the Q2 code**: live
  `scr_state` still shows `effective=73, total_pnl=+103.78`. Q2 becomes visible
  only after that server reloads (a Phase-3/operator concern, not a repo change).
- Today's runtime data is still polluted: `trade_history_20260720.ndjson` holds the
  6 phantom UNH closes; the gamma FIFO queue is empty; broker holds UNH=228.

---

## Baseline — CORRECTED after measuring (this is a methodology finding)

Running the full suite **in the worktree** does NOT reproduce the live-tree's
"known-5". Measured result in `/home/ubuntu/chad_w2a`:
**18 failed, 3828 passed, 5 skipped** — the failing SET is environment-dependent
because many tests are coupled to absolute live-tree paths / live runtime+config
while loading code from the worktree.

The worktree-18 failing set (record this — it is the real baseline):
```
test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage
test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes
test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop          [also fails live]
test_kraken_execution.py::test_intent_builder_btc_basic
test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified
test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only
test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only
test_repo_write_guard.py::  (6 tests — hardcodes REPO=Path("/home/ubuntu/chad_finale"))
test_routing_gates.py::test_e4_kraken_passive_order_params
test_routing_gates.py::test_e4_kraken_aggressive_order_params
test_tier_manager.py::test_1_scale_tier_at_182k                                       [also fails live; item 3 fixes]
test_tier_manager.py::test_2_starter_caps_at_2600                                     [also fails live; item 3 fixes]
test_tier_manager.py::test_7_promotion_immediate                                      [also fails live; item 3 fixes]
```

Cross-checked against the live tree (`/home/ubuntu/chad_finale`, main):
- The live-tree known-5 = `tier_manager`×3 + `quarantine_sidecar`×1 +
  `futures_expiry_gate`×1.
- **Only `tier_manager`×3 + `futures_expiry_gate`×1 fail in BOTH trees.**
- The worktree's other 14 are **worktree-execution artifacts** — verified PASS in
  the live tree for `repo_write_guard`×2 and `routing_gates`×2 (spot-checked); the
  rest (`backtest_unified`, `kraken_execution`, `phase_a_item5` "real SPY bar
  file", `pr03/pr04` "live posture", the other 4 `repo_write_guard`) are the same
  coupling class (they read `data/bars`, live posture/config, or hardcode the live
  repo path).
- Conversely the live-tree `quarantine_sidecar` failure does NOT reproduce in the
  worktree — the coupling cuts both ways.

**Methodology correction (this supersedes "green vs baseline by count"):**
Because the full-suite failing set is environment-specific, per-commit verification
uses a **failing-test-ID SET diff**, not a count:
1. Primary fast check: run the **affected test files** for the commit in the
   worktree and require them green (the new script tests use `tmp_path` and are
   coupling-free; the Stage-2 and tier tests are the ones we intend to change).
2. Gate: re-run the full suite in the worktree and require
   `new_failing_set ⊆ baseline_18` — i.e. **no new failing test ID**. Item 3 must
   REMOVE exactly the 3 `tier_manager` IDs from the set (→ worktree-15). None of
   the 14 artifacts overlap any W2A-touched test, so they cannot mask a regression.
3. Do NOT attempt to "fix the count" by de-coupling the 14 artifact tests here —
   that is a separate test-hygiene lane (see D8); Wave-2A only guarantees the set
   does not grow and shrinks by the 3 tier IDs.

This coupling is itself a finding worth surfacing: the suite is not hermetic
(absolute `/home/ubuntu/chad_finale` paths, live `runtime`/`config`/`data/bars`
reads), so worktree-based CI for any lane will see spurious failures until the
tests are made tree-relative.

---

## ITEM 1 — GHOST SCRUB (gated one-shot script)

**Goal:** make SCR **and** Stage-2 stop counting the 6 phantom UNH closes from the
PFF1-Q1 double-book, non-destructively (the trade_history is hash-chained; we do
NOT rewrite it).

**The exact 6 records to scrub** (`data/trades/trade_history_20260720.ndjson`, the
6 `gamma UNH` closes with NO `pnl_untrusted` marker — the re-buys closing phantom
shorts):

| seq | record_hash (prefix) | qty | pnl |
|-----|----------------------|-----|-----|
| 2 | `0c70922e1dc265aa` | 5  | +12.05 |
| 3 | `7bd3a872261ff9a4` | 5  | −3.55 |
| 4 | `e6f51f9923947089` | 32 | −22.72 |
| 5 | `687133278be20e0d` | 40 | −28.00 |
| 6 | `ead6c9f71dc97e01` | 80 | −56.00 |
| 7 | `3d0627208074e0dd` | 71 | −47.57 |

Sum = 233 sh, **−145.79**. (seq 1, the +625.17 untrusted seed close, is NOT
scrubbed — it is legitimately the seed close, already `excluded_untrusted=1`, and
its total_pnl leak is Q2's job.)

**Mechanism (non-destructive, hash-chain preserved):** the script writes
`runtime/quarantine_manifest_pff1_ghost_scrub.json`:
```json
{ "invalid_trades": [ {"record_hash": "<full 64-hex>", "reason": "pff1_phantom_double_book", ...} x6 ] }
```
- **SCR** honours this natively: `trade_stats_engine._load_all_trades` →
  `chad.utils.quarantine.get_exclusion_sets` unions every
  `runtime/quarantine_manifest_*.json` and drops any row whose `record_hash` is
  pinned, BEFORE parse (counted as `excluded_quarantined`). Record-hash pinning is
  surgical — it excludes exactly these 6 and nothing else.
- **Stage-2** does NOT read the manifest (`chad/validation/trade_log_adapter.py`
  judges rows purely by in-band markers via `trust_exclusion`; it reads
  `trade_history` and would ADMIT these 6 real-looking equity round-trips).
  → **A small repo change is required** so Stage-2 also honours the quarantine
  manifest (call `get_exclusion_sets` and drop pinned `record_hash` in
  `adapt_records`, tagged reason `quarantined`). This is NOT a live-loop change
  (Stage-2 is the on-demand edge-harness adapter). Covered by a new test.

**Script safety rails (identical for items 1 & 2):** dry-run default (prints the
full plan, mutates nothing); `--execute` requires typed token
`--confirm GHOST-SCRUB-PFF1`; fail-closed gates (exec mode ∈ {paper,dry_run};
refuse if any pinned record_hash not found; refuse if a manifest with the same
name already exists unless `--idempotent-ok`); timestamped `.bak` of any file it
writes; idempotent (re-run is a NOOP); writes an audit report to
`reports/ghost_scrub_pff1_<stamp>.json`. **No broker I/O, no order path.**

**Expected post-scrub numbers (the acceptance test):**
- `effective_trades`: 73 → **67** (−6). ✓ independent of Q2.
- `total_pnl`: reaches **−375.60 only once Q2 is active in the SCR server**:
  `+103.78 −625.17 (Q2 removes seed leak) −(−145.79) (scrub removes phantom) =
  −375.60`. **DEPENDENCY:** the −375.60 target requires the shadow SCR server to
  have reloaded Q2. Scrub alone (Q2 not yet active) yields `+103.78 + 145.79 =
  +249.57`. The plan's regression test asserts the **delta** (`effective −6`,
  `total_pnl_trusted += 145.79`) on a fixture, so it is deterministic; the live
  −375.60 is a Phase-3 verification gated on the server reload.

---

## ITEM 2 — UNH RE-ATTRIBUTION (gated one-shot script)

**Goal:** put the real +228 UNH long back under `gamma` with a **real, trusted**
cost basis so gamma and the exit overlay see and manage it — instead of it
surfacing as an untracked/`broker_sync` orphan.

**Does the existing tool already do this? — Checked. Partially, and not the way we
want.** `scripts/reconcile_ledger_to_broker.py` (the DRIFT-RECON tool) ADOPTs a
broker-truth position into the FIFO, BUT:
- attributes to the *dominant* FIFO strategy or `epoch3_adopted` (UNH's gamma FIFO
  is empty → it would land under `epoch3_adopted`, not `gamma`);
- marks the seed lot **`pnl_untrusted` + `UNATTRIBUTED_EPOCH3_ACCUMULATION`** at a
  *fabricated* basis — because in the general case the broker cost basis is unknown.

For UNH we have the **exact real re-buy fills**, so we can do better: a *trusted*
re-attribution. → Item 2 is a **new, narrow script** (not the general tool),
reusing the general tool's safety scaffold. Decision point D3 offers the reuse
alternative.

**Cost-basis derivation (cite exactly).** Broker truth: the account sold the 273
seed and re-bought 228; the held 228 are the last-bought lots. Real re-buy fills
(broker/harvester, account DUR119533):
- BUY 5 @ 425.57 (`8f5fc6d6`)
- BUY 40 @ 424.96 + 80 @ 424.96 + 80 @ 424.96 + 23 @ 424.95 (broker_sync `c400…/2970…/be0e…`) = 223
- **Broker-truth VWAP = (5·425.57 + 40·424.96 + 80·424.96 + 80·424.96 + 23·424.95) / 228 = 96 893.70 / 228 = 424.97.**

Alternative basis — the executor SIM re-buys (`58ff18a3` BUY 5 @ 422.45 + `799de7e5`
BUY 223 @ 425.57) → **VWAP = 425.50**. The two differ by ~$0.53/sh (~$121 on 228).
**Recommend the broker-truth VWAP (424.97)** because the position we are re-tracking
IS the broker's 228 and the overlay's anchor should equal what the broker paid.
(Decision point D4.)

**Mechanism.** The durable target is `runtime/trade_closer_state.json` (the FIFO
book), per the DRIFT-RECON doc: the guard is rebuilt from it every cycle. The
script:
1. Verifies broker truth UNH = 228 (reads `runtime/positions_snapshot.json` /
   position guard broker mirror; refuses if ≠ 228 or stale).
2. Verifies the gamma UNH FIFO is empty and no existing `gamma|UNH` lot (idempotency).
3. Writes ONE `gamma|UNH` lot: `qty=228, side=BUY, fill_price=424.97,
   ts_utc=<last re-buy ts>, meta={source: pff1_reattribution, basis: broker_truth_vwap,
   NOT pnl_untrusted}` into `trade_closer_state.json` (timestamped `.bak` first).
4. Writes `position_guard.json` in the exact rebuild shape for immediate
   consistency (next `_rebuild_guard_from_paper_ledger` reproduces it).

**OPEN RISK to resolve before building (study item):** `broker_sync|UNH` double-add.
`_rebuild_guard_from_broker` (condition 2) creates `broker_sync|<sym>` when "broker
holds a position the guard doesn't know about." I must confirm it dedups against a
*strategy* guard entry (gamma|UNH), not only against an existing `broker_sync|`
mirror — otherwise after re-attribution the guard could carry gamma|UNH=228 AND
broker_sync|UNH=228 (=451 vs broker 228, a NEW over-count). If it does not dedup
against strategy entries, the script must additionally clear any stale
`broker_sync|UNH`, or the plan escalates. **This is the top build-blocker for item 2.**

Same safety rails as item 1 (dry-run default, typed token `--confirm
REATTRIBUTE-UNH-PFF1`, `.bak`, idempotent NOOP, gates, no broker I/O). **Ordering:
run item 1 (scrub) and item 2 (re-attribution) only after PFF1-Q1 is live in the
trade_closer service** — else tomorrow's churn re-pollutes.

---

## ITEM 3 — CONFIG OFFICIALIZATION (repo-only, no runtime touch)

Three sub-tasks, each its own W2A commit, each green-vs-baseline:

**3a. Commit the CAD re-pricing.** `config/tiers.json` and
`config/withdrawal_policy.json` are tracked-but-modified in the LIVE tree
(uncommitted since ~June): `*_equity_usd`→`*_equity_cad` (×1.4160), and
`withdrawal_policy` v1→v2 (`seed_capital_cad=70800`, `max_monthly_salary_cad=2832`).
The live code already reads the CAD keys (`tier_manager._band_value` prefers
`_cad`, falls back to `_usd`; `withdrawal_manager` reads `total_equity_cad`), so
committing the current live bytes is **officialization only — zero content/runtime
change**. Procedure: copy the exact current bytes from
`/home/ubuntu/chad_finale/config/{tiers,withdrawal_policy}.json` into the worktree,
commit. (The live tree keeps showing them "modified" until this branch merges to
main and the live tree updates — expected, no runtime effect.)

**3b. Update the 3 tier tests to CAD.** `test_1_scale_tier_at_182k`,
`test_2_starter_caps_at_2600`, `test_7_promotion_immediate` assert the old USD
bands; with the CAD config, e.g. 182 000 now lands in PRO_GROWTH (SCALE
`min_equity_cad=226 560`). Re-price the test equities/asserted bands to CAD
(×1.4160), matching the already-present CAD tests (`test_main_cad_ok_*`).
**Wrinkle to fix or flag (D5):** the fixture reads an **absolute live-tree path**
(`REPO_ROOT = Path("/home/ubuntu/chad_finale")`), so it reads the LIVE config, not
the worktree's. Recommend switching the fixture to a tree-relative path
(`Path(__file__).resolve().parents[2]`) so the worktree suite is self-consistent;
otherwise the worktree test result silently depends on the live tree's config.

**3c. Remove the stale backup dir.** `config/_cii_backup_20260630T005102Z/`
(pre-CAD `tiers.json`+`withdrawal_policy.json` snapshots, 2026-06-30) is **untracked**
(not in git). `git rm` is a no-op; this is a working-tree `rm` — do it in the LIVE
tree as a Phase-3 housekeeping step under GO, NOT via the branch. Flag as D6 (it's
the one "removal" that touches the live tree, so it needs explicit GO and is NOT a
repo commit).

---

## ITEM 4 — SIM-MARK DIVERGENCE (report only; propose PA, no build)

**Observation:** the overlay's ACTIVE close of UNH printed the SIM fill at
`fill_price = 423.00` while the broker filled at **424.88** — a $1.88/sh ($513 on
273) divergence.

**Diagnosis (traced):** the paper executor prices the SIM fill from
`runtime/price_cache.json` via `paper_exec_evidence_writer.build_submit_quote_stamp`
(`source="price_cache_mid_or_last"`). The stamp on this fill:
`ref_price=423.0, quote_ts=13:50:00, quote_age_s=55.3, quote_ttl_s=300,
confidence="ref_only_no_nbbo"`. So the SIM mark is a **price-cache mid/last that was
55 s stale but still inside the 300 s TTL**, with no NBBO — on a name whose ATR is
~3.4%/day, 55 s is enough to diverge ~$1.9. The broker filled at the live tape; the
SIM booked the stale cache scalar. `_SUBMIT_QUOTE_TTL_S = 300` is the only staleness
bound and it is far too loose to be a *fill-price* gate.

**Proposed PA (no build in Wave-2A):** file
`ops/pending_actions/PA_SIM_MARK_freshness_2026-07-20.md` proposing one of:
- (a) a tight **fill-price freshness gate** (e.g. reject/repull if `quote_age_s`
  exceeds a small bound like 15–30 s for the *fill mark*, distinct from the 300 s
  evidence-TTL), falling back to `pnl_untrusted` when unmet; or
- (b) **reconcile the SIM fill price to broker truth** — when the harvester later
  reports the real fill, treat its price as the authoritative mark for the evidence
  record (forward-only; the harvester already carries `ibkr_exec_id`); or
- (c) require NBBO (`confidence != ref_only_no_nbbo`) for a *fill* mark.
Recommend (a)+(b). This is the same class of defect PFF1-Q4 flagged; it is
independent of the double-book.

---

## DECISION POINTS (please answer before Phase-2 build)

- **D1 — Scrub scope / Stage-2:** OK to (a) scrub via `quarantine_manifest` +
  (b) teach Stage-2 (`trade_log_adapter`) to honour the manifest? (The alternative,
  in-band rewriting of the hash-chained ledger, is rejected — it destroys the
  append-only chain.)
- **D2 — Deploy sequencing:** confirm PFF1-Q1 must be live in the trade_closer
  service (and Q2 reloaded in the SCR shadow server) BEFORE the scrub/re-attribution
  run, so the −375.60/effective-67 target holds and tomorrow does not re-pollute.
- **D3 — Re-attribution build vs reuse:** build the narrow *trusted* re-attribution
  script (recommended — gamma-attributed, real basis), or accept the existing
  `reconcile_ledger_to_broker.py` behaviour (untrusted, `epoch3_adopted`)?
- **D4 — UNH cost basis:** broker-truth VWAP **424.97** (recommended) vs executor
  SIM VWAP 425.50.
- **D5 — Tier fixture path:** fix the absolute-path fixture to tree-relative
  (recommended), or leave it and accept live/worktree coupling?
- **D6 — `_cii_backup` removal:** it is untracked, so removal is a live-tree `rm`
  (needs GO), not a branch commit — confirm you want it deleted (vs left as a dated
  local backup).
- **D7 — Item-2 build-blocker:** approve spending the first build step on confirming
  `_rebuild_guard_from_broker` dedups gamma|UNH vs re-creating broker_sync|UNH
  (the over-count risk) before writing the re-attribution script.
- **D8 — Green-vs-baseline methodology:** adopt the **failing-test-ID SET diff**
  against the recorded worktree-18 (require ⊆, item 3 removes the 3 tier IDs) —
  recommended, minimal — vs the larger detour of de-coupling the ~14 absolute-path
  artifact tests so the worktree suite is hermetic (a separate test-hygiene lane).
  Confirm which. (This is why the plan's original "known-5" baseline was wrong: the
  worktree suite fails 18, not 5, purely from environment coupling.)

## Phase-2 commit order (all `W2A`-prefixed, set-diff green-vs-baseline each)

1. `W2A: Stage-2 honours quarantine manifest (+ test)` [enables item 1]
2. `W2A: ghost-scrub one-shot script (dry-run default) + acceptance test`
3. `W2A: resolve item-2 guard-dedup risk; UNH re-attribution script + test`
4. `W2A: officialize CAD tiers+withdrawal config`
5. `W2A: retune 3 tier tests to CAD + tree-relative fixture`

Scripts (steps 2,3) and the `_cii_backup` rm execute against live `runtime/`/tree
**only on explicit Phase-3 operator GO**, dry-run first.

---

## PHASE 2 — CLOSURE RECORD (built 2026-07-20)

All eight decisions locked by the operator and built to the finish line. Worktree-only; live
tree untouched (the two runtime scripts are dry-run-by-default and were only dry-run against
live, read-only — verified: no output files written to `/home/ubuntu/chad_finale`).

| Decision | Resolution (locked) | Landed in |
|----------|---------------------|-----------|
| D1 | quarantine-manifest scrub + small Stage-2 change so both scorekeepers honour it; ledger never rewritten. Implemented isolation-safe (stdlib manifest reader, NOT a `chad.utils` import — the harness import-closure test forbids it). | W2A-1 |
| D2 | Phase-3 deploy MUST restart the SCR shadow server (:9618) so Q2 loads; −375.60 assumes it. Operator-gated. | W2A-7 runbook |
| D3 | build the narrow *trusted* re-attribution script (not the reconcile tool's fabricated untrusted basis). | W2A-4 |
| D4 | UNH cost basis = broker-truth VWAP **424.97** (corroborated live by broker avgCost ~424.98). | W2A-4 |
| D5 | fix the tier fixture to a tree-relative path; sweep/document the other hardcoded `/home/ubuntu/chad_finale` test paths. | W2A-6 + W2A-7 findings |
| D6 | delete `config/_cii_backup_20260630…` as a gated Phase-3 step (untracked → live-tree `rm`). | W2A-7 runbook |
| D7 | prove FIRST (test/replay) that re-attribution leaves NO `broker_sync|UNH` duplicate summing to 451/456; no script until proven. Proven — readers compare legs, never sum. | W2A-3 (then W2A-4) |
| D8 | failing-test-ID SET-diff gate (⊆ worktree-18; item 3 removes exactly the 3 tier IDs). Non-hermetic suite logged as a standing finding. | all commits; W2A-7 findings |

**Commits:** W2A-1 Stage-2 quarantine honouring · W2A-2 ghost-scrub script · W2A-3 D7 proof ·
W2A-4 UNH re-attribution script · W2A-5 CAD config officialization · W2A-6 tier tests CAD +
tree-relative fixture · W2A-7 docs (PA + findings + runbook) · W2A-8 this closure record.

**Gate result (definitive, `chad/tests/`):** 15 failed / 3867 passed / 5 skipped. The failing
set is **exactly** the worktree-18 baseline minus the 3 `test_tier_manager` IDs — zero new
failing IDs. `tests/validation/test_isolation.py` 11/11 green (D1 isolation preserved).
`full_cycle_preview` clean.

**Phase-2 status: COMPLETE.** Stopped at the push decision — branch `goal/wave2-books-cleanup`
is NOT pushed; the runtime scripts and `_cii_backup` rm await Phase-3 operator GO.

# W5B GO RECORD — R1 Portfolio Allocator (SHADOW)

Worktree `/home/ubuntu/chad_w5b`, branch `goal/wave5-allocator`.
Phase 1 (plan) `bca56e2`. Phase 2 built 2026-07-23 against `main@b5d0ee2`
(Lane A merged in at `7cfca0c`).

**STOPPED at the push/merge decision — same as W4A/W4B/W5A. Nothing pushed,
nothing merged, no runtime mutated, no service restarted.**

---

## 1. What shipped

| Commit | Contents |
|---|---|
| `76fd4f8` | W5B-0 — baseline capture + GO record + premise re-verification |
| `43294f4` | W5B-1 — exposure core (`ExposureVector`, `ProvisionalBook`) |
| `00223e1` | W5B-2 — `portfolio_limits.v1` + marginal evaluation engine |
| `52afefa` | W5B-3+4 — shadow observer at stage-3 + `allocator_state.v1` heartbeat |
| `15b3b9c` | W5B-5 — coach reject-streak NOTIFY + sentinel visibility |
| *(this)* | W5B-6 — closure record |

**117 new tests**, all passing. Set-diff green at every commit: **16 failed /
4498 passed / 5 skipped**, failing set **name-for-name identical** to
`audits/W5B_BASELINE.md`. `full_cycle_preview` exit 0 at both wiring points.
No allocator artifact was written anywhere in the live tree (the flag defaults
off; the observer in OFF mode reads neither the book nor the config).

New files: `chad/risk/portfolio_allocator.py`, `chad/risk/allocator_limits.py`,
`chad/risk/allocator_shadow_gate.py`, `config/portfolio_limits.json`, five test
modules. Modified: `chad/core/live_loop.py` (observer + two heartbeat call
sites), `chad/utils/coach_voice.py` (one template), `config/exterminator.json`
(one feeds row + one EXS7 pin), two test files.

## 2. Deviations from the plan — all four, stated plainly

1. **The correlation-diagnostic commit was CUT** (operator ruling). No rolling
   correlation is computed anywhere in W5B. §4 below records why the plan's own
   stated reason for deferring was wrong even though the ruling was right.
2. **W5B-3 and W5B-4 landed as one commit.** The heartbeat's entire payload is
   the observer's per-cycle state, so a standalone W5B-3 would have added
   counters with no reader. Both were verified separately before landing (30
   gate tests, then 46 gate+heartbeat tests). Honest cause: I wrote the
   heartbeat before committing the gate, which entangled two files past the
   point where a clean split was worth the churn.
3. **D4 flipped from report-only to binding-in-shadow** — a direct consequence
   of OPERATOR_VERIFY 2 supplying a derived per-sector number.
4. **One Lane-A test file was edited** (`test_w5a_embedded_schema_contract.py`).
   Test-only; no W5A production code and no frozen directory touched. §5.

## 3. The two derived thresholds — what they are and are not

Both carry `basis:"shadow_derivation_2026-07"`, `ratified:false`,
`enforce_era_requires_pa:true`, and their full derivation text inside the
config. A test pins those stamps so they cannot silently flip.

| | Value | Derivation | Effect on the live book |
|---|---|---|---|
| **gross** | $750,000 USD | 1.5 × `policy.py:665` $500k (the only USD firm-level exposure number in the repo; enforced but per-ORDER, and the book is already 1.34× it). Tier SCALE's `risk_profile` is all-null, so no competing ceiling exists. | Book $671,037 ⇒ **$78,963 headroom**. Not saturated: every row is a genuine marginal verdict. |
| **per-sector** | $375,000 USD | 0.5 × the gross ceiling ("no sector may exceed half the firm ceiling"). | Healthcare $316,936 ⇒ **84.5% of cap, $58,064 headroom**. The *next* healthcare ticket trips it. |

**They are not risk limits.** They exist so would-reject evidence is possible.
A ceiling *below* the live book would would-reject everything forever and the
corpus would carry no information; that is the failure these numbers avoid, and
it is the only property they were chosen for. The enforce-era numbers get their
own PA.

**Units matter here.** Both are USD notional and neither is a multiple of
equity. The book is USD-priced, the only `currency_ok` equity is CAD, and there
is no FX rate anywhere (`usd_ok=false`, `usdcad_rate_used=null`). The tempting
ratio `gross/equity = 0.676×` is mixed-unit and wrong by roughly the USDCAD
spread — nearer 0.93× at a plausible rate. `equity_basis.used_as_divisor` is
`false` and a test asserts it.

## 4. Correlation — the ruling was right, the plan's reasoning was not

Ruled: static sector buckets ratified, rolling correlation deferred to R2.
Effect: `allocator_correlation.py` was never written; no ρ is computed; a
structural test asserts the only numeric leaf in the config's correlation node
is a *citation*, not a measurement.

PLAN_W5B §7 justified deferral partly with "no correlation threshold is sourced
anywhere." **That is false.** `config/sizing_config.json::correlation_monitor.threshold
= 0.65` is sourced *and* wired into live sizing via
`chad/risk/correlation_monitor.py` at `execution_pipeline.py:264` — the R6
reducer — and the config even records the measured statistic ("a book averaging
0.654 pairwise correlation").

So the real argument for deferral is the *opposite* of the one given: a
correlation reducer already exists at per-order grain, and a second,
independently-thresholded correlation layer at portfolio grain would
**double-count the same effect through two unreconciled numbers**. That is a
stronger reason than "unsourced" ever was, and it hands R2 a concrete first
task: reconcile with `correlation_monitor`, do not duplicate it.

## 5. Findings

### Standing — these bound what the shadow evidence may claim

Carried in `standing_findings[]` on **every** `allocator_state.v1` heartbeat
(including the flag-off one), and pinned as a required key by the EXS7 contract
so a payload that lost them is a schema break.

- **W5B-SF1 — the stage-3 bypass makes the allocator partially blind.** The
  property that makes the observer safe (overlay, crypto-overlay, reconciler and
  flatten closes go `apply_close_intents → adapter` direct and never traverse
  stage-3) is the same property that hides those mutations from it. The
  provisional book is open-positions-at-cycle-start plus entries, so it can be
  one cycle stale. A **flip** is the sharp case: its closing leg bypasses while
  the executor and reconciler move the position. INC-0723 is the live precedent
  for how far a bypassing path can move real quantities.
  **Bound:** evidence supports *"this entry would have breached given the book
  as of cycle start"*; it does **not** support *"gross never exceeded X"*.
  **This heads the agenda of any future enforce-flip PA** — an enforcing
  allocator with a one-cycle-stale book could reject a legitimate entry or admit
  a breaching one, so enforce requires either a book re-read at evaluation time
  or explicit participation from the bypassing paths.
- **W5B-SF2 — the provisional book is an upper bound on submitted exposure.**
  Found while wiring, not anticipated by the plan. The observer sits at the
  stage-3 chokepoint, *upstream* of the cooldown gate, the ML veto and the
  dynamic risk gate, so an intent it counts can still be suppressed below. The
  book measures allocation **demand**, not predicted fills.

### New, filed as follow-ups

- **W5B-F1 — no execution join key on the IBKR lane.** `StrategyTradeIntent`
  (`ibkr_executor.py:26-80`) is frozen and defines neither `idempotency_key` nor
  `trace_id`, so PA-EP3's `execution_id=(getattr(...) or getattr(...) or "")` at
  `live_loop.py:3288-3292` always resolves to `""` for IBKR intents. PA-EP3's
  join spine is live on Kraken and **empty on IBKR**. Not W5B's to fix (an
  intent-schema change on a frozen dataclass in the execution hot path), but it
  bounds any future would-verdict → realized-cost join, and it is why allocator
  rows carry a soft correlation tuple rather than a fabricated hard key.
- **W5B-F2 — EXS7 evidence truncates breaks at 10 while `break_count` stays
  accurate** (`exterminator_sentinel.py:1128`, `breaks[:10]`). Under a tmp
  `repo_root` every enforced contract contributes a `missing` break, so adding
  the 10th contract pushed W5A's embedded break out of the evidence list and
  failed two Lane-A tests. The sentinel behaviour is defensible (the count is
  unbounded), but an operator reading `breaks` on a genuinely broken runtime can
  silently miss entries past the 10th. Fixed *in the test* (see below); the
  production truncation is left alone and filed here.
- **P7 caveat A (pre-existing, re-confirmed)** —
  `position_exit_overlay._asset_class` has no crypto/options branch and omits
  MYM from its futures set, so MYM misclassifies as equity. The allocator
  deliberately does not inherit that classifier.

### The Lane-A test edit

`test_w5a_embedded_schema_contract.py::_make` now writes a config copy with the
file-keyed `enforced` map emptied, so those tests isolate the embedded pins they
actually assert. The fragility was **latent** — any future enforced contract
would have tripped it; W5B was simply the first to add one. No W5A production
code and no frozen directory (`chad/validation/*`, the analytics modules, the
`to_payload` stamp, the overlay `_save_anchors` hook) was touched.

## 6. The premise corrections (detail in PLAN_W5B §12)

Five Phase-1 claims did not survive re-audit: the cited shadow-gate template
does not exist in any ref (substituted `fuse_gate.py`, which let the allocator
*import* the bypass predicate instead of copying it); the stage-3 line anchors
moved; the book gained IWM and is entirely long (gross ≡ net); the equity basis
was attributed to the wrong artifact; and a correlation threshold *is* sourced.

The sharpest live finding: **LLY ($215,488) and SPY ($182,637) already exceed
the ENFORCED $150k per-symbol cap**, because that cap is per-ORDER and a
position accumulated over several orders never meets it. The missing-aggregate-layer
thesis, demonstrated on today's book — and it means day-one would-rejects come
from a hard-sourced, already-ratified number rather than only from the derived
ones.

## 7. What W5B does NOT do

No enforce path (there is no `should_block`-True branch in the code at all, and
`CHAD_ALLOCATOR=enforce` is refused with a loud log). No beta weighting (ships
at `beta=1.0`/`default_1.0`; no beta source exists). No options greeks. No
full-size futures multipliers and no unification of the three point-value
tables. No MYM classifier fix. No USD equity basis. No rolling correlation. No
Kraken-lane observer — the D6 thin mirror was **not built**: the book has no
crypto today, the Kraken wallet cap is $184.58, and adding a second call site
for zero current exposure would have been wiring without evidence. Firm
gross/net already include both venues via the book snapshot. Filed as a
follow-up rather than silently dropped.

## 8. Activation

Everything is inert until two things happen, in this order:

1. **A gated live-loop restart** — the observer, both heartbeat call sites and
   the streak NOTIFY are new code in `live_loop.py`. Until then the sentinel
   WARNs on the missing `portfolio_allocator_state.json` feed, which is intended
   and documented in the feeds row (do not "fix" it by deleting the row).
2. **`CHAD_ALLOCATOR=shadow`** — a Pending Action, not applied here. Default off
   is byte-identical to today.

Recommended: restart first with the flag off (proves the heartbeat publishes and
the sentinel row goes green with zero behavioural change), then flip to shadow.

## 9. Open decision

**Push / merge to main is NOT done and is the operator's call.** Branch
`goal/wave5-allocator` is 5 commits ahead of `main@b5d0ee2` plus the merge
commit. If it merges after another lane that touches `config/exterminator.json`,
that lane rebases the file — the keys are disjoint.

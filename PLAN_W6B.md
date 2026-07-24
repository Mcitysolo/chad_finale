# PLAN_W6B — Wave 6 Lane B: Hardening Tail (EXS7 + EXS8 + P2 Remnants)

**Branch:** `goal/wave6-hardening` (worktree `/home/ubuntu/chad_w6b`, from `main@c76fbea`)
**Date:** 2026-07-23
**Phase:** 1 — PLAN ONLY. No code changed. This document is the sole commit.
**Commit prefix:** `W6B`

---

## 0. Scope and lane boundary

**In scope:** EXS7 schema-contract coverage, EXS8 ML veto baseline + manifest staleness,
the P2 remnant sweep (verify-then-fix), and the INCIDENT-0723 hold/in-flight-order PA.

**Lane A territory — NOT touched.** No changes to futures contract resolution
(`futures_contract_resolver.py`, `ibkr_historical_provider.py`), the nightly bars
refresh (`nightly_bars_refresh.py`), the futures roll publisher, or the futures expiry
gate. One deliberate cross-reference exists and is called out explicitly in §2: Lane A
identified `chad-xgb-train.service` as the blocker for its EXS5 goal and deferred it as
its decision point D4. **That unit is EXS8's own subject matter (it produces the ML
manifest), so Lane B claims it** — this resolves Lane A's D4 without either lane
touching the other's files.

**Standing constraints:** worktree only, no runtime mutation, no config mutation
(Pending Actions only), no systemd edits, no service restarts, set-diff methodology.

**Disclosure — one runtime write occurred during this audit.** I invoked
`python3 -m chad.ops.exterminator_sentinel` to read current check state; it resolved
`repo_root` to `/home/ubuntu/chad_finale` and appended its report to
`runtime/reports/EXTERMINATOR_SENTINEL_{LATEST.json,HISTORY.ndjson}`. That is the
sentinel's own output artifact and `chad-exterminator-sentinel.timer` writes it every
5 minutes regardless (last scheduled run 23:36:47, mine 23:38), so the effect is one
extra row identical in kind to a scheduled run — no state any other component consumes.
It still contradicts the "no runtime mutation" constraint, so it is recorded here rather
than glossed. For the remainder of the audit I read the timer's artifact instead of
invoking the scanner.

---

## 1. AUDIT SUMMARY — what is already done, stale, or misframed

Audit-first, as instructed. **Six of the eleven briefed items are already closed or
materially misframed.** Details in the sections that follow.

| Item | Briefed as | Audit verdict |
|---|---|---|
| (1) EXS7 schema pins | pin the stable ones; warn shrinks | **INVERTED** — the gap is enforcement *coverage*, not pinning. 74 files already carry a schema_version; only 10 are enforced |
| (2) EXS8 ML baseline | establish baseline from shadow history | **VIABLE, and the config's premise is wrong** — a baseline *is* computable; I computed it (§3) |
| P2-2 operator_intent | default posture in paper | **CLOSED-by-design** (June record) + superseded by the W4B-8 hold-preserve fix. Doc only |
| P2-3 winner/HALT_BOOST | reconcile clamp | **ALREADY IMPLEMENTED** — clamp + contradiction check both live. Residue is log noise |
| P2-5 EC2 SG 9618/9619/9620 | verify, Channel 1/3 if needed | **CLOSED — double-confirmed.** Ports are loopback-bound *and* absent from the SG. No change needed |
| P2-7 telegram urllib3 | fix | **BENIGN** — zero warnings in current journal; needs PTB 20.x, a real upgrade project |
| P2-8 APScheduler | assess risk, may defer | **DEAD DEPENDENCY — verified unimported.** Drop from requirements; near-zero risk |
| P2-9 options-chain silent failure + loud alert | fix | **CLOSED-healthy** + `OnFailure` already wired. One *new* finding: repo/live unit drift |
| P2-11 dual ledger | doc | **OPEN — genuinely undone.** Doc deliverable stands |
| (4) hold vs in-flight orders | design cancel step | **OPEN and real.** Both prior defects fixed; this is the surviving gap |
| EXS8 manifest staleness | refresh the check | **Check works** (74.9d > 30d). Root cause is upstream and now claimed by this lane |

`docs/DEFENSE_BOARD_RECONCILIATION_2026-06-17.md:27-37` already dispositioned the entire
P2 board on 2026-06-17. The brief's P2 list predates it. I re-verified each item
independently rather than trusting the record — results in §4.

---

## 2. ITEM 1 — EXS7 SCHEMA PINS (the brief's framing is inverted)

### Measured state

EXS7 currently returns **WARN** with `break_count: 0` — every enforced contract holds.
The warn comes solely from 5 entries in `unpinned_known`. But a full enumeration of
`runtime/*.json` tells a very different story:

```
files carrying a schema_version : 74
        ...of which ENFORCED    : 10
        ...of which unvalidated : 64   <-- pinned, but nothing checks them
files with NO schema_version    : 62
        ...documented in
           unpinned_known       :  5
        ...undocumented         : 57   <-- invisible to EXS7 entirely
```

**So the sentinel's WARN is scoped to a hand-curated list of 5 out of 57 genuinely
unpinned files, while 64 files carry a pinned schema that nothing validates.** The
briefed action — "pin the stable ones, warn shrinks to justified exceptions" — targets
the smaller and riskier half of the problem. The dominant, near-zero-risk win is
**adoption into `enforced`**, which is pure config and requires no publisher change.

Examples of already-pinned-but-unenforced contracts, several of them load-bearing:
`regime_state.v1`, `strategy_health.v1`, `winner_scaling.v1`, `tier_state.v3`,
`portfolio_state.v1`, `pnl_state.v1`, `positions_truth.v1`, `governor_state.v1`,
`lifecycle_replay_state.v3`, `options_chain_cache.v2`, `futures_roll_state.v1`.

### Why adoption is not simply "enforce all 64"

Adopting a file into `enforced` makes EXS7 **FAIL** (not warn) when it is missing or
malformed. Several pinned files are one-off or dated artifacts that must never gate a
sentinel — e.g. `quarantine_manifest_20260511.json`,
`quarantine_manifest_pff1_ghost_scrub.json`, `epoch_reset_state.json`. Enforcing those
manufactures false FAILs.

Adoption therefore needs a **liveness filter**: a contract is eligible only if it has a
live publisher, is rewritten on a known cadence, and is not a dated one-off. This is the
substance of W6B-1.

### Also found

- `runtime/__guard_probe_should_not_exist__.json` — a test probe artifact leaked into
  live `runtime/`. Flagged for disposition (W6B-3); not deleted by this lane.
- The 5 documented `unpinned_known` entries split cleanly by risk:
  - **Low risk to pin** (read-mostly publishers): `positions_snapshot.json`,
    `reconciliation_state.json`, `profit_lock_state.json`
  - **High risk to pin** (hot-path money-truth): `position_guard.json` (keys are
    `strategy|SYMBOL` pairs plus `_version`/`_written_by`), `trade_closer_state.json`
    (FIFO `queues` + `processed_fill_ids`)

For the high-risk pair, adding a `schema_version` key is *additive* and should be
invisible to readers that ignore unknown keys — but `position_guard.json` uses its key
space for position identity, so any reader iterating keys as positions would treat a new
top-level key as a position. That must be proven safe before pinning, not assumed.
Decision point D1.

### Deliverables

- **W6B-1** — liveness-filtered adoption of eligible already-pinned contracts into
  `schema_contracts.enforced` (config → **Pending Action**, not applied directly).
- **W6B-2** — pin the 3 low-risk unpinned publishers; document the 2 high-risk ones with
  a stated reason, so the residual warn is a justified exception set rather than an
  arbitrary list of 5.
- **W6B-3** — expand `unpinned_known` to cover all remaining genuinely-unpinned runtime
  files, so EXS7's warn describes the real gap instead of a curated sample.

---

## 3. ITEM 2 — EXS8 ML BASELINE (a baseline *is* computable; I computed it)

### The config's stated premise is wrong

`config/exterminator.json` asserts:

> "no live rate can be computed or compared" … "veto decisions leave only log lines
> (ML_SHADOW / ML_VETO), no durable artifact, no counter"

The second clause is true. **The first is not.** The `ML_SHADOW` lines are fully
structured and carry every field a baseline needs (`live_loop.py:2996-3009`):

```
ML_SHADOW symbol=GOOGL strategy=gamma intent_class=entry
  model_version=xgb_veto_20260510_020007 manifest_hash=sha256:74afb9e1c8f3e0fe4
  loss_prob=0.518 threshold=0.65 would_veto=False final_action=shadow_only
  reason=loss_probability_below_threshold
```

### Baseline computed from available shadow history

Parsing journald for the current boot (2026-07-23 17:00 → 23:45):

| Metric | Value |
|---|---|
| samples | **545** |
| `would_veto=True` | **6** |
| **production veto rate** | **0.0110 (1.10%)** |
| loss_prob mean | 0.275 |
| loss_prob p50 / p90 / p99 | 0.189 / 0.569 / 0.683 |
| model_version | `xgb_veto_20260510_020007` (single) |
| intent_class | `entry` (545/545) |
| strategy mix | gamma 510, omega_macro 31, beta 4 |

**Compare the manifest's training-time `val_veto_rate_at_0.65 = 0.7123 (71.2%)`.**

Production vetoes at **1.10%**; validation vetoed at **71.2%** — a **~65×
divergence**. The sentinel is right that comparing the two directly is a category
error, but the size of the gap is itself the headline finding: the model sees a
materially different distribution in production than in its training validation split.
This is exactly the "veto drift" the item wants detectable, and it is already enormous.

**Honest caveats, all of which shape the design:**
- 545 samples over ~6.75h of a **single boot**. journald here retains only the current
  boot (machine booted 17:21), so this window is volatile and cannot serve as a
  durable baseline on its own.
- 510/545 (94%) are `strategy=gamma` — this is close to "gamma's veto rate", not a
  portfolio-wide one. A baseline must be stratified or explicitly scoped.
- All samples are `intent_class=entry`; no protective-intent coverage.
- Single model version, so no cross-version comparison is possible yet.

### Manifest staleness — the check works; the root cause is upstream

EXS8 correctly reports `model_version_age_days: 74.9` against `manifest_stale_days: 30`
→ `manifest_stale: true`. The check needs no repair. **Why** it is stale does:

`chad-xgb-train.service` has failed since 2026-07-19. Reproduced faithfully under the
unit's own environment (`PYTHONPATH=/home/ubuntu/chad_finale`):

```
Training will exclude 5031 untrusted fill_ids and 322 quarantined trade hashes
Training set: 72 usable rows,
  exclusions={'quarantined': 6, 'untrusted': 2153, ...}
Training result: ok=False reason=Not enough samples for training (72 < 100)
```

**The unit is not crashing — it is refusing, correctly.** The Epoch-3 book yields only
**72 trusted closed trades** against a required 100, because 2153 rows are untrusted and
6 quarantined. Fail-closed behaviour working as designed.

This matters for how it is treated: **lowering the 100-sample threshold would be the
wrong fix** — it would train a live-money veto model on a book we have already declared
untrustworthy. The right dispositions are (a) let the unit report "declined" without
entering systemd `failed`, and/or (b) grow trusted data. Decision point D3.

*(A first pass of this reproduction, run from the worktree without the unit's
`PYTHONPATH`, showed 78 rows and warned "quarantine helper unavailable". That run was
not faithful to the unit environment; the correct figure is 72 with exclusions applied.
The conclusion holds either way, and more strongly at 72.)*

### Deliverables

- **W6B-4** — durable shadow-history collector: parse `ML_SHADOW` lines into an
  append-only `runtime/ml_veto_shadow.ndjson` (`ml_veto_shadow.v1`) so the baseline
  stops depending on journald retention. Read-only w.r.t. the veto path; the predictor
  is untouched and stays shadow-only.
- **W6B-5** — baseline publisher computing veto rate over a **declared window with a
  declared minimum sample count**, stratified by `strategy` and `intent_class` so a
  gamma-dominated window is never presented as portfolio-wide. Emits
  `ml_veto_baseline.v1` with the window, n, and stratification stamped in.
- **W6B-6** — wire `baseline_veto_rate` + `baseline_veto_rate_source` in
  `config/exterminator.json` as a **Pending Action**, with a drift band. Only after
  W6B-5 has produced a window meeting the minimum-n bar. Correct the `_doc` block's
  "no live rate can be computed" claim.
- **W6B-7** — xgb-train disposition per D3 (diagnosis is complete above; the fix is a
  decision, not a discovery).

---

## 4. ITEM 3 — P2 SWEEP (verify-then-fix)

Each item independently re-verified today; June record cited where it agrees.

### P2-2 — operator_intent default posture in paper → **CLOSED, doc only**

June: "by-design (allows paper entries; live blocked downstream)". Since then the
INCIDENT-0723 D4(a) defect — the 10-minute refresher unconditionally rewriting
`ALLOW_LIVE` and stomping an operator hold — has been **fixed**:
`operator_intent_refresher.py:347-359` now preserves an explicitly-set
`EXIT_ONLY`/`DENY_ALL` hold. D4(b) is also fixed: `live_gate.py:420-426` loads
operator intent and fails closed to `DENY_ALL`, and the live decision trace shows
`operator_mode=EXIT_ONLY` → *"Deny all lanes"*.

**Deliverable W6B-8:** doc-only. Update the P2-2 disposition to record that the posture
is by-design *and* that the hold is now durable and consumed. No code change.

### P2-3 — winner_multipliers vs HALT_BOOST clamp → **ALREADY IMPLEMENTED**

`dynamic_risk_allocator.py:388-406`: when a strategy is in `halted_lookup` and its
`winner_factor > 1.0`, the factor is clamped to `1.0`, `halt_clamp_applied` is recorded
in `applied_overlays`, and `HALT_BOOST_SUPPRESSED` is logged. A second, independent
`check_halt_boost_contradiction` exists at `exterminator.py:524-560`.

Nothing to reconcile — a halted strategy cannot be boosted. The only residue is the
June-noted log noise (WARNING per halted strategy per cycle).

**Deliverable W6B-9:** downgrade the repeated `HALT_BOOST_SUPPRESSED` WARNING to
INFO-with-dedupe (state already captured structurally in `applied_overlays`). Cosmetic,
optional. Decision point D4.

### P2-5 — EC2 security group for 9618/9619/9620 → **CLOSED, no change needed**

Report-only, as instructed. Two independent confirmations:

1. **All three listeners are loopback-bound** (`ss -tlnp`):
   `127.0.0.1:9618`, `127.0.0.1:9619`, `127.0.0.1:9620`. Not reachable off-host at any
   SG setting.
2. **The security group does not open them at all.** Instance `i-0c0d16702f524d9a6`,
   SG `CHAD-Prod-SG`; ingress is exactly: tcp/80 `0.0.0.0/0`, tcp/443 `0.0.0.0/0`,
   tcp/22 from three `/32` operator IPs. **No rule references 9618/9619/9620.**

**No Channel 1/3 action required.** Defense in depth is intact: even if the SG were
widened, loopback binding would still refuse. Recorded as a verified negative.

### P2-7 — telegram urllib3 → **BENIGN, defer**

`python-telegram-bot 13.15`, `urllib3 1.26.20` (both pinned in `requirements.txt`).
**Zero** urllib3/connectionpool warnings in the current journal window. The June
prescription (PTB 20.x) is a breaking major upgrade — PTB 20 is async-first and would
require rewriting `chad/utils/telegram_bot.py`'s handler surface (`ParseMode`,
`telegram.ext` imports at `:64-66`), which is the live alerting spine and the COACH-VOICE
L1/L2 delivery path.

**Recommendation: DEFER.** Not worth destabilising alerting for a warning that is not
currently firing. Decision point D5.

### P2-8 — APScheduler upgrade → **DEAD DEPENDENCY, drop**

`APScheduler==3.6.3` is pinned at `requirements.txt:7`. **Verified unimported** — zero
references across `chad/`, `ops/`, `scripts/`. The June record says the same.

**Deliverable W6B-10:** remove the pin from `requirements.txt`. No upgrade risk to
assess, because nothing imports it. This is the cheapest real win in the sweep.

### P2-9 — options-chain-refresh silent failure + loud alert → **CLOSED-healthy, one new finding**

Re-verified: unit exited `0/SUCCESS` at 12:30:06 today; `runtime/options_chains_cache.json`
is fresh (12:30:06Z), well-formed, and carries `schema_version: options_chain_cache.v2`.
The loud alert the brief asks for **already exists**: the live unit declares
`OnFailure=chad-service-alert@%N.service`.

Two residual findings, both new since the June record:

- **N1 — repo/live unit drift.** `/etc/systemd/system/chad-options-chain-refresh.service`
  exists and carries the `OnFailure` wiring, but **`deploy/chad-options-chain-refresh.service`
  does not exist in the repo.** The live unit is unversioned. This is a governance gap:
  a redeploy from repo would silently drop the alert wiring.
- **N2 — coverage is one symbol.** `chains` contains **SPY only**. That may be exactly
  the intent, but "exit 0 with 1 symbol" and "exit 0 with 0 useful symbols" are
  indistinguishable to the current check — which is the *actual* silent-failure surface
  the brief is reaching for. A coverage assertion (expected-symbol floor) would close it.

**Deliverable W6B-11:** capture the live unit into `deploy/` (N1) and add a coverage
floor to the freshness check (N2). Decision point D6 on whether SPY-only is correct.

### P2-11 — dual ledger authority declaration → **OPEN, doc deliverable stands**

The two ledgers:

| Ledger | Path | Consumers |
|---|---|---|
| Position-authority ledger | `runtime/ibkr_paper_ledger_state.json` (hash-keyed) | `chad/validators/position_authority.py:7,250` |
| Canonical closed-trade ledger | `data/trades/trade_history_YYYYMMDD.ndjson` | `tier_risk_enforcer`, `savage_allocator`, `edge_decay_monitor`, `per_strategy_loss_guard`, `dominance_allocator`, `symbol_performance_blocker`, `fuse_box`, `dashboard/api` (8+ modules) |

**Deliverable W6B-12:** `docs/LEDGER_AUTHORITY_DECLARATION.md` stating which ledger is
authoritative for which question (position identity vs realized P&L), that they are not
interchangeable, and which is canonical when they disagree. Doc only — no code.

---

## 5. ITEM 4 — INCIDENT-0723 FOLLOW-UP: holds do not stop in-flight orders

### The gap, precisely

Both INCIDENT-0723 D4 defects are now fixed (§4/P2-2): a hold **persists** and a hold
**is consulted**. What neither fix does is act on orders **already working at the
broker**. `live_gate`/`operator_intent` are admission control on *new* intents; an order
submitted seconds before the hold continues to live at IBKR and can still fill. The
incident record's own language — *"the hard brake remains operator-side:
`systemctl stop chad-live-loop`"* — is an admission that no in-band brake reaches
working orders.

### Existing mechanics to reuse (do not reinvent)

`chad/core/paper_position_closer.py:359-367` already implements broker order
cancellation: snapshot open orders, cancel each, settle. The design should **reuse this
proven mechanic**, but it **must not be reused as-is**, for two specific reasons:

1. **It cancels indiscriminately.** The loop cancels *every* open order. In a flatten
   that is intended; applied to a hold it would cancel **protective/exit orders**,
   leaving positions naked — strictly worse than the problem being solved. The hold
   variant must cancel **entry orders only**.
2. **It enumerates with a clientId-scoped call.** It uses `ib.openOrders()`. Per the
   standing IBKR probe methodology, clientId-scoped enumeration **silently returns
   nothing for other clients' orders** — it does not error. A hold-cancel built on it
   would report "cancelled 0" while orders placed under a different clientId keep
   working. The enumeration **must** use the all-clients call.

Trap (2) is the one most likely to produce a confident, wrong success message, so the
design treats it as the primary correctness requirement, not a footnote.

### Design — `W6B-13`, its own mini-PA, flag-gated, default OFF

**Shape:** a discrete, auditable step invoked *as part of applying a hold*, never on a
timer and never automatically.

- **Flag-gated, default OFF.** `CHAD_HOLD_CANCEL_ENTRIES` unset/false → the step does
  not run and hold application behaves exactly as today. Byte-identical default path.
- **Dry-run first.** Default output is a *plan*: which orders would be cancelled and
  why. `--execute` required to act, consistent with the drift-recon and flatten tools.
- **Entry-only, fail-closed classification.** An order is cancellable only if it is
  positively classified as an entry. Anything unclassifiable, protective, reduce-only,
  or a flip is **left alone**. Ambiguity must resolve to *do nothing* — reusing the
  existing exit/flip classification (`is_flip_signal`, `_is_exit_intent`,
  `intent_class`) rather than inventing a parallel notion.
- **All-clients enumeration**, per trap (2), with the order's `clientId` recorded per row.
- **Channel-1 posture.** The existing `chad-order-guard` hook already blocklists
  live-order/flatten entrypoints — it blocked two of my read-only greps during this
  audit, which is the guard working correctly. The new step belongs **behind that same
  guard**, operator-invoked in the terminal, never agent-invoked.
- **Evidence artifact.** `hold_cancel_report.v1` — orders seen, classification per order,
  action taken, per-order result, plus explicit `not_cancelled` reasons. A partial or
  zero-cancel outcome must be *loud*, never reported as clean success.
- **Paper-only fail-closed.** Refuses unless exec_mode ∈ {paper, dry_run}, matching
  `close_guard_entry.py`'s gate pattern.

**Mini-PA scope:** the PA covers only enabling the flag for a specific hold application.
Design, tests, and the dry-run path land dark under the default-off flag.

**Explicitly out of scope:** no change to hold semantics, `live_gate`, `operator_intent`,
or the exit/protective path. This step *adds* an action to hold application; it does not
alter what a hold means.

---

## 6. DECISION POINTS

**D1 — EXS7 high-risk pins (§2).** Pin `position_guard.json` and `trade_closer_state.json`?
 (a) document as justified exceptions, do not pin; (b) pin after proving no reader
 iterates top-level keys as positions; (c) pin now.
 *Recommendation: (b)* — with a named test proving key-iteration safety. `position_guard.json`
 uses its key space for position identity, so this is a real hazard, not a formality.
 Fall back to (a) if the proof does not come out clean.

**D2 — EXS7 adoption breadth (§2).** How aggressive should `enforced` adoption be?
 (a) liveness-filtered subset (live publisher + known cadence + not a dated one-off);
 (b) all 64 pinned files; (c) a hand-picked ~15 load-bearing contracts.
 *Recommendation: (a).* (b) manufactures false FAILs on dated artifacts; (c) recreates
 the arbitrary-curation problem that produced today's misleading warn.

**D3 — `chad-xgb-train.service` (§3).** The unit refuses correctly at 72 < 100 trusted rows.
 (a) let it report "declined" and exit 0, with the refusal surfaced as a data-quality
 signal (mirrors Lane A's W6A-6 exit-status pattern) — clears EXS5's last red;
 (b) leave failing as an honest signal that the book is too thin;
 (c) lower the 100-sample threshold.
 *Recommendation: (a). Explicitly NOT (c)* — training a live-money veto model on a book
 we have declared untrustworthy is the worst available outcome, and (b) leaves a
 permanently red sentinel that operators learn to ignore.

**D4 — P2-3 log noise (§4).** Downgrade `HALT_BOOST_SUPPRESSED` WARNING → INFO+dedupe?
 *Recommendation: yes, low priority.* Purely cosmetic; the structural record in
 `applied_overlays` is the real evidence.

**D5 — P2-7 PTB 20.x (§4).** *Recommendation: DEFER to its own lane.* An async-first
 rewrite of the live alerting spine, to silence a warning that is not currently firing,
 is a bad trade. Revisit if urllib3 churn actually resurfaces.

**D6 — P2-9 chain coverage (§4).** Is SPY-only the intended options-chain universe?
 If yes, the coverage floor is `>= 1` and this is a documentation fix. If no, the refresh
 is silently under-covering and the floor should be the real expected set.
 *Operator input needed — I cannot infer intent from code.*

**D7 — Item (4) hold-cancel scope (§5).** Entry-only cancellation as designed, or also
 cancel working *exit* orders on `DENY_ALL` (as opposed to `EXIT_ONLY`)?
 *Recommendation: entry-only for now.* Cancelling protective orders is how a hold turns
 into an incident; if `DENY_ALL` needs that later it should be its own explicit decision.

---

## 7. WORK BREAKDOWN & VERIFICATION

| ID | Item | Type | Risk |
|---|---|---|---|
| W6B-0 | This plan | doc | — |
| W6B-1 | Liveness-filtered `enforced` adoption | config PA | low |
| W6B-2 | Pin 3 low-risk publishers; document 2 high-risk | code+doc | low/med |
| W6B-3 | Expand `unpinned_known` to real coverage | config PA | none |
| W6B-4 | Durable ML shadow collector (`ml_veto_shadow.v1`) | code | low |
| W6B-5 | Stratified baseline publisher (`ml_veto_baseline.v1`) | code | low |
| W6B-6 | Wire `baseline_veto_rate` + correct `_doc` | config PA | low |
| W6B-7 | xgb-train disposition (per D3) | code | med |
| W6B-8 | P2-2 doc update | doc | none |
| W6B-9 | P2-3 log-noise downgrade | code | none |
| W6B-10 | Drop APScheduler pin | deps | none |
| W6B-11 | Capture options-chain unit to `deploy/`; coverage floor | code+ops | low |
| W6B-12 | `docs/LEDGER_AUTHORITY_DECLARATION.md` | doc | none |
| W6B-13 | Hold cancel-entry-orders step (flag-gated, default OFF) | code+PA | **high** |

**Ordering.** W6B-10, -8, -12, -3 first (zero-risk, immediate). Then EXS7 (-1, -2),
then EXS8 (-4 → -5 → -6, strictly sequential — the baseline cannot be wired before it
is measured). W6B-13 last and alone, given its risk class.

**Set-diff methodology.** Baseline captured on `goal/wave6-hardening` at `main@c76fbea`
before any change: **16 failed, 4498 passed, 5 skipped** — the standing baseline carried
since W5A, and byte-identical in membership to Lane A's independent capture (verified —
same 16 IDs). Verification compares the *set* of failing test IDs, never counts. Target
for this lane: **no additions, no removals** — W6B converts none of the known-fail rows
(that is Lane A's job); its obligation is to add nothing.

Recorded baseline fail-set (Appendix A):

```
test_backtest_unified_interface.py::test_backtest_unified_preserves_existing_pnl_run_completes
test_backtest_unified_interface.py::test_backtest_legacy_path_preserves_zero_slippage
test_futures_expiry_gate.py::test_bar_provider_skips_expired_in_polling_loop   (Lane A converts)
test_kraken_execution.py::test_intent_builder_btc_basic
test_phase_a_item5_liquidity.py::test_real_spy_bar_file_classified
test_pr03_ib_async_phase2_migration.py::test_live_posture_unchanged_paper_only
test_pr04_options_chain_refresh_remediation.py::test_live_posture_artifacts_unchanged_paper_only
test_repo_write_guard.py::test_guard_blocks_direct_write_under_data
test_repo_write_guard.py::test_guard_blocks_write_under_runtime
test_repo_write_guard.py::test_guard_blocks_mkdir_under_data
test_repo_write_guard.py::test_guard_reraises_even_when_caller_swallows
test_repo_write_guard.py::test_baseline_blocks_all_but_records_only_new_sinks
test_repo_write_guard.py::test_grandfathered_write_is_blocked_but_not_recorded
test_routing_gates.py::test_e4_kraken_passive_order_params
test_routing_gates.py::test_e4_kraken_aggressive_order_params
test_w4b8_flatten_bare_terminal.py::test_bare_terminal_drill_reaches_drill_complete
```

Note `test_pr04_options_chain_refresh_remediation.py` is in the standing baseline and
touches P2-9's subject matter — W6B-11 must not be assumed to convert it, and must not
disturb it.

**Per-change verification** (governance §4), in the worktree:
```
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q
CHAD_SKIP_IB_CONNECT=1 python3 -m chad.core.full_cycle_preview
```

Config changes ship as Pending Actions. No runtime mutation. No systemd edits. No restarts.

---

## 8. EVIDENCE INDEX

| Claim | Source |
|---|---|
| EXS7 warn, 0 breaks, 5 unpinned_known | `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` → `schema_breaks` |
| 74 pinned / 10 enforced / 62 unpinned / 5 documented | full enumeration of `runtime/*.json` vs `config/exterminator.json` |
| Production veto rate 1.10% (6/545) | journald `ML_SHADOW` parse, 2026-07-23 17:00–23:45 |
| Training val_veto_rate 71.2% | manifest `metrics.val_veto_rate_at_0.65` = 0.7123 |
| Manifest 74.9 days old vs 30-day threshold | EXS8 evidence block |
| xgb-train refuses at 72 < 100 | faithful re-run with unit `PYTHONPATH`; `train_xgb_model.py:433` |
| ML_SHADOW line schema | `chad/core/live_loop.py:2996-3009` |
| Hold now preserved | `chad/ops/operator_intent_refresher.py:347-359` |
| Hold now consumed | `chad/core/live_gate.py:420-426`; decision trace `operator_mode=EXIT_ONLY` → "Deny all lanes" |
| HALT_BOOST clamp implemented | `chad/risk/dynamic_risk_allocator.py:388-406`; `chad/ops/exterminator.py:524-560` |
| Ports loopback-bound | `ss -tlnp` — `127.0.0.1:{9618,9619,9620}` |
| SG opens only 22/80/443 | `aws ec2 describe-security-groups --group-names CHAD-Prod-SG` |
| APScheduler unimported | repo-wide search, zero hits; pinned `requirements.txt:7` |
| options-chain healthy + OnFailure | `systemctl status` `0/SUCCESS` 12:30:06; live unit `OnFailure=chad-service-alert@%N.service` |
| `deploy/` unit missing | `diff deploy/… /etc/systemd/system/…` → repo side empty |
| Chain covers SPY only | `runtime/options_chains_cache.json` → `chains` keys |
| Existing cancel mechanic | `chad/core/paper_position_closer.py:359-367` |
| clientId-scoped enumeration trap | `paper_position_closer.py:360` uses `ib.openOrders()`; standing IBKR probe methodology |
| June P2 dispositions | `docs/DEFENSE_BOARD_RECONCILIATION_2026-06-17.md:27-37` |
| Incident D4 / hard-brake admission | `audits/INCIDENT_20260723_DRILL_EXHAUST_FALSE_FLAT.md:154-193` |

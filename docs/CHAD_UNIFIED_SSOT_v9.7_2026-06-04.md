> # 🛑 DOCUMENT OF RECORD — SUPERSESSION NOTICE
> **This document is THE single comprehensive SSOT for CHAD. It supersedes and retires every prior scorecard, audit, checklist, register, and status document as a source of CHAD status.** That includes — each inventoried with path, item count, and disposition in §MS (MASTER SCORECARD) below — the Comprehensive Wiring Audit (2026-05-29, 82 findings), the SSOT-Calibrated Audit (2026-05-29, 132 claims), the Elite Completion Checklist (`docs/CHAD_GAPS_TO_CLOSE.md`, 73 items), the Paper Completion Register series (`reports/parity_audit/`, 4 generations), the L0–L8 red-team scorecard (`reports/redteam/scorecard/scorecard_20260604T124231Z.json`), the BOX-001→062 completion-matrix evidence set, and every SSOT cut v9.6 and earlier. Those artifacts remain FROZEN historical evidence and inputs to this document; **none of them may be cited as current status after this cut**. Status questions about CHAD are answered here and only here.
> *(Locked by operator GO 2026-06-04; the SSOT-of-record handoff from v9.6 is complete as of this commit + tag `SSOT_v9_7_2026-06-04`.)*

# CHAD Unified SSOT v9.7
# Post-Institutional-Closeout Cut: Bug B Futures Runaway, BOX-034 Currency Truth, Reliability Series

**Version:** 9.7 — **LOCKED 2026-06-04 (operator GO)**
**Date:** 2026-06-04 (UTC); drafted and locked same day
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v9.6_2026-05-27.md`. All prior cuts (v9.0 / v9.1 / v9.2 / v9.3 / v9.4-errata / v9.5-errata / v9.6) remain FROZEN historical artifacts.
**Predecessor reference commit:** `50d103a` (v9.6 lock commit, tag `SSOT_v9_6_2026-05-27`; merged to `main` at `d9b0c7f`).
**HEAD commit at draft:** `9385577` ("L1: Pending Action — root-fix Bug A event-loop/fd leak…")
**Commit span:** `git rev-list --count SSOT_v9_6_2026-05-27..HEAD` = **45 commits** (the merge `d9b0c7f` + 44 post-merge commits), every one accounted for in §6.
**Branch:** `main`
**Test baseline:** 2813 passing as of the last recorded full run (`3bfa924` execution log, 2026-06-04T00:21Z; +174 vs the 2639 collected at v9.6 lock).
**Lock type:** Full SSOT cut (the first full rewrite since v9.6). Tag: `SSOT_v9_7_2026-06-04`.
**Live order posture:** No new live execution authorization granted by this document. `ready_for_live` = false.

> Defense-board reconciliation (P0–P3) completed 2026-06-17 — see docs/DEFENSE_BOARD_RECONCILIATION_2026-06-17.md. All tiers reconciled; no active live exposure; ready_for_live remains FALSE.

> **LOCK STATUS:** Drafted 2026-06-04 for operator review; **LOCKED 2026-06-04 by explicit operator GO**
> (single-file commit + annotated tag `SSOT_v9_7_2026-06-04`). This file at
> `docs/CHAD_UNIFIED_SSOT_v9.7_2026-06-04.md` is now the SSOT of record, superseding
> `docs/CHAD_UNIFIED_SSOT_v9.6_2026-05-27.md` (which remains frozen and unmodified).

---

## CURRENT REALITY — The Three Questions (read this first)

### Question 1 — Does CHAD know its true financial state?

**Status: SUBSTANTIALLY IMPROVED — POSITION TRUTH GREEN; CURRENCY TRUTH TAGGED BUT NOT YET ENFORCED; EQUITY NUMBER REAL BUT PARTLY AN ARTIFACT.**

What is now known, with evidence:

- **Position truth is GREEN and broker-anchored.** At draft time (2026-06-04T21:14Z): `positions_truth.json` `broker_authority_status=GREEN`, `truth_ok=true`, ttl 90s; `reconciliation_state.json` `status=GREEN`, `drifts=[]`, counts 19/19/19, `broker_source=ibkr:clientId=83`. The recurring marginal-stale flicker (ttl == publisher cadence) was root-caused and fixed by raising the published ttl 60→90s (`3bfa924`), deploy live-verified the same day (`f4d8cea`).
- **Currency truth is now explicit instead of assumed.** The BOX-034/034A/034B series (full hash trail in §4 and §6) established CAD as canonical base, made the lifecycle publisher the single writer of `portfolio_snapshot.ibkr_equity` (`b807f1f`), made the collector fail closed on a NetLiquidation currency mismatch (`ec10789`), replaced the hardcoded 1.40 FX fallback with a validated live USDCAD rate that fails closed when unavailable (`dc5eb9b`), stopped the v2-collector tag-strip flicker (`c169c09`), and tagged `dynamic_caps.total_equity` and `pnl_state.account_equity` with `_currency`/`_ok` fields (`b6d333f`, `40a9c55`), live-verified (`fc51872`). At draft time `portfolio_snapshot.json` reads `ibkr_equity=289,106.09` with `ibkr_equity_currency="CAD"` — the currency is now a stated fact, not an inference.
- **However, currency assertions run in WARN-ONLY mode.** The five equity consumers carry `CURRENCY_WARN_*` assertions (`d0f7a78`, live-verified `6713101`) but the enforce-flip is still gated (the only remaining BOX-034A increment, per `33a92e9`). Until enforced, a wrong-currency value would warn, not block.
- **The equity number is real but its trajectory is contaminated.** Total equity at draft: **$289,330.97 CAD** (`dynamic_caps.json`, 2026-06-04T21:14Z). This includes mark-to-market on the Bug B runaway futures book (M6E +217 / M2K −25 / MCL +50; ~$3.99M USD gross notional, ~17.9× gross leverage; uPnL +$33.5K CAD at the 2026-06-04T01:21Z read and moving with prices — it has been as high as ~+$46.9K on the MCL leg alone and M6E has flipped negative). That book is an accident, not a thesis; equity and SCR statistics that include it overstate what the intended system produced. Disposition (flatten vs hold) is PENDING operator decision (`7744507`).
- **Residual gaps carried from v9.6:** `position_truth_v2` remains designed-and-stubbed, not wired (RISK-V9-6-02 carries forward); `tier_state.json` still names its field `current_equity_usd` while holding the CAD-valued figure (naming drift, value consistent with dynamic_caps); 78 runtime files still lack `schema_version` (SCHEMA-VERSION-1, deferred).

**Honest answer: CHAD's broker-position truth is verified GREEN and its currency provenance is now tagged end-to-end, but enforcement of currency correctness is pending, and the headline equity growth is partly an artifact of an unintended leveraged position that the operator has not yet disposed of.**

### Question 2 — Do CHAD's safety controls provably bind?

**Status: PARTIALLY PROVEN — ONE CONTROL OBSERVED BINDING IN PRODUCTION; THE PRIOR CONTROL ARCHITECTURE WAS PROVEN NOT TO BIND BY BUG B; REBUILD IN PROGRESS, CONTROL BY CONTROL.**

Evidence that controls bind:

- **The cumulative futures position cap (Bug B Fix A, `e4986ad`) is observed binding in production.** The chad-live-loop journal on 2026-06-04 shows `FUTURES_POSITION_CAP_BLOCK symbol=M6E strategy=omega_macro side=BUY qty=2.0 net=217.0 projected=219.0 cap=3 reason=cumulative_cap` firing on the ~10-minute cooldown cadence, on fresh GREEN truth (zero `CAP_UNVERIFIED` since the ttl fix). This is a guard demonstrably refusing real signals against real broker state — the strongest binding evidence in the system today.
- **The env gate is armed behind it (belt-and-suspenders).** `CHAD_DISABLE_FUTURES_EXECUTION` triple-flag stopgap (`82136d7`) live-verified via `FUTURES_EXECUTION_DISABLED_SKIP`; the cap fires first, the gate backstops.
- **The fill-feedback poisoning that defeated the same-side dedupe is fixed at source and consumer** (`b58bc9c`): harvester no longer emits phantom `closed_trade.v1` rounds; seeder skips `ibkr_harvest`/single-fill phantoms; 6 tests.
- **stop_bus latency trips now have symmetric hysteresis live** (`f3ab3d8`, activated `291bb84`) — single-cycle spikes no longer halt the book.
- **Fail-closed currency gates** on the equity collector (`ec10789`) and FX path (`dc5eb9b`) landed with tests.

Evidence that controls have NOT bound, or do not yet provably bind:

- **Bug B itself is proof of a multi-week silent control failure.** The harvester phantom-close poisoning (root-caused in `0d34315`) emptied the trade_closer FIFO, the guard auto-closed, same-side dedupe saw nothing open, and omega_macro re-entered every cycle: M6E reached 334 BUYs / 0 SELLs → +217 net before the 2026-05-30 stopgap. The per-order `max_contracts` "cap" was per-order only; no layer bound cumulatively. The controls in place at v9.6 lock did not bind.
- **The per-strategy loss-guard's enforcement bit is an env var alone.** Unset → silently report-only. It binds today only because the running service env sets it; the code-default-enforce hardening is PENDING GO (L3 PA, `d8441b3`).
- **No buying-power / margin pre-trade check exists at all** — confirmed absent during the L3 read-first (the 17.9× book formed with every existing layer in place). Split to its own future PA (`d8441b3` §10).
- **Bug A (event-loop/fd leak) is dormant, not fixed** (`9385577`): per-call thread+loop minting in `ibkr_adapter._call_with_timeout` caused the 2026-05-30 fd-exhaustion freeze; its only current mask is the Bug B env gate. Gate removal is now explicitly GATED on the Bug A root fix landing first.
- **The soak evaluator is still not built** (carried from v9.6): every `broker_authority_RED` window defaults to FAIL / NOT COUNTABLE, which is fail-closed but means no soak session can yet be mechanically graded EXPLAINED.

**Honest answer: today, at the futures chokepoint, a control provably binds — there is journal evidence of it refusing orders against live broker truth. But the system earned that proof by first demonstrating, via Bug B, that its previous control stack could silently fail open for weeks. Binding-ness is being rebuilt and proven one control at a time; it is not yet a system-wide property.**

### Question 3 — Is there cost-adjusted, out-of-sample edge?

**Status: UNPROVEN. No committed artifact demonstrates it.**

- A repository-wide search (docs/, reports/, ops/) for out-of-sample, walk-forward, or cost-adjusted validation artifacts found **no such artifact**. The only textual references are in historical SSOT cuts (v7.0, v8.3–v8.5), none of which contain or point to a completed validation.
- The paper statistics at draft (`scr_state.json`, 2026-06-04T21:14Z: effective_trades=205, win_rate=0.761, sharpe_like=3.530, total_pnl=+$27,299.82, max_drawdown=−$290, live_trades=0) are **in-sample, single-period, paper-fill** numbers. They embed paper execution assumptions, not measured live costs (slippage, real fill quality, financing).
- The statistics are additionally **contaminated by the Bug B runaway**: an accidental ~17.9×-gross-leverage futures book whose mark-to-market (+$33.5K CAD at last read, sign and size mobile) flows into equity, and whose mechanically-generated entries flowed through the same pipeline that produces the SCR inputs. Paper Epoch 3's clean-soak count remains effectively zero clean sessions of the intended system.
- The max_drawdown figure (−$290 against +$27.3K PnL) is not credible as a risk statistic for a book that carried multi-thousand-dollar open futures swings; it measures what the attribution layer recorded, not portfolio-level excursion.

**Plainly: there is no evidence of cost-adjusted, out-of-sample edge. Nothing in the repository proves the strategies make money outside the period and conditions they were observed in, net of realistic costs. Live promotion on current evidence would be unjustified, and this document does not move any live gate.**

---

## MS. MASTER SCORECARD — Reconciliation of All Prior Scorecards and Audits

This section is the single authoritative reconciliation of every prior scorecard, audit, and checklist. Each artifact is listed with **what it actually measured** and **its denominator**. All located in-repo on 2026-06-04 unless marked otherwise.

### MS.1 Inventory of prior status artifacts

| # | Artifact | Path | Items (counted, not quoted) | What it actually measured | Denominator |
|---|---|---|---|---|---|
| 1 | **Comprehensive Wiring Audit** (2026-05-29T01:37Z) | `reports/audits/CHAD_COMPREHENSIVE_WIRING_AUDIT_20260529T013753Z.md` | **82** tagged findings: 55 VERIFIED / 14 PARTIAL / 8 NOT_VERIFIED / 5 MISSING | "Is each component not just coded but **wired and observably running in production right now**" — per-component liveness across 12 layers, on that night | 82 wiring observations (55/82 ≈ **67%** VERIFIED) |
| 2 | **SSOT-Calibrated Comprehensive Audit** (2026-05-29T01:47Z) | `reports/audits/CHAD_SSOT_VERIFICATION_AUDIT_20260529T014743Z.md` | **132** tagged claims: 96 VERIFIED / 18 VERIFIED_FLAGGED / 11 DRIFT / 3 NOT_VERIFIED / 4 MISSING | "Do the **claims written in the SSOT documents** match repo + runtime reality" — documentation accuracy, not system completion | 132 SSOT claims ((96+18)/132 ≈ **86%**; the audit's own executive summary rounds to "~88%") |
| 3 | **Elite Completion Checklist** ("gaps_to_close", 2026-05-22) | `docs/CHAD_GAPS_TO_CLOSE.md` | **73** items, self-tallied **21 done / 52 remaining** | An **ambition checklist**: offense capabilities (A/B/C/D/ML), defense gaps (P0–P3), and a 16-step live-promotion path, mixed in one list | 73 heterogeneous items. **Frozen at 2026-05-22**: many "remaining" rows have since closed (e.g. P1-2 ib_async → PR-03 `d476e8c`; P2-9 options-chain → PR-04 `adacf6f`; P3-5 zero-fill classification → BOX-049; P3-1/2/3 hygiene → BOX-041/042/043). The 21-of-73 figure was true at its write date and is **stale now** |
| 4 | **Paper Completion Register** (latest generation 2026-05-27T00:13Z) | `reports/parity_audit/CHAD_FULL_PRODUCTION_PAPER_COMPLETION_REGISTER_20260527T001309Z.md` (supersedes 3 earlier same-name generations of 05-26/27) | **17 PR-side rows + 6 PO rows** (PR-01…PR-12, PR-M2K-MYM, PR-MYM-SIZING, PO-03, IBKR-LATENCY-RECOVERY; PO-01…PO-06); final tally 15 VERIFIED/LOCKED, 1 PARTIAL, 1 DEFERRED-by-design | "Is **paper mode** complete" — split into *build* vs *proof* vs *live-readiness* | Three separate denominators with three self-estimates: build **~97%**, proof **~70%**, live-readiness **~45%**. Pre-dates Bug B; its soak-clock and contamination picture is superseded by this document |
| 5 | **L0–L8 Red-Team Scorecard** (2026-06-04T12:42Z) | `reports/redteam/scorecard/scorecard_20260604T124231Z.json` — confirmed, single file in that directory | **9 layers** (L0–L8): 5 EVIDENCE-PRESENT / 3 PARTIAL (L1, L3, L4) / 1 UNSEEN (L6 edge) | Adversarial per-layer evidence sweep at HEAD `7744507` (dirty tree): "what evidence exists that each trust layer holds" | 9 layers; **publishes no percentage** (correctly) |
| 6 | **Pending Actions inventory** | `ops/pending_actions/` | **80 files** (79 tracked + 1 untracked: `PR-05_OFFICIAL_36_amendment_line84_patch_note_2026-05-26.md`) | The governance change-control queue — proposals, status logs, dispositions. **Not a completion measure at all** | n/a (a queue, not a scorecard) |
| 7 | **"75–80% overall complete"** | **NOT FOUND IN REPO** — figure cited from conversation only, unverified | — | No repository artifact computes or contains an overall completion percentage in this range | none — and this document deliberately does not create one |

Adjacent historical evidence sets (inputs, already absorbed into the artifacts above and into v9.5/v9.6): the forensic full-system audit (`reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`, 6 HIGH dispositions — §IB.3) and the Stage-3 completion matrix (`runtime/completion_matrix_evidence/`, 93 BOX evidence files, Boxes 001–062).

### MS.2 Why the percentages do not reconcile — and never will

The figures 67%, 86% (or "~88%"), 21-of-73, ~97/~70/~45%, and the conversation-only "75–80%" are **answers to different questions over different denominators at different dates**, and no arithmetic combines them:

- **67%** (wiring audit) answers *"of 82 things we looked at on the night of 05-29, how many were observably running in production?"* Its denominator includes deliberately-shadow-only layers (ML veto, drawdown guard) and install-pending Channel-1 items that are *governance-gated by design*, so it structurally cannot approach 100% without operator actions that were intentionally not taken.
- **86%** (SSOT-calibrated audit) answers *"of 132 sentences in the SSOT documents, how many are true?"* — a measure of **documentation honesty**, not of system completion. A system could be 10% built with a 100% score here.
- **21-of-73** (Elite checklist) answers *"of one author's combined offense+defense+live-path wishlist on 05-22, how many were done then?"* Its denominator mixes built features, open bugs, and a 16-item live-promotion sequence that is supposed to stay open until live GO — and it is frozen three weeks stale.
- **~97 / ~70 / ~45%** (Paper register) deliberately splits *build* from *proof* from *live-readiness* because the author recognized a single number would lie; it pre-dates Bug B, which retroactively invalidated its proof-side soak assumptions.
- **"75–80%"** corresponds to **no artifact in the repository**; it is an unverified conversational estimate and is hereby retired.

Any attempt to average or reconcile these into one number would be measuring the questions, not the system. **Accordingly, this SSOT publishes no overall completion percentage**, and retires all of the above as status sources (see Document-of-Record banner).

### MS.3 Bottom-line trust verdict (evidence-based, no percentage)

**What can be trusted today, on committed/live evidence:** broker-position truth (GREEN, 19/19/19, broker-anchored, fresh read 2026-06-04T22:32Z); currency provenance of the equity chain (tagged end-to-end, `b807f1f`…`40a9c55`, live-verified `fc51872`/`6713101`); and exactly **one risk control proven binding in production** — the cumulative futures cap (`e4986ad`, journal-evidenced `FUTURES_POSITION_CAP_BLOCK`). The documentation layer is now substantially accurate (86% of audited claims verified, drift items enumerated).

**What cannot be trusted:** any aggregate completion claim; the paper performance record (Bug-B-contaminated, in-sample, paper-fill-assumption-laden); system-wide safety (proven control-by-control only — Bug B demonstrated the previous stack failing silently open for weeks); and the equity trajectory (partly mark-to-market on an accidental 17.9× book). **Verdict: CHAD is a broadly built, honestly documented system whose trust is currently anchored to a small set of individually-verified controls — not yet to any system-wide property, and to no proven edge.**

---

## TM. L0–L8 TRUST-LAYER MATRIX

Layer names follow the red-team scorecard (`scorecard_20260604T124231Z.json`). For each layer: (a) what it requires, (b) what is DONE toward it (every item tied to a commit hash or a named live/runtime verification), (c) what is MISSING, (d) status from the controlled vocabulary {EXISTS / WIRED / SMOKE-TESTED / VERIFIED / LOCKED / DONE / PARTIAL / BLOCKED / DEFERRED / UNKNOWN-REQUIRES-AUDIT}. Statuses reflect **2026-06-04 reality**; stale prior statuses are explicitly superseded inline.

### TM.0 — L0: Money truth
**(a) Requires:** one currency-correct, broker-anchored, fail-closed source of truth for equity and positions, consumed by every sizing/risk decision.
**(b) DONE:**
- Broker-position truth GREEN and fresh: `positions_truth.json` `broker_authority_status=GREEN, truth_ok=true, ttl_seconds=90`; `reconciliation_state.json` GREEN, drifts=[], 19/19/19, `broker_source=ibkr:clientId=83` — **live runtime read 2026-06-04T22:32Z** (re-verified for this matrix).
- ttl marginal-stale flicker root-fixed `3bfa924`, deploy live-verified `f4d8cea`.
- Currency truth series landed end-to-end: single-writer CAD-canonical `ibkr_equity` `b807f1f`; fail-closed NetLiquidation currency gate `ec10789`; validated live USDCAD, 1.40 fallback removed `dc5eb9b`; tag-strip flicker fix `c169c09`; `_currency`/`_ok` tags on dynamic_caps `b6d333f` and pnl_state `40a9c55`, live-verified via gated restarts `fcc9986`/`fc51872`; warn-mode consumer assertions `d0f7a78`, live-verified `6713101`; hermetic reconciliation test `3e24bb7`.
- **SUPERSEDED stale status:** the earlier red-team claim "**L0 BROKEN — ~$84K equity gap**" is **retired**. The 2026-06-04 scorecard itself measured `dynamic_caps.total_equity` vs `portfolio_snapshot` sum at **abs_delta = 2.9e-09** and identified the apparent ~$86K gap as a **USD-display-vs-CAD-canonical comparison artifact** (`ibkr_equity_usd_display` vs `ibkr_equity`), not a real divergence. The gap claim is RESOLVED; record only the resolved state.
**(c) MISSING:** currency assertions are **warn-only** (`CURRENCY_WARN_*`; enforce-flip gated, last status `33a92e9`); `position_truth_v2` designed+stubbed, not wired (`1626f20`; RISK-V9-6-02); `tier_state.current_equity_usd` field-name drift (CAD value); 78 runtime files lack `schema_version` (SCHEMA-VERSION-1, DEFERRED); BOX-034B §3a de-minimis gaps (BTC omission, stale-FX round-trip, `6caecdf`).
**(d) Status: PARTIAL** — position truth VERIFIED, currency provenance VERIFIED, but enforcement not yet flipped, so the layer as a whole does not yet *bind* on currency error.

### TM.1 — L1: Stability
**(a) Requires:** services run for weeks without leak, freeze, or silent wedge; failures alert loudly.
**(b) DONE:** stop_bus symmetric latency hysteresis landed `f3ab3d8`, activated live `291bb84` (live runtime verification: no single-cycle-spike halts since); health-monitor alert pipeline verified operational `bfdcb7e`; Gateway nightly-restart timer + verifier built `70e150b`; Gateway version audit + R20 rule `3839bfb`; systemd patches regenerated against production reality `0799854`.
**(c) MISSING:** **Bug A fd/event-loop leak is dormant, not fixed** — root-fix PA PENDING GO (`9385577`); its only mask is the Bug B env gate. No committed artifact records a completed multi-week clean soak (scorecard L1 note: BOX-059 *defines* the criterion, does not prove it met). Channel 1 installs (nightly-restart, OnFailure directives, alert template) pending operator GO. BOX-034C oneshot watchdog PENDING (`7fa4a26` — the 13h publisher hang exposure).
**(d) Status: PARTIAL** — most important missing item: the Bug A root fix.

### TM.2 — L2: Execution authority
**(a) Requires:** a single authoritative execution mode; fail-closed gating of anything not explicitly authorized.
**(b) DONE:** runtime authority consistent and live-read: `CHAD_EXECUTION_MODE=paper`, `live_enabled=false`, `allow_ibkr_live=false` (live_readiness read 2026-06-04T22:32Z); futures-entry env gate landed and live-verified via `FUTURES_EXECUTION_DISABLED_SKIP` (`82136d7` + systemd drop-in); cumulative cap at the same chokepoint (`e4986ad`); cap-completeness vs the gate confirmed byte-identical-predicate/no-hole (disposition PA, `7744507`).
**(c) MISSING:** mode is declared in **479 scattered locations** across `chad/` (scorecard L2 — a diffuse authority surface, not a single source); the broad fail-closed submission gate for all non-`ibkr_live` lanes was landed `252bea9` and **same-day reverted** `f7ba4f7` — replaced only by the narrower futures stopgap; RISK-V9-7-07 carries.
**(d) Status: PARTIAL** — current posture VERIFIED, architecture (single authority point) not built.

### TM.3 — L3: Risk binding
**(a) Requires:** every risk limit provably *blocks* (not merely reports) when breached.
**(b) DONE:** the cumulative futures position cap is **observed binding in production** — journal evidence `FUTURES_POSITION_CAP_BLOCK … net=217.0 projected=219.0 cap=3` on fresh GREEN truth, 2026-06-04 (live verification; the strongest binding evidence in the system); per-strategy loss-guard runs in ENFORCE via `CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1` on the live service env (scorecard L3 service-env read, 2026-06-04 — this also **supersedes** the 05-29 wiring-audit's "unverifiable enforce-vs-report-only conflict": the authoritative later read found the flag set); fill-feedback poisoning fixed at source+consumer `b58bc9c`.
**(c) MISSING:** loss-guard enforcement is **env-var-fragile** — code default is report_only; enforce-by-default PA PENDING GO (`d8441b3`); **no buying-power/margin pre-trade check exists anywhere in `chad/`** (grep-confirmed absent in both the scorecard and the L3 read-first; the 17.9× book formed with every then-existing layer in place); drawdown guard and ML veto remain observe/shadow-only (wiring audit Layer 6); system-wide binding proof absent — see §RT.
**(d) Status: PARTIAL** — one control VERIFIED-binding; the layer property is not yet system-wide.

### TM.4 — L4: Paper record
**(a) Requires:** a fills/PnL ledger trustworthy enough to ground SCR, soak grading, and any edge claim.
**(b) DONE:** placeholder defense-in-depth catches 100% of $100 placeholders as `status=rejected, pnl_untrusted=true, broker_rejected` (trade_closer `cbded80`; strict predicate + read-only quarantine tool `29e7197`; PO-03 zero-public-fingerprint criterion committed `15b963e`); harvester phantom-close emission removed + seeder defense `b58bc9c`; 17/17 strategies honestly classified, 0 UNKNOWN (BOX-049 + Paper Completion Register 2026-05-27).
**(c) MISSING:** ~24.4% of all ledger rows are untrusted-tagged (scorecard L4: 4,483 of 18,336); the delta-strategy upstream placeholder emission is **still active** as of the latest fills (P0-1 remains PARTIAL); the Bug B runaway's mechanically-generated entries flowed through the same pipeline that feeds SCR — the record is **contaminated** for statistical purposes (RISK-V9-7-06); historical-quarantine `--apply` undecided.
**(d) Status: PARTIAL** — integrity *defenses* VERIFIED; the *record itself* is not clean.

### TM.5 — L5: Observability
**(a) Requires:** health endpoints, metrics, alert routing, and per-decision journal markers that make silent failure impossible.
**(b) DONE:** live endpoints respond (scorecard probe 2026-06-04: `:9618/shadow` 200, `:9620/metrics` 200); R01–R20 health-rule registry wired and firing (SSOT-calibrated audit §; R20 Gateway rule `3839bfb`); alert pipeline e2e-verified `bfdcb7e`; loud failure artifacts for options-chain (`b759cb6`, recovery verified in Paper Register 05-27 with `adacf6f` retry/backoff, discovery rewrite `4ba329e`); decision markers throughout (CAP_BLOCK, CURRENCY_WARN_*, EDGE_DECAY_FILTERED, etc. — journal-verified).
**(c) MISSING:** endpoint responses prove liveness, not correctness (no schema validation of metric bodies — scorecard L5 note); Channel 1 OnFailure directives + alert-template install pending; port-9618 0.0.0.0 binding closeout pending (PORT-BINDING-1 residual — mitigated only by the AWS Security Group, per SSOT-calibrated audit top-risk).
**(d) Status: PARTIAL** — wide and live, but with known blind spots that are install-gated.

### TM.6 — L6: Edge
**(a) Requires:** a committed, out-of-sample, cost-adjusted validation artifact demonstrating the strategy fleet makes money outside its observation period, net of realistic costs.
**(b) DONE:** **nothing qualifying.** Repo-wide search (docs/, reports/, ops/) found no such artifact. The only backtest artifact is `reports/backtest/gamma_reversion_backtest.json` — 1 strategy, 29 trades, no out-of-sample split, no cost fields (scorecard L6). Paper stats (scr_state 2026-06-04: effective_trades=205, win_rate=0.761, sharpe_like=3.530) are in-sample, paper-fill, Bug-B-contaminated.
**(c) MISSING:** the validation harness itself; a clean (uncontaminated) observation period; live-vs-paper cost calibration.
**(d) Status: UNPROVEN** (recorded plainly per Q3; nearest controlled-vocabulary mapping **UNKNOWN-REQUIRES-AUDIT**, where the required "audit" is the out-of-sample, cost-adjusted validation harness — roadmap §9 item 12). No committed artifact = no edge claim. Full stop.

### TM.7 — L7: Recovery
**(a) Requires:** the system detects, halts on, and recovers from broker/feed/process failure without operator heroics — and that this is periodically re-proven.
**(b) DONE:** real production auto-recovery on record — stop_bus trip cleared by `auto_recovery:broker_latency_clean_streak=5` 2026-05-28T18:38Z (runtime evidence in `stop_bus.json` history; scorecard L7); symmetric hysteresis now prevents spurious trips (`f3ab3d8`/`291bb84`); chaos-drill proof `reports/ratification/PROOF_CHAOS_DRILLS_20260403.json` all_passed=true (feed_stall, broker_disconnect).
**(c) MISSING:** the chaos-drill proof is dated **2026-04-03 — two months, one overhaul (04-19/21), and one Bug B ago**; no post-overhaul re-drill exists (scorecard L7 note). The soak evaluator is unbuilt (every RED window defaults FAIL/NOT-COUNTABLE — fail-closed but ungradeable). IBKR auto-recovery design PA still open. Bug A freeze-class recovery depends on the env-gate mask.
**(d) Status: PARTIAL** — recovery machinery live-evidenced once; the formal proof is SMOKE-TESTED and stale.

### TM.8 — L8: Live readiness
**(a) Requires:** a fail-closed readiness publisher that flips only when everything upstream is proven — and stays false until then.
**(b) DONE:** publisher functioning and fail-closed: `ready_for_live=false` (live read 2026-06-04T22:32Z); latest readiness report 9/10 checks ok with the sole failing check being paper mode itself — i.e. the gate is doing exactly its job (scorecard L8); LiveGate semantics locked (BOX-037).
**(c) MISSING:** everything upstream: clean-soak 0/5 (evaluator unbuilt, record contaminated); edge UNPROVEN (TM.6); pre-live operator tasks (CLAUDE.md) open; Bug B disposition + Bug A fix + enforce-flip all pending; the `operator_mode=ALLOW_LIVE`-while-paper intent tension (scorecard L8; GAP-016/BOX-037 context) flagged below as a new operator-confirmation item.
**(d) Status:** the gate itself **VERIFIED**; live promotion **BLOCKED** (correctly, by design, and by this document).

---

## RT. RED-TEAM FROM FIRST PRINCIPLES — The Strongest Current Case That CHAD Is NOT Done

Constructed from first principles against the 2026-06-04 evidence base, not carried from any prior red-team note. Every claim below is current; stale prior red-team statements are flagged at RT.9.

**RT.1 — The edge is unproven, and everything else is scaffolding around that absence.** A trading system's only irreducible justification is out-of-sample, cost-adjusted profitability. No committed artifact demonstrates it (TM.6). Every verified control, every GREEN reconciliation, every honest document is infrastructure in service of a hypothesis that has never been tested under the only conditions that matter. The +$27.3K paper PnL is in-sample, embeds paper-fill assumptions, and is contaminated (RT.3). If the edge is zero, CHAD is a perfectly-audited machine for slowly losing money with excellent logging.

**RT.2 — Safety is proven control-by-control, not as a system property — and the system has already demonstrated the difference.** Exactly one control has production-journal binding evidence (the futures cap, `e4986ad`). Bug B is the counterexample that disqualifies inference from "each control looks right" to "the system is safe": three individually-reasonable components (harvester, trade_closer seeder, same-side dedupe) composed into a multi-week silent fail-open that produced a 334-BUY runaway. Nothing proves the *current* stack lacks an equivalent composition defect; no end-to-end adversarial test of the whole control stack exists. The loss-guard binds only via a service env var (code default: report_only — `d8441b3` PENDING); the broad fail-closed submission gate was reverted (`f7ba4f7`); there is **no margin/buying-power check anywhere** (grep-confirmed) — the control class whose absence let the runaway reach ~$3.99M notional.

**RT.3 — The paper record — the entire evidentiary basis for confidence — is contaminated.** ~1 in 4 ledger rows is untrusted-tagged; the delta placeholder emitter is still firing; the Bug B mechanical entries flowed into the same SCR inputs that read CONFIDENT/1.0 today; max_drawdown=−$290 against a book that carried multi-thousand-dollar open futures swings is not a credible risk statistic; headline equity includes mark-to-market on an accident. SCR=CONFIDENT is therefore partly a statement about contamination, not skill. Clean-soak count is 0/5 — and the soak evaluator that would count it doesn't exist.

**RT.4 — A known freeze-class defect is dormant in the hot path, masked by the very gate the operator is being asked to remove.** Bug A (per-call thread+loop minting in `ibkr_adapter._call_with_timeout`) caused the 2026-05-30 fd-exhaustion freeze. It is unfixed (`9385577` PENDING GO). Its only mask is `CHAD_DISABLE_FUTURES_EXECUTION` — the same env gate queued for removal in the Bug B disposition. The system's stability is currently one env-var deletion away from re-arming a proven freeze mode.

**RT.5 — Currency correctness warns instead of blocking.** The BOX-034 series made currency truth *visible* (tags, `_ok` fields, fail-closed collector/FX) but the five equity consumers run `CURRENCY_WARN_*` assertions in warn-only mode (`d0f7a78`; enforce-flip gated, `33a92e9`). A wrong-currency equity value reaching sizing today would emit a log line and proceed. The 29% CAD/USD gap is larger than most daily risk budgets.

**RT.6 — A live 17.9×-gross-leverage accident sits open on the book, every hour, by decision-pending default.** M6E +217 / M2K −25 / MCL +50 is real market risk (uPnL has swung tens of thousands and changed sign), pinned only by the cap. Disposition is PENDING (`7744507`). Until flattened, CHAD's true risk posture is dominated not by its intended strategies but by its largest historical malfunction.

**RT.7 — Recovery is proven against last quarter's system.** The chaos-drill proof predates the 04-19/21 overhaul and Bug B. The soak-grading machinery (Decision 2 rule) is locked but mechanically inoperable without the unbuilt evaluator. CHAD can demonstrably auto-recover from a latency trip; nothing current demonstrates recovery from feed stall or broker disconnect *as the system exists today*.

**RT.8 — Bottom line.** CHAD today is: one verified truth layer (positions), one partially-enforced truth layer (currency), one binding control (futures cap), an honest but contaminated ledger, a stale recovery proof, a dormant freeze bug, an undisposed leveraged accident, and **zero evidence of edge**. The honest adversarial summary: *the system is excellent at knowing and documenting its own state, and has not yet proven it can either stay safe as a whole or make money at all.* Both proofs are future work (roadmap §9 items 12–14), and neither is shortened by any completion percentage.

**RT.9 — Prior red-team statements now STALE (superseded by this section):**
- "**L0 BROKEN / ~$84K equity gap**" — RESOLVED/SUPERSEDED: USD-display-vs-CAD artifact; measured abs_delta 2.9e-09 (scorecard 2026-06-04, TM.0).
- "**Order 5137 may collide with a flatten**" — RESOLVED: zero open orders account-wide via read-only `reqAllOpenOrdersAsync` (`d621aa8`); re-probe still required immediately before any actual flatten.
- "**CAP_UNVERIFIED flicker undermines the cap**" — RESOLVED: ttl 60→90 root fix (`3bfa924`), live-verified (`f4d8cea`); zero UNVERIFIED since (the formal §8(b) multi-hour observation remains open — see §8.1).
- "**Loss-guard enforce status unverifiable**" (wiring audit 05-29 conflict) — SUPERSEDED: the 06-04 scorecard read the live service env and found `CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1`; the residual risk is now precisely the env-var fragility (RT.2 / `d8441b3`), not uncertainty about today's state.
- "**Options-chain refresh broken / Greeks empty**" (wiring audit 05-29) — RESOLVED: full recovery verified in the Paper Completion Register (2026-05-27, status=ok), hardened by `adacf6f` and rewritten discovery `4ba329e`.

---

## 0. Preamble and Version Delta

### 0.1 Verified lineage (corrected and pinned)

| Cut | File | Nature | HEAD at cut |
|---|---|---|---|
| v9.0 (2026-05-04) | `docs/CHAD_UNIFIED_SSOT_v9.0_2026-05-04.md` | Full cut | tag `SSOT_V9_0_PAPER_EPOCH2_LOCK_20260504` |
| v9.1 (2026-05-13) | `docs/CHAD_UNIFIED_SSOT_v9.1_2026-05-13.md` | Full cut | — |
| v9.2 (2026-05-15) | `docs/CHAD_UNIFIED_SSOT_v9.2_2026-05-15.md` | Full cut | — |
| v9.3 (2026-05-17) | `docs/CHAD_UNIFIED_SSOT_v9_3_2026-05-17.md` | Full cut (last full rewrite before v9.6) | `72f361f` |
| v9.4 (2026-05-18) | `docs/CHAD_UNIFIED_SSOT_v9_4_2026-05-18.md` | **Forward-errata** (self-declared: "Lock type: Forward-errata SSOT") | `a5311d27` |
| v9.5 (2026-05-20) | `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md` | **Forward-errata** (Stage-3 box index) | `bbe7525` |
| v9.6 (2026-05-27, locked 05-28) | `docs/CHAD_UNIFIED_SSOT_v9.6_2026-05-27.md` | Full cut — **current SSOT of record** | `50d103a`, tag `SSOT_v9_6_2026-05-27`, merged to main at `d9b0c7f` |

Lineage findings (flagged for the operator):
1. **v9.4 and v9.5 both exist as real files but both are forward-errata, not full rewrites.** v9.6's own header is internally inconsistent: its lineage note calls v9.4 "the previous full SSOT" while its §0.1 (and v9.4's own header) correctly describe v9.4 as forward-errata. The last full rewrite before v9.6 was **v9.3**. This draft pins the corrected reading.
2. **v9.5's filename does not match the family glob.** `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md` is invisible to `docs/CHAD_UNIFIED_SSOT_v9*.md`; any tooling using that glob will silently skip v9.5. Naming drift also exists within the family (dots vs underscores: `v9.0/v9.1/v9.2/v9.6` vs `v9_3/v9_4`).
3. **No version gap.** The chain 9.0 → 9.1 → 9.2 → 9.3 → 9.4(errata) → 9.5(errata) → 9.6 is complete; v9.7 (this draft) is the correct next number.
4. v9.6 sha256 at draft time: `c9888b16c15b25033cf855d0e1423934eeba9c7f5b1c100b6b4bed57fb3ceecd` — unmodified by this draft.

### 0.2 What v9.7 captures
All 45 commits `SSOT_v9_6_2026-05-27..HEAD` (2026-05-28 → 2026-06-04), in five workstreams:

1. **Post-closeout reliability series (2026-05-28/29):** stop_bus latency hysteresis (Fixes A-series), Gateway nightly-restart timer, Gateway version audit, health-monitor alert pipeline verification, systemd patch regeneration, PR-05 options-chain discovery rewrite.
2. **The futures-submission P0 sequence (2026-05-31 → 06-01):** a fail-closed live-loop submission gate that was landed and same-day reverted, replaced by the `CHAD_DISABLE_FUTURES_EXECUTION` stopgap gate.
3. **BOX-034/034A/034B/034C — currency/equity truth (2026-06-01 → 06-03):** CAD-canonical single-writer equity, fail-closed currency gates, live FX, consumer currency tagging + warn-mode assertions, hermetic reconciliation test.
4. **Bug B — the futures runaway (2026-06-03 → 06-04):** root cause (harvester phantom-close poisoning + position-unaware entry), Fix B (source+consumer), Fix A (cumulative broker-truth cap, live-verified binding), positions_truth ttl fix, the order-5137 resolution, and the still-pending disposition decision.
5. **New pending hardening (2026-06-04):** L1 Bug A fd-leak root fix (PENDING GO), L3 loss-guard enforce-by-default (PENDING GO).
6. **Status-document consolidation (this expansion, 2026-06-04):** v9.7 absorbs and retires every prior scorecard/audit/checklist as a status source (Document-of-Record banner; full reconciliation in §MS; trust-layer matrix §TM; current red-team §RT).

### 0.3 What v9.7 is NOT
- NOT a live-trading authorization. `ready_for_live` remains false; no gate moves.
- NOT a disposition of the runaway futures book — that decision (flatten and/or gate removal) remains PENDING operator decision (`7744507`), with gate removal now additionally gated on the Bug A fix (`9385577`).
- NOT a closure of the BOX-034A enforce-flip, position_truth_v2 wiring, soak evaluator, Channel 1 install queue, or SCHEMA-VERSION-1 backfill — all carry forward.
- NOT a disposition of the four open operator items in §8.1 — locking this document does not close them.

### 0.4 Governance rules
Unchanged from CLAUDE.md / v9.6 §0.5 (ten rules; one-change-at-a-time, Pending Actions only, no live restarts without GO, etc.). Note `d80f44b` corrected the stale `full_cycle_preview` verification command in CLAUDE.md (module form, no `--dry-run`).

---

## 1. Mission and Architecture

### 1.1 Mission (unchanged)
Carried verbatim from v9.6 §1.1: CHAD is a multi-strategy, multi-broker algorithmic trading desk operating in **PAPER** posture, compounding a paper book through a governed, fail-closed execution stack until a clean multi-session soak and explicit operator GO authorize a live transition. Safety and auditability over throughput at every layer.

### 1.2 Architecture stack (v9.7 additions over v9.6)
The 8-layer hot-path stack plus the v9.6 horizontal layers (Position Truth v2 stub, Service Reliability, Soak Governance) carry forward unchanged. New in v9.7:

```
  FUTURES ENTRY GUARD STACK (NEW v9.7 — at the live_loop FUT-open chokepoint)
    1. Cumulative broker-truth per-symbol cap   (e4986ad; live, observed binding)
       — reads positions_truth (GREEN + ttl, fail-closed), refuses |net| growth
         beyond per-symbol cap; exits/flips pass; pending-adds tracked
       — markers: FUTURES_POSITION_CAP_BLOCK / FUTURES_POSITION_CAP_UNVERIFIED
    2. Env gate (redundant backstop)            (82136d7 + systemd drop-in 91-disable-futures-exec.conf)
       — CHAD_DISABLE_FUTURES_EXECUTION / CHAD_DISABLE_FUTURES / CHAD_FUTURES_EXECUTION_ENABLED=0
       — marker: FUTURES_EXECUTION_DISABLED_SKIP (now fires only behind the cap)

  FILL-FEEDBACK INTEGRITY (NEW v9.7)
    harvester: no closed_trade.v1 emission for single OPEN fills      (b58bc9c)
    trade_closer seeder: skips ibkr_harvest / len(fill_ids)<2 phantoms (b58bc9c)

  CURRENCY TRUTH LAYER (NEW v9.7 — BOX-034A/B, spans collector → publisher → consumers)
    single-writer ibkr_equity (CAD canonical; USD display-only)        (b807f1f)
    fail-closed NetLiquidation currency gate                           (ec10789)
    validated live USDCAD; 1.40 fallback removed; fail-closed FX       (dc5eb9b)
    v2-collector tag preservation (no strip flicker)                   (c169c09)
    per-value <value>_currency/_ok tags on dynamic_caps + pnl_state    (b6d333f, 40a9c55)
    warn-mode consumer assertions (CURRENCY_WARN_*; enforce-flip gated)(d0f7a78)

  GATEWAY RELIABILITY (NEW v9.7)
    stop_bus symmetric latency hysteresis (live)                       (f3ab3d8, 291bb84)
    nightly Gateway restart timer + verifier (Channel 1 install pending)(70e150b)
    Gateway version audit tool + R20 health rule                       (3839bfb)
```

### 1.3 Cross-cutting safety promises (v9.7 additions)
All v9.6 promises carry forward (no live BAG, no DOM, no Kraken-CA futures, no Coinglass, no scanner writeback, no position_truth_v2 production writes, port-binding allowlist, OnFailure completeness). New in v9.7:
- **No FUT open may grow |net| beyond the per-symbol cumulative cap**, evaluated against broker-authority truth, fail-closed on stale/non-GREEN truth (`e4986ad`).
- **No harvester-authored `closed_trade.v1` records for single fills**; the seeder must not consume harvester phantoms (`b58bc9c`).
- **No silent currency assumption on equity values**: canonical values carry `_currency`/`_ok` tags; the collector and FX path fail closed on mismatch/unavailability (`ec10789`, `dc5eb9b`).
- **No futures-execution re-enable (env-gate removal) before the Bug A fd-leak root fix lands** (`9385577` §5 coupling, recorded into the disposition PA at `d621aa8`-adjacent amendment).

---

## 2. Current Runtime Truth — 2026-06-04T21:14Z (draft read)

All values read directly from `runtime/` at draft time.

### 2.1 Equity, risk cap
From `runtime/dynamic_caps.json` (ts 2026-06-04T21:14:04Z):
- `total_equity` = **$289,330.97** (CAD-canonical post-BOX-034A; tagged via `b6d333f`)
- `portfolio_risk_cap` = **$14,466.55**
- From `runtime/portfolio_snapshot.json` (ts 21:14:31Z): `ibkr_equity` = $289,106.09, `ibkr_equity_currency` = **"CAD"**; `kraken_equity` = $256.67, `kraken_equity_currency` = "CAD".
- **Caveat (Q1):** equity includes mark-to-market on the undisposed Bug B futures book (§3); growth from v9.6's $265,577 is partly that artifact.

### 2.2 SCR posture
From `runtime/scr_state.json` (ts 2026-06-04T21:14:29Z):
- `state` = **CONFIDENT**, `sizing_factor` = 1.0, `paper_only` = false
- stats: effective_trades=205, sharpe_like=3.530, win_rate=0.761, total_pnl=+$27,299.82, paper_trades=4159, max_drawdown=−$290.00, excluded_manual=16, excluded_untrusted=321, live_trades=0
- **Caveat (Q3):** in-sample paper statistics, contaminated by Bug B; the drawdown figure does not reflect open-futures excursion.

### 2.3 Live readiness
From `runtime/live_readiness.json` (ts 2026-06-04T21:12:14Z): `ready_for_live` = **false** ✓

### 2.4 Tier state
From `runtime/tier_state.json` (ts 2026-06-04T21:12:25Z): `tier_name` = **SCALE**; `current_equity_usd` = 289,330.97 (field name is a known naming drift — value is the CAD-canonical figure consistent with dynamic_caps).

### 2.5 Reconciliation and positions truth
From `runtime/reconciliation_state.json` (ts 2026-06-04T21:12:31Z): `status` = **GREEN**, `drifts` = `[]`, counts `{chad_open: 19, chad_strategy_open: 19, broker_positions: 19}`, `broker_source` = `ibkr:clientId=83`.
From `runtime/positions_truth.json` (ts 2026-06-04T21:14:31Z): `broker_authority_status` = **GREEN**, `truth_ok` = true, **`ttl_seconds` = 90** (the `3bfa924` fix, live).

### 2.6 The futures book (broker truth; carried open, undisposed)
Per the disposition audit (`7744507`, positions_truth GREEN @ 2026-06-04T01:21:36Z):

| Symbol | conId | Net | Avg entry | uPnL USD (at read) |
|---|---|---|---|---|
| M6E | 838628915 | +217 | 1.16413 | −8,209 |
| M2K | 770561189 | −25 | 2890.75 | −1,381 |
| MCL | 661016559 | +50 | 88.38 | +33,733 |

Gross notional ~$3.99M USD (~17.9× gross vs ~$309K CAD equity at that read). Static since 2026-06-01T17:19:39Z (last futures fill, pre-fix); net pinned by `CAP_BLOCK` every cycle. **LIVE market risk, not frozen PnL.** Order 5137: RESOLVED — zero open orders account-wide via read-only `reqAllOpenOrdersAsync` probe (`d621aa8`); re-probe required immediately before any actual flatten.

### 2.7 Guard binding evidence (journal, 2026-06-04)
`FUTURES_POSITION_CAP_BLOCK symbol=M6E strategy=omega_macro side=BUY qty=2.0 net=217.0 projected=219.0 cap=3 reason=cumulative_cap` — firing at the ~10-min cooldown cadence; zero `FUTURES_POSITION_CAP_UNVERIFIED` since the ttl fix (§8(b) zero-UNVERIFIED-over-hours observation formally still open per `f4d8cea`).

### 2.8 Test baseline
2813 passed at the last recorded full run (logged in `POSITIONS_TRUTH_ttl_margin_fix` status log, 2026-06-04T00:21Z; commit `3bfa924`). Delta from v9.6: +174.

### 2.9 Git baseline
- HEAD: `9385577` (2026-06-04), branch `main`.
- v9.6 tag `SSOT_v9_6_2026-05-27` → commit `50d103a`; merged at `d9b0c7f`.
- Working tree note: one untracked file `ops/pending_actions/PR-05_OFFICIAL_36_amendment_line84_patch_note_2026-05-26.md` (dated 05-26, never committed) — flagged for operator disposition at lock time.

---

## 3. Bug B — Futures Runaway (root cause, fixes, open disposition)

### 3.1 Root cause (two composed defects; spec `0d34315`)
- **Defect B (ROOT — fill-feedback corruption):** `ibkr_paper_fill_harvester` wrote a phantom `closed_trade.v1` record for every harvested single OPEN fill (pnl=0, entry==exit, one fill_id, tag `ibkr_harvest`). The trade_closer seeder marked those fill_ids consumed → opens never enqueued → guard auto-closed → same-side dedupe saw nothing open → re-entry every cycle. M6E: 334 BUYs / 0 SELLs → +217.
- **Defect A (AMPLIFIER):** omega_macro/gamma_futures entries are position-unaware; `max_contracts` was per-ORDER only.

### 3.2 Fixes (landed and live)
- **Fix B** (`b58bc9c`): harvester `closed_trade.v1` emission removed (source) + seeder skips `ibkr_harvest`/single-fill phantoms (consumer defense). FILLS + dedupe untouched. 6 tests.
- **Fix A** (`e4986ad`): cumulative broker-truth per-symbol futures cap at the live_loop chokepoint; fail-closed on non-GREEN/stale truth; exits/flips pass; 14 tests. Activated by a gated chad-live-loop restart (PA `64545a8`); **live-verified binding** (§2.7; recorded as live-verified in `7744507`). *Doc-drift note: the restart PA's own status log still reads PENDING — supersede at lock.*
- **ttl margin** (`3bfa924`, executed per `f4d8cea`): positions_truth `ttl_seconds` 60→90, restoring freshness margin vs the ~61–65s publisher cadence; killed the benign `CAP_UNVERIFIED` flicker; zero-restart oneshot deploy.
- **Stopgap env gate** (`82136d7`, 2026-06-01 + systemd drop-in installed 2026-05-30): remains ARMED as redundant backstop.

### 3.3 Open disposition (PENDING operator decision — `7744507`)
Two independent decisions: **(A) flatten** the legacy book; **(B) remove the env gate** (cap becomes sole guard; cap-completeness confirmed — byte-identical predicate, evaluated before the gate, no hole). Sequencing facts: (b) flatten→remove is cleanest; (c) remove→flatten is dominated; **gate removal (b)/(c) is now GATED on the Bug A fix landing first** (`9385577` — the gate is the fd-leak's only mask). Order-5137 collision risk resolved (`d621aa8`).

---

## 4. BOX-034 Series — Currency/Equity Truth (landed)

Increment trail (all landed; full list in §6): spec `fe59de5`/`499d5a7` → Inc 1 single-writer `b807f1f` → Inc 2 fail-closed currency gate `ec10789` (naming convention `3cd3610`) → Inc 3 Step 0b live-FX `dc5eb9b` → BOX-034B tag-strip fix `c169c09` (root-cause corrected `6caecdf`; box raised `66f7c86`) → Inc 3 Step 1a/1b tags `b6d333f`/`40a9c55` (live via gated orchestrator restarts `fcc9986`/`fc51872`) → Step 2 warn-mode `d0f7a78` (live-verified `6713101`, published `8f93673`) → Inc 4 hermetic reconciliation test `3e24bb7` (`33a92e9`).
**Remaining:** the gated **enforce-flip** (warn → enforce) — the only open BOX-034A increment. BOX-034C (oneshot watchdog/TimeoutStartSec, `7fa4a26`) remains PENDING.

---

## 5. Pending Actions Register (as of draft)

New/changed since v9.6 lock:

| PA | Status | Blocks |
|---|---|---|
| `BUGB_futures_runaway_fill_feedback_fix_2026-06-03` | Fixes B+A LANDED and live | — (closed by code; formal status update at lock) |
| `BUGB_fixa_live_loop_restart_2026-06-03` | EXECUTED de facto (cap live, §2.7); status log not updated — doc drift | — |
| `BUGB_disposition_futures_book_and_gate_2026-06-04` | **PENDING operator decision** (A flatten / B gate removal; b/c gated on Bug A) | Book risk closure; cap-only operation |
| `POSITIONS_TRUTH_ttl_margin_fix_2026-06-03` | EXECUTED 2026-06-04 (`3bfa924`); §8(b) observation open | — |
| `L1_bug_a_event_loop_leak_2026-06-04` | **PENDING GO** (HIGH — masks only via env gate) | Gate removal; futures re-enable |
| `L3_loss_guard_enforce_code_default_2026-06-04` | **PENDING GO** (risk-tightening) | Loss-guard durability |
| `IBKR_GATEWAY_VERSION_UPGRADE_2026-05-28` | PENDING (current=1037, target=1045, stale) | Gateway currency |
| `HEALTH_ALERT_PIPELINE_operational_2026-05-28` | VERIFIED operational (`bfdcb7e`) | — |
| Gateway nightly-restart install (`70e150b` runbook) | Channel 1 PENDING | Socket-leak wedge prevention |
| BOX-034A enforce-flip | GATED, not yet raised as execution PA | Currency enforcement |
| BOX-034B residuals (§3a de-minimis: BTC omission, stale-FX round-trip) | OPEN (documented `6caecdf`) | — |
| BOX-034C oneshot watchdog (`7fa4a26`) | PENDING | Publisher-hang detection |
| Buying-power/margin pre-trade check | NOT YET FILED (split from L3 §10 — control confirmed absent) | Pre-trade margin safety |

Carried from v9.6 unchanged: R1/position_truth_v2 migration Steps 1–3, soak evaluator build, Channel 1 install queue (alert template, OnFailure directives, port-9618 closeout), IBKR auto-recovery design, SCHEMA-VERSION-1 backfill.

---

## 6. Commit Chain Since v9.6 (`SSOT_v9_6_2026-05-27..HEAD` = 45 commits)

Verified against `git rev-list --count SSOT_v9_6_2026-05-27..HEAD` = 45 (re-run 2026-06-04T22:32Z). Enumerated oldest→newest, grouped by workstream §0.2(1)–(5). No commit in the span is unaccounted for.

**Merge anchor:** `d9b0c7f` Merge institutional forensic closeout: HIGH-severity fixes + D1/D2/D3 + v9.6 SSOT.

**(1) Post-closeout reliability series (2026-05-28/29):**
```
f3ab3d8  stop_bus: symmetric hysteresis on broker_latency trip
291bb84  stop_bus: activate Fix A hysteresis (breach_streak_started_at; live in production)
70e150b  Fix B: Gateway nightly restart timer + service + verifier + Channel 1 runbook
3839bfb  Fix C: Gateway version audit tool + R20 health rule + upgrade PA (1037→1045, stale)
bfdcb7e  Fix D: health-monitor alert pipeline operational (verify only); e2e smoke tests
0799854  patches: regenerate against production reality (chad-backend ExecStart drift; %n→%N)
4ba329e  PR-05: options-chain discovery via reqSecDefOptParams (replaces pacing-throttled path)
```
**(2) Futures-submission P0 sequence (2026-05-31 → 06-01):**
```
252bea9  P0: fail closed live loop broker submissions outside ibkr_live
f7ba4f7  Revert "P0: fail closed live loop broker submissions outside ibkr_live"
82136d7  P0-stopgap: gate net-new futures entries behind CHAD_DISABLE_FUTURES_EXECUTION; live-verified
```
**(3) BOX-034/034A/034B/034C currency/equity truth (2026-06-01 → 06-03):**
```
fe59de5  docs(BOX-034A): canonical equity currency unification spec (CAD base; PENDING)
499d5a7  docs(BOX-034): amended by BOX-034A — supersession markers + dual-writer race root cause
b807f1f  fix(BOX-034A inc1): single-writer portfolio_snapshot.ibkr_equity (CAD canonical; USD display-only)
3cd3610  docs(BOX-034A): resolve currency field-naming convention (<value>_currency tags)
ec10789  feat(BOX-034A inc2): currency-aware fail-closed collector (NetLiquidation currency==CAD)
66f7c86  docs(BOX-034B): kraken_equity single-writer unification follow-up box (PENDING)
dc5eb9b  feat(BOX-034A inc3 step0b): validated live USDCAD; 1.40 fallback removed; fail-closed FX
c169c09  fix(BOX-034B): v2 collector preserves publisher-authored snapshot keys (tag-strip flicker fix)
6caecdf  docs(BOX-034B): correct root cause — v2-collector tag-strip; +§3a de-minimis gaps
7fa4a26  docs(BOX-034C): oneshot watchdog/timeout follow-up (13h publisher hang; PENDING)
62491b0  docs(BOX-034): cross-link follow-up lineage across 034/034A/034B/034C
b6d333f  BOX-034A inc3 step1a: tag dynamic_caps total_equity_currency/_ok
40a9c55  BOX-034A inc3 step1b: tag pnl_state account_equity_currency/_ok
d80f44b  docs(CLAUDE.md): fix stale full_cycle_preview verification command
fcc9986  BOX-034A inc3: PA — restart chad-orchestrator to activate Step 1a tagging live
fc51872  BOX-034A inc3: mark orchestrator-restart PA COMPLETED — Step 1a/1b live-verified
d0f7a78  BOX-034A inc3 step2: warn-mode currency assertions for 5 equity consumers (CURRENCY_WARN_*)
770a25b  docs(BOX-034A): record inc3 step1a/1b live + step2 warn-mode shipped; inc4 remaining
76a88f7  BOX-034A inc3: PA — restart chad-orchestrator to activate Step 2 warn-mode live
6713101  BOX-034A inc3: Step 2 warn-mode LIVE-verified — restart PA COMPLETED
8f93673  docs(BOX-034A): record inc3 series published to origin/main (GitHub-verified 6713101)
3e24bb7  BOX-034A inc4: rewrite reconciliation test — hermetic 4A + skew-gated currency-explicit 4B
33a92e9  docs(BOX-034A): mark inc4 DONE; only gated enforce-flip remains
```
**(4) Bug B futures runaway (2026-06-03 → 06-04):**
```
0d34315  Bug B: PA — fix futures runaway (Defect B harvester phantom-close + Defect A position-unaware entry)
b58bc9c  Bug B Fix B: stop harvester phantom-close poisoning (source + seeder consumer defense; 6 tests)
e4986ad  Bug B Fix A: cumulative broker-truth per-symbol futures position cap
64545a8  Bug B: PA — gated chad-live-loop restart to activate Fix A (PENDING)
d4358bc  PA — positions_truth ttl 60→90s freshness-margin fix (PENDING)
3bfa924  positions_truth: raise published ttl_seconds default 60→90s
f4d8cea  docs(PA): mark positions_truth ttl fix EXECUTED — deploy live-verified; §8(b) observation open
7744507  Bug B: PA — disposition decision facts (book ~17.9×, cap-complete, sequencing); PENDING decision
d621aa8  docs(PA): Bug B disposition §3 open-order caveat RESOLVED — order 5137 NOT PRESENT
```
**(5) New pending hardening (2026-06-04):**
```
d8441b3  L3: PA — per-strategy loss-guard ENFORCE as code default (shape B); PENDING GO
9385577  L1: PA — root-fix Bug A event-loop/fd leak (bounded executor); gate-removal coupling; PENDING GO
```

---

## 6A. Inherited SSOT Body — Carried from v9.6 (with v9.7 delta annotations)

This section carries the full v9.6 body forward so this document is self-contained; v9.6 (`c9888b1…` sha256, unmodified) remains the frozen historical artifact. Where v9.7 changed reality, the delta is annotated inline as **[v9.7Δ]**.

### 6A.1 Hot-path architecture stack (carried verbatim from v9.6 §1.2)
```
  1. Signal Layer            (strategies → RoutedSignal)
  2. Routing Layer           (signal_router, signal_guard, dedupe)
  3. Risk & Allocation Layer (dynamic_risk_allocator, profit_lock, tier_state)
  4. Execution Layer         (ibkr_adapter, paper_trade_executor, LiveGate)
  5. Attribution Layer       (paper_exec_evidence_writer, fills ledger)
  6. Reconciliation Layer    (reconcile_positions, position_guard, drift detector)
  7. Dashboard/Operator Layer(status server, metrics, Telegram)
  8. Governance Layer        (Pending Actions, SCR, stop_bus, epoch trackers)
```
Horizontal layers carried from v9.6: **Position Truth v2** (design+stub, NOT WIRED: `chad/core/position_truth_engine.py`, `chad/schemas/position_truth_v2.py`, `chad/validators/position_truth_v2.py`, design doc `docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md`; `1626f20`); **Service Reliability** (Channel 2 landed, Channel 1 install pending: `chad/ops/service_failure_alert.py`, `ops/systemd_templates/chad-service-alert@.service`, `chad/validators/port_binding.py`, `config/port_binding_allowlist.json`, installed pre-commit hook; `1e00568` **[v9.7Δ:** patches regenerated against production reality `0799854`**]**); **Soak Governance** (mechanical evidence-gate rule LOCKED `ac04e16`; evaluator NOT BUILT — RED defaults FAIL/NOT-COUNTABLE). **[v9.7Δ:** plus the new v9.7 layers in §1.2 — Futures Entry Guard Stack, Fill-Feedback Integrity, Currency Truth, Gateway Reliability.**]**

### 6A.2 Strategy fleet (carried from v9.6-era registers)
17 strategies registered (`chad/strategies/__init__.py`) ∩ 17 tier-enabled (SCALE). Classification of record (Paper Completion Register 2026-05-27 + BOX-049, 0 UNKNOWN): ACTIVE_PRODUCTIVE ×5 (alpha, alpha_futures, alpha_intraday, gamma_futures, omega_macro) · ACTIVE_BLOCKED_BY_DATA ×1 (delta — placeholder emission quarantined) · REGIME_SILENT ×10 (alpha_intraday_micro, alpha_options, beta, beta_trend, delta_pairs, gamma, gamma_reversion, omega, omega_momentum_options, omega_vol) · HALTED_EDGE_DECAY ×1 (alpha_crypto, consecutive_negative_10 since 2026-05-08). ALPHA_FOREX deferred (unregistered). **[v9.7Δ:** omega_macro is the Bug B re-entry strategy; its futures entries are now cap-bound (§2.7). Classifications otherwise carry; next reclassification at the post-disposition soak.**]**

### 6A.3 Forensic-audit HIGH dispositions (carried from v9.6 §3.1)
| HIGH_ID | Disposition at v9.6 | [v9.7Δ] |
|---|---|---|
| TRUTH-RECONCILE-1 | Architecture landed (`1626f20`) | unchanged — v2 wiring still pending (RISK-V9-6-02) |
| OPTIONS-CHAIN-1 | FIXED (`b759cb6`) | further hardened: `adacf6f` retry/backoff, discovery rewrite `4ba329e`; recovery verified 05-27 |
| STOP-BUS-RECOVERY-1 | PARTIALLY FIXED (`804abdd`) | hysteresis now LIVE (`f3ab3d8`, `291bb84`) |
| FUTURES-ROLL-1 | FIXED (`7a0c100`) | unchanged |
| PORT-BINDING-1 | PARTIALLY FIXED (`ae614bb`, `1e00568`) | unchanged — 9618 Channel-1 closeout still pending |
| HISTORICAL-PLACEHOLDER-1 | FIXED (`29e7197`) | unchanged — `--apply` quarantine decision still open |

MED items: SCHEMA-VERSION-1 (78 files) still DEFERRED; DOC-RECONCILE-1 closed by v9.6, kept closed by this cut.

### 6A.4 Services and runtime files (carried + current)
Service surface (as audited by the wiring + calibrated audits and the scorecard): `chad-orchestrator`, `chad-live-loop` (`CHAD_EXECUTION_MODE=paper`, `CHAD_PER_STRATEGY_LOSS_LIMIT_ENFORCE=1`, `CHAD_DISABLE_FUTURES_EXECUTION` drop-in `91-disable-futures-exec.conf`), `chad-reconciliation-publisher`, `chad-ibkr-collector` (+v2 collector), `chad-shadow-snapshot.timer` + `chad-scr-sync.timer` (canonical shadow Path A, `1575979`), `chad-ibkr-broker-events` + `chad-ibkr-paper-fill-harvester` (oneshots **[v9.7Δ:** harvester no longer emits closed_trade.v1, `b58bc9c`**]**), `chad-options-chain-refresh`, `chad-health-monitor` (R01–R20), `chad-backend`/dashboard (allowlisted `chad-dashboard:8765`; 9618 closeout pending), Telegram notify path. Built-not-installed: `chad-gateway-nightly-restart.timer` (`70e150b`), `chad-service-alert@.service` template.
Key runtime truth files (all read fresh 2026-06-04T22:32Z for this cut): `positions_truth.json` (GREEN, ttl 90), `reconciliation_state.json` (GREEN 19/19/19), `dynamic_caps.json`, `portfolio_snapshot.json` (CAD-tagged), `pnl_state.json` (currency-tagged), `scr_state.json` (CONFIDENT/1.0), `live_readiness.json` (false), `tier_state.json` (SCALE), `stop_bus.json` (clear), `position_guard.json` + `position_guard_drift.json`, `strategy_allocations.json` (edge-decay/halt SSOT), `execution_environment.json`. Named-but-absent paths (functional, state lives elsewhere — wiring audit): `edge_decay_state.json`, `futures_chain_state.json`, `ibkr_gateway_version.json`.

### 6A.5 Soak governance contract (carried verbatim-by-reference from v9.6 §4.1)
The mechanical evidence-gate rule (`ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md`, sha256 `800480a5…`, LOCKED `ac04e16`) remains the only authority for classifying broker_authority_RED windows: all 8 conditions mechanically verifiable or FAIL/NOT-COUNTABLE; no operator override; evaluator NOT BUILT. **[v9.7Δ:** clean-soak remains 0/5; the Bug B window additionally contaminates any session overlapping it (RISK-V9-7-06).**]**

### 6A.6 Channel 1 install queue (carried from v9.6 §8, extended)
1. Install `chad-service-alert@.service` template (runbook `ops/runbooks/INSTALL_chad_service_alert_template_2026-05-27.md`). 2. `chad-options-chain-refresh` OnFailure directive. 3. `chad-backend` OnFailure + uvicorn host flip to 127.0.0.1 (closes PORT-BINDING-1/9618). **[v9.7Δ adds:]** 4. Gateway nightly-restart timer install (`70e150b` runbook). 5. Gateway version upgrade 1037→1045 (PA `3839bfb`-adjacent). All require explicit operator GO; none authorized by this document.

### 6A.7 Governance rules (carried verbatim from v9.6 §0.5 / CLAUDE.md)
Ten rules unchanged: (1) one change at a time; (2) no full rewrites; (3) no direct config mutation — Pending Actions only; (4) verification sequence after every change; (5) never modify `runtime_FREEZE_*`/`data_FREEZE_*`; (6) never modify systemd units without explicit instruction; (7) never restart live services without explicit instruction; (8) commit+tag after each completed P0/P1/P2 item; (9) `python3` via venv; (10) pytest with `CHAD_SKIP_IB_CONNECT=1`. **[v9.7Δ:** `d80f44b` corrected the stale full_cycle_preview command in CLAUDE.md.**]**

---

## 7. Live Readiness (Unchanged)

Live promotion still requires operator GO, a clean multi-session paper soak, and clean equity/performance history. v9.7 grants nothing. Additionally, per CURRENT REALITY Q3: **no out-of-sample, cost-adjusted edge validation exists**; that artifact must exist before any live discussion is meaningful. Soak status: no clean-session credit claimed in this span (Bug B contamination + evaluator still unbuilt); clean-soak count remains **0 / 5**.

---

## 8. Known Issues and Residual Risks (v9.7)

Carried forward: RISK-V9-6-01 (SCR UNKNOWN flutter), RISK-V9-6-02 (position_truth_v2 unwired), RISK-V9-6-03 (soak evaluator unbuilt), RISK-V9-6-04 (Channel 1 installs pending). New:

- **RISK-V9-7-01 — Undisposed runaway futures book.** ~17.9× gross leverage live market risk; uPnL sign/size mobile. PENDING operator decision (`7744507`).
- **RISK-V9-7-02 — Bug A fd leak dormant, not fixed** (`9385577`). Masked solely by the env gate; gate removal before the fix re-arms the 2026-05-30 freeze class.
- **RISK-V9-7-03 — Loss-guard enforcement is env-var-fragile** until L3 lands (`d8441b3`).
- **RISK-V9-7-04 — No buying-power/margin pre-trade check exists** (confirmed absent; future PA).
- **RISK-V9-7-05 — Currency assertions warn-only** until the gated enforce-flip.
- **RISK-V9-7-06 — Statistics contamination.** SCR/equity series include the Bug B period; any soak or confidence claim must exclude or annotate it.
- **RISK-V9-7-07 — P0 fail-closed submission gate was reverted** (`252bea9` → `f7ba4f7`); the narrower stopgap (`82136d7`) + cap (`e4986ad`) replaced it. The broad fail-closed posture for non-ibkr_live submissions remains unimplemented.

### 8.1 OPEN ITEMS REQUIRING OPERATOR DISPOSITION

The following FOUR items remain open at lock. Locking this SSOT does **not** close any of them; each requires its own explicit operator disposition. They are recorded here so the document of record carries the complete open-items list.

**(a) Untracked PR-05 amendment note — keep/discard PENDING.**
`ops/pending_actions/PR-05_OFFICIAL_36_amendment_line84_patch_note_2026-05-26.md` (dated 2026-05-26) exists in the working tree, never committed. Deliberately **excluded** from the v9.7 lock commit (single-file commit discipline). Disposition options: commit it, move it to `_applied/`, or discard it.

**(b) Stale Fix-A restart PA status log.**
The gated chad-live-loop restart that activated Bug B Fix A is **EXECUTED de facto** — the cap is live and observed binding (§2.7, journal `FUTURES_POSITION_CAP_BLOCK`, recorded live-verified in `7744507`) — but the restart PA's own status log (`BUGB_fixa_live_loop_restart_2026-06-03`, raised at `64545a8`) still reads PENDING. Doc drift only; operator must authorize the status-log supersession edit (a PA-doc change, not made by this lock).

**(c) 2026-05-30 systemd drop-ins exist outside version control.**
`/etc/systemd/system/chad-live-loop.service.d/91-disable-futures-exec.conf` (2026-05-30 13:15 — the `CHAD_DISABLE_FUTURES_EXECUTION` env gate, §2.7/§3.2) and `90-fd-limit.conf` (2026-05-30 01:45 — the fd-limit raise from the Bug A freeze response) are live, load-bearing safety configuration with **no copy under `ops/systemd_templates/`** and no git history (verified 2026-06-04). A host rebuild or accidental deletion would silently remove the futures gate (currently the sole mask for dormant Bug A — RISK-V9-7-02). Disposition: operator-authorized capture of both drop-ins into `ops/systemd_templates/` (read-only copy; no unit modification involved).

**(d) §8(b) observation-window clock still running.**
The positions_truth ttl 60→90 fix is live-verified (`3bfa924`, `f4d8cea`), and zero `CAP_UNVERIFIED` markers have been observed since — but the formal §8(b) observation window declared in the PA (sustained zero-`CAP_UNVERIFIED` over a multi-hour window on fresh GREEN truth) has not been formally closed. Operator must confirm the window result and close the observation.

**Additional operator-confirmation flags (raised by the v9.7 expansion pass; not dispositions, but awaiting operator acknowledgement):**
- **`operator_mode=ALLOW_LIVE` while posture is paper** (red-team scorecard L8, 2026-06-04; reason `auto_refresh_allow_entries_non_live`). Documented tooling (e.g. the close-guard CLI) treats ALLOW_LIVE as a fail-closed trigger; the readiness publisher holds the line regardless, but the intent value contradicts the paper posture (GAP-016/BOX-037 context).
- **Chaos-drill proof staleness** — `reports/ratification/PROOF_CHAOS_DRILLS_20260403.json` (all_passed=true) predates the 04-19/21 overhaul and Bug B; confirm acceptance as-is or raise a re-drill PA (TM.7 / RT.7).

---

## 9. Phase Roadmap (post-v9.7)

Immediate (operator decisions): 1) Bug B disposition — flatten and/or gate-removal path (gate removal requires Bug A first). 2) GO/no-GO on L1 (Bug A executor fix). 3) GO/no-GO on L3 (loss-guard enforce default). 4) BOX-034A enforce-flip authorization.
Near-term (Channel 2): 5) Buying-power/margin pre-trade check read-first + PA. 6) Soak evaluator. 7) position_truth_v2 migration Step 1 (tune `ledger_ttl` first). 8) BOX-034C watchdog. 9) SCHEMA-VERSION-1 backfill.
Mid-term: 10) Channel 1 install queue (alert template, OnFailure, port-9618, Gateway nightly restart, Gateway 1037→1045). 11) position_truth_v2 shadow + repoint. 12) **Out-of-sample, cost-adjusted edge validation harness — prerequisite to any live conversation (Q3).**
Long-term: 13) Clean Paper Epoch 3 soak (5 sessions, post-disposition). 14) Operator GO for live.

---

## 10. Definition of Done (drafted → locked 2026-06-04)

- [x] Final document at `docs/CHAD_UNIFIED_SSOT_v9.7_2026-06-04.md` (renamed from the DRAFT filename at lock).
- [x] DOCUMENT-OF-RECORD supersession banner at top; all prior scorecards/audits/checklists retired as status sources.
- [x] MASTER SCORECARD section (§MS): every prior scorecard inventoried with path + counted items; non-reconciliation of percentages explained; no overall percentage computed.
- [x] L0–L8 TRUST MATRIX (§TM): per-layer requires/DONE/MISSING/status, every DONE claim hash- or runtime-cited; stale "L0 BROKEN/$84K" status explicitly superseded.
- [x] RED-TEAM FROM FIRST PRINCIPLES (§RT): current adversarial case; stale prior red-team claims flagged at RT.9.
- [x] Inherited v9.6 body carried (§6A) with delta annotations; v9.6 file unmodified (sha256 pinned §0.1).
- [x] Lineage verified and corrected (v9.4/v9.5 = forward-errata; v9.5 filename drift flagged).
- [x] All 45 commits since v9.6 accounted for and enumerated inline (§6); every asserted delta carries a hash.
- [x] CURRENT REALITY three-questions section present; edge recorded as UNPROVEN (Q3, TM.6).
- [x] No overall completion percentage anywhere in this document (artifact-internal figures are quoted only as attributed historical claims in §MS).
- [x] OPEN ITEMS REQUIRING OPERATOR DISPOSITION (§8.1): all four open items listed — (a) PR-05 note, (b) stale Fix-A restart PA log, (c) unversioned 2026-05-30 systemd drop-ins, (d) §8(b) observation window.
- [x] Operator review complete; explicit operator GO to lock received 2026-06-04.
- [x] Lock: single-file commit + annotated tag `SSOT_v9_7_2026-06-04` (lands with this commit).
- [ ] §8.1 items (a)–(d) — each requires its own subsequent operator disposition; NOT closed by this lock.

---

## 11. No-Live Confirmation

This document does not authorize live trading.
`ready_for_live` remains false. `allow_ibkr_live` remains false. `allow_ibkr_paper` remains true.
No broker orders may be placed or cancelled under this document.
The futures book disposition, env-gate removal, L1, L3, and the BOX-034A enforce-flip each require their own explicit operator GO.

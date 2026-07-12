# CHAD Unified SSOT v9.8
# Full Cut — Post-Epoch-3-Reset: Currency Truth (CAD-native), Edge-Validation Harness, Margin BLOCK, L1-CLD, Weekend-Fix Pre-Open Deployment

> **DOCUMENT OF RECORD.** This is a **full cut**, evidence-first, and **supersedes v9.7 as the single status source.**
> All prior scorecards, audits, parity sweeps, and checklists are retired as *status* sources; they survive only as
> attributed historical claims (see §MS). Runtime figures below are point-in-time reads (2026-07-12) against a live,
> paper-posture system — TTLs make several of them moving targets; re-read before relying on any single number.
>
> **Supersession pin.** Prior document of record: `docs/CHAD_UNIFIED_SSOT_v9.7_2026-06-04.md`
> sha256 `5f2b71fd5db203e1bdedbe4cfdcfb2c71707944398b1de9e7c5da69ccee2f53c` (unmodified by this cut; carried by reference).
> v9.7 tag: `SSOT_v9_7_2026-06-04`.
>
> **No overall completion percentage appears anywhere in this document.** Percentages are quoted only where a prior
> artifact asserted one, and only as an attributed historical claim.

---

## CURRENT REALITY — The Three Questions (read this first)

### Question 1 — Does CHAD know its true financial state?
**PARTIAL — account-level YES, position-level NO.** At the account level money-truth is CAD-honest and
broker-authoritative: `positions_truth.json` `truth_ok=true`, `broker_authority_status=GREEN`,
`truth_source=BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER`; `reconciliation_state.json` GREEN with zero
mismatches/drifts; equity is CAD-native (`equity_history.v2`, `usd_ok=false`, USD retired to null). **But**
position-level guard truth is NOT clean: `position_guard_drift.json` (v2) carries **5 `qty_mismatch` rows**
where CHAD's guard is ≈1.9× the broker's position on IWM/LLY/TLT/UNH/V (see §5). And the Epoch-3 reset
(2026-06-30) re-baselined equity to **~$1.0M CAD**, a ~3.8× jump over the pre-reset ~$264K CAD that is
consistent with a fresh/default-funded paper account but is not independently proven in the artifacts read (§5.4).

### Question 2 — Do CHAD's safety controls provably bind?
**STRENGTHENED this cycle, with one gate still shadow-only.** New controls proven binding in live journal:
RTH market-hours gate (equity/ETF blocked outside RTH, **zero orders, zero idempotency rows** — §5.2);
phantom-guard-on-fill + boot phantom-reconcile (`cleared=3` at boot — §5.1); futures execution hard-disabled
at the submit chokepoint (`CHAD_DISABLE_FUTURES_EXECUTION=1`, `91-disable-futures-exec` mask, commit `554a36a`);
L1-CLD cross-loop deadlock recovery activated (`fe25007`). **Not yet binding:** the margin/buying-power BLOCK
gate is wired into the IBKR chokepoint in **SHADOW mode only** — it evaluates and logs but never blocks
(`3a1fc25`, `6f58b39`). Residual contradiction: `operator_intent.json` = `ALLOW_LIVE` while posture is paper
(the readiness publisher holds `ready_for_live=false` regardless — §RT).

### Question 3 — Is there cost-adjusted, out-of-sample edge?
**UNPROVEN — and now explicitly zero-sample.** The Epoch-3 reset zeroed the trusted sample: SCR = **WARMUP**,
`effective_trades=0`, `sizing_factor=0.1`, `total_pnl=0.0`, `win_rate=0.0` (paper_trades=2148, all excluded/untrusted).
The out-of-sample, cost-adjusted **edge-validation harness** was BUILT this window (Phases 0–5, commits `e808408`…`09b46c2`)
but has produced **0 admissible windows** to date. No edge claim is made or implied by this document.

---

## MS. MASTER SCORECARD — Reconciliation of All Prior Scorecards and Audits

### MS.1 Inventory of prior status artifacts (retired as status sources by this cut)
| Artifact | Path / ref | Headline it asserted | Disposition here |
|----------|-----------|----------------------|------------------|
| SSOT v9.7 (prior doc of record) | `docs/CHAD_UNIFIED_SSOT_v9.7_2026-06-04.md` (sha256 pinned above) | Post-institutional-closeout; edge UNPROVEN; four §8.1 operator items | Superseded; §8.1 items all closed pre-reset (see §6/`37f0e6b`,`8b549e2`,`0bc47bc`,`0af767a`) |
| Forensic full-system audit | memory `forensic-audit-2026-06-11`; `reports/forensic_audits/` | "≈74% SSOT parity (98/132), 39 gaps (5 P1), no P0" | Historical; gaps mapped in §7. The 74% is an **attributed historical claim**, not adopted |
| P1 gap reconciliation (2026-06-16) | memory `p1_gap_reconciliation_2026_06_16` | 8 P1 "system-truth" gaps; 6 closed / 2 partial | Mapped in §7 |
| SSOT v9.7 parity audit (2026-06-09) | `audits/SSOT_v9_7_PARITY_2026-06-09.md` | zero reverse drift; 13 forward drifts | Historical |
| Master pre-reset audit (2026-06-21) | memory `master_pre_reset_audit_2026_06_21` | CONDITIONAL GO reset; 4 must-resolve (H2/H3/H4/H5) | Reset executed 2026-06-30 (§3) |
| Parity sweeps (May) | `reports/parity_audit/*` | per-line COMPLIANT/OPEN | Historical |
| WKF pre-open deployment PA | `ops/pending_actions/WKF_weekend_fix_pre_open_deploy_2026-07-11.md` (`f83146f`) | pre-open outcomes verified | **Adopted** into §5 |

### MS.2 Why the percentages do not reconcile — and never will
Prior artifacts counted different denominators (132 parity points vs 39 forensic gaps vs 8 P1 system-truth gaps
vs per-box worklists), at different dates, some pre- and some post-Epoch-3-reset. A single blended percentage
would average incommensurable samples across a reset boundary that zeroed the trade record. This cut therefore
computes **no** overall percentage and instead states a per-layer, evidence-cited verdict (§TM) and an explicit
open-debt register (§5).

### MS.3 Bottom-line trust verdict (evidence-based, no percentage)
CHAD is a **paper-posture, broker-connected system whose safety surface strengthened materially since v9.7**
(RTH gate, phantom-guard-on-fill, futures hard-disable, L1-CLD, Bug-A leak class, margin-gate scaffolding) and
whose **money-truth is now CAD-honest at the account level**. It is **not** live-ready: edge is unproven with a
freshly-zeroed sample, position-level guard drift is unreconciled, the margin gate does not yet block, and an
`ALLOW_LIVE` operator-intent value contradicts the paper posture. **No live promotion is authorized (§11).**

---

## TM. L0–L8 TRUST-LAYER MATRIX

| Layer | Verdict (2026-07-12) | Evidence |
|-------|----------------------|----------|
| **L0 — Money truth** | GREEN at account / **AMBER at position** | `positions_truth.truth_ok=true`, `broker_authority_status=GREEN`; `reconciliation_state`=GREEN(0/0/0); equity CAD-native `equity_history.v2` `usd_ok=false`. **BUT** `position_guard_drift.v2` = 5 `qty_mismatch` (guard ≈1.9× broker) + reset equity magnitude unconfirmed (§5.1, §5.4) |
| **L1 — Stability** | STRENGTHENED | L1-CLD connection-owner-loop deadlock fix built + activated (`8472862`…`fe25007`); Bug-A event-loop/fd leak class root-fixed (`ce9fad7`,`710e262`,`275acff`,`87c002f`,`480473e`) |
| **L2 — Execution authority** | STRENGTHENED | RTH gate live (`8b39730`; §5.2); phantom-guard-on-fill + boot reconcile (`0b4c8a9`; §5.1); futures exec hard-blocked at chokepoint (`554a36a` + env mask) |
| **L3 — Risk binding** | PARTIAL | Margin/BP BLOCK gate wired **SHADOW-only** (never blocks: `3a1fc25`,`6f58b39`,`51a5c81`); pre-registered `config/margin_block.json` frozen (`76f127d`); loss-guard + futures-block bind; capital pre-trade block does **not** yet enforce |
| **L4 — Paper record** | RESET / WARMUP | Epoch-3 reset 2026-06-30 (§3); `epoch_state` active `CHAD_v8.9_Paper_Epoch_3`; SCR WARMUP, `effective_trades=0`; evidence-pipeline Wave-1 (`f9454e1`…`3185502`) live |
| **L5 — Observability** | IMPROVED | drift-detector v2 (`d7db150`); soak status-history + RED-window classifier + evidence writers built (gated OFF: `c8a1b01`,`b611645`,`523f5a2`); health rules surface raising rules as ERROR (`d55f6be`) |
| **L6 — Edge** | UNPROVEN (zero-sample) | Edge-validation harness Phases 0–5 built (`e808408`…`09b46c2`); Stage-2 real-fill adapter (`d60c105`…`49cdfc0`); **0 admissible windows**; SCR `effective_trades=0` |
| **L7 — Recovery** | PARTIAL | Boot phantom-reconcile proven (`cleared=3`); L1-CLD forced-reconnect watchdog (`06b30f7`); chaos-drill proof still pre-overhaul/stale (carried from v9.7 RT.7) |
| **L8 — Live readiness** | NOT READY | `live_readiness.ready_for_live=false`; `CHAD_EXECUTION_MODE=paper`; no live code path; `operator_intent=ALLOW_LIVE` contradiction (readiness holds line) |

---

## RT. RED-TEAM FROM FIRST PRINCIPLES — The Strongest Current Case That CHAD Is NOT Done

1. **Zero edge evidence, and the clock was reset.** The 2026-06-30 Epoch-3 reset zeroed the trusted sample; SCR is
   WARMUP with `effective_trades=0`. Every prior in-sample stat (v9.7's sharpe≈3.55, +$27K) is now a retired
   pre-reset artifact. There is no out-of-sample, cost-adjusted edge — the harness that would prove it has 0 admissible windows.
2. **Position-truth divergence.** The guard reports ≈1.9× the broker's size on five symbols (§5.3). Guard-vs-broker
   disagreement on *quantity* is a money-truth-adjacent integrity gap: it is surfaced by drift-v2 but **not fixed** —
   FIFO entry-accumulation with exits never netted. Until reconciled, guard-derived sizing/exposure reads are wrong.
3. **The capital gate does not bind.** The margin/BP BLOCK is shadow-only; no pre-trade buying-power control actually
   rejects an order today. This is scaffolding, not a binding control (L3).
4. **`ALLOW_LIVE` intent under a paper posture.** `operator_intent.json`=`ALLOW_LIVE` (reason `auto_refresh_allow_entries_non_live`)
   contradicts paper posture. The readiness publisher fails-closed and holds `ready_for_live=false`, but a stray
   intent value is exactly the kind of latent contradiction a promotion path must not inherit.
5. **Unexplained equity magnitude.** Post-reset equity ~$1.0M CAD vs pre-reset ~$264K CAD (~3.8×). Plausibly a fresh
   default-funded paper account, but unproven in-artifact (§5.4).
6. **Cross-register gap-number collisions.** At least two gap registers assign overlapping `GAP-0NN` ids to different
   items (the 4002 bind is GAP-020; GAP-013 is the 9618/9619/9620 ports item in *both* registers). Colliding numbering
   across documents is itself a reconciliation hazard (§5.5, §7).
7. **Sockets still wildcard-bound.** IB Gateway `*:4002` (GAP-020) and metrics ports `9618/9619/9620` `0.0.0.0`
   (GAP-013) remain wildcard-bound, contained only by Security-Group + `TrustedIPs`/`ApiOnly` (defense-in-depth, not host-bind).

*Stale prior red-team claims:* v9.7's "L0 BROKEN/$84K" and pre-overhaul chaos-drill proofs remain superseded; do not re-cite.

---

## 0. Preamble and Version Delta

### 0.1 Verified lineage (corrected and pinned)
- **v9.7** — document of record until this cut. File sha256 `5f2b71fd5db203e1bdedbe4cfdcfb2c71707944398b1de9e7c5da69ccee2f53c`, tag `SSOT_v9_7_2026-06-04`. Carried by reference; **not modified.**
- **v9.8** — this file, `docs/CHAD_UNIFIED_SSOT_v9_8_2026-07-12.md`. Full cut. Lock lands as a single-file commit + annotated tag `SSOT_v9_8_2026-07-12` (see §10).
- Prior lineage caveats from v9.7 (v9.4/v9.5 = forward-errata; v9.5 filename drift) are unchanged and carried by reference.

### 0.2 What v9.8 captures (delta over v9.7)
1. **Epoch-3 reset executed 2026-06-30** (§3) — epoch_2 archived; SCR WARMUP; sample zeroed.
2. **Currency truth completed** — CAD-native equity intake (Bucket C, `d5073d5`), report-field relabel (Bucket A, `4671b87`), Kraken native CAD (`82bdfc7`), `usd_ok` fail-closed authoritative USD field (`2b2cde0`), drawdown reads continuous `_cad` (`5b9210c`).
3. **Edge-validation harness Phases 0–5** built (`e808408`…`09b46c2`) + Stage-2 real-fill adapter (`d60c105`…`49cdfc0`).
4. **Margin/Buying-Power BLOCK** Phases A/B + shadow-wired Phase C (`76f127d`…`3c80d40`, `3a1fc25`…`51a5c81`).
5. **L1-CLD cross-loop deadlock fix** built + activated (`8472862`…`fe25007`).
6. **Bug-A leak class** root-fixed (`ce9fad7`…`480473e`).
7. **Weekend-Fix (WKF)** phantom-guard-on-fill + RTH gate + drift-v2, **deployed at the 2026-07-12 pre-open restart** (§5; PA `f83146f`).
8. Registry canonicalization (`ff58803`,`1ff9a62`), tier/snapshot fixes (`0be62a7`,`94eefae`,`32e8cd6`), SCR honest-sample fix (`c18e4ae`), stop_bus latency fix (`26bc033`), GAP-004 nightly-restart fix (`5dc4f58`), EP1/3/4/6/8/2a evidence-pipeline (§6).

### 0.3 What v9.8 is NOT
Not a live authorization. Not an edge claim. Not a reconciliation of the 5 open guard drifts (surfaced, not fixed).
Not a fix for the socket binds or the cross-register gap-number collisions (documented, not closed).

### 0.4 Governance rules (carried verbatim from CLAUDE.md / v9.6 §0.5)
One change at a time; no full rewrites; no direct config mutation (Pending Actions only); verify after every change;
never modify `runtime_FREEZE_*`/`data_FREEZE_*`; no systemd/live-service restarts without explicit GO; commit+tag after each P0/P1/P2.

---

## 1. Mission and Architecture (carried, with v9.8 deltas)
Mission unchanged: compounding, risk-bounded, evidence-gated paper trading desk pending a proven-edge live conversation.
Architecture stack carried from v9.7 §1.2 / v9.6 §1.2 by reference. v9.8 additions to the hot path:
- **RTH gate** (`chad/execution/rth_gate.py`) at the equity/ETF submit chokepoint, sibling to the futures gate and the (shadow) margin gate.
- **Margin/BP gate** (`chad/risk/margin_block.py` + providers) wired at `ibkr_adapter._submit_intent` in SHADOW.
- **Broker connection-owner loop** (`chad/execution/broker_loop.py`) + `broker_executor` semaphore admission (L1-CLD).
- **Drift detector v2** (`chad/core/position_guard.detect_guard_vs_broker_drift_v2`) → `position_guard_drift.v2`.
- **Edge-validation harness** (`chad/validation/*`) — offline, isolated, out-of-band from the hot path.
- **Allocator reality (unchanged from CLAUDE.md):** live overlay is `correlation_overlay`; `CHAD_ALLOCATOR_MODE=V3` is inert (never instantiated).

---

## 2. Current Runtime Truth — 2026-07-12 (point-in-time reads)

### 2.1 Posture / execution mode
`CHAD_EXECUTION_MODE=paper` (live-loop); `dry_run` on some sibling units; `CHAD_KRAKEN_MODE=paper_kraken`;
`CHAD_DISABLE_FUTURES_EXECUTION=1` + `CHAD_FUTURES_EXECUTION_ENABLED=0`. Posture = **PAPER**.

### 2.2 SCR posture (`runtime/scr_state.json` @ 2026-07-12T01:02:02Z, ttl 180)
`state=WARMUP`, `paper_only=true`, `sizing_factor=0.1`, `effective_trades=0`, `paper_trades=2148`, `total_pnl=0.0`,
`win_rate=0.0`, `max_drawdown=0.0`. Reason: "Warmup: 0 effective trades (all trades excluded/untrusted)."
**This is the post-reset warmup state; all prior CONFIDENT/sharpe/PnL stats are retired.**

### 2.3 Live readiness (`runtime/live_readiness.json` @ 2026-07-12T00:56:52Z)
`ready_for_live=false`.

### 2.4 Reconciliation & positions truth
`reconciliation_state.json`: `status=GREEN`, `worst_diff=0.0`, `mismatches=0`, `drifts=0`, `diagnostic_drifts=0`.
`positions_truth.json` @ 2026-07-12T01:02:02Z: `truth_ok=true`, `broker_authority_status=GREEN`,
`truth_source=BROKER_SNAPSHOT_RECONCILED_WITH_LEDGER`, `replay_diagnostic_status=PARTIAL` (`blocks_truth=false`).
`position_guard_drift.json`: `position_guard_drift.v2`, `drift_count=5`, all `qty_mismatch` (see §5.3).
> Note the L0 split-brain: `reconciliation_state` (broker_sync register) is GREEN/0 while `position_guard_drift`
> (guard-vs-broker register) shows 5 — two different registers, both authoritative for their own scope (§5.5).

### 2.5 Equity (`runtime/equity_history.ndjson`, `equity_history.v2`, date_utc 2026-07-11)
`total_equity_cad=1,000,774.61`; `ibkr_equity_cad=1,000,521.76`; `kraken_equity_cad=252.85`; `coinbase_equity_cad=0.0`;
`total_equity_usd=null`; `usd_ok=false`. Equity is CAD-native; the USD leg is retired to null and fails closed.

### 2.6 Epoch state (`runtime/epoch_state.json`)
`active_epoch=CHAD_v8.9_Paper_Epoch_3`, `epoch_started_at_utc=2026-06-30T12:17:42Z`,
`previous_epoch_archive=runtime/archive/epoch_2_pre_20260630T121742Z`, `paper_only=true`, `ready_for_live=false`.

### 2.7 Operator intent (`runtime/operator_intent.json` @ 2026-07-12T00:56:53Z, ttl 900)
`operator_mode=ALLOW_LIVE`, `operator_reason=auto_refresh_allow_entries_non_live`. **Contradicts paper posture**
(carried red-team flag from v9.7; readiness publisher holds `ready_for_live=false` regardless).

### 2.8 Test baseline
`3194 tests collected` (verified this session). Last recorded full-suite run (WKF session, `dbf1c22`):
**3187 passed / 5 failed / 2 skipped**, the 5 fails being the known set (3× `test_tier_manager` CAD-repricing from
the uncommitted currency-audit config, `futures_expiry_gate`, `quarantine_sidecar`). *Not re-run in this session*;
cited as last-known baseline.

### 2.9 Git baseline
HEAD `f83146f` (WKF deployment PA); `main` in sync with `origin/main` at evidence-gather time. v9.8 lock commit lands on top.

---

## 3. The Epoch-3 Reset (executed 2026-06-30)

`epoch_state.json` shows `CHAD_v8.9_Paper_Epoch_3` started `2026-06-30T12:17:42Z`, with epoch_2 archived to
`runtime/archive/epoch_2_pre_20260630T121742Z`. Consequences visible in runtime today:
- Trade sample zeroed → SCR WARMUP, `effective_trades=0`, `paper_trades` re-baselined (4171 pre-reset → 2148).
- All pre-reset performance stats retired (v9.7's CONFIDENT/sharpe≈3.55/+$27K no longer valid status).
- Equity re-baselined to ~$1.0M CAD (§2.5; magnitude flagged §5.4).

The reset was the terminal step of the pre-reset audit stream (memory `master_pre_reset_audit_2026_06_21`,
`epoch_reset_bootstrap_tool`). This cut records the reset **as observed in `epoch_state.json`**; the precise reset
procedure/log is not re-verified line-by-line here (would require the archive manifest — a follow-up if audit-grade
reset provenance is needed).

---

## 4. Major Work Streams Since v9.7 (summary; full commit chain §6)
- **Currency truth (Buckets A/C).** Report-field relabel to `_cad` (`4671b87`); CAD-native intake + tier/withdrawal re-price at 1.4160 (`d5073d5`); Kraken native CAD (`82bdfc7`); authoritative USD + `usd_ok` fail-closed (`2b2cde0`); drawdown continuous `_cad` (`5b9210c`); business_phase/withdrawal authoritative USD (`6841eca`).
- **Edge-validation harness.** Phases 0–5 (`e808408`,`c20ea56`,`2474489`,`8aee51a`,`97c1565`,`09b46c2`) + real-Wilder-ADX regime fix (`769227d`); Stage-2 real-fill adapter (`d60c105`…`49cdfc0`). Offline; 0 admissible windows.
- **Margin/BP BLOCK.** Phase A providers (`51b370d`), Phase B ledger + gate (`fb46fa6`,`3c80d40`), frozen config (`76f127d`), Phase C shadow-wire (`3a1fc25`,`6f58b39`,`40782a6`,`51a5c81`). SHADOW only.
- **L1-CLD cross-loop deadlock.** Owner-loop + watchdog + semaphore + activation (`8472862`…`fe25007`).
- **Bug-A leak class.** SQLite/socket/event-loop fd leaks (`ce9fad7`,`710e262`,`275acff`,`87c002f`,`480473e`).
- **Weekend-Fix (WKF).** `0b4c8a9`,`8b39730`,`d7db150`,`dbf1c22` — deployed at pre-open restart (§5).
- **Reliability/registry/tier/soak.** `ff58803`,`1ff9a62`,`0be62a7`,`94eefae`,`32e8cd6`,`554a36a`,`c18e4ae`,`26bc033`,`5dc4f58`,`d55f6be`,`c8a1b01`,`b611645`,`523f5a2`,`45ef091`,`4c630c4`.
- **Evidence pipeline (EP).** `f9454e1`,`4ccb0f6`,`fc99c06`,`3185502`,`7991b26`,`8b72390`,`3d66339`,`a0f8c06`,`ee3e5f1`,`7383e78`.

---

## 5. OPEN RECONCILIATION DEBT

*This section is the substantive delta of v9.8. It records the verified pre-open (2026-07-12) deployment outcomes
of the Weekend-Fix and the reconciliation debts they surfaced but did not close. Source PA: `ops/pending_actions/WKF_weekend_fix_pre_open_deploy_2026-07-11.md` (`f83146f`).*

### 5.0 Operator addendum — recorded verbatim
> ADDENDUM — PRE-OPEN WINDOW OUTCOMES (2026-07-12 00:14Z, verify in journal/runtime):
> - Phantom reconcile swept THREE ghosts at boot: gamma|BAC, gamma|SPY, gamma|MSFT
>   (GUARD_PHANTOM_RECONCILED markers 00:14:47Z).
> - RTH gate proven live: 6 Saturday equity intents blocked, zero orders created.
> - BindAddress=127.0.0.1 applied but INERT for the API socket (4002 still *:4002);
>   GAP-013 remains OPEN-shielded (SG + TrustedIPs), disposition = manual GUI.
> - Drift v2 settled picture: 5 qty_mismatch rows, guard ≈ 2x broker on
>   TLT/IWM/UNH/LLY/V — FIFO entry-accumulation with exits never netted.
>   Surfaced-not-fixed; requires its own reconciliation PA. Record verbatim in §5.

### 5.1 Verification against journal/runtime (2026-07-12) — all CONFIRMED
| # | Claim | Evidence | Verdict |
|---|-------|----------|---------|
| 1 | Phantom sweep of gamma\|BAC, gamma\|SPY, gamma\|MSFT at 00:14:47Z | journald: 3× `GUARD_PHANTOM_RECONCILED … reason=no_broker_position_no_confirmed_fill` @ 00:14:47.380Z + `BOOT_PHANTOM_RECONCILE cleared=3 keys=['gamma\|BAC','gamma\|MSFT','gamma\|SPY']` @ 00:14:47.381Z | **CONFIRMED** (exact set) |
| 2 | RTH gate blocked Saturday equity intents, zero orders created | journald `RTH_GATE_BLOCK … status=market_closed` each paired with `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS` — no order, no idempotency row (see §5.2) | **CONFIRMED** |
| 3 | 4002 BindAddress applied but INERT; socket still `*:4002` | `config.ini` has `BindAddress=127.0.0.1` **and** `TrustedIPs=127.0.0.1`; `ss -tulpn` still shows `*:4002` (java pid) | **CONFIRMED** (id correction §5.5) |
| 4 | 5 `qty_mismatch` rows, guard ≈2× broker on TLT/IWM/UNH/LLY/V | `position_guard_drift.v2`, `drift_count=5`, all `qty_mismatch` (per-symbol ratios §5.3) | **CONFIRMED** |

### 5.2 RTH_GATE_BLOCK is per-cycle recurring — the invariant is zero-orders/zero-rows (correction)
The operator's "6 … blocked" is a point-in-time count. `RTH_GATE_BLOCK` **recurs every live-loop cycle while the
market is closed** — the strategies keep re-proposing on a Saturday, so the block fires each cycle (15 blocks in the
00:14–00:20Z window across the first cycles: BAC/gamma, MSFT/beta, SPY/gamma, then repeats). The count is not fixed
at 6; it grows with cycles. **The invariant that holds on every single block is: zero orders created and zero
idempotency rows written** (`status=market_closed` → `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS` →
`COOLDOWN_NOT_REARMED_UNCONFIRMED_STATUS`). Equity/ETF are gated; futures/crypto are exempt; `CHAD_RTH_GATE` DEFAULT ON.

### 5.3 Guard-vs-broker drift — 5 `qty_mismatch` rows, per-symbol ratios as measured (SURFACED, NOT FIXED)
| Symbol | Guard qty | Broker qty | Guard/Broker ratio |
|--------|-----------|-----------|--------------------|
| IWM | 392 | 200 | **1.96×** |
| LLY | 250 | 130 | **1.92×** |
| TLT | 780 | 420 | **1.86×** |
| UNH | 399 | 207 | **1.93×** |
| V   | 301 | 154 | **1.95×** |

Root pattern: **FIFO entry-accumulation with exits never netted** — the guard books entries but the corresponding
exits are not netted against them, so guard totals drift toward ≈2× the broker's true position. `drift_kind=qty_mismatch`
for all five; `phantom_guard_entry=0`, `broker_untracked_position=0`. **This is observability-only: the drift detector
surfaces the divergence; it does not mutate the guard.** Remediation requires its **own reconciliation PA** — candidate
paths: `scripts/close_guard_entry.py` under the GAP-028 permissive policy, or a netting rebuild. **Do not silently
mutate the guard**; this is a reconciliation decision, not a code cleanup.

### 5.4 Post-reset equity magnitude — unconfirmed (~3.8× jump)
Post-reset `total_equity_cad`≈$1,000,775 (2026-07-11) vs pre-reset ~$264K CAD (forensic 2026-06-11). Consistent with
a fresh/default-funded IBKR paper account created at the Epoch-3 reset, but **not independently proven** in the
artifacts read. Operator-confirm the funding basis, or raise a PA to reconcile the reset equity provenance.

### 5.5 Cross-register gap-number collision — the 4002 bind is GAP-020, not GAP-013 (correction + hazard line)
The addendum cited **GAP-013** for the 4002 IB Gateway socket. That id is **wrong across both registers**, and the
collision is itself a reconciliation hazard worth stating explicitly:

- **Register A — `docs/CHAD_GAPS_TO_CLOSE.md` / `ops/pending_actions/` (ops-PA numbering).** The 4002 IB Gateway
  `BindAddress` maintenance item is **GAP-020** (`ops/pending_actions/GAP-020_ibgateway_bindaddress_maintenance.md`).
  In this same register, **GAP-013 = ports 9618/9619/9620 bound to 0.0.0.0** (metrics/telegram/legend internal ports;
  `CHAD_GAPS_TO_CLOSE.md` line 91).
- **Register B — the 42-gap parity/forensic audit numbering.** Its **GAP-013 is also the 9618/9619/9620 ports item**
  (`reports/parity_audit/PARITY_AUDIT_20260521T232401Z.md`), **not** the 4002 socket.

So: the 4002 socket's canonical id is **GAP-020**; `GAP-013` denotes the *different* 9618/9619/9620 ports gap in both
registers. Both items are the same *shape* (wildcard-bound socket, contained by Security-Group + `TrustedIPs`/`ApiOnly`,
disposition = manual IB Gateway GUI / SG narrowing), which is exactly why they are easy to conflate. **Hazard:**
multiple documents assign overlapping `GAP-0NN` numbers to different items; any future gap-close bookkeeping must name
the register alongside the number, or closures will be mis-attributed. Both GAP-020 (4002) and GAP-013 (9618-20) remain
**OPEN-shielded**, not emergencies.

### 5.6 Split-brain reconciliation registers (L0)
`reconciliation_state.json` (broker_sync scope) reports GREEN/0 at the same instant `position_guard_drift.json`
(guard-vs-broker scope) reports 5. Each is authoritative for its own scope and both are correct; the divergence is a
*naming/scope* hazard for readers who treat "reconciliation" as one thing. Any live-readiness reasoning must consult
**both** registers (the PR-09 contract already routes `drift_count>0` as a fail-closed drift gate independent of the
broker_sync GREEN).

---

## 6. Commit Chain Since v9.7 (`SSOT_v9_7_2026-06-04..HEAD` = 86 commits; v9.8 lock lands on top)

*Every asserted delta above carries a hash. Full enumeration, newest→oldest:*

```
f83146f  WKF deployment outcomes PA 2026-07-12   ← this cut's WKF PA commit
dbf1c22  WKF U2-HF: conftest defaults CHAD_RTH_GATE=0
d7db150  WKF U3: drift semantics v2 (like-with-like totals + atomic snapshot + drift_kind)
8b39730  WKF U2: market-hours (RTH) gate on equity/ETF submit path
0b4c8a9  WKF U1: position_guard books on CONFIRMED FILL only + boot phantom reconcile
51a5c81  G3C-HF: kill test-write leak into data/margin_shadow + repo-wide write guard
40782a6  G3C U3: PA doc + review hardening
6f58b39  G3C U2: integration test — full submit path, shadow all-submit + enforce blocks
3a1fc25  G3C U1: wire margin/BP gate into IBKR chokepoint in SHADOW mode
49cdfc0  G4 U4: review hardening — Stage-2 adapter + CLI
06b431f  G4 U3: drop trade_log_adapter re-export from validation __init__
76d58f1  G4 U2: wire Stage-2 trade-log into the CLI + end-to-end test
d60c105  G4 U1: Stage-2 trade-log adapter — fail-closed trust gate + manifest + tests
fe25007  L1-CLD U7: activation — live_loop hands execution-connection ownership to broker owner loop
36ed7bc  L1-CLD U6: full-suite verification record + rollback range
a28e237  L1-CLD U5: cross-loop deadlock recovery integration test
2f431a3  L1-CLD U4: broker_executor semaphore admission gate
f7b1a3f  L1-CLD U3: ibkr_adapter routes broker calls through connection-owner loop
06b30f7  L1-CLD U2: reader-progress watchdog + forced reconnect + BROKER_READER_STALLED
725ed71  L1-CLD U1: broker_loop.py owner-loop thread + submit_coro + fail-closed
8472862  L1-CLD U0: PA doc — cross-loop deadlock autopsy + connection-owner design
3c80d40  margin-block: Phase B2 ALLOW/BLOCK gate
fb46fa6  margin-block: Phase B1 pending-exposure ledger
51b370d  margin-block: Phase A buying-power providers
76f127d  margin-block: freeze pre-registered config/margin_block.json
150c3bd  gitignore planning-with-files + recon plan files
1a9ac4b  docs(margin-block): v2.2 design as build SSOT
09b46c2  edge-harness: Phase 5 capstone — lockbox + freeze + verdict + report + CLI
97c1565  edge-harness: Phase 4 feature-parity gate + no-lookahead backtest engine
8aee51a  edge-harness: Phase 3 significance + ruin + pre-registered minimums
2474489  edge-harness: Phase 2 cost model + splits + regime labeler
c20ea56  edge-harness: Phase 1 scoring spine + isolation test
e808408  edge-harness: Phase 0 bar-corpus forensic auditor
ebab068  docs(edge-harness): v1.1 design as build SSOT
769227d  regime: real Wilder ADX (unblocks false trending_bear)
813dc12  chore(claude): add local-review skill + order-guard hook
f38372c  constants: add missing chad/constants/__init__.py
82bdfc7  kraken: value Kraken equity natively in CAD; retire USD round-trip
523f5a2  soak: GAP-039 clean-soak evaluator (SCR-authoritative clean_soak_state.json)
4c630c4  broker-authority: proven-flat GREEN path + cash passthrough into positions_snapshot
d5073d5  currency: C-ii re-price tier+withdrawal CAD-native @1.4160; equity intake live CAD (Bucket C)
26bc033  stop_bus: broker-latency trigger reads api_ms not connect-polluted latency_ms
45ef091  ibgateway: Paper Account Notice auto-dismiss watcher + PA + tests
db6a68c  ops: epoch_reset_bootstrap dry-run reports true would-preserve bytes
4ffd99b  ops: epoch_reset_bootstrap — guarded dry-run-first clean-reset (inert)
b611645  soak: RED-window mechanical classifier (inert CLI, rule sha 800480a5)
4671b87  currency: relabel report-only *_usd → *_cad + CAD dashboard (Bucket A)
c8a1b01  soak: status-history + companion evidence writers (gated OFF)
23eb5a5  docs(soak): spec status-history writer
366d3a8  docs(soak): canonicalize BOX-059 soak policy
480473e  bug-a: mark PA L1 IMPLEMENTED/LANDED; reconcile scope to L-01..L-05
87c002f  bug-a/L-05: cache backend ai_surface GPTClient as process singleton
275acff  bug-a/L-01,L-02: shared bounded broker executor (event-loop leak on timeout)
710e262  bug-a/L-04: close per-call SQLite conn in kraken_executor ExecStateStore
ce9fad7  bug-a/L-03: close per-call SQLite conn in _SQLiteIdempotencyStore
c18e4ae  scr: exclude CME-micro futures from confidence sample (CONFIDENT→CAUTIOUS)
d55f6be  health: run_all_rules surfaces raising rules as ERROR findings
6841eca  equity: business_phase + withdrawal read authoritative USD; kill CAD-in-_usd
69dc388  docs(defense): P0-P3 defense-board reconciliation (2026-06-17)
1ff9a62  registry: close P1-1 — validate per-tier enabled_strategies lists
ff58803  registry: single canonical strategy registry (18 declared/16 active/2 dormant)
32e8cd6  tier: pass real US-equity market_open into main() (mid-session demotions)
94eefae  snapshot: preserve unknown keys across portfolio_snapshot writers (USD block)
0be62a7  tier: select tier from authoritative USD equity, fail-closed
5b9210c  equity: relabel equity-history to _cad + forward-only USD; drawdown continuous _cad
2b2cde0  equity: authoritative USD equity field + usd_ok (fail-closed, never CAD fallback)
554a36a  futures-gate: hard-block all futures execution at submit chokepoint when disabled
c1cd533  ops: flatten script resolves futures exchange before submit (Warning 321)
480b1d8  ops: one-shot futures flatten script (live-read, preview+confirm)
8b72390  PA-EP2a: honest ref-price submit stamp (no NBBO; fail-open)
3d66339  PA-EP4: append-only 1m bar archive (UTC-normalized, dedup, retention)
ee3e5f1  PA-EP6b: co-writer ts_utc → true UTC + test
a0f8c06  PA-EP6: ts_utc → true UTC (instant-preserving) at bar writer + test
5dc4f58  Weekend maint: fix GAP-004 nightly-restart self-dependency + alert OnFailure
7383e78  S1: CLAUDE.md verified-fact corrections + Evidence Pipeline Wave-1 record
7991b26  docs(PA): PA-EP7v2 spec — root-kill $100 placeholder emission
3185502  PA-EP8: status canonicalization at evidence chokepoint (status_raw provenance)
fc99c06  PA-EP7-T: containment tripwire for placeholder records
4ccb0f6  PA-EP1: modeled IBKR Fixed commissions at evidence chokepoint (forward-only)
f9454e1  PA-EP3: thread intent idempotency_key into evidence execution_id (slippage join)
1f54927  test: hermetic alpha_intraday rvol/setup (decouple from live news_intel)
3c5f45c  docs(PA): refine L1 Bug A PA with audit deltas
37f0e6b  docs(PA): PR-05 line-84 patch note (APPLIED in 5961be2) — closes v9.7 §8.1(a)
8b549e2  docs(PA): close §8(b) observation window — closes v9.7 §8.1(d)
0bc47bc  docs(PA): BUGB Fix-A restart PA EXECUTED — closes v9.7 §8.1(b)
0af767a  ops(systemd): capture live-loop drop-ins 90-fd-limit + 91-disable-futures-exec — closes v9.7 §8.1(c)
```

*(86 commits; v9.7's four §8.1 operator items were all closed pre-reset by `37f0e6b`/`8b549e2`/`0bc47bc`/`0af767a`.)*

---

## 7. Gap-Register Mapping (42-gap forensic audit + July-2 list)

> **Provenance caveat (evidence-first).** No single literal artifact named "42-gap audit" or "July-2 list" was
> locatable on disk in this session. The nearest documented registers are: the **2026-06-11 forensic full-system
> audit** (memory `forensic-audit-2026-06-11`: **39 gaps** — 5 P1 / 15 P2 / 19 P3; the "42" likely counts the
> subsequent P1-reconciliation and post-reset additions), the **2026-06-16 P1 gap reconciliation** (8 P1 system-truth
> gaps), and the **post-2026-07-02 work** (Kraken CAD-native `82bdfc7`, GAP-039 soak evaluator `523f5a2`, constants
> init `f38372c`). The mapping below is reconstructed from those registers; where a literal row could not be
> confirmed, it is marked *(register unconfirmed)* rather than fabricated.

| Gap (forensic register) | Subject | Current disposition (evidence) |
|-------------------------|---------|--------------------------------|
| GAP-001 | CAD-in-`_usd` mislabel (drawdown/equity) | **CLOSED** — Bucket A/C relabel + authoritative USD/`usd_ok` (`4671b87`,`5b9210c`,`2b2cde0`,`6841eca`) |
| GAP-002 | −15.7% drawdown breach, enforce off | **RE-BASELINED** by Epoch-3 reset; drawdown reads continuous `_cad`; re-verify against fresh sample |
| GAP-003/006/007 | Silent VaR/Kelly publishers + failed alert handler | Health rules now surface raising rules as ERROR (`d55f6be`); VaR/Kelly publisher cadence — *(re-verify post-reset)* |
| GAP-004 | Nightly-restart self-dependency + alert OnFailure | **CLOSED** (`5dc4f58`) |
| GAP-005 | SSH world-open / SG | **PARTIAL** — per `master_pre_reset_audit` re-probe, SSH SG = 3 /32s (not world-open); 80/443/8765 posture unchanged |
| GAP-013 | Ports 9618/9619/9620 bound 0.0.0.0 | **OPEN-shielded** (SG + bind unchanged) — *distinct from the 4002 bind (§5.5)* |
| GAP-020 | IB Gateway 4002 `BindAddress` | **OPEN-shielded** — BindAddress set but socket still `*:4002`; SG + TrustedIPs + ApiOnly contain (§5.1, §5.5) |
| GAP-039 | Clean-soak evaluator | **BUILT** (`523f5a2`) — SCR-authoritative `clean_soak_state.json` |
| P1-1 registry | strategy-registry drift | **CLOSED** — canonical registry 18/16/2 + startup assertion (`ff58803`,`1ff9a62`) |
| P1-2 ib_async | ib_insync residue | **CLOSED** (migration complete; pinned by parity tests) |
| P1-3/4 drawdown/watchdog | fresh reads | **CLOSED** (`5b9210c`, snapshot writers `94eefae`) |
| P1-5 equity 3-source disagree | usdcad phantom drawdown | **ADDRESSED** by CAD-native intake; §5.4 magnitude still open |
| P1-8 sqlite lifecycle stale rows | orphan PendingSubmit | **PARTIAL** — live reqAllOpenOrders probe=0; stale local DB rows only (per `master_pre_reset_audit`) |
| H2/H3/H4/H5 (pre-reset must-resolve) | idem-DB sync / phantom HWM / epoch honored / equities-flatten | Reset executed 2026-06-30 (§3); confirm each was honored by the reset run — *(reset-provenance follow-up)* |
| July-2 list items | Kraken CAD-native / constants init / GAP-039 | **CLOSED/BUILT** (`82bdfc7`,`f38372c`,`523f5a2`) |

Any gap not enumerable from a confirmed register is deliberately **omitted rather than invented**; a follow-up
PA should reconcile the "42" count against a single canonical gap register (the collision hazard of §5.5 is why this matters).

---

## 8. Pending Actions Register (open, post-cut)
1. **Drift-v2 reconciliation PA (NEW, from §5.3)** — net the 5 `qty_mismatch` guard entries (guard ≈1.9× broker). Do not silently mutate the guard.
2. **Reset equity provenance (§5.4)** — confirm the ~$1.0M CAD funding basis.
3. **Margin/BP gate enforce-flip** — Phase C is shadow-only; enforce requires account-summary publisher + persistent ledger + burst protection (currently inert) + operator GO.
4. **GAP-020 (4002) + GAP-013 (9618-20) socket binds** — manual IB Gateway GUI / SG narrowing (Channel 3).
5. **operator_intent=ALLOW_LIVE (§2.7)** — reconcile the intent value with the paper posture.
6. **Gap-register unification (§7)** — single canonical register to end the `GAP-0NN` cross-document collision.
7. **Carried:** PA-EP7v2 (root-kill $100 placeholder), PA-EP8v2 (replay netting rebuild), chaos-drill re-drill (stale proof).

---

## 9. Live Readiness (unchanged)
`ready_for_live=false`. No live code path exists. Live promotion remains gated by, at minimum: proven out-of-sample
cost-adjusted edge (Q3 — currently zero-sample), a binding capital control (margin gate is shadow-only), position-guard
drift reconciliation (§5.3), and an explicit operator GO. None are satisfied.

---

## 10. Definition of Done (this cut)
- [x] Full cut, evidence-first; supersession banner; prior scorecards/audits retired as status sources (§MS).
- [x] Three-questions section present; edge recorded UNPROVEN / zero-sample (Q3, TM.6).
- [x] Master scorecard (§MS): prior artifacts inventoried; non-reconciliation explained; **no overall percentage**.
- [x] L0–L8 trust matrix (§TM), every verdict evidence- or hash-cited.
- [x] Red-team from first principles (§RT).
- [x] §5 OPEN RECONCILIATION DEBT: WKF addendum verbatim + verification table; RTH per-cycle/zero-orders correction; 1.9× per-symbol ratios; GAP-020-vs-GAP-013 register disambiguation; split-brain registers.
- [x] All 86 commits since v9.7 enumerated with hashes (§6).
- [x] Gap-register mapping (§7) with provenance caveat; nothing fabricated.
- [x] v9.7 sha256 pinned; v9.7 file unmodified.
- [x] No-live confirmation (§11).
- [ ] Lock: single-file commit + annotated tag `SSOT_v9_8_2026-07-12` + push (lands with this commit; push denials → pending operator actions).

---

## 11. No-Live Confirmation
This document does not authorize live trading. `ready_for_live` remains `false`; `CHAD_EXECUTION_MODE=paper`.
No broker orders may be placed or cancelled under this document. The margin-gate enforce-flip, the drift-v2
reconciliation, the socket binds, the `ALLOW_LIVE` intent reconciliation, and any live conversation each require
their own explicit operator GO. Edge is UNPROVEN with a freshly-zeroed sample; nothing in this cut constitutes edge evidence.

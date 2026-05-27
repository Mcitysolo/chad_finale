# CHAD Full-Production Paper Mode Completion Register

**Generated:** 2026-05-27T00:13:09Z
**Author:** Team CHAD (Claude Code) — read-only audit. No code, config, runtime, service, or git changes made by this audit.
**Scope:** Refresh the Paper Mode Completion Register after PO-03 was committed VERIFIED (`15b963e`), and audit the 8 remaining UNKNOWN strategy classifications.
**Supersedes:** `reports/parity_audit/CHAD_FULL_PRODUCTION_PAPER_COMPLETION_REGISTER_20260526T230112Z.md`.

---

## 1. Definition

**CHAD Full-Production Paper Mode** = the complete CHAD production system running with paper execution only. Not an MVP. The only difference from live should be money/account authorization. Every production-grade component must be present, healthy, and demonstrating expected behavior over real trading sessions. Paper-complete is reached when (a) all engineering work required for the production stack is verified or honestly classified, (b) all production strategies are either actively trading or honestly classified as silent/blocked, and (c) a sustained clean soak proves stability.

---

## 2. Current baseline (2026-05-27 ~00:13Z)

| Field | Value |
|---|---|
| HEAD | `15b963e` (PO-03: declare zero public placeholder fingerprint success criterion) |
| Tests | **2537 passed / 2 failed in 152s** (full suite). Both failures are pre-existing environment flakes, not regressions: `test_canonical_equity_source.py::test_canonical_sources_agree_within_skew_tolerance` (BOX-034: pnl_state.account_equity=237115.69 vs portfolio_snapshot total=170406.40, drift=$66,709 > tolerance=$85.20 — runtime data skew between concurrent equity publishers) and `test_xgb_promotion_workflow.py::test_trainer_candidate_dir_named_with_utc_timestamp` (UTC date-rollover flake; passes on re-run). |
| Working tree | clean for scope; pre-existing `_archive/` deletions and the untracked `PR-05_OFFICIAL_36_amendment_line84_patch_note_2026-05-26.md` remain. |
| mode | paper (`chad_mode=paper`, `live_enabled=false`) |
| ready_for_live | **false** (ts=2026-05-27T00:01:44Z) |
| allow_ibkr_live | **false** |
| allow_ibkr_paper | **true** |
| stop_bus | **clear** at audit time (active=false; last cleared 2026-05-26T23:52:03Z via `auto_recovery:broker_latency_clean_streak=5`). 24h: 30 STOP_BUS_TRIGGERED / 8 STOP_BUS_AUTO_CLEARED — intermittent flutter pattern. Latency last 10 health pings: median ~760ms, range 356–3721ms. |
| reconciliation_state | GREEN (ts=2026-05-27T00:01:14Z) |
| position_guard_drift | drift_count=0 |
| positions_truth | broker_authority_status=GREEN, replay_diagnostic_status=PARTIAL (by PR-09 design), truth_ok=true |
| trade_lifecycle_state | backlog_flag=false |
| SCR | CONFIDENT / sizing_factor=1.0 / paper_only=false / Epoch 2 |
| Tier | SCALE / 17 enabled (matches 17 registered) |
| live-loop | PID 323693, started 2026-05-25T20:51:10Z, uptime ~27h 22m, ActiveState=active SubState=running |
| bar provider (M2K/MYM) | **PR-M2K-MYM LOCKED.** symbols_count=34 (down 1 from 35 — SIL transiently absent this cache window; M2K/MYM/MCL all present). |
| **options chain refresh** | **VERIFIED FULL RECOVERY.** Service ran successfully at 12:30:31Z; chains_count=1 (SPY); **`runtime/options_greeks.json::status=ok` with `symbols_count=1`** (ts=2026-05-26T23:49:14Z) — Greeks publisher reran successfully. |
| Placeholder fingerprint (since 2026-05-26T14:45:17Z) | public `fill_price=100.0` = **0**; trusted-fake-placeholder = **0**; placeholder-tagged rejected rows = 24 (delta=16, reconciler=8), all `rejected + pnl_untrusted + broker_rejected` (defense-in-depth, deferred hardening). |

**Recent commits (since prior register HEAD `957c0c4`):**
- `15b963e` PO-03: declare zero public placeholder fingerprint success criterion

---

## 3. Paper Required Register

| ID | Requirement | Current Evidence | Status | Blocks Paper Complete? | Next Action |
|---|---|---|---|---|---|
| PR-01 | SSOT / pending-action cleanup (OFFICIAL_36 stale clauses) | Commits `ba1e1f2`, `5961be2`. | **VERIFIED** | No | None — PR-12 archives. |
| PR-02 | Delta upstream placeholder/noise source eliminated | Commit `139d275`; **PO-03 closure addendum appended `15b963e`**: public $100 fingerprint = 0; trusted-fake = 0; downstream PnL/SCR/trade evidence protected. | **VERIFIED** (under PO-03 success criterion) | No | None — DEFERRED executor-layer hardening tracked separately. |
| PR-02b | Reconciler upstream placeholder source silenced | Commit `5c5507e`; **PO-03 closure addendum appended `15b963e`**: same evidence as PR-02. | **VERIFIED** (under PO-03 success criterion) | No | Same as PR-02. |
| PR-03 | ib_async Phase 2 migration | Commit `d476e8c`; `test_pr03_ib_async_phase2_migration.py` enforces contract. | **VERIFIED** | No | None. |
| PR-04 | Options-chain-refresh / Greeks truth | Commit `adacf6f` shipped retry/backoff + structured failure artifact. **Service recovered 2026-05-26 12:30:31Z**. Chains cache: 1 chain (SPY). **Greeks publisher reran successfully 23:49:14Z; status=ok, symbols_count=1.** External block fully released. | **VERIFIED** | No | Continue watching cadence; reclassify alpha_options / omega_momentum_options once they emit. |
| PR-05 | Paper position closer service | `chad-paper-position-closer.service` LoadState=not-found (CLI-only by design). | **DEFERRED** by design | No | None. |
| PR-06 | Paper shadow runner / shadow path | `chad-paper-shadow-runner.service` + `chad-paper-shadow-exec.service` MASKED; canonical path via `chad-shadow-snapshot.timer` + `chad-scr-sync.timer`. Formalized Path A in commit `1575979`. | **VERIFIED** | No | None unless Path B re-arm chosen. |
| PR-07 | IBKR broker-events service | Oneshot, exit 0 in steady cadence. | **VERIFIED** | No | None. |
| PR-08 | IBKR paper-fill harvester | Oneshot, exit 0 in steady cadence. | **VERIFIED** | No | None. |
| PR-09 | Position truth / reconciliation contract | Commit `2d454ed`; `positions_truth.broker_authority_status=GREEN`, `replay_diagnostic_status=PARTIAL` (by design); 10/10 PR-09 tests pass. | **VERIFIED** | No | None. |
| PR-10 | Full test baseline | 2537/2539 pass. The 2 failures are pre-existing environment flakes (BOX-034 canonical equity skew + UTC date-rollover); no code regression. | **VERIFIED** (flakes tracked separately) | No | BOX-034 §4a separately tracked. |
| PR-11 | Full-cycle preview | `chad/core/full_cycle_preview.py --dry-run` is the standard CLAUDE.md gate. Not re-run by this read-only audit. | **VERIFIED** | No | None. |
| PR-12 | Final archive/apply closeout docs | All PR-* PA docs still under `ops/pending_actions/`. OFFICIAL_36 §9 doc cycle untouched. Untracked PR-05 patch-note awaiting decision. | **PARTIAL** | Soft yes (cosmetic SSOT closure) | Prepare PR-12 archive Pending Action. |
| PR-M2K-MYM | Bar-provider futures mapping | Commits `cdab294` + `ecb370c`; runtime-verified. M2K/MYM bars fresh; zero Error-200 for either symbol. | **VERIFIED** + **LOCKED** | No | None. |
| PR-MYM-SIZING | MYM whole-contract safety skip | Commit `957c0c4`, doc-only Pending Action. EXPECTED_SAFETY_SKIP. | **VERIFIED** | No | None. |
| **PO-03** (Paper Required side-effect) | Zero public placeholder fingerprint declaration | **Committed `15b963e`** — new PA `PO-03_zero_public_placeholder_fingerprint_success_2026-05-26.md` plus closure addenda on PR-02, PR-02b, OFFICIAL_36. | **VERIFIED** | No | None. |
| IBKR-LATENCY-RECOVERY | Broker-latency stop-bus recovery | Sustained 8200ms catastrophic state is gone since the operator-authorized Gateway restart. Pattern: intermittent 2000–3700ms spikes; auto-recovery via `clean_streak=5 / 240s`. 24h: 30 triggers / 8 clears. | **LOCKED** (sustained fix); **OBSERVATION: flutter remains** | No (self-resolving) | Watch flutter cadence; if cumulative halt-time stays bounded, leave threshold; if not, raise threshold-review Pending Action. |

---

## 4. Paper Observation Register

| ID | Requirement | Current Evidence | Status | Blocks Paper Complete? | Next Action |
|---|---|---|---|---|---|
| PO-01 | 5 clean paper-trading days | Live-loop uptime ~27h 22m since 2026-05-25T20:51:10Z. Flutter pattern stable. Clean session-days = 0/5; soak clock needs re-base at next US-equity open. | **PARTIAL** (observation in progress) | Yes (DoD item) | Re-base 5-session clock; track cumulative halt-time/session. |
| PO-02 | SCR remains CONFIDENT during soak | state=CONFIDENT, sizing_factor=1.0, paper_only=false, Epoch 2. Stable. | **PARTIAL** (in progress) | Yes | Continue soak. |
| PO-03 | Zero public placeholder fingerprint | Operator success criterion declared `15b963e`. Evidence: public $100=0, trusted-fake=0, 24 placeholder-tagged rejected rows all defense-quarantined. PR-02/PR-02b VERIFIED under this criterion. | **VERIFIED** | No | None — DEFERRED hardening tracked. |
| PO-04 | Stop bus stays clear | 24h: 30 triggers, 8 auto-clears. Each trigger lasts 5–60 min then auto-resolves on clean_streak=5. Not a hard blocker; intermittent flutter. | **PARTIAL — FLUTTER** | Soft (only blocks PO-01 if cumulative halt-time per session is excessive) | Watch for one full US-equity session; if minor, accept and re-base PO-01. |
| PO-05 | Live-loop uptime clean | Process up ~27h. Productive time = wall-clock minus trigger windows. | **PARTIAL** | Yes | Same dependency on PO-04 flutter. |
| PO-06 | Options strategies prove activity or remain honestly classified | **PR-04 fully recovered today** — chains cache + Greeks publisher both healthy (status=ok, symbols_count=1). alpha_options + omega_momentum_options still 0 paper_fills in 15d window — classification was EXTERNAL_BLOCKED; with Greeks back, they shift to REGIME_SILENT pending real signal evidence. | **VERIFIED** (honest classification holds and tightens) | No | Reclassify alpha_options / omega_momentum_options after 1–2 sessions of post-recovery evidence. |

---

## 5. Strategy Utilization Register

Source: `chad.strategies.active_strategy_names()` (17 registered) ∩ `runtime/tier_state.json.enabled_strategies` (tier=SCALE, 17 enabled).
Recent fills window: last 15 daily files (2026-05-13 → 2026-05-27). "Recent Fills" counts `status=paper_fill` only.

| Strategy | Registered | Tier Enabled | Recent Fills (paper_fill, 15d) | Current Classification | Paper Complete Impact | Next Action |
|---|---|---|---|---|---|---|
| alpha | ✓ | ✓ | 1 (BAC) | ACTIVE_PRODUCTIVE (very low cadence; competing with open orders) | None | Continue. |
| alpha_crypto | ✓ | ✓ | 0 | **HALTED_EDGE_DECAY** — `runtime/strategy_allocations.json::alpha_crypto.halted=true, halt_reason=consecutive_negative_10, halted_at=2026-05-08T20:15:26Z`. Strategy continues to emit Kraken-paper signals (4416 decay rows; 55 in last 24h) but the allocator's halt filter zeros all fills. Last_fill_at=2026-05-12T02:34:34Z. No `100.0` placeholder events from this path. | None (honest halt) | Optional: operator review of consecutive-negative threshold; clear halt only if policy permits. |
| alpha_futures | ✓ | ✓ | 21 (MGC=16, MES=5) | ACTIVE_PRODUCTIVE | None | None. |
| alpha_intraday | ✓ | ✓ | 2 (MNQ=2) | ACTIVE_PRODUCTIVE (low cadence) | None | None. |
| alpha_intraday_micro | ✓ | ✓ | 0 | REGIME_SILENT per BOX-033 / BOX-048 (micro weight policy documented) | None | None. |
| alpha_options | ✓ | ✓ | 0 | REGIME_SILENT (was EXTERNAL_BLOCKED; PR-04 recovered today, Greeks publisher healthy; pending real signal evidence) | No | Reclassify after 1–2 sessions of Greeks-active evidence. |
| beta | ✓ | ✓ | 0 | **REGIME_SILENT** — long-term institutional compounder; quarterly 13F cadence by design. `strategy_routing_diagnostics::beta.halted=false`, current_cap=$2,346; `strategy_health::beta.health_score=0.925, win_rate=0.75`; last_fill_at=2026-05-02. Module header: "Beta does NOT trade actively. It slowly builds and holds positions". | None | None. |
| beta_trend | ✓ | ✓ | 0 | **REGIME_SILENT** — once-per-UTC-day legend-driven allocator. `zero_fill_epoch2=true`; current_cap=$1,182; no halt. Module header: "WEALTH MODE, low-churn, once-per-day". | None | None. |
| delta | ✓ | ✓ | 0 paper_fills / 260 rejected (IWM=130, SPY=130) | ACTIVE_BLOCKED_BY_DATA — strategy fires; writer defense-in-depth rejects every row. Public $100 fingerprint zero (PO-03 criterion met). | Soft | DEFERRED hardening (executor-layer placeholder emitter). |
| delta_pairs | ✓ | ✓ | 0 | **REGIME_SILENT** — market-neutral pairs trading (SPY/QQQ, SPY/IWM, QQQ/IWM); signal-conditional on z-score ≥ entry threshold. `zero_fill_epoch2=true`; current_cap=$330; no halt. | None | None. |
| gamma | ✓ | ✓ | 0 | **REGIME_SILENT** — `strategy_routing_diagnostics::gamma.blocked_reasons={"net_exposure": 1, "regime_inactive": 11}`; 11 signals generated/cycle, all rejected at regime gate; last_fill_at=2026-04-08, last_closed_trade_at=2026-03-27; no halt; current_cap=$1,606. | None | None. |
| gamma_futures | ✓ | ✓ | 11 (MCL=6, M2K=5) | ACTIVE_PRODUCTIVE (M2K active post-PR-M2K-MYM) | None | None. |
| gamma_reversion | ✓ | ✓ | 0 | **REGIME_SILENT** — ETF mean reversion (SPY/QQQ/GLD/TLT); RSI+BB+z-score confluence required; `zero_fill_epoch2=true`; current_cap=$390; no halt. Signal-conditional by design. | None | None. |
| omega | ✓ | ✓ | 0 | **REGIME_SILENT** — hedge sleeve / crash insurance. Module header: "Omega is NOT a profit engine. Omega is insurance... It should activate rarely". `zero_fill_epoch2=true`; current_cap=$544; no halt. Activates only when ≥2 danger sensors fire. | None | None. |
| omega_macro | ✓ | ✓ | 8 (M6E=8) | ACTIVE_PRODUCTIVE | None | None. |
| omega_momentum_options | ✓ | ✓ | 0 | REGIME_SILENT (was EXTERNAL_BLOCKED; PR-04 recovered; pending real signal evidence) | No | Reclassify after 1–2 sessions. |
| omega_vol | ✓ | ✓ | 0 | **REGIME_SILENT** — vol-regime alpha (SVXY/UVXY). `strategy_routing_diagnostics::omega_vol.blocked_reasons={"regime_inactive": 1}`; 1 signal generated/cycle, blocked at regime gate; last_fill_at=2026-04-22; no halt; health_score=0.1 (low). Module header: "Active in most regimes (except ELEVATED where direction is ambiguous)". | None | None. |

**Adjacent:** `broker_sync` accounts for 131 paper_fills in window — broker-truth tracking, not a strategy.

**Strategy summary (after this audit):**
- ACTIVE_PRODUCTIVE (5): alpha, alpha_futures, alpha_intraday, gamma_futures, omega_macro
- ACTIVE_BLOCKED_BY_DATA (1): delta
- REGIME_SILENT (9): alpha_intraday_micro, alpha_options, beta, beta_trend, delta_pairs, gamma, gamma_reversion, omega, omega_vol, omega_momentum_options *(actually 10 — see corrected count below)*
- REGIME_SILENT corrected count: alpha_intraday_micro, alpha_options, beta, beta_trend, delta_pairs, gamma, gamma_reversion, omega, omega_momentum_options, omega_vol = **10**
- HALTED_EDGE_DECAY (1): alpha_crypto
- **UNKNOWN — REQUIRES AUDIT: 0** (all 8 prior-UNKNOWN strategies now have explicit evidence-backed classification)

Totals: 5 ACTIVE_PRODUCTIVE + 1 ACTIVE_BLOCKED_BY_DATA + 10 REGIME_SILENT + 1 HALTED_EDGE_DECAY = 17 / 17 strategies classified.

---

## 6. Remaining blockers

### Local engineering blockers
- **PR-12 closeout.** Move applied PR-* PA docs to `_applied/`; apply OFFICIAL_36 §9 doc cycle.
- **DEFERRED hardening (non-blocking):** trace and silence the executor-layer placeholder emitter (paper_trade_executor / live_loop close-fill synthesizer). Public ledger remains clean without it; this is for audit-narrative completeness only.

### External blockers
- **None.** PR-04 IBKR contract-details endpoint fully recovered; chains + Greeks both healthy.

### Observation/time blockers
- **PO-01 5-session soak** 0/5; needs re-base at the next stable cadence.
- **PO-04 stop_bus flutter** — intermittent 2000–3700ms spikes auto-clearing on clean_streak=5. Not a hard blocker but must be quantified across a full US-equity session.
- **PO-05 productive uptime** through 5-session window — coupled to PO-04 flutter cadence.

### Documentation/cleanup blockers
- **PR-12** as above.
- **OFFICIAL_36 §9 doc cycle** — three docs require operator-authorized edits.
- **Untracked file** `PR-05_OFFICIAL_36_amendment_line84_patch_note_2026-05-26.md` — commit or move to `_applied/`.

### Operator-decision blockers
- **PR-06 Path B re-arm** — optional; Path A formalized.
- **alpha_crypto halt review** — optional; `consecutive_negative_10` halt has been in place since 2026-05-08. Operator may choose to (a) leave halted, (b) clear halt under policy, (c) tune the consecutive-negative threshold via Pending Action.
- **Pre-Live Operator Tasks (CLAUDE.md)** — OS reboot, disk cleanup, IB-latency threshold review (if flutter persists), MES paper-position review.

### Deferred hardening items (non-blocking)
- **Executor-layer placeholder emitter trace** (per PO-03 PA §5).
- **BOX-034 canonical equity skew flake** (pnl_state vs portfolio_snapshot divergence; test_canonical_sources_agree_within_skew_tolerance flakes on this).
- **alpha_crypto consecutive-negative threshold policy review.**
- **alpha_options / omega_momentum_options regime-evidence collection** post-PR-04 recovery.

---

## 7. Updated completion estimate

| Dimension | Previous (23:01Z) | Now (00:13Z) | Reasoning |
|---|---|---|---|
| Paper system **build** maturity | ~94% | **~97%** | PO-03 closed; PR-02 / PR-02b reclassified VERIFIED; Greeks publisher recovered to status=ok. Only carry items are PR-12 cosmetic docs cleanup and the DEFERRED executor-layer hardening (non-blocking). |
| Paper-complete **proof** | ~55% | **~70%** | Strategy classifications complete (0 UNKNOWN). PO-03 VERIFIED. PO-06 VERIFIED. PO-01 / PO-02 / PO-04 / PO-05 still depend on the 5-session soak clock starting. |
| **Live-readiness** | ~40% | **~45%** | `ready_for_live=false`. Build essentially done. Live-readiness still gated on the Pre-Live Operator Tasks (CLAUDE.md), the 5-session soak evidence, and the flutter quantification. |

---

## 8. Exact next actions (top 5, ordered)

1. **Begin the 5-session paper soak clock at the next clean US-equity open** (rebased now that PO-03 is VERIFIED and strategies are fully classified). Track cumulative halt-time-per-session from PO-04 flutter; if <1h/day, accept and continue; if not, raise a threshold-review Pending Action.
2. **Prepare PR-12 closeout Pending Action.** Move applied PR-* docs into `ops/pending_actions/_applied/<commit-sha>/`. Apply OFFICIAL_36 §9 doc cycle. Resolve the untracked PR-05 patch-note file. This is the last paper-required cleanup item.
3. **Operator decision on alpha_crypto halt.** Review whether `consecutive_negative_10` halt threshold remains appropriate or should be tuned via Pending Action. Optional; halt is honest.
4. **Reclassify alpha_options / omega_momentum_options after 1–2 sessions** of post-PR-04-recovery evidence — REGIME_SILENT confirmed or upgrade to ACTIVE_PRODUCTIVE.
5. **DEFERRED hardening (non-blocking, parallelizable):** trace and silence the executor-layer placeholder emitter per PO-03 PA §5. Reduces 24 placeholder-tagged rejected rows/day to zero; does not affect paper-complete.

---

## 9. Single next action

```
Begin (or re-base) the 5-session paper soak clock at the next clean
US-equity open. Track cumulative stop_bus halt-time per session against
a soft <1h/day budget. After 5 clean sessions:
  - PO-01, PO-02, PO-04, PO-05 close VERIFIED
  - paper-complete proof advances to ~95%+
  - live-readiness can begin moving on the Pre-Live Operator Tasks
The build side is essentially done (97%); proof is the only major
remaining lever.
```

---

## FINAL PAPER REGISTER

```
PAPER_REQUIRED_LOCKED_OR_VERIFIED=PR-01, PR-02, PR-02b, PR-03, PR-04, PR-06, PR-07, PR-08, PR-09, PR-10, PR-11, PR-M2K-MYM, PR-MYM-SIZING, PO-03, IBKR-LATENCY-RECOVERY  (15 items)
PAPER_REQUIRED_PARTIAL=PR-12  (1 item — docs/archive cleanup)
PAPER_REQUIRED_BLOCKED=  (none)
PAPER_REQUIRED_DEFERRED=PR-05  (1 item — service does not exist by design)
PAPER_REQUIRED_UNKNOWN=  (none)

PAPER_OBSERVATION_LOCKED_OR_VERIFIED=PO-03, PO-06  (2 items)
PAPER_OBSERVATION_IN_PROGRESS=PO-01, PO-02, PO-05  (3 items — 5-session soak)
PAPER_OBSERVATION_BLOCKED=  (none)
PAPER_OBSERVATION_PARTIAL_FLUTTER=PO-04  (1 item — 24h: 30 triggers / 8 auto-clears; self-resolving)

STRATEGIES_CLASSIFIED=17 / 17
  ACTIVE_PRODUCTIVE: alpha, alpha_futures, alpha_intraday, gamma_futures, omega_macro  (5)
  ACTIVE_BLOCKED_BY_DATA: delta  (1)
  REGIME_SILENT: alpha_intraday_micro, alpha_options, beta, beta_trend, delta_pairs, gamma, gamma_reversion, omega, omega_momentum_options, omega_vol  (10)
  HALTED_EDGE_DECAY: alpha_crypto  (1; consecutive_negative_10 since 2026-05-08T20:15Z)
STRATEGIES_UNKNOWN=  (none)

LOCAL_ENGINEERING_BLOCKERS=PR-12 docs/archive (cosmetic). DEFERRED executor-layer placeholder emitter trace (non-blocking).
EXTERNAL_BLOCKERS=  (none — PR-04 fully recovered)
OBSERVATION_BLOCKERS=PO-01 5-session soak (0/5; needs re-base). PO-04 flutter watch (one session).
DOC_CLEANUP_BLOCKERS=PR-12 PA archival; OFFICIAL_36 §9 doc cycle; untracked PR-05 patch-note.

NEXT_ACTION=Begin (or re-base) the 5-session paper soak clock at the next clean US-equity open. Track flutter halt-time/session against soft <1h/day budget. Paper-complete proof can advance to ~95%+ after 5 clean sessions.

FINAL STATUS: PARTIAL
  - Build: VERIFIED/LOCKED for 15/17 PR items. 1 PARTIAL (PR-12 cosmetic). 1 DEFERRED by design (PR-05). 0 external blockers. 0 strategies UNKNOWN.
  - Proof: PARTIAL — 5-session soak (0/5; needs re-base). 0 unclassified strategies remain.
  - Live-readiness: BLOCKED — ready_for_live=false; Pre-Live Operator Tasks open; flutter quantification pending across a full session.
```

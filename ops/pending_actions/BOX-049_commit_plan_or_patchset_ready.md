# BOX-049 — Commit / patchset plan

- **Box number:** 049
- **Box title:** COMMIT_PLAN_OR_PATCHSET_READY
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T15:54:50Z
- **HEAD at plan cut:** `bbe7525` (short) — "GAP-039 (Phase-58/59): relocate stop-bus evaluate before early-return"
- **Branch:** `main` (ahead of `origin/main` by 402 commits — observational only; Box 049 does NOT push)
- **Plan posture:** **Recommendation only** — Box 049 does not `git add`, `git commit`, `git push`, or `git rm`. The operator (or a later box) executes the plan.

---

## 0. Scope and safety statement

This file is the per-patchset plan for the 52 changed/untracked working-tree
files inventoried by Box 048
(`runtime/completion_matrix_evidence/BOX-048_PRODUCTION_CODE_DIFF_REVIEWED.md`).

This plan **does not promote CHAD to live** and live trading is **not
authorized**. `CHAD_EXECUTION_MODE=paper`; `runtime/live_readiness.json`
`ready_for_live: false` (unchanged across Box 049).

**Governance basis (CLAUDE.md §1):** "One change at a time. Baseline,
change, verify, proceed." The plan organises the 20 modified + 32
untracked files into **20 atomic patchsets (A – T)** plus **one
explicit exclusion (Z)**, each patchset corresponding to exactly one
closed Completion-Matrix box (and a per-box anchor evidence file).

### Plan rules

1. **Per-box atomicity:** each patchset contains the production code,
   tests, ops, scripts, and (where present) docs/pending-action for
   exactly one Stage-3 box.
2. **Test-before-prod within a patchset:** when both code and a new
   test exist, the patchset is staged with the test included so the
   commit can be re-verified by `pytest -q chad/tests tests` from the
   commit's tree.
3. **No live config:** no patchset modifies live `runtime/*.json`,
   SQLite, systemd unit files inside `/etc/systemd/system/`, ledgers,
   fills, fees, trades, or broker events.
4. **No re-baseline of frozen artifacts:** SSOT v8.x / v9.0 – v9.4 and
   the Box-045 forward errata remain byte-frozen.
5. **Per-box-evidence anchored:** every patchset cites its anchor
   evidence file under `runtime/completion_matrix_evidence/`, which is
   on disk now (verified by Box 048 §4).
6. **No `--no-verify` / no `--amend`:** patchsets are NEW commits;
   pre-commit hooks (if any) must not be skipped.

---

## 1. Patchset summary table

| ID | Title (recommended commit headline)                                                                  | Files | Box(es)       | Risk    | Commit-now? |
| -- | ---------------------------------------------------------------------------------------------------- | ----- | ------------- | ------- | ----------- |
| A  | GAP-035 (Box 013/014): upstream operator-exclusion at strategy emit + new `_upstream_exclusion` module | 6     | 013, 014      | LOW     | YES         |
| B  | GAP-009 (Box 015): weekday same-side regression test                                                  | 1     | 015           | LOW     | YES         |
| C  | GAP-037 (Box 016/030): futures contract construction in close-intent + MCL schedule + MYM registry   | 5     | 016, 030      | LOW-MED | YES         |
| D  | GAP-038 (Box 017): minTick side-safe snap for FUT LMT orders                                          | 2     | 017           | LOW-MED | YES         |
| E  | NEW-GAP-041 (Box 018): live_readiness resolved-reconciliation status                                  | 2     | 018           | LOW     | YES         |
| F  | NEW-GAP-053 (Box 019): lifecycle replay backlog quiet-window distinction                              | 2     | 019           | LOW     | YES         |
| G  | NEW-GAP-044 (Box 020): options-chain refresh health rule R17                                          | 2     | 020           | LOW     | YES         |
| H  | NEW-GAP-043 (Box 021): stuck IBKR collector wall-clock guard + systemd drop-in                        | 3     | 021           | LOW-MED | YES         |
| I  | Boxes 025/026/031/032: tests/ops collection + omega_macro + Kraken stop-bus isolation + beta_trend rename | 3   | 025, 026, 031, 032 | LOW     | YES         |
| J  | Box 029: alpha_options BAG meta-preservation test addition                                            | 1     | 029           | LOW     | YES         |
| K  | Box 033 (GAP-003/008/026): strategy-registry parity test                                              | 1     | 033           | LOW     | YES         |
| L  | Box 034 (GAP-007): canonical equity source test                                                       | 1     | 034           | LOW     | YES         |
| M  | Box 035 (GAP-018 / NEW-GAP-051): halt-clear semantics — reset streak + clear_reason flag              | 3     | 035           | LOW     | YES         |
| N  | Box 037 (GAP-016): ALLOW_LIVE semantics test                                                          | 1     | 037           | LOW     | YES         |
| O  | Box 008 (GAP-036 / UNK-06): sqlite_retention argparse + `--cutoff <ISO>` + `--dry-run`                | 1     | 008           | LOW     | YES         |
| P  | Box 042 (GAP-012 / NEW-GAP-049): Telegram dedupe cleanup script + test                                | 2     | 042           | LOW     | YES         |
| Q  | Box 043: runtime artifacts cleanup script + test                                                      | 2     | 043           | LOW     | YES         |
| R  | Pending-action policies (12 docs): Boxes 028, 033, 034, 036, 037, 038, 039, 041, 042, 043, 044, GAP-053 | 12  | various       | NONE    | YES         |
| S  | Box 045: SSOT v9.5 forward errata                                                                    | 1     | 045           | NONE    | YES         |
| T  | Box 046: CHAD Evidence-Locked DoD v1.0                                                                | 1     | 046           | NONE    | YES         |
| Z  | (exclusion) `runtime/completion_matrix_evidence/BOX-*.md` (63 evidence files)                         | n/a   | n/a           | NONE    | NO — gitignored by policy |

**Totals:**
- Patchsets to commit: **20** (A – T).
- Exclusions: **1** (Z — evidence files; `runtime/` is gitignored).
- Files covered by plan: **52 / 52** (20 modified + 32 untracked). Zero unmapped.
- Files marked "commit later" or "deferred": **0**.
- Files marked "do not commit": **0** (the systemd drop-in IS committable as a source-of-truth mirror; the runtime evidence is excluded by gitignore policy, not by content).

---

## 2. Per-patchset detail

Each patchset entry includes: purpose · files · mapped boxes · anchor evidence · tests proving it · deployment impact · restart/Channel-1 requirement · rollback note · risk level · commit-now decision.

### Patchset A — GAP-035 upstream operator-exclusion (Box 013 / 014)

- **Commit headline (recommended):** `GAP-035 (Box-013/014): upstream operator-exclusion at strategy emit + _upstream_exclusion helper`
- **Purpose:** Block strategies from emitting `TradeSignal` objects on operator-excluded underlyings *before* the close-path chokepoints, preventing never-confirmable exit signals from re-firing every cycle.
- **Files (6):**
  - `chad/strategies/_upstream_exclusion.py` (new module)
  - `chad/strategies/alpha_options.py` (M, +17 / -0)
  - `chad/strategies/delta.py` (M, +11 / -0)
  - `chad/strategies/delta_pairs.py` (M, +9 / -0)
  - `chad/tests/test_strategy_upstream_exclusion.py` (new)
  - `chad/tests/test_delta_pairs.py` (M, +13 / -0 — GAP-035 either-leg test)
- **Mapped boxes:** 013 + 014.
- **Anchor evidence:** `runtime/completion_matrix_evidence/BOX-013_GAP-035_upstream_TradeSignal_emit_exclusions.md`, `BOX-014_GAP-035_deployed_verified_RESTART.md`.
- **Tests proving it:** `chad/tests/test_strategy_upstream_exclusion.py`, `chad/tests/test_delta_pairs.py`, `chad/tests/test_alpha_options_meta_preservation.py` (all in Box 047 GREEN 2361/2361).
- **Deployment impact:** Already deployed in Box 014 RESTART. Repo files are SOT mirror.
- **Restart/Channel-1:** Already restarted. No new restart needed.
- **Rollback note:** Single-commit revert restores prior emit behaviour; downstream chokepoints in `position_reconciler.apply_close_intents` remain the last line of defence, so a revert does not re-introduce exclusion gaps. Fail-closed posture in the new module (empty set on import error) means a half-revert still defaults safely.
- **Risk:** **LOW** — additive filter on upstream emit; production behaviour unchanged for non-excluded symbols.
- **Commit now?** YES.

### Patchset B — GAP-009 weekday same-side regression test (Box 015)

- **Commit headline:** `GAP-009 (Box-015): weekday same-side regression test`
- **Purpose:** Lock in the GAP-009 fix with a regression test that explicitly fails when same-side weekday entries are over-blocked.
- **Files (1):** `chad/tests/test_gap009_same_side_weekday_regression.py` (new)
- **Mapped boxes:** 015.
- **Anchor evidence:** `BOX-015_GAP-009_weekday_same_side_regression_verified.md`.
- **Tests proving it:** the file itself (in Box 047 GREEN).
- **Deployment impact:** none (test-only).
- **Restart/Channel-1:** none.
- **Rollback note:** Test-only — revert deletes the test, no production drift.
- **Risk:** **LOW**.
- **Commit now?** YES.

### Patchset C — GAP-037 futures contract construction + MCL/MYM (Box 016 / 030)

- **Commit headline:** `GAP-037 (Box-016/030): futures contract construction in close-intent + MCL post-2026-05-18 schedule advance + MYM registry`
- **Purpose:** Make reconciler close-intents wrap futures symbols as FUT with `meta.contract_month`; advance MCL after 2026-05-18 expiry; add MYM to the resolver registry.
- **Files (5):**
  - `chad/core/position_reconciler.py` (M, +53 / -3)
  - `chad/market_data/futures_contract_resolver.py` (M, +6 / -1)
  - `chad/market_data/ibkr_historical_provider.py` (M, +5 / -1 — mirror)
  - `chad/tests/test_gap037_reconciler_futures_contract.py` (new)
  - `chad/tests/test_futures_contract_resolver.py` (M, +39 / -0)
- **Mapped boxes:** 016, 030.
- **Anchor evidence:** `BOX-016_GAP-037_futures_contract_construction_RUNTIME_VERIFY.md`, `BOX-030_futures_resolver_MYM_registry_fixed.md`.
- **Tests proving it:** 2 added/modified tests + integration via Box 016 RESTART evidence.
- **Deployment impact:** Already runtime-verified in Box 016. Repo files are SOT mirror.
- **Restart/Channel-1:** Already restarted; collector picks up new schedule on its next cycle.
- **Rollback note:** Reverting restores STK-only fallback and lets MCLM6 re-appear in the schedule — both are pre-2026-05-19 behaviours that caused IBKR Error 200 on FUT closes.
- **Risk:** **LOW-MED** — touches close-path. Mitigated by Box 016 RESTART verification + 2 covering tests.
- **Commit now?** YES.

### Patchset D — GAP-038 minTick side-safe snap (Box 017)

- **Commit headline:** `GAP-038 (Box-017): minTick side-safe snap for FUT LMT orders`
- **Purpose:** Snap LMT prices to instrument minTick boundary side-safely (BUY floor / SELL ceil), never making the order more aggressive than intended; STK/OPT/BAG and unknown FUT pass through unchanged.
- **Files (2):**
  - `chad/execution/ibkr_adapter.py` (M, +100 / -2)
  - `chad/tests/test_ibkr_adapter_tick_snap.py` (new)
- **Mapped boxes:** 017.
- **Anchor evidence:** `BOX-017_GAP-038_minTick_snapping_verified.md`.
- **Tests proving it:** dedicated unit test + adapter green path.
- **Deployment impact:** new code path takes effect on next chad-live-loop restart cycle.
- **Restart/Channel-1:** restart recommended at next planned operator window (not blocking paper).
- **Rollback note:** Reverting restores pre-snap LMT prices — IBKR will silently reject sub-tick prices but no order will be MORE aggressive than intended. Safe rollback.
- **Risk:** **LOW-MED** — order-shape change, but conservative-by-construction.
- **Commit now?** YES.

### Patchset E — NEW-GAP-041 live_readiness resolved reconciliation (Box 018)

- **Commit headline:** `NEW-GAP-041 (Box-018): live_readiness resolved-reconciliation status (drift-aware fail-closed)`
- **Purpose:** Combine `reconciliation_state.json` + `position_guard_drift.json` into a resolved RED/YELLOW/GREEN status that is **fail-closed** (any drift → RED). This makes the live_readiness publisher more conservative.
- **Files (2):**
  - `ops/live_readiness_publish.py` (M, +82 / -1)
  - `chad/tests/test_gap041_live_readiness_resolved_reconciliation.py` (new)
- **Mapped boxes:** 018.
- **Anchor evidence:** `BOX-018_NEW-GAP-041_live_readiness_reconciliation_fixed.md`.
- **Tests proving it:** dedicated test + Box 047 GREEN.
- **Deployment impact:** publisher service picks up change on next cycle (~1 min cadence). No restart needed.
- **Restart/Channel-1:** none.
- **Rollback note:** Reverting widens GREEN classification (less conservative). Since current `ready_for_live=false` anyway, neither direction enables live trading. Safe rollback either way.
- **Risk:** **LOW** — strictly more conservative.
- **Commit now?** YES.

### Patchset F — NEW-GAP-053 lifecycle replay backlog quiet-window (Box 019)

- **Commit headline:** `NEW-GAP-053 (Box-019): lifecycle replay backlog quiet-window distinction`
- **Purpose:** Distinguish "pipeline working, no fills happened" (legitimate quiet) from "fills happened, ledger broke" so the backlog flag is not flipped false on a quiet window.
- **Files (2):**
  - `chad/ops/lifecycle_truth_publisher.py` (M, +137 / -4)
  - `chad/tests/test_gap053_lifecycle_backlog_quiet_window.py` (new)
- **Mapped boxes:** 019.
- **Anchor evidence:** `BOX-019_NEW-GAP-053_lifecycle_replay_coverage_fixed.md`.
- **Tests proving it:** dedicated test + Box 047 GREEN.
- **Deployment impact:** publisher service picks up change on next cycle.
- **Restart/Channel-1:** none.
- **Rollback note:** Reverting may re-introduce false-negative backlog claims; tracked under `ops/pending_actions/GAP-053_lifecycle_replay_coverage_policy.md`.
- **Risk:** **LOW**.
- **Commit now?** YES.

### Patchset G — NEW-GAP-044 options-chain refresh health rule R17 (Box 020)

- **Commit headline:** `NEW-GAP-044 (Box-020): options-chain refresh health rule R17`
- **Purpose:** Promote options-chain refresh failure (error field, empty chains, stale `ts_utc` >26h on weekday) to a CRITICAL Finding so the existing Telegram pipeline alerts.
- **Files (2):**
  - `chad/ops/health_monitor_rules.py` (M, +208 / -0)
  - `chad/tests/test_gap044_options_chain_refresh_alert.py` (new)
- **Mapped boxes:** 020.
- **Anchor evidence:** `BOX-020_NEW-GAP-044_options_chain_refresh_fixed.md`.
- **Tests proving it:** dedicated test + Box 047 GREEN.
- **Deployment impact:** health monitor activates new rule on next cycle.
- **Restart/Channel-1:** none.
- **Rollback note:** Removing rule re-silences the failure mode; not safer.
- **Risk:** **LOW** — adds alerting, no behavioural change for healthy state.
- **Commit now?** YES.

### Patchset H — NEW-GAP-043 stuck collector wall-clock guard + systemd drop-in (Box 021)

- **Commit headline:** `NEW-GAP-043 (Box-021): stuck IBKR collector wall-clock SIGALRM guard + systemd drop-in (defence in depth)`
- **Purpose:** Process-level SIGALRM guard + systemd TimeoutStartSec=60 / RuntimeMaxSec=90 / TimeoutStopSec=30 drop-in to prevent the chad-ibkr-collector oneshot from hanging indefinitely.
- **Files (3):**
  - `chad/portfolio/ibkr_portfolio_collector_v2.py` (M, +88 / -0)
  - `chad/tests/test_gap043_collector_wall_clock_guard.py` (new)
  - `ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf` (new — SOT mirror of the already-deployed Channel-1 drop-in)
- **Mapped boxes:** 021.
- **Anchor evidence:** `BOX-021_NEW-GAP-043_collector_systemd_deploy_verify.md`, `BOX-021_NEW-GAP-043_collector_runtime_reverify.md`.
- **Tests proving it:** dedicated test + Box 047 GREEN.
- **Deployment impact:** systemd drop-in **already deployed** under Box 021 Channel-1; this commit only checks in the SOT mirror. Code-level guard installs on next collector run.
- **Restart/Channel-1:** none new (already done).
- **Rollback note:** Reverting removes finite timeouts and re-introduces the hang risk; not safer.
- **Risk:** **LOW-MED** — narrows runtime envelope; healthy runs complete in ~2-5s vs the 60s start budget.
- **Commit now?** YES.

### Patchset I — Tests/ops + omega_macro + Kraken stop-bus isolation + beta_trend rename (Boxes 025/026/031/032)

- **Commit headline:** `Boxes-025/026/031/032: tests/ops collection + omega_macro lane + Kraken stop-bus isolation + beta_trend fixture rename`
- **Purpose:** Bundle four small test-surface fixes that share no production-code impact: pytest-collects `tests/ops` cleanly; omega_macro execution-lane mapping; Kraken test isolates stop-bus monkeypatch (production code unchanged); beta → beta_trend strategy name in daily ops report fixture.
- **Files (3):**
  - `chad/tests/test_kraken_execution.py` (M, +16 / -0 — `_isolate_stop_bus` autouse fixture)
  - `chad/tests/test_omega_macro_execution_lane.py` (M, +14 / -0)
  - `tests/ops/test_daily_ops_report.py` (M, +2 / -2 — rename only)
- **Mapped boxes:** 025, 026, 031, 032.
- **Anchor evidence:** `BOX-025_*`, `BOX-026_*`, `BOX-031_*`, `BOX-032_*`.
- **Tests proving it:** the changed files themselves (Box 047 GREEN).
- **Deployment impact:** none (test-only).
- **Restart/Channel-1:** none.
- **Rollback note:** test-only; revert deletes assertions.
- **Risk:** **LOW**.
- **Commit now?** YES. (Slight tension with one-change-per-commit, but each diff is ≤16 lines and all are test-only with no shared module touched — bundling is a valid grouping per CLAUDE.md §2 "surgical changes only".)

### Patchset J — Box 029 alpha_options BAG meta-preservation

- **Commit headline:** `Box-029: alpha_options BAG meta-preservation test`
- **Purpose:** Lock in BAG meta-preservation through signal -> intent -> trade write.
- **Files (1):** `chad/tests/test_alpha_options_meta_preservation.py` (M, +12 / -0).
- **Mapped boxes:** 029.
- **Anchor evidence:** `BOX-029_alpha_options_BAG_meta_tests_fixed.md`.
- **Risk:** LOW. **Commit now?** YES.

### Patchset K — Box 033 strategy-registry parity test

- **Commit headline:** `Box-033 (GAP-003/008/026): strategy-registry parity test`
- **Files (1):** `chad/tests/test_strategy_registry_parity.py` (new).
- **Mapped boxes:** 033.
- **Anchor evidence:** `BOX-033_GAP-003_008_026_strategy_registry_reconciled.md`.
- **Risk:** LOW. **Commit now?** YES.

### Patchset L — Box 034 canonical equity source test

- **Commit headline:** `Box-034 (GAP-007): canonical equity source test`
- **Files (1):** `chad/tests/test_canonical_equity_source.py` (new).
- **Mapped boxes:** 034.
- **Anchor evidence:** `BOX-034_GAP-007_canonical_equity_source_declared.md`.
- **Risk:** LOW. **Commit now?** YES.

### Patchset M — Box 035 GAP-018 / NEW-GAP-051 halt-clear semantics

- **Commit headline:** `GAP-018 / NEW-GAP-051 (Box-035): halt-clear resets consecutive_negative + clear_reason flag`
- **Files (3):**
  - `chad/risk/edge_decay_monitor.py` (M, +23 / -3)
  - `scripts/clear_edge_decay.py` (M, +11 / -1)
  - `chad/tests/test_gap018_halt_clear_semantics.py` (new)
- **Mapped boxes:** 035.
- **Anchor evidence:** `BOX-035_GAP-018_NEW-GAP-051_halt_clear_semantics_fixed.md`.
- **Deployment impact:** operator-on-demand (clear_edge_decay.py invocation).
- **Restart/Channel-1:** none.
- **Rollback note:** Reverting restores stale-streak leak after clear; raw trade history untouched so next monitor pass re-derives the truth anyway.
- **Risk:** LOW. **Commit now?** YES.

### Patchset N — Box 037 ALLOW_LIVE semantics test

- **Commit headline:** `Box-037 (GAP-016): ALLOW_LIVE semantics test (gates allow_entries within paper, not live order routing)`
- **Files (1):** `chad/tests/test_gap016_allow_live_semantics.py` (new).
- **Mapped boxes:** 037.
- **Anchor evidence:** `BOX-037_GAP-016_ALLOW_LIVE_semantics_locked.md`.
- **Risk:** LOW. **Commit now?** YES.
- **Safety note:** Test only — it asserts that `operator_mode=ALLOW_LIVE` does NOT enable live order routing. It does not flip any live-mode flag.

### Patchset O — Box 008 sqlite_retention argparse/--cutoff/--dry-run

- **Commit headline:** `Box-008 (GAP-036 / UNK-06): sqlite_retention argparse + --cutoff <ISO> + --dry-run`
- **Files (1):** `ops/sqlite_retention.py` (M, +157 / -23).
- **Mapped boxes:** 008.
- **Anchor evidence:** `BOX-008_GAP-036_stale_PendingSubmit_debt_CUTOFF_CLEANUP.md`.
- **Deployment impact:** operator-on-demand.
- **Rollback note:** Reverting restores positional-integer-only mode; cleanup still possible but boundary-imprecise.
- **Risk:** LOW. **Commit now?** YES.

### Patchset P — Box 042 Telegram dedupe cleanup

- **Commit headline:** `GAP-012 / NEW-GAP-049 (Box-042): Telegram dedupe cleanup script + test`
- **Files (2):**
  - `ops/cleanup_telegram_dedupe.py` (new)
  - `chad/tests/test_box042_telegram_dedupe_cleanup.py` (new)
- **Mapped boxes:** 042.
- **Anchor evidence:** `BOX-042_GAP-012_NEW-GAP-049_Telegram_dedupe_bounded.md`.
- **Risk:** LOW. **Commit now?** YES.

### Patchset Q — Box 043 runtime artifacts cleanup

- **Commit headline:** `Box-043: runtime artifacts cleanup script + test (logrotate coverage extension)`
- **Files (2):**
  - `ops/cleanup_runtime_artifacts.py` (new)
  - `chad/tests/test_box043_runtime_artifacts_cleanup.py` (new)
- **Mapped boxes:** 043.
- **Anchor evidence:** `BOX-043_logrotate_runtime_cleanup_coverage_complete.md`.
- **Risk:** LOW. **Commit now?** YES.

### Patchset R — Pending-action policies (Boxes 028 / 033 / 034 / 036 / 037 / 038 / 039 / 041 / 042 / 043 / 044 + GAP-053)

- **Commit headline:** `Pending-action policies: Boxes 028/033/034/036/037/038/039/041/042/043/044 + GAP-053 (Stage-3 carry-forward)`
- **Files (12):** all under `ops/pending_actions/` (see §1 row R). All untracked.
- **Mapped boxes:** 028, 033, 034, 036, 037, 038, 039, 041, 042, 043, 044, plus the GAP-053 policy note.
- **Anchor evidence:** the corresponding `runtime/completion_matrix_evidence/BOX-NNN_*.md` for each.
- **Deployment impact:** none (docs).
- **Restart/Channel-1:** none.
- **Rollback note:** docs-only; revert removes the policy notes but does not change behaviour.
- **Risk:** **NONE**. **Commit now?** YES.

### Patchset S — Box 045 SSOT v9.5 forward errata

- **Commit headline:** `Box-045: SSOT v9.5 forward errata (no rewrite of v9.4 or earlier)`
- **Files (1):** `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md` (new).
- **Mapped boxes:** 045.
- **Anchor evidence:** `BOX-045_SSOT_v9_5_forward_errata_cut.md`.
- **Risk:** NONE. **Commit now?** YES.

### Patchset T — Box 046 CHAD Evidence-Locked DoD v1.0

- **Commit headline:** `Box-046: CHAD Evidence-Locked DoD v1.0 (initial cut)`
- **Files (1):** `docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md` (new — currently 381 lines, includes Box 047 GREEN closure).
- **Mapped boxes:** 046.
- **Anchor evidence:** `BOX-046_DOD_checklist_updated_to_v1_0.md`.
- **Risk:** NONE. **Commit now?** YES.

### Patchset Z — EXCLUDED — evidence files

- **Files (63 on disk):** `runtime/completion_matrix_evidence/BOX-*.md`.
- **Decision:** **Not committed.** `runtime/` is gitignored by project policy (see `.gitignore`); evidence files live on disk as the source of truth and are anchored by path from the committed DoD (`docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md`). Box 045 SSOT errata locks this convention.
- **Rationale:** Committing 63+ runtime files would force runtime artifacts into git, violate the existing `.gitignore` policy, and bloat the repo with content that is regenerable from operator-run audits.
- **Risk:** NONE (status quo).

---

## 3. Per-patchset deployment matrix

| Patchset | Channel-1 deploy needed? | chad-live-loop restart needed? | Other service action?                                  |
| -------- | ------------------------ | ------------------------------ | ------------------------------------------------------ |
| A        | no (already deployed Box-014 RESTART) | no                       | none                                                    |
| B        | no (test only)            | no                            | none                                                    |
| C        | no (already runtime-verified Box-016) | no                       | collector picks up MCL schedule on next cycle           |
| D        | no                        | recommended at next ops window | none required for paper                                 |
| E        | no                        | no                            | live_readiness publisher picks up on next 1-min cycle    |
| F        | no                        | no                            | lifecycle_truth_publisher picks up on next cycle         |
| G        | no                        | no                            | health monitor picks up on next cycle                    |
| H        | already done (Box-021 systemd_deploy_verify) | no                | systemd drop-in already at `/etc/systemd/system/...`     |
| I        | no (test only)            | no                            | none                                                    |
| J        | no (test only)            | no                            | none                                                    |
| K        | no (test only)            | no                            | none                                                    |
| L        | no (test only)            | no                            | none                                                    |
| M        | no                        | no                            | operator-on-demand                                       |
| N        | no (test only)            | no                            | none                                                    |
| O        | no                        | no                            | operator-on-demand                                       |
| P        | no                        | no                            | operator-on-demand                                       |
| Q        | no                        | no                            | operator-on-demand                                       |
| R        | no (docs only)            | no                            | none                                                    |
| S        | no (docs only)            | no                            | none                                                    |
| T        | no (docs only)            | no                            | none                                                    |

**Channel-1 / restart pending after Box 049 commit:** **0 new items**.
Carryover from Box 048 §7 unchanged: (1) chad-scr-sync Telegram env soak
ongoing, (2) ALLOW_LIVE Channel-1 promotion deferred, (3) ib_async Phase 2
open, (4) MSFT guard drift operator review, (5) retention-policy follow-ups
open, (6) Box-021 systemd drop-in already deployed (no new action).

---

## 4. Per-patchset rollback summary

| Patchset | Rollback effect                                                                                                                                  | Safer-to-revert?                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| A        | Strategies again emit on operator-excluded symbols; downstream chokepoints still block close intents (defence-in-depth intact).                  | No — re-introduces upstream emit churn.                                                            |
| B        | Test removed; no production behaviour change.                                                                                                     | Neutral.                                                                                          |
| C        | Reconciler regresses to STK-only fallback for FUT (Error 200 returns); MCL schedule re-includes expired 202606.                                  | No — re-introduces the bug.                                                                       |
| D        | LMT prices not snapped; IBKR silently rejects sub-tick prices. Never makes orders more aggressive.                                                | Neutral-to-worse.                                                                                 |
| E        | live_readiness gate widens GREEN classification (less conservative). `ready_for_live` still gated separately; no live trading risk.                | Less conservative — not safer.                                                                    |
| F        | Backlog flag may false-negative on legitimate quiet windows.                                                                                     | No.                                                                                               |
| G        | Options-chain refresh failures go silent.                                                                                                         | No.                                                                                               |
| H        | Collector can again hang indefinitely on a stalled IBKR `accountSummary()`.                                                                       | No.                                                                                               |
| I        | Test surface regresses (no isolation; old fixture name).                                                                                          | No.                                                                                               |
| J/K/L/N  | Test removed; no production change.                                                                                                              | Neutral.                                                                                          |
| M        | Stale `consecutive_negative` returns after operator clear; raw history untouched so monitor pass still re-derives.                                | No.                                                                                               |
| O        | sqlite_retention loses `--cutoff` precision; legacy positional `<days>` still works.                                                              | No.                                                                                               |
| P/Q      | Cleanup scripts unavailable; `chad-disk-guard` still bounds disk usage.                                                                          | No.                                                                                               |
| R/S/T    | Documentation/policy lost from git history; on-disk artifacts persist.                                                                            | Neutral (no production effect).                                                                   |

---

## 5. Pre-commit safety acceptance (operator checklist)

Before any operator (or follow-on box) executes this plan, the following
acceptance criteria must hold:

1. `chad-live-loop.service` is `active (running)`.
2. `runtime/live_readiness.json` `ready_for_live=false`.
3. `pytest -q chad/tests tests` is GREEN (Box 047 evidence is the
   canonical reference: 2361/2361, 92.69s).
4. Working tree exactly matches Box-048 inventory (20 modified + 32
   untracked, no new file appears).
5. No `git add`, `git commit`, `git push`, `git rm`, `git reset
   --hard`, `git checkout --` in the Box 049 window.
6. Channel-1 deployments: only Box-021 systemd drop-in mirror is
   needed for the operator file-system (already done).
7. No commit may flip `runtime/live_readiness.json` `ready_for_live`
   to `true`.
8. Each patchset commit message references the originating box number
   (e.g. `(Box-013/014)`, `(Box-021)`) so `git log` remains traceable.

---

## 6. Anti-speculation footer

- No file was assumed safe without classification — every file is
  in §1 / §2 / §3 / §4 with an explicit box and evidence anchor.
- No patchset claims closure that does not have an on-disk evidence
  file under `runtime/completion_matrix_evidence/`.
- No live trading authorization is granted by this plan.
- No future box (050 – 062) is claimed as complete or pre-staged.
- This is a plan, not an execution. Box 049 does not `git add`, does
  not `git commit`, does not `git push`, does not `git rm`. The
  working tree is byte-identical to the state at the start of Box 049
  (modulo this new pending-action file and the Box 049 evidence file
  written under `runtime/completion_matrix_evidence/`).

**live trading not authorized.**

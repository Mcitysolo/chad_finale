# BOX-054 — Final Deployment Actions List

- **Box number:** 054
- **Box title:** FINAL_DEPLOYMENT_ACTIONS_LIST_READY
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T17:24:03Z
- **HEAD at cut:** `bbe7525` (short) — "GAP-039 (Phase-58/59): relocate stop-bus evaluate before early-return"
- **Branch:** `main`
- **List posture:** **Recommendation only** — no action below is executed by Box 054. Each Channel-1 action requires separate operator approval.

---

## 0. Scope and safety statement

This file lists every remaining deployment / operator / engineering
action needed for CHAD to progress from current PAPER posture toward
live readiness. It consolidates:

- 26 reconciled open items (G-01 – G-26) from Box 052
  (`ops/pending_actions/BOX-052_final_open_gaps_reconciled.md`).
- 7 current runtime blockers from Box 053
  (`runtime/completion_matrix_evidence/BOX-053_FINAL_RUNTIME_READINESS_SNAPSHOT_CAPTURED.md` §12).
- 20 patchsets (A – T) + 1 exclusion (Z) from Box 049
  (`ops/pending_actions/BOX-049_commit_plan_or_patchset_ready.md`).
- 17 follow-ups from Box 050 changelog §7.

Items appearing in multiple sources are deduplicated under their canonical
G-NN id (from Box 052) or assigned a new D-NN id (deployment) below.

### Canonical safety posture (non-negotiable)

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This document is **not** authorization
  to execute any of the actions below. Each Channel-1 action requires
  **separate explicit operator approval** before it may be performed.
- **`ready_for_live` remains `false`** (`runtime/live_readiness.json`,
  publisher-driven). No action below mutates that flag manually.
- **This document does not promote CHAD to live**, does not make CHAD
  live-ready, does not authorize live trading, and does not close any
  box beyond Box 054.

### Channel conventions

- **Channel 1 — Terminal / operator-only:** `systemctl`, broker actions,
  `scripts/close_guard_entry.py`, cleanup-script invocations, OS-level
  changes. Requires explicit operator approval per action.
- **Channel 2 — Code / docs commits:** `git add` / `git commit` per
  Box-049 patchsets, doc back-edits. Requires the Box 051 commit
  decision to be approved first.
- **Manual external:** action that originates upstream (e.g. IB Gateway
  latency remediation by Interactive Brokers, kernel update by AWS).

---

## 1. Runtime blocker summary (from Box 053 snapshot)

| #  | Blocker (current observation)                                                | Source                                                  | Severity | Cross-ref          |
| -- | ---------------------------------------------------------------------------- | ------------------------------------------------------- | -------- | ------------------ |
| 1  | `ready_for_live: false`                                                      | `runtime/live_readiness.json`                           | gate     | publisher-driven   |
| 2  | `stop_bus.active: true` (broker_latency 8258 ms > 2000 ms)                    | `runtime/stop_bus.json`                                 | **P1**   | G-26 / D-04        |
| 3  | `reconciliation_state.status: RED` (`broker_source: "unavailable:"`)          | `runtime/reconciliation_state.json`                     | **P1**   | G-12, G-26 / D-03  |
| 4  | `position_guard_drift.drift_count: 2` (MSFT both legs `side_mismatch`)        | `runtime/position_guard_drift.json`                     | P1       | G-01 / D-01        |
| 5  | `trade_lifecycle_state.backlog_flag: true` (GAP-053 quiet-window not accepted) | `runtime/trade_lifecycle_state.json`                    | P2       | G-20 / D-05        |
| 6  | 6 `chad-ibkr-*` services in `failed` state                                    | `systemctl is-active …`                                  | P2       | G-26 / D-06        |
| 7  | ALLOW_LIVE Channel-1 promotion deferred                                       | DoD §2.2 / Box 037                                      | **P1**   | G-11 / D-13 (gate) |

---

## 2. Final deployment action list (D-01 – D-26)

Legend:
- **Priority:** P0 / P1 / P2 / P3 / Maintenance.
- **Channel:** C1 = Channel 1 terminal · C2 = Channel 2 code/docs · MX = Manual external.
- **Blocks live?:** yes / no / gate (gate = the live-readiness publisher's own gate, not an action).

### 2.A Live-blocking actions

| ID    | Title                                                                                                 | Cat                         | Pri   | Source (box / G-id)                  | Source evidence path                                                                                              | Blocks live? | Channel | Exact safe next action                                                                                                                                                       | Prerequisites                                  | Expected proof                                                                                                                                              | Rollback / undo                                                                                                  | Owner / module                                |
| ----- | ----------------------------------------------------------------------------------------------------- | --------------------------- | ----- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| D-01  | Close stale MSFT position-guard drifts (alpha\|MSFT, alpha_intraday\|MSFT)                              | operator decision           | **P1** | Box 052 G-01 + Box 053 #4            | `runtime/position_guard_drift.json`; `ops/pending_actions/GAP-028_position_guard_broker_truth_drift_wiring.md`     | yes          | C1      | `scripts/close_guard_entry.py --strategy alpha --symbol MSFT --reason "close stale guard per GAP-028 PERMISSIVE"` then same for `alpha_intraday`. Then re-verify `runtime/position_guard_drift.json drift_count: 0`. | Operator GO; SCR ∈ {CONFIDENT, CAUTIOUS}; exec_mode=paper. | `runtime/position_guard_drift.json drift_count: 0` on next publisher cycle.                                                                                  | Re-open requires re-running open intent; documented by GAP-028 PERMISSIVE design (rebuilder will repopulate from `trade_closer_state.queues`). | operator + `chad/core/position_guard.py`        |
| D-02  | Resolve `stop_bus.active=true` via the **canonical path** (do NOT edit `runtime/stop_bus.json` by hand) | engineering + operator       | **P1** | Box 053 #2 + Box 052 G-26             | `runtime/stop_bus.json`; CLAUDE.md Pre-Live Operator Tasks (IB Gateway latency)                                    | yes          | MX → C1 | (1) Drive the underlying broker-latency cause to <2000 ms (MX, see D-04). (2) The clean-streak hysteresis (GAP-034 / Phase-44 durable stop-bus auto-clear) will then flip `stop_bus.active=false` on its own. **No manual JSON edit.** | D-04 (broker-latency remediation) completes. | `runtime/stop_bus.json active: false` with `cleared_at` set; verified across several live-loop cycles.                                                       | None needed (auto-clear is the design). If hysteresis misfires, follow GAP-034 / Phase-44 evidence path.            | live-loop / stop-bus state machine             |
| D-03  | Resolve `reconciliation_state.status=RED` (broker_source unavailable; 22 chad_open vs 0 broker_positions) | engineering                  | **P1** | Box 053 #3 + Box 052 G-12 / G-26      | `runtime/reconciliation_state.json`                                                                                | yes          | MX → C1 | Restore IB Gateway broker connectivity (D-04 / D-06 dependencies). Once broker truth becomes available, the publisher will re-derive `RED → GREEN`. **No manual JSON edit.** | D-04 broker latency reduced; D-06 ibkr-* services recovered. | `runtime/reconciliation_state.json status: GREEN`; `broker_source != "unavailable:"`; `counts.broker_positions > 0`.                                          | None (publisher-derived).                                                                                         | reconciliation publisher / IB Gateway          |
| D-04  | Investigate / remediate IB Gateway weekday latency (dangerous classification, >750 ms; current observed ~8.2 s) | future hardening / external | **P1** | Box 052 G-26 + CLAUDE.md Pre-Live Operator Tasks #3 | `ops/pending_actions/GAP-020_ibgateway_bindaddress_maintenance.md`; `runtime/completion_matrix_evidence/BOX-024_GAP-020_weekday_latency_proof.md` | yes          | MX → C1 | (1) Operator audits IB Gateway connection (network path, JVM heap, throttle settings). (2) If known root cause is the bind-address / TWS configuration, apply the operator-side fix per Box 024 evidence. (3) Re-measure broker latency.                                                  | Operator GO; access to IB Gateway host.        | Broker latency < 2000 ms sustained ≥ 1 trading hour; `runtime/stop_bus.json` auto-clears via D-02.                                                            | Roll-back the IB Gateway config change; the stop-bus will re-engage on any future latency spike >2000 ms.            | operator + IB Gateway                          |
| D-05  | Bring `trade_lifecycle_state.backlog_flag` back to `false` legitimately (GAP-053 quiet-window policy)   | engineering                  | P2     | Box 053 #5 + Box 052 G-20             | `runtime/trade_lifecycle_state.json`; `ops/pending_actions/GAP-053_lifecycle_replay_coverage_policy.md`            | partial      | none / engineering | This is a **derived signal** — `backlog_flag` will flip to `false` when the publisher proves a quiet window OR fresh broker-fill activity arrives. Resolution flows from D-03 / D-04. **No manual JSON edit.** | D-03, D-04 progress.                                | `backlog_flag: false` AND `quiet_window_accepted: true` OR `had_recent_broker_fill_activity: true`.                                                          | None (derived).                                                                                                  | `chad/ops/lifecycle_truth_publisher.py`        |
| D-06  | Inspect / fix 6 `chad-ibkr-*` services in `failed` state                                              | engineering / operator       | P2     | Box 053 #6 + Box 045 §7 row 5         | `systemctl status chad-ibkr-collector.service`, `chad-ibkr-broker-events.service`, etc.                          | partial      | C1      | (1) Per service, run `systemctl status <unit>` + `journalctl -u <unit> -n 200 --no-pager` to capture failure mode. (2) For collector specifically, the wall-clock SIGALRM guard (Box 021) is in place — if hangs were the cause, the recovery is to invoke `systemctl reset-failed <unit>` and wait for the next timer tick. (3) For broker-events / paper-fill-harvester / paper-ledger-watcher / options-chain-refresh / positions-snapshot, follow per-service runbooks (some are oneshots; `failed` between fires is expected per Box 045 §7). | Operator GO; per-service rationale before each restart. | All 6 services either `active`, `inactive` (expected for oneshots), or a documented per-service exception.                                                    | If `reset-failed` mis-triggers, the timer-driven services will re-attempt on next tick.                              | operator + systemd                              |
| D-07  | `ib_async` migration Phase 2 — port remaining 5 files (and resolve `qualify_timeout_ib_async_phase2`)  | future hardening (code)      | **P1** | Box 052 G-12 + Box 050 §7 #10         | `ops/pending_actions/BOX-038_ib_insync_migration_policy.md`; `ops/pending_actions/qualify_timeout_ib_async_phase2.md` | yes          | C2      | Engineering work: identify the 5 remaining `ib_insync` files (per CLAUDE.md migration count); port each to `ib_async` with the qualify-timeout fix; run `pytest`. Box 047 GREEN baseline must remain GREEN after the migration. | Test baseline GREEN; one-change-at-a-time per CLAUDE.md §1. | All 5 files import `ib_async` (no `ib_insync` residue); `qualifyContracts` completes without 10s timeout; full suite still GREEN.                            | Per-file `git revert` of each migration commit; ib_insync remains importable as a fallback during transition.       | execution / ibkr_adapter / qualifyContracts    |
| D-08  | Re-baseline `CLAUDE.md` `ib_async` migration count when D-07 completes                                | documentation lag            | P3     | Box 052 G-13 + Box 050 §7 #11         | `CLAUDE.md` line "Phase 1 complete (18 files), Phase 2 pending (5 files remaining)"                                | no           | C2      | After D-07 finishes, edit `CLAUDE.md` to read "Phase 1 + Phase 2 complete (23 files migrated)" or equivalent ground truth.                                                                                                              | D-07 complete.                                  | `CLAUDE.md` count matches actual `ib_async`-imported files; `grep -r "ib_insync" chad/` returns 0 in production code.                                         | `git revert` the CLAUDE.md edit.                                                                                  | docs / CLAUDE.md                                |
| D-09  | `alpha_intraday_micro` weight policy / operator decision                                              | operator decision            | P2     | Box 052 G-08                          | `ops/pending_actions/BOX-033_alpha_intraday_micro_weight_policy.md`                                                | yes (sizing) | C1      | Operator selects a weight policy for `alpha_intraday_micro` and ratifies via pending-action update + (if needed) a config change recorded as a Pending Action per CLAUDE.md §3.                                                          | Operator GO; per-CLAUDE.md governance, no direct config mutation. | Pending action updated with chosen weight + rationale.                                                                                                       | Revert pending action text; no production code change needed at the decision stage.                                | operator + risk allocator                       |
| D-10  | Canonical equity source policy enforcement                                                            | operator decision            | P2     | Box 052 G-09                          | `ops/pending_actions/BOX-034_canonical_equity_source_policy.md`                                                    | yes          | C1      | Operator ratifies the policy; engineering wires enforcement into the equity-source resolver (if not already there per Box 034 test).                                                                                                    | Operator policy ratification.                  | `test_canonical_equity_source.py` continues to pass + enforcement is asserted in the resolver.                                                               | Per-commit revert.                                                                                                | operator + risk allocator                       |
| D-11  | XGB veto retrain decision (2026-05-17 dirty-tree)                                                     | operator decision            | P3     | Box 052 G-10                          | `docs/XGB_VETO_WEEKLY_RETRAIN_DIRTY_TREE_DECISION_2026-05-17.md` + `ops/pending_actions/BOX-036_XGB_veto_documentation.md` | partial      | C1      | Operator decides whether to retrain XGB on a clean tree or accept the dirty-tree snapshot; record decision in the pending action.                                                                                                       | Operator GO.                                  | Pending action updated; if retrained, new model file with hygiene metadata.                                                                                  | Revert the pending action update; model file rollback per its own versioning.                                       | operator + ML                                   |
| D-12  | OS reboot policy + disk cleanup + LiveGate posture flip (pre-live operator checklist from CLAUDE.md)  | operator                     | P3 / Maintenance | Box 052 G-24 + G-25; CLAUDE.md Pre-Live Operator Tasks | `CLAUDE.md` "Pre-Live Operator Tasks"                                                                              | yes          | C1      | (1) Apply reboot at next kernel update window. (2) Prune `backups/` below 75% usage. (3) When all P1 blockers resolved, prepare LiveGate posture-flip pending action (per CLAUDE.md §3 governance: prepared as Pending Action, never directly mutated). | All prior P1 blockers resolved; explicit operator GO. | `df -h` shows < 75%; LiveGate pending action filed; ALLOW_LIVE Channel-1 promotion request prepared (gated by D-13). | None for cleanup; LiveGate pending action revert per its file.                                                  | operator + OS + LiveGate                         |
| D-13  | ALLOW_LIVE Channel-1 promotion (final live activation)                                                | operator decision            | **P1** | Box 052 G-11 + DoD v1.0 §2.2          | `ops/pending_actions/BOX-037_ALLOW_LIVE_semantics.md`                                                              | yes          | C1      | **The final operator GO.** Requires all 5 DoD §2.2 conditions to hold simultaneously: (1) all 62 boxes closed with evidence; (2) P0/P1 truth clean; (3) tests + docs pass; (4) live_readiness publisher writes `ready_for_live=true` (NOT a manual edit); (5) explicit operator GO recorded + LiveGate accepts. **DO NOT execute until items D-01 through D-12 and Boxes 054 – 062 are closed.** | All 5 DoD §2.2 conditions; D-01 through D-12 resolved; Boxes 055 – 062 closed. | `runtime/live_readiness.json ready_for_live: true` published by canonical publisher (not manual); LiveGate accepts; recorded operator GO. | Reverting live promotion is a destructive-to-position operation; consult the live-promotion runbook before any rollback. | operator (final authority)                       |

### 2.B Non-live-blocking hardening

| ID    | Title                                                                                                 | Cat                         | Pri   | Source                                | Source evidence path                                                                                              | Blocks live? | Channel | Exact safe next action                                                                                                                                                       | Prerequisites                                  | Expected proof                                                                                                                                              | Rollback / undo                                                                                                  | Owner / module                                |
| ----- | ----------------------------------------------------------------------------------------------------- | --------------------------- | ----- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| D-14  | Lifecycle replay coverage policy / lifecycle-replay-engine "no timer" guard (subsidiary: D-01 of Box 052) | future hardening             | P2     | Box 052 G-20                          | `ops/pending_actions/GAP-053_lifecycle_replay_coverage_policy.md`; `ops/pending_actions/lifecycle_replay_engine_no_timer.md` (subsidiary CLOSED 2026-05-12) | no           | C2      | Operator decision on whether to wire a systemd timer for `lifecycle_replay_engine.py`; if no, document as "intentional no-timer with covering invariants". Then update pending action. | Operator decision.                              | Pending action updated; if timer added, systemd unit + Box-021-style timeout guards.                                                                         | Per-commit revert.                                                                                                | operator + lifecycle_replay_engine             |
| D-15  | Localhost-bind hardening for ports 9618 / 9619 / 9620 (operational maintenance)                       | maintenance                  | P3     | Box 052 G-06                          | `ops/pending_actions/GAP-020_ibgateway_bindaddress_maintenance.md`                                                 | no           | C1      | Apply localhost-bind drop-in at next operator window (per pending-action runbook).                                                                                                                                                      | Operator GO; next maintenance window.          | `ss -ltnp` shows ports 9618/9619/9620 bound to 127.0.0.1 only; tests still GREEN.                                                                            | Revert the drop-in via `systemctl revert <unit>` or remove the override file.                                       | operator + systemd                              |
| D-16  | Deterministic failure-cluster review schedule                                                         | operator decision            | P3     | Box 052 G-07                          | `ops/pending_actions/BOX-028_deterministic_failure_cluster_classification.md`                                       | no           | C1      | Operator schedules a periodic review cadence; record in pending action.                                                                                                                                                                  | Operator GO.                                   | Pending action updated with cadence + ownership.                                                                                                              | Revert pending action update.                                                                                       | operator                                       |
| D-17  | STK sub-cent price snap edge cases (deferred; revisit on observed regression)                          | deferred policy              | P3     | Box 052 G-03                          | annotated in `runtime/completion_matrix_evidence/BOX-017_GAP-038_minTick_snapping_verified.md`                       | no           | none    | No action unless a regression is observed.                                                                                                                                                                                                | n/a                                            | n/a                                                                                                                                                          | n/a                                                                                                                | execution / ibkr_adapter                       |
| D-18  | Options-chain SPY contract-details timeout (deferred; re-fire on next cadence; R17 alert in place)    | deferred policy              | P3     | Box 052 G-04                          | annotated in `BOX-020_NEW-GAP-044_options_chain_refresh_fixed.md`                                                    | no           | none    | No action unless R17 alert fires.                                                                                                                                                                                                         | n/a                                            | R17 health rule remains silent.                                                                                                                              | n/a                                                                                                                | options-chain refresh                          |
| D-19  | MES paper-ledger stale position (pre-existing; paper only)                                            | maintenance                  | P2     | Box 052 G-19                          | `ops/pending_actions/GAP-027_mes_paper_ledger_stale_position.md`                                                    | no           | C1      | Operator-on-demand cleanup of stale paper-ledger entry per pending-action runbook.                                                                                                                                                       | Operator GO; SCR posture acceptable.            | Paper ledger entry removed; reconciliation re-runs cleanly.                                                                                                  | Paper-only; restore from backup if needed.                                                                          | operator + paper_exec                          |
| D-20  | RuntimeMaxSec-on-oneshot cosmetic warning + monotonic timer drift (documented harmless)               | documented harmless          | Maint  | Box 052 G-05 + G-18                   | `ops/pending_actions/BOX-044_systemd_timer_drift_policy.md`                                                          | no           | none    | None.                                                                                                                                                                                                                                     | n/a                                            | n/a                                                                                                                                                          | n/a                                                                                                                | systemd                                         |
| D-21  | `chad-scr-sync` Telegram env post-deploy soak (deployed; soak continuing)                              | maintenance                  | P3     | Box 052 G-14                          | `ops/pending_actions/BOX-039_chad_scr_sync_telegram_env.md`                                                          | no           | none    | Observe; no action unless an alert fires.                                                                                                                                                                                                 | n/a                                            | No new Telegram-env-missing warnings in journal.                                                                                                              | n/a                                                                                                                | scr-sync / systemd                              |

### 2.C Documentation / commit / archive actions

| ID    | Title                                                                                                 | Cat                         | Pri   | Source                                | Source evidence path                                                                                              | Blocks live? | Channel | Exact safe next action                                                                                                                                                       | Prerequisites                                  | Expected proof                                                                                                                                              | Rollback / undo                                                                                                  | Owner / module                                |
| ----- | ----------------------------------------------------------------------------------------------------- | --------------------------- | ----- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| D-22  | Execute Box-049 commit plan (Patchsets A – T) per Box-051 recommendation = Option A                    | operator decision (commit)   | P3     | Box 052 G-22 + G-23                   | `ops/pending_actions/BOX-049_commit_plan_or_patchset_ready.md`; `ops/pending_actions/BOX-051_commit_or_archive_decision_recorded.md` | no           | C2      | Per Box-051 §6.2: operator runs the baseline checks, then commits one patchset at a time (re-`pytest` between each), per the Box-049 plan §2 A – T file lists.                                                                            | Operator approves Option A; Box 047 GREEN baseline re-confirmed at commit time. | 20 commits in `git log` with per-patchset headlines + anchor evidence + proving tests; `pytest -q chad/tests tests` remains GREEN.                          | `git revert <commit>` per patchset (Box 049 plan §4 rollback table is authoritative).                              | operator + git                                  |
| D-23  | Push to origin (separate operator approval; NOT in Option A)                                          | operator decision (push)     | P3     | Box 051 §6.3                          | `ops/pending_actions/BOX-051_commit_or_archive_decision_recorded.md` §6.3                                          | no           | C1      | `git push origin main` — only after D-22 completes AND operator separately approves push. Push is **not** implicit in Option A.                                                                                                          | D-22 complete; explicit push approval.          | `git status` reports branch up to date with origin/main (and 402+N commits pushed where N = patchsets committed).                                              | `git push --force-with-lease` is not authorized; operator must coordinate any history-rewrite via a separate pending action. | operator + git                                  |
| D-24  | Back-edit DoD v1.0 §3.3 rows 048 / 049 / 050 / 051 (and 052 / 053 / 054 once they close) from placeholder OPEN to CLOSED with anchor evidence paths | documentation lag            | P3     | Box 052 G-21 (newly surfaced)         | `docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md` §3.3                                                            | no           | C2      | Edit the four rows (and successor rows as those boxes close) to record `CLOSED` + the anchor evidence file path; update §1 tracker counts accordingly. **Do not flip any other row.**                                                  | Box closures themselves are already done (evidence on disk). | DoD §3.3 rows 048 – 051 (and 052 – 054 once closed) read `CLOSED`; §1 tracker reads accurately.                                                              | `git revert` the DoD edit.                                                                                          | docs / DoD                                       |

### 2.D Optional cleanup actions

| ID    | Title                                                                                                 | Cat                         | Pri   | Source                                | Source evidence path                                                                                              | Blocks live? | Channel | Exact safe next action                                                                                                                                                       | Prerequisites                                  | Expected proof                                                                                                                                              | Rollback / undo                                                                                                  | Owner / module                                |
| ----- | ----------------------------------------------------------------------------------------------------- | --------------------------- | ----- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| D-25  | `dynamic_caps` cleanup script + retention policy apply                                                | maintenance / cleanup        | P3     | Box 052 G-15                          | `ops/pending_actions/BOX-041_dynamic_caps_retention_policy.md`                                                      | no           | C1      | Operator applies retention policy or writes/runs cleanup script per pending action.                                                                                                                                                       | Operator GO.                                   | `dynamic_caps` directory size bounded; no new alerts.                                                                                                       | Disk-guard 30-day sweep is the existing fallback.                                                                  | operator + ops                                  |
| D-26  | Telegram dedupe retention apply (`ops/cleanup_telegram_dedupe.py` ready) + runtime cleanup / logrotate scope expansion install (`ops/cleanup_runtime_artifacts.py` ready) | maintenance / cleanup        | P3     | Box 052 G-16 + G-17                   | `ops/pending_actions/BOX-042_Telegram_dedupe_retention_policy.md`; `ops/pending_actions/BOX-043_runtime_cleanup_retention_policy.md` | no           | C1      | Operator invokes the cleanup scripts on a scheduled basis (or wires them into a timer); apply retention policy.                                                                                                                          | Operator GO.                                   | Telegram dedupe size bounded; `chad-disk-guard` no longer surfaces aged runtime artifacts.                                                                  | Per-script revert; disk-guard 30-day sweep is the existing fallback.                                                | operator + ops                                  |

---

## 3. Counts by category and channel

| Bucket                                              | Count | IDs                                                                                                              |
| --------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------- |
| **Total actions**                                   | **26** | D-01 – D-26                                                                                                       |
| Live-blocking actions (§2.A)                        | **13** | D-01, D-02, D-03, D-04, D-05 (partial), D-06 (partial), D-07, D-08, D-09, D-10, D-11 (partial), D-12, D-13       |
| Non-live-blocking hardening (§2.B)                  | **8**  | D-14, D-15, D-16, D-17, D-18, D-19, D-20, D-21                                                                    |
| Documentation / commit / archive (§2.C)             | **3**  | D-22, D-23, D-24                                                                                                  |
| Optional cleanup (§2.D)                             | **2**  | D-25, D-26                                                                                                        |
| **Channel 1 (terminal / operator-only)**            | **15** | D-01, D-04 (final ops), D-06, D-09, D-10, D-11, D-12, D-13, D-15, D-16, D-19, D-23, D-25, D-26 (×2 — Telegram + runtime cleanup) |
| **Channel 2 (code / docs commits)**                 | **5**  | D-07, D-08, D-14, D-22, D-24                                                                                      |
| **Manual external**                                 | **2**  | D-04 (IB Gateway latency upstream), D-03 (broker connectivity upstream — same root cause)                          |
| **No action required (deferred / harmless)**        | **5**  | D-05 (derived), D-17, D-18, D-20, D-21                                                                            |
| Priority **P1**                                      | **7**  | D-01, D-02, D-03, D-04, D-07, D-13, D-12 (subset)                                                                  |
| Priority P2                                          | **5**  | D-05, D-06, D-09, D-10, D-19                                                                                       |
| Priority P3                                          | **12** | D-08, D-11, D-12 (subset), D-14, D-15, D-16, D-18, D-21, D-22, D-23, D-24, D-25                                    |
| Priority Maintenance                                | **2**  | D-12 (subset), D-20, D-26                                                                                          |

**Note:** Some actions span multiple priorities/channels (e.g. D-12
covers reboot + disk + LiveGate prep). Counts above use the
operator-facing primary classification.

---

## 4. Dependency / ordering hints (recommended)

(Not authorization to execute — recommended sequencing only.)

1. **First wave (live-blocking removal):**
   - D-04 (IB Gateway latency) → enables D-02 (stop-bus auto-clear) and D-03 (recon GREEN).
   - D-06 (failed ibkr-* services) — investigate in parallel with D-04.
   - D-01 (MSFT guard close) — independent; can run any time SCR ∈ {CONFIDENT, CAUTIOUS}.
2. **Second wave (engineering live-blocking):**
   - D-07 (ib_async Phase 2). Re-run pytest after each file.
   - D-08 (CLAUDE.md count re-baseline) after D-07.
3. **Third wave (operator policy):**
   - D-09 (alpha_intraday_micro weight).
   - D-10 (canonical equity source enforcement).
4. **Fourth wave (commit + doc):**
   - D-22 (execute Box-049 patchsets per Box-051 Option A).
   - D-24 (DoD §3.3 back-edit) — can be part of Patchset T or a follow-on commit.
   - D-23 (push — separate operator approval).
5. **Pre-live operator checklist:**
   - D-11 (XGB retrain decision).
   - D-12 (reboot policy + disk + LiveGate prep).
6. **Final activation:**
   - D-13 (ALLOW_LIVE Channel-1 promotion) — **only after all 5 DoD §2.2
     conditions hold simultaneously, including a publisher-driven
     `ready_for_live=true` and explicit operator GO**.
7. **Non-blocking / optional (any time):**
   - D-14, D-15, D-16, D-19, D-25, D-26.
   - D-17, D-18, D-20, D-21 are passive (no action unless triggered).

---

## 5. False-closure guardrails

This document is **NOT**:

- authorization to execute any action.
- authorization to flip `ready_for_live` to `true`.
- authorization to enter live trading.
- closure of any box beyond Box 054.
- a claim that CHAD is complete.
- a claim that CHAD is live-ready.

Each Channel-1 action requires **separate operator approval** before
execution. The Box-049 commit decision (Box 051) recommends Option A
but does not execute it; D-22 is the execution step and requires
operator GO.

**live trading not authorized. CHAD remains PAPER. `ready_for_live=false`.
This document does not promote CHAD to live and does not authorize live trading.**

---

## 6. Anti-speculation footer

- Box 054 does not execute any action in this list.
- Box 054 does not run `git add`, `git commit`, `git push`, `git rm`,
  or any file-delete command.
- Box 054 does not authorize live trading.
- Box 054 does not modify any runtime JSON, SQLite, ledgers, fills,
  fees, trades, broker events, or order state.
- Box 054 does not restart, start, stop, or daemon-reload any service.
- Box 054 does not kill any process.
- No future-box state (055 – 062) is claimed as complete.
- Every action in §2 is mapped to a source box / G-id / evidence path;
  none is invented.
- The runtime-blocker section §1 is sourced verbatim from Box 053 §12;
  no value was fabricated.

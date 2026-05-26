# PR-06 — Paper Shadow Runner / Shadow Exec Quiesce Formalization (Path A)

**Date:** 2026-05-26
**Author:** Team CHAD (Claude Code)
**Status:** DOCS-ONLY pending action. No services started, restarted, unmasked,
armed, or modified. No runtime JSON edited. No broker orders. No live
enablement. CHAD posture unchanged: `ready_for_live=false`,
`allow_ibkr_live=false`, `allow_ibkr_paper=true`.

## Plain-English summary
The PR-05/PR-08 paper-service audit (2026-05-26) confirmed that
`chad-paper-shadow-runner.service` and `chad-paper-shadow-exec.service`
have been MASKED at the systemd level since 2026-05-06 (see
`/etc/systemd/system/_masked_archive_20260506/` referenced in
PR-03 Pending Action lines 89-91). They have produced **no journal entries
in the last 24 h** and are not on any active timer.

The system has not regressed as a result. The canonical SCR/shadow path is
now the snapshot + sync timer pair. This Path A pending action formalizes
that quiesce so the named masked services are no longer considered a gap
against "paper complete".

This is **Path A** from the 2026-05-26 audit. Path B (operator-authorized
re-arm) is **out of scope** of this document and is described only as a
forward reference in §6.

## Facts of record (read-only evidence)

1. **chad-paper-shadow-runner.service and chad-paper-shadow-exec.service
   are intentionally MASKED under the 2026-05-06 quiesce policy.**
   - `systemctl show` for both reports `LoadState=masked`,
     `UnitFileState=masked`, `ActiveState=inactive`, `SubState=dead`.
   - Drop-in directories are preserved (`20-after-orchestrator.conf`,
     `90-wrapper-v2.conf`, `95-hold-seconds.conf`, `96-ibkr-env.conf`,
     `99-evidence-cycle.conf` on the runner; `60-wrapper-v2.conf` on
     the exec). Preservation is intentional so a future Path B unmask
     restores the wrapper-v2 contract without reconstruction.
   - PR-03 Pending Action (`PR-03_ib_async_phase2_migration_2026-05-25.md`,
     lines 87-91) records: "`chad-paper-shadow-runner.service` — MASKED
     (`/etc/systemd/system/_masked_archive_20260506/`)" and
     "`chad-paper-shadow-exec.service` — MASKED".

2. **The active canonical SCR/shadow path is `chad-shadow-snapshot.timer`
   + `chad-scr-sync.timer`.**
   - `chad-shadow-snapshot.service` (oneshot, triggered by
     `chad-shadow-snapshot.timer`) runs
     `python -m chad.analytics.shadow_state_snapshot` and writes
     `data/shadow/shadow_state.json`. Last exit at audit time:
     `code=exited, status=0/SUCCESS` with payload
     `ShadowState snapshot: state=WARMUP sizing_factor=0.100 paper_only=True`.
   - `chad-scr-sync.service` (oneshot, triggered by
     `chad-scr-sync.timer`) runs `ops/scr_state_sync.py` and writes
     both `runtime/scr_state.json` and `runtime/shadow_state.json`.
     Last exit at audit time: `code=exited, status=0/SUCCESS` with payload
     `{"ok": true, "out_scr": ".../runtime/scr_state.json",
     "out_shadow": ".../runtime/shadow_state.json",
     "ts_utc": "2026-05-26T00:44:35Z"}`.
   - These two timers, not the masked runner/exec pair, are now the
     SSOT-canonical publishers of the paper-shadow state surface.

3. **`runtime/scr_state.json` is fresh and currently reports
   `state=CONFIDENT`, `sizing_factor=1.0`.**
   - Audit-time mtime: `2026-05-26T00:44:37+00:00` (age ≈ 28 s).
   - Schema: `scr_state.v1`. TTL: 180 s. Within TTL.
   - The CLAUDE.md trading-posture line ("SCR=CONFIDENT, sizing_factor=1.0,
     paper_only=false") matches the runtime file.
   - Adjacent observation (NOT in PR-06 scope): the upstream
     `data/shadow/shadow_state.json` reports
     `state=WARMUP sizing_factor=0.100 paper_only=True`, while the
     post-sync `runtime/scr_state.json` reports
     `state=CONFIDENT sizing_factor=1.0 paper_only=false`. This is the
     Epoch-2 promotion logic inside `ops/scr_state_sync.py` and is
     unrelated to the masked-service status. It is flagged here only so a
     future SSOT review does not mistake the discrepancy for breakage.

4. **The named masked services should NOT block "paper complete" unless
   the operator explicitly chooses to re-arm them.**
   - Paper-runtime artifacts all fresh at audit time:
     `positions_truth.json` (29 s), `trade_lifecycle_state.json` (29 s),
     `reconciliation_state.json` (106 s, status=GREEN),
     `position_guard_drift.json` (110 s, drift_count=0),
     `decision_trace_heartbeat.json` (5 s),
     `scr_state.json` (28 s, CONFIDENT, 1.0×),
     `live_readiness.json` (399 s, ready_for_live=False).
   - PR-05/PR-07/PR-08 audit results: PR-07 and PR-08 VERIFIED (67/67
     oneshot success in 48 h on both `chad-ibkr-broker-events.service`
     and `chad-ibkr-paper-fill-harvester.service`); PR-05 is a
     non-existent unit (see §7).
   - With the alternate SCR/shadow publishers healthy and all downstream
     paper artifacts fresh, the masked runner/exec pair do not
     constitute a runtime gap. They are a **policy choice**, not an
     outage.

5. **Re-arming requires a separate operator-approved Path B.**
   - Path B is **out of scope of PR-06**. It must be prepared as its own
     Pending Action and explicitly authorized by the operator before any
     unmask/start command runs.
   - Indicative Path B steps (NOT to be executed under PR-06):
     - `sudo systemctl unmask chad-paper-shadow-runner.service`
     - `sudo systemctl unmask chad-paper-shadow-exec.service`
     - `sudo systemctl daemon-reload`
     - Verify drop-in files are intact and ib_async migration is
       honored (PR-03 already migrated both modules; see
       `chad/tests/test_pr03_ib_async_phase2_migration.py`).
     - Stage environment: `CHAD_PAPER_SHADOW_ARM=1` plus any other
       wrapper-v2 env required by `96-ibkr-env.conf`.
     - `sudo systemctl start chad-paper-shadow-runner.service` (or
       trigger via the wrapper-v2 path that historically invoked
       `chad-paper-shadow-exec`).
     - Watch the first 3 cycles for clean ib_async session lifecycle.
   - Until Path B is approved and executed, the canonical SCR/shadow
     path remains `chad-shadow-snapshot.timer` + `chad-scr-sync.timer`.

6. **PR-05 should be corrected separately because
   `chad-paper-position-closer.service` does not exist as a systemd
   unit.**
   - `systemctl show chad-paper-position-closer.service` returns
     `LoadState=not-found`, `FragmentPath=` (empty),
     `UnitFileState=` (empty).
   - BOX-038 §3.1 (`BOX-038_ib_insync_migration_policy.md` line 62)
     documents: "no dedicated systemd timer/service runs
     `paper_position_closer.main` today".
   - PR-03 Pending Action (lines 87-88) records:
     "`chad-paper-position-closer.service` — does not exist at the
     systemd level."
   - A small note has been prepared (see companion file
     `PR-05_OFFICIAL_36_amendment_line84_patch_note_2026-05-26.md`,
     UNCOMMITTED, awaiting operator authorization) proposing that
     `OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md` line 84 be
     amended to strike the stale "obtain operator GO for restarting
     `chad-paper-position-closer.service`" clause. That patch is **not
     applied by this PR-06 document**.

7. **No live posture changed by this Pending Action.**
   - `ready_for_live = false` (verified in
     `runtime/live_readiness.json` at audit time).
   - `allow_ibkr_live = false` (verified in
     `runtime/decision_trace_heartbeat.json`).
   - `allow_ibkr_paper = true` (verified in
     `runtime/decision_trace_heartbeat.json`).
   - PR-06 introduces no code change, no config mutation, no service
     state change, and no runtime JSON edit.

## Decision recorded
The 2026-05-06 quiesce of `chad-paper-shadow-runner.service` and
`chad-paper-shadow-exec.service` is hereby formalized as the **current
SSOT-canonical policy**. The SCR/shadow surface is owned by
`chad-shadow-snapshot.timer` (publisher of `data/shadow/shadow_state.json`)
+ `chad-scr-sync.timer` (publisher of `runtime/scr_state.json` and
`runtime/shadow_state.json`). The masked runner/exec pair are **not** to
be treated as a gap against paper-complete.

## Out of scope
- Re-arming the masked services (that is Path B, requires its own
  Pending Action and explicit operator GO).
- Editing `OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md` (a
  separate patch-note draft is prepared but uncommitted).
- The Epoch-2 SCR promotion-logic discrepancy between
  `data/shadow/shadow_state.json` (WARMUP, 0.1×) and
  `runtime/scr_state.json` (CONFIDENT, 1.0×). Flagged in §3 for
  future review only.
- Any change to the PR-07 / PR-08 timer-driven oneshot services
  (both VERIFIED in the 2026-05-26 audit).

## Final status
**PR-06 = VERIFIED (formalized via Path A quiesce).**
The named masked services are not blocking paper-complete. The active
canonical SCR/shadow path is healthy and within TTL. No operator action
is required unless and until a Path B re-arm is desired.

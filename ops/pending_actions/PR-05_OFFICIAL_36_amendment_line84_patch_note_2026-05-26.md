# PR-05 — OFFICIAL_36 Amendment Line 84 Patch Note (PREPARED, NOT APPLIED)

**Date:** 2026-05-26
**Author:** Team CHAD (Claude Code)
**Status:** DRAFT — patch text prepared but **NOT applied** to
`OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md`. This file is
companion to `PR-06_shadow_runner_quiesce_formalization_2026-05-26.md`
and is **uncommitted** until the operator explicitly authorizes the
patch.

## Why this patch is needed
The 2026-05-26 paper-service audit (PR-05/PR-08) established two facts
that make a clause in
`ops/pending_actions/OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md`
line 84 stale:

1. **PR-03 is complete.** Commit `d476e8c` (2026-05-26) finished the
   ib_async Phase 2 migration. Both `chad/core/paper_position_closer.py`
   and `chad/core/paper_shadow_runner.py` are now on `ib_async`.
   `chad/tests/test_pr03_ib_async_phase2_migration.py` pins the
   contract. The "migrate the 2 runtime files from `ib_insync` to
   `ib_async`" remaining-work clause is satisfied.
2. **`chad-paper-position-closer.service` does not exist.** Per
   `systemctl show chad-paper-position-closer.service`:
   `LoadState=not-found`, `FragmentPath=` (empty),
   `UnitFileState=` (empty). BOX-038 §3.1 documents:
   "no dedicated systemd timer/service runs `paper_position_closer.main`
   today". PR-03 Pending Action lines 87-88 record the same fact.
   The "operator GO for restarting `chad-paper-position-closer.service`"
   clause refers to a unit that has never existed.

## Current text on line 84
> Remaining work: migrate the 2 runtime files from `ib_insync` to
> `ib_async`; add `chad/tests/test_ib_insync_zero_imports_in_production.py`;
> obtain operator GO for restarting
> `chad-paper-position-closer.service` and
> `chad-paper-shadow-runner.service`.

## Proposed replacement text (NOT YET APPLIED)
> Remaining work (status 2026-05-26): the 2 runtime files are migrated
> to `ib_async` via PR-03 (commit `d476e8c`); the
> `test_ib_insync_zero_imports_in_production.py` enforcement is in
> `chad/tests/test_pr03_ib_async_phase2_migration.py`;
> `chad-paper-position-closer.service` does not exist as a systemd unit
> (see PR-03 Pending Action lines 87-88 and BOX-038 §3.1 — it is a
> CLI/oneshot module, no service to restart);
> `chad-paper-shadow-runner.service` remains MASKED under the
> 2026-05-06 quiesce policy formalized by
> `PR-06_shadow_runner_quiesce_formalization_2026-05-26.md` (Path A) and
> any future re-arm requires a separate operator-approved Path B
> Pending Action.

## How to apply (when authorized)
- Edit `ops/pending_actions/OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md`,
  replacing the existing line 84 paragraph with the replacement text
  above.
- Verify with `sed -n '78,95p'` that surrounding context (lines 78-83
  and 85-95) is unchanged.
- Stage only that one file: `git add ops/pending_actions/OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md`.
- Suggested commit message:
  `Pending Action: amend OFFICIAL_36 line 84 — strike stale PR-05 service-restart clause`.

## Out of scope of this note
- Any other line in `OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md`
  (e.g. line 143 contains similar language and may also warrant a
  follow-up patch note, but is not addressed here).
- Re-arming any masked service.
- Modifying CLAUDE.md.

## Authorization needed
Operator must explicitly authorize before this patch is applied. Until
then, this file is the only artifact of the proposed change.

---

**APPLIED — 2026-06-06:** the patch payload above (PR-03 ib_async Phase 2 completeness via commit `d476e8c`; `chad-paper-position-closer.service` never existed per BOX-038 §3.1, systemctl `LoadState=not-found`; shadow-runner remains MASKED per PR-06 Path A) landed verbatim in canonical `ops/pending_actions/OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23.md` via commit `5961be2` ("PR-05/PR-06: remove stale service restart language from OFFICIAL_36"). This note file is preserved in place for audit rationale (BOX-038 cite + systemctl `LoadState=not-found` evidence, neither duplicated in the canonical file or commit message). Committed in place per v9.7 §8.1(a) disposition; no new `applied/` directory convention introduced. Closes v9.7 §8.1(a).

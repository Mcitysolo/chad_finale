# W2A Phase-3 deploy runbook (OPERATOR-GATED)

Everything here requires an explicit operator GO. Nothing runs automatically. The two
runtime-mutating scripts are DRY-RUN by default and must be dry-run FIRST, then re-run with the
typed `--confirm` token. All commands run in the LIVE tree `/home/ubuntu/chad_finale` unless
noted. Governance: one step at a time — verify, then proceed.

Branch under deploy: `goal/wave2-books-cleanup` (commits W2A-1..W2A-6). Repo-side changes
(Stage-2 quarantine honouring, CAD config officialization, tier tests) activate on merge; the
runtime scrubs activate only when the scripts are executed.

## Preconditions (MUST hold before the runtime scrubs — D2)

1. **PFF1-Q1 live in the trade_closer service.** The double-book root fix must be running in the
   live `chad-trade-closer` (or equivalent) service, else tomorrow's churn re-pollutes the book
   and re-creates the phantoms after the scrub. Confirm the deployed service is on a commit at
   or after `31f5517` (PFF1 merge) and has been restarted since.
2. **PFF1-Q2 reloaded in the SCR shadow server (:9618).** `runtime/scr_state.json.source` =
   `http://127.0.0.1:9618/shadow`; that long-running server has NOT reloaded the Q2 code
   (live scr_state still shows the pre-Q2 `effective=73, total_pnl=+103.78`). The −375.60 /
   effective-67 target assumes Q2 is active, so **restart the SCR shadow server (:9618)** as part
   of this deploy. Until it reloads, the scrub alone yields `+103.78 + 145.79 = +249.57`
   (Q2 not yet removing the +625.17 seed leak).
3. **Exec mode = paper.** `CHAD_EXECUTION_MODE=paper` (the scripts fail-closed otherwise).
4. **Broker truth fresh.** `runtime/positions_snapshot.json` fresh with UNH=228, and
   `position_guard.json broker_sync|UNH` open BUY 228 (the re-attribution script verifies both).

## Step 1 — merge the repo-side changes (activates W2A-1/5/6)

Merge `goal/wave2-books-cleanup` → `main`. This activates:
- W2A-1: Stage-2 (`trade_log_adapter`) honours the quarantine manifest (needed for the scrub to
  bind on the Stage-2 scorekeeper as well as SCR).
- W2A-5: CAD tiers/withdrawal config officialized (zero runtime change — code already reads CAD).
- W2A-6: tier tests pass tree-relative.
The scripts (W2A-2/W2A-4) ship with the branch but mutate nothing until executed in Step 3/4.

## Step 2 — verify (post-merge, pre-scrub)

```
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m pytest chad/tests/test_tier_manager.py chad/tests/test_w2a_stage2_quarantine.py \
    chad/tests/test_w2a_ghost_scrub.py chad/tests/test_w2a_reattribute_unh.py \
    chad/tests/test_w2a_reattribution_guard_dedup.py -q
python3 -m pytest tests/validation/test_isolation.py -q     # harness isolation still green
```

## Step 3 — ghost-scrub (item 1) — DRY-RUN then EXECUTE

```
# DRY-RUN (mutates nothing; verifies the 6 records against the live ledger):
python3 scripts/ghost_scrub_pff1.py
# On GO — EXECUTE:
python3 scripts/ghost_scrub_pff1.py --execute --confirm GHOST-SCRUB-PFF1
```
Writes `runtime/quarantine_manifest_pff1_ghost_scrub.json` (pins the 6 record_hashes) +
`reports/ghost_scrub_pff1_<stamp>.json`. Idempotent (re-run is a NOOP). The hash-chained ledger
is NOT rewritten.

## Step 4 — UNH re-attribution (item 2) — DRY-RUN then EXECUTE

Run only AFTER Step 3 and with the Preconditions holding.
```
python3 scripts/reattribute_unh_pff1.py                          # DRY-RUN
python3 scripts/reattribute_unh_pff1.py --execute --confirm REATTRIBUTE-UNH-PFF1
```
Writes ONE trusted `gamma|UNH`=228 @424.97 lot into `runtime/trade_closer_state.json` + the
matching `position_guard.json` entry (both `.bak`'d first) + `reports/reattribute_unh_pff1_…`.
`broker_sync|UNH` is NEVER touched (D7). Idempotent.

## Step 5 — restart the SCR shadow server (:9618) — D2

Restart the shadow SCR server so PFF1-Q2 loads and the scrubbed manifest is picked up. Only then
does the live `total_pnl` reach the −375.60 target.

## Step 6 — remove the stale config backup dir (D6)

`config/_cii_backup_20260630T005102Z/` is a pre-CAD snapshot, **untracked** (not in git), so this
is a live-tree `rm`, not a branch commit:
```
rm -rf /home/ubuntu/chad_finale/config/_cii_backup_20260630T005102Z
```
(Optional: `tar czf ~/chad_revert_points/cii_backup_20260630.tar.gz config/_cii_backup_20260630T005102Z`
first if a dated off-tree copy is wanted.)

## Step 7 — post-deploy verification

- SCR: `runtime/scr_state.json` → `effective_trades` 73 → **67** (−6); `total_pnl` → **−375.60**
  (only after Step 5 reload; pre-reload expect +249.57).
- Stage-2: `python3 -m chad.validation.trade_log_adapter --since 2026-07-20 --until 2026-07-20`
  → `excluded_by_reason['quarantined'] == 6`; the manifest is listed in the run notes.
- Guard: `position_guard.json gamma|UNH` open BUY 228; `broker_sync|UNH` still 228;
  `detect_guard_vs_broker_drift_v2` → UNH matched (no `broker_untracked_position`, no
  `qty_mismatch`).
- Exit overlay now sees `gamma|UNH` (managed reduce-only, capped at broker truth 228).

## Rollback

- Un-scrub: remove `runtime/quarantine_manifest_pff1_ghost_scrub.json` (both scorekeepers stop
  excluding the 6). The original ledger was never modified.
- Un-re-attribute: restore `runtime/trade_closer_state.json` and `runtime/position_guard.json`
  from their `*.bak_reattr_unh_<stamp>` copies (taken automatically at execute).
- Repo-side: `git checkout RATIFICATION_MASTER_20260402` (full rollback) or revert the W2A merge.

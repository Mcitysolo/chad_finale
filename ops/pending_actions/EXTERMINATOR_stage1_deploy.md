# PENDING ACTION — Exterminator Sentinel Stage 1 deploy

- **Status:** PROPOSED — awaiting explicit operator GO
- **Filed:** 2026-07-15
- **Type:** timer install (systemd) + first-run review
- **Risk:** LOW — read-only scanner; no order path, no runtime mutation, no restarts
- **Governance:** CLAUDE.md rules #6 (never modify systemd units without instruction)
  and #7 (never restart services without explicit instruction). This PA exists
  because installing a timer is an operator step the agent must not take itself.

## What is being proposed

Install and enable `chad-exterminator-sentinel.timer` (5-minute cadence) so
`chad/ops/exterminator_sentinel.py` runs continuously instead of never.

Nothing has been installed. The unit files, config, module, tests and INSTALL
doc are in the repo only:

| Artifact | Path |
|---|---|
| Module | `chad/ops/exterminator_sentinel.py` |
| Config (TTL table, allowlists, contracts) | `config/exterminator.json` |
| Tests (63) | `chad/tests/test_exterminator_sentinel.py` |
| Unit files | `ops/systemd/chad-exterminator-sentinel.{service,timer}` |
| Install runbook | `docs/EXTERMINATOR_SENTINEL_INSTALL.md` |

## Why

The existing scanner `chad/ops/exterminator.py` last ran **2026-05-05** (71
days ago, Epoch 2), has **zero production importers**, and was never scheduled.
It is an orphan CLI someone ran twice by hand. Meanwhile the conditions it was
built to catch went unreported for months — `var_state.json` has been ~69 days
stale while publishing as fresh, and `chad-ibkr-daily-bars-refresh.service` sits
in `failed`.

Worse, its reconciliation check would have slept through XOV-2345: it read
`reconciliation_state.status`, which in paper mode is a `broker_sync`-only
self-comparison. Stage 1 rebuilds that check on the **independent-leg rule**
(see the INSTALL doc), so a same-source false GREEN can no longer hide a
false-flatted book.

## Read-only contract (what makes this LOW risk)

The module may write **exactly two paths** —
`runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` and
`…_HISTORY.ndjson`. This is locked by regression tests, not convention:

- `test_run_does_not_mutate_any_runtime_byte` — every runtime file is
  byte-identical after a run.
- `test_sentinel_writes_exactly_two_paths_and_nothing_else` — the created-file
  set is compared against the allowlist.
- `test_no_mutation_tokens_in_source` — no `systemctl restart|stop|start|
  reset-failed`, no `git commit|push|add|tag|checkout|reset`, no `rmtree`,
  `os.remove`, `unlink`.
- `test_subprocess_calls_are_read_only_queries` — git is limited to
  `rev-parse`/`status`; systemd to `list-units --state=failed`.
- `test_failing_check_does_not_trigger_any_repair` — a `fail` verdict leaves
  every runtime byte untouched.
- `test_defaults_refuse_real_runtime_under_pytest` — the suite can never append
  to the real history it inspects.

The unit additionally confines the process with `ProtectSystem=full` +
`ReadWritePaths=/home/ubuntu/chad_finale/runtime/reports`, so the read-only
property holds at the OS level too, not only in code.

Alerting is `NOTIFY_ONLY` and fires **only on a fail**, deduped on a
by-construction-stable key (`exterminator_sentinel_EXS<n>`), so a persistent
finding cannot re-alert every 5 minutes the way R13's SCR-gap did (CTF-T2).

## Expected first-run result: `overall=fail`

This is correct behavior, not an install failure. The scan surfaces
pre-existing conditions (all independently verified 2026-07-15):

| Check | Verdict | Finding |
|---|---|---|
| EXS1 | fail | `var_state.json` ~69d stale vs its own 3600s TTL (~1663× breach), published as fresh |
| EXS3 | fail | 1 `pnl_untrusted` TLT row (INCIDENT-0713) in `data/trades/trade_history_20260713.ndjson`; SCR drops it via the *manual* bucket today, so it is contaminating but not currently scored |
| EXS5 | fail | `chad-ibkr-daily-bars-refresh.service` in `failed` |
| EXS2 | ok | placeholder fills present and all correctly demoted |
| EXS4 | ok | guard matches the independent collector; 4 mixed-ownership info rows |
| EXS6 | warn | uncommitted `config/tiers.json`, `config/withdrawal_policy.json` |
| EXS7 | warn | 5 runtime contracts with no pinned `schema_version` |
| EXS8 | warn | `no_baseline`; live manifest `xgb_veto_20260510_020007` is 66.7d old vs its 30d threshold |

**Operator decision required:** the three fails are real and pre-date this
work. Deploying the sentinel will alert on them. Either accept the alert and
open follow-ups, or disposition them first. They are not caused by this change.

## Steps (operator, on GO)

Follow `docs/EXTERMINATOR_SENTINEL_INSTALL.md`. Summary:

1. `mkdir -p runtime/reports` (the unit's `ReadWritePaths` must pre-exist).
2. `cp` both unit files to `/etc/systemd/system/`, `systemctl daemon-reload`.
3. `systemctl start chad-exterminator-sentinel.service` **once**; read
   `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` before going further.
4. Only if step 3 looks right: `systemctl enable --now
   chad-exterminator-sentinel.timer`.
5. Confirm `systemctl list-timers chad-exterminator-sentinel` shows the next
   elapse, and that `EXTERMINATOR_SENTINEL_HISTORY.ndjson` gains one line per run.

## Verification after deploy

- `runtime/reports/EXTERMINATOR_SENTINEL_LATEST.json` refreshes every ~5 min
  with `read_only_confirmed: true`, `runtime_files_modified: []`,
  `services_restarted: false`.
- `chad-exterminator-sentinel.service` does **not** appear in
  `systemctl list-units --state=failed` (it exits 0 on findings by design — a
  non-zero exit would put its own unit into `failed` and trip its own EXS5).
- At most one Telegram alert per dedupe TTL per distinct failing check set.

## Rollback

```bash
sudo systemctl disable --now chad-exterminator-sentinel.timer
sudo rm -f /etc/systemd/system/chad-exterminator-sentinel.{service,timer}
sudo systemctl daemon-reload
```

No other state to undo — only the two report artifacts were ever written.

## Allowlist entry pending disposition

`config/exterminator.json::git.allowlist` currently suppresses the **20 staged
`_archive/` deletions** (`_archive/bak_purge_20260506/*`,
`_archive/bak_quarantine_20260402/*`) that predate this sentinel. This is a
documented, temporary entry.

**Remove it** once those deletions are committed or reverted — otherwise EXS6
is permanently blind to that path. Tracked here rather than in the config alone
so it cannot quietly become permanent.

---

## Pre-registered follow-on: Stage 2 (SEPARATE future PA — not authorized here)

**Stage 2 is explicitly OUT OF SCOPE for this PA.** It is recorded now only so
the boundary is unambiguous and so Stage 1's read-only contract cannot be
eroded by increments.

- **Proposed PA:** `ops/pending_actions/EXTERMINATOR_stage2_propose_only.md`
  (to be filed separately; requires its own GO)
- **Scope:** **PROPOSE-ONLY.** Stage 2 may emit a *suggested remediation* per
  finding — as a Pending Action document — and may not apply it. No auto-fix,
  no auto-restart, no config mutation, no git staging. A human applies every
  proposal.
- **Hard boundary:** Stage 1's anti-auto-healing tests must continue to pass
  unmodified after Stage 2 lands. If a Stage 2 change requires weakening
  `test_no_mutation_tokens_in_source`, `test_sentinel_writes_exactly_two_paths_
  and_nothing_else`, or `test_failing_check_does_not_trigger_any_repair`, that
  is a design error in Stage 2, not a stale test.
- **Rationale:** the value of a sentinel is that its report is trustworthy. A
  scanner that also repairs can mask the very drift it is meant to expose — and
  CHAD has already been burned by components that quietly "corrected"
  themselves into agreement (the same-source GREEN of XOV-2345). Observation and
  remediation stay in separate processes with separate authorization.

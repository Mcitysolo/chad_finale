# BOX-051 — Commit / archive / split / hold / exclude decision record

- **Box number:** 051
- **Box title:** COMMIT_OR_ARCHIVE_DECISION_RECORDED
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Decision-record cut (UTC):** 2026-05-20T16:57:39Z
- **HEAD at decision cut:** `bbe7525` (short) — "GAP-039 (Phase-58/59): relocate stop-bus evaluate before early-return"
- **Branch:** `main` (ahead of `origin/main` by 402 commits — observational only; Box 051 does NOT push)
- **Decision status:** **Recommendation only** — operator approval required before any of Options A – E is executed.

---

## 0. Scope and safety statement

This file records the operator-decision options for the patchset planned
in Box 049, references Box 050's operator changelog, and **recommends
the safest option**. It does **not** execute the decision: no `git
add`, `git commit`, `git push`, `git rm`, or file deletion is
performed during the closure of Box 051.

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** No option in this record (including
  the recommended one) flips `ready_for_live` to true.
- **`ready_for_live` remains `false`.** Source of truth:
  `runtime/live_readiness.json`.
- **This decision does not make CHAD live-ready, does not authorize
  live trading, and does not close Boxes 052 – 062.**

---

## 1. Current patchset state

| Field                                                | Value at Box-051 cut                                                  |
| ---------------------------------------------------- | --------------------------------------------------------------------- |
| Modified tracked files                                | **20** (per `git diff --name-status`)                                  |
| Untracked entries                                     | **34** (32 from Box 048 + 1 Box-049 plan + 1 Box-050 changelog)        |
| Staged changes                                        | **0**                                                                  |
| HEAD                                                  | `bbe7525acbbe34864cbb4a7e4084789bd45ea906` (unchanged across Box 049/050/051) |
| Box 047 full-suite result                             | **GREEN — 2361 / 2361 passed, 92.69s**                                 |
| Box 048 secret/live-enablement scan                   | **clean** (0 hits)                                                     |
| Box 049 patchset plan                                 | **20 atomic per-box patchsets (A – T)** + 1 documented exclusion (Z)   |
| Box 050 operator changelog                            | **present** (`docs/CHAD_OPERATOR_CHANGELOG_BOXES_001_050_2026-05-20.md`, 449 lines) |
| `runtime/live_readiness.json` `ready_for_live`        | `false`                                                                |
| `chad-live-loop.service`                              | `active (running)`                                                     |

Working-tree state is byte-identical to Box 048 / 049 / 050 inventories
(modulo the per-box pending-action / evidence / changelog files those
boxes themselves authored). No new production-code file has appeared
since Box 047.

---

## 2. Decision options

### Option A — **Commit now as planned patchsets** (one commit per patchset)

- **Description:** Execute Box 049's plan, committing patchsets A – T
  in order (each commit corresponds to exactly one closed box).
- **Pros:**
  - Aligns with CLAUDE.md governance rule #8 ("Commit and tag git
    after each completed P0, P1, or P2 item").
  - Backed by Box 047 GREEN (2361 / 2361) — no regression risk.
  - Box 048 verified no secrets, no live enablement.
  - Per-patchset granularity preserves rollback granularity
    (per Box 049 §4).
  - The repository is 402 commits ahead of `origin/main` already;
    further delay increases drift risk.
- **Cons:**
  - 20 commits in one batch — pre-commit hooks (if any) run 20×.
  - Operator must commit-and-tag at each step (per governance §8) to
    preserve traceability.
- **Risk:** **LOW** — well-rehearsed pattern; tests green; no live
  config touched.
- **Operator approval required?** YES.
- **Push?** NO (not in scope of Option A; push is a separate operator
  decision).

### Option B — **Archive patchset without commit**

- **Description:** Bundle the 52 changed/untracked files into a tarball
  under `_archive/` (or similar), preserve the patchset plan + evidence,
  and leave the working tree untouched.
- **Pros:**
  - Preserves the work without committing if the operator wants to
    decouple Stage-3 closure from git history.
  - Useful if the operator plans to rewrite history or rebase.
- **Cons:**
  - Loses per-commit traceability (no `git log` entries).
  - Working tree remains dirty across future boxes (which complicates
    Box 048-style diff reviews going forward).
  - Recovery from archive is a manual `tar -xf … && git apply` flow
    that is error-prone.
- **Risk:** **MED** — preserves audit but degrades git ergonomics.
- **Operator approval required?** YES.

### Option C — **Split into smaller review batches**

- **Description:** Subdivide Box 049's 20 patchsets into sub-batches
  (e.g. safety/runtime fixes first → execution correctness →
  strategy filtering → tests → ops → docs/pending-actions), each
  reviewed and committed independently with re-tests between.
- **Pros:**
  - Strictest interpretation of CLAUDE.md §1 ("One change at a time.
    Baseline, change, verify, proceed.").
  - Smallest blast radius per commit batch.
- **Cons:**
  - Slow: re-running `pytest -q chad/tests tests` between batches
    adds ~90s × N batches.
  - The Box-049 plan already enforces per-box atomicity, so most of
    the safety benefit of "split" is already realised in Option A.
- **Risk:** **LOW** (functionally equivalent to Option A but slower).
- **Operator approval required?** YES.

### Option D — **Hold working tree until remaining boxes close**

- **Description:** Defer all commits until Boxes 052 – 062 are also
  closed; commit only at the very end.
- **Pros:**
  - Single mega-commit-batch with everything tested together.
- **Cons:**
  - Violates CLAUDE.md §8 (commit-after-each-completed-item).
  - Loses per-box traceability — Boxes 003 – 050 commits would all
    land at once instead of being atomic units.
  - 12 more boxes (052 – 062) of working-tree drift before commit
    increases the surface for regressions/conflicts.
  - Existing 402-commits-ahead-of-origin pile grows further.
- **Risk:** **MED-HIGH** — drift accumulation and lost traceability.
- **Operator approval required?** YES.

### Option E — **Exclude evidence/generated files and commit code/docs/tests only**

- **Description:** Execute Patchsets A – T from Box 049 (which already
  exclude `runtime/completion_matrix_evidence/` by `.gitignore` policy
  — i.e. Patchset Z is already an exclusion). Evidence files remain
  on disk under `runtime/completion_matrix_evidence/` as the canonical
  audit trail, but never enter git.
- **Pros:**
  - **This is exactly what Box 049 already planned.** Patchset Z
    documents the exclusion explicitly. There is no need for a
    separate "Option E" — Option A *is* Option E.
- **Cons:**
  - n/a (this is the default).
- **Risk:** **LOW** (identical to Option A).
- **Operator approval required?** YES.

---

## 3. Recommended decision

**Recommended option:** **Option A — Commit now as planned patchsets.**

**Rationale:**

1. **Test backing:** Box 047 GREEN (2361 / 2361, 92.69s, 0 failed/errors/skipped)
   means there is no known regression to commit.
2. **Safety scan backing:** Box 048 verified 0 secrets and 0 live-enablement
   changes across diff and untracked sets (broader regex re-run in Box 049 §6 — clean).
3. **Per-patchset atomicity already designed:** Box 049 plan groups
   each closed box's full file set into a single commit unit (A – T),
   with rollback notes per patchset.
4. **Governance compatibility:** CLAUDE.md §8 ("Commit and tag git
   after each completed P0, P1, or P2 item") prefers per-item commits
   over batched mega-commits.
5. **Drift cost of deferral:** the repo is already 402 commits ahead
   of `origin/main`; deferring Stage-3 commits further widens the
   working-tree drift window.
6. **Evidence policy preserved:** evidence files (Patchset Z) remain
   on disk under `runtime/completion_matrix_evidence/` and are
   excluded from git by existing `.gitignore` — Option A *is* Option E
   in operational terms; the recommended path commits code/docs/tests
   and keeps evidence on disk.

**This recommendation is advisory.** Operator may choose any of
Options A – E. Box 051 closes only when the decision (whatever the
choice) is recorded here.

**Decision pending operator approval:** the operator has not yet
approved or selected an option as of this Box-051 cut. The
recommendation above is the auditor's safest-path read; operator
selection happens on Channel-1.

---

## 4. Operator approval requirement

- **No commit may be executed without explicit operator approval.**
- The operator selects exactly one of Options A – E (or "defer" =
  hold without a chosen option; see Option D for the implications).
- Once selected, the operator (or a follow-up box) runs the commands
  in §6 below — **never Box 051 itself**.
- If the operator selects Option A but wants to push afterwards, push
  is a **separate** operator approval, not implicit in Option A.

---

## 5. Risk notes

| Risk class                                  | Notes                                                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Untracked evidence / policy / docs volume   | 34 untracked entries; 32 are committable per Box 049 plan, 1 is Box 049 plan itself, 1 is Box 050 changelog. None are "should-not-commit". |
| Systemd / drop-in staged-vs-deployed        | `ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf` is already deployed under `/etc/systemd/system/...` (Box 021 verify). Committing the repo file is SOT-mirror only; **no Channel-1 redeploy required.** |
| Runtime posture                             | PAPER (`CHAD_EXECUTION_MODE=paper`). Commit decision does not change posture.                                                        |
| `ready_for_live`                            | `false`. Commit decision does not flip it. Only the live_readiness publisher writes that field.                                       |
| Pre-commit hooks                            | If hooks exist, they MUST run for each commit (no `--no-verify`). If a hook fails, fix the underlying issue per CLAUDE.md Git Safety Protocol (do NOT `--no-verify` and do NOT amend the previous commit). |
| Pending Channel-1 / operator follow-ups     | 20 non-blocking follow-ups carried forward (Box 050 §7). None blocks the commit decision; all block the live promotion.              |

---

## 6. Future operator commands (NOT EXECUTED by Box 051)

The following commands are listed for operator convenience. **Box 051
does not run any of these.** They are for the operator (or a follow-on
box) to execute once an option is approved.

### 6.1 Baseline checks (always run before any commit)

```bash
# Confirm current state matches Box 049 plan baseline
cd /home/ubuntu/chad_finale && git status --short
cd /home/ubuntu/chad_finale && git diff --cached --name-only | wc -l   # expect 0
cd /home/ubuntu/chad_finale && git rev-parse HEAD                       # expect bbe7525...

# Confirm test baseline (per CLAUDE.md verification sequence)
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m pytest -q chad/tests tests --tb=long --disable-warnings | tail -5
# expect: "2361 passed, 14 warnings in ~92s"
```

### 6.2 Option A / E — commit by patchset (recommended)

(Run **one patchset at a time**; verify, then proceed. Do **not** run
all 20 in a single shell loop.)

```bash
# Patchset A — GAP-035 upstream operator-exclusion (Box 013/014)
git add chad/strategies/_upstream_exclusion.py \
        chad/strategies/alpha_options.py \
        chad/strategies/delta.py \
        chad/strategies/delta_pairs.py \
        chad/tests/test_strategy_upstream_exclusion.py \
        chad/tests/test_delta_pairs.py
git commit -m "GAP-035 (Box-013/014): upstream operator-exclusion at strategy emit + _upstream_exclusion helper

Anchor evidence: runtime/completion_matrix_evidence/BOX-013_GAP-035_upstream_TradeSignal_emit_exclusions.md
                 runtime/completion_matrix_evidence/BOX-014_GAP-035_deployed_verified_RESTART.md
Tests: chad/tests/test_strategy_upstream_exclusion.py, chad/tests/test_delta_pairs.py
       (full suite GREEN at 2361/2361 — see BOX-047 evidence)
Risk: LOW. Already deployed under Box 014 RESTART.
Rollback: single-commit revert restores prior emit behaviour; downstream chokepoints remain."

# Re-verify after each patchset
python3 -m pytest -q chad/tests tests --tb=long --disable-warnings | tail -5

# Patchsets B – T: see ops/pending_actions/BOX-049_commit_plan_or_patchset_ready.md §2 for the
# full file lists, headlines, and rollback notes. Each patchset commit message should:
#   - lead with the box number(s) and the GAP id(s)
#   - cite the anchor evidence file(s) under runtime/completion_matrix_evidence/
#   - cite the proving test(s)
#   - cite the Box 047 GREEN baseline reference
#   - include a one-line rollback note
```

### 6.3 Push (separate operator approval)

```bash
# Only after every approved patchset has been committed AND re-verified.
# Push is NOT part of Box 051's recommendation.
# git push origin main      # requires explicit operator approval
```

### 6.4 Option B — archive without commit

```bash
# Archive into _archive/ (not git-tracked)
DEST="_archive/stage3_patchset_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DEST"
# Modified files
git diff > "$DEST/modified.diff"
# Untracked files (preserve directory structure)
git ls-files --others --exclude-standard | tar -czf "$DEST/untracked.tar.gz" -T -
# Evidence remains in place (do not move out of runtime/completion_matrix_evidence/)
```

### 6.5 Option C — split into smaller review batches

```bash
# Each batch is a strict subset of §6.2's patchsets. Commit one batch,
# re-run pytest, only proceed when GREEN. Operator selects the split
# boundaries (suggested: safety/runtime first → execution → strategies
# → tests → ops → docs).
```

### 6.6 Option D — hold

No commands. Working tree is left as-is; commit decision deferred.
The operator must remember that working-tree state will continue to
accumulate as Boxes 052 – 062 close.

---

## 7. Rollback / archive guidance

### 7.1 If a committed patchset turns out to regress something

```bash
# Find the commit
git log --oneline -20

# Revert it (creates a NEW commit; does NOT rewrite history)
git revert <commit-sha>
python3 -m pytest -q chad/tests tests --tb=long --disable-warnings | tail -5
```

Per Box 049 §4 rollback summary:

- Test-only patchsets (B / I / J / K / L / N): rollback is neutral.
- Reconciliation / fail-closed patchsets (E): rollback widens GREEN
  classification (less conservative). `ready_for_live` still gated
  separately.
- Execution-correctness patchsets (C / D / H): rollback re-introduces
  the original bug. Avoid unless a proven regression is observed.
- Cleanup tools (M / O / P / Q): rollback removes operator helpers.
- Docs / SSOT / DoD / pending actions (R / S / T): rollback removes
  audit-trail records from git history.

### 7.2 Preserve evidence files

**Do NOT delete files under `runtime/completion_matrix_evidence/`.**
Those are the canonical audit trail and are intentionally not
git-tracked (excluded by `.gitignore`); deleting them destroys the
audit trail without git history to recover from. If preservation
beyond the live disk is needed:

```bash
# Operator-driven backup (does not change git state)
tar -czf "/home/ubuntu/chad_revert_points/completion_matrix_evidence_$(date -u +%Y%m%dT%H%M%SZ).tar.gz" \
        runtime/completion_matrix_evidence/
```

### 7.3 Scorched-earth rollback (last resort)

Per CLAUDE.md "Rollback Command":

```bash
git checkout RATIFICATION_MASTER_20260402
```

This restores the pre-hardening baseline and discards everything from
Boxes 003 – 050. Per-patchset revert is strongly preferred under
normal circumstances.

---

## 8. False-closure guardrails

The following claims must NOT be made on the basis of Box 051's
closure (or any future commit decision):

- **"CHAD is complete."** — DoD v1.0 tracker: 49 / 62 closed. Boxes
  051 – 062 remain OPEN (and Box 051 closes itself by recording this
  record; it does not close 052 – 062).
- **"CHAD is live-ready."** — `runtime/live_readiness.json`
  `ready_for_live=false`. Commit decision does not flip it.
- **"live trading authorized."** — `live trading not authorized`.
  Commit decision does not authorize live trading.
- **"All remaining boxes are closed."** — Boxes 052 – 062 still OPEN.
- **"Committing makes CHAD live-ready."** — false. Live promotion has
  its own publisher gate + operator GO per DoD v1.0 §2.2.

---

## 9. Anti-speculation footer

- Box 051 does not execute the commit decision. The operator (or a
  follow-on box) does.
- Box 051 does not run `git add`, `git commit`, `git push`, `git rm`,
  or any file-delete command.
- Box 051 does not authorize live trading and does not modify any
  runtime JSON, SQLite, ledgers, fills, fees, trades, broker events,
  or order state.
- Box 051 does not pre-judge the operator's option choice; the
  recommendation in §3 is advisory only.
- No future-box state (052 – 062) is claimed as complete.

**live trading not authorized.**
